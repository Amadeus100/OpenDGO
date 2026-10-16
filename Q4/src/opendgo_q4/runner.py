from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from opendgo_q1.data import domain_task_path, load_tasks
from opendgo_q1.domains import get_domain
from opendgo_q1.organizations import Organization
from opendgo_q1.runner import BudgetState, EventRecorder, ExperimentRunner
from opendgo_q1.schema import TaskPackage, Usage
from opendgo_q1.topologies import (
    FIXED_EDGES,
    adjacency,
    sequential_inputs,
    sequential_layers,
    topology_record,
)
from opendgo_q1.util import (
    estimate_tokens,
    extract_json,
    normalize_for_loop_detection,
    safe_component,
    sha256_json,
    sha256_text,
    utc_now,
)

from .config import Q4Config
from .faults import FaultPlan, build_fault_plan
from .util import write_json, write_jsonl


class Q4BudgetState(BudgetState):
    def accounted_tokens(self) -> int:
        return int(self.usage.total_tokens + self.message_tokens_estimated)

    def collaboration_allowed(self) -> bool:
        return (
            self.accounted_tokens() < self.max_total_tokens - self.final_reserve
            and self.messages_delivered < self.max_messages
        )

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        value.update(
            {
                "agent_input_output_tokens": self.usage.total_tokens,
                "inter_agent_message_tokens_estimated": self.message_tokens_estimated,
                "accounted_total_tokens": self.accounted_tokens(),
                "accounting_rule": "provider input/output plus delivered inter-agent message estimate",
            }
        )
        return value


def select_tasks(
    config: Q4Config, domains: list[str] | None = None,
    structures: list[str] | None = None, limit: int | None = None,
) -> list[TaskPackage]:
    selected: list[TaskPackage] = []
    structure_names = set(structures or config.experiment["task_structures"])
    for domain in domains or list(config.experiment["domains"]):
        tasks = load_tasks(domain_task_path(config.q1_data_root, domain), domain)
        for structure in sorted(structure_names):
            cell = sorted(
                (task for task in tasks if task.task_structure == structure),
                key=lambda task: task.task_id,
            )
            selected.extend(cell[:limit] if limit is not None else cell)
    return selected


class Q4Runner(ExperimentRunner):
    def __init__(self, config: Q4Config) -> None:
        super().__init__(config.q1_config())
        self.q4_config = config
        self._active_cap = int(config.experiment["natural_max_total_tokens"])
        self._active_fault = FaultPlan(0, 0, 0, (), (), 0, 0.0, "")

    def _make_budget(self) -> Q4BudgetState:
        return Q4BudgetState(
            max_total_tokens=self._active_cap,
            final_reserve=int(self.experiment["final_token_reserve"]),
            max_messages=int(self.experiment["max_messages"]),
            usage=Usage(),
        )

    def _record_delivery(
        self, recorder: EventRecorder, budget: BudgetState, *, sender: int,
        recipient: int, round_index: int, phase: str, content: str,
    ) -> bool:
        fault = self._active_fault
        message_tokens = estimate_tokens(content)
        drop_reason = ""
        if sender in fault.failed_agents or recipient in fault.failed_agents:
            drop_reason = "failed_agent_endpoint"
        elif (sender, recipient) in fault.failed_channels:
            drop_reason = "failed_message_channel"
        elif budget.messages_delivered >= budget.max_messages:
            drop_reason = "message_count_cap"
        elif isinstance(budget, Q4BudgetState) and (
            budget.accounted_tokens() + message_tokens
            > budget.max_total_tokens - budget.final_reserve
        ):
            drop_reason = "token_cap_final_reserve"
        delivered = not drop_reason
        budget.record_message(content, delivered)
        event = {
            "event": "message",
            "phase": phase,
            "round": round_index,
            "sender": sender,
            "recipient": recipient,
            "delivered": delivered,
            "drop_reason": drop_reason,
            "content_sha256": sha256_text(content),
            "chars": len(content),
            "tokens_estimated": message_tokens,
        }
        if bool(self.experiment.get("record_message_content", True)):
            event["content"] = content
        recorder.append(event)
        return delivered

    async def _run_multi_agent(
        self, task: TaskPackage, organization: Organization,
        run_seed: int, recorder: EventRecorder,
    ) -> dict[str, Any]:
        assert organization.edges is not None
        graph = adjacency(organization.edges)
        sink = int(self.experiment.get("synthesizer_node", 0))
        failed = set(self._active_fault.failed_agents)
        budget = self._make_budget()
        previous = [""] * 6
        watchdog_reason = ""
        if sink in failed:
            recorder.append({"event": "synthesizer_failed", "node": sink})
            return {
                "status": "incomplete",
                "raw_answer": "",
                "parsed_answer": None,
                "parse_failed": True,
                "format_repair_attempts": 0,
                "budget": budget.to_dict(),
                "watchdog_stop_reason": "synthesizer_failed",
            }

        if task.task_structure == "sequential_dependency":
            inputs = sequential_inputs(organization.edges, sink)
            for layer_index, layer in enumerate(sequential_layers(organization.edges, sink)):
                nodes = [node for node in layer if node not in failed]
                if not nodes:
                    continue
                if not budget.collaboration_allowed():
                    watchdog_reason = "collaboration_budget_exhausted"
                    break
                inbound: dict[int, list[tuple[int, str]]] = {node: [] for node in nodes}
                for recipient in nodes:
                    for sender in inputs[recipient]:
                        if previous[sender] and self._record_delivery(
                            recorder,
                            budget,
                            sender=sender,
                            recipient=recipient,
                            round_index=layer_index,
                            phase="sequential_dependency",
                            content=previous[sender],
                        ):
                            inbound[recipient].append((sender, previous[sender]))
                completions = await asyncio.gather(
                    *(
                        self._agent_call(
                            task,
                            node,
                            inbound[node],
                            run_seed=run_seed,
                            round_index=0,
                            phase="sequential_dependency",
                            recorder=recorder,
                        )
                        for node in nodes
                    )
                )
                for node, completion in zip(nodes, completions):
                    previous[node] = completion.content
                    budget.usage.add(completion.usage)
        else:
            active_nodes = [node for node in range(6) if node not in failed]
            completions = await asyncio.gather(
                *(
                    self._agent_call(
                        task,
                        node,
                        [],
                        run_seed=run_seed,
                        round_index=0,
                        phase="independent",
                        recorder=recorder,
                    )
                    for node in active_nodes
                )
            )
            for node, completion in zip(active_nodes, completions):
                previous[node] = completion.content
                budget.usage.add(completion.usage)
            repeat_counts: list[Counter[str]] = [Counter() for _ in range(6)]
            earlier_state: tuple[str, ...] | None = None
            unchanged_rounds = 0
            for round_index in range(1, int(self.experiment["communication_rounds"]) + 1):
                if not budget.collaboration_allowed():
                    watchdog_reason = "collaboration_budget_exhausted"
                    break
                inbound = self._inbound_from_edges(
                    previous,
                    graph,
                    recorder,
                    budget,
                    round_index=round_index,
                    phase=task.task_structure,
                )
                completions = await asyncio.gather(
                    *(
                        self._agent_call(
                            task,
                            node,
                            inbound[node],
                            run_seed=run_seed,
                            round_index=round_index,
                            phase=task.task_structure,
                            recorder=recorder,
                        )
                        for node in active_nodes
                    )
                )
                current = [""] * 6
                for node, completion in zip(active_nodes, completions):
                    current[node] = completion.content
                    budget.usage.add(completion.usage)
                    repeat_counts[node][normalize_for_loop_detection(completion.content)] += 1
                state = tuple(normalize_for_loop_detection(value) for value in current)
                unchanged_rounds = unchanged_rounds + 1 if state == earlier_state else 0
                earlier_state = state
                previous = current
                if any(
                    counts.most_common(1)[0][1] >= int(self.experiment.get("repeat_limit", 3))
                    for counts in repeat_counts if counts
                ):
                    watchdog_reason = "repeated_output_loop"
                    break
                if unchanged_rounds >= int(self.experiment.get("unchanged_round_limit", 2)):
                    watchdog_reason = "unchanged_shared_state"
                    break

        final_inbound: list[tuple[int, str]] = []
        for sender in graph[sink]:
            if previous[sender] and self._record_delivery(
                recorder,
                budget,
                sender=sender,
                recipient=sink,
                round_index=-1,
                phase="final_delivery",
                content=previous[sender],
            ):
                final_inbound.append((sender, previous[sender]))
        raw, parsed, repairs = await self._final_call(
            task,
            final_inbound,
            previous[sink],
            run_seed=run_seed,
            budget=budget,
            recorder=recorder,
        )
        return {
            "status": "completed",
            "raw_answer": raw,
            "parsed_answer": parsed,
            "parse_failed": parsed is None,
            "format_repair_attempts": repairs,
            "budget": budget.to_dict(),
            "watchdog_stop_reason": watchdog_reason,
        }

    async def _run_external_q4_coding(
        self, task: TaskPackage, organization: Organization,
        run_seed: int, recorder: EventRecorder,
    ) -> dict[str, Any]:
        settings = dict(self.config.executors.get("coding", {}))
        command = settings.get("command", [])
        if not isinstance(command, list) or not command:
            raise RuntimeError("Coding requires executors.coding.command")
        request = {
            "protocol": "opendgo_q4_coding_executor_v1",
            "task": task.public_dict(),
            "organization": {
                "name": organization.name,
                "edges": [list(edge) for edge in organization.edges or ()],
            },
            "seed": run_seed,
            "token_cap": self._active_cap,
            "final_token_reserve": int(self.experiment["final_token_reserve"]),
            "fault_plan": self._active_fault.to_dict(),
        }
        process = await asyncio.create_subprocess_exec(
            *(str(part) for part in command),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.config.project_root),
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        timeout = float(settings.get("timeout_seconds", 7200))
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(request, ensure_ascii=False).encode("utf-8")),
                timeout=timeout,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"coding executor exceeded {timeout} seconds")
        if process.returncode != 0:
            raise RuntimeError(
                f"coding executor exited {process.returncode}: "
                f"{stderr.decode('utf-8', errors='replace')[-2000:]}"
            )
        payload = extract_json(stdout.decode("utf-8", errors="replace"))
        if not isinstance(payload, dict):
            raise RuntimeError("coding executor did not return a JSON object")
        recorder.append(
            {"event": "coding_executor", "return_code": 0, "payload_sha256": sha256_json(payload)}
        )
        return {
            "status": str(payload.get("status", "completed")),
            "raw_answer": payload.get("raw_answer", payload.get("patch", "")),
            "parsed_answer": payload.get("parsed_answer", {"patch": payload.get("patch", "")}),
            "parse_failed": bool(payload.get("parse_failed", False)),
            "format_repair_attempts": int(payload.get("format_repair_attempts", 0)),
            "budget": dict(payload.get("budget", {})),
            "watchdog_stop_reason": str(payload.get("watchdog_stop_reason", "")),
            "official_evaluation": payload.get("official_evaluation"),
            "external_executor_metadata": payload.get("metadata", {}),
        }

    def _condition_dir(
        self, task: TaskPackage, organization: str, seed: int,
        condition_type: str, condition_value: int,
    ) -> Path:
        return (
            self.q4_config.results_root
            / "runs"
            / condition_type
            / str(condition_value)
            / task.domain
            / task.task_structure
            / organization
            / str(seed)
            / safe_component(task.task_id)
        )

    async def run_condition(
        self, task: TaskPackage, organization_name: str, run_seed: int,
        condition_type: str, condition_value: int,
    ) -> dict[str, Any]:
        if organization_name not in FIXED_EDGES:
            raise ValueError(f"Q4 only accepts fixed topology: {organization_name}")
        if condition_type == "budget_cap":
            self._active_cap = int(condition_value)
            pressure = 0
        elif condition_type == "fault_pressure":
            self._active_cap = int(self.q4_config.experiment["natural_max_total_tokens"])
            pressure = int(condition_value)
        else:
            raise ValueError(f"unknown condition type: {condition_type}")
        self._active_fault = build_fault_plan(
            FIXED_EDGES[organization_name],
            pressure,
            task_id=task.task_id,
            task_structure=task.task_structure,
            organization=organization_name,
            seed=run_seed,
        )
        run_dir = self._condition_dir(
            task, organization_name, run_seed, condition_type, condition_value
        )
        result_path = run_dir / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        run_dir.mkdir(parents=True, exist_ok=True)
        recorder = EventRecorder(run_dir / "events.jsonl")
        organization = Organization(
            organization_name, "fixed", FIXED_EDGES[organization_name], {}
        )
        started = time.monotonic()
        recorder.append(
            {
                "event": "run_started",
                "protocol": "opendgo_q4_v1",
                "condition_type": condition_type,
                "condition_value": condition_value,
                "fault_plan": self._active_fault.to_dict(),
            }
        )
        error = ""
        try:
            if get_domain(task.domain).requires_external_executor:
                outcome = await self._run_external_q4_coding(
                    task, organization, run_seed, recorder
                )
            else:
                outcome = await self._run_multi_agent(
                    task, organization, run_seed, recorder
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            recorder.append({"event": "run_error", "error": error})
            outcome = {
                "status": "incomplete",
                "raw_answer": "",
                "parsed_answer": None,
                "parse_failed": True,
                "format_repair_attempts": 0,
                "budget": {},
                "watchdog_stop_reason": "execution_error",
            }
        budget_record = dict(outcome.get("budget", {}))
        provider_tokens = int(
            budget_record.get("agent_input_output_tokens", budget_record.get("provider_usage", {}).get("total_tokens", 0))
        )
        message_tokens = int(budget_record.get("inter_agent_message_tokens_estimated", budget_record.get("message_tokens_estimated", 0)))
        result = {
            "protocol_version": "opendgo_q4_v1",
            "run_key": sha256_text(
                f"q4|{condition_type}|{condition_value}|{task.task_id}|{organization_name}|{run_seed}"
            ),
            "condition_type": condition_type,
            "condition_value": int(condition_value),
            "token_cap": self._active_cap,
            "fault_pressure_percent": pressure,
            "fault_plan": self._active_fault.to_dict(),
            "task_id": task.task_id,
            "source_id": task.source_id,
            "domain": task.domain,
            "dataset": get_domain(task.domain).dataset,
            "task_structure": task.task_structure,
            "organization": organization_name,
            "organization_category": "fixed",
            "organization_record": topology_record(organization_name, organization.edges or ()),
            "seed": run_seed,
            "model": self.backend.model,
            "visible_input_sha256": task.visible_input_sha256 or task.compute_visible_hash(),
            "config_sha256": self.q4_config.fingerprint,
            **outcome,
            "token_accounting": {
                "agent_input_output_tokens": provider_tokens,
                "inter_agent_message_tokens_estimated": message_tokens,
                "accounted_total_tokens": provider_tokens + message_tokens,
            },
            "error": error,
            "runtime_seconds": time.monotonic() - started,
            "completed_at_utc": utc_now(),
        }
        write_json(result_path, result)
        recorder.append(
            {"event": "run_finished", "status": result["status"], "result_sha256": sha256_json(result)}
        )
        return result

    async def run_plan(
        self, tasks: list[TaskPackage], organizations: list[str], seeds: list[int],
        condition_type: str, values: list[int],
    ) -> dict[str, Any]:
        if any(task.domain == "coding" for task in tasks):
            if not self.config.executors.get("coding", {}).get("command", []):
                raise RuntimeError("selected tasks include Coding but coding executor is not configured")
        rows = [
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "task_structure": task.task_structure,
                "organization": organization,
                "seed": seed,
                "condition_type": condition_type,
                "condition_value": value,
            }
            for task in tasks
            for organization in organizations
            for seed in seeds
            for value in values
        ]
        plan_id = sha256_json({"config": self.q4_config.fingerprint, "rows": rows})[:16]
        write_jsonl(self.q4_config.results_root / "plans" / f"plan_{plan_id}.jsonl", rows)
        if any(task.domain != "coding" for task in tasks):
            print(json.dumps({"model_health": await self.backend.health()}, ensure_ascii=False), flush=True)
        statuses: Counter[str] = Counter()
        completed = 0
        for task in tasks:
            for organization in organizations:
                for seed in seeds:
                    for value in values:
                        result = await self.run_condition(
                            task, organization, seed, condition_type, value
                        )
                        completed += 1
                        statuses[str(result["status"])] += 1
                        print(
                            json.dumps(
                                {
                                    "completed": completed,
                                    "total": len(rows),
                                    "task_id": task.task_id,
                                    "organization": organization,
                                    condition_type: value,
                                    "seed": seed,
                                    "status": result["status"],
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
        return {"plan_id": plan_id, "total": len(rows), "statuses": dict(statuses)}


async def run_selected(
    config: Q4Config, *, condition_type: str,
    domains: list[str] | None = None, structures: list[str] | None = None,
    organizations: list[str] | None = None, seeds: list[int] | None = None,
    values: list[int] | None = None, limit: int | None = None,
) -> dict[str, Any]:
    tasks = select_tasks(config, domains, structures, limit)
    default_values = (
        config.experiment["token_caps"]
        if condition_type == "budget_cap"
        else config.experiment["fault_pressures_percent"]
    )
    selected_values = [int(value) for value in (values or default_values)]
    unknown_values = sorted(set(selected_values) - {int(value) for value in default_values})
    if unknown_values:
        raise ValueError(
            f"condition values are outside the frozen protocol: {unknown_values}"
        )
    runner = Q4Runner(config)
    try:
        return await runner.run_plan(
            tasks,
            organizations or list(config.experiment["organizations"]),
            seeds or list(config.experiment["seeds"]),
            condition_type,
            selected_values,
        )
    finally:
        close = getattr(runner.backend, "close", None)
        if close is not None:
            await close()
