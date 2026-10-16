from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .allocation import AllocationResult, allocate
from .backend import ChatBackend, Completion, OpenAICompatibleBackend
from .conditions import Assignment, FIXED_EDGES, PositionCondition, adjacency, assignment_for, oriented_layers
from .config import Q2Config
from .domains import STRUCTURE_INSTRUCTIONS, get_domain
from .schema import EvidenceUnit, InterventionAnnotation, Q2Task, Usage
from .util import (
    estimate_tokens,
    extract_json,
    normalize_for_loop_detection,
    safe_component,
    sha256_json,
    sha256_text,
    utc_now,
    write_json,
)


BASE_SYSTEM = """You are one node in a controlled six-agent reasoning system.
Use only the common task, your assigned role, your private evidence, permitted tools, and messages
actually delivered to you. Never assume access to another role's evidence, a hidden answer, or an
undelivered source. Cite evidence IDs when available. Keep findings concise and auditable."""


def _call_seed(seed: int, phase: str, node: int = 0, round_index: int = 0) -> int:
    return int(sha256_text(f"{seed}|{phase}|{node}|{round_index}")[:8], 16) % (2**31 - 1)


def _evidence_text(units: Iterable[EvidenceUnit]) -> str:
    rows = [unit.to_dict() for unit in units]
    return json.dumps(rows, ensure_ascii=False, indent=2) if rows else "[]"


def _messages_text(messages: list[tuple[int, str]], cap: int) -> str:
    if not messages:
        return "No neighbor messages were delivered."
    each = max(240, cap // len(messages))
    return "\n\n".join(
        f"[message {index + 1}]\n{content[:each]}"
        for index, (_, content) in enumerate(messages)
    )[:cap]


def _schema_satisfied(value: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    required = schema.get("required", [])
    return isinstance(required, list) and all(str(key) in value for key in required)


@dataclass
class BudgetState:
    max_total_tokens: int
    final_reserve: int
    max_messages: int
    usage: Usage
    messages_attempted: int = 0
    messages_delivered: int = 0
    messages_suppressed: int = 0
    message_chars: int = 0
    message_tokens_estimated: int = 0

    def collaboration_allowed(self) -> bool:
        return (
            self.usage.total_tokens < self.max_total_tokens - self.final_reserve
            and self.messages_delivered < self.max_messages
        )

    def record_message(self, content: str, delivered: bool) -> None:
        self.messages_attempted += 1
        if delivered:
            self.messages_delivered += 1
            self.message_chars += len(content)
            self.message_tokens_estimated += estimate_tokens(content)
        else:
            self.messages_suppressed += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "nominal_max_total_tokens": self.max_total_tokens,
            "final_token_reserve": self.final_reserve,
            "provider_usage": self.usage.to_dict(),
            "messages_attempted": self.messages_attempted,
            "messages_delivered": self.messages_delivered,
            "messages_suppressed": self.messages_suppressed,
            "message_chars": self.message_chars,
            "message_tokens_estimated": self.message_tokens_estimated,
        }


class EventRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"at": utc_now(), **event}, ensure_ascii=False) + "\n")


class Q2Runner:
    def __init__(self, config: Q2Config, backend: ChatBackend | None = None) -> None:
        self.config = config
        self.backend = backend or OpenAICompatibleBackend(config.llm)
        self.experiment = config.experiment
        self._semaphore = asyncio.Semaphore(int(config.llm.get("max_concurrent_requests", 6)))

    async def _complete(self, system: str, user: str, max_tokens: int, seed: int) -> Completion:
        async with self._semaphore:
            return await self.backend.complete(system, user, max_tokens=max_tokens, seed=seed)

    def _run_dir(
        self,
        task: Q2Task,
        condition: PositionCondition,
        allocation_mode: str,
        seed: int,
    ) -> Path:
        return (
            self.config.results_root
            / "runs"
            / task.domain
            / task.task_structure
            / condition.topology
            / condition.slug
            / allocation_mode
            / str(seed)
            / safe_component(task.task_id)
        )

    def _budget(self) -> BudgetState:
        return BudgetState(
            max_total_tokens=int(self.experiment["max_total_tokens"]),
            final_reserve=int(self.experiment["final_token_reserve"]),
            max_messages=int(self.experiment["max_messages"]),
            usage=Usage(),
        )

    def _role_prompt(self, task: Q2Task, assignment: Assignment, node: int) -> str:
        role_index = assignment.node_to_role_index[node]
        return (
            f"Role: {task.roles[role_index]}\n"
            f"Responsibility: {task.responsibilities[role_index]}\n"
            f"Task guidance: {STRUCTURE_INSTRUCTIONS[task.task_structure]}"
        )

    def _user_prompt(
        self,
        task: Q2Task,
        assignment: Assignment,
        views: tuple[tuple[EvidenceUnit, ...], ...],
        node: int,
        inbound: list[tuple[int, str]],
    ) -> str:
        return (
            f"Task:\n{task.question}\n\n"
            f"Shared context:\n{json.dumps(task.shared_context, ensure_ascii=False)}\n\n"
            f"{self._role_prompt(task, assignment, node)}\n\n"
            f"Private evidence:\n{_evidence_text(views[node])}\n\n"
            f"Delivered neighbor messages:\n"
            f"{_messages_text(inbound, int(self.experiment['communication_budget_chars']))}\n\n"
            "Return one concise evidence-grounded finding for the next reasoning step."
        )

    def _deliver(
        self,
        recorder: EventRecorder,
        budget: BudgetState,
        sender: int,
        recipient: int,
        round_index: int,
        content: str,
    ) -> bool:
        delivered = budget.messages_delivered < budget.max_messages
        budget.record_message(content, delivered)
        event: dict[str, Any] = {
            "event": "message",
            "round": round_index,
            "sender": sender,
            "recipient": recipient,
            "delivered": delivered,
            "content_sha256": sha256_text(content),
            "chars": len(content),
            "tokens_estimated": estimate_tokens(content),
        }
        if bool(self.experiment.get("record_message_content", True)):
            event["content"] = content
        recorder.append(event)
        return delivered

    async def _call_nodes(
        self,
        task: Q2Task,
        assignment: Assignment,
        views: tuple[tuple[EvidenceUnit, ...], ...],
        inbound: dict[int, list[tuple[int, str]]],
        nodes: list[int],
        phase: str,
        round_index: int,
        budget: BudgetState,
        recorder: EventRecorder,
    ) -> dict[int, str]:
        async def call(node: int) -> tuple[int, Completion]:
            completion = await self._complete(
                BASE_SYSTEM,
                self._user_prompt(task, assignment, views, node, inbound.get(node, [])),
                int(self.config.llm["max_tokens"]),
                _call_seed(int(self._current_seed), phase, node, round_index),
            )
            return node, completion

        batch = await asyncio.gather(*(call(node) for node in nodes))
        outputs: dict[int, str] = {}
        for node, completion in batch:
            budget.usage.add(completion.usage)
            outputs[node] = completion.content
            recorder.append(
                {
                    "event": "agent_output",
                    "phase": phase,
                    "round": round_index,
                    "node": node,
                    "role_index": assignment.node_to_role_index[node],
                    "content": completion.content,
                    "content_sha256": sha256_text(completion.content),
                    "usage": completion.usage.to_dict(),
                    "request_attempts": completion.request_attempts,
                }
            )
        return outputs

    async def _collaborate(
        self,
        task: Q2Task,
        assignment: Assignment,
        allocation_result: AllocationResult,
        budget: BudgetState,
        recorder: EventRecorder,
    ) -> tuple[list[str], str]:
        topology = assignment.condition.topology
        graph = adjacency(topology)
        repeat_limit = int(self.experiment.get("repeat_limit", 3))
        unchanged_limit = int(self.experiment.get("unchanged_round_limit", 2))
        repeat_counts: Counter[str] = Counter()
        unchanged = 0
        previous_state = ""

        if task.task_structure == "sequential_dependency":
            layers, upstream = oriented_layers(topology, assignment.synthesizer_node)
            outputs = ["" for _ in range(6)]
            for round_index, layer in enumerate(layers):
                inbound: dict[int, list[tuple[int, str]]] = {node: [] for node in layer}
                for node in layer:
                    for source in upstream[node]:
                        if outputs[source] and self._deliver(
                            recorder, budget, source, node, round_index, outputs[source]
                        ):
                            inbound[node].append((source, outputs[source]))
                fresh = await self._call_nodes(
                    task,
                    assignment,
                    allocation_result.views,
                    inbound,
                    layer,
                    "dependency_stage",
                    round_index,
                    budget,
                    recorder,
                )
                for node, content in fresh.items():
                    outputs[node] = content
                if not budget.collaboration_allowed():
                    return outputs, "budget_or_message_cap"
            return outputs, ""

        initial = await self._call_nodes(
            task,
            assignment,
            allocation_result.views,
            {node: [] for node in range(6)},
            list(range(6)),
            "independent_analysis",
            0,
            budget,
            recorder,
        )
        outputs = [initial[node] for node in range(6)]
        for round_index in range(1, int(self.experiment["communication_rounds"]) + 1):
            if not budget.collaboration_allowed():
                return outputs, "budget_or_message_cap"
            inbound: dict[int, list[tuple[int, str]]] = {node: [] for node in range(6)}
            for sender in range(6):
                for recipient in graph[sender]:
                    if self._deliver(
                        recorder, budget, sender, recipient, round_index, outputs[sender]
                    ):
                        inbound[recipient].append((sender, outputs[sender]))
            fresh = await self._call_nodes(
                task,
                assignment,
                allocation_result.views,
                inbound,
                list(range(6)),
                "neighbor_revision",
                round_index,
                budget,
                recorder,
            )
            outputs = [fresh[node] for node in range(6)]
            normalized = [normalize_for_loop_detection(value) for value in outputs]
            for value in normalized:
                repeat_counts[value] += 1
            if normalized and max(repeat_counts.values()) >= repeat_limit:
                return outputs, "repeated_output_loop"
            state = sha256_json(normalized)
            unchanged = unchanged + 1 if state == previous_state else 0
            previous_state = state
            if unchanged >= unchanged_limit:
                return outputs, "unchanged_shared_state"
        return outputs, ""

    async def _finalize(
        self,
        task: Q2Task,
        assignment: Assignment,
        outputs: list[str],
        budget: BudgetState,
        recorder: EventRecorder,
    ) -> tuple[str, Any, int]:
        synth = assignment.synthesizer_node
        graph = adjacency(assignment.condition.topology)
        reachable: list[tuple[int, str]] = []
        if outputs[synth]:
            reachable.append((synth, outputs[synth]))
        final_round = int(self.experiment["communication_rounds"]) + 1
        for node in graph[synth]:
            if outputs[node] and self._deliver(
                recorder, budget, node, synth, final_round, outputs[node]
            ):
                reachable.append((node, outputs[node]))
        user = (
            f"Task:\n{task.question}\n\n"
            f"Shared context:\n{json.dumps(task.shared_context, ensure_ascii=False)}\n\n"
            f"{self._role_prompt(task, assignment, synth)}\n\n"
            f"Your private evidence:\n{_evidence_text(self._current_views[synth])}\n\n"
            f"Reachable findings:\n{_messages_text(reachable, int(self.experiment['communication_budget_chars']))}\n\n"
            f"{get_domain(task.domain).final_instruction}"
        )
        completion = await self._complete(
            BASE_SYSTEM,
            user,
            int(self.config.llm["final_max_tokens"]),
            _call_seed(int(self._current_seed), "final", synth, 0),
        )
        budget.usage.add(completion.usage)
        raw = completion.content
        parsed = extract_json(raw)
        repairs = 0
        while (
            not _schema_satisfied(parsed, task.answer_schema)
            and repairs < int(self.experiment.get("parse_repair_attempts", 1))
        ):
            repairs += 1
            repair = await self._complete(
                "Convert the supplied answer to the required JSON schema. Do not add facts.",
                f"Schema:\n{json.dumps(task.answer_schema, ensure_ascii=False)}\n\nAnswer:\n{raw}",
                int(self.config.llm["final_max_tokens"]),
                _call_seed(int(self._current_seed), "format_repair", synth, repairs),
            )
            budget.usage.add(repair.usage)
            raw = repair.content
            parsed = extract_json(raw)
        recorder.append(
            {
                "event": "final_answer",
                "node": synth,
                "raw_answer": raw,
                "parsed_answer": parsed,
                "parse_failed": not _schema_satisfied(parsed, task.answer_schema),
                "format_repair_attempts": repairs,
            }
        )
        return raw, parsed, repairs

    async def _run_text(
        self,
        task: Q2Task,
        assignment: Assignment,
        allocation_result: AllocationResult,
        recorder: EventRecorder,
    ) -> dict[str, Any]:
        budget = self._budget()
        self._current_views = allocation_result.views
        outputs, stop_reason = await self._collaborate(
            task, assignment, allocation_result, budget, recorder
        )
        raw, parsed, repairs = await self._finalize(task, assignment, outputs, budget, recorder)
        return {
            "status": "completed",
            "raw_answer": raw,
            "parsed_answer": parsed,
            "parse_failed": not _schema_satisfied(parsed, task.answer_schema),
            "format_repair_attempts": repairs,
            "budget": budget.to_dict(),
            "watchdog_stop_reason": stop_reason,
        }

    async def _run_coding(
        self,
        task: Q2Task,
        assignment: Assignment,
        allocation_result: AllocationResult,
        seed: int,
        recorder: EventRecorder,
    ) -> dict[str, Any]:
        settings = dict(self.config.executors.get("coding", {}))
        command = settings.get("command", [])
        if not isinstance(command, list) or not command:
            raise RuntimeError("Coding requires executors.coding.command; no proxy is permitted")
        request = {
            "protocol": "opendgo_q2_coding_executor_v1",
            "task": task.public_dict(),
            "condition": {
                "edges": [list(edge) for edge in FIXED_EDGES[assignment.condition.topology]],
                "assignment": assignment.to_dict(),
                "allocation_manifest": allocation_result.manifest,
                "evidence_views": [
                    [unit.to_dict() for unit in view] for view in allocation_result.views
                ],
            },
            "seed": seed,
            "frozen_config": self.config.frozen_public_config(),
        }
        process = await asyncio.create_subprocess_exec(
            *(str(item) for item in command),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.config.project_root),
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        timeout = float(settings.get("timeout_seconds", 7200))
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(request, ensure_ascii=False).encode()),
                timeout=timeout,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"coding executor exceeded {timeout} seconds")
        if process.returncode != 0:
            raise RuntimeError(
                f"coding executor exited {process.returncode}: "
                f"{stderr.decode(errors='replace')[-2000:]}"
            )
        payload = extract_json(stdout.decode(errors="replace"))
        if not isinstance(payload, dict):
            raise RuntimeError("coding executor stdout did not contain one JSON object")
        recorder.append(
            {
                "event": "coding_executor",
                "return_code": process.returncode,
                "stderr_tail": stderr.decode(errors="replace")[-2000:],
                "payload_sha256": sha256_json(payload),
            }
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

    async def run_one(
        self,
        task: Q2Task,
        annotation: InterventionAnnotation,
        condition: PositionCondition,
        allocation_mode: str,
        seed: int,
    ) -> dict[str, Any]:
        run_dir = self._run_dir(task, condition, allocation_mode, seed)
        result_path = run_dir / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        run_dir.mkdir(parents=True, exist_ok=True)
        assignment = assignment_for(condition, annotation.expert_role_index)
        allocated = allocate(task, annotation, assignment, allocation_mode)
        write_json(run_dir / "allocation_manifest.json", allocated.manifest)
        recorder = EventRecorder(run_dir / "events.jsonl")
        self._current_seed = seed
        started = time.monotonic()
        recorder.append(
            {
                "event": "run_started",
                "task_id": task.task_id,
                "domain": task.domain,
                "task_structure": task.task_structure,
                "condition_sha256": sha256_json(
                    {
                        "assignment": assignment.to_dict(),
                        "allocation_manifest_sha256": allocated.manifest["manifest_sha256"],
                    }
                ),
                "visible_input_sha256": task.visible_input_sha256 or task.compute_hash(),
                "config_sha256": self.config.fingerprint,
                "seed": seed,
            }
        )
        error = ""
        try:
            if get_domain(task.domain).requires_external_executor:
                outcome = await self._run_coding(task, assignment, allocated, seed, recorder)
            else:
                outcome = await self._run_text(task, assignment, allocated, recorder)
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
        result = {
            "protocol_version": "opendgo_q2_v1",
            "run_key": sha256_text(
                f"{task.task_id}|{task.task_structure}|{condition.topology}|"
                f"{condition.slug}|{allocation_mode}|{seed}"
            ),
            "task_id": task.task_id,
            "source_id": task.source_id,
            "domain": task.domain,
            "dataset": get_domain(task.domain).dataset,
            "task_structure": task.task_structure,
            "topology": condition.topology,
            "position": condition.label,
            "position_slug": condition.slug,
            "equivalent_placement_control": condition.equivalent_control,
            "information_allocation": allocation_mode,
            "expert_role_index": annotation.expert_role_index,
            "expert_node": assignment.expert_node,
            "synthesizer_node": assignment.synthesizer_node,
            "node_to_role_index": list(assignment.node_to_role_index),
            "role_order_by_node": [task.roles[index] for index in assignment.node_to_role_index],
            "edges": [list(edge) for edge in FIXED_EDGES[condition.topology]],
            "allocation_manifest_sha256": allocated.manifest["manifest_sha256"],
            "evidence_inventory_sha256": allocated.manifest["evidence_inventory_sha256"],
            "seed": seed,
            "model": self.backend.model,
            "visible_input_sha256": task.visible_input_sha256 or task.compute_hash(),
            "config_sha256": self.config.fingerprint,
            **outcome,
            "error": error,
            "runtime_seconds": time.monotonic() - started,
            "completed_at_utc": utc_now(),
        }
        write_json(result_path, result)
        recorder.append(
            {"event": "run_finished", "status": result["status"], "result_sha256": sha256_json(result)}
        )
        return result

    def _write_plan(
        self,
        tasks: list[Q2Task],
        conditions: list[PositionCondition],
        allocation_modes: list[str],
        seeds: list[int],
    ) -> Path:
        identity = {
            "config_sha256": self.config.fingerprint,
            "tasks": [task.task_id for task in tasks],
            "conditions": [(item.topology, item.slug) for item in conditions],
            "allocations": allocation_modes,
            "seeds": seeds,
        }
        plan_id = sha256_json(identity)[:16]
        path = self.config.results_root / "plans" / f"plan_{plan_id}.jsonl"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for task in tasks:
                    for condition in conditions:
                        for mode in allocation_modes:
                            for seed in seeds:
                                handle.write(
                                    json.dumps(
                                        {
                                            "task_id": task.task_id,
                                            "domain": task.domain,
                                            "task_structure": task.task_structure,
                                            "topology": condition.topology,
                                            "position": condition.label,
                                            "position_slug": condition.slug,
                                            "information_allocation": mode,
                                            "seed": seed,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
        return path

    async def run_plan(
        self,
        tasks: list[Q2Task],
        annotations: dict[str, InterventionAnnotation],
        conditions: list[PositionCondition],
        allocation_modes: list[str],
        seeds: list[int],
    ) -> dict[str, Any]:
        if any(task.domain == "coding" for task in tasks):
            command = self.config.executors.get("coding", {}).get("command", [])
            if not isinstance(command, list) or not command:
                raise RuntimeError("selected plan includes Coding but no coding executor is configured")
        missing = sorted({task.task_id for task in tasks} - set(annotations))
        if missing:
            raise RuntimeError(f"missing annotations: {missing[:20]}")
        plan_path = self._write_plan(tasks, conditions, allocation_modes, seeds)
        if any(task.domain != "coding" for task in tasks):
            print(json.dumps({"model_health": await self.backend.health()}), flush=True)
        total = len(tasks) * len(conditions) * len(allocation_modes) * len(seeds)
        done = 0
        statuses: Counter[str] = Counter()
        for task in tasks:
            for condition in conditions:
                for mode in allocation_modes:
                    for seed in seeds:
                        result = await self.run_one(
                            task, annotations[task.task_id], condition, mode, seed
                        )
                        done += 1
                        statuses[str(result["status"])] += 1
                        print(
                            json.dumps(
                                {
                                    "completed": done,
                                    "total": total,
                                    "task_id": task.task_id,
                                    "topology": condition.topology,
                                    "position": condition.label,
                                    "allocation": mode,
                                    "seed": seed,
                                    "status": result["status"],
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
        return {"plan_path": str(plan_path), "total": total, "statuses": dict(statuses)}

