from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .backend import ChatBackend, Completion, OpenAICompatibleBackend
from .config import Q1Config
from .domains import get_domain
from .organizations import (
    LEARNED_ORGANIZATIONS,
    LearnedGraphStore,
    Organization,
    resolve_organization,
)
from .schema import EvidenceUnit, TaskPackage, Usage
from .topologies import adjacency, sequential_inputs, sequential_layers, topology_record
from .util import (
    estimate_tokens,
    extract_json,
    normalize_for_loop_detection,
    safe_component,
    sha256_json,
    sha256_text,
    utc_now,
)


BASE_SYSTEM = """You are one node in a controlled OpenDGO Q1 experiment.
Use only the common task, your assigned private evidence, permitted tools, and messages delivered to
you. Never assume access to another role's evidence, a gold answer, or an undelivered message. Cite
evidence IDs when available. Keep the response concise because it may be transmitted unchanged."""

SINGLE_SYSTEM = """You are the controlled single-agent baseline in OpenDGO Q1. You receive exactly the
union of the evidence that is collectively model-visible to the six-agent conditions. Use no hidden
gold labels or outside documents. Return the frozen answer schema only."""


def _call_seed(run_seed: int, phase: str, node: int = 0, round_index: int = 0) -> int:
    digest = sha256_text(f"{run_seed}|{phase}|{node}|{round_index}")
    return int(digest[:8], 16) % (2**31 - 1)


def _json_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def _evidence_text(units: Iterable[EvidenceUnit]) -> str:
    values = [unit.model_dict() for unit in units]
    return json.dumps(values, ensure_ascii=False, indent=2) if values else "[]"


def _truncate_messages(messages: list[tuple[int, str]], budget_chars: int) -> str:
    if not messages:
        return "No messages were delivered to you in this step."
    per_message = max(240, budget_chars // len(messages))
    chunks = [f"[from node {sender}]\n{content[:per_message]}" for sender, content in messages]
    return "\n\n".join(chunks)[:budget_chars]


def _answer_schema_is_satisfied(parsed: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(parsed, dict):
        return False
    required = schema.get("required", [])
    return isinstance(required, list) and all(str(key) in parsed for key in required)


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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def append(self, event: dict[str, Any]) -> None:
        payload = {"at": utc_now(), **event}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


class ExperimentRunner:
    def __init__(
        self,
        config: Q1Config,
        backend: ChatBackend | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or OpenAICompatibleBackend(config.llm)
        self.learned_store = LearnedGraphStore(config.learned_graph_root)
        self.experiment = config.experiment
        self._semaphore = asyncio.Semaphore(int(config.llm.get("max_concurrent_requests", 6)))

    async def _complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        seed: int,
    ) -> Completion:
        async with self._semaphore:
            return await self.backend.complete(system, user, max_tokens=max_tokens, seed=seed)

    def _run_dir(self, task: TaskPackage, organization: str, seed: int) -> Path:
        return (
            self.config.results_root
            / "runs"
            / task.domain
            / task.task_structure
            / organization
            / str(seed)
            / safe_component(task.task_id)
        )

    def _make_budget(self) -> BudgetState:
        return BudgetState(
            max_total_tokens=int(self.experiment["max_total_tokens"]),
            final_reserve=int(self.experiment["final_token_reserve"]),
            max_messages=int(self.experiment["max_messages"]),
            usage=Usage(),
        )

    def _record_delivery(
        self,
        recorder: EventRecorder,
        budget: BudgetState,
        *,
        sender: int,
        recipient: int,
        round_index: int,
        phase: str,
        content: str,
    ) -> bool:
        delivered = budget.messages_delivered < budget.max_messages
        budget.record_message(content, delivered)
        event = {
            "event": "message",
            "phase": phase,
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

    def _inbound_from_edges(
        self,
        previous: list[str],
        graph: dict[int, list[int]],
        recorder: EventRecorder,
        budget: BudgetState,
        *,
        round_index: int,
        phase: str,
    ) -> dict[int, list[tuple[int, str]]]:
        inbound: dict[int, list[tuple[int, str]]] = {node: [] for node in graph}
        for sender in sorted(graph):
            if not previous[sender]:
                continue
            for recipient in graph[sender]:
                if self._record_delivery(
                    recorder,
                    budget,
                    sender=sender,
                    recipient=recipient,
                    round_index=round_index,
                    phase=phase,
                    content=previous[sender],
                ):
                    inbound[recipient].append((sender, previous[sender]))
        return inbound

    async def _agent_call(
        self,
        task: TaskPackage,
        node: int,
        inbound: list[tuple[int, str]],
        *,
        run_seed: int,
        round_index: int,
        phase: str,
        recorder: EventRecorder,
    ) -> Completion:
        spec = get_domain(task.domain)
        prompt = f"""Domain: {spec.display_name}
Dataset: {spec.dataset}
Task ID: {task.task_id}
Task structure: {task.task_structure}
Node: {node}
Role: {task.roles[node]}
Role responsibility: {spec.role_brief(node, task.task_structure)}

Question:
{task.question}

Shared model-visible context:
{_json_text(task.shared_context)}

Your private evidence view:
{_evidence_text(task.evidence_views[node])}

Permitted tools:
{json.dumps(list(task.tools), ensure_ascii=False)}

Messages delivered through communication edges:
{_truncate_messages(inbound, int(self.experiment['communication_budget_chars']))}

Return a concise evidence-grounded finding for the next step. Do not return a final JSON answer unless
your role is explicitly responsible for final integration."""
        started = time.monotonic()
        completion = await self._complete(
            BASE_SYSTEM,
            prompt,
            max_tokens=int(self.config.llm.get("max_tokens", 700)),
            seed=_call_seed(run_seed, phase, node, round_index),
        )
        recorder.append(
            {
                "event": "agent_output",
                "phase": phase,
                "round": round_index,
                "node": node,
                "role": task.roles[node],
                "received_from": [sender for sender, _ in inbound],
                "private_evidence_ids": [unit.evidence_id for unit in task.evidence_views[node]],
                "output": completion.content,
                "output_sha256": sha256_text(completion.content),
                "usage": completion.usage.to_dict(),
                "finish_reason": completion.finish_reason,
                "request_attempts": completion.request_attempts,
                "latency_seconds": time.monotonic() - started,
            }
        )
        return completion

    async def _parse_or_repair(
        self,
        raw_answer: str,
        task: TaskPackage,
        run_seed: int,
        budget: BudgetState,
        recorder: EventRecorder,
    ) -> tuple[Any | None, int]:
        parsed = extract_json(raw_answer)
        if _answer_schema_is_satisfied(parsed, task.answer_schema):
            return parsed, 0
        attempts = int(self.experiment.get("parse_repair_attempts", 0))
        current = raw_answer
        for repair_index in range(1, attempts + 1):
            prompt = f"""Convert the following already-produced answer into valid JSON that satisfies the
given schema. Preserve the answer's substance exactly. Do not solve the task again, add evidence, or
change a label or number.

Schema:
{json.dumps(task.answer_schema, ensure_ascii=False, indent=2)}

Answer to reformat:
{current}

Return JSON only."""
            completion = await self._complete(
                "You are a deterministic output-format repair step.",
                prompt,
                max_tokens=int(self.config.llm.get("final_max_tokens", 1800)),
                seed=_call_seed(run_seed, "format_repair", 0, repair_index),
            )
            budget.usage.add(completion.usage)
            current = completion.content
            parsed = extract_json(current)
            recorder.append(
                {
                    "event": "format_repair",
                    "attempt": repair_index,
                    "output": current,
                    "usage": completion.usage.to_dict(),
                    "valid": _answer_schema_is_satisfied(parsed, task.answer_schema),
                }
            )
            if _answer_schema_is_satisfied(parsed, task.answer_schema):
                return parsed, repair_index
        return None, attempts

    async def _final_call(
        self,
        task: TaskPackage,
        inbound: list[tuple[int, str]],
        own_state: str,
        *,
        run_seed: int,
        budget: BudgetState,
        recorder: EventRecorder,
    ) -> tuple[str, Any | None, int]:
        sink = int(self.experiment.get("synthesizer_node", 0))
        spec = get_domain(task.domain)
        prompt = f"""You are the final answer synthesizer.

Question:
{task.question}

Shared model-visible context:
{_json_text(task.shared_context)}

Your private evidence view:
{_evidence_text(task.evidence_views[sink])}

Your latest local state:
{own_state or 'No earlier local state.'}

Final messages reachable through your graph neighborhood:
{_truncate_messages(inbound, int(self.experiment['communication_budget_chars']))}

{spec.final_instruction}
Required JSON schema:
{json.dumps(task.answer_schema, ensure_ascii=False, indent=2)}
Do not mention agent roles, graph topology, organization method, or hidden evaluation data."""
        started = time.monotonic()
        completion = await self._complete(
            BASE_SYSTEM,
            prompt,
            max_tokens=int(self.config.llm.get("final_max_tokens", 1800)),
            seed=_call_seed(run_seed, "final", sink, 0),
        )
        budget.usage.add(completion.usage)
        recorder.append(
            {
                "event": "final_output",
                "node": sink,
                "role": task.roles[sink],
                "received_from": [sender for sender, _ in inbound],
                "output": completion.content,
                "usage": completion.usage.to_dict(),
                "finish_reason": completion.finish_reason,
                "request_attempts": completion.request_attempts,
                "latency_seconds": time.monotonic() - started,
            }
        )
        parsed, repairs = await self._parse_or_repair(
            completion.content, task, run_seed, budget, recorder
        )
        return completion.content, parsed, repairs

    async def _run_single_agent(
        self,
        task: TaskPackage,
        run_seed: int,
        recorder: EventRecorder,
    ) -> dict[str, Any]:
        spec = get_domain(task.domain)
        budget = self._make_budget()
        prompt = f"""Domain: {spec.display_name}
Dataset: {spec.dataset}
Task ID: {task.task_id}
Task structure: {task.task_structure}

Question:
{task.question}

Shared model-visible context:
{_json_text(task.shared_context)}

Complete evidence union:
{_evidence_text(task.evidence_union())}

Permitted tools:
{json.dumps(list(task.tools), ensure_ascii=False)}

{spec.final_instruction}
Required JSON schema:
{json.dumps(task.answer_schema, ensure_ascii=False, indent=2)}"""
        started = time.monotonic()
        completion = await self._complete(
            SINGLE_SYSTEM,
            prompt,
            max_tokens=int(self.config.llm.get("final_max_tokens", 1800)),
            seed=_call_seed(run_seed, "single_agent"),
        )
        budget.usage.add(completion.usage)
        recorder.append(
            {
                "event": "single_agent_output",
                "output": completion.content,
                "visible_evidence_ids": [unit.evidence_id for unit in task.evidence_union()],
                "usage": completion.usage.to_dict(),
                "finish_reason": completion.finish_reason,
                "request_attempts": completion.request_attempts,
                "latency_seconds": time.monotonic() - started,
            }
        )
        parsed, repairs = await self._parse_or_repair(
            completion.content, task, run_seed, budget, recorder
        )
        return {
            "status": "completed",
            "raw_answer": completion.content,
            "parsed_answer": parsed,
            "parse_failed": parsed is None,
            "format_repair_attempts": repairs,
            "budget": budget.to_dict(),
            "watchdog_stop_reason": "",
        }

    async def _run_multi_agent(
        self,
        task: TaskPackage,
        organization: Organization,
        run_seed: int,
        recorder: EventRecorder,
    ) -> dict[str, Any]:
        assert organization.edges is not None
        graph = adjacency(organization.edges)
        sink = int(self.experiment.get("synthesizer_node", 0))
        budget = self._make_budget()
        previous = [""] * 6
        watchdog_reason = ""

        if task.task_structure == "sequential_dependency":
            inputs = sequential_inputs(organization.edges, sink)
            for layer_index, nodes in enumerate(sequential_layers(organization.edges, sink)):
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
                    for node in range(6)
                )
            )
            previous = [completion.content for completion in completions]
            for completion in completions:
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
                        for node in range(6)
                    )
                )
                current = [completion.content for completion in completions]
                for node, completion in enumerate(completions):
                    budget.usage.add(completion.usage)
                    repeat_counts[node][normalize_for_loop_detection(completion.content)] += 1
                state = tuple(normalize_for_loop_detection(value) for value in current)
                unchanged_rounds = unchanged_rounds + 1 if state == earlier_state else 0
                earlier_state = state
                previous = current
                if any(
                    counts.most_common(1)[0][1] >= int(self.experiment.get("repeat_limit", 3))
                    for counts in repeat_counts
                    if counts
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

    async def _run_external_coding(
        self,
        task: TaskPackage,
        organization: Organization,
        run_seed: int,
        recorder: EventRecorder,
    ) -> dict[str, Any]:
        settings = dict(self.config.executors.get("coding", {}))
        command = settings.get("command", [])
        if not isinstance(command, list) or not command:
            raise RuntimeError(
                "Coding requires executors.coding.command; no proxy evaluator is permitted"
            )
        request = {
            "protocol": "opendgo_q1_coding_executor_v1",
            "task": task.public_dict(),
            "organization": {
                "name": organization.name,
                "category": organization.category,
                "edges": [list(edge) for edge in organization.edges] if organization.edges else None,
            },
            "seed": run_seed,
            "frozen_config": self.config.frozen_public_config(),
        }
        started = time.monotonic()
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
            raise RuntimeError("coding executor did not return one JSON object on stdout")
        recorder.append(
            {
                "event": "coding_executor",
                "return_code": process.returncode,
                "runtime_seconds": time.monotonic() - started,
                "stderr_tail": stderr.decode("utf-8", errors="replace")[-2000:],
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
        task: TaskPackage,
        organization_name: str,
        run_seed: int,
    ) -> dict[str, Any]:
        run_dir = self._run_dir(task, organization_name, run_seed)
        result_path = run_dir / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        run_dir.mkdir(parents=True, exist_ok=True)
        recorder = EventRecorder(run_dir / "events.jsonl")
        organization = resolve_organization(
            organization_name, task, run_seed, self.learned_store
        )
        started = time.monotonic()
        recorder.append(
            {
                "event": "run_started",
                "task_id": task.task_id,
                "source_id": task.source_id,
                "domain": task.domain,
                "task_structure": task.task_structure,
                "organization": organization.name,
                "seed": run_seed,
                "visible_input_sha256": task.visible_input_sha256
                or task.compute_visible_hash(),
                "config_sha256": self.config.fingerprint,
            }
        )
        error = ""
        try:
            if get_domain(task.domain).requires_external_executor:
                outcome = await self._run_external_coding(
                    task, organization, run_seed, recorder
                )
            elif organization.edges is None:
                outcome = await self._run_single_agent(task, run_seed, recorder)
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
        result = {
            "protocol_version": "opendgo_q1_v1",
            "run_key": sha256_text(
                f"{task.task_id}|{task.task_structure}|{organization_name}|{run_seed}"
            ),
            "task_id": task.task_id,
            "source_id": task.source_id,
            "domain": task.domain,
            "dataset": get_domain(task.domain).dataset,
            "task_structure": task.task_structure,
            "organization": organization.name,
            "organization_category": organization.category,
            "organization_record": (
                topology_record(organization.name, organization.edges)
                if organization.edges is not None
                else {"name": "single_agent", "nodes": 1, "edges": []}
            ),
            "designer_metadata": organization.designer_metadata,
            "role_order": list(task.roles),
            "synthesizer_node": int(self.experiment.get("synthesizer_node", 0)),
            "seed": run_seed,
            "model": self.backend.model,
            "visible_input_sha256": task.visible_input_sha256 or task.compute_visible_hash(),
            "config_sha256": self.config.fingerprint,
            **outcome,
            "error": error,
            "runtime_seconds": time.monotonic() - started,
            "completed_at_utc": utc_now(),
        }
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(result_path)
        recorder.append(
            {
                "event": "run_finished",
                "status": result["status"],
                "result_sha256": sha256_json(result),
            }
        )
        return result

    def validate_prerequisites(
        self,
        tasks: list[TaskPackage],
        organizations: list[str],
        seeds: list[int],
    ) -> None:
        if any(task.domain == "coding" for task in tasks):
            command = self.config.executors.get("coding", {}).get("command", [])
            if not isinstance(command, list) or not command:
                raise RuntimeError(
                    "selected plan contains Coding but executors.coding.command is empty"
                )
        for method in organizations:
            if method in LEARNED_ORGANIZATIONS:
                errors = self.learned_store.validate_complete(method, tasks, seeds)
                if errors:
                    preview = "\n".join(errors[:20])
                    raise RuntimeError(
                        f"learned graph validation failed for {method} "
                        f"({len(errors)} errors):\n{preview}"
                    )

    def _write_plan(
        self,
        tasks: list[TaskPackage],
        organizations: list[str],
        seeds: list[int],
    ) -> Path:
        plan_identity = {
            "config_sha256": self.config.fingerprint,
            "tasks": [task.task_id for task in tasks],
            "organizations": organizations,
            "seeds": seeds,
        }
        plan_id = sha256_json(plan_identity)[:16]
        plan_root = self.config.results_root / "plans"
        plan_root.mkdir(parents=True, exist_ok=True)
        path = plan_root / f"plan_{plan_id}.jsonl"
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for task in tasks:
                    for seed in seeds:
                        for organization in organizations:
                            row = {
                                "task_id": task.task_id,
                                "source_id": task.source_id,
                                "domain": task.domain,
                                "task_structure": task.task_structure,
                                "organization": organization,
                                "seed": seed,
                                "visible_input_sha256": task.visible_input_sha256
                                or task.compute_visible_hash(),
                            }
                            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "plan_id": plan_id,
            "created_at_utc": utc_now(),
            "runs": len(tasks) * len(seeds) * len(organizations),
            "tasks": len(tasks),
            "organizations": organizations,
            "seeds": seeds,
            "frozen_config": self.config.frozen_public_config(),
        }
        (plan_root / f"plan_{plan_id}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    async def run_plan(
        self,
        tasks: list[TaskPackage],
        organizations: list[str],
        seeds: list[int],
    ) -> dict[str, Any]:
        self.validate_prerequisites(tasks, organizations, seeds)
        plan_path = self._write_plan(tasks, organizations, seeds)
        if any(task.domain != "coding" for task in tasks):
            health = await self.backend.health()
            print(json.dumps({"model_health": health}, ensure_ascii=False), flush=True)
        total = len(tasks) * len(organizations) * len(seeds)
        completed = 0
        statuses: Counter[str] = Counter()
        for task in tasks:
            for seed in seeds:
                for organization in organizations:
                    result = await self.run_one(task, organization, seed)
                    completed += 1
                    statuses[str(result["status"])] += 1
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": total,
                                "task_id": task.task_id,
                                "organization": organization,
                                "seed": seed,
                                "status": result["status"],
                                "runtime_seconds": round(float(result["runtime_seconds"]), 2),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        return {
            "plan_path": str(plan_path),
            "total": total,
            "statuses": dict(statuses),
        }

