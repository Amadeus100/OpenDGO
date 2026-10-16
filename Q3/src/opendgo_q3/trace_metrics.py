from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from opendgo_q1.data import domain_task_path, load_tasks
from opendgo_q1.schema import TaskPackage
from opendgo_q1.util import estimate_tokens

from .config import Q3Config
from .util import read_jsonl, write_json


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _annotations(config: Q3Config, domain: str) -> dict[str, dict[str, Any]]:
    path = config.trace_annotation_root / domain / "trace_annotations.jsonl"
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(row["task_id"]): row for row in rows}


def _tasks(config: Q3Config, domain: str) -> dict[str, TaskPackage]:
    return {
        task.task_id: task
        for task in load_tasks(domain_task_path(config.q1_data_root, domain), domain)
    }


def _mentions(text: str, evidence_ids: set[str]) -> set[str]:
    return {
        evidence_id
        for evidence_id in evidence_ids
        if re.search(rf"(?<![\w-]){re.escape(evidence_id)}(?![\w-])", text)
    }


def _jaccard(left: str, right: str) -> float:
    a = {token.lower() for token in TOKEN_RE.findall(left)}
    b = {token.lower() for token in TOKEN_RE.findall(right)}
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a and b else 0.0


def _partition_map(partition: Any) -> dict[int, int]:
    if not isinstance(partition, list) or len(partition) < 2:
        return {}
    return {int(node): group for group, nodes in enumerate(partition) for node in nodes}


def compute_trace_metrics(
    result: dict[str, Any], events: list[dict[str, Any]], task: TaskPackage,
    annotation: dict[str, Any], duplicate_threshold: float,
) -> dict[str, Any]:
    sink = int(result.get("synthesizer_node", 0))
    all_evidence_ids = {unit.evidence_id for unit in task.evidence_union()}
    gold_ids = {str(value) for value in annotation.get("gold_supporting_evidence_ids", [])}
    sink_private = {unit.evidence_id for unit in task.evidence_views[sink]}
    delivered = [
        event for event in events
        if event.get("event") == "message" and bool(event.get("delivered"))
    ]
    sink_messages = [event for event in delivered if int(event.get("recipient", -1)) == sink]
    reached = set(sink_private)
    for event in sink_messages:
        reached.update(_mentions(str(event.get("content", "")), all_evidence_ids))
    coverage = len(reached & gold_ids) / len(gold_ids) if gold_ids else None

    max_round = int(result.get("communication_rounds", 0))
    cumulative = set(sink_private)
    latency: int | None = None
    for round_index in sorted({max(0, int(event.get("round", 0))) for event in sink_messages}):
        for event in sink_messages:
            if max(0, int(event.get("round", 0))) <= round_index:
                cumulative.update(_mentions(str(event.get("content", "")), all_evidence_ids))
        if gold_ids and gold_ids <= cumulative:
            latency = round_index
            break
    latency_censored = bool(gold_ids and latency is None)
    if latency_censored:
        latency = max_round + 1

    duplicate_count = 0
    earlier: list[str] = []
    for event in delivered:
        content = str(event.get("content", ""))
        duplicate_count += int(any(_jaccard(content, old) >= duplicate_threshold for old in earlier))
        earlier.append(content)
    duplicate_rate = duplicate_count / len(delivered) if delivered else 0.0

    conflicts = annotation.get("conflicts", [])
    raw_answer = str(result.get("raw_answer", ""))
    explicit = result.get("parsed_answer", {})
    explicit_ids = set(explicit.get("resolved_conflict_evidence_ids", [])) if isinstance(explicit, dict) else set()
    contradiction: float | None
    if not conflicts:
        contradiction = None
        contradiction_status = "not_applicable"
    elif any(not str(item.get("resolution_regex", "")).strip() for item in conflicts):
        contradiction = None
        contradiction_status = "unavailable_missing_audit_rule"
    else:
        resolved = sum(
            str(item["evidence_id"]) in explicit_ids
            or bool(re.search(str(item["resolution_regex"]), raw_answer))
            for item in conflicts
        )
        contradiction = resolved / len(conflicts)
        contradiction_status = "audited"

    groups = _partition_map(result.get("graph_record", {}).get("community_partition"))
    evidence_messages = [
        event for event in delivered
        if _mentions(str(event.get("content", "")), all_evidence_ids)
    ]
    cross = sum(
        groups.get(int(event["sender"])) != groups.get(int(event["recipient"]))
        for event in evidence_messages
    )
    cross_flow = cross / len(evidence_messages) if groups and evidence_messages else None
    final_event = next(
        (event for event in reversed(events) if event.get("event") == "final_output"), None
    )
    provider_prompt_tokens = (
        int(final_event.get("usage", {}).get("prompt_tokens", 0))
        if isinstance(final_event, dict)
        else 0
    )
    if provider_prompt_tokens > 0:
        synth_load = provider_prompt_tokens
        synth_load_method = "provider-reported final synthesis prompt tokens"
    else:
        final_messages = [event for event in sink_messages if event.get("phase") == "final_delivery"]
        latest_sink_output = next(
            (
                str(event.get("output", ""))
                for event in reversed(events)
                if event.get("event") == "agent_output" and int(event.get("node", -1)) == sink
            ),
            "",
        )
        synth_load = estimate_tokens(
            task.question
            + json.dumps(task.shared_context, ensure_ascii=False)
            + json.dumps(task.answer_schema, ensure_ascii=False)
            + "".join(unit.text for unit in task.evidence_views[sink])
            + latest_sink_output
            + "".join(str(event.get("content", "")) for event in final_messages)
        )
        synth_load_method = "ceil(chars/4) reconstruction of final synthesis prompt"
    return {
        "run_key": result["run_key"],
        "task_id": result["task_id"],
        "domain": result["domain"],
        "task_structure": result["task_structure"],
        "graph_id": result["graph_id"],
        "seed": result["seed"],
        "evidence_coverage": coverage,
        "evidence_coverage_status": "audited" if gold_ids else "unavailable_missing_annotation",
        "propagation_latency": latency,
        "propagation_latency_censored": latency_censored,
        "duplicate_message_rate": duplicate_rate,
        "contradiction_resolution": contradiction,
        "contradiction_resolution_status": contradiction_status,
        "cross_group_flow": cross_flow,
        "cross_group_flow_status": "computed" if cross_flow is not None else "not_applicable",
        "synthesizer_load_tokens_estimated": synth_load,
        "synthesizer_load_method": synth_load_method,
    }


def extract_tree(config: Q3Config) -> dict[str, Any]:
    task_cache: dict[str, dict[str, TaskPackage]] = {}
    annotation_cache: dict[str, dict[str, dict[str, Any]]] = {}
    written = 0
    errors: list[str] = []
    threshold = float(config.experiment.get("duplicate_jaccard_threshold", 0.90))
    for result_path in sorted((config.results_root / "runs").rglob("result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            domain = str(result["domain"])
            if domain not in task_cache:
                task_cache[domain] = _tasks(config, domain)
            if domain not in annotation_cache:
                annotation_cache[domain] = _annotations(config, domain)
            events = read_jsonl(result_path.parent / "events.jsonl")
            metrics = compute_trace_metrics(
                result,
                events,
                task_cache[domain][str(result["task_id"])],
                annotation_cache[domain].get(str(result["task_id"]), {}),
                threshold,
            )
            write_json(result_path.parent / "trace_metrics.json", metrics)
            written += 1
        except Exception as exc:
            errors.append(f"{result_path}: {type(exc).__name__}: {exc}")
    summary = {"written": written, "errors": errors, "ready": written > 0 and not errors}
    write_json(config.results_root / "trace_metrics_summary.json", summary)
    return summary
