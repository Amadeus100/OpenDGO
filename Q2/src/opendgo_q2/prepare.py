from __future__ import annotations

from pathlib import Path
from typing import Any

from .domains import RESPONSIBILITIES
from .schema import EvidenceUnit, InterventionAnnotation, Q2Task
from .util import read_jsonl, sha256_json, write_jsonl


FORBIDDEN_TASK_KEYS = {
    "answer",
    "gold",
    "gold_answer",
    "correct_answer",
    "resolved",
    "official_evaluation",
}


def _evidence_union(q1_row: dict[str, Any]) -> tuple[EvidenceUnit, ...]:
    raw_views = q1_row.get("evidence_views")
    if not isinstance(raw_views, list) or len(raw_views) != 6:
        raise ValueError(f"{q1_row.get('task_id')}: Q1 evidence_views must have six entries")
    by_id: dict[str, EvidenceUnit] = {}
    for view in raw_views:
        if not isinstance(view, list):
            raise ValueError("each Q1 evidence view must be a list")
        for raw_unit in view:
            unit = EvidenceUnit.from_dict(raw_unit)
            earlier = by_id.get(unit.evidence_id)
            if earlier is not None and earlier != unit:
                raise ValueError(f"inconsistent copies of evidence {unit.evidence_id}")
            by_id[unit.evidence_id] = unit
    return tuple(by_id[key] for key in sorted(by_id))


def prepare_domain(
    domain: str,
    q1_tasks_path: Path,
    q1_gold_path: Path,
    annotations_path: Path,
    output_tasks_path: Path,
    output_gold_path: Path,
) -> dict[str, Any]:
    annotation_rows = read_jsonl(annotations_path)
    annotations = {
        item.task_id: item
        for item in (InterventionAnnotation.from_dict(row) for row in annotation_rows)
    }
    if len(annotations) != len(annotation_rows):
        raise ValueError(f"duplicate annotation task ID in {annotations_path}")

    output_tasks: list[dict[str, Any]] = []
    q1_rows = read_jsonl(q1_tasks_path)
    for row in q1_rows:
        leaked = FORBIDDEN_TASK_KEYS.intersection(row)
        if leaked:
            raise ValueError(f"Q1 public task contains forbidden keys: {sorted(leaked)}")
        if str(row.get("domain")) != domain:
            raise ValueError(f"{q1_tasks_path} contains a non-{domain} task")
        task_id = str(row["task_id"])
        if task_id not in annotations:
            raise ValueError(f"{task_id}: missing Q2 intervention annotation")
        task = Q2Task(
            task_id=task_id,
            source_id=str(row["source_id"]),
            domain=domain,
            task_structure=str(row["task_structure"]),
            question=str(row["question"]),
            roles=tuple(str(item) for item in row["roles"]),
            responsibilities=RESPONSIBILITIES[domain],
            evidence_units=_evidence_union(row),
            shared_context=row.get("shared_context", {}),
            tools=tuple(str(item) for item in row.get("tools", [])),
            answer_schema=dict(row.get("answer_schema", {})),
            provenance={
                **dict(row.get("provenance", {})),
                "q2_annotation_sha256": sha256_json(
                    next(item for item in annotation_rows if str(item["task_id"]) == task_id)
                ),
                "q2_annotation_source": annotations[task_id].annotation_source,
            },
            q1_visible_input_sha256=str(row.get("visible_input_sha256", "")),
        )
        task.validate()
        annotations[task_id].validate_for(task)
        output_tasks.append(task.public_dict())

    task_ids = {str(item["task_id"]) for item in output_tasks}
    if set(annotations) != task_ids:
        extra = sorted(set(annotations) - task_ids)
        raise ValueError(f"annotations contain tasks absent from Q1 package: {extra[:20]}")

    gold_rows = read_jsonl(q1_gold_path)
    selected_gold = [row for row in gold_rows if str(row.get("task_id")) in task_ids]
    gold_ids = {str(row["task_id"]) for row in selected_gold}
    if gold_ids != task_ids or len(selected_gold) != len(gold_ids):
        raise ValueError("Q1 gold IDs do not match Q2 task IDs exactly")

    write_jsonl(output_tasks_path, output_tasks)
    write_jsonl(output_gold_path, selected_gold)
    return {
        "domain": domain,
        "tasks_written": len(output_tasks),
        "tasks_path": str(output_tasks_path),
        "gold_path": str(output_gold_path),
        "annotations_path": str(annotations_path),
    }

