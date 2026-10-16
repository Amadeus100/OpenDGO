from __future__ import annotations

from pathlib import Path

from .util import read_jsonl, write_jsonl


def create_annotation_template(q1_tasks_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing annotation file: {output_path}")
    rows = []
    for task in read_jsonl(q1_tasks_path):
        evidence_ids = sorted(
            {
                str(unit["evidence_id"])
                for view in task.get("evidence_views", [])
                for unit in view
            }
        )
        rows.append(
            {
                "task_id": str(task["task_id"]),
                "expert_role_index": None,
                "decisive_evidence_ids": [],
                "supporting_evidence_ids": [],
                "conflicting_evidence_ids": [],
                "available_evidence_ids": evidence_ids,
                "annotation_source": "",
                "reviewed_by": "",
                "metadata": {"status": "requires_review"},
            }
        )
    write_jsonl(output_path, rows)
    return {"written": len(rows), "output": str(output_path), "status": "requires_review"}

