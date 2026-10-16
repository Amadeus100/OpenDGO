from __future__ import annotations

from pathlib import Path
from typing import Any

from .domains import get_domain
from .schema import EvidenceUnit, GoldRecord, N_AGENTS, STRUCTURES, TaskPackage
from .util import read_jsonl, safe_component, write_jsonl


def _units_by_id(raw: dict[str, Any]) -> dict[str, EvidenceUnit]:
    units: dict[str, EvidenceUnit] = {}
    for value in raw.get("evidence_units", []):
        unit = EvidenceUnit.from_dict(value)
        if unit.evidence_id in units:
            raise ValueError(f"{raw.get('source_id')}: duplicate evidence ID {unit.evidence_id}")
        units[unit.evidence_id] = unit
    if not units:
        raise ValueError(f"{raw.get('source_id')}: evidence_units must not be empty")
    return units


def _explicit_views(
    source_id: str,
    structure: str,
    raw_views: Any,
    units: dict[str, EvidenceUnit],
) -> tuple[tuple[EvidenceUnit, ...], ...]:
    if not isinstance(raw_views, list) or len(raw_views) != N_AGENTS:
        raise ValueError(f"{source_id}/{structure}: structure view must contain six lists")
    views: list[tuple[EvidenceUnit, ...]] = []
    assigned: set[str] = set()
    for raw_view in raw_views:
        if not isinstance(raw_view, list):
            raise ValueError(f"{source_id}/{structure}: every evidence view must be a list")
        unknown = [evidence_id for evidence_id in raw_view if evidence_id not in units]
        if unknown:
            raise ValueError(f"{source_id}/{structure}: unknown evidence IDs {unknown}")
        assigned.update(str(evidence_id) for evidence_id in raw_view)
        views.append(tuple(units[str(evidence_id)] for evidence_id in raw_view))
    if assigned != set(units):
        raise ValueError(
            f"{source_id}/{structure}: every evidence unit must be assigned at least once; "
            f"missing={sorted(set(units) - assigned)}"
        )
    return tuple(views)


def _automatic_views(
    source_id: str,
    structure: str,
    units: dict[str, EvidenceUnit],
) -> tuple[tuple[EvidenceUnit, ...], ...]:
    views: list[list[EvidenceUnit]] = [[] for _ in range(N_AGENTS)]
    ordered = [units[key] for key in sorted(units)]
    for index, unit in enumerate(ordered):
        if unit.role_hint is not None:
            node = unit.role_hint
        elif structure == "sequential_dependency" and unit.stage is not None:
            node = min(5, max(1, unit.stage))
        else:
            node = 1 + (index % 5)
        views[node].append(unit)
    if structure == "evidence_integration":
        holders = {
            node
            for node, view in enumerate(views)
            if any(unit.required for unit in view)
        }
        if len(holders) < 2:
            raise ValueError(
                f"{source_id}: automatic evidence integration requires required=true units "
                "assigned to at least two roles; provide structure_views to fix the allocation"
            )
    return tuple(tuple(view) for view in views)


def prepare_domain(
    input_path: Path,
    data_root: Path,
    domain: str,
) -> dict[str, Any]:
    spec = get_domain(domain)
    task_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for raw in read_jsonl(input_path):
        source_id = str(raw["source_id"]).strip()
        if not source_id or source_id in seen_sources:
            raise ValueError(f"source_id must be non-empty and unique: {source_id!r}")
        seen_sources.add(source_id)
        units = _units_by_id(raw)
        structure_views = dict(raw.get("structure_views", {}))
        for structure in STRUCTURES:
            views = (
                _explicit_views(source_id, structure, structure_views[structure], units)
                if structure in structure_views
                else _automatic_views(source_id, structure, units)
            )
            if structure == "evidence_integration":
                required_holders = {
                    node
                    for node, view in enumerate(views)
                    if any(unit.required for unit in view)
                }
                if len(required_holders) < 2:
                    raise ValueError(
                        f"{source_id}: evidence integration must split required evidence across >=2 roles"
                    )
            task_id = f"{domain}__{safe_component(source_id)}__{structure}"
            provenance = {
                "dataset": str(raw.get("dataset", spec.dataset)),
                "dataset_version": str(raw.get("dataset_version", "")),
                "split": str(raw.get("split", "")),
                "source_document_id": str(raw.get("source_document_id", "")),
                "selection_rule": str(raw.get("selection_rule", "pre-frozen source order")),
                "difficulty": str(raw.get("difficulty", "")),
                "source_category": str(raw.get("source_category", "")),
            }
            answer_schema = dict(
                raw.get(
                    "answer_schema",
                    {"type": "object", "required": [spec.answer_key]},
                )
            )
            task = TaskPackage(
                task_id=task_id,
                source_id=source_id,
                domain=domain,
                task_structure=structure,
                question=str(raw["question"]),
                roles=spec.roles,
                evidence_views=views,
                shared_context=raw.get("shared_context", {}),
                tools=tuple(str(tool) for tool in raw.get("tools", [])),
                answer_schema=answer_schema,
                provenance=provenance,
            )
            task.validate()
            task_rows.append(task.public_dict())
            gold_rows.append(
                GoldRecord(
                    task_id=task_id,
                    source_id=source_id,
                    domain=domain,
                    answer=raw.get("answer"),
                    metadata=dict(raw.get("gold_metadata", {})),
                ).to_dict()
            )
            audit_rows.append(
                {
                    "task_id": task_id,
                    "source_id": source_id,
                    "task_structure": structure,
                    "evidence_union": sorted(units),
                    "views": [
                        [
                            {
                                "evidence_id": unit.evidence_id,
                                "required": unit.required,
                                "truth_status": unit.truth_status,
                                "stage": unit.stage,
                            }
                            for unit in view
                        ]
                        for view in views
                    ],
                }
            )
    domain_root = data_root / domain
    write_jsonl(domain_root / "processed" / "tasks.jsonl", task_rows)
    write_jsonl(domain_root / "private" / "gold.jsonl", gold_rows)
    write_jsonl(domain_root / "private" / "allocation_audit.jsonl", audit_rows)
    return {
        "domain": domain,
        "sources": len(seen_sources),
        "tasks": len(task_rows),
        "per_structure": len(seen_sources),
        "task_path": str(domain_root / "processed" / "tasks.jsonl"),
        "gold_path": str(domain_root / "private" / "gold.jsonl"),
    }

