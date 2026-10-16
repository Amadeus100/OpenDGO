from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .allocation import allocate, validate_mode_family
from .conditions import POSITIONS, assignment_for
from .config import Q2Config
from .schema import ALLOCATIONS, GoldRecord, InterventionAnnotation, Q2Task
from .util import read_jsonl


def task_path(root: Path, domain: str) -> Path:
    return root / domain / "processed" / "tasks.jsonl"


def gold_path(root: Path, domain: str) -> Path:
    return root / domain / "private" / "gold.jsonl"


def annotation_path(root: Path, domain: str) -> Path:
    return root / domain / "annotations" / "interventions.jsonl"


def load_tasks(path: Path, expected_domain: str | None = None) -> list[Q2Task]:
    tasks = [Q2Task.from_dict(row) for row in read_jsonl(path)]
    if expected_domain and any(task.domain != expected_domain for task in tasks):
        raise ValueError(f"{path} contains a task from another domain")
    duplicates = [key for key, count in Counter(x.task_id for x in tasks).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate task IDs: {duplicates[:20]}")
    return tasks


def load_annotations(path: Path) -> dict[str, InterventionAnnotation]:
    rows = read_jsonl(path)
    values = [InterventionAnnotation.from_dict(row) for row in rows]
    result = {item.task_id: item for item in values}
    if len(result) != len(values):
        raise ValueError(f"duplicate annotation task ID in {path}")
    return result


def load_gold(path: Path, expected_domain: str | None = None) -> dict[str, GoldRecord]:
    records: dict[str, GoldRecord] = {}
    for row in read_jsonl(path):
        record = GoldRecord.from_dict(row)
        if expected_domain and record.domain != expected_domain:
            raise ValueError(f"{record.task_id}: wrong gold domain")
        if record.task_id in records:
            raise ValueError(f"duplicate gold task ID {record.task_id}")
        records[record.task_id] = record
    return records


def load_config_tasks(config: Q2Config) -> list[Q2Task]:
    structures = set(config.experiment["task_structures"])
    tasks: list[Q2Task] = []
    for domain in config.experiment["domains"]:
        tasks.extend(
            task
            for task in load_tasks(task_path(config.data_root, domain), domain)
            if task.task_structure in structures
        )
    return tasks


def load_config_annotations(config: Q2Config) -> dict[str, InterventionAnnotation]:
    combined: dict[str, InterventionAnnotation] = {}
    for domain in config.experiment["domains"]:
        for task_id, item in load_annotations(annotation_path(config.data_root, domain)).items():
            if task_id in combined:
                raise ValueError(f"task ID appears in multiple domains: {task_id}")
            combined[task_id] = item
    return combined


def validate_data(config: Q2Config) -> dict[str, Any]:
    expected = int(config.experiment["tasks_per_cell"])
    report: dict[str, Any] = {"ready": True, "domains": {}}
    for domain in config.experiment["domains"]:
        item: dict[str, Any] = {"errors": [], "counts": {}}
        try:
            tasks = load_tasks(task_path(config.data_root, domain), domain)
            annotations = load_annotations(annotation_path(config.data_root, domain))
            gold = load_gold(gold_path(config.data_root, domain), domain)
            task_ids = {task.task_id for task in tasks}
            if task_ids != set(annotations):
                item["errors"].append("task/annotation ID mismatch")
            if task_ids != set(gold):
                item["errors"].append("task/gold ID mismatch")
            for structure in config.experiment["task_structures"]:
                count = sum(task.task_structure == structure for task in tasks)
                item["counts"][structure] = count
                if count != expected:
                    item["errors"].append(f"{structure}: expected {expected}, found {count}")
            for task in tasks:
                annotation = annotations[task.task_id]
                annotation.validate_for(task)
                for topology in config.experiment["topologies"]:
                    equivalent_records: dict[str, list[tuple[Any, Any]]] = {
                        mode: [] for mode in ALLOCATIONS
                    }
                    for condition in POSITIONS[topology]:
                        assignment = assignment_for(condition, annotation.expert_role_index)
                        family = {
                            mode: allocate(task, annotation, assignment, mode)
                            for mode in ALLOCATIONS
                        }
                        validate_mode_family(family)
                        if condition.equivalent_control:
                            for mode, result in family.items():
                                equivalent_records[mode].append((assignment, result))
                    if topology in {"ring", "mesh"}:
                        for mode, records in equivalent_records.items():
                            base_assignment, base_result = records[0]
                            base_destinations = {
                                row["evidence_id"]: row["initial_node"]
                                for row in base_result.manifest["assignments"]
                            }
                            base_roles = list(base_assignment.node_to_role_index)
                            for assignment, result in records[1:]:
                                rotation = assignment.condition.rotation
                                expected_roles = [-1] * 6
                                for node, role in enumerate(base_roles):
                                    expected_roles[(node + rotation) % 6] = role
                                if list(assignment.node_to_role_index) != expected_roles:
                                    raise AssertionError("equivalent control did not rotate all roles")
                                observed = {
                                    row["evidence_id"]: row["initial_node"]
                                    for row in result.manifest["assignments"]
                                }
                                expected = {
                                    evidence_id: (node + rotation) % 6
                                    for evidence_id, node in base_destinations.items()
                                }
                                if observed != expected:
                                    raise AssertionError(
                                        f"{topology}/{mode}: equivalent control did not rotate all evidence"
                                    )
        except Exception as exc:
            item["errors"].append(f"{type(exc).__name__}: {exc}")
        item["ready"] = not item["errors"]
        report["ready"] = report["ready"] and item["ready"]
        report["domains"][domain] = item
    positions = sum(len(POSITIONS[name]) for name in config.experiment["topologies"])
    report["position_conditions"] = positions
    report["expected_runs"] = (
        len(config.experiment["domains"])
        * len(config.experiment["task_structures"])
        * positions
        * len(config.experiment["information_allocations"])
        * len(config.experiment["seeds"])
        * expected
    )
    return report

