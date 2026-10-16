from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .config import Q1Config
from .domains import get_domain
from .organizations import LEARNED_ORGANIZATIONS, LearnedGraphStore
from .schema import GoldRecord, TaskPackage
from .util import read_jsonl


FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "gold",
    "gold_answer",
    "correct_answer",
    "resolved",
    "official_evaluation",
}


def domain_task_path(data_root: Path, domain: str) -> Path:
    return data_root / domain / "processed" / "tasks.jsonl"


def domain_gold_path(data_root: Path, domain: str) -> Path:
    return data_root / domain / "private" / "gold.jsonl"


def load_tasks(path: Path, expected_domain: str | None = None) -> list[TaskPackage]:
    rows = read_jsonl(path)
    tasks: list[TaskPackage] = []
    for line_number, row in enumerate(rows, start=1):
        leaked = FORBIDDEN_PUBLIC_KEYS.intersection(row)
        if leaked:
            raise ValueError(f"{path}:{line_number} contains forbidden public keys: {sorted(leaked)}")
        task = TaskPackage.from_dict(row)
        if expected_domain and task.domain != expected_domain:
            raise ValueError(f"{path}:{line_number} declares domain {task.domain!r}")
        spec = get_domain(task.domain)
        if task.roles != spec.roles:
            raise ValueError(f"{task.task_id}: frozen roles do not match domain registry")
        tasks.append(task)
    duplicates = sorted(
        task_id for task_id, count in Counter(task.task_id for task in tasks).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate task IDs in {path}: {duplicates[:10]}")
    return tasks


def load_gold(path: Path, expected_domain: str | None = None) -> dict[str, GoldRecord]:
    records: dict[str, GoldRecord] = {}
    for row in read_jsonl(path):
        record = GoldRecord.from_dict(row)
        if expected_domain and record.domain != expected_domain:
            raise ValueError(f"{path}: gold record {record.task_id} has wrong domain")
        if record.task_id in records:
            raise ValueError(f"duplicate gold task ID in {path}: {record.task_id}")
        records[record.task_id] = record
    return records


def load_config_tasks(config: Q1Config) -> list[TaskPackage]:
    selected_structures = set(config.experiment["task_structures"])
    tasks: list[TaskPackage] = []
    for domain in config.experiment["domains"]:
        domain_tasks = load_tasks(domain_task_path(config.data_root, domain), domain)
        tasks.extend(task for task in domain_tasks if task.task_structure in selected_structures)
    return tasks


def validate_data(config: Q1Config) -> dict[str, Any]:
    expected = int(config.experiment["tasks_per_cell"])
    report: dict[str, Any] = {"ready": True, "domains": {}, "learned_graph_errors": []}
    all_tasks: list[TaskPackage] = []
    for domain in config.experiment["domains"]:
        task_path = domain_task_path(config.data_root, domain)
        gold_path = domain_gold_path(config.data_root, domain)
        domain_report: dict[str, Any] = {
            "task_path": str(task_path),
            "gold_path": str(gold_path),
            "counts": {},
            "errors": [],
        }
        try:
            tasks = load_tasks(task_path, domain)
            gold = load_gold(gold_path, domain)
            task_ids = {task.task_id for task in tasks}
            gold_ids = set(gold)
            if task_ids != gold_ids:
                domain_report["errors"].append(
                    f"task/gold ID mismatch: tasks_only={len(task_ids - gold_ids)}, "
                    f"gold_only={len(gold_ids - task_ids)}"
                )
            for structure in config.experiment["task_structures"]:
                count = sum(task.task_structure == structure for task in tasks)
                domain_report["counts"][structure] = count
                if count != expected:
                    domain_report["errors"].append(
                        f"{structure}: expected {expected}, found {count}"
                    )
            all_tasks.extend(
                task
                for task in tasks
                if task.task_structure in set(config.experiment["task_structures"])
            )
        except Exception as exc:
            domain_report["errors"].append(f"{type(exc).__name__}: {exc}")
        domain_report["ready"] = not domain_report["errors"]
        report["ready"] = report["ready"] and domain_report["ready"]
        report["domains"][domain] = domain_report

    store = LearnedGraphStore(config.learned_graph_root)
    seeds = list(config.experiment["seeds"])
    for method in config.experiment["organizations"]:
        if method in LEARNED_ORGANIZATIONS:
            errors = store.validate_complete(method, all_tasks, seeds)
            report["learned_graph_errors"].extend(errors)
    if report["learned_graph_errors"]:
        report["ready"] = False
    return report

