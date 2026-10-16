from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import TaskPackage
from .topologies import FIXED_EDGES, normalize_edges
from .util import read_jsonl


SINGLE_AGENT = "single_agent"
LEARNED_ORGANIZATIONS = ("g_designer", "arg_designer", "goagent")
FIXED_ORGANIZATIONS = tuple(FIXED_EDGES)
ALL_ORGANIZATIONS = (SINGLE_AGENT, *FIXED_ORGANIZATIONS, *LEARNED_ORGANIZATIONS)


@dataclass(frozen=True)
class Organization:
    name: str
    category: str
    edges: tuple[tuple[int, int], ...] | None
    designer_metadata: dict[str, Any]


class LearnedGraphStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}

    def _load_method(self, method: str) -> dict[tuple[str, str, int], dict[str, Any]]:
        if method not in LEARNED_ORGANIZATIONS:
            raise KeyError(method)
        if method in self._cache:
            return self._cache[method]
        path = self.root / f"{method}.jsonl"
        rows = read_jsonl(path)
        records: dict[tuple[str, str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["task_id"]), str(row["task_structure"]), int(row["seed"]))
            if key in records:
                raise ValueError(f"duplicate learned graph key in {path}: {key}")
            normalized = normalize_edges(row["edges"])
            records[key] = {**row, "edges": normalized}
        self._cache[method] = records
        return records

    def resolve(self, method: str, task: TaskPackage, seed: int) -> Organization:
        records = self._load_method(method)
        key = (task.task_id, task.task_structure, seed)
        if key not in records:
            raise KeyError(f"missing {method} graph for task/structure/seed {key}")
        row = records[key]
        expected_hash = task.visible_input_sha256 or task.compute_visible_hash()
        artifact_hash = str(row.get("input_sha256", ""))
        if artifact_hash and artifact_hash != expected_hash:
            raise ValueError(f"{method} graph input hash does not match frozen task {task.task_id}")
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"task_id", "task_structure", "seed", "edges"}
        }
        return Organization(method, "learned", tuple(row["edges"]), metadata)

    def validate_complete(
        self, method: str, tasks: list[TaskPackage], seeds: list[int]
    ) -> list[str]:
        errors: list[str] = []
        try:
            records = self._load_method(method)
        except Exception as exc:
            return [f"{method}: {type(exc).__name__}: {exc}"]
        for task in tasks:
            for seed in seeds:
                key = (task.task_id, task.task_structure, seed)
                if key not in records:
                    errors.append(f"{method}: missing {key}")
        return errors


def resolve_organization(
    name: str,
    task: TaskPackage,
    seed: int,
    learned_store: LearnedGraphStore,
) -> Organization:
    if name == SINGLE_AGENT:
        return Organization(name, "baseline", None, {})
    if name in FIXED_EDGES:
        return Organization(name, "fixed", normalize_edges(FIXED_EDGES[name]), {})
    if name in LEARNED_ORGANIZATIONS:
        return learned_store.resolve(name, task, seed)
    raise KeyError(f"unknown organization: {name}")

