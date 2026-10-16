from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .conditions import FIXED_EDGES
from .domains import DOMAIN_SPECS
from .schema import ALLOCATIONS, STRUCTURES
from .util import sha256_json


@dataclass(frozen=True)
class Q2Config:
    source_path: Path
    project_root: Path
    data_root: Path
    q1_data_root: Path
    results_root: Path
    llm: dict[str, Any]
    experiment: dict[str, Any]
    executors: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        public_llm = {key: value for key, value in self.llm.items() if key != "api_key"}
        return sha256_json({"llm": public_llm, "experiment": self.experiment})

    def frozen_public_config(self) -> dict[str, Any]:
        return {
            "llm": {key: value for key, value in self.llm.items() if key != "api_key"},
            "experiment": self.experiment,
            "config_sha256": self.fingerprint,
        }


def _resolve(root: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"paths.{key} must be a non-empty string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> Q2Config:
    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    root = source.parent.parent
    paths = dict(raw.get("paths", {}))
    llm = dict(raw.get("llm", {}))
    experiment = dict(raw.get("experiment", {}))
    executors = dict(raw.get("executors", {}))

    checks = (
        ("domains", DOMAIN_SPECS),
        ("task_structures", STRUCTURES),
        ("topologies", FIXED_EDGES),
        ("information_allocations", ALLOCATIONS),
    )
    for key, allowed in checks:
        values = list(experiment.get(key, []))
        unknown = sorted(set(values) - set(allowed))
        if not values or unknown:
            raise ValueError(f"experiment.{key} is empty or invalid: {unknown}")
    seeds = list(experiment.get("seeds", []))
    if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(x, int) for x in seeds):
        raise ValueError("experiment.seeds must contain unique integers")
    for key in (
        "tasks_per_cell",
        "communication_rounds",
        "communication_budget_chars",
        "max_total_tokens",
        "final_token_reserve",
        "max_messages",
        "bootstrap_resamples",
    ):
        if int(experiment.get(key, 0)) <= 0:
            raise ValueError(f"experiment.{key} must be positive")
    for key in ("base_url", "model"):
        if not str(llm.get(key, "")).strip():
            raise ValueError(f"llm.{key} must not be empty")

    return Q2Config(
        source_path=source,
        project_root=root,
        data_root=_resolve(root, paths.get("data_root"), "data_root"),
        q1_data_root=_resolve(root, paths.get("q1_data_root"), "q1_data_root"),
        results_root=_resolve(root, paths.get("results_root"), "results_root"),
        llm=llm,
        experiment=experiment,
        executors=executors,
    )

