from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .domains import DOMAIN_SPECS
from .organizations import ALL_ORGANIZATIONS
from .schema import N_AGENTS, STRUCTURES
from .util import sha256_json


@dataclass(frozen=True)
class Q1Config:
    source_path: Path
    project_root: Path
    data_root: Path
    results_root: Path
    learned_graph_root: Path
    llm: dict[str, Any]
    experiment: dict[str, Any]
    executors: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        llm = {key: value for key, value in self.llm.items() if key != "api_key"}
        return sha256_json({"llm": llm, "experiment": self.experiment})

    def frozen_public_config(self) -> dict[str, Any]:
        return {
            "llm": {key: value for key, value in self.llm.items() if key != "api_key"},
            "experiment": self.experiment,
            "config_sha256": self.fingerprint,
        }


def _resolve(project_root: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"paths.{key} must be a non-empty string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: Path) -> Q1Config:
    source_path = path.resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    project_root = source_path.parent.parent
    paths = dict(raw.get("paths", {}))
    llm = dict(raw.get("llm", {}))
    experiment = dict(raw.get("experiment", {}))
    executors = dict(raw.get("executors", {}))

    domains = list(experiment.get("domains", []))
    structures = list(experiment.get("task_structures", []))
    organizations = list(experiment.get("organizations", []))
    seeds = list(experiment.get("seeds", []))
    unknown_domains = sorted(set(domains) - set(DOMAIN_SPECS))
    unknown_structures = sorted(set(structures) - set(STRUCTURES))
    unknown_organizations = sorted(set(organizations) - set(ALL_ORGANIZATIONS))
    if unknown_domains:
        raise ValueError(f"unknown domains: {unknown_domains}")
    if unknown_structures:
        raise ValueError(f"unknown task structures: {unknown_structures}")
    if unknown_organizations:
        raise ValueError(f"unknown organizations: {unknown_organizations}")
    if not domains or not structures or not organizations or not seeds:
        raise ValueError("domains, task_structures, organizations, and seeds must be non-empty")
    if len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("experiment.seeds must contain unique integers")
    if int(experiment.get("synthesizer_node", 0)) not in range(N_AGENTS):
        raise ValueError("experiment.synthesizer_node must be in 0..5")
    for positive_key in (
        "tasks_per_cell",
        "communication_rounds",
        "communication_budget_chars",
        "max_total_tokens",
        "final_token_reserve",
        "max_messages",
    ):
        if int(experiment.get(positive_key, 0)) <= 0:
            raise ValueError(f"experiment.{positive_key} must be positive")
    for key in ("base_url", "model"):
        if not str(llm.get(key, "")).strip():
            raise ValueError(f"llm.{key} must not be empty")

    return Q1Config(
        source_path=source_path,
        project_root=project_root,
        data_root=_resolve(project_root, paths.get("data_root"), "data_root"),
        results_root=_resolve(project_root, paths.get("results_root"), "results_root"),
        learned_graph_root=_resolve(
            project_root, paths.get("learned_graph_root"), "learned_graph_root"
        ),
        llm=llm,
        experiment=experiment,
        executors=executors,
    )

