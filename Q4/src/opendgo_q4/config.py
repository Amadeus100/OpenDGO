from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from opendgo_q1.config import Q1Config
from opendgo_q1.domains import DOMAIN_SPECS
from opendgo_q1.schema import STRUCTURES
from opendgo_q1.topologies import FIXED_EDGES

from .util import sha256_json


@dataclass(frozen=True)
class Q4Config:
    source_path: Path
    project_root: Path
    q1_data_root: Path
    results_root: Path
    llm: dict[str, Any]
    experiment: dict[str, Any]
    executors: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        llm = {key: value for key, value in self.llm.items() if key != "api_key"}
        return sha256_json({"llm": llm, "experiment": self.experiment})

    def q1_config(self) -> Q1Config:
        experiment = {
            **self.experiment,
            "max_total_tokens": int(self.experiment["natural_max_total_tokens"]),
        }
        return Q1Config(
            source_path=self.source_path,
            project_root=self.project_root,
            data_root=self.q1_data_root,
            results_root=self.results_root,
            learned_graph_root=self.project_root / "unused_learned_graphs",
            llm=self.llm,
            experiment=experiment,
            executors=self.executors,
        )


def _path(root: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"paths.{key} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> Q4Config:
    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    root = source.parent.parent
    paths = dict(raw.get("paths", {}))
    llm = dict(raw.get("llm", {}))
    experiment = dict(raw.get("experiment", {}))
    executors = dict(raw.get("executors", {}))
    domains = list(experiment.get("domains", []))
    structures = list(experiment.get("task_structures", []))
    organizations = list(experiment.get("organizations", []))
    seeds = list(experiment.get("seeds", []))
    caps = [int(value) for value in experiment.get("token_caps", [])]
    pressures = [int(value) for value in experiment.get("fault_pressures_percent", [])]
    if not domains or set(domains) - set(DOMAIN_SPECS):
        raise ValueError("experiment.domains is empty or invalid")
    if not structures or set(structures) - set(STRUCTURES):
        raise ValueError("experiment.task_structures is empty or invalid")
    if not organizations or set(organizations) - set(FIXED_EDGES):
        raise ValueError("Q4 organizations must be fixed Q1 topologies")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("experiment.seeds must contain unique values")
    if not caps or caps != sorted(set(caps)) or caps[0] <= int(experiment["final_token_reserve"]):
        raise ValueError("token_caps must be sorted unique values above final reserve")
    if not pressures or pressures != sorted(set(pressures)) or pressures[0] != 0 or pressures[-1] > 100:
        raise ValueError("fault pressures must be sorted, unique, start at 0, and not exceed 100")
    agent_probability = float(experiment.get("fault_agent_probability", 0.5))
    channel_probability = float(experiment.get("fault_channel_probability", 0.5))
    if abs(agent_probability - 0.5) > 1e-9 or abs(channel_probability - 0.5) > 1e-9:
        raise ValueError("final Q4 protocol requires 0.5 agent and 0.5 channel probability")
    for key in ("base_url", "model"):
        if not str(llm.get(key, "")).strip():
            raise ValueError(f"llm.{key} must not be empty")
    return Q4Config(
        source_path=source,
        project_root=root,
        q1_data_root=_path(root, paths.get("q1_data_root"), "q1_data_root"),
        results_root=_path(root, paths.get("results_root"), "results_root"),
        llm=llm,
        experiment=experiment,
        executors=executors,
    )
