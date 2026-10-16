from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from opendgo_q1.config import Q1Config
from opendgo_q1.domains import DOMAIN_SPECS
from opendgo_q1.schema import STRUCTURES

from .util import sha256_json


@dataclass(frozen=True)
class Q3Config:
    source_path: Path
    project_root: Path
    q1_data_root: Path
    graph_library_root: Path
    trace_annotation_root: Path
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
            "organizations": ["star"],
            "synthesizer_node": int(self.experiment.get("synthesizer_node", 0)),
        }
        return Q1Config(
            source_path=self.source_path,
            project_root=self.project_root,
            data_root=self.q1_data_root,
            results_root=self.results_root,
            learned_graph_root=self.graph_library_root,
            llm=self.llm,
            experiment=experiment,
            executors=self.executors,
        )


def _path(root: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"paths.{key} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> Q3Config:
    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    root = source.parent.parent
    paths = dict(raw.get("paths", {}))
    experiment = dict(raw.get("experiment", {}))
    llm = dict(raw.get("llm", {}))
    executors = dict(raw.get("executors", {}))
    domains = list(experiment.get("domains", []))
    structures = list(experiment.get("task_structures", []))
    seeds = list(experiment.get("seeds", []))
    if not domains or set(domains) - set(DOMAIN_SPECS):
        raise ValueError("experiment.domains is empty or contains unknown domains")
    if not structures or set(structures) - set(STRUCTURES):
        raise ValueError("experiment.task_structures is empty or invalid")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("experiment.seeds must contain unique values")
    for key in ("base_url", "model"):
        if not str(llm.get(key, "")).strip():
            raise ValueError(f"llm.{key} must not be empty")
    return Q3Config(
        source_path=source,
        project_root=root,
        q1_data_root=_path(root, paths.get("q1_data_root"), "q1_data_root"),
        graph_library_root=_path(root, paths.get("graph_library_root"), "graph_library_root"),
        trace_annotation_root=_path(
            root, paths.get("trace_annotation_root"), "trace_annotation_root"
        ),
        results_root=_path(root, paths.get("results_root"), "results_root"),
        llm=llm,
        experiment=experiment,
        executors=executors,
    )
