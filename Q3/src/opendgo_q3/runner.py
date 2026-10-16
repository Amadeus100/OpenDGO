from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from opendgo_q1.data import domain_task_path, load_tasks
from opendgo_q1.domains import get_domain
from opendgo_q1.organizations import Organization
from opendgo_q1.runner import EventRecorder, ExperimentRunner
from opendgo_q1.schema import TaskPackage
from opendgo_q1.util import safe_component, sha256_json, sha256_text, utc_now

from .config import Q3Config
from .graph_library import load_graph_records
from .util import write_json, write_jsonl


def select_tasks(
    config: Q3Config,
    domains: list[str] | None = None,
    structures: list[str] | None = None,
    limit: int | None = None,
) -> list[TaskPackage]:
    domain_names = domains or list(config.experiment["domains"])
    structure_names = set(structures or config.experiment["task_structures"])
    selected: list[TaskPackage] = []
    for domain in domain_names:
        tasks = load_tasks(domain_task_path(config.q1_data_root, domain), domain)
        for structure in sorted(structure_names):
            cell = sorted(
                (task for task in tasks if task.task_structure == structure),
                key=lambda task: task.task_id,
            )
            selected.extend(cell[:limit] if limit is not None else cell)
    return selected


class Q3Runner(ExperimentRunner):
    def __init__(self, config: Q3Config) -> None:
        super().__init__(config.q1_config())
        self.q3_config = config

    def _q3_run_dir(self, task: TaskPackage, graph_id: str, seed: int) -> Path:
        return (
            self.q3_config.results_root
            / "runs"
            / task.domain
            / task.task_structure
            / graph_id
            / str(seed)
            / safe_component(task.task_id)
        )

    async def run_graph(
        self, task: TaskPackage, graph_record: dict[str, Any], run_seed: int
    ) -> dict[str, Any]:
        graph_id = str(graph_record["graph_id"])
        run_dir = self._q3_run_dir(task, graph_id, run_seed)
        result_path = run_dir / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        run_dir.mkdir(parents=True, exist_ok=True)
        recorder = EventRecorder(run_dir / "events.jsonl")
        organization = Organization(
            name=graph_id,
            category=str(graph_record["graph_category"]),
            edges=tuple(tuple(int(node) for node in edge) for edge in graph_record["edges"]),
            designer_metadata={"q3_properties": graph_record["properties"]},
        )
        started = time.monotonic()
        recorder.append(
            {
                "event": "run_started",
                "protocol": "opendgo_q3_v1",
                "task_id": task.task_id,
                "domain": task.domain,
                "task_structure": task.task_structure,
                "graph_id": graph_id,
                "seed": run_seed,
                "visible_input_sha256": task.visible_input_sha256 or task.compute_visible_hash(),
                "config_sha256": self.q3_config.fingerprint,
            }
        )
        error = ""
        try:
            if get_domain(task.domain).requires_external_executor:
                outcome = await self._run_external_coding(task, organization, run_seed, recorder)
            else:
                outcome = await self._run_multi_agent(task, organization, run_seed, recorder)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            recorder.append({"event": "run_error", "error": error})
            outcome = {
                "status": "incomplete",
                "raw_answer": "",
                "parsed_answer": None,
                "parse_failed": True,
                "format_repair_attempts": 0,
                "budget": {},
                "watchdog_stop_reason": "execution_error",
            }
        result = {
            "protocol_version": "opendgo_q3_v1",
            "run_key": sha256_text(f"q3|{task.task_id}|{graph_id}|{run_seed}"),
            "task_id": task.task_id,
            "source_id": task.source_id,
            "domain": task.domain,
            "dataset": get_domain(task.domain).dataset,
            "task_structure": task.task_structure,
            "organization": graph_id,
            "organization_category": organization.category,
            "graph_id": graph_id,
            "graph_record": graph_record,
            "graph_properties": graph_record["properties"],
            "role_order": list(task.roles),
            "synthesizer_node": int(self.experiment.get("synthesizer_node", 0)),
            "communication_rounds": int(self.experiment["communication_rounds"]),
            "seed": run_seed,
            "model": self.backend.model,
            "visible_input_sha256": task.visible_input_sha256 or task.compute_visible_hash(),
            "config_sha256": self.q3_config.fingerprint,
            **outcome,
            "error": error,
            "runtime_seconds": time.monotonic() - started,
            "completed_at_utc": utc_now(),
        }
        write_json(result_path, result)
        recorder.append(
            {"event": "run_finished", "status": result["status"], "result_sha256": sha256_json(result)}
        )
        return result

    async def run_q3_plan(
        self,
        tasks: list[TaskPackage],
        graph_ids: list[str],
        seeds: list[int],
    ) -> dict[str, Any]:
        graphs = load_graph_records(self.q3_config.graph_library_root)
        missing = sorted(set(graph_ids) - set(graphs))
        if missing:
            raise ValueError(f"unknown or unfrozen graph IDs: {missing}")
        if any(task.domain == "coding" for task in tasks):
            command = self.config.executors.get("coding", {}).get("command", [])
            if not command:
                raise RuntimeError("selected tasks include Coding but coding executor is not configured")
        plan_rows = [
            {
                "task_id": task.task_id,
                "source_id": task.source_id,
                "domain": task.domain,
                "task_structure": task.task_structure,
                "graph_id": graph_id,
                "seed": seed,
            }
            for task in tasks
            for seed in seeds
            for graph_id in graph_ids
        ]
        plan_id = sha256_json(
            {"config": self.q3_config.fingerprint, "rows": plan_rows}
        )[:16]
        write_jsonl(self.q3_config.results_root / "plans" / f"plan_{plan_id}.jsonl", plan_rows)
        if any(task.domain != "coding" for task in tasks):
            health = await self.backend.health()
            print(json.dumps({"model_health": health}, ensure_ascii=False), flush=True)
        statuses: Counter[str] = Counter()
        total = len(plan_rows)
        completed = 0
        for task in tasks:
            for seed in seeds:
                for graph_id in graph_ids:
                    result = await self.run_graph(task, graphs[graph_id], seed)
                    completed += 1
                    statuses[str(result["status"])] += 1
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": total,
                                "task_id": task.task_id,
                                "graph_id": graph_id,
                                "seed": seed,
                                "status": result["status"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        return {"plan_id": plan_id, "total": total, "statuses": dict(statuses)}


async def run_selected(
    config: Q3Config,
    *,
    domains: list[str] | None = None,
    structures: list[str] | None = None,
    graph_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    tasks = select_tasks(config, domains, structures, limit)
    frozen = load_graph_records(config.graph_library_root)
    runner = Q3Runner(config)
    try:
        return await runner.run_q3_plan(
            tasks,
            graph_ids or sorted(frozen),
            seeds or list(config.experiment["seeds"]),
        )
    finally:
        close = getattr(runner.backend, "close", None)
        if close is not None:
            await close()
