from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .aggregate import aggregate_results
from .annotation_template import create_annotation_template
from .conditions import POSITIONS, PositionCondition, find_condition
from .config import Q2Config, load_config
from .data import (
    annotation_path,
    gold_path,
    load_annotations,
    load_tasks,
    task_path,
    validate_data,
)
from .evaluation import evaluate_tree
from .prepare import prepare_domain
from .runner import Q2Runner
from .schema import Q2Task
from .status import experiment_status


def _select_tasks(
    config: Q2Config,
    domain: str | None,
    structure: str | None,
    limit: int | None,
) -> list[Q2Task]:
    selected_domains = [domain] if domain else list(config.experiment["domains"])
    tasks = []
    for selected_domain in selected_domains:
        tasks.extend(
            task
            for task in load_tasks(task_path(config.data_root, selected_domain), selected_domain)
            if task.task_structure in config.experiment["task_structures"]
            and (structure is None or task.task_structure == structure)
        )
    if not limit:
        return sorted(tasks, key=lambda item: (item.domain, item.task_structure, item.task_id))
    grouped: dict[tuple[str, str], list[Q2Task]] = defaultdict(list)
    for task in sorted(tasks, key=lambda item: item.task_id):
        grouped[(task.domain, task.task_structure)].append(task)
    return [task for key in sorted(grouped) for task in grouped[key][:limit]]


def _select_conditions(
    config: Q2Config,
    topology: str | None,
    position: str | None,
) -> list[PositionCondition]:
    if topology and topology not in config.experiment["topologies"]:
        raise ValueError(f"topology {topology!r} is not enabled by this config")
    topologies = [topology] if topology else list(config.experiment["topologies"])
    if position:
        if len(topologies) != 1:
            raise ValueError("--position requires exactly one --topology")
        return [find_condition(topologies[0], position)]
    return [condition for name in topologies for condition in POSITIONS[name]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenDGO Q2 experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build one Q2 domain from Q1 tasks and annotations")
    prepare.add_argument("--config", type=Path, default=Path("configs/q2_tree.yaml"))
    prepare.add_argument("--domain", required=True)
    prepare.add_argument("--q1-tasks", type=Path)
    prepare.add_argument("--q1-gold", type=Path)
    prepare.add_argument("--annotations", type=Path)

    template = sub.add_parser("annotation-template", help="write a review-required annotation template")
    template.add_argument("--config", type=Path, default=Path("configs/q2_tree.yaml"))
    template.add_argument("--domain", required=True)
    template.add_argument("--output", type=Path)

    validate = sub.add_parser("validate", help="validate Q2 data and all selected interventions")
    validate.add_argument("--config", type=Path, default=Path("configs/q2_tree.yaml"))

    run = sub.add_parser("run", help="run selected Q2 cells")
    run.add_argument("--config", type=Path, default=Path("configs/q2_tree.yaml"))
    run.add_argument("--domain")
    run.add_argument("--structure")
    run.add_argument("--topology")
    run.add_argument("--position")
    run.add_argument("--allocation", choices=("centralized", "distributed", "conflicting"))
    run.add_argument("--seed", type=int)
    run.add_argument("--limit", type=int)

    evaluate = sub.add_parser("evaluate", help="score completed runs with evaluator-only gold")
    evaluate.add_argument("--config", type=Path, default=Path("configs/q2_tree.yaml"))

    aggregate = sub.add_parser("aggregate", help="build Q2 tables and figure data")
    aggregate.add_argument("--config", type=Path, default=Path("configs/q2_tree.yaml"))

    status = sub.add_parser("status", help="show plan and result progress")
    status.add_argument("--config", type=Path, default=Path("configs/q2_tree.yaml"))
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        domain = args.domain
        result = prepare_domain(
            domain,
            args.q1_tasks or task_path(config.q1_data_root, domain),
            args.q1_gold or gold_path(config.q1_data_root, domain),
            args.annotations or annotation_path(config.data_root, domain),
            task_path(config.data_root, domain),
            gold_path(config.data_root, domain),
        )
        _print(result)
        return
    if args.command == "annotation-template":
        output = args.output or annotation_path(config.data_root, args.domain)
        _print(create_annotation_template(task_path(config.q1_data_root, args.domain), output))
        return
    if args.command == "validate":
        result = validate_data(config)
        _print(result)
        if not result["ready"]:
            raise SystemExit(1)
        return
    if args.command == "run":
        tasks = _select_tasks(config, args.domain, args.structure, args.limit)
        if not tasks:
            raise ValueError("the selected filters contain no frozen Q2 tasks")
        conditions = _select_conditions(config, args.topology, args.position)
        modes = [args.allocation] if args.allocation else list(config.experiment["information_allocations"])
        seeds = [args.seed] if args.seed is not None else list(config.experiment["seeds"])
        if any(seed not in config.experiment["seeds"] for seed in seeds):
            raise ValueError("selected seed is not part of the frozen configuration")
        annotations = {}
        for selected_domain in sorted({task.domain for task in tasks}):
            annotations.update(load_annotations(annotation_path(config.data_root, selected_domain)))
        result = asyncio.run(Q2Runner(config).run_plan(tasks, annotations, conditions, modes, seeds))
        _print(result)
        return
    if args.command == "evaluate":
        result = evaluate_tree(config)
        _print(result)
        if not result["ready"]:
            raise SystemExit(1)
        return
    if args.command == "status":
        _print(experiment_status(config))
        return
    result = aggregate_results(config)
    _print(result)
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

