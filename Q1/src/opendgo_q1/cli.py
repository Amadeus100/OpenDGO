from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .aggregate import aggregate_results
from .config import Q1Config, load_config
from .data import load_config_tasks, validate_data
from .domains import DOMAIN_SPECS
from .evaluation import evaluate_tree
from .organizations import ALL_ORGANIZATIONS
from .prepare import prepare_domain
from .runner import ExperimentRunner
from .schema import STRUCTURES, TaskPackage
from .status import experiment_status


def _select_tasks(
    config: Q1Config,
    *,
    domains: list[str] | None,
    structures: list[str] | None,
    limit: int | None,
) -> list[TaskPackage]:
    domain_filter = set(domains or config.experiment["domains"])
    structure_filter = set(structures or config.experiment["task_structures"])
    tasks = [
        task
        for task in load_config_tasks(config)
        if task.domain in domain_filter and task.task_structure in structure_filter
    ]
    groups: dict[tuple[str, str], list[TaskPackage]] = defaultdict(list)
    for task in tasks:
        groups[(task.domain, task.task_structure)].append(task)
    selected: list[TaskPackage] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda task: task.task_id)
        selected.extend(ordered[:limit] if limit is not None else ordered)
    return selected


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    tasks = _select_tasks(
        config,
        domains=args.domains,
        structures=args.structures,
        limit=args.limit,
    )
    organizations = args.organizations or list(config.experiment["organizations"])
    seeds = args.seeds or list(config.experiment["seeds"])
    runner = ExperimentRunner(config)
    try:
        return await runner.run_plan(tasks, organizations, seeds)
    finally:
        close = getattr(runner.backend, "close", None)
        if close is not None:
            await close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenDGO Q1 six-domain benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="build public tasks and private gold")
    prepare_parser.add_argument("--config", type=Path, default=Path("configs/q1.yaml"))
    prepare_parser.add_argument("--domain", required=True, choices=sorted(DOMAIN_SPECS))
    prepare_parser.add_argument("--input", type=Path)

    validate_parser = subparsers.add_parser("validate", help="validate frozen Q1 inputs")
    validate_parser.add_argument("--config", type=Path, default=Path("configs/q1.yaml"))

    run_parser = subparsers.add_parser("run", help="execute Q1 runs with resume support")
    run_parser.add_argument("--config", type=Path, default=Path("configs/q1.yaml"))
    run_parser.add_argument("--domain", action="append", choices=sorted(DOMAIN_SPECS), dest="domains")
    run_parser.add_argument("--structure", action="append", choices=STRUCTURES, dest="structures")
    run_parser.add_argument(
        "--organization", action="append", choices=ALL_ORGANIZATIONS, dest="organizations"
    )
    run_parser.add_argument("--seed", action="append", type=int, dest="seeds")
    run_parser.add_argument(
        "--limit", type=int, help="take the first N frozen tasks in every selected domain-structure cell"
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="score runs with evaluator-only gold")
    evaluate_parser.add_argument("--config", type=Path, default=Path("configs/q1.yaml"))

    aggregate_parser = subparsers.add_parser("aggregate", help="build Table 1 data files")
    aggregate_parser.add_argument("--config", type=Path, default=Path("configs/q1.yaml"))

    status_parser = subparsers.add_parser("status", help="show plan and result progress")
    status_parser.add_argument("--config", type=Path, default=Path("configs/q1.yaml"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        config = load_config(args.config)
        input_path = args.input or (
            config.data_root / args.domain / "raw" / "normalized.jsonl"
        )
        result = prepare_domain(input_path.resolve(), config.data_root, args.domain)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "validate":
        result = validate_data(load_config(args.config))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ready"]:
            raise SystemExit(1)
        return
    if args.command == "run":
        if args.limit is not None and args.limit <= 0:
            raise SystemExit("--limit must be positive")
        result = asyncio.run(_run(args))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate":
        result = evaluate_tree(load_config(args.config))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ready"]:
            raise SystemExit(1)
        return
    if args.command == "status":
        result = experiment_status(load_config(args.config))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = aggregate_results(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


