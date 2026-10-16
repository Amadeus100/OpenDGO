from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from opendgo_q1.domains import DOMAIN_SPECS
from opendgo_q1.schema import STRUCTURES
from opendgo_q1.topologies import FIXED_EDGES

from .aggregate import aggregate
from .config import load_config
from .evaluation import evaluate_tree
from .runner import run_selected
from .status import experiment_status


def _run_parser(sub: argparse._SubParsersAction, name: str, value_name: str) -> None:
    parser = sub.add_parser(name)
    parser.add_argument("--config", type=Path, default=Path("configs/q4.yaml"))
    parser.add_argument("--domain", action="append", choices=sorted(DOMAIN_SPECS), dest="domains")
    parser.add_argument("--structure", action="append", choices=STRUCTURES, dest="structures")
    parser.add_argument("--organization", action="append", choices=sorted(FIXED_EDGES), dest="organizations")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument(f"--{value_name}", action="append", type=int, dest="values")
    parser.add_argument("--limit", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenDGO Q4 quality-cost-robustness")
    sub = parser.add_subparsers(dest="command", required=True)
    _run_parser(sub, "run-budget", "cap")
    _run_parser(sub, "run-fault", "pressure")
    for name in ("evaluate", "aggregate", "status"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/q4.yaml"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command in {"run-budget", "run-fault"}:
        if args.limit is not None and args.limit <= 0:
            raise SystemExit("--limit must be positive")
        condition_type = "budget_cap" if args.command == "run-budget" else "fault_pressure"
        result = asyncio.run(
            run_selected(
                config,
                condition_type=condition_type,
                domains=args.domains,
                structures=args.structures,
                organizations=args.organizations,
                seeds=args.seeds,
                values=args.values,
                limit=args.limit,
            )
        )
    elif args.command == "evaluate":
        result = evaluate_tree(config)
    elif args.command == "aggregate":
        result = aggregate(config)
    else:
        result = experiment_status(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
