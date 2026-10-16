from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from opendgo_q1.domains import DOMAIN_SPECS
from opendgo_q1.schema import STRUCTURES

from .analysis import analyze
from .config import load_config
from .evaluation import evaluate_tree
from .graph_library import build_graph_library
from .runner import run_selected
from .status import experiment_status
from .trace_metrics import extract_tree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenDGO Q3 structural mechanism experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("graph-library", "evaluate", "metrics", "analyze", "status"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/q3.yaml"))
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/q3.yaml"))
    run.add_argument("--domain", action="append", choices=sorted(DOMAIN_SPECS), dest="domains")
    run.add_argument("--structure", action="append", choices=STRUCTURES, dest="structures")
    run.add_argument("--graph", action="append", dest="graphs")
    run.add_argument("--seed", action="append", type=int, dest="seeds")
    run.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "graph-library":
        result = build_graph_library(config)
    elif args.command == "run":
        if args.limit is not None and args.limit <= 0:
            raise SystemExit("--limit must be positive")
        result = asyncio.run(
            run_selected(
                config,
                domains=args.domains,
                structures=args.structures,
                graph_ids=args.graphs,
                seeds=args.seeds,
                limit=args.limit,
            )
        )
    elif args.command == "evaluate":
        result = evaluate_tree(config)
    elif args.command == "metrics":
        result = extract_tree(config)
    elif args.command == "analyze":
        result = analyze(config)
    else:
        result = experiment_status(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
