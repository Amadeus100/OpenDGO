from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from .config import Q2Config
from .util import read_jsonl


def experiment_status(config: Q2Config) -> dict[str, Any]:
    planned: set[tuple[str, str, str, str, str, int]] = set()
    for path in sorted((config.results_root / "plans").glob("plan_*.jsonl")):
        for row in read_jsonl(path):
            planned.add(
                (
                    str(row["task_id"]),
                    str(row["task_structure"]),
                    str(row["topology"]),
                    str(row["position_slug"]),
                    str(row["information_allocation"]),
                    int(row["seed"]),
                )
            )
    observed: set[tuple[str, str, str, str, str, int]] = set()
    statuses: Counter[str] = Counter()
    cells: dict[tuple[str, str, str, str, str, int], Counter[str]] = defaultdict(Counter)
    for path in sorted((config.results_root / "runs").rglob("result.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            key = (
                str(row["task_id"]),
                str(row["task_structure"]),
                str(row["topology"]),
                str(row["position_slug"]),
                str(row["information_allocation"]),
                int(row["seed"]),
            )
            observed.add(key)
            status = str(row.get("status", "unknown"))
            statuses[status] += 1
            cell = (
                str(row["domain"]),
                str(row["task_structure"]),
                str(row["topology"]),
                str(row["position_slug"]),
                str(row["information_allocation"]),
                int(row["seed"]),
            )
            cells[cell][status] += 1
        except Exception:
            statuses["unreadable_result"] += 1
    complete = len(planned & observed)
    return {
        "results_root": str(config.results_root),
        "planned_runs": len(planned),
        "observed_runs": len(observed),
        "pending_runs": max(0, len(planned) - complete),
        "progress_percent": round(100 * complete / len(planned), 4) if planned else None,
        "status_counts": dict(statuses),
        "cells": [
            {
                "domain": key[0],
                "task_structure": key[1],
                "topology": key[2],
                "position": key[3],
                "information_allocation": key[4],
                "seed": key[5],
                **dict(value),
                "total": sum(value.values()),
            }
            for key, value in sorted(cells.items())
        ],
    }

