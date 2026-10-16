from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import Q1Config
from .util import read_jsonl


def experiment_status(config: Q1Config) -> dict[str, Any]:
    planned: set[tuple[str, str, str, int]] = set()
    for plan_path in sorted((config.results_root / "plans").glob("plan_*.jsonl")):
        for row in read_jsonl(plan_path):
            planned.add(
                (
                    str(row["task_id"]),
                    str(row["task_structure"]),
                    str(row["organization"]),
                    int(row["seed"]),
                )
            )

    status_counts: Counter[str] = Counter()
    observed_keys: set[tuple[str, str, str, int]] = set()
    cells: dict[tuple[str, str, str, int], Counter[str]] = defaultdict(Counter)
    for result_path in sorted((config.results_root / "runs").rglob("result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            status_counts["unreadable_result"] += 1
            continue
        status = str(result.get("status", "unknown"))
        key = (
            str(result["task_id"]),
            str(result["task_structure"]),
            str(result["organization"]),
            int(result["seed"]),
        )
        observed_keys.add(key)
        status_counts[status] += 1
        cell = (
            str(result["domain"]),
            str(result["task_structure"]),
            str(result["organization"]),
            int(result["seed"]),
        )
        cells[cell][status] += 1

    cell_rows = [
        {
            "domain": cell[0],
            "task_structure": cell[1],
            "organization": cell[2],
            "seed": cell[3],
            **dict(counts),
            "total": sum(counts.values()),
        }
        for cell, counts in sorted(cells.items())
    ]
    expected = len(planned)
    observed = len(observed_keys)
    return {
        "results_root": str(config.results_root),
        "planned_runs": expected,
        "observed_runs": observed,
        "pending_runs": max(0, expected - len(planned & observed_keys)),
        "progress_percent": round(100 * len(planned & observed_keys) / expected, 4)
        if expected
        else None,
        "status_counts": dict(status_counts),
        "cells": cell_rows,
    }

