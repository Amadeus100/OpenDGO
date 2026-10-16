from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .config import Q4Config


def experiment_status(config: Q4Config) -> dict[str, Any]:
    planned: set[tuple[str, str, str, int, str, int]] = set()
    for path in sorted((config.results_root / "plans").glob("plan_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                planned.add(
                    (
                        str(row["task_id"]),
                        str(row["task_structure"]),
                        str(row["organization"]),
                        int(row["seed"]),
                        str(row["condition_type"]),
                        int(row["condition_value"]),
                    )
                )
    observed: set[tuple[str, str, str, int, str, int]] = set()
    statuses: Counter[str] = Counter()
    for path in sorted((config.results_root / "runs").rglob("result.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            observed.add(
                (
                    str(row["task_id"]),
                    str(row["task_structure"]),
                    str(row["organization"]),
                    int(row["seed"]),
                    str(row["condition_type"]),
                    int(row["condition_value"]),
                )
            )
            statuses[str(row.get("status", "unknown"))] += 1
        except Exception:
            statuses["unreadable_result"] += 1
    return {
        "results_root": str(config.results_root),
        "planned_runs": len(planned),
        "observed_runs": len(observed),
        "pending_runs": len(planned - observed),
        "progress_percent": 100 * len(planned & observed) / len(planned) if planned else None,
        "status_counts": dict(statuses),
    }
