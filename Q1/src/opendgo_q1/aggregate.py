from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .config import Q1Config
from .domains import get_domain
from .organizations import FIXED_ORGANIZATIONS, LEARNED_ORGANIZATIONS


DISPLAY_NAMES = {
    "single_agent": "Single Agent",
    "star": "Star",
    "chain": "Chain",
    "tree": "Tree",
    "ring": "Ring",
    "mesh": "Mesh",
    "community": "Community",
    "g_designer": "G-Designer",
    "arg_designer": "ARG-Designer",
    "goagent": "GoAgent",
}


def _category(method: str) -> str:
    if method == "single_agent":
        return "Baseline"
    if method in FIXED_ORGANIZATIONS:
        return "Fixed"
    if method in LEARNED_ORGANIZATIONS:
        return "Learned"
    return "Unknown"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(config: Q1Config) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    for path in sorted((config.results_root / "runs").rglob("evaluation.json")):
        evaluations.append(json.loads(path.read_text(encoding="utf-8")))

    expected_n = int(config.experiment["tasks_per_cell"])
    expected_seeds = list(config.experiment["seeds"])
    seed_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluations:
        seed_groups[
            (
                str(row["domain"]),
                str(row["task_structure"]),
                str(row["organization"]),
                int(row["seed"]),
            )
        ].append(row)

    seed_rows: list[dict[str, Any]] = []
    seed_scores: dict[tuple[str, str, str, int], float] = {}
    ready = True
    missing_groups: list[str] = []
    for domain in config.experiment["domains"]:
        for structure in config.experiment["task_structures"]:
            for organization in config.experiment["organizations"]:
                for seed in expected_seeds:
                    key = (domain, structure, organization, seed)
                    rows = seed_groups.get(key, [])
                    scores = [row.get("primary_score") for row in rows]
                    valid_scores = [float(score) for score in scores if score is not None]
                    cell_ready = len(rows) == expected_n and len(valid_scores) == expected_n
                    if not cell_ready:
                        ready = False
                        missing_groups.append(
                            f"{domain}/{structure}/{organization}/{seed}: "
                            f"observed={len(rows)}, scored={len(valid_scores)}, expected={expected_n}"
                        )
                    seed_mean = mean(valid_scores) if valid_scores else None
                    if seed_mean is not None:
                        seed_scores[key] = seed_mean
                    seed_rows.append(
                        {
                            "domain": domain,
                            "task_structure": structure,
                            "category": _category(organization),
                            "organization": organization,
                            "method": DISPLAY_NAMES.get(organization, organization),
                            "seed": seed,
                            "n_expected": expected_n,
                            "n_observed": len(rows),
                            "n_scored": len(valid_scores),
                            "seed_mean": seed_mean,
                            "completion_rate": (
                                sum(row.get("run_status") == "completed" for row in rows) / expected_n
                                if expected_n
                                else None
                            ),
                            "ready": cell_ready,
                        }
                    )

    long_rows: list[dict[str, Any]] = []
    cell_values: dict[tuple[str, str, str], tuple[float | None, float | None]] = {}
    for domain in config.experiment["domains"]:
        for structure in config.experiment["task_structures"]:
            for organization in config.experiment["organizations"]:
                values = [
                    seed_scores[(domain, structure, organization, seed)]
                    for seed in expected_seeds
                    if (domain, structure, organization, seed) in seed_scores
                ]
                cell_mean = mean(values) if values else None
                cell_sd = stdev(values) if len(values) >= 2 else (0.0 if len(values) == 1 else None)
                cell_values[(domain, structure, organization)] = (cell_mean, cell_sd)
                long_rows.append(
                    {
                        "domain": domain,
                        "dataset": get_domain(domain).dataset,
                        "task_structure": structure,
                        "category": _category(organization),
                        "organization": organization,
                        "method": DISPLAY_NAMES.get(organization, organization),
                        "primary_metric": get_domain(domain).primary_metric,
                        "seed_count": len(values),
                        "mean": cell_mean,
                        "sample_sd": cell_sd,
                        "mean_percent": cell_mean * 100 if cell_mean is not None else None,
                        "sample_sd_percent": cell_sd * 100 if cell_sd is not None else None,
                    }
                )

    domain_columns = list(config.experiment["domains"])
    wide_rows: list[dict[str, Any]] = []
    for structure in config.experiment["task_structures"]:
        for organization in config.experiment["organizations"]:
            row: dict[str, Any] = {
                "task_structure": structure,
                "category": _category(organization),
                "organization": organization,
                "method": DISPLAY_NAMES.get(organization, organization),
            }
            for domain in domain_columns:
                cell_mean, cell_sd = cell_values[(domain, structure, organization)]
                row[domain] = (
                    f"{cell_mean * 100:.1f}±{cell_sd * 100:.1f}"
                    if cell_mean is not None and cell_sd is not None
                    else ""
                )
            macro_by_seed: list[float] = []
            for seed in expected_seeds:
                values = [
                    seed_scores[(domain, structure, organization, seed)]
                    for domain in domain_columns
                    if (domain, structure, organization, seed) in seed_scores
                ]
                if len(values) == len(domain_columns):
                    macro_by_seed.append(mean(values))
            if macro_by_seed:
                macro_mean = mean(macro_by_seed)
                macro_sd = stdev(macro_by_seed) if len(macro_by_seed) >= 2 else 0.0
                row["average"] = f"{macro_mean * 100:.1f}±{macro_sd * 100:.1f}"
            else:
                row["average"] = ""
            wide_rows.append(row)

    aggregate_root = config.results_root / "aggregate"
    _write_csv(
        aggregate_root / "seed_summary.csv",
        seed_rows,
        [
            "domain",
            "task_structure",
            "category",
            "organization",
            "method",
            "seed",
            "n_expected",
            "n_observed",
            "n_scored",
            "seed_mean",
            "completion_rate",
            "ready",
        ],
    )
    _write_csv(
        aggregate_root / "q1_table_long.csv",
        long_rows,
        [
            "domain",
            "dataset",
            "task_structure",
            "category",
            "organization",
            "method",
            "primary_metric",
            "seed_count",
            "mean",
            "sample_sd",
            "mean_percent",
            "sample_sd_percent",
        ],
    )
    _write_csv(
        aggregate_root / "q1_table_wide.csv",
        wide_rows,
        ["task_structure", "category", "organization", "method", *domain_columns, "average"],
    )
    report = {
        "ready": ready,
        "evaluation_files": len(evaluations),
        "expected_runs": (
            len(config.experiment["domains"])
            * len(config.experiment["task_structures"])
            * len(config.experiment["organizations"])
            * len(expected_seeds)
            * expected_n
        ),
        "missing_or_invalid_groups": missing_groups,
        "outputs": {
            "seed_summary": str(aggregate_root / "seed_summary.csv"),
            "long_table": str(aggregate_root / "q1_table_long.csv"),
            "wide_table": str(aggregate_root / "q1_table_wide.csv"),
        },
    }
    (aggregate_root / "aggregation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

