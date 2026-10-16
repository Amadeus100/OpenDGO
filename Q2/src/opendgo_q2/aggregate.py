from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from .conditions import POSITIONS
from .config import Q2Config
from .util import sha256_text, write_json


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap(values: list[float], resamples: int, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    draws = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(resamples)
    )
    return draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]


def _cell_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, int]:
    return (
        str(row["domain"]),
        str(row["task_structure"]),
        str(row["topology"]),
        str(row["position_slug"]),
        str(row["information_allocation"]),
        int(row["seed"]),
    )


def aggregate_results(config: Q2Config) -> dict[str, Any]:
    evaluations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((config.results_root / "runs").rglob("evaluation.json"))
    ]
    grouped: dict[tuple[str, str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluations:
        grouped[_cell_key(row)].append(row)

    expected_n = int(config.experiment["tasks_per_cell"])
    seeds = list(config.experiment["seeds"])
    seed_rows: list[dict[str, Any]] = []
    seed_means: dict[tuple[str, str, str, str, str, int], float] = {}
    missing: list[str] = []
    for domain in config.experiment["domains"]:
        for structure in config.experiment["task_structures"]:
            for topology in config.experiment["topologies"]:
                for condition in POSITIONS[topology]:
                    for allocation in config.experiment["information_allocations"]:
                        for seed in seeds:
                            key = (domain, structure, topology, condition.slug, allocation, seed)
                            rows = grouped.get(key, [])
                            scores = [
                                float(row["primary_score"])
                                for row in rows
                                if row.get("primary_score") is not None
                            ]
                            ready = len(rows) == expected_n and len(scores) == expected_n
                            if not ready:
                                missing.append(
                                    "/".join(map(str, key))
                                    + f": observed={len(rows)}, scored={len(scores)}, expected={expected_n}"
                                )
                            value = mean(scores) if scores else None
                            if value is not None:
                                seed_means[key] = value
                            seed_rows.append(
                                {
                                    "domain": domain,
                                    "task_structure": structure,
                                    "topology": topology,
                                    "position": condition.label,
                                    "position_slug": condition.slug,
                                    "information_allocation": allocation,
                                    "seed": seed,
                                    "n_expected": expected_n,
                                    "n_observed": len(rows),
                                    "n_scored": len(scores),
                                    "seed_mean": value,
                                    "completion_rate": (
                                        sum(row.get("run_status") == "completed" for row in rows) / expected_n
                                        if expected_n
                                        else None
                                    ),
                                    "ready": ready,
                                }
                            )

    condition_rows: list[dict[str, Any]] = []
    for domain in config.experiment["domains"]:
        for structure in config.experiment["task_structures"]:
            for topology in config.experiment["topologies"]:
                for condition in POSITIONS[topology]:
                    for allocation in config.experiment["information_allocations"]:
                        values = [
                            seed_means[(domain, structure, topology, condition.slug, allocation, seed)]
                            for seed in seeds
                            if (domain, structure, topology, condition.slug, allocation, seed) in seed_means
                        ]
                        condition_rows.append(
                            {
                                "domain": domain,
                                "task_structure": structure,
                                "topology": topology,
                                "position": condition.label,
                                "position_slug": condition.slug,
                                "information_allocation": allocation,
                                "seed_count": len(values),
                                "mean": mean(values) if values else None,
                                "sample_sd": stdev(values) if len(values) >= 2 else (0.0 if values else None),
                                "mean_percent": 100 * mean(values) if values else None,
                                "sample_sd_percent": (
                                    100 * stdev(values)
                                    if len(values) >= 2
                                    else (0.0 if values else None)
                                ),
                            }
                        )

    resamples = int(config.experiment["bootstrap_resamples"])
    base_seed = int(config.experiment.get("bootstrap_seed", 2026))
    curve_rows: list[dict[str, Any]] = []
    for structure in config.experiment["task_structures"]:
        for topology in config.experiment["topologies"]:
            for condition in POSITIONS[topology]:
                for allocation in config.experiment["information_allocations"]:
                    values = [
                        value
                        for (domain, s, t, p, a, seed), value in seed_means.items()
                        if s == structure and t == topology and p == condition.slug and a == allocation
                    ]
                    deterministic = base_seed + int(
                        sha256_text(f"{structure}|{topology}|{condition.slug}|{allocation}")[:8], 16
                    )
                    low, high = _bootstrap(values, resamples, deterministic)
                    curve_rows.append(
                        {
                            "task_structure": structure,
                            "topology": topology,
                            "position": condition.label,
                            "position_slug": condition.slug,
                            "information_allocation": allocation,
                            "domain_seed_cells": len(values),
                            "macro_mean": mean(values) if values else None,
                            "bootstrap_95_low": low,
                            "bootstrap_95_high": high,
                            "macro_mean_percent": 100 * mean(values) if values else None,
                            "bootstrap_95_low_percent": 100 * low if low is not None else None,
                            "bootstrap_95_high_percent": 100 * high if high is not None else None,
                        }
                    )

    best_rows: list[dict[str, Any]] = []
    if "tree" in config.experiment["topologies"]:
        for domain in config.experiment["domains"]:
            for structure in config.experiment["task_structures"]:
                candidates = [
                    row
                    for row in condition_rows
                    if row["domain"] == domain
                    and row["task_structure"] == structure
                    and row["topology"] == "tree"
                    and row["mean"] is not None
                ]
                ranked = sorted(candidates, key=lambda row: (-float(row["mean"]), row["position_slug"], row["information_allocation"]))
                if ranked:
                    best_rows.append(
                        {
                            "domain": domain,
                            "task_structure": structure,
                            "position": ranked[0]["position"],
                            "information_allocation": ranked[0]["information_allocation"],
                            "winning_mean": ranked[0]["mean"],
                            "margin_to_second": (
                                float(ranked[0]["mean"]) - float(ranked[1]["mean"])
                                if len(ranked) > 1
                                else None
                            ),
                            "winning_quality_percent": 100 * float(ranked[0]["mean"]),
                            "margin_points": (
                                100 * (float(ranked[0]["mean"]) - float(ranked[1]["mean"]))
                                if len(ranked) > 1
                                else None
                            ),
                        }
                    )

    interaction_rows: list[dict[str, Any]] = []
    if "tree" in config.experiment["topologies"]:
        interaction_by_domain: dict[tuple[str, str], list[float]] = defaultdict(list)
        for domain in config.experiment["domains"]:
            for structure in config.experiment["task_structures"]:
                for seed in seeds:
                    keys = {
                        "cc": (domain, structure, "tree", "center", "centralized", seed),
                        "ec": (domain, structure, "tree", "edge", "centralized", seed),
                        "cd": (domain, structure, "tree", "center", "distributed", seed),
                        "ed": (domain, structure, "tree", "edge", "distributed", seed),
                    }
                    if all(key in seed_means for key in keys.values()):
                        gamma = (
                            seed_means[keys["cc"]]
                            - seed_means[keys["ec"]]
                            - seed_means[keys["cd"]]
                            + seed_means[keys["ed"]]
                        )
                        interaction_by_domain[(domain, structure)].append(gamma)
                values = interaction_by_domain[(domain, structure)]
                interaction_rows.append(
                    {
                        "domain": domain,
                        "task_structure": structure,
                        "contrast": "(Center-Centralized - Edge-Centralized) - (Center-Distributed - Edge-Distributed)",
                        "seed_count": len(values),
                        "mean_interaction": mean(values) if values else None,
                        "sample_sd": stdev(values) if len(values) >= 2 else (0.0 if values else None),
                        "mean_interaction_points": 100 * mean(values) if values else None,
                        "sample_sd_points": (
                            100 * stdev(values)
                            if len(values) >= 2
                            else (0.0 if values else None)
                        ),
                    }
                )

    root = config.results_root / "aggregate"
    seed_fields = ["domain", "task_structure", "topology", "position", "position_slug", "information_allocation", "seed", "n_expected", "n_observed", "n_scored", "seed_mean", "completion_rate", "ready"]
    condition_fields = ["domain", "task_structure", "topology", "position", "position_slug", "information_allocation", "seed_count", "mean", "sample_sd", "mean_percent", "sample_sd_percent"]
    curve_fields = ["task_structure", "topology", "position", "position_slug", "information_allocation", "domain_seed_cells", "macro_mean", "bootstrap_95_low", "bootstrap_95_high", "macro_mean_percent", "bootstrap_95_low_percent", "bootstrap_95_high_percent"]
    _write_csv(root / "seed_summary.csv", seed_rows, seed_fields)
    _write_csv(root / "q2_condition_long.csv", condition_rows, condition_fields)
    _write_csv(root / "q2_all_topology_curves.csv", curve_rows, curve_fields)
    _write_csv(root / "q2_tree_main.csv", [row for row in curve_rows if row["topology"] == "tree"], curve_fields)
    _write_csv(root / "q2_best_pairing_by_domain.csv", best_rows, ["domain", "task_structure", "position", "information_allocation", "winning_mean", "margin_to_second", "winning_quality_percent", "margin_points"])
    _write_csv(root / "q2_interaction_contrasts.csv", interaction_rows, ["domain", "task_structure", "contrast", "seed_count", "mean_interaction", "sample_sd", "mean_interaction_points", "sample_sd_points"])
    report = {
        "ready": bool(evaluations) and not missing,
        "evaluation_files": len(evaluations),
        "missing_or_invalid_groups": missing,
        "outputs": {
            "seed_summary": str(root / "seed_summary.csv"),
            "condition_long": str(root / "q2_condition_long.csv"),
            "tree_main": str(root / "q2_tree_main.csv"),
            "best_pairing": str(root / "q2_best_pairing_by_domain.csv"),
            "interaction_contrasts": str(root / "q2_interaction_contrasts.csv"),
            "all_topology_curves": str(root / "q2_all_topology_curves.csv"),
        },
    }
    write_json(root / "aggregation_report.json", report)
    return report

