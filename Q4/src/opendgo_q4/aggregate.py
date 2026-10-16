from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from .config import Q4Config
from .util import write_csv, write_json


def _load_rows(config: Q4Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((config.results_root / "runs").rglob("evaluation.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not value.get("valid_for_primary_metric") or value.get("primary_score") is None:
            continue
        rows.append(
            {
                "task_id": str(value["task_id"]),
                "domain": str(value["domain"]),
                "task_structure": str(value["task_structure"]),
                "organization": str(value["organization"]),
                "seed": int(value["seed"]),
                "condition_type": str(value["condition_type"]),
                "condition_value": int(value["condition_value"]),
                "quality": float(value["primary_score"]),
                "tokens": float(value["accounted_total_tokens"]),
            }
        )
    return rows


def _macro(rows: list[dict[str, Any]], key: str) -> float | None:
    domain_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        domain_values[str(row["domain"])].append(float(row[key]))
    if not domain_values:
        return None
    return float(np.mean([np.mean(values) for values in domain_values.values()]))


def _bootstrap(
    rows: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float | None],
    resamples: int, seed: int,
) -> tuple[float | None, float | None, int]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["task_id"]))].append(row)
    clusters = sorted(grouped)
    if not clusters:
        return None, None, 0
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample = [
            row
            for index in rng.integers(0, len(clusters), len(clusters))
            for row in grouped[clusters[int(index)]]
        ]
        value = statistic(sample)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return None, None, 0
    low, high = np.quantile(np.asarray(values), [0.025, 0.975])
    return float(low), float(high), len(values)


def _budget_curve(
    rows: list[dict[str, Any]], resamples: int, seed: int,
) -> list[dict[str, Any]]:
    budget = [row for row in rows if row["condition_type"] == "budget_cap"]
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in budget:
        groups[(row["task_structure"], row["organization"], row["condition_value"])].append(row)
    output: list[dict[str, Any]] = []
    for index, (key, cell) in enumerate(sorted(groups.items())):
        low, high, successful = _bootstrap(
            cell, lambda sample: _macro(sample, "quality"), resamples, seed + index
        )
        output.append(
            {
                "task_structure": key[0],
                "organization": key[1],
                "token_cap": key[2],
                "quality_macro": _macro(cell, "quality"),
                "quality_ci95_low": low,
                "quality_ci95_high": high,
                "realized_tokens_macro": _macro(cell, "tokens"),
                "n_runs": len(cell),
                "n_task_clusters": len({(row["domain"], row["task_id"]) for row in cell}),
                "bootstrap_successful": successful,
            }
        )
    return output


def _pareto(
    rows: list[dict[str, Any]], natural_cap: int,
) -> list[dict[str, Any]]:
    natural = [
        row for row in rows
        if row["condition_type"] == "budget_cap" and row["condition_value"] == natural_cap
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in natural:
        groups[(row["task_structure"], row["organization"])].append(row)
    output = [
        {
            "task_structure": structure,
            "organization": organization,
            "quality_macro": _macro(cell, "quality"),
            "tokens_macro": _macro(cell, "tokens"),
            "n_runs": len(cell),
        }
        for (structure, organization), cell in sorted(groups.items())
    ]
    for row in output:
        alternatives = [other for other in output if other["task_structure"] == row["task_structure"]]
        dominated_by = [
            str(other["organization"])
            for other in alternatives
            if float(other["tokens_macro"]) < float(row["tokens_macro"])
            and float(other["quality_macro"]) > float(row["quality_macro"])
        ]
        row["pareto_efficient"] = not dominated_by
        row["dominated_by"] = ";".join(dominated_by)
    return output


def _fault_retained(
    rows: list[dict[str, Any]], resamples: int, seed: int,
) -> list[dict[str, Any]]:
    fault = [row for row in rows if row["condition_type"] == "fault_pressure"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fault:
        groups[(row["task_structure"], row["organization"])].append(row)
    output: list[dict[str, Any]] = []
    counter = 0
    for (structure, organization), group in sorted(groups.items()):
        pressures = sorted({int(row["condition_value"]) for row in group})
        for pressure in pressures:
            paired = [row for row in group if row["condition_value"] in {0, pressure}]

            def retained(sample: list[dict[str, Any]]) -> float | None:
                base = _macro([row for row in sample if row["condition_value"] == 0], "quality")
                stressed = _macro(
                    [row for row in sample if row["condition_value"] == pressure], "quality"
                )
                return None if base is None or stressed is None or base == 0 else stressed / base

            point = retained(paired)
            low, high, successful = _bootstrap(
                paired, retained, resamples, seed + counter
            )
            counter += 1
            stressed_rows = [row for row in group if row["condition_value"] == pressure]
            output.append(
                {
                    "task_structure": structure,
                    "organization": organization,
                    "fault_pressure_percent": pressure,
                    "quality_macro": _macro(stressed_rows, "quality"),
                    "retained_quality": point,
                    "retained_quality_ci95_low": low,
                    "retained_quality_ci95_high": high,
                    "n_runs": len(stressed_rows),
                    "bootstrap_successful": successful,
                    "normalization": "topology-and-structure-specific zero-failure macro score",
                }
            )
    return output


def aggregate(config: Q4Config) -> dict[str, Any]:
    rows = _load_rows(config)
    root = config.results_root / "aggregate"
    resamples = int(config.experiment.get("bootstrap_resamples", 10000))
    seed = int(config.experiment.get("bootstrap_seed", 2026))
    budget = _budget_curve(rows, resamples, seed)
    pareto = _pareto(rows, int(config.experiment["natural_max_total_tokens"]))
    fault = _fault_retained(rows, resamples, seed + 100000)
    write_csv(root / "budget_curve.csv", budget)
    write_csv(root / "pareto_frontier.csv", pareto)
    write_csv(root / "fault_retained_quality.csv", fault)
    observed_domains = sorted({str(row["domain"]) for row in rows})
    expected_domains = sorted(str(value) for value in config.experiment["domains"])
    observed_caps = sorted(
        {int(row["condition_value"]) for row in rows if row["condition_type"] == "budget_cap"}
    )
    observed_pressures = sorted(
        {int(row["condition_value"]) for row in rows if row["condition_type"] == "fault_pressure"}
    )
    complete_design = (
        observed_domains == expected_domains
        and observed_caps == sorted(int(value) for value in config.experiment["token_caps"])
        and observed_pressures
        == sorted(int(value) for value in config.experiment["fault_pressures_percent"])
    )
    report = {
        "protocol": "opendgo_q4_aggregate_v1",
        "evaluated_rows": len(rows),
        "budget_rows": len(budget),
        "pareto_rows": len(pareto),
        "fault_rows": len(fault),
        "domain_aggregation": "equal-weight macro average across available domains",
        "bootstrap_unit": "domain + task_id; all seeds and paired conditions retained together",
        "bootstrap_resamples_requested": resamples,
        "pareto_rule": "not strictly lower tokens and strictly higher quality dominated",
        "expected_domains": expected_domains,
        "observed_domains": observed_domains,
        "observed_token_caps": observed_caps,
        "observed_fault_pressures_percent": observed_pressures,
        "complete_design": complete_design,
        "ready": bool(rows and budget and pareto and fault and complete_design),
    }
    write_json(root / "aggregation_report.json", report)
    return report
