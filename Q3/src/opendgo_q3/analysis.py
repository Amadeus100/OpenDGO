from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import Q3Config
from .graph_library import load_graph_records
from .graph_properties import PROPERTY_NAMES
from .util import write_csv, write_json


PRE_SPECIFIED: dict[str, tuple[str, ...]] = {
    "sequential_dependency": ("diameter", "edge_connectivity", "density"),
    "parallel_decomposition": ("modularity", "clustering"),
    "evidence_integration": ("cycle_redundancy", "edge_connectivity"),
}

MEDIATORS = (
    "evidence_coverage",
    "propagation_latency",
    "duplicate_message_rate",
    "contradiction_resolution",
    "cross_group_flow",
    "synthesizer_load_tokens_estimated",
)


def _load_observations(config: Q3Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation_path in sorted((config.results_root / "runs").rglob("evaluation.json")):
        metrics_path = evaluation_path.parent / "trace_metrics.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if not evaluation.get("valid_for_primary_metric"):
            continue
        row = {
            "task_id": evaluation["task_id"],
            "domain": evaluation["domain"],
            "task_structure": evaluation["task_structure"],
            "graph_id": evaluation["graph_id"],
            "seed": int(evaluation["seed"]),
            "quality": float(evaluation["primary_score"]),
            **{name: float(evaluation["graph_properties"][name]) for name in PROPERTY_NAMES},
        }
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row.update({name: metrics.get(name) for name in MEDIATORS})
        rows.append(row)
    return rows


def _z(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    return (values - float(values.mean())) / scale if scale > 0 else np.zeros_like(values)


def _fit(
    rows: list[dict[str, Any]], outcome: str, target: str,
    controls: list[str], mediator: str | None = None,
) -> dict[str, Any] | None:
    required = [outcome, target, *controls, *([mediator] if mediator else [])]
    clean = [row for row in rows if all(row.get(key) is not None for key in required)]
    if len(clean) < 4:
        return None
    columns: list[tuple[str, np.ndarray]] = [("intercept", np.ones(len(clean)))]
    ordered = [target, *([mediator] if mediator else []), *controls]
    for name in ordered:
        columns.append((name, _z(np.asarray([float(row[name]) for row in clean]))))
    domains = sorted({str(row["domain"]) for row in clean})
    for domain in domains[1:]:
        columns.append((f"domain={domain}", np.asarray([float(row["domain"] == domain) for row in clean])))
    kept_names: list[str] = []
    kept_columns: list[np.ndarray] = []
    dropped: list[str] = []
    for name, column in columns:
        trial = np.column_stack([*kept_columns, column]) if kept_columns else column[:, None]
        if np.linalg.matrix_rank(trial) > len(kept_columns):
            kept_names.append(name)
            kept_columns.append(column)
        else:
            dropped.append(name)
    design = np.column_stack(kept_columns)
    y = _z(np.asarray([float(row[outcome]) for row in clean]))
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    coefficients = dict(zip(kept_names, (float(value) for value in beta)))
    return {
        "coefficients": coefficients,
        "n": len(clean),
        "rank": int(rank),
        "controls_used": [name for name in controls if name in kept_names],
        "dropped_collinear": dropped,
        "rows": clean,
    }


def _cluster_bootstrap(
    rows: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float | None],
    resamples: int, seed: int,
) -> tuple[float | None, float | None, int]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["task_id"]))].append(row)
    clusters = sorted(grouped)
    if not clusters or resamples <= 0:
        return None, None, 0
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        indices = rng.integers(0, len(clusters), len(clusters))
        sample = [row for index in indices for row in grouped[clusters[int(index)]]]
        value = statistic(sample)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return None, None, 0
    low, high = np.quantile(np.asarray(values), [0.025, 0.975])
    return float(low), float(high), len(values)


def analyze(config: Q3Config) -> dict[str, Any]:
    observations = _load_observations(config)
    graphs = load_graph_records(config.graph_library_root)
    pairs = json.loads(
        (config.graph_library_root / "matched_pairs.json").read_text(encoding="utf-8")
    )
    aggregate_root = config.results_root / "aggregate"
    profile_rows = [
        {
            "graph_id": graph_id,
            "graph_category": record["graph_category"],
            **record["properties"],
        }
        for graph_id, record in sorted(graphs.items())
    ]
    write_csv(aggregate_root / "graph_profiles.csv", profile_rows)
    resamples = int(config.experiment.get("bootstrap_resamples", 10000))
    bootstrap_seed = int(config.experiment.get("bootstrap_seed", 2026))
    effect_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    counter = 0
    for structure, targets in PRE_SPECIFIED.items():
        structure_rows = [row for row in observations if row["task_structure"] == structure]
        for target in targets:
            allowed = {
                str(item[key])
                for item in pairs.get(target, [])
                for key in ("high_graph_id", "low_graph_id")
            }
            rows = [row for row in structure_rows if row["graph_id"] in allowed]
            controls = [name for name in PROPERTY_NAMES if name != target]
            fit = _fit(rows, "quality", target, controls)
            if fit is None:
                warnings.append(f"{structure}/{target}: insufficient evaluated observations")
                continue
            point = fit["coefficients"].get(target)
            if point is None:
                warnings.append(f"{structure}/{target}: target is constant or collinear")
                continue

            def property_stat(sample: list[dict[str, Any]]) -> float | None:
                fitted = _fit(sample, "quality", target, controls)
                return None if fitted is None else fitted["coefficients"].get(target)

            low, high, successful = _cluster_bootstrap(
                fit["rows"], property_stat, resamples, bootstrap_seed + counter
            )
            counter += 1
            effect_rows.append(
                {
                    "task_structure": structure,
                    "target_property": target,
                    "standardized_partial_effect": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_observations": fit["n"],
                    "n_task_clusters": len({(row["domain"], row["task_id"]) for row in fit["rows"]}),
                    "bootstrap_successful": successful,
                    "controls_used": ";".join(fit["controls_used"]),
                    "dropped_collinear": ";".join(fit["dropped_collinear"]),
                }
            )
            for mediator in MEDIATORS:
                behavior_fit = _fit(rows, mediator, target, controls)
                quality_fit = _fit(rows, "quality", target, controls, mediator=mediator)
                if behavior_fit is None or quality_fit is None:
                    continue
                complete_rows = quality_fit["rows"]
                base_mechanism_fit = _fit(complete_rows, "quality", target, controls)
                if base_mechanism_fit is None:
                    continue
                a = behavior_fit["coefficients"].get(target)
                b = quality_fit["coefficients"].get(mediator)
                adjusted = quality_fit["coefficients"].get(target)
                base_mechanism = base_mechanism_fit["coefficients"].get(target)
                if a is None or b is None or adjusted is None or base_mechanism is None:
                    continue

                def mechanism_stat(
                    sample: list[dict[str, Any]], component: str
                ) -> float | None:
                    behavior = _fit(sample, mediator, target, controls)
                    quality = _fit(sample, "quality", target, controls, mediator=mediator)
                    base = _fit(sample, "quality", target, controls)
                    if behavior is None or quality is None or base is None:
                        return None
                    if component == "a":
                        return behavior["coefficients"].get(target)
                    if component == "b":
                        return quality["coefficients"].get(mediator)
                    adjusted_value = quality["coefficients"].get(target)
                    base_value = base["coefficients"].get(target)
                    if adjusted_value is None or base_value is None:
                        return None
                    return adjusted_value - base_value

                a_low, a_high, a_successful = _cluster_bootstrap(
                    complete_rows,
                    lambda sample: mechanism_stat(sample, "a"),
                    resamples,
                    bootstrap_seed + counter,
                )
                counter += 1
                b_low, b_high, b_successful = _cluster_bootstrap(
                    complete_rows,
                    lambda sample: mechanism_stat(sample, "b"),
                    resamples,
                    bootstrap_seed + counter,
                )
                counter += 1
                change_low, change_high, change_successful = _cluster_bootstrap(
                    complete_rows,
                    lambda sample: mechanism_stat(sample, "change"),
                    resamples,
                    bootstrap_seed + counter,
                )
                counter += 1
                mechanism_rows.append(
                    {
                        "task_structure": structure,
                        "target_property": target,
                        "behavior": mediator,
                        "property_to_behavior": a,
                        "property_to_behavior_ci95_low": a_low,
                        "property_to_behavior_ci95_high": a_high,
                        "behavior_to_quality_conditional": b,
                        "behavior_to_quality_ci95_low": b_low,
                        "behavior_to_quality_ci95_high": b_high,
                        "property_effect_without_behavior": base_mechanism,
                        "property_effect_with_behavior": adjusted,
                        "property_effect_change": adjusted - base_mechanism,
                        "property_effect_change_ci95_low": change_low,
                        "property_effect_change_ci95_high": change_high,
                        "n_observations": quality_fit["n"],
                        "bootstrap_successful_a": a_successful,
                        "bootstrap_successful_b": b_successful,
                        "bootstrap_successful_change": change_successful,
                        "interpretation": "diagnostic_not_causal_mediation",
                    }
                )
    write_csv(aggregate_root / "q3_property_effects.csv", effect_rows)
    write_csv(aggregate_root / "q3_mechanism_diagnostics.csv", mechanism_rows)
    report = {
        "protocol": "opendgo_q3_analysis_v1",
        "observation_count": len(observations),
        "effect_rows": len(effect_rows),
        "mechanism_rows": len(mechanism_rows),
        "bootstrap_unit": "domain + task_id; all organizations and seeds retained together",
        "bootstrap_resamples_requested": resamples,
        "interval": "asymmetric percentile 95%",
        "fixed_effects": "domain",
        "mechanism_interpretation": "diagnostic, not causal mediation",
        "warnings": warnings,
    }
    write_json(aggregate_root / "analysis_report.json", report)
    return report
