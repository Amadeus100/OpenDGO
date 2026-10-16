from __future__ import annotations

import json
from pathlib import Path

from opendgo_q1.aggregate import aggregate_results
from opendgo_q1.evaluation import evaluate_result, finance_exact_match


def test_finance_normalization_handles_currency_units_percent_and_parentheses():
    assert finance_exact_match("$284 million", "284") == 1.0
    assert finance_exact_match("13%", "13") == 1.0
    assert finance_exact_match("(7)", "-7") == 1.0


def test_incomplete_run_is_retained_and_scored_zero():
    result = {
        "task_id": "t1",
        "source_id": "s1",
        "domain": "legal",
        "task_structure": "parallel_decomposition",
        "organization": "star",
        "organization_category": "fixed",
        "seed": 101,
        "status": "incomplete",
    }
    evaluation = evaluate_result(result, "yes", {})
    assert evaluation["primary_score"] == 0.0
    assert evaluation["evaluation_status"] == "incomplete_run_scored_zero"


def test_aggregate_computes_seed_level_table(make_config, tmp_path: Path):
    config = make_config(
        domains=["finance"],
        task_structures=["parallel_decomposition"],
        organizations=["star"],
        seeds=[101],
        tasks_per_cell=1,
    )
    run_dir = (
        config.results_root
        / "runs"
        / "finance"
        / "parallel_decomposition"
        / "star"
        / "101"
        / "task"
    )
    run_dir.mkdir(parents=True)
    evaluation = {
        "task_id": "t1",
        "source_id": "s1",
        "domain": "finance",
        "task_structure": "parallel_decomposition",
        "organization": "star",
        "organization_category": "fixed",
        "seed": 101,
        "run_status": "completed",
        "primary_metric": "exact_match",
        "primary_score": 1.0,
        "valid_for_primary_metric": True,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    report = aggregate_results(config)
    assert report["ready"] is True
    wide = Path(report["outputs"]["wide_table"]).read_text(encoding="utf-8-sig")
    assert "100.0±0.0" in wide

