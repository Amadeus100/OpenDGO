from __future__ import annotations

import json
from pathlib import Path

import pytest

from opendgo_q1.prepare import prepare_domain
from opendgo_q1.schema import TaskPackage
from opendgo_q1.util import read_jsonl


def test_prepare_separates_public_task_from_gold_and_evidence_labels(tmp_path: Path):
    source = {
        "source_id": "finqa-1",
        "question": "What is 40 plus 2?",
        "answer": "42",
        "dataset": "FinQA",
        "dataset_version": "test-revision",
        "split": "test",
        "evidence_units": [
            {
                "evidence_id": f"e{index}",
                "text": f"evidence {index}",
                "required": index in {1, 2},
                "truth_status": "supporting" if index in {1, 2} else "irrelevant",
                "role_hint": index % 6,
            }
            for index in range(6)
        ],
        "structure_views": {
            structure: [[f"e{node}"] for node in range(6)]
            for structure in (
                "sequential_dependency",
                "parallel_decomposition",
                "evidence_integration",
            )
        },
        "gold_metadata": {"program": "add(40,2)"},
    }
    raw = tmp_path / "normalized.jsonl"
    raw.write_text(json.dumps(source) + "\n", encoding="utf-8")
    report = prepare_domain(raw, tmp_path / "data", "finance")
    assert report["tasks"] == 3

    public_rows = read_jsonl(tmp_path / "data" / "finance" / "processed" / "tasks.jsonl")
    serialized = json.dumps(public_rows)
    assert '"answer"' not in public_rows[0]
    for row in public_rows:
        for view in row["evidence_views"]:
            for unit in view:
                assert "required" not in unit
                assert "truth_status" not in unit
                assert "role_hint" not in unit
    assert "add(40,2)" not in serialized
    for row in public_rows:
        TaskPackage.from_dict(row)

    gold_rows = read_jsonl(tmp_path / "data" / "finance" / "private" / "gold.jsonl")
    assert {row["answer"] for row in gold_rows} == {"42"}


def test_frozen_visible_hash_detects_tampering(tmp_path: Path):
    from conftest import make_finance_task

    row = make_finance_task("parallel_decomposition").public_dict()
    row["question"] = "tampered"
    with pytest.raises(ValueError, match="visible_input_sha256 mismatch"):
        TaskPackage.from_dict(row)

