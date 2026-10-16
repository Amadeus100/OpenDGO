from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_finance_task
from opendgo_q1.organizations import LearnedGraphStore


def test_learned_graph_is_keyed_by_task_structure_and_seed(tmp_path: Path):
    task = make_finance_task("parallel_decomposition")
    root = tmp_path / "graphs"
    root.mkdir()
    row = {
        "task_id": task.task_id,
        "task_structure": task.task_structure,
        "seed": 101,
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
        "input_sha256": task.compute_visible_hash(),
        "designer_version": "official-test",
    }
    (root / "g_designer.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    organization = LearnedGraphStore(root).resolve("g_designer", task, 101)
    assert organization.category == "learned"
    assert organization.edges == ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))


def test_learned_graph_rejects_disconnected_or_wrong_hash(tmp_path: Path):
    task = make_finance_task("parallel_decomposition")
    root = tmp_path / "graphs"
    root.mkdir()
    bad = {
        "task_id": task.task_id,
        "task_structure": task.task_structure,
        "seed": 101,
        "edges": [[0, 1], [1, 2]],
        "input_sha256": "wrong",
    }
    (root / "goagent.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="connect all six nodes"):
        LearnedGraphStore(root).resolve("goagent", task, 101)

