from __future__ import annotations

import json

import pytest

from conftest import MockBackend, make_finance_task
from opendgo_q1.runner import ExperimentRunner


@pytest.mark.asyncio
async def test_single_agent_receives_union_once(make_config):
    backend = MockBackend()
    runner = ExperimentRunner(make_config(), backend)
    result = await runner.run_one(
        make_finance_task("evidence_integration"), "single_agent", 101
    )
    assert result["status"] == "completed"
    assert len(backend.calls) == 1
    prompt = backend.calls[0]["user"]
    for index in range(6):
        assert f'"evidence_id": "e{index}"' in prompt


@pytest.mark.asyncio
async def test_sequential_chain_calls_each_non_sink_once_then_integrates(make_config):
    backend = MockBackend()
    runner = ExperimentRunner(make_config(), backend)
    result = await runner.run_one(
        make_finance_task("sequential_dependency"), "chain", 101
    )
    assert result["status"] == "completed"
    assert len(backend.calls) == 6  # five dependency nodes + one final synthesis
    assert all("Topology:" not in call["user"] for call in backend.calls)
    assert all("chain" not in call["user"].lower() for call in backend.calls)


@pytest.mark.asyncio
async def test_parallel_uses_real_edge_messages_without_method_name(make_config):
    backend = MockBackend()
    runner = ExperimentRunner(make_config(communication_rounds=1), backend)
    task = make_finance_task("parallel_decomposition")
    result = await runner.run_one(task, "star", 101)
    assert result["status"] == "completed"
    assert len(backend.calls) == 13  # 6 independent + 6 communication + final
    assert all("star" not in call["user"].lower() for call in backend.calls)
    events_path = runner._run_dir(task, "star", 101) / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    messages = [event for event in events if event["event"] == "message"]
    assert any(event["sender"] == 0 and event["recipient"] == 1 for event in messages)
    assert not any(event["sender"] == 1 and event["recipient"] == 2 for event in messages)


@pytest.mark.asyncio
async def test_missing_coding_executor_fails_before_model_calls(make_config):
    from dataclasses import replace
    from opendgo_q1.domains import get_domain
    from opendgo_q1.schema import EvidenceUnit, TaskPackage

    backend = MockBackend()
    runner = ExperimentRunner(make_config(domains=["coding"]), backend)
    task = TaskPackage(
        task_id="coding__sample__parallel_decomposition",
        source_id="sample",
        domain="coding",
        task_structure="parallel_decomposition",
        question="Fix the issue",
        roles=get_domain("coding").roles,
        evidence_views=tuple((EvidenceUnit(f"e{i}", f"view {i}"),) for i in range(6)),
        answer_schema={"type": "object", "required": ["patch"]},
    )
    with pytest.raises(RuntimeError, match="executors.coding.command is empty"):
        runner.validate_prerequisites([task], ["star"], [101])
    assert backend.calls == []

