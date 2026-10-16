from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from opendgo_q1.backend import Completion
from opendgo_q1.config import Q1Config
from opendgo_q1.schema import EvidenceUnit, TaskPackage, Usage


class MockBackend:
    model = "mock-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def health(self) -> dict[str, Any]:
        return {"models": [self.model]}

    async def complete(
        self, system: str, user: str, *, max_tokens: int, seed: int
    ) -> Completion:
        self.calls.append(
            {"system": system, "user": user, "max_tokens": max_tokens, "seed": seed}
        )
        if "Required JSON schema" in user or "Complete evidence union" in user:
            content = '{"answer": "42", "supporting_evidence_ids": ["e1", "e2"], "reasoning_steps": []}'
        elif "Convert the following already-produced answer" in user:
            content = '{"answer": "42"}'
        else:
            content = f"finding-{len(self.calls)}"
        return Completion(
            content=content,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, api_calls=1),
            finish_reason="stop",
            request_attempts=1,
        )


@pytest.fixture
def make_config(tmp_path: Path):
    def factory(**experiment_overrides: Any) -> Q1Config:
        experiment = {
            "domains": ["finance"],
            "task_structures": ["sequential_dependency", "parallel_decomposition", "evidence_integration"],
            "organizations": ["single_agent", "star", "chain", "tree", "ring", "mesh", "community"],
            "seeds": [101],
            "tasks_per_cell": 1,
            "synthesizer_node": 0,
            "communication_rounds": 1,
            "communication_budget_chars": 3000,
            "max_total_tokens": 12000,
            "final_token_reserve": 500,
            "max_messages": 100,
            "parse_repair_attempts": 0,
            "repeat_limit": 3,
            "unchanged_round_limit": 2,
            "record_message_content": True,
        }
        experiment.update(experiment_overrides)
        return Q1Config(
            source_path=tmp_path / "configs" / "test.yaml",
            project_root=tmp_path,
            data_root=tmp_path / "data",
            results_root=tmp_path / "results",
            learned_graph_root=tmp_path / "learned_graphs",
            llm={
                "base_url": "http://unused/v1",
                "api_key": "EMPTY",
                "model": "mock-model",
                "temperature": 0,
                "max_tokens": 100,
                "final_max_tokens": 200,
                "timeout_seconds": 5,
                "request_retries": 0,
                "max_concurrent_requests": 6,
            },
            experiment=experiment,
            executors={"coding": {"command": []}},
        )

    return factory


def make_finance_task(structure: str, task_id: str | None = None) -> TaskPackage:
    from opendgo_q1.domains import get_domain

    units = tuple(
        (EvidenceUnit(f"e{node}", f"evidence for node {node}", source=f"source-{node}"),)
        for node in range(6)
    )
    task = TaskPackage(
        task_id=task_id or f"finance__sample__{structure}",
        source_id="sample",
        domain="finance",
        task_structure=structure,
        question="What is the value?",
        roles=get_domain("finance").roles,
        evidence_views=units,
        shared_context={"currency": "USD"},
        answer_schema={"type": "object", "required": ["answer"]},
        provenance={"dataset": "FinQA", "split": "test"},
    )
    task.validate()
    return task

