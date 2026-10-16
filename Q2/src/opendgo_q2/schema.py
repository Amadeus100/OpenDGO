from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .util import sha256_json


N_AGENTS = 6
STRUCTURES = (
    "sequential_dependency",
    "parallel_decomposition",
    "evidence_integration",
)
ALLOCATIONS = ("centralized", "distributed", "conflicting")


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    text: str
    source: str = ""
    kind: str = "text"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceUnit":
        unit = cls(
            evidence_id=str(value["evidence_id"]).strip(),
            text=str(value["text"]),
            source=str(value.get("source", "")),
            kind=str(value.get("kind", "text")),
        )
        if not unit.evidence_id or not unit.text.strip():
            raise ValueError("evidence_id and evidence text must not be empty")
        return unit

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Q2Task:
    task_id: str
    source_id: str
    domain: str
    task_structure: str
    question: str
    roles: tuple[str, ...]
    responsibilities: tuple[str, ...]
    evidence_units: tuple[EvidenceUnit, ...]
    shared_context: Any = field(default_factory=dict)
    tools: tuple[str, ...] = ()
    answer_schema: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    q1_visible_input_sha256: str = ""
    visible_input_sha256: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Q2Task":
        task = cls(
            task_id=str(value["task_id"]).strip(),
            source_id=str(value.get("source_id", "")).strip(),
            domain=str(value["domain"]).strip(),
            task_structure=str(value["task_structure"]).strip(),
            question=str(value["question"]),
            roles=tuple(str(item) for item in value.get("roles", [])),
            responsibilities=tuple(str(item) for item in value.get("responsibilities", [])),
            evidence_units=tuple(
                EvidenceUnit.from_dict(item) for item in value.get("evidence_units", [])
            ),
            shared_context=value.get("shared_context", {}),
            tools=tuple(str(item) for item in value.get("tools", [])),
            answer_schema=dict(value.get("answer_schema", {})),
            provenance=dict(value.get("provenance", {})),
            q1_visible_input_sha256=str(value.get("q1_visible_input_sha256", "")),
            visible_input_sha256=str(value.get("visible_input_sha256", "")),
        )
        task.validate()
        if task.visible_input_sha256 and task.visible_input_sha256 != task.compute_hash():
            raise ValueError(f"{task.task_id}: frozen Q2 visible-input hash mismatch")
        return task

    def validate(self) -> None:
        if not self.task_id or not self.source_id or not self.question.strip():
            raise ValueError("task_id, source_id, and question must not be empty")
        if self.task_structure not in STRUCTURES:
            raise ValueError(f"unknown task structure {self.task_structure!r}")
        if len(self.roles) != N_AGENTS or len(self.responsibilities) != N_AGENTS:
            raise ValueError(f"{self.task_id}: Q2 requires six roles and responsibilities")
        ids = [unit.evidence_id for unit in self.evidence_units]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{self.task_id}: evidence IDs must be non-empty and unique")

    def public_dict(self, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "domain": self.domain,
            "task_structure": self.task_structure,
            "question": self.question,
            "roles": list(self.roles),
            "responsibilities": list(self.responsibilities),
            "evidence_units": [unit.to_dict() for unit in self.evidence_units],
            "shared_context": self.shared_context,
            "tools": list(self.tools),
            "answer_schema": self.answer_schema,
            "provenance": self.provenance,
            "q1_visible_input_sha256": self.q1_visible_input_sha256,
        }
        if include_hash:
            value["visible_input_sha256"] = self.visible_input_sha256 or self.compute_hash()
        return value

    def compute_hash(self) -> str:
        return sha256_json(self.public_dict(include_hash=False))


@dataclass(frozen=True)
class InterventionAnnotation:
    task_id: str
    expert_role_index: int
    decisive_evidence_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    conflicting_evidence_ids: tuple[str, ...]
    annotation_source: str
    reviewed_by: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterventionAnnotation":
        item = cls(
            task_id=str(value["task_id"]),
            expert_role_index=int(value["expert_role_index"]),
            decisive_evidence_ids=tuple(str(x) for x in value.get("decisive_evidence_ids", [])),
            supporting_evidence_ids=tuple(str(x) for x in value.get("supporting_evidence_ids", [])),
            conflicting_evidence_ids=tuple(str(x) for x in value.get("conflicting_evidence_ids", [])),
            annotation_source=str(value.get("annotation_source", "")),
            reviewed_by=str(value.get("reviewed_by", "")),
            metadata=dict(value.get("metadata", {})),
        )
        if item.expert_role_index not in range(N_AGENTS):
            raise ValueError(f"{item.task_id}: expert_role_index must be in 0..5")
        if not item.decisive_evidence_ids:
            raise ValueError(f"{item.task_id}: decisive_evidence_ids must not be empty")
        if not item.annotation_source.strip() or not item.reviewed_by.strip():
            raise ValueError(f"{item.task_id}: annotation provenance is required")
        if not set(item.decisive_evidence_ids) <= set(item.supporting_evidence_ids):
            raise ValueError(f"{item.task_id}: decisive evidence must be supporting evidence")
        if set(item.supporting_evidence_ids) & set(item.conflicting_evidence_ids):
            raise ValueError(f"{item.task_id}: supporting and conflicting evidence overlap")
        if not item.conflicting_evidence_ids:
            raise ValueError(f"{item.task_id}: conflicting evidence must be source-grounded and non-empty")
        return item

    def validate_for(self, task: Q2Task) -> None:
        if self.task_id != task.task_id:
            raise ValueError("annotation/task ID mismatch")
        available = {unit.evidence_id for unit in task.evidence_units}
        labeled = (
            set(self.supporting_evidence_ids)
            | set(self.decisive_evidence_ids)
            | set(self.conflicting_evidence_ids)
        )
        missing = labeled - available
        if missing:
            raise ValueError(f"{task.task_id}: annotation references missing evidence {sorted(missing)}")


@dataclass(frozen=True)
class GoldRecord:
    task_id: str
    source_id: str
    domain: str
    answer: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoldRecord":
        return cls(
            task_id=str(value["task_id"]),
            source_id=str(value.get("source_id", "")),
            domain=str(value["domain"]),
            answer=value.get("answer"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.api_calls += other.api_calls

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

