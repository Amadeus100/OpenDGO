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


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    text: str
    source: str = ""
    kind: str = "text"
    required: bool = False
    stage: int | None = None
    role_hint: int | None = None
    truth_status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceUnit":
        unit = cls(
            evidence_id=str(value["evidence_id"]).strip(),
            text=str(value["text"]),
            source=str(value.get("source", "")),
            kind=str(value.get("kind", "text")),
            required=bool(value.get("required", False)),
            stage=int(value["stage"]) if value.get("stage") is not None else None,
            role_hint=int(value["role_hint"]) if value.get("role_hint") is not None else None,
            truth_status=str(value.get("truth_status", "unknown")),
            metadata=dict(value.get("metadata", {})),
        )
        if not unit.evidence_id:
            raise ValueError("evidence_id must not be empty")
        if not unit.text.strip():
            raise ValueError(f"evidence {unit.evidence_id!r} has empty text")
        if unit.role_hint is not None and unit.role_hint not in range(N_AGENTS):
            raise ValueError(f"evidence {unit.evidence_id!r} has invalid role_hint")
        return unit

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_dict(self) -> dict[str, Any]:
        """Return only fields that may be shown to the model.

        Construction labels such as ``required`` and ``truth_status`` are kept
        while preparing a task but are never serialized into the public package.
        """
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "source": self.source,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class TaskPackage:
    task_id: str
    source_id: str
    domain: str
    task_structure: str
    question: str
    roles: tuple[str, ...]
    evidence_views: tuple[tuple[EvidenceUnit, ...], ...]
    shared_context: Any = field(default_factory=dict)
    tools: tuple[str, ...] = ()
    answer_schema: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    visible_input_sha256: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskPackage":
        raw_views = value.get("evidence_views")
        if not isinstance(raw_views, list):
            raise ValueError("evidence_views must be a six-element list")
        task = cls(
            task_id=str(value["task_id"]).strip(),
            source_id=str(value["source_id"]).strip(),
            domain=str(value["domain"]).strip(),
            task_structure=str(value["task_structure"]).strip(),
            question=str(value["question"]),
            roles=tuple(str(role) for role in value.get("roles", [])),
            evidence_views=tuple(
                tuple(EvidenceUnit.from_dict(item) for item in view) for view in raw_views
            ),
            shared_context=value.get("shared_context", {}),
            tools=tuple(str(tool) for tool in value.get("tools", [])),
            answer_schema=dict(value.get("answer_schema", {})),
            provenance=dict(value.get("provenance", {})),
            visible_input_sha256=str(value.get("visible_input_sha256", "")),
        )
        task.validate()
        expected_hash = task.compute_visible_hash()
        if task.visible_input_sha256 and task.visible_input_sha256 != expected_hash:
            raise ValueError(
                f"{task.task_id}: visible_input_sha256 mismatch; task package changed after freezing"
            )
        return task

    def validate(self) -> None:
        if not self.task_id or not self.source_id:
            raise ValueError("task_id and source_id must not be empty")
        if self.task_structure not in STRUCTURES:
            raise ValueError(f"unknown task structure: {self.task_structure}")
        if not self.question.strip():
            raise ValueError(f"{self.task_id}: question must not be empty")
        if len(self.roles) != N_AGENTS:
            raise ValueError(f"{self.task_id}: roles must contain exactly six entries")
        if len(self.evidence_views) != N_AGENTS:
            raise ValueError(f"{self.task_id}: evidence_views must contain exactly six entries")
        seen: dict[str, EvidenceUnit] = {}
        for view in self.evidence_views:
            for unit in view:
                earlier = seen.get(unit.evidence_id)
                if earlier is not None and earlier != unit:
                    raise ValueError(
                        f"{self.task_id}: evidence ID {unit.evidence_id!r} has inconsistent copies"
                    )
                seen[unit.evidence_id] = unit
        if not seen:
            raise ValueError(f"{self.task_id}: evidence union must not be empty")
        if self.task_structure == "evidence_integration":
            holders = {
                node for node, view in enumerate(self.evidence_views) if view
            }
            if len(holders) < 2:
                raise ValueError(
                    f"{self.task_id}: evidence integration needs evidence on >=2 roles"
                )

    def evidence_union(self) -> tuple[EvidenceUnit, ...]:
        by_id: dict[str, EvidenceUnit] = {}
        for view in self.evidence_views:
            for unit in view:
                by_id.setdefault(unit.evidence_id, unit)
        return tuple(by_id[key] for key in sorted(by_id))

    def public_dict(self, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "domain": self.domain,
            "task_structure": self.task_structure,
            "question": self.question,
            "roles": list(self.roles),
            "evidence_views": [
                [unit.model_dict() for unit in view] for view in self.evidence_views
            ],
            "shared_context": self.shared_context,
            "tools": list(self.tools),
            "answer_schema": self.answer_schema,
            "provenance": self.provenance,
        }
        if include_hash:
            value["visible_input_sha256"] = self.visible_input_sha256 or self.compute_visible_hash()
        return value

    def compute_visible_hash(self) -> str:
        return sha256_json(self.public_dict(include_hash=False))


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

