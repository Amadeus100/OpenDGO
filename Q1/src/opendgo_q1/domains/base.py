from __future__ import annotations

from dataclasses import dataclass

from ..schema import N_AGENTS, STRUCTURES


STRUCTURE_INSTRUCTIONS = {
    "sequential_dependency": (
        "Produce the dependency-stage artifact your role is responsible for. "
        "Use upstream messages when present, preserve intermediate values, and do not skip prerequisites."
    ),
    "parallel_decomposition": (
        "Analyze your assigned view independently, then use neighbor messages to compare independent "
        "findings. Report disagreements instead of averaging them away."
    ),
    "evidence_integration": (
        "Track evidence IDs explicitly. Determine which claims are supporting, conflicting, or irrelevant, "
        "and state what complementary evidence is still missing from your local view."
    ),
}


@dataclass(frozen=True)
class DomainSpec:
    slug: str
    display_name: str
    dataset: str
    primary_metric: str
    roles: tuple[str, ...]
    responsibilities: tuple[str, ...]
    final_instruction: str
    answer_key: str
    requires_external_executor: bool = False

    def __post_init__(self) -> None:
        if len(self.roles) != N_AGENTS or len(self.responsibilities) != N_AGENTS:
            raise ValueError(f"{self.slug} must define exactly six roles and responsibilities")

    def role_brief(self, node: int, structure: str) -> str:
        if structure not in STRUCTURES:
            raise KeyError(structure)
        return f"{self.responsibilities[node]} {STRUCTURE_INSTRUCTIONS[structure]}"

