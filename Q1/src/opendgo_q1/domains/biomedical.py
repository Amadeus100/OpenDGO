from .base import DomainSpec


SPEC = DomainSpec(
    slug="biomedical",
    display_name="Biomedical",
    dataset="BioASQ Task B",
    primary_metric="exact_answer_accuracy",
    roles=(
        "answer integrator",
        "question type analyst",
        "literature evidence analyst",
        "biomedical evidence extractor",
        "biomedical reasoner",
        "fact checker",
    ),
    responsibilities=(
        "Integrate an exact answer supported by reachable biomedical evidence.",
        "Identify question type, entities, scope, and the required exact-answer form.",
        "Review assigned articles or snippets and report relevant evidence IDs.",
        "Extract relations, mechanisms, populations, interventions, and outcomes.",
        "Connect complementary evidence while separating association from causation.",
        "Check factual support, contradictions, overclaims, and answer completeness.",
    ),
    final_instruction=(
        "Return JSON only with keys exact_answer, ideal_answer, and evidence_ids. "
        "For list questions, exact_answer must be a JSON list."
    ),
    answer_key="exact_answer",
)

