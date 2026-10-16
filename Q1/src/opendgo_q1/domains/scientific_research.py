from .base import DomainSpec


SPEC = DomainSpec(
    slug="scientific_research",
    display_name="Scientific Research",
    dataset="SciFact",
    primary_metric="label_accuracy",
    roles=(
        "claim decision integrator",
        "claim analyst",
        "abstract evidence analyst",
        "support evidence analyst",
        "refutation evidence analyst",
        "rationale verifier",
    ),
    responsibilities=(
        "Integrate one claim label and an evidence-grounded rationale.",
        "Decompose the claim into testable propositions, scope, and qualifiers.",
        "Review assigned abstracts and identify relevant sentence IDs.",
        "Build the strongest evidence-grounded support case.",
        "Build the strongest evidence-grounded refutation or insufficiency case.",
        "Check whether the selected sentences actually entail the proposed label.",
    ),
    final_instruction=(
        "Return JSON only with keys label, evidence_sentence_ids, and rationale. "
        "Use the frozen dataset label vocabulary."
    ),
    answer_key="label",
)

