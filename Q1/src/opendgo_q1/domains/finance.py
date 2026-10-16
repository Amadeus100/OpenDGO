from .base import DomainSpec


SPEC = DomainSpec(
    slug="finance",
    display_name="Finance",
    dataset="FinQA",
    primary_metric="exact_match",
    roles=(
        "answer integrator",
        "question analyst",
        "text evidence analyst",
        "table numeric analyst",
        "calculation specialist",
        "reasoning verifier",
    ),
    responsibilities=(
        "Synthesize one precise answer with correct units from reachable evidence.",
        "Determine the requested quantity, answer type, dates, units, and required operations.",
        "Extract relevant narrative facts and their source evidence IDs.",
        "Extract table cells, row and column labels, dates, signs, and scales.",
        "Carry out explicit calculations and preserve intermediate values.",
        "Challenge evidence selection, arithmetic, unit conversion, and assumptions.",
    ),
    final_instruction=(
        "Return JSON only with keys answer, supporting_evidence_ids, and reasoning_steps. "
        "The answer must preserve the requested unit."
    ),
    answer_key="answer",
)

