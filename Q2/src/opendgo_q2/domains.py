from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainSpec:
    dataset: str
    primary_metric: str
    final_instruction: str
    answer_key: str
    requires_external_executor: bool = False


DOMAIN_SPECS = {
    "coding": DomainSpec(
        "SWE-bench Verified",
        "official_resolve_rate",
        "Return the repository patch in the external executor schema.",
        "patch",
        True,
    ),
    "finance": DomainSpec(
        "FinQA",
        "exact_match",
        "Return JSON only with answer, supporting_evidence_ids, and reasoning_steps.",
        "answer",
    ),
    "biomedical": DomainSpec(
        "BioASQ Task B",
        "exact_answer_accuracy",
        "Return JSON only with exact_answer, ideal_answer, and evidence_ids.",
        "exact_answer",
    ),
    "legal": DomainSpec(
        "LegalBench",
        "label_accuracy",
        "Return JSON only with label, applied_rule, and rationale.",
        "label",
    ),
    "scientific_research": DomainSpec(
        "SciFact",
        "label_accuracy",
        "Return JSON only with label, evidence_sentence_ids, and rationale.",
        "label",
    ),
    "recommendation": DomainSpec(
        "Amazon Reviews 2023",
        "hit_at_1",
        "Return JSON only with ranked_item_ids and rationale.",
        "ranked_item_ids",
    ),
}


RESPONSIBILITIES: dict[str, tuple[str, ...]] = {
    "coding": (
        "Integrate a minimal patch after checking all received repository evidence.",
        "Clarify the issue contract, failure behavior, and acceptance criteria.",
        "Locate relevant files, symbols, call paths, and repository conventions.",
        "Design the smallest implementation change consistent with the repository.",
        "Identify or run relevant tests and interpret failures without inventing results.",
        "Check regressions, edge cases, compatibility, and patch completeness.",
    ),
    "finance": (
        "Synthesize one precise answer with correct units from reachable evidence.",
        "Determine the requested quantity, answer type, dates, units, and operations.",
        "Extract relevant narrative facts and their evidence IDs.",
        "Extract table cells, labels, dates, signs, and scales.",
        "Carry out explicit calculations and preserve intermediate values.",
        "Challenge evidence selection, arithmetic, units, and assumptions.",
    ),
    "biomedical": (
        "Integrate an exact answer supported by reachable biomedical evidence.",
        "Identify question type, entities, scope, and exact-answer form.",
        "Review assigned articles or snippets and report relevant evidence IDs.",
        "Extract relations, mechanisms, populations, interventions, and outcomes.",
        "Connect complementary evidence while separating association from causation.",
        "Check factual support, contradictions, overclaims, and completeness.",
    ),
    "legal": (
        "Integrate the permitted label from reachable facts, rules, and exceptions.",
        "Identify the legal issue and the decision represented by the label.",
        "Separate legally relevant facts from background and assumptions.",
        "State the applicable rule using only supplied legal evidence.",
        "Check exceptions, thresholds, defenses, and limiting conditions.",
        "Construct and test the strongest counterargument before judgment.",
    ),
    "scientific_research": (
        "Integrate one claim label and an evidence-grounded rationale.",
        "Decompose the claim into propositions, scope, and qualifiers.",
        "Review assigned abstracts and identify relevant sentence IDs.",
        "Build the strongest evidence-grounded support case.",
        "Build the strongest evidence-grounded refutation or insufficiency case.",
        "Check whether selected sentences entail the proposed label.",
    ),
    "recommendation": (
        "Integrate a complete ranking over the frozen candidate IDs.",
        "Infer stable preferences and recency signals from supplied history.",
        "Extract item qualities and user sentiment from review evidence.",
        "Compare candidate attributes, category, price, and metadata.",
        "Score candidates using explicit signals and preserve IDs exactly.",
        "Check constraints, duplication, assumptions, and ranking completeness.",
    ),
}


def get_domain(slug: str) -> DomainSpec:
    try:
        return DOMAIN_SPECS[slug]
    except KeyError as exc:
        raise KeyError(f"unknown domain {slug!r}") from exc


STRUCTURE_INSTRUCTIONS = {
    "sequential_dependency": (
        "Preserve prerequisite order and explicit intermediate values. Use only upstream findings "
        "that were actually delivered."
    ),
    "parallel_decomposition": (
        "Develop your assigned analysis independently, then compare delivered findings and expose "
        "disagreements instead of averaging them away."
    ),
    "evidence_integration": (
        "Track evidence IDs and distinguish support, conflict, and missing information without "
        "assuming access to an undelivered source."
    ),
}

