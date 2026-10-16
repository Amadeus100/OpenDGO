from .base import DomainSpec


SPEC = DomainSpec(
    slug="recommendation",
    display_name="Recommendation",
    dataset="Amazon Reviews 2023",
    primary_metric="hit_at_1",
    roles=(
        "ranking integrator",
        "user profile analyst",
        "review text analyst",
        "item metadata analyst",
        "candidate scoring specialist",
        "constraint reviewer",
    ),
    responsibilities=(
        "Integrate a complete ranking over the frozen candidate IDs.",
        "Infer stable preferences and recency signals from the supplied history.",
        "Extract item qualities and user sentiment from assigned review evidence.",
        "Compare candidate attributes, categories, price, and compatibility metadata.",
        "Score candidates using explicit signals and preserve candidate IDs exactly.",
        "Check hard constraints, duplication, unsupported assumptions, and ranking completeness.",
    ),
    final_instruction=(
        "Return JSON only with keys ranked_item_ids and rationale. ranked_item_ids must contain each "
        "frozen candidate ID exactly once, best item first."
    ),
    answer_key="ranked_item_ids",
)

