from .base import DomainSpec


SPEC = DomainSpec(
    slug="legal",
    display_name="Legal",
    dataset="LegalBench",
    primary_metric="label_accuracy",
    roles=(
        "decision integrator",
        "issue spotter",
        "fact analyst",
        "rule analyst",
        "exception analyst",
        "counterargument reviewer",
    ),
    responsibilities=(
        "Integrate the permitted label from reachable facts, rules, and exceptions.",
        "Identify the legal issue and the precise decision the label represents.",
        "Separate legally relevant facts from background and unsupported assumptions.",
        "State the applicable rule using only the supplied legal evidence.",
        "Check exceptions, thresholds, defenses, and limiting conditions.",
        "Construct and test the strongest counterargument before final judgment.",
    ),
    final_instruction=(
        "Return JSON only with keys label, applied_rule, and rationale. "
        "The label must use the dataset's frozen label vocabulary."
    ),
    answer_key="label",
)

