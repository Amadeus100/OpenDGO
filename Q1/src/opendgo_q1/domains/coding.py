from .base import DomainSpec


SPEC = DomainSpec(
    slug="coding",
    display_name="Coding",
    dataset="SWE-bench Verified",
    primary_metric="official_resolve_rate",
    roles=(
        "patch integrator",
        "issue analyst",
        "repository navigator",
        "implementation specialist",
        "test analyst",
        "regression reviewer",
    ),
    responsibilities=(
        "Integrate a minimal patch after checking all received repository evidence.",
        "Clarify the issue contract, failure behavior, and acceptance criteria.",
        "Locate relevant files, symbols, call paths, and repository conventions.",
        "Design the smallest implementation change consistent with the repository.",
        "Identify or run relevant tests and interpret failures without inventing results.",
        "Check regressions, edge cases, compatibility, and patch completeness.",
    ),
    final_instruction=(
        "Return the repository patch in the external executor's required schema. Primary success is decided "
        "only by the SWE-bench official harness."
    ),
    answer_key="patch",
    requires_external_executor=True,
)

