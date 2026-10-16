from __future__ import annotations

from .base import DomainSpec
from .biomedical import SPEC as BIOMEDICAL
from .coding import SPEC as CODING
from .finance import SPEC as FINANCE
from .legal import SPEC as LEGAL
from .recommendation import SPEC as RECOMMENDATION
from .scientific_research import SPEC as SCIENTIFIC_RESEARCH


DOMAIN_SPECS: dict[str, DomainSpec] = {
    spec.slug: spec
    for spec in (CODING, FINANCE, BIOMEDICAL, LEGAL, SCIENTIFIC_RESEARCH, RECOMMENDATION)
}


def get_domain(slug: str) -> DomainSpec:
    try:
        return DOMAIN_SPECS[slug]
    except KeyError as exc:
        raise KeyError(f"unknown domain {slug!r}; choose from {sorted(DOMAIN_SPECS)}") from exc

