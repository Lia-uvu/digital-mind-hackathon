"""Frozen persona-template identities for the 2 x 2 factorial design."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaSpec:
    """One prompt template and its two experimental factor levels."""

    prompt_key: str
    quadrant_id: str
    extraversion: str
    neuroticism: str
    template_id: str


PERSONA_SPECS = (
    PersonaSpec("persona.high_e_high_n", "high_e_high_n", "high", "high", "v1"),
    PersonaSpec("persona.high_e_high_n.v2", "high_e_high_n", "high", "high", "v2"),
    PersonaSpec("persona.high_e_high_n.v3", "high_e_high_n", "high", "high", "v3"),
    PersonaSpec("persona.high_e_low_n", "high_e_low_n", "high", "low", "v1"),
    PersonaSpec("persona.high_e_low_n.v2", "high_e_low_n", "high", "low", "v2"),
    PersonaSpec("persona.high_e_low_n.v3", "high_e_low_n", "high", "low", "v3"),
    PersonaSpec("persona.low_e_high_n", "low_e_high_n", "low", "high", "v1"),
    PersonaSpec("persona.low_e_high_n.v2", "low_e_high_n", "low", "high", "v2"),
    PersonaSpec("persona.low_e_high_n.v3", "low_e_high_n", "low", "high", "v3"),
    PersonaSpec("persona.low_e_low_n", "low_e_low_n", "low", "low", "v1"),
    PersonaSpec("persona.low_e_low_n.v2", "low_e_low_n", "low", "low", "v2"),
    PersonaSpec("persona.low_e_low_n.v3", "low_e_low_n", "low", "low", "v3"),
)

PERSONA_BY_ID = {spec.prompt_key: spec for spec in PERSONA_SPECS}
PERSONA_KEYS = tuple(PERSONA_BY_ID)
PERSONA_QUADRANTS = tuple(dict.fromkeys(spec.quadrant_id for spec in PERSONA_SPECS))


def get_persona_spec(persona_id: str) -> PersonaSpec:
    """Return frozen factor metadata for a runtime persona prompt key."""
    try:
        return PERSONA_BY_ID[persona_id]
    except KeyError as error:
        raise ValueError(f"unknown persona: {persona_id}") from error


def expected_template_ids(quadrant_id: str) -> tuple[str, ...]:
    """Return the template IDs required for one balanced quadrant block."""
    templates = tuple(
        spec.template_id for spec in PERSONA_SPECS if spec.quadrant_id == quadrant_id
    )
    if not templates:
        raise ValueError(f"unknown persona quadrant: {quadrant_id}")
    return templates
