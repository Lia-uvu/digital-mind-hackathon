from collections import Counter
from pathlib import Path

import pytest

from encouragement_lab.personas import (
    PERSONA_BY_ID,
    PERSONA_KEYS,
    PERSONA_QUADRANTS,
    PERSONA_SPECS,
    expected_template_ids,
    get_persona_spec,
)
from encouragement_lab.prompt_loader import load_prompts


ROOT = Path(__file__).resolve().parents[1]


def test_persona_registry_is_balanced_three_templates_per_quadrant() -> None:
    assert len(PERSONA_KEYS) == len(set(PERSONA_KEYS)) == 12
    assert len(PERSONA_BY_ID) == 12
    assert len(PERSONA_QUADRANTS) == 4
    assert Counter(spec.quadrant_id for spec in PERSONA_SPECS) == {
        quadrant: 3 for quadrant in PERSONA_QUADRANTS
    }
    for quadrant in PERSONA_QUADRANTS:
        assert expected_template_ids(quadrant) == ("v1", "v2", "v3")


def test_persona_registry_exposes_factors_and_rejects_unknown_ids() -> None:
    spec = get_persona_spec("persona.low_e_high_n.v3")
    assert (spec.extraversion, spec.neuroticism, spec.template_id) == (
        "low",
        "high",
        "v3",
    )
    with pytest.raises(ValueError, match="unknown persona"):
        get_persona_spec("persona.unknown")


def test_parallel_persona_wordings_are_approximately_length_matched() -> None:
    prompts = load_prompts(ROOT / "prompts.md")
    for template_id in ("v1", "v2", "v3"):
        word_counts = [
            len(prompts[spec.prompt_key].split())
            for spec in PERSONA_SPECS
            if spec.template_id == template_id
        ]
        assert max(word_counts) - min(word_counts) <= 3
