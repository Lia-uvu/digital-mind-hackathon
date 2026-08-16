from __future__ import annotations

import pytest

from recompute_post_guess_emotions import (
    balanced_summary,
    extract_post_guess_messages,
    main,
    recompute_branch_record,
)
from encouragement_lab.personas import PERSONA_SPECS


PROMPTS = {
    "condition.encouragement": "supportive intervention",
    "condition.neutral": "neutral intervention",
    "willingness": "rate willingness",
}


def _projection(value: float) -> dict:
    return {
        axis: {"layers": {"19": value}, "median": value}
        for axis in ("positive", "negative", "frustration")
    }


def _record() -> dict:
    return {
        "run_id": "persona.high_e_high_n.seed-1001",
        "seed": 1001,
        "checkpoint_id": "persona.high_e_high_n.seed-1001.round-5",
        "persona_id": "persona.high_e_high_n",
        "persona_quadrant": "high_e_high_n",
        "persona_template_id": "v1",
        "condition": "encouragement",
        "transcript": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "last feedback"},
            {"role": "user", "content": "supportive intervention"},
            {"role": "assistant", "content": '{"guess":"0123"}'},
            {"role": "user", "content": "rate willingness"},
            {"role": "assistant", "content": '{"willingness":8}'},
        ],
        "emotion_projections": {
            "pre_intervention": _projection(0.1),
            "post_guess": _projection(0.2),
        },
    }


class _Backend:
    def __init__(self) -> None:
        self.calls = []

    def render_messages(self, messages, *, add_generation_prompt=True):
        self.calls.append((messages, add_generation_prompt))
        return "rendered"


class _Probe:
    def score_text(self, rendered_text):
        assert rendered_text == "rendered"
        return _projection(0.4)


def test_recompute_uses_exact_assistant_ending_without_generation_prompt() -> None:
    backend = _Backend()

    result = recompute_branch_record(_record(), PROMPTS, backend, _Probe())

    messages, add_generation_prompt = backend.calls[0]
    assert add_generation_prompt is False
    assert messages[-1] == {"role": "assistant", "content": '{"guess":"0123"}'}
    assert result["old_post_guess_delta"] == pytest.approx(
        {axis: 0.1 for axis in ("positive", "negative", "frustration")}
    )
    assert result["corrected_post_guess_delta"] == pytest.approx(
        {axis: 0.3 for axis in ("positive", "negative", "frustration")}
    )


def test_extraction_rejects_non_frozen_branch_suffix() -> None:
    record = _record()
    record["transcript"][-2]["content"] = "different willingness prompt"

    with pytest.raises(ValueError, match="frozen prompt"):
        extract_post_guess_messages(record, PROMPTS)


def test_balanced_summary_averages_templates_within_seed() -> None:
    corrections = []
    for spec in PERSONA_SPECS:
        for condition, corrected_value in (("encouragement", 1.0), ("neutral", 0.0)):
            corrections.append(
                {
                    "run_id": f"{spec.prompt_key}.seed-1001",
                    "seed": 1001,
                    "checkpoint_id": f"{spec.prompt_key}.seed-1001.round-5",
                    "persona_id": spec.prompt_key,
                    "persona_quadrant": spec.quadrant_id,
                    "persona_template_id": spec.template_id,
                    "condition": condition,
                    "old_post_guess_delta": {
                        axis: corrected_value / 2
                        for axis in ("positive", "negative", "frustration")
                    },
                    "corrected_post_guess_delta": {
                        axis: corrected_value
                        for axis in ("positive", "negative", "frustration")
                    },
                }
            )

    summary = balanced_summary(corrections)

    assert summary["branch_count"] == 24
    assert summary["pair_count"] == 12
    assert summary["seeds"] == [1001]
    corrected = summary["axes"]["frustration"]["corrected"]
    assert corrected["by_seed"] == (
        {
            "seed": 1001,
            "average_treatment": 1.0,
            "extraversion": 0.0,
            "neuroticism": 0.0,
            "interaction": 0.0,
        },
    )
    assert corrected["average_treatment"]["mean"] == pytest.approx(1.0)


def test_cli_refuses_to_overwrite_before_loading_model(tmp_path) -> None:
    output = tmp_path / "existing.jsonl"
    output.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(
            [
                "--output",
                str(output),
                "--summary-output",
                str(tmp_path / "summary.json"),
            ]
        )

    assert output.read_text(encoding="utf-8") == "keep me\n"
