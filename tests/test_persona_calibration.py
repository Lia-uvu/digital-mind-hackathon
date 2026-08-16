from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from encouragement_lab.persona_calibration import (
    CALIBRATION_LAYERS,
    CALIBRATION_SUFFIX_KEYS,
    audit_persona_tokens,
    calibration_messages,
    calibration_personas,
    calibration_suffixes,
    contamination_audit,
    evaluate_leave_one_template_out,
    render_calibration_inputs,
)
from encouragement_lab.persona_baseline import baseline_rows, summarize_baseline, write_snapshot
from encouragement_lab.personas import PERSONA_SPECS
from encouragement_lab.prompt_loader import load_prompts


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        tokens = text.replace(".", " .").replace(",", " ,").split()
        return {
            "input_ids": [
                self.vocabulary.setdefault(token, len(self.vocabulary) + 1)
                for token in tokens
            ]
        }


def _render(messages, *, add_generation_prompt=True):
    assert add_generation_prompt is True
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _drafts():
    prompts = load_prompts(ROOT / "formal_v2_personas.md")
    return calibration_personas(prompts), calibration_suffixes(prompts)


def _v1_drafts():
    prompts = load_prompts(ROOT / "formal_v2_personas_v1.md")
    return calibration_personas(prompts), calibration_suffixes(prompts)


def _v2_drafts():
    prompts = load_prompts(ROOT / "formal_v2_personas_v2.md")
    return calibration_personas(prompts), calibration_suffixes(prompts)


def _v3_drafts():
    prompts = load_prompts(ROOT / "formal_v2_personas_v3.md")
    return calibration_personas(prompts), calibration_suffixes(prompts)


def _hidden():
    rows = {}
    for suffix_index, suffix in enumerate(CALIBRATION_SUFFIX_KEYS):
        rows[suffix] = {}
        for spec in PERSONA_SPECS:
            e = 1.0 if spec.extraversion == "high" else -1.0
            n = 1.0 if spec.neuroticism == "high" else -1.0
            rows[suffix][spec.prompt_key] = {
                layer: np.asarray([e, n, float(suffix_index)], dtype=np.float64)
                for layer in CALIBRATION_LAYERS
            }
    return rows


def test_draft_token_audit_checks_sentences_personas_renders_and_fixed_spans():
    personas, suffixes = _drafts()

    audits = audit_persona_tokens(_Tokenizer(), personas, suffixes, _render)

    assert [audit.template_id for audit in audits] == ["v1", "v2", "v3"]
    for audit in audits:
        assert len(set(audit.sentence_token_counts.values())) == 1
        assert len(set(audit.persona_token_counts.values())) == 1
        assert len(set(audit.rendered_token_counts.values())) == 1
        assert len(audit.axis_difference_spans["extraversion"]) == 2
        assert len(audit.axis_difference_spans["neuroticism"]) == 2


def test_token_audit_rejects_a_replacement_that_moves_token_indices():
    personas, suffixes = _drafts()
    personas = dict(personas)
    personas["persona.low_e_high_n"] = personas["persona.low_e_high_n"].replace(
        "mildly drawn", "drawn mildly"
    )

    with pytest.raises(ValueError, match="fixed token indices"):
        audit_persona_tokens(_Tokenizer(), personas, suffixes, _render)


def test_v1_uses_only_more_less_scalar_axis_differences_and_no_banned_n_language():
    personas, suffixes = _v1_drafts()
    banned = (
        "shift", "change", "vary", "over time", "dwell", "linger", "recover",
        "uncertainty", "problem", "calm", "steady", "even", "failure", "support",
    )
    assert not any(term in " ".join(personas.values()).lower() for term in banned)
    audits = audit_persona_tokens(_Tokenizer(), personas, suffixes, _render)
    assert all(
        len(audit.axis_difference_spans[axis]) == 2
        for audit in audits
        for axis in ("extraversion", "neuroticism")
    )


def test_v2_n_carriers_are_paraphrases_after_scalar_slots_are_normalized():
    personas, suffixes = _v2_drafts()
    by_template = {}
    for template, key in (("v1", "persona.high_e_high_n"), ("v2", "persona.high_e_high_n.v2"), ("v3", "persona.high_e_high_n.v3")):
        n_sentence = personas[key].split(". ", 1)[1]
        by_template[template] = re.sub(r"\b(?:more|less)\b", "{degree}", n_sentence)

    assert len(set(by_template.values())) == 3
    audits = audit_persona_tokens(_Tokenizer(), personas, suffixes, _render)
    assert all(
        len(audit.axis_difference_spans[axis]) == 2
        for audit in audits
        for axis in ("extraversion", "neuroticism")
    )


def test_v3_only_changes_the_four_v1_worry_naturalness_substitutions():
    v2 = load_prompts(ROOT / "formal_v2_personas_v2.md")
    v3 = load_prompts(ROOT / "formal_v2_personas_v3.md")
    changed = [key for key in v2 if v2[key] != v3[key]]

    assert changed == [
        "persona.high_e_high_n",
        "persona.high_e_low_n",
        "persona.low_e_high_n",
        "persona.low_e_low_n",
    ]
    assert all(v2[key].replace("feel worry", "feel worried") == v3[key] for key in changed)
    personas, suffixes = _v3_drafts()
    assert len(audit_persona_tokens(_Tokenizer(), personas, suffixes, _render)) == 3


def test_baseline_raw_rows_summary_and_tidy_csv_are_consistent_and_immutable(tmp_path):
    hidden = _hidden()
    suffixes = {key: f"suffix {index}" for index, key in enumerate(CALIBRATION_SUFFIX_KEYS)}
    directions = {
        axis: {layer: np.asarray([1.0, 0.5, 0.0]) for layer in CALIBRATION_LAYERS}
        for axis in ("positive", "negative", "frustration")
    }
    rows = baseline_rows(hidden, directions, {layer: layer for layer in CALIBRATION_LAYERS}, suffixes)
    summary = summarize_baseline(rows)
    assert len(rows) == 3 * 12 * 3 * 5
    assert set(summary["by_suffix"]) == set(CALIBRATION_SUFFIX_KEYS)

    csv_rows = [
        {**row, "schema_version": 1, "snapshot_id": "test", "suffix_text_sha256": "x" * 64}
        for row in rows
    ]
    payload = {"schema_version": 1, "raw_scores": rows, "descriptive_summary": summary}
    json_path, csv_path = tmp_path / "snapshot.json", tmp_path / "snapshot.csv"
    write_snapshot(json_path, csv_path, payload, csv_rows)
    assert len(json_path.read_text(encoding="utf-8").splitlines()) > 1
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == len(rows) + 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_snapshot(json_path, csv_path, payload, csv_rows)


def test_calibration_context_has_no_game_and_no_generation_interface():
    messages = calibration_messages("A general persona.", "The window is closed.")

    assert messages == [
        {"role": "system", "content": "A general persona."},
        {"role": "user", "content": "The window is closed."},
    ]
    rendered = render_calibration_inputs(
        {"persona.high_e_high_n": "A general persona."},
        {"calibration.suffix.1": "The window is closed."},
        _render,
    )
    assert "mastermind" not in rendered["calibration.suffix.1"]["persona.high_e_high_n"].lower()
    assert "failure" not in rendered["calibration.suffix.1"]["persona.high_e_high_n"].lower()


def test_leave_one_template_out_pass_uses_only_six_prespecified_median_signs():
    result = evaluate_leave_one_template_out(_hidden())

    assert result["pass_rule"] == {
        "required_fold_axis_signs": 6,
        "observed_correct_fold_axis_signs": 6,
        "passed": True,
        "suffix_specific_signs": 18,
        "suffix_specific_correct_signs": 18,
        "robust": True,
        "suffix_sensitive": False,
    }
    assert len(result["combined"]) == 6


def test_existing_frustration_probe_is_contamination_only_not_a_pass_input():
    hidden = _hidden()
    baseline = evaluate_leave_one_template_out(hidden)["pass_rule"]
    directions = {
        "positive": {layer: np.asarray([1.0, 0.0, 0.0]) for layer in CALIBRATION_LAYERS},
        "negative": {layer: np.asarray([0.0, 1.0, 0.0]) for layer in CALIBRATION_LAYERS},
        "frustration": {layer: np.asarray([-1.0, -1.0, 0.0]) for layer in CALIBRATION_LAYERS},
    }

    audit = contamination_audit(
        hidden,
        directions,
        direction_layer_map={layer: layer for layer in CALIBRATION_LAYERS},
    )

    assert set(audit) == {"positive", "negative", "frustration"}
    assert evaluate_leave_one_template_out(hidden)["pass_rule"] == baseline


def test_calibration_requires_all_three_frozen_suffixes():
    hidden = _hidden()
    hidden.pop("calibration.suffix.3")

    with pytest.raises(ValueError, match="exactly the three frozen suffixes"):
        evaluate_leave_one_template_out(hidden)
