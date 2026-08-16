import json

import pytest

from analyze_response_pilot import analyze_paths, pair_records
from encouragement_lab.personas import PERSONA_SPECS


QUADRANT_DELTAS = {
    "high_e_high_n": 4,
    "high_e_low_n": 2,
    "low_e_high_n": 1,
    "low_e_low_n": 0,
}


def _projection(value):
    return {
        axis: {"median": value * scale, "layers": {"1": value * scale}}
        for axis, scale in (("positive", 0.1), ("negative", -0.2), ("frustration", 0.3))
    }


def _trajectory(tokens, start=100):
    return [
        {
            "token_position": start + offset,
            "token_id": offset + 1,
            "token_text": token,
            "projections": _projection(offset + 1),
        }
        for offset, token in enumerate(tokens)
    ]


def _record(spec, seed, condition, *, tokens=("{", "}"), response=None, **changes):
    treatment = QUADRANT_DELTAS[spec.quadrant_id] if condition == "encouragement" else 0
    values = {
        "schema_version": 1,
        "checkpoint_id": f"{spec.prompt_key}.seed-{seed}.round-5",
        "run_id": f"{spec.prompt_key}.seed-{seed}",
        "seed": seed,
        "persona_id": spec.prompt_key,
        "persona_quadrant": spec.quadrant_id,
        "persona_template_id": spec.template_id,
        "condition": condition,
        "generation_seed": seed * 101,
        "sampling": {"temperature": 0.5, "top_p": 0.9},
        "model": {"name": "test-model", "snapshot_sha256": "a" * 64},
        "source_file": "formal.jsonl",
        "source_sha256": "b" * 64,
        "intervention_prompt_sha256": "c" * 64,
        "direction_artifact_sha256": "d" * 64,
        "intervention": f"{condition} message",
        "response": response if response is not None else json.dumps(
            {"willingness": 5 + treatment}, separators=(",", ":")
        ),
        "generated_token_count": len(tokens),
        "pre_intervention_emotion": _projection(0),
        "prompt_end_emotion": _projection(treatment),
        "token_trajectory": _trajectory(tokens),
        "frustration_summary": {"first_token": 0.0, "final_token": 0.0},
    }
    values.update(changes)
    return values


def _balanced_records(seeds=(11, 12)):
    return [
        _record(spec, seed, condition)
        for seed in seeds
        for spec in PERSONA_SPECS
        for condition in ("encouragement", "neutral")
    ]


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def test_analysis_combines_files_and_uses_balanced_seed_contrasts(tmp_path):
    records = _balanced_records()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [record for record in records if record["seed"] == 11])
    _write_jsonl(second, [record for record in records if record["seed"] == 12])

    result = analyze_paths([first, second])

    assert result["completeness"] == {
        "record_count": 48,
        "condition_record_counts": {"encouragement": 24, "neutral": 24},
        "checkpoint_pair_count": 24,
        "seed_count": 2,
        "seeds": [11, 12],
        "persona_template_count": 12,
        "quadrant_seed_block_count": 8,
        "parsed_willingness_record_count": 48,
        "unparsed_willingness_record_count": 0,
        "valid_willingness_pair_count": 24,
        "invalid_willingness_pair_count": 0,
    }
    willingness = result["metrics"]["willingness"]
    assert willingness["average_treatment"]["mean"] == pytest.approx(1.75)
    assert willingness["average_treatment"]["exact_sign_flip_p"] == pytest.approx(0.5)
    assert willingness["planned_contrasts"]["extraversion"]["mean"] == pytest.approx(2.5)
    assert willingness["planned_contrasts"]["neuroticism"]["mean"] == pytest.approx(1.5)
    assert willingness["planned_contrasts"]["interaction"]["mean"] == pytest.approx(1.0)
    assert all(
        willingness["planned_contrasts"][name]["holm_adjusted_p"] is not None
        for name in ("extraversion", "neuroticism", "interaction")
    )
    assert result["metrics"]["prompt_end_positive"]["average_treatment"][
        "mean"
    ] == pytest.approx(0.175)
    assert result["metrics"]["prompt_end_negative"]["average_treatment"][
        "mean"
    ] == pytest.approx(-0.35)
    assert result["metrics"]["prompt_end_frustration"]["average_treatment"][
        "mean"
    ] == pytest.approx(0.525)
    diagnostics = result["output_token_diagnostics"]
    assert diagnostics["analysis_role"] == "descriptive_only_no_token_position_is_an_outcome"
    assert diagnostics["matched_length_pair_count"] == 24
    assert diagnostics["mismatched_length_pair_count"] == 0
    assert len(diagnostics["positions"]) == 2
    assert "projections" not in json.dumps(diagnostics)


def test_unparsed_willingness_is_reported_and_omits_its_seed_from_inference(tmp_path):
    records = _balanced_records()
    target = next(
        record
        for record in records
        if record["seed"] == 11
        and record["persona_id"] == "persona.high_e_high_n.v2"
        and record["condition"] == "neutral"
    )
    target["response"] = "no rating supplied"
    target["generated_token_count"] = 1
    target["token_trajectory"] = _trajectory(("no",))
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, records)

    result = analyze_paths([path])

    assert result["completeness"]["unparsed_willingness_record_count"] == 1
    assert result["completeness"]["invalid_willingness_pair_count"] == 1
    assert result["metrics"]["willingness"]["omitted_seeds"] == [11]
    assert result["metrics"]["willingness"]["average_treatment"]["count"] == 1
    assert result["output_token_diagnostics"]["mismatched_length_pair_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "must have exactly one branch"),
        (lambda rows: rows.append(dict(rows[0])), "duplicate 'encouragement' branch"),
        (
            lambda rows: rows[1].update(source_file="different.jsonl"),
            "mismatched shared field 'source_file'",
        ),
    ],
)
def test_pairing_rejects_incomplete_duplicate_and_mismatched_pairs(mutation, message):
    records = _balanced_records(seeds=(11,))
    mutation(records)

    with pytest.raises(ValueError, match=message):
        pair_records(records)


def test_pairing_rejects_incomplete_template_block():
    records = _balanced_records(seeds=(11,))
    records = [
        record
        for record in records
        if record["persona_id"] != "persona.high_e_high_n.v3"
    ]

    with pytest.raises(ValueError, match="incomplete template block"):
        pair_records(records)


def test_pairing_rejects_mixed_study_metadata():
    records = _balanced_records(seeds=(11,))
    records[-1]["intervention_prompt_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="mix study metadata"):
        pair_records(records)


def test_record_rejects_trajectory_length_mismatch():
    records = _balanced_records(seeds=(11,))
    records[0]["generated_token_count"] = 99

    with pytest.raises(ValueError, match="does not match token_trajectory length"):
        pair_records(records)
