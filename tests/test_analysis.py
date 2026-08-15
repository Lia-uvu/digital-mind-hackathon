import pytest

from encouragement_lab.analysis import (
    analyze_jsonl,
    pair_records,
    summarize_by_persona,
    summarize_by_quadrant,
)
from encouragement_lab.factorial import analyze_factorial
from encouragement_lab.records import ExperimentRecord, append_record


def make_record(**changes):
    values = {
        "run_id": "run-1",
        "seed": 7,
        "branch_seed": 7001,
        "checkpoint_id": "round-3",
        "model": {"name": "test-model", "revision": "abc"},
        "sampling": {"temperature": 0},
        "persona_id": "high_e_high_n",
        "prompt_checksum": "a" * 64,
        "condition": "encouragement",
        "transcript": [{"role": "user", "content": "shared pre-checkpoint history"}],
        "pre_intervention_trajectory": [],
        "failure_rounds": 3,
        "candidate_count_at_checkpoint": 451,
        "guess": "0123",
        "feedback": [0, 1],
        "raw_information_efficiency": 0.2,
        "optimal_information_efficiency": 0.4,
        "normalized_information_efficiency": 0.8,
        "rule_violations": [],
        "emotion_projections": {},
        "emotion_summary": {
            "message_delta": {"positive": 0.3, "negative": -0.2, "frustration": 0.4},
            "post_guess_delta": {"positive": 0.1, "negative": -0.1, "frustration": 0.2},
        },
        "willingness_to_continue": 8,
        "code_version": "deadbeef",
        "dependency_versions": {"repeng": "pinned"},
    }
    values.update(changes)
    return ExperimentRecord(**values).to_dict()


def neutral_record(**changes):
    values = {
        "condition": "neutral",
        "normalized_information_efficiency": 0.5,
        "willingness_to_continue": 5,
        "emotion_summary": {
            "message_delta": {"positive": 0.1, "negative": -0.4, "frustration": 0.1},
            "post_guess_delta": {"positive": -0.1, "negative": -0.3, "frustration": -0.2},
        },
    }
    values.update(changes)
    return make_record(**values)


def test_analyze_jsonl_returns_encouragement_minus_neutral_deltas(tmp_path):
    output = tmp_path / "records.jsonl"
    append_record(output, make_record())
    append_record(output, neutral_record())

    pair = analyze_jsonl(output)[0]
    assert pair.normalized_information_efficiency == pytest.approx(0.3)
    assert pair.normalized_information_efficiency_delta == pytest.approx(0.3)
    assert pair.willingness_to_continue == pytest.approx(3)
    assert pair.positive_message_delta == pytest.approx(0.2)
    assert pair.negative_message_delta == pytest.approx(0.2)
    assert pair.frustration_message_delta == pytest.approx(0.3)
    assert pair.positive_post_guess_delta == pytest.approx(0.2)
    assert pair.negative_post_guess_delta == pytest.approx(0.2)
    assert pair.frustration_post_guess_delta == pytest.approx(0.4)
    assert pair.hard_rule_violation_rate == 0


@pytest.mark.parametrize(
    ("encouragement_violations", "neutral_violations", "expected_delta"),
    [
        ([], [], 0),
        (["wrong_length"], [], 1),
        (["invalid_willingness_response"], [], 0),
    ],
)
def test_hard_rule_violation_counts_only_game_guess_violations(
    encouragement_violations, neutral_violations, expected_delta
):
    pair = pair_records(
        [
            make_record(rule_violations=encouragement_violations),
            neutral_record(rule_violations=neutral_violations),
        ]
    )[0]
    assert pair.hard_rule_violation_rate == expected_delta


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([make_record()], "missing branch"),
        ([make_record(), make_record()], "duplicate"),
        ([make_record(), neutral_record(candidate_count_at_checkpoint=999)], "mismatched checkpoint metadata"),
    ],
)
def test_pairing_rejects_incomplete_duplicate_and_mismatched_checkpoints(records, message):
    with pytest.raises(ValueError, match=message):
        pair_records(records)


def test_summary_averages_checkpoints_before_counting_independent_runs():
    first_checkpoint = pair_records([make_record(), neutral_record()])[0]
    second_checkpoint = pair_records(
        [
            make_record(
                checkpoint_id="round-4",
                normalized_information_efficiency=1.0,
                rule_violations=["wrong_length"],
            ),
            neutral_record(checkpoint_id="round-4", normalized_information_efficiency=0.5),
        ]
    )[0]
    second_run = pair_records(
        [
            make_record(run_id="run-2", normalized_information_efficiency=0.6),
            neutral_record(run_id="run-2", normalized_information_efficiency=0.5),
        ]
    )[0]

    summary = summarize_by_persona([first_checkpoint, second_checkpoint, second_run])["high_e_high_n"]
    # run-1 first averages to (0.3 + 0.5) / 2 = 0.4; run-2 is 0.1.
    assert summary.normalized_information_efficiency.count == 2
    assert summary.normalized_information_efficiency.mean == pytest.approx(0.25)
    assert summary.normalized_information_efficiency.sample_stdev == pytest.approx(0.212132034)
    assert summary.willingness_to_continue.sample_stdev == pytest.approx(0)
    assert summary.frustration_message_delta.mean == pytest.approx(0.3)
    assert summary.frustration_message_delta.sample_stdev == pytest.approx(0)
    assert summary.frustration_post_guess_delta.mean == pytest.approx(0.4)
    assert summary.frustration_post_guess_delta.sample_stdev == pytest.approx(0)
    # run-1 averages the no-violation and encouragement-only-violation
    # checkpoints to 0.5; run-2 is 0.
    assert summary.hard_rule_violation_rate.count == 2
    assert summary.hard_rule_violation_rate.mean == pytest.approx(0.25)
    assert summary.hard_rule_violation_rate.sample_stdev == pytest.approx(0.353553391)


def test_summary_uses_none_stdev_for_one_independent_run():
    pair = pair_records([make_record(), neutral_record()])[0]
    summary = summarize_by_persona([pair])["high_e_high_n"]
    assert summary.normalized_information_efficiency.count == 1
    assert summary.normalized_information_efficiency.sample_stdev is None


def test_missing_willingness_does_not_discard_primary_pair_metrics():
    pair = pair_records(
        [
            make_record(willingness_to_continue=None),
            neutral_record(willingness_to_continue=6),
        ]
    )[0]

    assert pair.willingness_to_continue is None
    assert pair.normalized_information_efficiency == pytest.approx(0.3)
    summary = summarize_by_persona([pair])["high_e_high_n"]
    assert summary.willingness_to_continue.count == 0
    assert summary.willingness_to_continue.mean is None
    assert summary.normalized_information_efficiency.count == 1


def _template_pair(
    seed,
    persona_id,
    template_id,
    encouragement_efficiency,
    *,
    quadrant="high_e_high_n",
):
    shared = {
        "run_id": f"{persona_id}.seed-{seed}",
        "seed": seed,
        "checkpoint_id": f"{persona_id}.seed-{seed}.round-5",
        "persona_id": persona_id,
        "persona_quadrant": quadrant,
        "persona_template_id": template_id,
    }
    return pair_records(
        [
            make_record(
                **shared,
                normalized_information_efficiency=encouragement_efficiency,
            ),
            neutral_record(**shared, normalized_information_efficiency=0.5),
        ]
    )[0]


def test_quadrant_summary_averages_templates_before_counting_seeds():
    pairs = []
    persona_ids = (
        "persona.high_e_high_n",
        "persona.high_e_high_n.v2",
        "persona.high_e_high_n.v3",
    )
    for seed, encouragement_values in ((11, (0.6, 0.7, 0.8)), (12, (0.9, 1.0, 1.1))):
        for persona_id, template_id, value in zip(
            persona_ids, ("v1", "v2", "v3"), encouragement_values, strict=True
        ):
            pairs.append(_template_pair(seed, persona_id, template_id, value))

    summary = summarize_by_quadrant(pairs)["high_e_high_n"]

    assert summary.template_ids == ("v1", "v2", "v3")
    assert summary.normalized_information_efficiency.count == 2
    assert summary.normalized_information_efficiency.mean == pytest.approx(0.35)
    assert summary.normalized_information_efficiency.sample_stdev == pytest.approx(
        0.212132034
    )


def test_quadrant_summary_rejects_an_incomplete_template_block():
    pair = _template_pair(11, "persona.high_e_high_n", "v1", 0.6)

    with pytest.raises(ValueError, match="incomplete template block"):
        summarize_by_quadrant([pair])


def test_factorial_analysis_uses_balanced_seed_level_planned_contrasts():
    quadrant_values = {
        "high_e_high_n": 0.4,
        "high_e_low_n": 0.2,
        "low_e_high_n": 0.1,
        "low_e_low_n": 0.0,
    }
    pairs = []
    for seed in (11, 12):
        for quadrant, delta in quadrant_values.items():
            for template_id in ("v1", "v2", "v3"):
                suffix = "" if template_id == "v1" else f".{template_id}"
                pairs.append(
                    _template_pair(
                        seed,
                        f"persona.{quadrant}{suffix}",
                        template_id,
                        0.5 + delta,
                        quadrant=quadrant,
                    )
                )

    result = analyze_factorial(pairs)

    assert result.omitted_seeds == ()
    assert result.average_treatment.mean == pytest.approx(0.175)
    assert result.extraversion.mean == pytest.approx(0.25)
    assert result.neuroticism.mean == pytest.approx(0.15)
    assert result.interaction.mean == pytest.approx(0.1)
    assert result.interaction.count == 2
    assert result.interaction.exact_sign_flip_p == pytest.approx(0.5)
    assert result.interaction.holm_adjusted_p is not None


def test_factorial_secondary_metric_omits_seed_with_any_missing_template_value():
    pairs = []
    for quadrant in (
        "high_e_high_n",
        "high_e_low_n",
        "low_e_high_n",
        "low_e_low_n",
    ):
        for template_id in ("v1", "v2", "v3"):
            suffix = "" if template_id == "v1" else f".{template_id}"
            pair = _template_pair(
                11,
                f"persona.{quadrant}{suffix}",
                template_id,
                0.7,
                quadrant=quadrant,
            )
            if quadrant == "high_e_high_n" and template_id == "v2":
                pair = pair.__class__(
                    **{**pair.__dict__, "willingness_to_continue": None}
                )
            pairs.append(pair)

    result = analyze_factorial(pairs, metric="willingness_to_continue")

    assert result.by_seed == ()
    assert result.omitted_seeds == (11,)
    assert result.average_treatment.count == 0
