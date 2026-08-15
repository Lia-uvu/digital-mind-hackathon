import json
from pathlib import Path

import pytest

from analyze_frustration import main
from encouragement_lab.manipulation import (
    FrustrationRun,
    analyze_frustration_jsonl,
    analyze_frustration_records,
    evaluate_manipulation_pass,
    summarize_runs,
)
from encouragement_lab.records import ExperimentRecord, append_record


def _axis(value):
    return {"frustration": {"median": value, "layers": {"10": value, "11": value}}}


def _trajectory(values=(0.2, 0.4, 0.8), violations=([], ["response_format"], [])):
    return [
        {
            "round": index,
            "emotion_after_feedback": _axis(value),
            "rule_violations": list(round_violations),
        }
        for index, (value, round_violations) in enumerate(zip(values, violations), start=1)
    ]


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
        "transcript": [],
        "pre_intervention_trajectory": _trajectory(),
        "failure_rounds": 3,
        "candidate_count_at_checkpoint": 451,
        "guess": "0123",
        "feedback": [0, 1],
        "raw_information_efficiency": 0.2,
        "optimal_information_efficiency": 0.4,
        "normalized_information_efficiency": 0.5,
        "rule_violations": [],
        "emotion_projections": {"baseline": _axis(0.1)},
        "emotion_summary": {},
        "willingness_to_continue": 6,
        "code_version": "deadbeef",
        "dependency_versions": {"repeng": "pinned"},
    }
    values.update(changes)
    return ExperimentRecord(**values).to_dict()


def neutral_record(**changes):
    values = {"condition": "neutral"}
    values.update(changes)
    return make_record(**values)


def test_jsonl_analysis_uses_shared_pre_intervention_trajectory_once(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    append_record(path, make_record(guess="9999", willingness_to_continue=1))
    append_record(path, neutral_record(guess="0000", willingness_to_continue=10))

    result = analyze_frustration_jsonl(path)
    assert len(result.runs) == 1
    run = result.runs[0]
    assert run.baseline == pytest.approx(0.1)
    assert run.round_frustration_medians == pytest.approx((0.2, 0.4, 0.8))
    assert run.final_round == 3
    assert run.baseline_to_final_delta == pytest.approx(0.7)
    assert run.round1_to_final_delta == pytest.approx(0.6)
    assert run.round_slope == pytest.approx(0.3)
    assert run.pre_intervention_guess_rule_violation_rounds == 1
    summary = result.by_persona["high_e_high_n"]
    assert summary.n == 1
    assert summary.final_round == 3
    assert summary.baseline_to_final_delta_median == pytest.approx(0.7)
    assert summary.round1_to_final_positive_fraction == 1
    assert summary.round_medians == pytest.approx((0.2, 0.4, 0.8))
    assert summary.fraction_above_round1_by_round == (None, 1, 1)


def test_cli_emits_json(tmp_path: Path, capsys):
    path = tmp_path / "records.jsonl"
    append_record(path, make_record())
    append_record(path, neutral_record())

    assert main([str(path), "--indent", "0"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["pooled"]["n"] == 1
    assert output["runs"][0]["final_round"] == 3
    assert output["pooled"]["round_medians"] == pytest.approx([0.2, 0.4, 0.8])
    assert output["pooled"]["fraction_above_round1_by_round"] == [None, 1, 1]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([make_record()], "missing branch"),
        ([make_record(), make_record()], "duplicate"),
        ([make_record(), neutral_record(seed=8)], "mismatched branch metadata"),
        ([make_record(), neutral_record(pre_intervention_trajectory=_trajectory((0.2, 0.3, 0.8)))], "mismatched branch metadata"),
    ],
)
def test_pairing_rejects_missing_duplicate_and_mismatched_pre_intervention_data(records, message):
    with pytest.raises(ValueError, match=message):
        analyze_frustration_records(records)


@pytest.mark.parametrize(
    ("alter_record", "message"),
    [
        (
            lambda record: record.update(emotion_projections={"baseline": {}}),
            "missing the frustration axis",
        ),
        (
            lambda record: record.update(
                pre_intervention_trajectory=[{"rule_violations": [], "emotion_after_feedback": {}}]
            ),
            "missing the frustration axis",
        ),
        (
            lambda record: record.update(
                pre_intervention_trajectory=[
                    {"rule_violations": [], "emotion_after_feedback": _axis(float("inf"))}
                ]
            ),
            "must be a finite number",
        ),
    ],
)
def test_validation_rejects_missing_frustration_axis_and_non_finite_values(alter_record, message):
    record = make_record()
    neutral = neutral_record()
    alter_record(record)
    alter_record(neutral)
    with pytest.raises(ValueError, match=message):
        analyze_frustration_records([record, neutral])


def test_invalid_willingness_is_not_a_pre_intervention_guess_violation():
    trajectory = _trajectory(
        values=(0.2, 0.4),
        violations=(["invalid_willingness_response"], ["wrong_length"]),
    )
    result = analyze_frustration_records(
        [make_record(pre_intervention_trajectory=trajectory), neutral_record(pre_intervention_trajectory=trajectory)]
    )
    assert result.runs[0].pre_intervention_guess_rule_violation_rounds == 1


def test_preregistered_pass_requires_all_thresholds():
    runs = tuple(
        FrustrationRun(
            run_id=f"run-{index}",
            persona_id="persona",
            baseline=0.1,
            round_frustration_medians=(0.2, 0.5),
            final_round=2,
            baseline_to_final_delta=0.4,
            round1_to_final_delta=0.3,
            round_slope=0.3,
            pre_intervention_guess_rule_violation_rounds=0,
        )
        for index in range(20)
    )
    summary = summarize_runs(runs)
    assert evaluate_manipulation_pass(runs, summary).passed is True

    failed = evaluate_manipulation_pass(runs[:19], summarize_runs(runs[:19]))
    assert failed.passed is False


def test_roundwise_summary_reports_medians_and_fraction_above_each_runs_round_one():
    runs = (
        FrustrationRun(
            "run-1", "persona", 0.1, (0.1, 0.3, 0.4, 0.6, 0.9), 5, 0.8, 0.8, 0.2, 0
        ),
        FrustrationRun(
            "run-2", "persona", 0.2, (0.2, 0.1, 0.5, 0.7, 1.0), 5, 0.8, 0.8, 0.2, 0
        ),
        FrustrationRun(
            "run-3", "persona", 0.3, (0.3, 0.4, 0.2, 0.8, 1.1), 5, 0.8, 0.8, 0.2, 0
        ),
    )

    summary = summarize_runs(runs)
    assert summary.round_medians == pytest.approx((0.2, 0.3, 0.4, 0.7, 1.0))
    assert summary.fraction_above_round1_by_round[0] is None
    assert summary.fraction_above_round1_by_round[1:] == pytest.approx((2 / 3, 2 / 3, 1, 1))
    assert summary.round_medians[-1] == max(summary.round_medians)


def test_roundwise_summary_is_none_for_inconsistent_run_lengths():
    two_rounds = FrustrationRun(
        "run-1", "persona", 0.1, (0.2, 0.5), 2, 0.4, 0.3, 0.3, 0
    )
    three_rounds = FrustrationRun(
        "run-2", "persona", 0.1, (0.2, 0.4, 0.7), 3, 0.6, 0.5, 0.25, 0
    )

    summary = summarize_runs((two_rounds, three_rounds))
    assert summary.final_round is None
    assert summary.round_medians is None
    assert summary.fraction_above_round1_by_round is None
    assert evaluate_manipulation_pass((two_rounds, three_rounds), summary).equal_final_rounds is False
