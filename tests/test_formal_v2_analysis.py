from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pytest

from encouragement_lab.experiment import DryRunBackend
from encouragement_lab.formal_v2_analysis import (
    DERIVED_CONTRAST,
    PRIMARY_CONTRASTS,
    analyze_records,
    holm_adjust,
)
from encouragement_lab.formal_v2_runner import FormalV2Runner
from encouragement_lab.formal_v2_records import append
from encouragement_lab.formal_v2_figures import write_figure_bundle
from encouragement_lab.model import SamplingConfig
from encouragement_lab.personas import PERSONA_SPECS
from run_formal_v2_analysis import TABLE_FIELDS, write_analysis_bundle


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "formal_v2_prompts.md"
SAMPLING = SamplingConfig(temperature=0)
QUADRANT_NEUTRAL_EFFECT = {
    "high_e_high_n": 4.0,
    "high_e_low_n": 2.0,
    "low_e_high_n": 0.0,
    "low_e_low_n": -2.0,
}
QUADRANT_SUPPORT_EFFECT = {
    "high_e_high_n": 1.0,
    "high_e_low_n": 3.0,
    "low_e_high_n": 5.0,
    "low_e_low_n": 7.0,
}
TEMPLATE_OFFSET = {"v1": -1.0, "v2": 0.0, "v3": 1.0}


class Probe:
    def score_text(self, text: str) -> dict:
        return {"frustration": {"layers": {"-5": 0.0}, "median": 0.0}}


@lru_cache(maxsize=1)
def _base_records() -> dict[str, dict]:
    runner = FormalV2Runner(DryRunBackend(), PROMPTS, probe=Probe())
    return {
        arm: runner.run("persona.high_e_high_n", 1, arm, SAMPLING)
        for arm in ("feedback_only", "neutral", "supportive")
    }


def _record(persona_id: str, seed: int, arm: str, run_slope: float) -> dict:
    spec = next(item for item in PERSONA_SPECS if item.prompt_key == persona_id)
    record = deepcopy(_base_records()[arm])
    record.update(
        {
            "run_id": f"{persona_id}.seed-{seed}.{arm}",
            "seed": seed,
            "arm": arm,
            "persona_id": persona_id,
            "persona_quadrant": spec.quadrant_id,
            "persona_template_id": spec.template_id,
        }
    )
    for round_number, attempt in enumerate(record["trajectory"], start=1):
        value = 10.0 + run_slope * (round_number - 1)
        attempt["readout"] = {
            "frustration": {"layers": {"-5": value}, "median": value}
        }
    return record


def _design(*, positive_feedback_seeds: int = 7) -> list[dict]:
    records: list[dict] = []
    for seed in range(1, 11):
        feedback_slope = 1.0 if seed <= positive_feedback_seeds else -1.0
        for spec in PERSONA_SPECS:
            template_offset = TEMPLATE_OFFSET[spec.template_id]
            neutral_effect = (
                QUADRANT_NEUTRAL_EFFECT[spec.quadrant_id] + template_offset
            )
            support_effect = (
                QUADRANT_SUPPORT_EFFECT[spec.quadrant_id] - template_offset
            )
            slopes = {
                "feedback_only": feedback_slope,
                "neutral": feedback_slope + neutral_effect,
                "supportive": feedback_slope + neutral_effect + support_effect,
            }
            for arm, value in slopes.items():
                records.append(_record(spec.prompt_key, seed, arm, value))
    return records


def _seed_row(result: dict, contrast: str, seed: int = 1) -> dict:
    return next(
        row
        for row in result["seed_contrasts"]
        if row["contrast"] == contrast and row["seed"] == seed
    )


def test_complete_design_preserves_aggregation_order_and_planned_families():
    result = analyze_records(_design(), expected_seeds=tuple(range(1, 11)))

    assert len(result["round_rows"]) == 360 * 5
    assert len(result["run_rows"]) == 360
    assert len(result["template_contrasts"]) == 10 * 12 * 3
    assert len(result["quadrant_contrasts"]) == 10 * 4 * 3
    assert len(result["seed_contrasts"]) == 10 * 3
    assert result["run_exclusions"] == []
    assert result["missing_records"] == []
    assert result["contrast_missing_map"] == []

    neutral = _seed_row(result, PRIMARY_CONTRASTS[0])
    assert neutral["overall_slope"] == pytest.approx(1.0)
    assert neutral["extraversion_slope"] == pytest.approx(4.0)
    assert neutral["neuroticism_slope"] == pytest.approx(2.0)
    assert neutral["interaction_slope"] == pytest.approx(0.0)
    assert neutral["overall_r5_minus_r1"] == pytest.approx(4.0)

    supportive = _seed_row(result, PRIMARY_CONTRASTS[1])
    assert supportive["overall_slope"] == pytest.approx(4.0)
    assert supportive["extraversion_slope"] == pytest.approx(-4.0)
    assert supportive["neuroticism_slope"] == pytest.approx(-2.0)
    assert supportive["interaction_slope"] == pytest.approx(0.0)

    derived = _seed_row(result, DERIVED_CONTRAST)
    for effect in ("overall", "extraversion", "neuroticism", "interaction"):
        assert derived[f"{effect}_slope"] == pytest.approx(
            neutral[f"{effect}_slope"] + supportive[f"{effect}_slope"]
        )

    inference = result["inference"]
    assert len(inference["co_primary"]) == 2
    assert all(row["count"] == 10 for row in inference["co_primary"])
    assert all(row["holm_adjusted_p"] is not None for row in inference["co_primary"])
    assert len(inference["moderation"]) == 6
    assert all(row["holm_adjusted_p"] is not None for row in inference["moderation"])
    assert inference["derived_total"]["confirmatory_test"] is None
    assert all(
        row["confirmatory_test"] is None
        for row in inference["r5_minus_r1_robustness"]
    )

    manipulation = result["manipulation"]
    assert manipulation["eligible_seed_count"] == 10
    assert manipulation["positive_r5_minus_r1_count"] == 7
    assert manipulation["slope_median"] > 0
    assert manipulation["r5_minus_r1_median"] > 0
    assert manipulation["passed"] is True


def test_six_positive_endpoint_seeds_fail_only_the_direction_consistency_gate():
    result = analyze_records(
        _design(positive_feedback_seeds=6), expected_seeds=tuple(range(1, 11))
    )
    manipulation = result["manipulation"]

    assert manipulation["slope_median"] > 0
    assert manipulation["r5_minus_r1_median"] > 0
    assert manipulation["positive_r5_minus_r1_count"] == 6
    assert manipulation["criteria"] == {
        "exactly_ten_eligible_seeds": True,
        "positive_seed_slope_median": True,
        "positive_seed_r5_minus_r1_median": True,
        "at_least_seven_positive_seed_r5_minus_r1": False,
    }
    assert manipulation["passed"] is False
    assert "retain all data" in manipulation["failure_interpretation"]


def test_ineligible_runs_leave_no_partial_rows_and_omit_whole_seed_blocks():
    records = _design()
    missing_probe = next(
        row
        for row in records
        if row["seed"] == 1
        and row["persona_id"] == "persona.high_e_high_n"
        and row["arm"] == "neutral"
    )
    missing_probe["trajectory"][2]["readout"] = {}

    early_win = next(
        row
        for row in records
        if row["seed"] == 1
        and row["persona_id"] == "persona.high_e_high_n.v2"
        and row["arm"] == "supportive"
    )
    first = early_win["trajectory"][0]
    first.update(
        {
            "failure_count_after": 0,
            "outcome": "win",
            "feedback": [4, 0],
            "feedback_frame": "win",
            "filler_id": None,
            "filler_text": None,
            "user_prompt": None,
            "readout": None,
        }
    )
    early_win["status"] = "early_win"
    early_win["completed_failure_rounds"] = 0
    early_win["trajectory"] = [first]
    early_win["transcript"] = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "game"},
        {"role": "assistant", "content": first["raw_response"]},
    ]
    early_win["final_candidate_state"] = first["candidate_state_after"]

    result = analyze_records(records, expected_seeds=tuple(range(1, 11)))
    excluded_ids = {row["run_id"] for row in result["run_exclusions"]}
    assert excluded_ids == {missing_probe["run_id"], early_win["run_id"]}
    assert not any(
        row["run_id"] in excluded_ids for row in result["round_rows"]
    )

    seed_one_contrasts = {
        row["contrast"]
        for row in result["seed_contrasts"]
        if row["seed"] == 1
    }
    assert seed_one_contrasts == set()
    assert {
        row["contrast"]
        for row in result["contrast_missing_map"]
        if row["seed"] == 1
    } == set((*PRIMARY_CONTRASTS, DERIVED_CONTRAST))
    assert result["manipulation"]["eligible_seed_count"] == 10


def test_missing_record_is_reported_and_removes_only_affected_seed_contrasts():
    records = _design()
    removed = next(
        row
        for row in records
        if row["seed"] == 2
        and row["persona_id"] == "persona.low_e_low_n.v3"
        and row["arm"] == "supportive"
    )
    records.remove(removed)
    result = analyze_records(records, expected_seeds=tuple(range(1, 11)))

    assert result["missing_records"] == [
        {
            "persona_id": "persona.low_e_low_n.v3",
            "seed": 2,
            "arm": "supportive",
            "reason": "missing_record",
        }
    ]
    seed_two = {
        row["contrast"]
        for row in result["seed_contrasts"]
        if row["seed"] == 2
    }
    assert seed_two == {PRIMARY_CONTRASTS[0]}
    assert result["manipulation"]["eligible_seed_count"] == 10


def test_duplicate_provenance_and_persona_metadata_are_strictly_rejected():
    record = _record("persona.high_e_high_n", 1, "neutral", 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        analyze_records([record, deepcopy(record)], expected_seeds=(1,))

    drifted = _record("persona.high_e_high_n", 1, "supportive", 1.0)
    drifted["provenance"]["code_version"] = "different"
    with pytest.raises(ValueError, match="provenance mismatch"):
        analyze_records([record, drifted], expected_seeds=(1,))

    bad_metadata = deepcopy(record)
    bad_metadata["persona_quadrant"] = "low_e_low_n"
    with pytest.raises(ValueError, match="persona metadata mismatch"):
        analyze_records([bad_metadata], expected_seeds=(1,))


def test_holm_adjustment_is_monotone_in_sorted_order():
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )


def test_bundle_writer_is_tidy_hashed_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source.jsonl"
    records = [record for record in _design() if record["seed"] == 1]
    for record in records:
        append(source, record)
    destination = tmp_path / "analysis"

    summary = write_analysis_bundle(
        source, destination, expected_seeds=(1,)
    )

    assert (destination / "summary.json").is_file()
    assert (destination / "DATA_DICTIONARY.md").is_file()
    assert all((destination / name).is_file() for name in TABLE_FIELDS)
    assert summary["row_counts"]["rounds.csv"] == 36 * 5
    assert summary["row_counts"]["runs.csv"] == 36
    assert summary["manipulation"]["passed"] is False
    assert set(summary["artifact_sha256"]) == {
        *TABLE_FIELDS,
        "DATA_DICTIONARY.md",
    }

    with pytest.raises(FileExistsError, match="already exists"):
        write_analysis_bundle(source, destination, expected_seeds=(1,))


def test_figure_bundle_exports_paper_formats_and_self_contained_html(tmp_path):
    source = tmp_path / "source.jsonl"
    for record in _design():
        append(source, record)
    analysis = tmp_path / "analysis"
    write_analysis_bundle(source, analysis, expected_seeds=tuple(range(1, 11)))
    figures = tmp_path / "figures"

    manifest = write_figure_bundle(analysis, figures)

    expected = {
        "formal-v2-trajectory.svg",
        "formal-v2-trajectory.pdf",
        "formal-v2-trajectory.png",
        "formal-v2-slope-contrasts.svg",
        "formal-v2-slope-contrasts.pdf",
        "formal-v2-slope-contrasts.png",
        "formal-v2-report.html",
    }
    assert set(manifest["artifacts_sha256"]) == expected
    assert manifest["complete_all_arm_seeds"] == list(range(1, 11))
    assert (figures / "formal-v2-trajectory.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert (figures / "formal-v2-trajectory.pdf").read_bytes().startswith(b"%PDF")
    svg = (figures / "formal-v2-trajectory.svg").read_text(encoding="utf-8")
    assert "frustration-direction" in svg
    html = (figures / "formal-v2-report.html").read_text(encoding="utf-8")
    assert "data:image/svg+xml;base64," in html
    assert "Co-primary slope contrasts" in html
    assert "subjective experience" in html
