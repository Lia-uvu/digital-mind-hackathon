from pathlib import Path

import pytest

from encouragement_lab.experiment import (
    DryRunBackend,
    ExperimentRunner,
    RunConfig,
    derive_generation_seed,
    parse_guess,
    parse_willingness,
)
from encouragement_lab.model import SamplingConfig
from encouragement_lab.personas import PERSONA_KEYS
from encouragement_lab.records import append_record, file_checksum
from run_experiment import _resume_state, code_version, source_snapshot_checksum


ROOT = Path(__file__).resolve().parents[1]


def test_guess_and_willingness_parsing_distinguish_format_violations() -> None:
    assert parse_guess('{"guess":"0012"}').guess == "0012"
    salvaged = parse_guess("My guess is 0012")
    assert salvaged.guess == "0012"
    assert "response_format" in salvaged.violations
    assert parse_guess("0012 or 3456").guess is None
    assert parse_willingness('{"willingness":10}') == 10
    assert parse_willingness("I would say 6.") == 6
    assert parse_willingness("between 5 and 6") is None


def test_generation_seed_streams_do_not_overlap_adjacent_runs() -> None:
    first = {
        derive_generation_seed(101, "pre_intervention", round_index)
        for round_index in range(5)
    }
    second = {
        derive_generation_seed(102, "pre_intervention", round_index)
        for round_index in range(5)
    }
    assert first.isdisjoint(second)
    assert derive_generation_seed(101, "branch_guess") not in first


def test_code_version_includes_exact_source_snapshot_checksum() -> None:
    checksum = source_snapshot_checksum()
    assert len(checksum) == 64
    assert f"source-sha256:{checksum}" in code_version()


def test_dry_run_pair_shares_checkpoint_and_writes_both_conditions() -> None:
    runner = ExperimentRunner(DryRunBackend(), ROOT / "prompts.md")
    config = RunConfig(
        seed=11,
        failure_rounds=5,
        sampling=SamplingConfig(temperature=0),
    )
    encouragement, neutral = runner.run_pair("persona.high_e_high_n", config)

    assert encouragement.condition == "encouragement"
    assert neutral.condition == "neutral"
    assert encouragement.checkpoint_id == neutral.checkpoint_id
    assert (
        encouragement.candidate_count_at_checkpoint
        == neutral.candidate_count_at_checkpoint
        == 24
    )
    # Everything through the checkpoint is byte-for-byte paired.
    checkpoint_length = 2 + 2 * config.failure_rounds
    assert (
        encouragement.transcript[:checkpoint_length]
        == neutral.transcript[:checkpoint_length]
    )
    assert encouragement.rule_violations == []
    assert neutral.rule_violations == []
    assert encouragement.willingness_to_continue == 7
    assert neutral.willingness_to_continue == 7
    assert encouragement.persona_quadrant == neutral.persona_quadrant == "high_e_high_n"
    assert encouragement.persona_template_id == neutral.persona_template_id == "v1"


def test_all_persona_templates_pass_lint_and_dry_checkpoint_creation() -> None:
    runner = ExperimentRunner(DryRunBackend(), ROOT / "prompts.md")
    config = RunConfig(seed=23, failure_rounds=1, sampling=SamplingConfig(temperature=0))

    assert len(PERSONA_KEYS) == 12
    for persona_id in PERSONA_KEYS:
        checkpoint = runner.make_checkpoint(persona_id, config)
        assert checkpoint.persona_id == persona_id


def test_resume_indexes_one_partial_branch_and_rejects_metadata_drift(tmp_path) -> None:
    runner = ExperimentRunner(
        DryRunBackend(), ROOT / "prompts.md", code_version="frozen-source"
    )
    sampling = SamplingConfig(temperature=0)
    record = runner.run_pair(
        "persona.high_e_high_n", RunConfig(seed=31, failure_rounds=5, sampling=sampling)
    )[0]
    output = tmp_path / "partial.jsonl"
    append_record(output, record)
    arguments = {
        "enabled": True,
        "personas": ["persona.high_e_high_n"],
        "seeds": [31],
        "failure_rounds": 5,
        "sampling": {
            "max_new_tokens": 48,
            "temperature": 0,
            "top_p": 0.9,
        },
        "prompt_hash": file_checksum(ROOT / "prompts.md"),
        "model_metadata": DryRunBackend().metadata(),
        "probe": None,
        "version": "frozen-source",
    }

    state = _resume_state(output, **arguments)

    assert state == {("persona.high_e_high_n", 31): {"encouragement"}}
    with pytest.raises(SystemExit, match="code_version"):
        _resume_state(output, **{**arguments, "version": "changed-source"})


def test_willingness_prompt_cannot_anchor_a_numeric_answer(tmp_path: Path) -> None:
    source = (ROOT / "prompts.md").read_text(encoding="utf-8")
    anchored = source.replace(
        'only the key "willingness" and your integer rating',
        'the form {"willingness":7}',
    )
    prompts = tmp_path / "prompts.md"
    prompts.write_text(anchored, encoding="utf-8")

    try:
        ExperimentRunner(DryRunBackend(), prompts)
    except ValueError as error:
        assert "must not anchor" in str(error)
    else:
        raise AssertionError("numeric willingness anchor was not rejected")
