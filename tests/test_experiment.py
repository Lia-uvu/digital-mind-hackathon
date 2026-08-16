from pathlib import Path

import pytest

from encouragement_lab.experiment import (
    DryRunBackend,
    ExperimentRunner,
    RunConfig,
    _emotion_score,
    derive_generation_seed,
    parse_guess,
    parse_willingness,
)
from encouragement_lab.model import LocalChatModel, SamplingConfig
from encouragement_lab.personas import PERSONA_KEYS
from encouragement_lab.records import append_record, file_checksum
from run_experiment import _resume_state, code_version, source_snapshot_checksum


ROOT = Path(__file__).resolve().parents[1]


class _RecordingRenderBackend(DryRunBackend):
    def __init__(self) -> None:
        self.add_generation_prompt_calls: list[bool] = []

    def render_messages(self, messages, *, add_generation_prompt=True):
        self.add_generation_prompt_calls.append(add_generation_prompt)
        return super().render_messages(
            messages, add_generation_prompt=add_generation_prompt
        )


class _RecordingEmotionProbe:
    def __init__(self) -> None:
        self.rendered_texts: list[str] = []

    def score_text(self, rendered_text: str) -> dict:
        self.rendered_texts.append(rendered_text)
        return {}


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


def test_post_guess_emotion_scores_completed_assistant_turn() -> None:
    backend = _RecordingRenderBackend()
    runner = ExperimentRunner(
        backend,
        ROOT / "prompts.md",
        emotion_probe=_RecordingEmotionProbe(),
    )
    config = RunConfig(
        seed=11,
        failure_rounds=1,
        sampling=SamplingConfig(temperature=0),
    )
    checkpoint = runner.make_checkpoint("persona.high_e_high_n", config)
    backend.add_generation_prompt_calls.clear()

    runner.run_branch(checkpoint, "encouragement", config, optimum=1.0)

    # post-message is a user-ending next-generation boundary; post-guess is an
    # assistant-ending completed-turn boundary.
    assert backend.add_generation_prompt_calls == [True, False]


def test_qwen_assistant_ending_boundary_does_not_add_empty_turn() -> None:
    transformers = pytest.importorskip("transformers")
    model_path = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
    if not model_path.is_dir():
        pytest.skip("local frozen Qwen tokenizer is unavailable")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    backend = LocalChatModel(
        model=None,
        tokenizer=tokenizer,
        model_name=str(model_path),
        device=None,
    )
    probe = _RecordingEmotionProbe()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "guess"},
        {"role": "assistant", "content": '{"guess":"0123"}'},
    ]

    _emotion_score(backend, probe, messages)

    expected = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    assert probe.rendered_texts == [expected]
    assert expected.endswith("<|im_end|>\n")
    assert not expected.endswith("<|im_start|>assistant\n")


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
