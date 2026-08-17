from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from encouragement_lab.experiment import DryRunBackend
from encouragement_lab.formal_v2_records import (
    append,
    iter_runs,
    resume_index,
    validate,
)
from encouragement_lab.formal_v2_runner import (
    ARMS,
    FormalV2Runner,
    derive_generation_seed,
)
from encouragement_lab.model import SamplingConfig
from encouragement_lab.prompt_loader import load_prompts
import encouragement_lab.formal_v2_runner as runner_module


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "formal_v2_prompts.md"
SAMPLING = SamplingConfig(temperature=0)


class RecordingProbe:
    def __init__(self):
        self.rendered: list[str] = []

    def score_text(self, text: str) -> dict:
        self.rendered.append(text)
        return {"frustration": {"layers": {"-5": 1.0}, "median": 1.0}}


class SequenceBackend(DryRunBackend):
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls: list[dict] = []

    def generate(self, messages, *, seed, sampling):
        self.calls.append(
            {
                "messages": deepcopy(list(messages)),
                "seed": seed,
                "sampling": sampling,
            }
        )
        return self.outputs[len(self.calls) - 1]


def _runner(backend=None, *, probe=None) -> FormalV2Runner:
    return FormalV2Runner(backend or DryRunBackend(), PROMPTS, probe=probe)


def _complete_record(arm: str = "neutral") -> dict:
    return _runner().run("persona.high_e_high_n", 9001, arm, SAMPLING)


def test_runtime_candidate_preserves_accepted_source_text_exactly():
    historical = load_prompts(ROOT / "prompts.md")
    personas = load_prompts(ROOT / "formal_v2_personas_v3.md")
    fillers = load_prompts(ROOT / "formal_v2_filler_candidates_v4.md")
    candidate = load_prompts(PROMPTS)

    assert candidate["system.base"] == historical["system.base"]
    assert candidate["game.intro"] == historical["game.intro"]
    assert candidate["game.final_instruction"].strip() == (
        'To continue, Reply with exactly one JSON object in the form '
        '{"guess":"0123"} containing your next guess.'
    )
    for key, value in personas.items():
        if key.startswith("persona."):
            assert candidate[key] == value
    for key, value in fillers.items():
        if key.startswith("filler."):
            assert candidate[key] == value


def test_three_arms_have_five_shared_seed_prompt_boundary_rounds():
    probe = RecordingProbe()
    runner = _runner(probe=probe)
    rows = [
        runner.run("persona.high_e_high_n", 9001, arm, SAMPLING)
        for arm in ARMS
    ]

    assert [row["arm"] for row in rows] == list(ARMS)
    assert all(row["status"] == "complete_five_failures" for row in rows)
    assert all(row["completed_failure_rounds"] == 5 for row in rows)
    assert all(len(row["trajectory"]) == 5 for row in rows)
    assert len(probe.rendered) == 18  # 3 auxiliary baselines + 15 main readouts
    for attempt_index in range(5):
        seeds = {
            row["trajectory"][attempt_index]["generation_seed"] for row in rows
        }
        assert seeds == {derive_generation_seed(9001, attempt_index + 1)}

    for arm_index, row in enumerate(rows):
        assert row["transcript"][-1]["role"] == "user"
        assert len(row["transcript"]) == 12  # intro + five guesses + five feedbacks
        assert [item["attempt_index"] for item in row["trajectory"]] == [1, 2, 3, 4, 5]
        assert [item["failure_count_after"] for item in row["trajectory"]] == [1, 2, 3, 4, 5]
        for failure_count, item in enumerate(row["trajectory"], start=1):
            assert item["user_prompt"].endswith(
                runner.prompts["game.final_instruction"].strip()
            )
            expected_filler = (
                None
                if arm_index == 0
                else f"filler.{row['arm']}.{failure_count}"
            )
            assert item["filler_id"] == expected_filler


def test_invalid_output_counts_as_failure_without_changing_candidates():
    backend = SequenceBackend(
        [
            "not a guess",
            '{"guess":"0123"}',
            '{"guess":"4567"}',
            '{"guess":"8901"}',
            '{"guess":"2345"}',
        ]
    )
    record = _runner(backend).run(
        "persona.high_e_high_n", 9001, "neutral", SAMPLING
    )
    first = record["trajectory"][0]

    assert first["outcome"] == "invalid"
    assert first["failure_count_after"] == 1
    assert first["guess"] is None
    assert first["candidate_state_before"] == first["candidate_state_after"]
    assert first["feedback_frame"] == "invalid_guess"
    assert first["filler_id"] == "filler.neutral.1"


def test_one_unambiguous_four_digit_code_is_salvaged_and_scored():
    backend = SequenceBackend(
        [
            "My answer is 0123",
            '{"guess":"4567"}',
            '{"guess":"8901"}',
            '{"guess":"2345"}',
            '{"guess":"6789"}',
        ]
    )
    record = _runner(backend).run(
        "persona.high_e_high_n", 9001, "feedback_only", SAMPLING
    )
    first = record["trajectory"][0]

    assert first["guess"] == "0123"
    assert first["outcome"] == "unsuccessful"
    assert "response_format" in first["rule_violations"]
    assert first["candidate_state_after"] != first["candidate_state_before"]


class WinningState:
    def __init__(self, candidates=("0000", "1111")):
        self.candidates = tuple(candidates)

    def play(self, guess: str):
        return (4, 0), WinningState((guess,))


def test_early_win_is_retained_without_fake_failure_prompt_or_readout(monkeypatch):
    monkeypatch.setattr(runner_module, "AbsurdleState", WinningState)
    backend = SequenceBackend(['{"guess":"0000"}'])
    record = _runner(backend).run(
        "persona.high_e_high_n", 9001, "supportive", SAMPLING
    )

    assert record["status"] == "early_win"
    assert record["completed_failure_rounds"] == 0
    assert len(record["trajectory"]) == 1
    winning = record["trajectory"][0]
    assert winning["failure_count_after"] == 0
    assert winning["feedback"] == [4, 0]
    assert winning["filler_id"] is None
    assert winning["user_prompt"] is None
    assert winning["readout"] is None
    assert record["transcript"][-1]["role"] == "assistant"
    validate(record)


def test_record_validation_rejects_identity_state_and_nonfinite_errors():
    record = _complete_record()
    assert validate(record) == record

    wrong_id = deepcopy(record)
    wrong_id["run_id"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        validate(wrong_id)

    broken_state = deepcopy(record)
    broken_state["trajectory"][1]["candidate_state_before"] = {
        "candidate_count": 1,
        "candidate_set_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="discontinuous"):
        validate(broken_state)

    nonfinite = deepcopy(record)
    nonfinite["trajectory"][0]["raw_information_efficiency"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate(nonfinite)


def test_resume_rejects_unscheduled_duplicate_and_provenance_drift(tmp_path):
    record = _complete_record()
    output = tmp_path / "runs.jsonl"
    append(output, record)
    key = (record["persona_id"], record["seed"], record["arm"])

    assert resume_index(
        output,
        expected_provenance=record["provenance"],
        scheduled={key},
    ) == {key}
    with pytest.raises(ValueError, match="unscheduled"):
        resume_index(
            output,
            expected_provenance=record["provenance"],
            scheduled=set(),
        )
    with pytest.raises(ValueError, match="provenance mismatch"):
        resume_index(output, expected_provenance={}, scheduled={key})

    append(output, record)
    with pytest.raises(ValueError, match="duplicate"):
        resume_index(
            output,
            expected_provenance=record["provenance"],
            scheduled={key},
        )


def test_cli_requires_mode_refuses_overwrite_and_resumes_without_duplicates(tmp_path):
    output = tmp_path / "dry.jsonl"
    command = [sys.executable, str(ROOT / "run_formal_v2.py")]

    blocked = subprocess.run(
        [*command, "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0
    assert "choose exactly one" in blocked.stderr

    first = subprocess.run(
        [*command, "--dry-run", "--output", str(output), "--seed", "9001"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote" in first.stdout
    assert len(list(iter_runs(output))) == 3

    refused = subprocess.run(
        [*command, "--dry-run", "--output", str(output), "--seed", "9001"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "already contains records" in refused.stderr

    resumed = subprocess.run(
        [
            *command,
            "--dry-run",
            "--output",
            str(output),
            "--seed",
            "9001",
            "--resume",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert resumed.stdout.count("skipped complete") == 3
    assert len(list(iter_runs(output))) == 3


def test_formal_cli_rejects_non_mps_before_loading_model(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_formal_v2.py"),
            "--formal",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "formal.jsonl"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires explicit --device mps" in result.stderr
