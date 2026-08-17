from __future__ import annotations

from pathlib import Path

from encouragement_lab.experiment import DryRunBackend
from encouragement_lab.formal_v2_runner import derive_generation_seed
from encouragement_lab.formal_v3_records import iter_runs, validate
from encouragement_lab.formal_v3_runner import ARMS, FormalV3Runner
from encouragement_lab.model import SamplingConfig


ROOT = Path(__file__).resolve().parents[1]


class ThreeConceptProbe:
    def score_text(self, text: str) -> dict:
        return {
            axis: {"layers": {"17": value}, "median": value}
            for axis, value in (
                ("joyful", 0.1),
                ("grief_stricken", -0.2),
                ("furious", -0.3),
            )
        }


def test_v3_emits_new_identity_and_namespaced_shared_seeds() -> None:
    runner = FormalV3Runner(
        DryRunBackend(),
        ROOT / "formal_v2_prompts.md",
        probe=ThreeConceptProbe(),
        seed_namespace="formal-v3",
    )
    records = [
        runner.run(
            "persona.high_e_high_n", 3001, arm, SamplingConfig(temperature=0)
        )
        for arm in ARMS
    ]

    assert all(validate(record)["record_kind"] == "formal_v3_run" for record in records)
    for index in range(5):
        observed = {row["trajectory"][index]["generation_seed"] for row in records}
        assert observed == {
            derive_generation_seed(3001, index + 1, namespace="formal-v3")
        }
        assert set(records[0]["trajectory"][index]["readout"]) >= {
            "joyful", "grief_stricken", "furious", "rendered_prompt_sha256",
            "rendered_prompt_utf8_bytes",
        }


def test_v3_reader_rejects_v2_record_kind(tmp_path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"record_kind":"formal_v2_run"}\n', encoding="utf-8")
    try:
        list(iter_runs(source))
    except ValueError as error:
        assert "formal-v3" in str(error)
    else:
        raise AssertionError("v2 identity was accepted as formal-v3")
