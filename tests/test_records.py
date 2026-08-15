from dataclasses import dataclass
from enum import Enum
import hashlib
import json

import pytest

from encouragement_lab.records import (
    SCHEMA_VERSION,
    ExperimentRecord,
    append_record,
    file_checksum,
    prompt_checksum,
    read_records,
    to_jsonable,
)


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
        "transcript": [{"role": "user", "content": "你好"}],
        "pre_intervention_trajectory": [],
        "failure_rounds": 3,
        "candidate_count_at_checkpoint": 451,
        "guess": "0123",
        "feedback": [0, 1],
        "raw_information_efficiency": 0.2,
        "optimal_information_efficiency": 0.4,
        "normalized_information_efficiency": 0.5,
        "rule_violations": [],
        "emotion_projections": {"baseline": {"layers": {"12": 0.1}}},
        "emotion_summary": {"positive": 0.1, "negative": -0.1},
        "willingness_to_continue": 6,
        "code_version": "deadbeef",
        "dependency_versions": {"repeng": "pinned"},
    }
    values.update(changes)
    return ExperimentRecord(**values)


def test_append_is_utf8_jsonl_and_does_not_overwrite(tmp_path):
    output = tmp_path / "nested" / "records.jsonl"
    append_record(output, make_record(run_id="first"))
    append_record(output, make_record(run_id="second", condition="neutral"))

    lines = output.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["first", "second"]
    assert json.loads(lines[0])["transcript"][0]["content"] == "你好"
    assert [record["run_id"] for record in read_records(output)] == ["first", "second"]


def test_reader_rejects_wrong_schema_and_missing_fields(tmp_path):
    output = tmp_path / "records.jsonl"
    malformed = make_record().to_dict()
    malformed["schema_version"] = SCHEMA_VERSION + 1
    output.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_records(output)

    malformed = make_record().to_dict()
    del malformed["guess"]
    output.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        read_records(output)


@dataclass
class NestedValue:
    value: tuple[int, int]


class Status(Enum):
    READY = "ready"


def test_recursive_json_conversion_and_checksums(tmp_path):
    converted = to_jsonable({"nested": NestedValue((1, 2)), "status": Status.READY})
    assert converted == {"nested": {"value": [1, 2]}, "status": "ready"}
    with pytest.raises(ValueError, match="non-finite"):
        to_jsonable(float("nan"))

    prompt = "鼓励一下"
    assert prompt_checksum(prompt) == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    source = tmp_path / "prompt.md"
    source.write_bytes(prompt.encode("utf-8"))
    assert file_checksum(source) == prompt_checksum(prompt)
