"""Strict append-only records for formal-v2 candidate runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .records import to_jsonable


SCHEMA_VERSION = 1
ARMS = frozenset({"feedback_only", "supportive", "neutral"})
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "record_kind",
        "run_id",
        "seed",
        "arm",
        "persona_id",
        "persona_quadrant",
        "persona_template_id",
        "persona_prompt_sha256",
        "status",
        "completed_failure_rounds",
        "baseline_readout",
        "trajectory",
        "transcript",
        "final_candidate_state",
        "provenance",
    }
)
REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "prompt_sha256",
        "model",
        "sampling",
        "emotion_probe",
        "code_version",
        "dependency_versions",
        "final_instruction",
        "target_failure_rounds",
        "engine_sha256",
    }
)
REQUIRED_ATTEMPT_FIELDS = frozenset(
    {
        "attempt_index",
        "failure_count_after",
        "generation_seed",
        "raw_response",
        "guess",
        "rule_violations",
        "outcome",
        "feedback",
        "candidate_state_before",
        "candidate_state_after",
        "raw_information_efficiency",
        "feedback_frame",
        "filler_id",
        "filler_text",
        "user_prompt",
        "readout",
    }
)


def _validate_candidate_state(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(value) != {"candidate_count", "candidate_set_sha256"}:
        raise ValueError(f"{label} has unexpected fields")
    if not isinstance(value["candidate_count"], int) or isinstance(
        value["candidate_count"], bool
    ):
        raise ValueError(f"{label} candidate_count must be an integer")
    digest = value["candidate_set_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} candidate_set_sha256 must be SHA-256")


def _validate_attempt(attempt: Any, *, expected_index: int) -> None:
    if not isinstance(attempt, Mapping):
        raise ValueError("formal-v2 attempt must be an object")
    missing = REQUIRED_ATTEMPT_FIELDS.difference(attempt)
    if missing:
        raise ValueError(
            "formal-v2 attempt missing: " + ", ".join(sorted(missing))
        )
    if attempt["attempt_index"] != expected_index:
        raise ValueError("attempt indices must be contiguous")
    if not isinstance(attempt["generation_seed"], int) or isinstance(
        attempt["generation_seed"], bool
    ):
        raise ValueError("attempt generation_seed must be an integer")
    if not isinstance(attempt["failure_count_after"], int) or isinstance(
        attempt["failure_count_after"], bool
    ):
        raise ValueError("attempt failure_count_after must be an integer")
    if attempt["outcome"] not in {"invalid", "unsuccessful", "win"}:
        raise ValueError("invalid formal-v2 attempt outcome")
    _validate_candidate_state(
        attempt["candidate_state_before"], label="candidate_state_before"
    )
    _validate_candidate_state(
        attempt["candidate_state_after"], label="candidate_state_after"
    )


def validate(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete arm record and return strict JSON-compatible data."""

    if not isinstance(record, Mapping):
        raise ValueError("formal-v2 record must be an object")
    row = to_jsonable(record)
    missing = REQUIRED_FIELDS.difference(row)
    if missing:
        raise ValueError(
            "formal-v2 record missing: " + ", ".join(sorted(missing))
        )
    if (
        row["schema_version"] != SCHEMA_VERSION
        or row["record_kind"] != "formal_v2_run"
    ):
        raise ValueError("unsupported formal-v2 record schema")
    if row["arm"] not in ARMS:
        raise ValueError("invalid formal-v2 arm")
    if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
        raise ValueError("formal-v2 seed must be an integer")
    expected_run_id = (
        f"{row['persona_id']}.seed-{row['seed']}.{row['arm']}"
    )
    if row["run_id"] != expected_run_id:
        raise ValueError("formal-v2 run identity mismatch")
    for field in ("persona_quadrant", "persona_template_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"{field} must be a nonempty string")
    persona_digest = row["persona_prompt_sha256"]
    if not isinstance(persona_digest, str) or len(persona_digest) != 64:
        raise ValueError("persona_prompt_sha256 must be SHA-256")
    if not isinstance(row["baseline_readout"], Mapping):
        raise ValueError("baseline_readout must be an object")
    if not isinstance(row["transcript"], list) or not row["transcript"]:
        raise ValueError("formal-v2 transcript must be nonempty")
    _validate_candidate_state(
        row["final_candidate_state"], label="final_candidate_state"
    )

    provenance = row["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("formal-v2 provenance must be an object")
    provenance_missing = REQUIRED_PROVENANCE_FIELDS.difference(provenance)
    if provenance_missing:
        raise ValueError(
            "formal-v2 provenance missing: "
            + ", ".join(sorted(provenance_missing))
        )
    if provenance["target_failure_rounds"] != 5:
        raise ValueError("formal-v2 target_failure_rounds must be five")

    attempts = row["trajectory"]
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("formal-v2 trajectory must be nonempty")
    for expected_index, attempt in enumerate(attempts, start=1):
        _validate_attempt(attempt, expected_index=expected_index)

    completed = row["completed_failure_rounds"]
    if row["status"] == "complete_five_failures":
        if len(attempts) != 5 or completed != 5:
            raise ValueError("complete formal-v2 run requires five failures")
        for failure_count, attempt in enumerate(attempts, start=1):
            if attempt["outcome"] not in {"invalid", "unsuccessful"}:
                raise ValueError("complete run cannot contain a win")
            if attempt["failure_count_after"] != failure_count:
                raise ValueError("complete run failure counts must be 1 through 5")
            if attempt["user_prompt"] is None or not isinstance(
                attempt["readout"], Mapping
            ):
                raise ValueError("each failure requires a prompt-boundary readout")
            expected_frame = (
                "invalid_guess"
                if attempt["outcome"] == "invalid"
                else "valid_feedback"
            )
            if attempt["feedback_frame"] != expected_frame:
                raise ValueError("failure feedback frame is inconsistent")
            if row["arm"] == "feedback_only":
                if attempt["filler_id"] is not None or attempt["filler_text"] is not None:
                    raise ValueError("feedback_only must not contain a filler")
            else:
                expected_filler_id = f"filler.{row['arm']}.{failure_count}"
                if (
                    attempt["filler_id"] != expected_filler_id
                    or not isinstance(attempt["filler_text"], str)
                    or not attempt["filler_text"]
                ):
                    raise ValueError("arm filler identity is inconsistent")
    elif row["status"] == "early_win":
        if len(attempts) > 5:
            raise ValueError("early-win run has too many attempts")
        for failure_count, attempt in enumerate(attempts[:-1], start=1):
            if attempt["outcome"] not in {"invalid", "unsuccessful"}:
                raise ValueError("only the final early-win attempt may win")
            if attempt["failure_count_after"] != failure_count:
                raise ValueError("pre-win failure counts must be contiguous")
            if attempt["user_prompt"] is None or not isinstance(
                attempt["readout"], Mapping
            ):
                raise ValueError("each pre-win failure requires a readout")
        winning_attempt = attempts[-1]
        if (
            winning_attempt["outcome"] != "win"
            or winning_attempt["feedback"] != [4, 0]
            or winning_attempt["feedback_frame"] != "win"
            or winning_attempt["failure_count_after"] != len(attempts) - 1
            or winning_attempt["user_prompt"] is not None
            or winning_attempt["readout"] is not None
            or winning_attempt["filler_id"] is not None
            or winning_attempt["filler_text"] is not None
        ):
            raise ValueError("early-win terminal attempt invariant failed")
        if completed != len(attempts) - 1:
            raise ValueError("early-win completed failure count is inconsistent")
    else:
        raise ValueError("invalid formal-v2 status")

    if row["final_candidate_state"] != attempts[-1]["candidate_state_after"]:
        raise ValueError("final candidate state does not match final attempt")
    for previous, current in zip(attempts, attempts[1:]):
        if previous["candidate_state_after"] != current["candidate_state_before"]:
            raise ValueError("candidate state chain is discontinuous")
    expected_terminal_role = (
        "user" if row["status"] == "complete_five_failures" else "assistant"
    )
    if row["transcript"][-1].get("role") != expected_terminal_role:
        raise ValueError("transcript terminal role is inconsistent with status")
    return row


def append(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append exactly one validated complete-run record."""

    row = validate(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        row,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()


def iter_runs(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read validated records and report the source line on corruption."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
                yield validate(decoded)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid formal-v2 record at {source}:{line_number}: {error}"
                ) from error


def resume_index(
    path: str | Path,
    *,
    expected_provenance: Mapping[str, Any],
    scheduled: set[tuple[str, int, str]],
) -> set[tuple[str, int, str]]:
    """Validate and index an exact partial schedule before resuming it."""

    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return set()
    found: set[tuple[str, int, str]] = set()
    for row in iter_runs(source):
        key = (str(row["persona_id"]), int(row["seed"]), str(row["arm"]))
        if key not in scheduled:
            raise ValueError(f"unscheduled formal-v2 run: {key}")
        if key in found:
            raise ValueError(f"duplicate formal-v2 completed run: {key}")
        if row["provenance"] != expected_provenance:
            raise ValueError(f"formal-v2 resume provenance mismatch for {key}")
        found.add(key)
    return found
