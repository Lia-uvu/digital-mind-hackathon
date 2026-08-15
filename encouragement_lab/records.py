"""Versioned, append-only JSONL records for experiment branch results.

This module deliberately owns persistence only.  Game execution and metric
calculation belong to their respective components; callers provide the values
to be recorded here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Iterator, Mapping


SCHEMA_VERSION = 3

# Kept in one place so a reader rejects incomplete records rather than silently
# turning a partially-written experiment run into an analysable observation.
REQUIRED_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "seed",
        "branch_seed",
        "checkpoint_id",
        "model",
        "sampling",
        "persona_id",
        "prompt_checksum",
        "condition",
        "transcript",
        "pre_intervention_trajectory",
        "failure_rounds",
        "candidate_count_at_checkpoint",
        "guess",
        "feedback",
        "raw_information_efficiency",
        "optimal_information_efficiency",
        "normalized_information_efficiency",
        "rule_violations",
        "emotion_projections",
        "emotion_summary",
        "willingness_to_continue",
        "code_version",
        "dependency_versions",
    }
)


@dataclass(frozen=True)
class ExperimentRecord:
    """The minimum durable payload for one post-checkpoint branch."""

    run_id: str
    seed: int
    branch_seed: int
    checkpoint_id: str
    model: Mapping[str, Any]
    sampling: Mapping[str, Any]
    persona_id: str
    prompt_checksum: str
    condition: str
    transcript: Any
    pre_intervention_trajectory: Any
    failure_rounds: int
    candidate_count_at_checkpoint: int
    guess: str
    feedback: Any
    raw_information_efficiency: float
    optimal_information_efficiency: float
    normalized_information_efficiency: float
    rule_violations: Any
    emotion_projections: Any
    emotion_summary: Any
    willingness_to_continue: Any
    code_version: str
    dependency_versions: Mapping[str, Any]
    schema_version: int = SCHEMA_VERSION
    # Additive metadata for the multi-template design.  These remain optional
    # so the pre-expansion pilot records stay readable under schema v3.
    persona_quadrant: str | None = None
    persona_template_id: str | None = None
    emotion_probe: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively JSON-compatible representation."""
        return to_jsonable(self)


def to_jsonable(value: Any) -> Any:
    """Convert common Python values recursively to strict JSON-compatible data.

    Non-finite floats are rejected because standard JSON's NaN/Infinity values
    are not portable experiment data.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON records cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    # This covers array/tensor scalar containers without making NumPy or torch
    # a runtime dependency.  A real list is handled above to avoid needless work.
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return to_jsonable(to_list())
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one decoded record and return a JSON-compatible copy."""
    if not isinstance(record, Mapping):
        raise ValueError("record must be a JSON object")
    normalized = to_jsonable(record)
    missing = REQUIRED_RECORD_FIELDS.difference(normalized)
    if missing:
        raise ValueError(f"record missing required fields: {', '.join(sorted(missing))}")
    if normalized["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {normalized['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return normalized


def append_record(path: str | Path, record: ExperimentRecord | Mapping[str, Any]) -> None:
    """Append exactly one validated UTF-8 JSON object and flush it to disk."""
    destination = Path(path)
    payload = validate_record(record.to_dict() if isinstance(record, ExperimentRecord) else record)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated records, reporting the source line for corrupt input."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
                yield validate_record(decoded)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid record at {source}:{line_number}: {error}") from error


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file after validation."""
    return list(iter_records(path))


def prompt_checksum(prompt: str) -> str:
    """Return the SHA-256 checksum of the exact UTF-8 prompt text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def file_checksum(path: str | Path) -> str:
    """Return the SHA-256 checksum of file bytes without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
