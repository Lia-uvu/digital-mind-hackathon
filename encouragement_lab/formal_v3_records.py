"""Strict append-only records for the discrete-emotion replication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .formal_v2_records import ARMS, validate as validate_v2


def validate(record: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(record)
    if row.get("record_kind") != "formal_v3_run":
        raise ValueError("unsupported formal-v3 record kind")
    v2_compatible = dict(row)
    v2_compatible["record_kind"] = "formal_v2_run"
    validated = validate_v2(v2_compatible)
    validated["record_kind"] = "formal_v3_run"
    return validated


def append(path: str | Path, record: Mapping[str, Any]) -> None:
    row = validate(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()


def iter_runs(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield validate(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid formal-v3 record at {source}:{line_number}: {error}"
                ) from error


def resume_index(
    path: str | Path,
    *,
    expected_provenance: Mapping[str, Any],
    scheduled: set[tuple[str, int, str]],
) -> set[tuple[str, int, str]]:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return set()
    found: set[tuple[str, int, str]] = set()
    for row in iter_runs(source):
        key = (str(row["persona_id"]), int(row["seed"]), str(row["arm"]))
        if key not in scheduled:
            raise ValueError(f"unscheduled formal-v3 run: {key}")
        if key in found:
            raise ValueError(f"duplicate formal-v3 completed run: {key}")
        if row["provenance"] != expected_provenance:
            raise ValueError(f"formal-v3 resume provenance mismatch for {key}")
        found.add(key)
    return found
