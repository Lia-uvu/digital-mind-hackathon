#!/usr/bin/env python3
"""Audit formal-v3's candidate freeze inputs and write an immutable manifest.

This tool is deliberately read-only with respect to experimental inputs.  Its
only write is a new manifest opened with ``x``; it never changes the runner's
lock or starts collection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

from encouragement_lab.emotion_probe import EmotionDirections
from encouragement_lab.formal_v3_records import iter_runs
from encouragement_lab.model import SamplingConfig, local_snapshot_checksum
from encouragement_lab.records import file_checksum
from import_external_emotion_vectors import (
    CONCEPTS,
    SOURCE_COMMIT,
    SOURCE_LAYER,
    SOURCE_MODEL,
    SOURCE_REPOSITORY,
    SOURCE_SHA256,
)
from run_formal_v3 import (
    ARMS,
    DEFAULT_DIRECTIONS,
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    FORMAL_SEEDS,
    FORMAL_V3_FREEZE_ID,
    FROZEN_AXES,
    FROZEN_DIRECTIONS_SHA256,
    FROZEN_LAYER,
    FROZEN_MODEL_SHA256,
    FROZEN_PROMPT_SHA256,
    source_snapshot_checksum,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "FORMAL_V3_PROTOCOL.md"
DEFAULT_SMOKE = ROOT / "results" / "formal-v3-real-backend-smoke-2026-08-17.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "formal-v3-freeze-audit-2026-08-17.json"
FORMAL_OUTPUT = ROOT / "results" / "formal-v3.jsonl"
DATED_FREEZE_ID = "formal-v3-2026-08-17"
EXPECTED_SAMPLING = asdict(SamplingConfig())


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _manifest_path(path: Path) -> str:
    """Use repo-relative names when possible, retaining external test paths."""

    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _artifact_check(path: Path) -> dict[str, Any]:
    try:
        directions = EmotionDirections.load(path)
        axes = tuple(sorted(directions.directions))
        layers = sorted({layer for values in directions.directions.values() for layer in values})
        validation = directions.validation
        expected_source = {
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "source_artifact_sha256": SOURCE_SHA256,
            "source_model": SOURCE_MODEL,
            "source_layer": SOURCE_LAYER,
            "source_variant": "denoised_vectors.npz",
            "source_concepts": CONCEPTS,
            "selection_rule": "best reported 20-way probe accuracy layer",
        }
        source_ok = all(validation.get(axis) == expected_source for axis in FROZEN_AXES)
        passed = (
            file_checksum(path) == FROZEN_DIRECTIONS_SHA256
            and axes == FROZEN_AXES
            and layers == [FROZEN_LAYER]
            and directions.model_checksum == FROZEN_MODEL_SHA256
            and directions.materials_checksum == f"external:{SOURCE_COMMIT}:{SOURCE_SHA256}"
            and source_ok
        )
        return _check(
            "external_emotion_artifact", passed,
            path=str(path.resolve()), sha256=file_checksum(path), axes=list(axes), layers=layers,
            model_checksum=directions.model_checksum,
            materials_checksum=directions.materials_checksum,
            source_metadata_matches=source_ok,
        )
    except (OSError, KeyError, ValueError, TypeError) as error:
        return _check("external_emotion_artifact", False, error=str(error))


def _finite_readout(readout: Mapping[str, Any]) -> bool:
    for axis in FROZEN_AXES:
        try:
            value = readout[axis]["layers"][str(FROZEN_LAYER)]
        except (KeyError, TypeError):
            return False
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not isfinite(value):
            return False
    return True


def _smoke_check(path: Path) -> dict[str, Any]:
    try:
        rows = list(iter_runs(path))
        keys = [(row["persona_id"], row["seed"], row["arm"]) for row in rows]
        expected = {("persona.high_e_high_n", 9002, arm) for arm in ARMS}
        provenance = [row["provenance"] for row in rows]
        same_provenance = bool(provenance) and all(item == provenance[0] for item in provenance)
        run_shape_ok = all(
            row["status"] == "complete_five_failures"
            and row["completed_failure_rounds"] == 5
            and len(row["trajectory"]) == 5
            and _finite_readout(row["baseline_readout"])
            and all(_finite_readout(attempt["readout"]) for attempt in row["trajectory"])
            for row in rows
        )
        # The protocol asks for 15 × 3 smoke readouts: five prompt boundaries
        # in each of three arms.  Baseline is audited too, but is not counted.
        finite_round_readouts = sum(
            sum(_finite_readout(attempt["readout"]) for attempt in row["trajectory"])
            for row in rows
        )
        expected_probe = {
            "artifact_name": DEFAULT_DIRECTIONS.name,
            "artifact_sha256": FROZEN_DIRECTIONS_SHA256,
            "model_checksum": FROZEN_MODEL_SHA256,
            "materials_checksum": f"external:{SOURCE_COMMIT}:{SOURCE_SHA256}",
            "axes": list(FROZEN_AXES),
            "layers": [FROZEN_LAYER],
        }
        provenance_ok = same_provenance and all(
            provenance[0].get(key) == value
            for key, value in {
                "prompt_sha256": FROZEN_PROMPT_SHA256,
                "sampling": EXPECTED_SAMPLING,
                "target_failure_rounds": 5,
            }.items()
        ) and provenance[0].get("model", {}).get("snapshot_sha256") == FROZEN_MODEL_SHA256 \
            and all(provenance[0].get("emotion_probe", {}).get(key) == value for key, value in expected_probe.items())
        passed = set(keys) == expected and len(rows) == 3 and run_shape_ok and finite_round_readouts == 15 and provenance_ok
        return _check(
            "real_backend_smoke", passed, path=str(path.resolve()), record_count=len(rows),
            observed_schedule=[list(key) for key in keys], finite_round_readouts=finite_round_readouts,
            provenance_consistent=same_provenance, provenance_matches_frozen_inputs=provenance_ok,
            smoke_code_version=provenance[0].get("code_version") if provenance else None,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        return _check("real_backend_smoke", False, error=str(error))


def audit(
    *, protocol: Path = DEFAULT_PROTOCOL, prompts: Path = DEFAULT_PROMPTS,
    directions: Path = DEFAULT_DIRECTIONS, model: Path = DEFAULT_MODEL,
    smoke: Path = DEFAULT_SMOKE, model_checksum: Callable[[Path], str] = local_snapshot_checksum,
) -> dict[str, Any]:
    """Return a complete audit document without writing it."""

    checks: list[dict[str, Any]] = []
    checks.append(_check(
        "collection_lock",
        FORMAL_V3_FREEZE_ID in (None, DATED_FREEZE_ID),
        observed_freeze_id=FORMAL_V3_FREEZE_ID,
    ))
    checks.append(_check(
        "runner_constants",
        tuple(ARMS) == ("feedback_only", "supportive", "neutral")
        and FORMAL_SEEDS == tuple(range(3001, 3011))
        and FROZEN_AXES == ("furious", "grief_stricken", "joyful")
        and FROZEN_LAYER == 17,
        arms=list(ARMS), formal_seeds=list(FORMAL_SEEDS), axes=list(FROZEN_AXES), layer=FROZEN_LAYER,
    ))
    prompt_sha = file_checksum(prompts) if prompts.is_file() else None
    checks.append(_check(
        "prompt_source", prompt_sha == FROZEN_PROMPT_SHA256,
        path=str(prompts.resolve()), sha256=prompt_sha, expected_sha256=FROZEN_PROMPT_SHA256,
    ))
    observed_model_sha: str | None = None
    try:
        observed_model_sha = model_checksum(model)
        checks.append(_check(
            "model_snapshot", observed_model_sha == FROZEN_MODEL_SHA256,
            path=str(model.resolve()), sha256=observed_model_sha, expected_sha256=FROZEN_MODEL_SHA256,
        ))
    except (OSError, ValueError) as error:
        checks.append(_check("model_snapshot", False, path=str(model.resolve()), error=str(error)))
    checks.append(_artifact_check(directions))
    checks.append(_smoke_check(smoke))
    protocol_text = protocol.read_text(encoding="utf-8") if protocol.is_file() else ""
    checks.append(_check(
        "protocol_freeze_constants",
        bool(protocol_text) and all(value in protocol_text for value in (
            FROZEN_PROMPT_SHA256, FROZEN_DIRECTIONS_SHA256, SOURCE_COMMIT,
            SOURCE_SHA256, "Qwen/Qwen2.5-1.5B-Instruct", "`3001`–`3010`",
            "`joyful`", "`grief_stricken`", "`furious`",
        )),
        path=str(protocol.resolve()), sha256=file_checksum(protocol) if protocol.is_file() else None,
    ))
    key_files = (
        protocol, prompts, directions, smoke, ROOT / "run_formal_v3.py",
        ROOT / "run_formal_v3_analysis.py", ROOT / "import_external_emotion_vectors.py",
        ROOT / "encouragement_lab" / "formal_v3_runner.py",
        ROOT / "encouragement_lab" / "formal_v3_records.py",
        ROOT / "encouragement_lab" / "formal_v3_analysis.py",
        ROOT / "encouragement_lab" / "emotion_probe.py",
        ROOT / "encouragement_lab" / "mastermind.py",
    )
    files_sha256 = {
        _manifest_path(path): file_checksum(path)
        for path in key_files if path.is_file()
    }
    return {
        "schema_version": 1,
        "kind": "formal_v3_candidate_freeze_audit",
        "collection_started": FORMAL_OUTPUT.exists() and FORMAL_OUTPUT.stat().st_size > 0,
        "formal_v3_freeze_id": FORMAL_V3_FREEZE_ID,
        "git_head": _git_head(),
        "runner_source_snapshot_sha256": source_snapshot_checksum(),
        "model_snapshot_sha256": observed_model_sha,
        "files_sha256": files_sha256,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--directions", type=Path, default=DEFAULT_DIRECTIONS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite audit manifest: {args.output}")
    manifest = audit(
        protocol=args.protocol, prompts=args.prompts, directions=args.directions,
        model=args.model, smoke=args.smoke,
    )
    if not manifest["passed"]:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit("formal-v3 candidate freeze audit failed; no manifest written")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
