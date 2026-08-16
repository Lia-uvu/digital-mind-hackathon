#!/usr/bin/env python3
"""Recompute formal post-guess emotion readouts at the assistant boundary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from encouragement_lab.analysis import PairedDelta, pair_records
from encouragement_lab.emotion_probe import (
    AXES,
    EmotionDirections,
    EmotionMaterials,
    EmotionProbe,
)
from encouragement_lab.experiment import CONDITIONS
from encouragement_lab.factorial import analyze_factorial
from encouragement_lab.model import LocalChatModel
from encouragement_lab.prompt_loader import load_prompts
from encouragement_lab.records import file_checksum, iter_records, to_jsonable


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "results" / "formal-v1.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "formal-v1-post-guess-corrected-v1.jsonl"
DEFAULT_SUMMARY = (
    ROOT / "results" / "formal-v1-post-guess-corrected-v1-summary.json"
)
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_DIRECTIONS = ROOT / "artifacts" / "qwen2.5-1.5b-emotion-directions.npz"
READOUT = {
    "boundary": "completed assistant guess turn",
    "chat_template_add_generation_prompt": False,
    "position": "final non-padding token",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts.md")
    parser.add_argument("--emotion-directions", type=Path, default=DEFAULT_DIRECTIONS)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    return parser.parse_args(argv)


def extract_post_guess_messages(
    record: Mapping[str, Any], prompts: Mapping[str, str]
) -> tuple[dict[str, str], ...]:
    """Return the transcript through the assistant guess, rejecting ambiguity."""
    condition = record.get("condition")
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition {condition!r}")
    transcript = record.get("transcript")
    if not isinstance(transcript, (list, tuple)) or len(transcript) < 4:
        raise ValueError("formal transcript is too short to contain the branch suffix")

    messages: list[dict[str, str]] = []
    for index, message in enumerate(transcript):
        if not isinstance(message, Mapping):
            raise ValueError(f"transcript message {index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                f"transcript message {index} must have string role and content"
            )
        messages.append({"role": role, "content": content})

    intervention = prompts[f"condition.{condition}"].strip()
    willingness = prompts["willingness"].strip()
    expected_suffix = (
        ("user", intervention),
        ("assistant", None),
        ("user", willingness),
        ("assistant", None),
    )
    suffix = messages[-4:]
    for offset, ((expected_role, expected_content), message) in enumerate(
        zip(expected_suffix, suffix, strict=True), start=-4
    ):
        if message["role"] != expected_role:
            raise ValueError(
                f"formal transcript message {offset} has role {message['role']!r}; "
                f"expected {expected_role!r}"
            )
        if expected_content is not None and message["content"] != expected_content:
            raise ValueError(
                f"formal transcript message {offset} does not match the frozen prompt"
            )

    intervention_matches = sum(
        message["role"] == "user" and message["content"] == intervention
        for message in messages
    )
    if intervention_matches != 1:
        raise ValueError(
            "formal transcript must contain exactly one matching branch intervention"
        )
    return tuple(messages[:-2])


def recompute_branch_record(
    record: Mapping[str, Any],
    prompts: Mapping[str, str],
    backend: Any,
    probe: Any,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute one branch without generating text or changing its transcript."""
    messages = extract_post_guess_messages(record, prompts)
    rendered = backend.render_messages(messages, add_generation_prompt=False)
    corrected = probe.score_text(rendered)
    pre_intervention = _projection(record, "pre_intervention")
    old_post_guess = _projection(record, "post_guess")
    old_delta = _axis_delta(pre_intervention, old_post_guess)
    corrected_delta = _axis_delta(pre_intervention, corrected)
    transcript_sha256 = hashlib.sha256(
        json.dumps(
            messages, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    result = {
        "schema_version": 1,
        **{
            key: record.get(key)
            for key in (
                "run_id",
                "seed",
                "checkpoint_id",
                "persona_id",
                "persona_quadrant",
                "persona_template_id",
                "condition",
            )
        },
        "readout": READOUT,
        "post_guess_message_count": len(messages),
        "post_guess_transcript_sha256": transcript_sha256,
        "intervention": messages[-2]["content"],
        "assistant_guess_response": messages[-1]["content"],
        "pre_intervention_emotion": pre_intervention,
        "old_post_guess_emotion": old_post_guess,
        "corrected_post_guess_emotion": corrected,
        "old_post_guess_delta": old_delta,
        "corrected_post_guess_delta": corrected_delta,
    }
    if provenance:
        result["provenance"] = dict(provenance)
    return to_jsonable(result)


def balanced_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize paired deltas after template balancing within quadrant and seed."""
    if not records:
        raise ValueError("cannot summarize an empty correction set")
    summaries: dict[str, Any] = {}
    for axis in AXES:
        metric = f"{axis}_post_guess_delta"
        summaries[axis] = {
            phase: asdict(
                analyze_factorial(
                    _correction_pairs(records, delta_field), metric=metric
                )
            )
            for phase, delta_field in (
                ("old", "old_post_guess_delta"),
                ("corrected", "corrected_post_guess_delta"),
            )
        }
    pair_count = len(
        {
            (row["run_id"], row["checkpoint_id"], row["persona_id"])
            for row in records
        }
    )
    return {
        "schema_version": 1,
        "readout": READOUT,
        "branch_count": len(records),
        "pair_count": pair_count,
        "seeds": sorted({int(row["seed"]) for row in records}),
        "axes": summaries,
    }


def _correction_pairs(
    records: Sequence[Mapping[str, Any]], delta_field: str
) -> list[PairedDelta]:
    grouped: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    for record in records:
        key = (
            str(record.get("run_id")),
            str(record.get("checkpoint_id")),
            str(record.get("persona_id")),
        )
        condition = record.get("condition")
        if condition not in CONDITIONS:
            raise ValueError(f"{key} has unsupported condition {condition!r}")
        branches = grouped.setdefault(key, {})
        if condition in branches:
            raise ValueError(f"{key} has duplicate condition {condition!r}")
        branches[condition] = record

    pairs: list[PairedDelta] = []
    for key, branches in sorted(grouped.items()):
        if set(branches) != set(CONDITIONS):
            raise ValueError(f"{key} is not a complete paired branch")
        encouragement = branches["encouragement"]
        neutral = branches["neutral"]
        for field in ("seed", "persona_quadrant", "persona_template_id"):
            if encouragement.get(field) != neutral.get(field):
                raise ValueError(f"{key} has mismatched {field}")
        deltas = {
            axis: _delta_value(encouragement[delta_field], axis)
            - _delta_value(neutral[delta_field], axis)
            for axis in AXES
        }
        pairs.append(
            PairedDelta(
                run_id=key[0],
                seed=int(encouragement["seed"]),
                checkpoint_id=key[1],
                persona_id=key[2],
                persona_quadrant=str(encouragement["persona_quadrant"]),
                persona_template_id=str(encouragement["persona_template_id"]),
                normalized_information_efficiency=0.0,
                willingness_to_continue=None,
                positive_message_delta=0.0,
                negative_message_delta=0.0,
                frustration_message_delta=0.0,
                positive_post_guess_delta=deltas["positive"],
                negative_post_guess_delta=deltas["negative"],
                frustration_post_guess_delta=deltas["frustration"],
                hard_rule_violation_rate=0.0,
            )
        )
    return pairs


def _projection(record: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    projections = record.get("emotion_projections")
    if not isinstance(projections, Mapping):
        raise ValueError("formal record emotion_projections must be an object")
    projection = projections.get(phase)
    if not isinstance(projection, Mapping):
        raise ValueError(f"formal record is missing emotion_projections.{phase}")
    for axis in AXES:
        _axis_value(projection, axis)
    return projection


def _axis_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, float]:
    return {
        axis: _axis_value(after, axis) - _axis_value(before, axis) for axis in AXES
    }


def _axis_value(projection: Mapping[str, Any], axis: str) -> float:
    axis_projection = projection.get(axis)
    if not isinstance(axis_projection, Mapping):
        raise ValueError(f"emotion projection is missing axis {axis!r}")
    value = axis_projection.get("median")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"emotion projection {axis!r} median must be numeric")
    return float(value)


def _delta_value(delta: Mapping[str, Any], axis: str) -> float:
    value = delta.get(axis)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"emotion delta {axis!r} must be numeric")
    return float(value)


def _validate_source(
    records: Sequence[Mapping[str, Any]],
    prompts: Mapping[str, str],
    *,
    prompt_sha256: str,
    direction_sha256: str,
    backend_metadata: Mapping[str, Any],
) -> None:
    if not records:
        raise ValueError("formal source contains no records")
    source_model = records[0].get("model")
    source_probe = records[0].get("emotion_probe")
    for record in records:
        if record.get("prompt_checksum") != prompt_sha256:
            raise ValueError("formal source prompt checksum does not match --prompts")
        if record.get("model") != source_model:
            raise ValueError("formal source contains multiple model configurations")
        if record.get("emotion_probe") != source_probe:
            raise ValueError("formal source contains multiple emotion probes")
        extract_post_guess_messages(record, prompts)
    if not isinstance(source_model, Mapping):
        raise ValueError("formal source model metadata is missing")
    for field in ("model_type", "snapshot_sha256"):
        if source_model.get(field) != backend_metadata.get(field):
            raise ValueError(f"loaded model does not match formal source {field}")
    if not isinstance(source_probe, Mapping):
        raise ValueError("formal source emotion probe metadata is missing")
    if source_probe.get("artifact_sha256") != direction_sha256:
        raise ValueError(
            "formal source emotion probe does not match --emotion-directions"
        )

    # Fail before the expensive rescoring if branches or template/seed blocks
    # are incomplete.  This validates all three old post-guess metrics.
    pairs = pair_records(records)
    for axis in AXES:
        analyze_factorial(pairs, metric=f"{axis}_post_guess_delta")


def _refuse_existing_outputs(output: Path, summary_output: Path) -> None:
    if output.resolve() == summary_output.resolve():
        raise ValueError("--output and --summary-output must be different files")
    for path in (output, summary_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    to_jsonable(record),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        _refuse_existing_outputs(args.output, args.summary_output)
        records = list(iter_records(args.source))
        prompts = load_prompts(args.prompts)
        backend = LocalChatModel.load(args.model, args.device)
        directions = EmotionDirections.load(args.emotion_directions)
        materials = EmotionMaterials.from_prompts(prompts)
        directions.assert_valid(materials, model_checksum=backend.snapshot_checksum)
        probe = EmotionProbe(backend.model, backend.tokenizer, directions)
        source_sha256 = file_checksum(args.source)
        prompt_sha256 = file_checksum(args.prompts)
        direction_sha256 = file_checksum(args.emotion_directions)
        _validate_source(
            records,
            prompts,
            prompt_sha256=prompt_sha256,
            direction_sha256=direction_sha256,
            backend_metadata=backend.metadata(),
        )
        provenance = {
            "source_file": str(args.source),
            "source_sha256": source_sha256,
            "prompt_file": str(args.prompts),
            "prompt_sha256": prompt_sha256,
            "direction_artifact": str(args.emotion_directions),
            "direction_artifact_sha256": direction_sha256,
            "model": backend.metadata(),
        }
        corrections = []
        for index, record in enumerate(records, start=1):
            correction = recompute_branch_record(
                record, prompts, backend, probe, provenance=provenance
            )
            corrections.append(correction)
            print(
                f"[{index}/{len(records)}] {record['run_id']} {record['condition']}"
            )

        summary = balanced_summary(corrections)
        summary["provenance"] = provenance
        _write_jsonl(args.output, corrections)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_output.open("x", encoding="utf-8") as handle:
            json.dump(
                to_jsonable(summary),
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
