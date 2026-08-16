#!/usr/bin/env python3
"""Analyze one or more response-pilot JSONL files as paired seed blocks.

The response pilot is exploratory.  Prompt-end projections and willingness are
paired encouragement-minus-neutral outcomes.  Generated-token trajectories are
reported only as coverage, length, and token-alignment diagnostics; this script
does not select or test any token position as an outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

from encouragement_lab.experiment import CONDITIONS, parse_willingness
from encouragement_lab.personas import (
    PERSONA_BY_ID,
    PERSONA_QUADRANTS,
    expected_template_ids,
)


AXES = ("positive", "negative", "frustration")
METRICS = ("willingness", *(f"prompt_end_{axis}" for axis in AXES))
PLANNED_CONTRASTS = ("extraversion", "neuroticism", "interaction")
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_id",
        "run_id",
        "seed",
        "persona_id",
        "persona_quadrant",
        "persona_template_id",
        "condition",
        "generation_seed",
        "sampling",
        "model",
        "source_file",
        "source_sha256",
        "intervention_prompt_sha256",
        "direction_artifact_sha256",
        "intervention",
        "response",
        "generated_token_count",
        "pre_intervention_emotion",
        "prompt_end_emotion",
        "token_trajectory",
        "frustration_summary",
    }
)
SHARED_PAIR_FIELDS = (
    "schema_version",
    "checkpoint_id",
    "run_id",
    "seed",
    "persona_id",
    "persona_quadrant",
    "persona_template_id",
    "generation_seed",
    "sampling",
    "model",
    "source_file",
    "source_sha256",
    "intervention_prompt_sha256",
    "direction_artifact_sha256",
    "pre_intervention_emotion",
)
SHARED_STUDY_FIELDS = (
    "schema_version",
    "sampling",
    "model",
    "source_sha256",
    "intervention_prompt_sha256",
    "direction_artifact_sha256",
)


@dataclass(frozen=True)
class ResponsePair:
    checkpoint_id: str
    run_id: str
    seed: int
    persona_id: str
    persona_quadrant: str
    persona_template_id: str
    encouragement_willingness: int | None
    neutral_willingness: int | None
    willingness: float | None
    prompt_end_positive: float
    prompt_end_negative: float
    prompt_end_frustration: float
    encouragement_token_count: int
    neutral_token_count: int
    encouragement_tokens: tuple[str, ...]
    neutral_tokens: tuple[str, ...]


@dataclass(frozen=True)
class SeedContrasts:
    seed: int
    average_treatment: float
    extraversion: float
    neuroticism: float
    interaction: float


@dataclass(frozen=True)
class ContrastSummary:
    count: int
    mean: float | None
    sample_stdev: float | None
    standard_error: float | None
    exact_sign_flip_p: float | None
    holm_adjusted_p: float | None = None


def read_jsonl(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read response-pilot records and attach no file-specific semantics."""
    if not paths:
        raise ValueError("at least one response-pilot JSONL file is required")
    records: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
                try:
                    records.append(_validate_record(record))
                except ValueError as error:
                    raise ValueError(f"invalid record at {path}:{line_number}: {error}") from error
    if not records:
        raise ValueError("response-pilot input contains no records")
    return records


def pair_records(records: Iterable[Mapping[str, Any]]) -> list[ResponsePair]:
    """Require exactly one encouragement and one neutral row per checkpoint."""
    validated_records = [_validate_record(record) for record in records]
    if not validated_records:
        raise ValueError("response-pilot input contains no records")
    _validate_shared_study_metadata(validated_records)

    grouped: dict[tuple[str, str, int, str], dict[str, Mapping[str, Any]]] = {}
    for record in validated_records:
        key = (
            record["checkpoint_id"],
            record["run_id"],
            record["seed"],
            record["persona_id"],
        )
        branches = grouped.setdefault(key, {})
        condition = record["condition"]
        if condition in branches:
            raise ValueError(f"{_format_key(key)} has duplicate {condition!r} branch")
        branches[condition] = record

    pairs: list[ResponsePair] = []
    structural_keys: set[tuple[int, str, str]] = set()
    for key in sorted(grouped):
        branches = grouped[key]
        if len(branches) != 2 or set(branches) != set(CONDITIONS):
            missing = sorted(set(CONDITIONS).difference(branches))
            extra = sorted(set(branches).difference(CONDITIONS))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("unexpected " + ", ".join(extra))
            raise ValueError(
                f"{_format_key(key)} must have exactly one branch per condition: "
                + "; ".join(details)
            )
        encouragement = branches["encouragement"]
        neutral = branches["neutral"]
        _validate_shared_pair(key, encouragement, neutral)

        structural_key = (
            encouragement["seed"],
            encouragement["persona_quadrant"],
            encouragement["persona_template_id"],
        )
        if structural_key in structural_keys:
            raise ValueError(
                "duplicate persona template within a seed block: "
                f"seed={structural_key[0]}, quadrant={structural_key[1]!r}, "
                f"template={structural_key[2]!r}"
            )
        structural_keys.add(structural_key)

        encouragement_willingness = parse_willingness(encouragement["response"])
        neutral_willingness = parse_willingness(neutral["response"])
        willingness = (
            float(encouragement_willingness - neutral_willingness)
            if encouragement_willingness is not None
            and neutral_willingness is not None
            else None
        )
        pairs.append(
            ResponsePair(
                checkpoint_id=encouragement["checkpoint_id"],
                run_id=encouragement["run_id"],
                seed=encouragement["seed"],
                persona_id=encouragement["persona_id"],
                persona_quadrant=encouragement["persona_quadrant"],
                persona_template_id=encouragement["persona_template_id"],
                encouragement_willingness=encouragement_willingness,
                neutral_willingness=neutral_willingness,
                willingness=willingness,
                **{
                    f"prompt_end_{axis}": _prompt_end_median(encouragement, axis)
                    - _prompt_end_median(neutral, axis)
                    for axis in AXES
                },
                encouragement_token_count=encouragement["generated_token_count"],
                neutral_token_count=neutral["generated_token_count"],
                encouragement_tokens=tuple(
                    token["token_text"] for token in encouragement["token_trajectory"]
                ),
                neutral_tokens=tuple(
                    token["token_text"] for token in neutral["token_trajectory"]
                ),
            )
        )
    _validate_complete_template_blocks(pairs)
    return pairs


def analyze_paths(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Return the complete JSON-compatible response-pilot analysis."""
    records = read_jsonl(paths)
    pairs = pair_records(records)
    seeds = sorted({pair.seed for pair in pairs})
    analyses = {metric: _analyze_metric(pairs, metric) for metric in METRICS}
    condition_counts = Counter(record["condition"] for record in records)
    parsed_records = sum(parse_willingness(record["response"]) is not None for record in records)
    valid_willingness_pairs = sum(pair.willingness is not None for pair in pairs)
    return {
        "analysis_schema_version": 1,
        "input_files": [str(Path(path)) for path in paths],
        "study_metadata": {
            field: records[0][field] for field in SHARED_STUDY_FIELDS
        },
        "completeness": {
            "record_count": len(records),
            "condition_record_counts": {
                condition: condition_counts[condition] for condition in CONDITIONS
            },
            "checkpoint_pair_count": len(pairs),
            "seed_count": len(seeds),
            "seeds": seeds,
            "persona_template_count": len({pair.persona_id for pair in pairs}),
            "quadrant_seed_block_count": len(
                {(pair.seed, pair.persona_quadrant) for pair in pairs}
            ),
            "parsed_willingness_record_count": parsed_records,
            "unparsed_willingness_record_count": len(records) - parsed_records,
            "valid_willingness_pair_count": valid_willingness_pairs,
            "invalid_willingness_pair_count": len(pairs) - valid_willingness_pairs,
        },
        "template_pairs": [
            {
                "checkpoint_id": pair.checkpoint_id,
                "run_id": pair.run_id,
                "seed": pair.seed,
                "persona_id": pair.persona_id,
                "persona_quadrant": pair.persona_quadrant,
                "persona_template_id": pair.persona_template_id,
                "willingness": {
                    "encouragement": pair.encouragement_willingness,
                    "neutral": pair.neutral_willingness,
                    "encouragement_minus_neutral": pair.willingness,
                },
                "prompt_end_encouragement_minus_neutral": {
                    axis: getattr(pair, f"prompt_end_{axis}") for axis in AXES
                },
                "generated_token_count": {
                    "encouragement": pair.encouragement_token_count,
                    "neutral": pair.neutral_token_count,
                    "matched": pair.encouragement_token_count
                    == pair.neutral_token_count,
                },
            }
            for pair in pairs
        ],
        "output_token_diagnostics": _token_diagnostics(pairs),
        "metrics": analyses,
    }


def _validate_record(raw_record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_record, Mapping):
        raise ValueError("record must be a JSON object")
    record = dict(raw_record)
    missing = REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError("missing required fields: " + ", ".join(sorted(missing)))
    if record["schema_version"] != 1:
        raise ValueError(f"unsupported schema_version {record['schema_version']!r}")
    for field in ("checkpoint_id", "run_id", "persona_id"):
        _nonempty_string(record[field], field)
    seed = record["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    generation_seed = record["generation_seed"]
    if isinstance(generation_seed, bool) or not isinstance(generation_seed, int):
        raise ValueError("generation_seed must be an integer")
    if record["condition"] not in CONDITIONS:
        raise ValueError(f"unsupported condition {record['condition']!r}")
    for field in (
        "source_file",
        "source_sha256",
        "intervention_prompt_sha256",
        "direction_artifact_sha256",
        "intervention",
    ):
        _nonempty_string(record[field], field)
    for field in ("sampling", "model", "pre_intervention_emotion", "frustration_summary"):
        if not isinstance(record[field], Mapping):
            raise ValueError(f"{field} must be an object")

    persona_id = record["persona_id"]
    spec = PERSONA_BY_ID.get(persona_id)
    if spec is None:
        raise ValueError(f"unknown persona_id {persona_id!r}")
    if record["persona_quadrant"] != spec.quadrant_id:
        raise ValueError(f"persona_quadrant does not match {persona_id!r}")
    if record["persona_template_id"] != spec.template_id:
        raise ValueError(f"persona_template_id does not match {persona_id!r}")

    if not isinstance(record["response"], str):
        raise ValueError("response must be a string")
    count = record["generated_token_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("generated_token_count must be a non-negative integer")
    trajectory = record["token_trajectory"]
    if not isinstance(trajectory, list):
        raise ValueError("token_trajectory must be a list")
    if len(trajectory) != count:
        raise ValueError("generated_token_count does not match token_trajectory length")
    token_positions: list[int] = []
    for offset, token in enumerate(trajectory):
        if not isinstance(token, Mapping):
            raise ValueError(f"token_trajectory[{offset}] must be an object")
        _nonempty_string(token.get("token_text"), f"token_trajectory[{offset}].token_text")
        token_position = token.get("token_position")
        token_id = token.get("token_id")
        if isinstance(token_position, bool) or not isinstance(token_position, int):
            raise ValueError(f"token_trajectory[{offset}].token_position must be an integer")
        token_positions.append(token_position)
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"token_trajectory[{offset}].token_id must be an integer")
        projections = token.get("projections")
        if not isinstance(projections, Mapping):
            raise ValueError(f"token_trajectory[{offset}].projections must be an object")
        for axis in AXES:
            _projection_median(projections, axis, f"token_trajectory[{offset}].projections")
    if token_positions and token_positions != list(
        range(token_positions[0], token_positions[0] + len(token_positions))
    ):
        raise ValueError("token_trajectory token positions must be consecutive")
    for axis in AXES:
        _prompt_end_median(record, axis)
    return record


def _validate_shared_pair(
    key: tuple[str, str, int, str],
    encouragement: Mapping[str, Any],
    neutral: Mapping[str, Any],
) -> None:
    for field in SHARED_PAIR_FIELDS:
        if encouragement[field] != neutral[field]:
            raise ValueError(f"{_format_key(key)} has mismatched shared field {field!r}")


def _validate_shared_study_metadata(records: Sequence[Mapping[str, Any]]) -> None:
    reference = records[0]
    for field in SHARED_STUDY_FIELDS:
        if any(record[field] != reference[field] for record in records[1:]):
            raise ValueError(f"response-pilot inputs mix study metadata field {field!r}")


def _validate_complete_template_blocks(pairs: Sequence[ResponsePair]) -> None:
    seeds = sorted({pair.seed for pair in pairs})
    templates_by_block: dict[tuple[int, str], set[str]] = {}
    for pair in pairs:
        templates_by_block.setdefault((pair.seed, pair.persona_quadrant), set()).add(
            pair.persona_template_id
        )
    expected_blocks = {(seed, quadrant) for seed in seeds for quadrant in PERSONA_QUADRANTS}
    if set(templates_by_block) != expected_blocks:
        missing = sorted(expected_blocks.difference(templates_by_block))
        extra = sorted(set(templates_by_block).difference(expected_blocks))
        details = []
        if missing:
            details.append(f"missing blocks {missing!r}")
        if extra:
            details.append(f"unexpected blocks {extra!r}")
        raise ValueError("incomplete quadrant-by-seed design: " + "; ".join(details))
    for block, templates in sorted(templates_by_block.items()):
        expected = set(expected_template_ids(block[1]))
        if templates != expected:
            raise ValueError(
                f"seed={block[0]} quadrant={block[1]!r} has incomplete template block; "
                f"expected {sorted(expected)!r}, found {sorted(templates)!r}"
            )


def _analyze_metric(pairs: Sequence[ResponsePair], metric: str) -> dict[str, Any]:
    quadrant_values: dict[tuple[int, str], float | None] = {}
    for seed in sorted({pair.seed for pair in pairs}):
        for quadrant in PERSONA_QUADRANTS:
            block = [
                pair for pair in pairs if pair.seed == seed and pair.persona_quadrant == quadrant
            ]
            values = [getattr(pair, metric) for pair in block]
            quadrant_values[(seed, quadrant)] = (
                mean(float(value) for value in values if value is not None)
                if all(value is not None for value in values)
                else None
            )

    by_seed: list[SeedContrasts] = []
    omitted_seeds: list[int] = []
    for seed in sorted({pair.seed for pair in pairs}):
        values = {quadrant: quadrant_values[(seed, quadrant)] for quadrant in PERSONA_QUADRANTS}
        if any(value is None for value in values.values()):
            omitted_seeds.append(seed)
            continue
        hh = float(values["high_e_high_n"])
        hl = float(values["high_e_low_n"])
        lh = float(values["low_e_high_n"])
        ll = float(values["low_e_low_n"])
        by_seed.append(
            SeedContrasts(
                seed=seed,
                average_treatment=mean((hh, hl, lh, ll)),
                extraversion=(hh + hl - lh - ll) / 2,
                neuroticism=(hh + lh - hl - ll) / 2,
                interaction=hh - hl - lh + ll,
            )
        )

    summaries = {
        name: _summarize(tuple(getattr(row, name) for row in by_seed))
        for name in ("average_treatment", *PLANNED_CONTRASTS)
    }
    adjusted = _holm_adjust(
        {name: summaries[name].exact_sign_flip_p for name in PLANNED_CONTRASTS}
    )
    for name in PLANNED_CONTRASTS:
        summaries[name] = replace(summaries[name], holm_adjusted_p=adjusted[name])
    return {
        "direction": "encouragement_minus_neutral",
        "template_pair_count": len(pairs),
        "available_template_pair_count": sum(
            getattr(pair, metric) is not None for pair in pairs
        ),
        "omitted_template_pair_count": sum(
            getattr(pair, metric) is None for pair in pairs
        ),
        "quadrant_by_seed": [
            {
                "seed": seed,
                "quadrant": quadrant,
                "value": quadrant_values[(seed, quadrant)],
            }
            for seed in sorted({pair.seed for pair in pairs})
            for quadrant in PERSONA_QUADRANTS
        ],
        "by_seed": [asdict(row) for row in by_seed],
        "omitted_seeds": omitted_seeds,
        "average_treatment": asdict(summaries["average_treatment"]),
        "planned_contrasts": {
            name: asdict(summaries[name]) for name in PLANNED_CONTRASTS
        },
    }


def _token_diagnostics(pairs: Sequence[ResponsePair]) -> dict[str, Any]:
    length_pairs = Counter(
        (pair.encouragement_token_count, pair.neutral_token_count) for pair in pairs
    )
    maximum_length = max(
        max(pair.encouragement_token_count, pair.neutral_token_count) for pair in pairs
    )
    positions = []
    for offset in range(maximum_length):
        present = [
            pair
            for pair in pairs
            if offset < pair.encouragement_token_count and offset < pair.neutral_token_count
        ]
        token_pairs = Counter(
            (pair.encouragement_tokens[offset], pair.neutral_tokens[offset])
            for pair in present
        )
        positions.append(
            {
                "generated_token_offset": offset,
                "pair_count_both_present": len(present),
                "pair_count_same_token_text": sum(
                    count for (left, right), count in token_pairs.items() if left == right
                ),
                "token_text_pairs": [
                    {"encouragement": left, "neutral": right, "count": count}
                    for (left, right), count in sorted(
                        token_pairs.items(), key=lambda item: (-item[1], item[0])
                    )
                ],
            }
        )
    return {
        "analysis_role": "descriptive_only_no_token_position_is_an_outcome",
        "matched_length_pair_count": sum(
            pair.encouragement_token_count == pair.neutral_token_count for pair in pairs
        ),
        "mismatched_length_pair_count": sum(
            pair.encouragement_token_count != pair.neutral_token_count for pair in pairs
        ),
        "length_pairs": [
            {"encouragement": left, "neutral": right, "count": count}
            for (left, right), count in sorted(length_pairs.items())
        ],
        "positions": positions,
    }


def _summarize(values: tuple[float, ...]) -> ContrastSummary:
    if not values:
        return ContrastSummary(0, None, None, None, None)
    deviation = stdev(values) if len(values) > 1 else None
    return ContrastSummary(
        count=len(values),
        mean=mean(values),
        sample_stdev=deviation,
        standard_error=deviation / sqrt(len(values)) if deviation is not None else None,
        exact_sign_flip_p=_exact_sign_flip_p(values),
    )


def _exact_sign_flip_p(values: tuple[float, ...]) -> float:
    observed = abs(mean(values))
    extreme = 0
    for signs in product((-1, 1), repeat=len(values)):
        permuted = abs(mean(value * sign for value, sign in zip(values, signs, strict=True)))
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / (2 ** len(values))


def _holm_adjust(values: Mapping[str, float | None]) -> dict[str, float | None]:
    available = sorted(
        ((name, value) for name, value in values.items() if value is not None),
        key=lambda item: item[1],
    )
    result = {name: None for name in values}
    running = 0.0
    for index, (name, value) in enumerate(available):
        running = max(running, min(1.0, (len(available) - index) * value))
        result[name] = running
    return result


def _prompt_end_median(record: Mapping[str, Any], axis: str) -> float:
    projections = record.get("prompt_end_emotion")
    if not isinstance(projections, Mapping):
        raise ValueError("prompt_end_emotion must be an object")
    return _projection_median(projections, axis, "prompt_end_emotion")


def _projection_median(container: Mapping[str, Any], axis: str, label: str) -> float:
    projection = container.get(axis)
    if not isinstance(projection, Mapping):
        raise ValueError(f"{label}.{axis} must be an object")
    return _finite_number(projection.get("median"), f"{label}.{axis}.median")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _format_key(key: tuple[str, str, int, str]) -> str:
    return (
        f"(checkpoint_id={key[0]!r}, run_id={key[1]!r}, "
        f"seed={key[2]}, persona_id={key[3]!r})"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+", help="response-pilot JSONL file(s)")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)
    result = analyze_paths(args.input)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=args.indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
