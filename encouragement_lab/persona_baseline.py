"""Immutable paper snapshot helpers for persona-only probe baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from .emotion_probe import AXES
from .persona_calibration import CALIBRATION_LAYERS, _cosine, _main_effect
from .personas import PERSONA_SPECS


def baseline_rows(
    hidden_by_suffix: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    directions: Mapping[str, Mapping[int, np.ndarray]],
    direction_layer_map: Mapping[int, int],
    suffix_texts: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Return one raw score row per suffix/persona/axis/layer.

    The values are deterministic prompt endpoints, not stochastic samples.
    """

    by_id = {spec.prompt_key: spec for spec in PERSONA_SPECS}
    rows: list[dict[str, Any]] = []
    for suffix_id, by_persona in hidden_by_suffix.items():
        for persona_id, by_layer in by_persona.items():
            spec = by_id[persona_id]
            for axis in AXES:
                scores = {
                    layer: _cosine(by_layer[layer], directions[axis][direction_layer_map[layer]])
                    for layer in CALIBRATION_LAYERS
                }
                axis_median = float(median(scores.values()))
                for layer, score in scores.items():
                    rows.append(
                        {
                            "suffix_id": suffix_id,
                            "suffix_text": suffix_texts[suffix_id],
                            "persona_id": persona_id,
                            "persona_quadrant": spec.quadrant_id,
                            "persona_template_id": spec.template_id,
                            "extraversion": spec.extraversion,
                            "neuroticism": spec.neuroticism,
                            "axis": axis,
                            "layer_relative": layer,
                            "layer_absolute": direction_layer_map[layer],
                            "cosine_score": float(score),
                            "persona_axis_median": axis_median,
                        }
                    )
    return rows


def summarize_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe raw scores without treating suffixes or templates as samples."""

    grouped: dict[tuple[str, str, str, int], list[float]] = {}
    for row in rows:
        key = (str(row["suffix_id"]), str(row["axis"]), str(row["persona_id"]), int(row["layer_relative"]))
        grouped.setdefault(key, []).append(float(row["cosine_score"]))
    if any(len(values) != 1 for values in grouped.values()):
        raise ValueError("baseline rows must have exactly one score per suffix/persona/axis/layer")

    by_suffix: dict[str, Any] = {}
    for suffix in sorted({str(row["suffix_id"]) for row in rows}):
        suffix_rows = [row for row in rows if row["suffix_id"] == suffix]
        by_suffix[suffix] = {
            axis: _axis_summary([row for row in suffix_rows if row["axis"] == axis])
            for axis in AXES
        }
    return {
        "analysis_role": "descriptive_only_not_calibrated_emotion_or_independent_samples",
        "by_suffix": by_suffix,
    }


def write_snapshot(
    json_path: str | Path, csv_path: str | Path, payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write JSON and tidy CSV once; refuse either existing destination."""

    json_destination = Path(json_path)
    csv_destination = Path(csv_path)
    if json_destination.exists() or csv_destination.exists():
        existing = [str(path) for path in (json_destination, csv_destination) if path.exists()]
        raise FileExistsError("refusing to overwrite snapshot output: " + ", ".join(existing))
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    columns = (
        "schema_version", "snapshot_id", "suffix_id", "suffix_text_sha256", "persona_id",
        "persona_quadrant", "persona_template_id", "extraversion", "neuroticism", "axis",
        "layer_relative", "layer_absolute", "cosine_score", "persona_axis_median",
    )
    try:
        with csv_destination.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in columns})
    except Exception:
        # JSON remains deliberately immutable if CSV persistence fails; expose
        # the failure rather than replacing either artifact.
        raise


def _axis_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = {
        persona: {
            str(row["layer_relative"]): float(row["cosine_score"])
            for row in rows
            if row["persona_id"] == persona
        }
        for persona in sorted({str(row["persona_id"]) for row in rows})
    }
    specs = [spec for spec in PERSONA_SPECS]
    factorial = {
        str(layer): {
            factor: float(
                _main_effect(
                    {persona: {layer: scores[str(layer)]} for persona, scores in values.items()},
                    specs,
                    factor,
                    layer,
                )
            )
            for factor in ("extraversion", "neuroticism")
        }
        for layer in CALIBRATION_LAYERS
    }
    return {
        "by_template_quadrant": values,
        "factorial_main_effects": factorial,
        "factorial_five_layer_medians": {
            factor: float(median(factorial[str(layer)][factor] for layer in CALIBRATION_LAYERS))
            for factor in ("extraversion", "neuroticism")
        },
    }
