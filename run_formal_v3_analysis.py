#!/usr/bin/env python3
"""Validate formal-v3 JSONL and write reproducible tidy analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from encouragement_lab.formal_v3_analysis import analyze_records
from encouragement_lab.formal_v3_records import iter_runs
from encouragement_lab.personas import PERSONA_KEYS
from encouragement_lab.records import file_checksum


FORMAL_SEEDS = tuple(range(3001, 3011))
TABLE_FIELDS = {
    "rounds.csv": (
        "run_id",
        "seed",
        "arm",
        "persona_id",
        "quadrant",
        "template",
        "status",
        "axis",
        "round",
        "attempt_outcome",
        "feedback_frame",
        "filler_id",
        "generation_seed",
        "candidate_before_count",
        "candidate_before_sha256",
        "candidate_after_count",
        "candidate_after_sha256",
        "projection",
    ),
    "runs.csv": (
        "run_id",
        "seed",
        "arm",
        "persona_id",
        "quadrant",
        "template",
        "axis",
        "slope",
        "r5_minus_r1",
    ),
    "template_contrasts.csv": (
        "seed",
        "quadrant",
        "template",
        "persona_id",
        "axis",
        "contrast",
        "slope",
        "r5_minus_r1",
    ),
    "quadrant_contrasts.csv": (
        "seed",
        "quadrant",
        "axis",
        "contrast",
        "template_count",
        "slope",
        "r5_minus_r1",
    ),
    "seed_contrasts.csv": (
        "seed",
        "axis",
        "contrast",
        "overall_slope",
        "extraversion_slope",
        "neuroticism_slope",
        "interaction_slope",
        "overall_r5_minus_r1",
        "extraversion_r5_minus_r1",
        "neuroticism_r5_minus_r1",
        "interaction_r5_minus_r1",
    ),
    "run_exclusions.csv": ("run_id", "seed", "arm", "persona_id", "reason"),
    "missing_records.csv": ("persona_id", "seed", "arm", "reason"),
    "contrast_missing_map.csv": (
        "seed",
        "axis",
        "contrast",
        "persona_id",
        "arm",
        "reason",
    ),
    "manipulation_seeds.csv": ("axis", "seed", "slope", "r5_minus_r1"),
    "manipulation_summary.csv": (
        "axis", "eligible_seed_count", "slope_median", "r5_minus_r1_median",
        "positive_r5_minus_r1_count", "confirmatory_test",
    ),
    "manipulation_missing_map.csv": ("axis", "seed", "persona_id", "arm", "reason"),
    "co_primary.csv": (
        "axis",
        "contrast",
        "effect",
        "count",
        "mean",
        "sample_stdev",
        "standard_error",
        "exact_sign_flip_p",
        "holm_adjusted_p",
    ),
    "moderation.csv": (
        "axis",
        "contrast",
        "effect",
        "count",
        "mean",
        "sample_stdev",
        "standard_error",
        "exact_sign_flip_p",
        "holm_adjusted_p",
    ),
    "r5_minus_r1_robustness.csv": (
        "axis",
        "contrast",
        "effect",
        "count",
        "mean",
        "sample_stdev",
        "standard_error",
        "confirmatory_test",
    ),
    "neutral_minus_feedback_only_diagnostic.csv": (
        "axis",
        "contrast",
        "effect",
        "count",
        "mean",
        "sample_stdev",
        "standard_error",
        "confirmatory_test",
    ),
    "supportive_minus_feedback_only_descriptive.csv": (
        "axis",
        "contrast",
        "effect",
        "count",
        "mean",
        "sample_stdev",
        "standard_error",
        "confirmatory_test",
    ),
}


DATA_DICTIONARY = """# formal-v3 derived-data dictionary

The immutable JSONL remains the source of truth. These files are deterministic
derivatives and contain no model execution.

- `rounds.csv`: one eligible run-round prompt-boundary projection per axis.
- `runs.csv`: one eligible five-failure run-axis, with OLS slope and R5−R1.
- `template_contrasts.csv`: arm differences within persona-template × seed.
- `quadrant_contrasts.csv`: v1–v3 mean after within-template arm subtraction.
- `seed_contrasts.csv`: four-quadrant overall, E, N, and E×N effects per seed.
- `co_primary.csv`: supportive−neutral overall slope tests across joyful,
  grief-stricken, and furious; the three p-values form one Holm family.
- `moderation.csv`: the nine supportive−neutral slope moderation tests
  (three axes × three moderators) form a separate Holm family.
- `neutral_minus_feedback_only_diagnostic.csv` and
  `supportive_minus_feedback_only_descriptive.csv`: descriptive slopes only.
- `r5_minus_r1_robustness.csv`: endpoint robustness effects, with no
  confirmatory p-value.
- exclusion and missing-map tables: all incomplete runs/cells and the seed
  blocks they prevent from entering a contrast.
- manipulation tables: descriptive feedback-only trajectory summaries; no
  confirmatory test is attached.

`projection` is a single-layer (layer 17) cosine projection onto an external
discrete-emotion vector. It is not a measure of subjective experience.
"""


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _table_rows(analysis: Mapping[str, Any]) -> dict[str, Sequence[Mapping[str, Any]]]:
    return {
        "rounds.csv": analysis["round_rows"],
        "runs.csv": analysis["run_rows"],
        "template_contrasts.csv": analysis["template_contrasts"],
        "quadrant_contrasts.csv": analysis["quadrant_contrasts"],
        "seed_contrasts.csv": analysis["seed_contrasts"],
        "run_exclusions.csv": analysis["run_exclusions"],
        "missing_records.csv": analysis["missing_records"],
        "contrast_missing_map.csv": analysis["contrast_missing_map"],
        "manipulation_seeds.csv": analysis["manipulation"]["seed_rows"],
        "manipulation_summary.csv": analysis["manipulation"]["axis_summaries"],
        "manipulation_missing_map.csv": analysis["manipulation"]["missing_map"],
        "co_primary.csv": analysis["inference"]["co_primary"],
        "moderation.csv": analysis["inference"]["moderation"],
        "r5_minus_r1_robustness.csv": analysis["inference"][
            "r5_minus_r1_robustness"
        ],
        "neutral_minus_feedback_only_diagnostic.csv": analysis["inference"]["neutral_minus_feedback_only_diagnostic"],
        "supportive_minus_feedback_only_descriptive.csv": analysis["inference"]["supportive_minus_feedback_only_descriptive"],
    }


def write_analysis_bundle(
    source: Path,
    destination: Path,
    *,
    expected_seeds: Sequence[int],
    expected_personas: Sequence[str] = PERSONA_KEYS,
) -> dict[str, Any]:
    """Write a complete bundle atomically; refuse an existing destination."""

    if destination.exists():
        raise FileExistsError(f"analysis destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    analysis = analyze_records(
        iter_runs(source),
        expected_seeds=expected_seeds,
        expected_personas=expected_personas,
    )
    tables = _table_rows(analysis)

    with TemporaryDirectory(prefix="formal-v3-analysis-", dir=destination.parent) as raw:
        staging = Path(raw)
        for filename, fields in TABLE_FIELDS.items():
            _write_csv(staging / filename, tables[filename], fields)
        dictionary = staging / "DATA_DICTIONARY.md"
        dictionary.write_text(DATA_DICTIONARY, encoding="utf-8")

        artifact_hashes = {
            path.name: file_checksum(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manipulation = dict(analysis["manipulation"])
        manipulation.pop("seed_rows")
        manipulation.pop("missing_map")
        summary = {
            "schema_version": 1,
            "source": str(source.resolve()),
            "source_sha256": file_checksum(source),
            "expected_seeds": list(expected_seeds),
            "expected_personas": list(expected_personas),
            "row_counts": {name: len(rows) for name, rows in tables.items()},
            "inference": analysis["inference"],
            "manipulation": manipulation,
            "provenance": analysis["provenance"],
            "artifact_sha256": artifact_hashes,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--persona", action="append", dest="personas")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds) if args.seeds else FORMAL_SEEDS
    personas = tuple(args.personas) if args.personas else PERSONA_KEYS
    summary = write_analysis_bundle(
        args.input,
        args.output_dir,
        expected_seeds=seeds,
        expected_personas=personas,
    )
    print(
        f"wrote formal-v3 analysis bundle to {args.output_dir} "
        f"from {summary['source_sha256']}"
    )


if __name__ == "__main__":
    main()
