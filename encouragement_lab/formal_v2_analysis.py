"""Validated seed-block analysis for formal-v2 frustration trajectories.

This module is independent of the formal-v1 paired-branch analysis. It performs
no model execution and never edits the immutable source JSONL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite, sqrt
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping, Sequence

from .formal_v2_records import ARMS, validate
from .personas import PERSONA_BY_ID, PERSONA_KEYS, PERSONA_QUADRANTS


PRIMARY_CONTRASTS = (
    "neutral_minus_feedback_only",
    "supportive_minus_neutral",
)
DERIVED_CONTRAST = "supportive_minus_feedback_only"
ALL_CONTRASTS = (*PRIMARY_CONTRASTS, DERIVED_CONTRAST)
CONTRAST_ARMS = {
    "neutral_minus_feedback_only": ("neutral", "feedback_only"),
    "supportive_minus_neutral": ("supportive", "neutral"),
    "supportive_minus_feedback_only": ("supportive", "feedback_only"),
}
MODERATORS = ("extraversion", "neuroticism", "interaction")


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    seed: int
    arm: str
    persona_id: str
    quadrant: str
    template: str
    slope: float
    r5_minus_r1: float


def ols_slope(values: Sequence[float]) -> float:
    """Return the OLS slope against one-indexed equally spaced rounds."""

    if len(values) < 2:
        raise ValueError("a trajectory slope requires at least two rounds")
    x_mean = (len(values) + 1) / 2
    y_mean = mean(values)
    denominator = sum(
        (round_number - x_mean) ** 2
        for round_number in range(1, len(values) + 1)
    )
    return sum(
        (round_number - x_mean) * (value - y_mean)
        for round_number, value in enumerate(values, start=1)
    ) / denominator


def exact_sign_flip_p(values: Sequence[float]) -> float | None:
    """Two-sided exact sign-flip p-value for paired seed-level effects."""

    if not values:
        return None
    observed = abs(mean(values))
    extreme = 0
    for signs in product((-1, 1), repeat=len(values)):
        permuted = abs(
            mean(
                value * sign
                for value, sign in zip(values, signs, strict=True)
            )
        )
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / (2 ** len(values))


def holm_adjust(
    p_values: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Holm-adjust one predeclared family, retaining unavailable entries."""

    available = sorted(
        ((name, value) for name, value in p_values.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    family_size = len(available)
    for index, (name, value) in enumerate(available):
        running = max(running, min(1.0, (family_size - index) * value))
        adjusted[name] = running
    return adjusted


def _descriptive(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "sample_stdev": None,
            "standard_error": None,
        }
    deviation = stdev(values) if len(values) > 1 else None
    return {
        "count": len(values),
        "mean": mean(values),
        "sample_stdev": deviation,
        "standard_error": (
            deviation / sqrt(len(values)) if deviation is not None else None
        ),
    }


def _frustration_median(attempt: Mapping[str, Any]) -> float:
    try:
        value = attempt["readout"]["frustration"]["median"]
    except (KeyError, TypeError) as error:
        raise ValueError("missing frustration readout") from error
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("frustration median must be numeric")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError("frustration median must be finite")
    return numeric


def _round_row(
    record: Mapping[str, Any], attempt: Mapping[str, Any], value: float
) -> dict[str, Any]:
    before = attempt["candidate_state_before"]
    after = attempt["candidate_state_after"]
    return {
        "run_id": record["run_id"],
        "seed": record["seed"],
        "arm": record["arm"],
        "persona_id": record["persona_id"],
        "quadrant": record["persona_quadrant"],
        "template": record["persona_template_id"],
        "status": record["status"],
        "round": attempt["attempt_index"],
        "attempt_outcome": attempt["outcome"],
        "feedback_frame": attempt["feedback_frame"],
        "filler_id": attempt["filler_id"],
        "generation_seed": attempt["generation_seed"],
        "candidate_before_count": before["candidate_count"],
        "candidate_before_sha256": before["candidate_set_sha256"],
        "candidate_after_count": after["candidate_count"],
        "candidate_after_sha256": after["candidate_set_sha256"],
        "frustration_median": value,
    }


def _validate_persona_metadata(record: Mapping[str, Any]) -> None:
    try:
        spec = PERSONA_BY_ID[record["persona_id"]]
    except KeyError as error:
        raise ValueError(f"unknown formal-v2 persona: {record['persona_id']}") from error
    if (
        record["persona_quadrant"] != spec.quadrant_id
        or record["persona_template_id"] != spec.template_id
    ):
        raise ValueError(f"persona metadata mismatch for {record['persona_id']}")


def extract_eligible_runs(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    expected_personas: Sequence[str] = PERSONA_KEYS,
) -> dict[str, Any]:
    """Validate records and atomically extract complete five-point runs.

    A run with an early win or any unusable readout contributes no round rows.
    This prevents a partially valid prefix from leaking into a tidy table.
    """

    seed_set = set(expected_seeds)
    persona_set = set(expected_personas)
    if len(seed_set) != len(expected_seeds):
        raise ValueError("expected_seeds contains duplicates")
    if len(persona_set) != len(expected_personas):
        raise ValueError("expected_personas contains duplicates")

    round_rows: list[dict[str, Any]] = []
    run_rows: list[RunSummary] = []
    exclusions: list[dict[str, Any]] = []
    present: dict[tuple[str, int, str], str] = {}
    reference_provenance: Mapping[str, Any] | None = None

    for raw in records:
        record = validate(raw)
        _validate_persona_metadata(record)
        identity = (record["persona_id"], record["seed"], record["arm"])
        if record["seed"] not in seed_set or record["persona_id"] not in persona_set:
            raise ValueError(f"unscheduled formal-v2 record: {identity}")
        if identity in present:
            raise ValueError(f"duplicate formal-v2 record: {identity}")
        present[identity] = record["run_id"]

        if reference_provenance is None:
            reference_provenance = record["provenance"]
        elif record["provenance"] != reference_provenance:
            raise ValueError(f"formal-v2 provenance mismatch for {identity}")

        if record["status"] != "complete_five_failures":
            exclusions.append(
                {
                    "run_id": record["run_id"],
                    "seed": record["seed"],
                    "arm": record["arm"],
                    "persona_id": record["persona_id"],
                    "reason": "early_win",
                }
            )
            continue

        local_rounds: list[dict[str, Any]] = []
        values: list[float] = []
        failure_reason: str | None = None
        for attempt in record["trajectory"]:
            try:
                value = _frustration_median(attempt)
            except ValueError as error:
                failure_reason = str(error)
                break
            values.append(value)
            local_rounds.append(_round_row(record, attempt, value))

        if failure_reason is not None:
            exclusions.append(
                {
                    "run_id": record["run_id"],
                    "seed": record["seed"],
                    "arm": record["arm"],
                    "persona_id": record["persona_id"],
                    "reason": failure_reason,
                }
            )
            continue

        round_rows.extend(local_rounds)
        run_rows.append(
            RunSummary(
                run_id=record["run_id"],
                seed=record["seed"],
                arm=record["arm"],
                persona_id=record["persona_id"],
                quadrant=record["persona_quadrant"],
                template=record["persona_template_id"],
                slope=ols_slope(values),
                r5_minus_r1=values[-1] - values[0],
            )
        )

    missing_records = [
        {
            "persona_id": persona_id,
            "seed": seed,
            "arm": arm,
            "reason": "missing_record",
        }
        for seed in expected_seeds
        for persona_id in expected_personas
        for arm in sorted(ARMS)
        if (persona_id, seed, arm) not in present
    ]
    return {
        "round_rows": round_rows,
        "run_rows": [asdict(row) for row in run_rows],
        "exclusions": exclusions,
        "missing_records": missing_records,
        "provenance": reference_provenance,
    }


def _factorial_values(quadrants: Mapping[str, float]) -> dict[str, float]:
    hh = quadrants["high_e_high_n"]
    hl = quadrants["high_e_low_n"]
    lh = quadrants["low_e_high_n"]
    ll = quadrants["low_e_low_n"]
    return {
        "overall": mean((hh, hl, lh, ll)),
        "extraversion": (hh + hl - lh - ll) / 2,
        "neuroticism": (hh + lh - hl - ll) / 2,
        "interaction": hh - hl - lh + ll,
    }


def _seed_contrasts(
    run_rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    expected_personas: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    by_cell = {
        (row["seed"], row["persona_id"], row["arm"]): row for row in run_rows
    }
    template_rows: list[dict[str, Any]] = []
    quadrant_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    missing_map: list[dict[str, Any]] = []

    for contrast in ALL_CONTRASTS:
        left_arm, right_arm = CONTRAST_ARMS[contrast]
        for seed in expected_seeds:
            cells: list[dict[str, Any]] = []
            missing: list[tuple[str, str]] = []
            for persona_id in expected_personas:
                left = by_cell.get((seed, persona_id, left_arm))
                right = by_cell.get((seed, persona_id, right_arm))
                if left is None:
                    missing.append((persona_id, left_arm))
                if right is None:
                    missing.append((persona_id, right_arm))
                if left is None or right is None:
                    continue
                spec = PERSONA_BY_ID[persona_id]
                cells.append(
                    {
                        "seed": seed,
                        "quadrant": spec.quadrant_id,
                        "template": spec.template_id,
                        "persona_id": persona_id,
                        "contrast": contrast,
                        "slope": left["slope"] - right["slope"],
                        "r5_minus_r1": left["r5_minus_r1"]
                        - right["r5_minus_r1"],
                    }
                )

            template_rows.extend(cells)
            if missing:
                missing_map.extend(
                    {
                        "seed": seed,
                        "contrast": contrast,
                        "persona_id": persona_id,
                        "arm": arm,
                        "reason": "ineligible_or_missing_run",
                    }
                    for persona_id, arm in sorted(set(missing))
                )
                continue

            by_quadrant: dict[str, list[dict[str, Any]]] = {
                quadrant: [] for quadrant in PERSONA_QUADRANTS
            }
            for cell in cells:
                by_quadrant[cell["quadrant"]].append(cell)
            if any(len(items) != 3 for items in by_quadrant.values()):
                raise ValueError("complete contrast block lacks three templates")

            slope_quadrants: dict[str, float] = {}
            delta_quadrants: dict[str, float] = {}
            for quadrant in PERSONA_QUADRANTS:
                items = by_quadrant[quadrant]
                slope_value = mean(item["slope"] for item in items)
                delta_value = mean(item["r5_minus_r1"] for item in items)
                slope_quadrants[quadrant] = slope_value
                delta_quadrants[quadrant] = delta_value
                quadrant_rows.append(
                    {
                        "seed": seed,
                        "quadrant": quadrant,
                        "contrast": contrast,
                        "template_count": 3,
                        "slope": slope_value,
                        "r5_minus_r1": delta_value,
                    }
                )

            slope_effects = _factorial_values(slope_quadrants)
            delta_effects = _factorial_values(delta_quadrants)
            seed_rows.append(
                {
                    "seed": seed,
                    "contrast": contrast,
                    **{
                        f"{effect}_slope": value
                        for effect, value in slope_effects.items()
                    },
                    **{
                        f"{effect}_r5_minus_r1": value
                        for effect, value in delta_effects.items()
                    },
                }
            )

    return {
        "template_contrasts": template_rows,
        "quadrant_contrasts": quadrant_rows,
        "seed_contrasts": seed_rows,
        "contrast_missing_map": missing_map,
    }


def _inferential_summaries(
    seed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary: list[dict[str, Any]] = []
    raw_primary_p: dict[str, float | None] = {}
    for contrast in PRIMARY_CONTRASTS:
        values = [
            row["overall_slope"]
            for row in seed_rows
            if row["contrast"] == contrast
        ]
        p_value = exact_sign_flip_p(values)
        raw_primary_p[contrast] = p_value
        primary.append(
            {
                "contrast": contrast,
                "effect": "overall",
                **_descriptive(values),
                "exact_sign_flip_p": p_value,
                "holm_adjusted_p": None,
            }
        )
    adjusted_primary = holm_adjust(raw_primary_p)
    for row in primary:
        row["holm_adjusted_p"] = adjusted_primary[row["contrast"]]

    moderation: list[dict[str, Any]] = []
    raw_moderation_p: dict[str, float | None] = {}
    for contrast in PRIMARY_CONTRASTS:
        for moderator in MODERATORS:
            values = [
                row[f"{moderator}_slope"]
                for row in seed_rows
                if row["contrast"] == contrast
            ]
            key = f"{contrast}:{moderator}"
            p_value = exact_sign_flip_p(values)
            raw_moderation_p[key] = p_value
            moderation.append(
                {
                    "contrast": contrast,
                    "effect": moderator,
                    **_descriptive(values),
                    "exact_sign_flip_p": p_value,
                    "holm_adjusted_p": None,
                }
            )
    adjusted_moderation = holm_adjust(raw_moderation_p)
    for row in moderation:
        key = f"{row['contrast']}:{row['effect']}"
        row["holm_adjusted_p"] = adjusted_moderation[key]

    derived_values = [
        row["overall_slope"]
        for row in seed_rows
        if row["contrast"] == DERIVED_CONTRAST
    ]
    derived = {
        "contrast": DERIVED_CONTRAST,
        "effect": "overall",
        **_descriptive(derived_values),
        "confirmatory_test": None,
    }

    robustness: list[dict[str, Any]] = []
    for contrast in ALL_CONTRASTS:
        for effect in ("overall", *MODERATORS):
            values = [
                row[f"{effect}_r5_minus_r1"]
                for row in seed_rows
                if row["contrast"] == contrast
            ]
            robustness.append(
                {
                    "contrast": contrast,
                    "effect": effect,
                    **_descriptive(values),
                    "confirmatory_test": None,
                }
            )

    return {
        "co_primary": primary,
        "moderation": moderation,
        "derived_total": derived,
        "r5_minus_r1_robustness": robustness,
    }


def _feedback_only_manipulation(
    run_rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    expected_personas: Sequence[str],
) -> dict[str, Any]:
    by_cell = {
        (row["seed"], row["persona_id"]): row
        for row in run_rows
        if row["arm"] == "feedback_only"
    }
    seed_rows: list[dict[str, Any]] = []
    missing_map: list[dict[str, Any]] = []

    for seed in expected_seeds:
        missing = [
            persona_id
            for persona_id in expected_personas
            if (seed, persona_id) not in by_cell
        ]
        if missing:
            missing_map.extend(
                {
                    "seed": seed,
                    "persona_id": persona_id,
                    "arm": "feedback_only",
                    "reason": "ineligible_or_missing_run",
                }
                for persona_id in missing
            )
            continue

        quadrant_slopes: dict[str, list[float]] = {
            quadrant: [] for quadrant in PERSONA_QUADRANTS
        }
        quadrant_deltas: dict[str, list[float]] = {
            quadrant: [] for quadrant in PERSONA_QUADRANTS
        }
        for persona_id in expected_personas:
            spec = PERSONA_BY_ID[persona_id]
            row = by_cell[(seed, persona_id)]
            quadrant_slopes[spec.quadrant_id].append(row["slope"])
            quadrant_deltas[spec.quadrant_id].append(row["r5_minus_r1"])
        if any(len(values) != 3 for values in quadrant_slopes.values()):
            raise ValueError("complete feedback-only seed lacks three templates")

        slope = mean(
            mean(quadrant_slopes[quadrant]) for quadrant in PERSONA_QUADRANTS
        )
        delta = mean(
            mean(quadrant_deltas[quadrant]) for quadrant in PERSONA_QUADRANTS
        )
        seed_rows.append(
            {"seed": seed, "slope": slope, "r5_minus_r1": delta}
        )

    slopes = [row["slope"] for row in seed_rows]
    deltas = [row["r5_minus_r1"] for row in seed_rows]
    slope_median = median(slopes) if slopes else None
    delta_median = median(deltas) if deltas else None
    positive_delta_count = sum(value > 0 for value in deltas)
    exactly_ten_eligible = len(expected_seeds) == 10 and len(seed_rows) == 10
    criteria = {
        "exactly_ten_eligible_seeds": exactly_ten_eligible,
        "positive_seed_slope_median": bool(
            slope_median is not None and slope_median > 0
        ),
        "positive_seed_r5_minus_r1_median": bool(
            delta_median is not None and delta_median > 0
        ),
        "at_least_seven_positive_seed_r5_minus_r1": (
            positive_delta_count >= 7
        ),
    }
    passed = all(criteria.values())
    return {
        "eligible_seed_count": len(seed_rows),
        "expected_seed_count": len(expected_seeds),
        "seed_rows": seed_rows,
        "slope_median": slope_median,
        "r5_minus_r1_median": delta_median,
        "positive_r5_minus_r1_count": positive_delta_count,
        "criteria": criteria,
        "passed": passed,
        "missing_map": missing_map,
        "failure_interpretation": (
            None
            if passed
            else "retain all data; repeated-failure buildup not reliably reproduced"
        ),
    }


def analyze_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    expected_personas: Sequence[str] = PERSONA_KEYS,
) -> dict[str, Any]:
    """Run the approved formal-v2 analysis without file I/O."""

    extracted = extract_eligible_runs(
        records,
        expected_seeds=expected_seeds,
        expected_personas=expected_personas,
    )
    contrasts = _seed_contrasts(
        extracted["run_rows"],
        expected_seeds=expected_seeds,
        expected_personas=expected_personas,
    )
    return {
        "schema_version": 1,
        "round_rows": extracted["round_rows"],
        "run_rows": extracted["run_rows"],
        "run_exclusions": extracted["exclusions"],
        "missing_records": extracted["missing_records"],
        **contrasts,
        "inference": _inferential_summaries(contrasts["seed_contrasts"]),
        "manipulation": _feedback_only_manipulation(
            extracted["run_rows"],
            expected_seeds=expected_seeds,
            expected_personas=expected_personas,
        ),
        "provenance": extracted["provenance"],
    }
