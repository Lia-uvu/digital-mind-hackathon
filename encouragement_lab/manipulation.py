"""Pre-intervention frustration manipulation checks from paired JSONL records.

All reported values are fixed before the encouragement/neutral intervention.
The module deliberately reads only a shared checkpoint's baseline and
``pre_intervention_trajectory``; it never inspects downstream branch outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from encouragement_lab.records import iter_records


_SHARED_METADATA_FIELDS = (
    "schema_version",
    "run_id",
    "seed",
    "branch_seed",
    "checkpoint_id",
    "model",
    "sampling",
    "persona_id",
    "prompt_checksum",
    "failure_rounds",
    "candidate_count_at_checkpoint",
    "pre_intervention_trajectory",
    "code_version",
    "dependency_versions",
    "emotion_probe",
)


@dataclass(frozen=True)
class FrustrationRun:
    """One independent run's pre-intervention frustration trajectory."""

    run_id: str
    persona_id: str
    baseline: float
    round_frustration_medians: tuple[float, ...]
    final_round: int
    baseline_to_final_delta: float
    round1_to_final_delta: float
    round_slope: float | None
    pre_intervention_guess_rule_violation_rounds: int


@dataclass(frozen=True)
class FrustrationSummary:
    """Run-level frustration summary for a persona or the pooled sample."""

    n: int
    final_round: int | None
    baseline_to_final_delta_median: float | None
    round1_to_final_delta_median: float | None
    round_slope_median: float | None
    round1_to_final_positive_fraction: float | None
    round_medians: tuple[float, ...] | None
    fraction_above_round1_by_round: tuple[float | None, ...] | None


@dataclass(frozen=True)
class ManipulationPass:
    """The preregistered pass decision and each of its transparent checks."""

    passed: bool
    at_least_twenty_runs: bool
    equal_final_rounds: bool
    final_round_at_least_two: bool
    positive_round1_to_final_median: bool
    positive_slope_median: bool
    positive_fraction_at_least_point_seven: bool


@dataclass(frozen=True)
class ManipulationAnalysis:
    """Complete, JSON-serializable result of the pre-intervention check."""

    runs: tuple[FrustrationRun, ...]
    by_persona: dict[str, FrustrationSummary]
    pooled: FrustrationSummary
    manipulation_pass: ManipulationPass


def analyze_frustration_jsonl(path: str | Path) -> ManipulationAnalysis:
    """Read a JSONL file exclusively through :func:`records.iter_records`."""
    return analyze_frustration_records(iter_records(path))


def analyze_frustration_records(
    records: Iterable[Mapping[str, Any]],
) -> ManipulationAnalysis:
    """Validate paired branches and calculate pre-intervention run statistics."""
    pairs = _pair_records(records)
    runs_by_identity: dict[tuple[str, str], FrustrationRun] = {}
    for key, encouragement, neutral in pairs:
        _validate_shared_branches(key, encouragement, neutral)
        run = _run_from_shared_checkpoint(key, encouragement)
        run_key = (run.run_id, run.persona_id)
        previous = runs_by_identity.get(run_key)
        if previous is None:
            runs_by_identity[run_key] = run
        elif previous != run:
            raise ValueError(
                f"{_format_key(key)} conflicts with another checkpoint for "
                f"independent run {run_key!r}"
            )

    runs = tuple(sorted(runs_by_identity.values(), key=lambda run: (run.persona_id, run.run_id)))
    by_persona = {
        persona_id: summarize_runs(persona_runs)
        for persona_id, persona_runs in _group_by_persona(runs).items()
    }
    pooled = summarize_runs(runs)
    return ManipulationAnalysis(
        runs=runs,
        by_persona=by_persona,
        pooled=pooled,
        manipulation_pass=evaluate_manipulation_pass(runs, pooled),
    )


def summarize_runs(runs: Iterable[FrustrationRun]) -> FrustrationSummary:
    """Summarize independent runs without treating branches as observations."""
    materialized = tuple(runs)
    if not materialized:
        return FrustrationSummary(0, None, None, None, None, None, None, None)

    final_rounds = {run.final_round for run in materialized}
    trajectory_lengths = {len(run.round_frustration_medians) for run in materialized}
    shared_final_round = (
        next(iter(final_rounds))
        if len(final_rounds) == len(trajectory_lengths) == 1
        and next(iter(final_rounds)) == next(iter(trajectory_lengths))
        else None
    )
    slopes = [run.round_slope for run in materialized]
    return FrustrationSummary(
        n=len(materialized),
        final_round=shared_final_round,
        baseline_to_final_delta_median=median(
            run.baseline_to_final_delta for run in materialized
        ),
        round1_to_final_delta_median=median(
            run.round1_to_final_delta for run in materialized
        ),
        round_slope_median=(
            median(slope for slope in slopes if slope is not None)
            if all(slope is not None for slope in slopes)
            else None
        ),
        round1_to_final_positive_fraction=(
            sum(run.round1_to_final_delta > 0 for run in materialized) / len(materialized)
        ),
        round_medians=(
            tuple(
                median(run.round_frustration_medians[round_index] for run in materialized)
                for round_index in range(shared_final_round)
            )
            if shared_final_round is not None
            else None
        ),
        fraction_above_round1_by_round=(
            (None,)
            + tuple(
                sum(
                    run.round_frustration_medians[round_index]
                    > run.round_frustration_medians[0]
                    for run in materialized
                )
                / len(materialized)
                for round_index in range(1, shared_final_round)
            )
            if shared_final_round is not None
            else None
        ),
    )


def evaluate_manipulation_pass(
    runs: Iterable[FrustrationRun], pooled: FrustrationSummary | None = None
) -> ManipulationPass:
    """Evaluate the fixed preregistered frustration manipulation criteria."""
    materialized = tuple(runs)
    summary = summarize_runs(materialized) if pooled is None else pooled
    at_least_twenty_runs = len(materialized) >= 20
    equal_final_rounds = summary.final_round is not None
    final_round_at_least_two = bool(summary.final_round and summary.final_round >= 2)
    positive_round1_to_final_median = bool(
        summary.round1_to_final_delta_median is not None
        and summary.round1_to_final_delta_median > 0
    )
    positive_slope_median = bool(
        summary.round_slope_median is not None and summary.round_slope_median > 0
    )
    positive_fraction_at_least_point_seven = bool(
        summary.round1_to_final_positive_fraction is not None
        and summary.round1_to_final_positive_fraction >= 0.70
    )
    checks = (
        at_least_twenty_runs,
        equal_final_rounds,
        final_round_at_least_two,
        positive_round1_to_final_median,
        positive_slope_median,
        positive_fraction_at_least_point_seven,
    )
    return ManipulationPass(bool(all(checks)), *checks)


def _pair_records(
    records: Iterable[Mapping[str, Any]],
) -> list[tuple[tuple[str, str, str], Mapping[str, Any], Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    for record in records:
        key = _pair_key(record)
        condition = record.get("condition")
        if condition not in {"encouragement", "neutral"}:
            raise ValueError(f"{_format_key(key)} has unsupported condition {condition!r}")
        branches = grouped.setdefault(key, {})
        if condition in branches:
            raise ValueError(f"{_format_key(key)} has duplicate {condition!r} branch")
        branches[condition] = record

    pairs = []
    for key in sorted(grouped):
        branches = grouped[key]
        missing = {"encouragement", "neutral"}.difference(branches)
        if missing:
            raise ValueError(f"{_format_key(key)} is missing branch(es): {', '.join(sorted(missing))}")
        pairs.append((key, branches["encouragement"], branches["neutral"]))
    return pairs


def _validate_shared_branches(
    key: tuple[str, str, str], encouragement: Mapping[str, Any], neutral: Mapping[str, Any]
) -> None:
    for field in _SHARED_METADATA_FIELDS:
        if encouragement.get(field) != neutral.get(field):
            raise ValueError(f"{_format_key(key)} has mismatched branch metadata: {field}")
    encouragement_baseline = _baseline_axis(encouragement, key)
    neutral_baseline = _baseline_axis(neutral, key)
    if encouragement_baseline != neutral_baseline:
        raise ValueError(f"{_format_key(key)} has mismatched branch metadata: baseline")


def _run_from_shared_checkpoint(
    key: tuple[str, str, str], record: Mapping[str, Any]
) -> FrustrationRun:
    baseline = _axis_median(_baseline_axis(record, key), "baseline.frustration")
    trajectory = record.get("pre_intervention_trajectory")
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        raise ValueError(f"{_format_key(key)} pre_intervention_trajectory must be a list")
    if not trajectory:
        raise ValueError(f"{_format_key(key)} pre_intervention_trajectory must not be empty")

    round_medians: list[float] = []
    violation_rounds = 0
    for round_number, step in enumerate(trajectory, start=1):
        if not isinstance(step, Mapping):
            raise ValueError(f"{_format_key(key)} trajectory round {round_number} must be an object")
        emotion = step.get("emotion_after_feedback")
        if not isinstance(emotion, Mapping):
            raise ValueError(
                f"{_format_key(key)} trajectory round {round_number} is missing emotion_after_feedback"
            )
        round_medians.append(
            _axis_median(
                emotion,
                f"trajectory round {round_number}.emotion_after_feedback.frustration",
            )
        )
        if _has_guess_rule_violation(step, key, round_number):
            violation_rounds += 1

    final = round_medians[-1]
    return FrustrationRun(
        run_id=key[0],
        persona_id=key[2],
        baseline=baseline,
        round_frustration_medians=tuple(round_medians),
        final_round=len(round_medians),
        baseline_to_final_delta=final - baseline,
        round1_to_final_delta=final - round_medians[0],
        round_slope=_least_squares_slope(round_medians),
        pre_intervention_guess_rule_violation_rounds=violation_rounds,
    )


def _baseline_axis(record: Mapping[str, Any], key: tuple[str, str, str]) -> Mapping[str, Any]:
    projections = record.get("emotion_projections")
    if not isinstance(projections, Mapping):
        raise ValueError(f"{_format_key(key)} emotion_projections must be an object")
    baseline = projections.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError(f"{_format_key(key)} is missing emotion_projections.baseline")
    return baseline


def _axis_median(container: Mapping[str, Any], label: str) -> float:
    axis = container.get("frustration")
    if not isinstance(axis, Mapping):
        raise ValueError(f"{label} is missing the frustration axis")
    value = _finite_number(axis.get("median"), f"{label}.median")
    layers = axis.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError(f"{label}.layers must be a non-empty object")
    for layer, layer_value in layers.items():
        _finite_number(layer_value, f"{label}.layers[{layer!r}]")
    return value


def _has_guess_rule_violation(
    step: Mapping[str, Any], key: tuple[str, str, str], round_number: int
) -> bool:
    violations = step.get("rule_violations")
    if not isinstance(violations, (list, tuple)):
        raise ValueError(
            f"{_format_key(key)} trajectory round {round_number}.rule_violations must be a list"
        )
    return any(not _is_willingness_violation(item) for item in violations)


def _is_willingness_violation(violation: Any) -> bool:
    if violation == "invalid_willingness_response":
        return True
    if isinstance(violation, Mapping):
        return any(
            violation.get(field) == "invalid_willingness_response"
            for field in ("type", "code", "kind")
        )
    return False


def _least_squares_slope(values: Sequence[float]) -> float | None:
    """Slope for round numbers 1..n; one point has no identifiable slope."""
    count = len(values)
    if count < 2:
        return None
    mean_x = (count + 1) / 2
    mean_y = sum(values) / count
    numerator = sum((round_number - mean_x) * (value - mean_y) for round_number, value in enumerate(values, 1))
    denominator = sum((round_number - mean_x) ** 2 for round_number in range(1, count + 1))
    return numerator / denominator


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _pair_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    try:
        return (str(record["run_id"]), str(record["checkpoint_id"]), str(record["persona_id"]))
    except KeyError as error:
        raise ValueError(f"record cannot be paired without {error.args[0]!r}") from error


def _group_by_persona(runs: Sequence[FrustrationRun]) -> dict[str, tuple[FrustrationRun, ...]]:
    grouped: dict[str, list[FrustrationRun]] = {}
    for run in runs:
        grouped.setdefault(run.persona_id, []).append(run)
    return {persona_id: tuple(persona_runs) for persona_id, persona_runs in sorted(grouped.items())}


def _format_key(key: tuple[str, str, str]) -> str:
    return f"(run_id={key[0]!r}, checkpoint_id={key[1]!r}, persona_id={key[2]!r})"
