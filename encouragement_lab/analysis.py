"""Pair and summarize completed experiment branches from versioned JSONL data."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

from encouragement_lab.personas import PERSONA_BY_ID, expected_template_ids
from encouragement_lab.records import iter_records


# These values describe the shared state at the fork.  Do not compare fields
# such as guess, feedback, condition, or the full post-intervention transcript:
# they are expected to diverge after branching.
CHECKPOINT_METADATA_FIELDS = (
    "schema_version",
    "run_id",
    "seed",
    "checkpoint_id",
    "model",
    "sampling",
    "persona_id",
    "prompt_checksum",
    "failure_rounds",
    "candidate_count_at_checkpoint",
    "pre_intervention_trajectory",
)
OPTIONAL_CHECKPOINT_METADATA_FIELDS = (
    "branch_seed",
    "checkpoint_rng_state",
    "checkpoint_transcript_checksum",
    "checkpoint_candidate_checksum",
    "persona_quadrant",
    "persona_template_id",
    "emotion_probe",
)
_METRIC_NAMES = (
    "normalized_information_efficiency",
    "willingness_to_continue",
    "positive_message_delta",
    "negative_message_delta",
    "frustration_message_delta",
    "positive_post_guess_delta",
    "negative_post_guess_delta",
    "frustration_post_guess_delta",
    "hard_rule_violation_rate",
)


@dataclass(frozen=True)
class PairedDelta:
    """One encouragement-minus-neutral comparison from one checkpoint."""

    run_id: str
    seed: int
    checkpoint_id: str
    persona_id: str
    persona_quadrant: str | None
    persona_template_id: str | None
    normalized_information_efficiency: float
    willingness_to_continue: float | None
    positive_message_delta: float
    negative_message_delta: float
    frustration_message_delta: float
    positive_post_guess_delta: float
    negative_post_guess_delta: float
    frustration_post_guess_delta: float
    hard_rule_violation_rate: float

    @property
    def normalized_information_efficiency_delta(self) -> float:
        """Explicit alias for the primary encouragement-minus-neutral metric."""
        return self.normalized_information_efficiency

    @property
    def willingness_to_continue_delta(self) -> float:
        """Explicit alias for the encouragement-minus-neutral willingness metric."""
        return self.willingness_to_continue


@dataclass(frozen=True)
class ScalarSummary:
    """A metric summarized across independent runs for one persona."""

    count: int
    mean: float | None
    sample_stdev: float | None


@dataclass(frozen=True)
class PersonaSummary:
    """Per-template summaries after within-run checkpoint averaging."""

    persona_id: str
    normalized_information_efficiency: ScalarSummary
    willingness_to_continue: ScalarSummary
    positive_message_delta: ScalarSummary
    negative_message_delta: ScalarSummary
    frustration_message_delta: ScalarSummary
    positive_post_guess_delta: ScalarSummary
    negative_post_guess_delta: ScalarSummary
    frustration_post_guess_delta: ScalarSummary
    hard_rule_violation_rate: ScalarSummary


@dataclass(frozen=True)
class QuadrantSummary:
    """Per-quadrant summaries after averaging templates within each seed."""

    quadrant_id: str
    template_ids: tuple[str, ...]
    normalized_information_efficiency: ScalarSummary
    willingness_to_continue: ScalarSummary
    positive_message_delta: ScalarSummary
    negative_message_delta: ScalarSummary
    frustration_message_delta: ScalarSummary
    positive_post_guess_delta: ScalarSummary
    negative_post_guess_delta: ScalarSummary
    frustration_post_guess_delta: ScalarSummary
    hard_rule_violation_rate: ScalarSummary


def analyze_jsonl(path: str | Path) -> list[PairedDelta]:
    """Read records only through :func:`records.iter_records` and pair them."""
    return pair_records(iter_records(path))


def pair_records(records: Iterable[Mapping[str, Any]]) -> list[PairedDelta]:
    """Strictly pair encouragement and neutral records by checkpoint identity.

    ``emotion_summary`` must expose per-direction deltas in either equivalent
    layout::

        {"message_delta": {"positive": 0.1, "negative": -0.2, "frustration": 0.4},
         "post_guess_delta": {"positive": 0.0, "negative": 0.3, "frustration": -0.1}}

    or::

        {"positive": {"message_delta": 0.1, "post_guess_delta": 0.0},
         "negative": {"message_delta": -0.2, "post_guess_delta": 0.3},
         "frustration": {"message_delta": 0.4, "post_guess_delta": -0.1}}
    """
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

    paired: list[PairedDelta] = []
    for key in sorted(grouped):
        branches = grouped[key]
        missing = {"encouragement", "neutral"}.difference(branches)
        if missing:
            raise ValueError(f"{_format_key(key)} is missing branch(es): {', '.join(sorted(missing))}")
        encouragement = branches["encouragement"]
        neutral = branches["neutral"]
        _validate_shared_checkpoint(key, encouragement, neutral)
        quadrant_id, template_id = _persona_metadata(encouragement)
        paired.append(
            PairedDelta(
                run_id=key[0],
                seed=_seed(encouragement),
                checkpoint_id=key[1],
                persona_id=key[2],
                persona_quadrant=quadrant_id,
                persona_template_id=template_id,
                normalized_information_efficiency=_difference(
                    encouragement, neutral, "normalized_information_efficiency"
                ),
                willingness_to_continue=_optional_difference(
                    encouragement, neutral, "willingness_to_continue"
                ),
                positive_message_delta=_emotion_difference(
                    encouragement, neutral, "positive", "message_delta"
                ),
                negative_message_delta=_emotion_difference(
                    encouragement, neutral, "negative", "message_delta"
                ),
                frustration_message_delta=_emotion_difference(
                    encouragement, neutral, "frustration", "message_delta"
                ),
                positive_post_guess_delta=_emotion_difference(
                    encouragement, neutral, "positive", "post_guess_delta"
                ),
                negative_post_guess_delta=_emotion_difference(
                    encouragement, neutral, "negative", "post_guess_delta"
                ),
                frustration_post_guess_delta=_emotion_difference(
                    encouragement, neutral, "frustration", "post_guess_delta"
                ),
                hard_rule_violation_rate=(
                    _hard_rule_violation(encouragement) - _hard_rule_violation(neutral)
                ),
            )
        )
    return paired


def summarize_by_persona(pairs: Iterable[PairedDelta]) -> dict[str, PersonaSummary]:
    """Summarize deltas per persona using independent runs as observations.

    Multiple checkpoints in one run are averaged first, so they never inflate
    sample count.  ``sample_stdev`` is ``None`` for a single independent run.
    """
    by_persona_run: dict[tuple[str, str], list[PairedDelta]] = {}
    for pair in pairs:
        by_persona_run.setdefault((pair.persona_id, pair.run_id), []).append(pair)

    values_by_persona: dict[str, dict[str, list[float]]] = {}
    for (persona_id, _run_id), run_pairs in by_persona_run.items():
        persona_values = values_by_persona.setdefault(
            persona_id, {metric: [] for metric in _METRIC_NAMES}
        )
        for metric in _METRIC_NAMES:
            available = [
                value
                for pair in run_pairs
                if (value := getattr(pair, metric)) is not None
            ]
            if available:
                persona_values[metric].append(mean(available))

    return {
        persona_id: PersonaSummary(
            persona_id=persona_id,
            **{metric: _summarize(values) for metric, values in metric_values.items()},
        )
        for persona_id, metric_values in sorted(values_by_persona.items())
    }


def summarize_by_quadrant(
    pairs: Iterable[PairedDelta],
) -> dict[str, QuadrantSummary]:
    """Average all three prompt templates within seed, then summarize seeds.

    A quadrant/seed block must contain exactly one complete ``v1``/``v2``/``v3``
    set.  This prevents the extra wording templates from silently tripling the
    nominal sample size or giving an incomplete template unequal weight.
    Multiple checkpoints from the same run are averaged before that template
    contributes to its seed block.
    """
    by_run: dict[tuple[str, int, str, str], list[PairedDelta]] = {}
    for pair in pairs:
        if pair.persona_quadrant is None or pair.persona_template_id is None:
            raise ValueError(
                f"persona {pair.persona_id!r} lacks quadrant/template metadata"
            )
        key = (
            pair.persona_quadrant,
            pair.seed,
            pair.persona_template_id,
            pair.run_id,
        )
        by_run.setdefault(key, []).append(pair)

    blocks: dict[tuple[str, int], dict[str, dict[str, float | None]]] = {}
    for (quadrant_id, seed, template_id, _run_id), run_pairs in by_run.items():
        template_values: dict[str, float | None] = {}
        for metric in _METRIC_NAMES:
            available = [
                value
                for pair in run_pairs
                if (value := getattr(pair, metric)) is not None
            ]
            template_values[metric] = mean(available) if available else None
        templates = blocks.setdefault((quadrant_id, seed), {})
        if template_id in templates:
            raise ValueError(
                f"quadrant {quadrant_id!r} seed {seed} has duplicate "
                f"template {template_id!r} runs"
            )
        templates[template_id] = template_values

    values_by_quadrant: dict[str, dict[str, list[float]]] = {}
    templates_by_quadrant: dict[str, tuple[str, ...]] = {}
    for (quadrant_id, seed), templates in sorted(blocks.items()):
        expected = expected_template_ids(quadrant_id)
        if set(templates) != set(expected):
            missing = sorted(set(expected).difference(templates))
            extra = sorted(set(templates).difference(expected))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("unexpected " + ", ".join(extra))
            raise ValueError(
                f"quadrant {quadrant_id!r} seed {seed} has incomplete template block: "
                + "; ".join(details)
            )
        templates_by_quadrant[quadrant_id] = expected
        quadrant_values = values_by_quadrant.setdefault(
            quadrant_id, {metric: [] for metric in _METRIC_NAMES}
        )
        for metric in _METRIC_NAMES:
            template_metric_values = [
                templates[template_id][metric] for template_id in expected
            ]
            if all(value is not None for value in template_metric_values):
                quadrant_values[metric].append(
                    mean(value for value in template_metric_values if value is not None)
                )

    return {
        quadrant_id: QuadrantSummary(
            quadrant_id=quadrant_id,
            template_ids=templates_by_quadrant[quadrant_id],
            **{metric: _summarize(values) for metric, values in metric_values.items()},
        )
        for quadrant_id, metric_values in sorted(values_by_quadrant.items())
    }


def _pair_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    try:
        return (str(record["run_id"]), str(record["checkpoint_id"]), str(record["persona_id"]))
    except KeyError as error:
        raise ValueError(f"record cannot be paired without {error.args[0]!r}") from error


def _seed(record: Mapping[str, Any]) -> int:
    value = record.get("seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer")
    return value


def _persona_metadata(record: Mapping[str, Any]) -> tuple[str | None, str | None]:
    quadrant_id = record.get("persona_quadrant")
    template_id = record.get("persona_template_id")
    if quadrant_id is not None or template_id is not None:
        if not isinstance(quadrant_id, str) or not quadrant_id:
            raise ValueError("persona_quadrant must be a nonempty string")
        if not isinstance(template_id, str) or not template_id:
            raise ValueError("persona_template_id must be a nonempty string")
        return quadrant_id, template_id
    spec = PERSONA_BY_ID.get(str(record.get("persona_id")))
    if spec is None:
        return None, None
    return spec.quadrant_id, spec.template_id


def _validate_shared_checkpoint(
    key: tuple[str, str, str], encouragement: Mapping[str, Any], neutral: Mapping[str, Any]
) -> None:
    for field in CHECKPOINT_METADATA_FIELDS:
        if encouragement.get(field) != neutral.get(field):
            raise ValueError(f"{_format_key(key)} has mismatched checkpoint metadata: {field}")
    for field in OPTIONAL_CHECKPOINT_METADATA_FIELDS:
        present_in_encouragement = field in encouragement
        present_in_neutral = field in neutral
        if present_in_encouragement != present_in_neutral or (
            present_in_encouragement and encouragement[field] != neutral[field]
        ):
            raise ValueError(f"{_format_key(key)} has mismatched checkpoint metadata: {field}")


def _difference(
    encouragement: Mapping[str, Any], neutral: Mapping[str, Any], field: str
) -> float:
    return _number(encouragement.get(field), field) - _number(neutral.get(field), field)


def _optional_difference(
    encouragement: Mapping[str, Any], neutral: Mapping[str, Any], field: str
) -> float | None:
    encouragement_value = encouragement.get(field)
    neutral_value = neutral.get(field)
    if encouragement_value is None or neutral_value is None:
        return None
    return _number(encouragement_value, field) - _number(neutral_value, field)


def _emotion_difference(
    encouragement: Mapping[str, Any], neutral: Mapping[str, Any], direction: str, phase: str
) -> float:
    label = f"emotion_summary.{direction}.{phase}"
    return _number(_emotion_value(encouragement, direction, phase), label) - _number(
        _emotion_value(neutral, direction, phase), label
    )


def _emotion_value(record: Mapping[str, Any], direction: str, phase: str) -> Any:
    summary = record.get("emotion_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("emotion_summary must be an object containing message and post-guess deltas")
    phase_values = summary.get(phase)
    if isinstance(phase_values, Mapping) and direction in phase_values:
        return phase_values[direction]
    direction_values = summary.get(direction)
    if isinstance(direction_values, Mapping) and phase in direction_values:
        return direction_values[phase]
    raise ValueError(f"emotion_summary is missing {direction}.{phase}")


def _hard_rule_violation(record: Mapping[str, Any]) -> float:
    """Return whether a branch has a Mastermind-guess rule violation.

    ``invalid_willingness_response`` belongs to the post-guess questionnaire,
    not to the game, and therefore never contributes to this metric.  All
    other recorded rule violations are treated as game-guess violations.
    """
    violations = record.get("rule_violations")
    if not isinstance(violations, (list, tuple)):
        raise ValueError("rule_violations must be a list or tuple")
    return float(any(not _is_invalid_willingness_response(item) for item in violations))


def _is_invalid_willingness_response(violation: Any) -> bool:
    if violation == "invalid_willingness_response":
        return True
    if isinstance(violation, Mapping):
        return any(
            violation.get(field) == "invalid_willingness_response"
            for field in ("type", "code", "kind")
        )
    return False


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _summarize(values: list[float]) -> ScalarSummary:
    if not values:
        return ScalarSummary(count=0, mean=None, sample_stdev=None)
    return ScalarSummary(
        count=len(values),
        mean=mean(values),
        sample_stdev=stdev(values) if len(values) > 1 else None,
    )


def _format_key(key: tuple[str, str, str]) -> str:
    return f"(run_id={key[0]!r}, checkpoint_id={key[1]!r}, persona_id={key[2]!r})"
