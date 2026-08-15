"""Predeclared balanced 2 x 2 persona contrasts for formal experiment data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import sqrt
from statistics import mean, stdev
from typing import Iterable

from encouragement_lab.analysis import PairedDelta
from encouragement_lab.personas import PERSONA_QUADRANTS, expected_template_ids


PLANNED_CONTRASTS = ("extraversion", "neuroticism", "interaction")
FACTORIAL_METRICS = (
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
class SeedContrasts:
    """Four balanced quadrant values reduced to planned contrasts for one seed."""

    seed: int
    average_treatment: float
    extraversion: float
    neuroticism: float
    interaction: float


@dataclass(frozen=True)
class ContrastSummary:
    """Across-seed uncertainty without treating templates as extra samples."""

    count: int
    mean: float | None
    sample_stdev: float | None
    standard_error: float | None
    exact_sign_flip_p: float | None
    holm_adjusted_p: float | None = None


@dataclass(frozen=True)
class FactorialAnalysis:
    """Formal balanced analysis for one encouragement-minus-neutral metric."""

    metric: str
    by_seed: tuple[SeedContrasts, ...]
    omitted_seeds: tuple[int, ...]
    average_treatment: ContrastSummary
    extraversion: ContrastSummary
    neuroticism: ContrastSummary
    interaction: ContrastSummary


def analyze_factorial(
    pairs: Iterable[PairedDelta],
    *,
    metric: str = "normalized_information_efficiency",
) -> FactorialAnalysis:
    """Apply the frozen template-within-seed, then 2 x 2 contrast analysis.

    The four quadrant values are the means of complete ``v1``/``v2``/``v3``
    blocks.  The reported effects are:

    - extraversion: high-E marginal mean minus low-E marginal mean;
    - neuroticism: high-N marginal mean minus low-N marginal mean;
    - interaction: ``HH - HL - LH + LL`` (difference of differences).

    Exact sign-flip p-values use seed-level contrasts.  Holm adjustment covers
    the three persona contrasts; the average encouragement effect is reported
    separately and is not part of that family.
    """
    if metric not in FACTORIAL_METRICS:
        raise ValueError(f"unsupported factorial metric: {metric}")
    materialized = tuple(pairs)
    run_values: dict[tuple[int, str, str, str], list[float]] = {}
    for pair in materialized:
        if pair.persona_quadrant is None or pair.persona_template_id is None:
            raise ValueError(
                f"persona {pair.persona_id!r} lacks quadrant/template metadata"
            )
        key = (
            pair.seed,
            pair.persona_quadrant,
            pair.persona_template_id,
            pair.run_id,
        )
        value = getattr(pair, metric)
        if value is not None:
            run_values.setdefault(key, []).append(value)

    template_values: dict[tuple[int, str, str], float | None] = {}
    structural_keys = {
        (pair.seed, pair.persona_quadrant, pair.persona_template_id, pair.run_id)
        for pair in materialized
    }
    for seed, quadrant, template, run_id in structural_keys:
        assert quadrant is not None and template is not None
        short_key = (seed, quadrant, template)
        if short_key in template_values:
            raise ValueError(
                f"quadrant {quadrant!r} seed {seed} has duplicate template "
                f"{template!r} runs"
            )
        values = run_values.get((seed, quadrant, template, run_id), [])
        template_values[short_key] = mean(values) if values else None

    seeds = sorted({pair.seed for pair in materialized})
    quadrant_values: dict[tuple[int, str], float | None] = {}
    for seed in seeds:
        for quadrant in PERSONA_QUADRANTS:
            expected = expected_template_ids(quadrant)
            present = {
                template
                for value_seed, value_quadrant, template in template_values
                if value_seed == seed and value_quadrant == quadrant
            }
            if present != set(expected):
                missing = ", ".join(sorted(set(expected).difference(present)))
                raise ValueError(
                    f"quadrant {quadrant!r} seed {seed} has incomplete template "
                    f"block; missing {missing}"
                )
            values = [template_values[(seed, quadrant, template)] for template in expected]
            quadrant_values[(seed, quadrant)] = (
                mean(value for value in values if value is not None)
                if all(value is not None for value in values)
                else None
            )

    by_seed: list[SeedContrasts] = []
    omitted: list[int] = []
    for seed in seeds:
        values = {
            quadrant: quadrant_values[(seed, quadrant)]
            for quadrant in PERSONA_QUADRANTS
        }
        if any(value is None for value in values.values()):
            omitted.append(seed)
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
        name: _summarize(tuple(getattr(item, name) for item in by_seed))
        for name in ("average_treatment", *PLANNED_CONTRASTS)
    }
    adjusted = _holm_adjust(
        {name: summaries[name].exact_sign_flip_p for name in PLANNED_CONTRASTS}
    )
    for name in PLANNED_CONTRASTS:
        summaries[name] = replace(summaries[name], holm_adjusted_p=adjusted[name])
    return FactorialAnalysis(
        metric=metric,
        by_seed=tuple(by_seed),
        omitted_seeds=tuple(omitted),
        average_treatment=summaries["average_treatment"],
        extraversion=summaries["extraversion"],
        neuroticism=summaries["neuroticism"],
        interaction=summaries["interaction"],
    )


def _summarize(values: tuple[float, ...]) -> ContrastSummary:
    if not values:
        return ContrastSummary(0, None, None, None, None)
    deviation = stdev(values) if len(values) > 1 else None
    return ContrastSummary(
        count=len(values),
        mean=mean(values),
        sample_stdev=deviation,
        standard_error=(deviation / sqrt(len(values)) if deviation is not None else None),
        exact_sign_flip_p=_exact_sign_flip_p(values),
    )


def _exact_sign_flip_p(values: tuple[float, ...]) -> float:
    observed = abs(mean(values))
    extreme = 0
    total = 2 ** len(values)
    for signs in product((-1, 1), repeat=len(values)):
        permuted = abs(mean(value * sign for value, sign in zip(values, signs, strict=True)))
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / total


def _holm_adjust(p_values: dict[str, float | None]) -> dict[str, float | None]:
    available = sorted(
        ((name, value) for name, value in p_values.items() if value is not None),
        key=lambda item: item[1],
    )
    result = {name: None for name in p_values}
    running = 0.0
    count = len(available)
    for index, (name, value) in enumerate(available):
        running = max(running, min(1.0, (count - index) * value))
        result[name] = running
    return result
