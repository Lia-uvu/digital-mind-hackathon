"""Prompt-only factorial activation calibration for formal-v2 persona drafts.

This module deliberately never samples or generates text.  It checks whether
the controlled persona wording is legible in a common neutral chat endpoint;
it is not an emotion probe and does not make a claim about model experience.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .emotion_probe import AXES, DEFAULT_LAYERS
from .personas import PERSONA_SPECS, PersonaSpec


CALIBRATION_SUFFIX_KEYS = (
    "calibration.suffix.1",
    "calibration.suffix.2",
    "calibration.suffix.3",
)
CALIBRATION_LAYERS = DEFAULT_LAYERS


@dataclass(frozen=True)
class TokenAudit:
    """Token-count and fixed-span checks for one persona template block."""

    template_id: str
    sentence_token_counts: Mapping[str, tuple[int, ...]]
    persona_token_counts: Mapping[str, int]
    rendered_token_counts: Mapping[str, tuple[int, ...]]
    axis_difference_spans: Mapping[str, tuple[tuple[int, int], ...]]


def calibration_messages(persona: str, suffix: str) -> list[dict[str, str]]:
    """Build the only allowed calibration context: persona system + neutral user."""

    return [
        {"role": "system", "content": persona.strip()},
        {"role": "user", "content": suffix.strip()},
    ]


def calibration_personas(prompts: Mapping[str, str]) -> dict[str, str]:
    """Return exactly the registered 12 persona drafts from a prompt mapping."""

    missing = [spec.prompt_key for spec in PERSONA_SPECS if spec.prompt_key not in prompts]
    if missing:
        raise ValueError("calibration prompts missing personas: " + ", ".join(missing))
    return {spec.prompt_key: prompts[spec.prompt_key].strip() for spec in PERSONA_SPECS}


def calibration_suffixes(prompts: Mapping[str, str]) -> dict[str, str]:
    missing = [key for key in CALIBRATION_SUFFIX_KEYS if key not in prompts]
    if missing:
        raise ValueError("calibration prompts missing suffixes: " + ", ".join(missing))
    return {key: prompts[key].strip() for key in CALIBRATION_SUFFIX_KEYS}


def audit_persona_tokens(
    tokenizer: Any,
    personas: Mapping[str, str],
    suffixes: Mapping[str, str],
    render_messages: Callable[..., str],
) -> tuple[TokenAudit, ...]:
    """Verify equal persona/render lengths and fixed replacement spans.

    The caller supplies the final Qwen chat-template renderer.  This makes the
    same audit usable in a tokenizer-only command and in the full activation
    runner without loading a model twice.
    """

    audits: list[TokenAudit] = []
    for template_id in ("v1", "v2", "v3"):
        specs = [spec for spec in PERSONA_SPECS if spec.template_id == template_id]
        texts = {spec.prompt_key: personas[spec.prompt_key] for spec in specs}
        sentence_counts = {
            spec.prompt_key: tuple(
                len(_token_ids(tokenizer, sentence)) for sentence in _sentences(texts[spec.prompt_key])
            )
            for spec in specs
        }
        if len({len(value) for value in sentence_counts.values()}) != 1:
            raise ValueError(f"{template_id} has unequal sentence counts")
        if len(set(sentence_counts.values())) != 1:
            raise ValueError(f"{template_id} has unequal corresponding-sentence token counts")

        persona_ids = {spec.prompt_key: _token_ids(tokenizer, texts[spec.prompt_key]) for spec in specs}
        if len({len(value) for value in persona_ids.values()}) != 1:
            raise ValueError(f"{template_id} has unequal complete-persona token counts")

        rendered_counts: dict[str, tuple[int, ...]] = {}
        for spec in specs:
            rendered_counts[spec.prompt_key] = tuple(
                len(
                    _token_ids(
                        tokenizer,
                        render_messages(calibration_messages(texts[spec.prompt_key], suffix)),
                    )
                )
                for suffix in suffixes.values()
            )
        if len(set(rendered_counts.values())) != 1:
            raise ValueError(f"{template_id} has unequal rendered chat-template token counts")

        spans = {
            "extraversion": _axis_spans(specs, persona_ids, axis="extraversion"),
            "neuroticism": _axis_spans(specs, persona_ids, axis="neuroticism"),
        }
        audits.append(
            TokenAudit(
                template_id=template_id,
                sentence_token_counts=sentence_counts,
                persona_token_counts={key: len(value) for key, value in persona_ids.items()},
                rendered_token_counts=rendered_counts,
                axis_difference_spans=spans,
            )
        )
    return tuple(audits)


def render_calibration_inputs(
    personas: Mapping[str, str],
    suffixes: Mapping[str, str],
    render_messages: Callable[..., str],
) -> dict[str, dict[str, str]]:
    """Render common user-ending endpoints without generating any assistant text."""

    return {
        suffix_key: {
            persona_id: render_messages(calibration_messages(persona, suffix))
            for persona_id, persona in personas.items()
        }
        for suffix_key, suffix in suffixes.items()
    }


def evaluate_leave_one_template_out(
    hidden_by_suffix: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
) -> dict[str, Any]:
    """Evaluate E/N directions trained on two templates at the third template.

    ``hidden_by_suffix`` contains exactly one deterministic endpoint per
    suffix/persona/layer.  The returned pass decision only uses the six
    fold×axis medians across the three predeclared suffixes.  Existing emotion
    probes may be reported separately but cannot affect this decision.
    """

    _validate_hidden_inputs(hidden_by_suffix)
    suffix_results: dict[str, dict[str, Any]] = {}
    for suffix_key, hidden in hidden_by_suffix.items():
        suffix_results[suffix_key] = _evaluate_one_suffix(hidden)

    combined: list[dict[str, Any]] = []
    for heldout in ("v1", "v2", "v3"):
        for axis in ("extraversion", "neuroticism"):
            rows = [suffix_results[key][heldout][axis] for key in hidden_by_suffix]
            median_margin = float(median(row["median_margin"] for row in rows))
            combined.append(
                {
                    "heldout_template": heldout,
                    "axis": axis,
                    "suffix_median_margins": {
                        key: suffix_results[key][heldout][axis]["median_margin"]
                        for key in hidden_by_suffix
                    },
                    "median_across_suffixes": median_margin,
                    "correct": median_margin > 0,
                }
            )
    suffix_signs = [
        row["median_margin"] > 0
        for suffix in suffix_results.values()
        for fold in suffix.values()
        for row in fold.values()
    ]
    passed = all(row["correct"] for row in combined)
    robust = all(suffix_signs)
    return {
        "pass_rule": {
            "required_fold_axis_signs": 6,
            "observed_correct_fold_axis_signs": sum(row["correct"] for row in combined),
            "passed": passed,
            "suffix_specific_signs": len(suffix_signs),
            "suffix_specific_correct_signs": sum(suffix_signs),
            "robust": robust,
            "suffix_sensitive": passed and not robust,
        },
        "combined": combined,
        "by_suffix": suffix_results,
        "cross_template_cosines": _cross_template_cosines(hidden_by_suffix),
    }


def contamination_audit(
    hidden_by_suffix: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    directions: Mapping[str, Mapping[int, np.ndarray]],
    *,
    direction_layer_map: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Describe old emotion-probe shifts without permitting them to decide pass."""

    result: dict[str, Any] = {}
    layer_map = direction_layer_map or {layer: layer for layer in CALIBRATION_LAYERS}
    for axis in AXES:
        if axis not in directions:
            raise ValueError(f"contamination directions missing {axis}")
        by_suffix: dict[str, Any] = {}
        for suffix_key, hidden in hidden_by_suffix.items():
            scores = {
                persona: {
                    layer: _cosine(vector, directions[axis][layer_map[layer]])
                    for layer, vector in by_layer.items()
                }
                for persona, by_layer in hidden.items()
            }
            by_suffix[suffix_key] = _factorial_score_summary(scores)
        result[axis] = by_suffix
    return result


def token_audit_json(audits: Sequence[TokenAudit]) -> list[dict[str, Any]]:
    return [asdict(audit) for audit in audits]


def _sentences(text: str) -> tuple[str, ...]:
    parts = tuple(part.strip() + "." for part in text.strip().split(".") if part.strip())
    if not parts:
        raise ValueError("persona must contain at least one sentence")
    return parts


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return tuple(int(value) for value in values)


def _axis_spans(
    specs: Sequence[PersonaSpec], token_ids: Mapping[str, tuple[int, ...]], *, axis: str
) -> tuple[tuple[int, int], ...]:
    if axis == "extraversion":
        pairs = [
            ("high", "low", neuroticism)
            for neuroticism in ("high", "low")
        ]
        attr = "extraversion"
    elif axis == "neuroticism":
        pairs = [
            ("high", "low", extraversion)
            for extraversion in ("high", "low")
        ]
        attr = "neuroticism"
    else:
        raise ValueError(f"unknown axis {axis!r}")
    comparison_spans: list[tuple[tuple[int, int], ...]] = []
    for high, low, held in pairs:
        high_spec = next(
            spec for spec in specs
            if getattr(spec, attr) == high
            and getattr(spec, "neuroticism" if axis == "extraversion" else "extraversion") == held
        )
        low_spec = next(
            spec for spec in specs
            if getattr(spec, attr) == low
            and getattr(spec, "neuroticism" if axis == "extraversion" else "extraversion") == held
        )
        comparison_spans.append(
            _difference_spans(token_ids[high_spec.prompt_key], token_ids[low_spec.prompt_key])
        )
    if not comparison_spans:
        raise ValueError(f"{axis} has no fixed scalar-token differences")
    if len(set(comparison_spans)) != 1:
        raise ValueError(f"{axis} replacement spans are not at fixed token indices")
    return comparison_spans[0]


def _difference_spans(first: Sequence[int], second: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if len(first) != len(second):
        raise ValueError("replacement token spans differ in length")
    changed = [index for index, (left, right) in enumerate(zip(first, second)) if left != right]
    if not changed:
        raise ValueError("axis comparison has no token difference")
    spans: list[tuple[int, int]] = []
    start = previous = changed[0]
    for index in changed[1:]:
        if index != previous + 1:
            spans.append((start, previous + 1))
            start = index
        previous = index
    spans.append((start, previous + 1))
    return tuple(spans)


def _validate_hidden_inputs(hidden_by_suffix: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]]) -> None:
    if set(hidden_by_suffix) != set(CALIBRATION_SUFFIX_KEYS):
        raise ValueError("calibration must use exactly the three frozen suffixes")
    expected_personas = {spec.prompt_key for spec in PERSONA_SPECS}
    for suffix, rows in hidden_by_suffix.items():
        if set(rows) != expected_personas:
            raise ValueError(f"{suffix} does not contain the complete 12-persona block")
        for persona, by_layer in rows.items():
            if set(by_layer) != set(CALIBRATION_LAYERS):
                raise ValueError(f"{suffix} {persona} has wrong calibration layers")


def _evaluate_one_suffix(hidden: Mapping[str, Mapping[int, np.ndarray]]) -> dict[str, Any]:
    by_template = _template_specs()
    result: dict[str, Any] = {}
    for heldout, held_specs in by_template.items():
        training = [template for template in by_template if template != heldout]
        result[heldout] = {}
        for axis in ("extraversion", "neuroticism"):
            direction = {
                layer: np.mean(
                    [_main_effect(hidden, by_template[template], axis, layer) for template in training],
                    axis=0,
                )
                for layer in CALIBRATION_LAYERS
            }
            scores = {
                spec.prompt_key: {
                    layer: _cosine(hidden[spec.prompt_key][layer], direction[layer])
                    for layer in CALIBRATION_LAYERS
                }
                for spec in held_specs
            }
            margins = {
                layer: _main_effect(scores, held_specs, axis, layer)
                for layer in CALIBRATION_LAYERS
            }
            other_axis = "neuroticism" if axis == "extraversion" else "extraversion"
            cross_talk = {
                layer: _main_effect(scores, held_specs, other_axis, layer)
                for layer in CALIBRATION_LAYERS
            }
            interaction = {
                layer: _interaction(scores, held_specs, layer)
                for layer in CALIBRATION_LAYERS
            }
            result[heldout][axis] = {
                "per_layer_margin": {str(layer): float(margins[layer]) for layer in CALIBRATION_LAYERS},
                "median_margin": float(median(margins.values())),
                "per_layer_cross_talk": {str(layer): float(cross_talk[layer]) for layer in CALIBRATION_LAYERS},
                "per_layer_interaction": {str(layer): float(interaction[layer]) for layer in CALIBRATION_LAYERS},
            }
    return result


def _template_specs() -> dict[str, list[PersonaSpec]]:
    return {
        template: [spec for spec in PERSONA_SPECS if spec.template_id == template]
        for template in ("v1", "v2", "v3")
    }


def _main_effect(
    values: Mapping[str, Mapping[int, Any]], specs: Sequence[PersonaSpec], axis: str, layer: int
) -> Any:
    high = [values[spec.prompt_key][layer] for spec in specs if getattr(spec, axis) == "high"]
    low = [values[spec.prompt_key][layer] for spec in specs if getattr(spec, axis) == "low"]
    return (high[0] + high[1] - low[0] - low[1]) / 2


def _interaction(values: Mapping[str, Mapping[int, Any]], specs: Sequence[PersonaSpec], layer: int) -> Any:
    table = {(spec.extraversion, spec.neuroticism): values[spec.prompt_key][layer] for spec in specs}
    return table["high", "high"] - table["high", "low"] - table["low", "high"] + table["low", "low"]


def _cross_template_cosines(hidden_by_suffix: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]]) -> dict[str, Any]:
    templates = _template_specs()
    result: dict[str, Any] = {}
    for axis in ("extraversion", "neuroticism"):
        rows: dict[str, Any] = {}
        for suffix, hidden in hidden_by_suffix.items():
            contrasts = {
                template: {
                    layer: _main_effect(hidden, specs, axis, layer)
                    for layer in CALIBRATION_LAYERS
                }
                for template, specs in templates.items()
            }
            rows[suffix] = {
                f"{left}__{right}": {
                    str(layer): _cosine(contrasts[left][layer], contrasts[right][layer])
                    for layer in CALIBRATION_LAYERS
                }
                for left, right in combinations(templates, 2)
            }
        result[axis] = rows
    return result


def _factorial_score_summary(scores: Mapping[str, Mapping[int, float]]) -> dict[str, Any]:
    specs = list(PERSONA_SPECS)
    return {
        axis: {
            str(layer): float(_main_effect(scores, specs, axis, layer))
            for layer in CALIBRATION_LAYERS
        }
        for axis in ("extraversion", "neuroticism")
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not denom:
        raise ValueError("calibration direction has zero magnitude")
    return float(np.asarray(left) @ np.asarray(right) / denom)
