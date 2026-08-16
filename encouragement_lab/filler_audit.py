"""Candidate-only Qwen token audit for formal-v2 filler pairs.

This module is deliberately independent of the formal-v2 runner: it cannot
start a game, generate a guess, or write collection data.  It verifies only
the paired wording proposed for a later human/freeze review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Callable, Mapping

from .persona_calibration import _token_ids


FINAL_INSTRUCTION = (
    'To continue, Reply with exactly one JSON object in the form '
    '{"guess":"0123"} containing your next guess.'
)
VALID_FEEDBACK = (
    "Feedback for 0123: 0 digit(s) are in the correct position and 0 additional "
    "digit(s) are correct but in the wrong position."
)
INVALID_GUESS_FEEDBACK = (
    "Your previous response did not contain one valid four-digit guess. "
    "It counts as an unsuccessful turn."
)
PAIR_IDS = ("1", "2", "3", "4", "5")
_FORBIDDEN = (
    "continue", "keep", "persist", "try", "guess", "strategy", "should", "could", "next", "again",
)


@dataclass(frozen=True)
class FillerPairAudit:
    round_id: str
    supportive_tokens: int
    neutral_tokens: int
    valid_feedback_turn_supportive_tokens: int
    valid_feedback_turn_neutral_tokens: int
    invalid_guess_turn_supportive_tokens: int
    invalid_guess_turn_neutral_tokens: int
    rendered_valid_token_counts: Mapping[str, tuple[int, int]]
    rendered_invalid_token_counts: Mapping[str, tuple[int, int]]
    filler_difference_spans: tuple[tuple[int, int], ...]


def candidate_pairs(prompts: Mapping[str, str]) -> dict[str, tuple[str, str]]:
    """Load exactly five ordered candidate pairs and reject accidental extras."""

    pairs: dict[str, tuple[str, str]] = {}
    for round_id in PAIR_IDS:
        support = f"filler.supportive.{round_id}"
        neutral = f"filler.neutral.{round_id}"
        if support not in prompts or neutral not in prompts:
            raise ValueError(f"missing candidate filler pair {round_id}")
        pairs[round_id] = (prompts[support].strip(), prompts[neutral].strip())
    found = {key for key in prompts if key.startswith("filler.")}
    expected = {f"filler.{arm}.{round_id}" for arm in ("supportive", "neutral") for round_id in PAIR_IDS}
    if found != expected:
        raise ValueError("candidate document must contain exactly five supportive/neutral pairs")
    return pairs


def validate_candidate_language(pairs: Mapping[str, tuple[str, str]]) -> None:
    """Guard against direct continuation/response/strategy cues in either arm."""

    for round_id, texts in pairs.items():
        for arm, text in zip(("supportive", "neutral"), texts):
            lower = text.lower()
            hits = [term for term in _FORBIDDEN if re.search(rf"\b{re.escape(term)}\b", lower)]
            if hits:
                raise ValueError(f"{arm} filler {round_id} contains prohibited cue(s): {', '.join(hits)}")


def feedback_turn(filler: str, *, round_id: str, frame: str) -> str:
    """Return an exact protocol feedback frame with its actual candidate round."""

    if frame == "valid_feedback":
        feedback = VALID_FEEDBACK
    elif frame == "invalid_guess":
        feedback = INVALID_GUESS_FEEDBACK
    else:
        raise ValueError(f"unknown feedback frame: {frame}")

    return "\n".join(
        (
            feedback,
            f"Consecutive unsuccessful guesses: {round_id}.",
            filler,
            FINAL_INSTRUCTION,
        )
    )


def audit_filler_pairs(
    tokenizer: Any,
    pairs: Mapping[str, tuple[str, str]],
    personas: Mapping[str, str],
    render_messages: Callable[..., str],
) -> tuple[FillerPairAudit, ...]:
    """Audit raw, full-turn, and chat-rendered counts for every candidate pair."""

    validate_candidate_language(pairs)
    audits: list[FillerPairAudit] = []
    for round_id in PAIR_IDS:
        supportive, neutral = pairs[round_id]
        support_ids, neutral_ids = _token_ids(tokenizer, supportive), _token_ids(tokenizer, neutral)
        valid_support_turn = feedback_turn(supportive, round_id=round_id, frame="valid_feedback")
        valid_neutral_turn = feedback_turn(neutral, round_id=round_id, frame="valid_feedback")
        invalid_support_turn = feedback_turn(supportive, round_id=round_id, frame="invalid_guess")
        invalid_neutral_turn = feedback_turn(neutral, round_id=round_id, frame="invalid_guess")
        valid_support_ids, valid_neutral_ids = _token_ids(tokenizer, valid_support_turn), _token_ids(tokenizer, valid_neutral_turn)
        invalid_support_ids, invalid_neutral_ids = _token_ids(tokenizer, invalid_support_turn), _token_ids(tokenizer, invalid_neutral_turn)
        if len(support_ids) != len(neutral_ids):
            raise ValueError(f"round {round_id} filler token counts differ")
        if len(valid_support_ids) != len(valid_neutral_ids):
            raise ValueError(f"round {round_id} valid-feedback token counts differ")
        if len(invalid_support_ids) != len(invalid_neutral_ids):
            raise ValueError(f"round {round_id} invalid-guess token counts differ")
        rendered_valid: dict[str, tuple[int, int]] = {}
        rendered_invalid: dict[str, tuple[int, int]] = {}
        for persona_id, persona in personas.items():
            valid_support_rendered = render_messages([
                {"role": "system", "content": persona.strip()},
                {"role": "user", "content": valid_support_turn},
            ])
            valid_neutral_rendered = render_messages([
                {"role": "system", "content": persona.strip()},
                {"role": "user", "content": valid_neutral_turn},
            ])
            invalid_support_rendered = render_messages([
                {"role": "system", "content": persona.strip()},
                {"role": "user", "content": invalid_support_turn},
            ])
            invalid_neutral_rendered = render_messages([
                {"role": "system", "content": persona.strip()},
                {"role": "user", "content": invalid_neutral_turn},
            ])
            valid_counts = (len(_token_ids(tokenizer, valid_support_rendered)), len(_token_ids(tokenizer, valid_neutral_rendered)))
            invalid_counts = (len(_token_ids(tokenizer, invalid_support_rendered)), len(_token_ids(tokenizer, invalid_neutral_rendered)))
            if valid_counts[0] != valid_counts[1]:
                raise ValueError(f"round {round_id} valid rendered counts differ for {persona_id}")
            if invalid_counts[0] != invalid_counts[1]:
                raise ValueError(f"round {round_id} invalid rendered counts differ for {persona_id}")
            rendered_valid[persona_id] = valid_counts
            rendered_invalid[persona_id] = invalid_counts
        audits.append(FillerPairAudit(
            round_id=round_id,
            supportive_tokens=len(support_ids), neutral_tokens=len(neutral_ids),
            valid_feedback_turn_supportive_tokens=len(valid_support_ids), valid_feedback_turn_neutral_tokens=len(valid_neutral_ids),
            invalid_guess_turn_supportive_tokens=len(invalid_support_ids), invalid_guess_turn_neutral_tokens=len(invalid_neutral_ids),
            rendered_valid_token_counts=rendered_valid,
            rendered_invalid_token_counts=rendered_invalid,
            filler_difference_spans=_difference_spans(support_ids, neutral_ids),
        ))
    return tuple(audits)


def audit_json(audits: tuple[FillerPairAudit, ...]) -> list[dict[str, Any]]:
    return [asdict(audit) for audit in audits]


def _difference_spans(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id and start is None:
            start = index
        elif left_id == right_id and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(left)))
    return tuple(spans)
