"""Pure Mastermind scoring and an Absurdle-style adversarial state.

Codes are always four decimal characters.  Candidate collections are immutable
tuples so an experiment checkpoint can safely be forked into paired branches.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, TypeAlias

import numpy as np

Code: TypeAlias = str
Feedback: TypeAlias = tuple[int, int]

CODE_LENGTH = 4
ALL_CODES: tuple[Code, ...] = tuple(f"{number:04d}" for number in range(10_000))
_ALL_CODE_DIGITS = np.asarray(
    [[ord(digit) - ord("0") for digit in code] for code in ALL_CODES],
    dtype=np.uint8,
)
_ALL_CODE_COUNTS = np.zeros((len(ALL_CODES), 10), dtype=np.uint8)
for _position in range(CODE_LENGTH):
    np.add.at(
        _ALL_CODE_COUNTS,
        (np.arange(len(ALL_CODES)), _ALL_CODE_DIGITS[:, _position]),
        1,
    )
_FEEDBACK_CODE_COUNT = (CODE_LENGTH + 1) ** 2


def is_valid_code(code: object) -> bool:
    """Return whether *code* is a four-character decimal Mastermind code."""
    return isinstance(code, str) and len(code) == CODE_LENGTH and code.isascii() and code.isdecimal()


def _require_code(code: object, *, name: str) -> Code:
    if not is_valid_code(code):
        raise ValueError(f"{name} must be exactly four ASCII decimal digits, got {code!r}")
    return code


def _normalise_candidates(candidates: Iterable[Code]) -> tuple[Code, ...]:
    result = tuple(candidates)
    if not result:
        raise ValueError("candidates must not be empty")
    for candidate in result:
        _require_code(candidate, name="candidate")
    if len(set(result)) != len(result):
        raise ValueError("candidates must not contain duplicates")
    return result


def feedback(secret: Code, guess: Code) -> Feedback:
    """Score *guess* against *secret* as ``(exact, misplaced)``.

    ``misplaced`` counts the multiset overlap after all exact-position matches
    have been removed, which is the standard Mastermind rule for repeats.
    """
    secret = _require_code(secret, name="secret")
    guess = _require_code(guess, name="guess")
    exact = sum(secret_digit == guess_digit for secret_digit, guess_digit in zip(secret, guess))
    shared_digits = sum((Counter(secret) & Counter(guess)).values())
    return exact, shared_digits - exact


def partition_candidates(candidates: Iterable[Code], guess: Code) -> dict[Feedback, tuple[Code, ...]]:
    """Group candidates by the feedback they would produce for *guess*."""
    candidate_tuple = _normalise_candidates(candidates)
    guess = _require_code(guess, name="guess")
    buckets: dict[Feedback, list[Code]] = {}
    for candidate in candidate_tuple:
        buckets.setdefault(feedback(candidate, guess), []).append(candidate)
    return {pattern: tuple(bucket) for pattern, bucket in buckets.items()}


def adversarial_feedback(candidates: Iterable[Code], guess: Code) -> Feedback:
    """Choose the Absurdle feedback that retains the largest candidate bucket.

    Equal-size buckets are resolved by lower total matches, then lower exact
    matches, then lexicographically smaller feedback tuple.
    """
    buckets = partition_candidates(candidates, guess)
    return min(
        buckets,
        key=lambda pattern: (-len(buckets[pattern]), sum(pattern), pattern[0], pattern),
    )


def filter_candidates(candidates: Iterable[Code], guess: Code, pattern: Feedback) -> tuple[Code, ...]:
    """Return candidates consistent with a known feedback *pattern*."""
    candidate_tuple = _normalise_candidates(candidates)
    guess = _require_code(guess, name="guess")
    if not _is_valid_feedback(pattern):
        raise ValueError(f"invalid feedback pattern: {pattern!r}")
    result = tuple(candidate for candidate in candidate_tuple if feedback(candidate, guess) == pattern)
    if not result:
        raise ValueError("feedback pattern is inconsistent with candidates and guess")
    return result


def _is_valid_feedback(pattern: object) -> bool:
    return (
        isinstance(pattern, tuple)
        and len(pattern) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= CODE_LENGTH for value in pattern)
        and sum(pattern) <= CODE_LENGTH
    )


def _largest_bucket_size(candidates: tuple[Code, ...], guess: Code) -> int:
    counts: dict[Feedback, int] = {}
    for candidate in candidates:
        pattern = feedback(candidate, guess)
        counts[pattern] = counts.get(pattern, 0) + 1
    return max(counts.values())


def information_efficiency(candidates: Iterable[Code], guess: Code) -> float:
    """Return ``1 - largest_feedback_bucket / candidate_count`` for *guess*."""
    candidate_tuple = _normalise_candidates(candidates)
    guess = _require_code(guess, name="guess")
    return 1.0 - _largest_bucket_size(candidate_tuple, guess) / len(candidate_tuple)


def optimal_information_efficiency(
    candidates: Iterable[Code], guesses: Iterable[Code] | None = None
) -> float:
    """Return the best raw efficiency over *guesses* (all legal codes by default)."""
    candidate_tuple = _normalise_candidates(candidates)
    guess_tuple = ALL_CODES if guesses is None else tuple(guesses)
    if not guess_tuple:
        raise ValueError("guesses must not be empty")
    for guess in guess_tuple:
        _require_code(guess, name="guess")
    largest_buckets = _largest_bucket_sizes_vectorized(candidate_tuple, guess_tuple)
    return 1.0 - int(largest_buckets.min()) / len(candidate_tuple)


def _largest_bucket_sizes_vectorized(
    candidates: tuple[Code, ...], guesses: tuple[Code, ...], *, batch_size: int = 128
) -> np.ndarray:
    """Return exact largest-bucket sizes for many guesses using bounded NumPy batches.

    The calculation is identical to repeated :func:`feedback` calls.  Batching
    removes Python's 10,000 × candidate-count inner loop without allocating the
    full all-guesses-by-all-candidates feedback matrix.
    """
    candidate_indices = np.fromiter(
        (int(code) for code in candidates), dtype=np.intp, count=len(candidates)
    )
    guess_indices = np.fromiter(
        (int(code) for code in guesses), dtype=np.intp, count=len(guesses)
    )
    candidate_digits = _ALL_CODE_DIGITS[candidate_indices]
    candidate_counts = _ALL_CODE_COUNTS[candidate_indices]
    largest = np.empty(len(guesses), dtype=np.int64)
    for start in range(0, len(guesses), batch_size):
        stop = min(start + batch_size, len(guesses))
        batch_indices = guess_indices[start:stop]
        guess_digits = _ALL_CODE_DIGITS[batch_indices]
        guess_counts = _ALL_CODE_COUNTS[batch_indices]
        exact = (guess_digits[:, None, :] == candidate_digits[None, :, :]).sum(
            axis=2
        )
        shared = np.minimum(
            guess_counts[:, None, :], candidate_counts[None, :, :]
        ).sum(axis=2)
        encoded = exact * (CODE_LENGTH + 1) + (shared - exact)
        offsets = np.arange(stop - start, dtype=np.int64)[:, None] * _FEEDBACK_CODE_COUNT
        bucket_counts = np.bincount(
            (encoded.astype(np.int64) + offsets).ravel(),
            minlength=(stop - start) * _FEEDBACK_CODE_COUNT,
        ).reshape(stop - start, _FEEDBACK_CODE_COUNT)
        largest[start:stop] = bucket_counts.max(axis=1)
    return largest


def normalized_information_efficiency(
    candidates: Iterable[Code], guess: Code, guesses: Iterable[Code] | None = None
) -> float:
    """Return raw efficiency divided by the position's optimal efficiency.

    A solved one-candidate state has no remaining information to gain, so its
    normalized efficiency is defined as ``0.0`` rather than dividing by zero.
    """
    candidate_tuple = _normalise_candidates(candidates)
    raw = information_efficiency(candidate_tuple, guess)
    optimum = optimal_information_efficiency(candidate_tuple, guesses)
    return 0.0 if optimum == 0.0 else raw / optimum


@dataclass(frozen=True)
class AbsurdleState:
    """An immutable candidate set for one point in an Absurdle game."""

    candidates: tuple[Code, ...] = ALL_CODES

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", _normalise_candidates(self.candidates))

    def copy(self) -> "AbsurdleState":
        """Return an independent state object with the same immutable candidates."""
        return AbsurdleState(tuple([*self.candidates]))

    def play(self, guess: Code) -> tuple[Feedback, "AbsurdleState"]:
        """Return adversarial feedback and the separate state after that feedback."""
        pattern = adversarial_feedback(self.candidates, guess)
        return pattern, AbsurdleState(filter_candidates(self.candidates, guess, pattern))
