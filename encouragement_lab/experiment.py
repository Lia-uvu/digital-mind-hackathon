"""Paired encouragement/neutral experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

from .mastermind import (
    AbsurdleState,
    adversarial_feedback,
    information_efficiency,
    is_valid_code,
    optimal_information_efficiency,
)
from .model import SamplingConfig, sampling_metadata
from .personas import PERSONA_KEYS, get_persona_spec
from .prompt_loader import load_prompts, validate_persona
from .records import ExperimentRecord, file_checksum

CONDITIONS = ("encouragement", "neutral")
_FOUR_DIGITS = re.compile(r"(?<!\d)\d{4}(?!\d)")


class ChatBackend(Protocol):
    def render_messages(
        self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = True
    ) -> str: ...

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int,
        sampling: SamplingConfig,
    ) -> str: ...

    def metadata(self) -> dict[str, Any]: ...


class EmotionBackend(Protocol):
    def score_text(self, rendered_text: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RunConfig:
    seed: int
    failure_rounds: int = 5
    sampling: SamplingConfig = SamplingConfig()
    track_round_emotions: bool = False


@dataclass(frozen=True)
class GuessParse:
    guess: str | None
    violations: tuple[str, ...]


@dataclass(frozen=True)
class Checkpoint:
    run_id: str
    checkpoint_id: str
    persona_id: str
    messages: tuple[dict[str, str], ...]
    trajectory: tuple[dict[str, Any], ...]
    state: AbsurdleState
    baseline_emotion: Mapping[str, Any]
    pre_intervention_emotion: Mapping[str, Any]


def parse_guess(text: str) -> GuessParse:
    """Parse strict JSON first, then salvage one unambiguous four-digit code."""
    violations: list[str] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and set(payload) == {"guess"}:
        guess = payload["guess"]
        if is_valid_code(guess):
            return GuessParse(guess=guess, violations=())
        violations.append("invalid_guess_value")
    else:
        violations.append("response_format")

    matches = _FOUR_DIGITS.findall(text)
    if len(matches) == 1 and is_valid_code(matches[0]):
        return GuessParse(guess=matches[0], violations=tuple(violations))
    violations.append("missing_or_ambiguous_guess")
    return GuessParse(guess=None, violations=tuple(dict.fromkeys(violations)))


def parse_willingness(text: str) -> int | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        value = payload.get("willingness")
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 10:
            return value
    numbers = [int(value) for value in re.findall(r"(?<!\d)(10|[1-9])(?!\d)", text)]
    return numbers[0] if len(numbers) == 1 else None


def derive_generation_seed(run_seed: int, stream: str, index: int = 0) -> int:
    """Derive non-overlapping reproducible RNG streams from one run seed."""
    payload = f"encouragement-lab-v1:{run_seed}:{stream}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _emotion_score(
    backend: ChatBackend,
    probe: EmotionBackend | None,
    messages: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    if probe is None:
        return {}
    return probe.score_text(backend.render_messages(messages))


def _emotion_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for axis in ("positive", "negative", "frustration"):
        if axis in before and axis in after:
            result[axis] = float(after[axis]["median"] - before[axis]["median"])
    return result


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "repeng", "numpy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


class ExperimentRunner:
    def __init__(
        self,
        backend: ChatBackend,
        prompts_path: str | Path,
        *,
        emotion_probe: EmotionBackend | None = None,
        emotion_probe_metadata: Mapping[str, Any] | None = None,
        code_version: str = "uncommitted",
    ):
        self.backend = backend
        self.prompts_path = Path(prompts_path)
        self.prompts = load_prompts(self.prompts_path)
        self.emotion_probe = emotion_probe
        self.emotion_probe_metadata = emotion_probe_metadata
        self.code_version = code_version
        self._validate_prompts()

    def _validate_prompts(self) -> None:
        required = {
            "system.base",
            "game.intro",
            "game.feedback",
            "game.invalid_guess",
            "condition.encouragement",
            "condition.neutral",
            "willingness",
            *PERSONA_KEYS,
        }
        missing = required.difference(self.prompts)
        if missing:
            raise ValueError("prompts.md missing keys: " + ", ".join(sorted(missing)))
        if re.search(
            r'"willingness"\s*:\s*(?:10|[1-9])', self.prompts["willingness"]
        ):
            raise ValueError(
                "willingness prompt must not anchor the response with a numeric JSON example"
            )
        for key in PERSONA_KEYS:
            validate_persona(self.prompts[key], name=key)

    def make_checkpoint(self, persona_id: str, config: RunConfig) -> Checkpoint:
        get_persona_spec(persona_id)
        if config.track_round_emotions and self.emotion_probe is None:
            raise ValueError("track_round_emotions requires a validated emotion probe")
        persona = self.prompts[persona_id].strip()
        system = self.prompts["system.base"].format(persona=persona).strip()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": self.prompts["game.intro"].strip()},
        ]
        state = AbsurdleState()
        trajectory: list[dict[str, Any]] = []
        baseline = _emotion_score(self.backend, self.emotion_probe, messages)
        failures = 0
        for round_index in range(config.failure_rounds):
            generation_seed = derive_generation_seed(
                config.seed, "pre_intervention", round_index
            )
            output = self.backend.generate(
                messages,
                seed=generation_seed,
                sampling=config.sampling,
            )
            messages.append({"role": "assistant", "content": output})
            parsed = parse_guess(output)
            failures += 1
            candidate_count_before = len(state.candidates)
            if parsed.guess is None:
                pattern = None
                raw_efficiency = 0.0
                next_prompt = self.prompts["game.invalid_guess"].format(
                    failure_count=failures
                )
            else:
                raw_efficiency = information_efficiency(
                    state.candidates, parsed.guess
                )
                pattern, next_state = state.play(parsed.guess)
                if pattern == (4, 0):
                    raise RuntimeError(
                        f"model won before checkpoint at round {round_index + 1}; "
                        "choose an earlier failure_rounds baseline"
                    )
                state = next_state
                next_prompt = self.prompts["game.feedback"].format(
                    guess=parsed.guess,
                    exact=pattern[0],
                    misplaced=pattern[1],
                    failure_count=failures,
                )
            messages.append({"role": "user", "content": next_prompt.strip()})
            round_emotion = (
                _emotion_score(self.backend, self.emotion_probe, messages)
                if config.track_round_emotions
                else {}
            )
            trajectory.append(
                {
                    "round": round_index + 1,
                    "generation_seed": generation_seed,
                    "raw_response": output,
                    "guess": parsed.guess,
                    "rule_violations": list(parsed.violations),
                    "candidate_count_before": candidate_count_before,
                    "candidate_count_after": len(state.candidates),
                    "feedback": list(pattern) if pattern is not None else None,
                    "raw_information_efficiency": raw_efficiency,
                    "emotion_after_feedback": round_emotion,
                }
            )

        pre_intervention = (
            trajectory[-1]["emotion_after_feedback"]
            if config.track_round_emotions
            else _emotion_score(self.backend, self.emotion_probe, messages)
        )
        run_id = f"{persona_id}.seed-{config.seed}"
        return Checkpoint(
            run_id=run_id,
            checkpoint_id=f"{run_id}.round-{config.failure_rounds}",
            persona_id=persona_id,
            messages=tuple(dict(message) for message in messages),
            trajectory=tuple(trajectory),
            state=state,
            baseline_emotion=baseline,
            pre_intervention_emotion=pre_intervention,
        )

    def run_branch(
        self,
        checkpoint: Checkpoint,
        condition: str,
        config: RunConfig,
        *,
        optimum: float | None = None,
    ) -> ExperimentRecord:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")
        messages = [dict(message) for message in checkpoint.messages]
        messages.append(
            {
                "role": "user",
                "content": self.prompts[f"condition.{condition}"].strip(),
            }
        )
        post_message = _emotion_score(self.backend, self.emotion_probe, messages)
        branch_seed = derive_generation_seed(config.seed, "branch_guess")
        output = self.backend.generate(
            messages,
            seed=branch_seed,
            sampling=config.sampling,
        )
        messages.append({"role": "assistant", "content": output})
        post_guess = _emotion_score(self.backend, self.emotion_probe, messages)
        parsed = parse_guess(output)

        candidates = checkpoint.state.candidates
        optimum = (
            optimal_information_efficiency(candidates)
            if optimum is None
            else optimum
        )
        if parsed.guess is None:
            raw_efficiency = 0.0
            normalized_efficiency = 0.0
            feedback_value: list[int] | None = None
        else:
            raw_efficiency = information_efficiency(candidates, parsed.guess)
            normalized_efficiency = (
                0.0 if optimum == 0.0 else raw_efficiency / optimum
            )
            feedback_value = list(adversarial_feedback(candidates, parsed.guess))

        messages.append(
            {"role": "user", "content": self.prompts["willingness"].strip()}
        )
        willingness_output = self.backend.generate(
            messages,
            seed=derive_generation_seed(config.seed, "willingness"),
            sampling=config.sampling,
        )
        messages.append({"role": "assistant", "content": willingness_output})
        willingness = parse_willingness(willingness_output)
        violations = list(parsed.violations)
        if willingness is None:
            violations.append("invalid_willingness_response")

        emotion_projections = {
            "baseline": checkpoint.baseline_emotion,
            "pre_intervention": checkpoint.pre_intervention_emotion,
            "post_message": post_message,
            "post_guess": post_guess,
        }
        emotion_summary = {
            "buildup_delta": _emotion_delta(
                checkpoint.baseline_emotion,
                checkpoint.pre_intervention_emotion,
            ),
            "message_delta": _emotion_delta(
                checkpoint.pre_intervention_emotion, post_message
            ),
            "post_guess_delta": _emotion_delta(
                checkpoint.pre_intervention_emotion, post_guess
            ),
        }
        persona_spec = get_persona_spec(checkpoint.persona_id)
        return ExperimentRecord(
            run_id=checkpoint.run_id,
            seed=config.seed,
            branch_seed=branch_seed,
            checkpoint_id=checkpoint.checkpoint_id,
            model=self.backend.metadata(),
            sampling=sampling_metadata(config.sampling),
            persona_id=checkpoint.persona_id,
            persona_quadrant=persona_spec.quadrant_id,
            persona_template_id=persona_spec.template_id,
            emotion_probe=self.emotion_probe_metadata,
            prompt_checksum=file_checksum(self.prompts_path),
            condition=condition,
            transcript=messages,
            pre_intervention_trajectory=checkpoint.trajectory,
            failure_rounds=config.failure_rounds,
            candidate_count_at_checkpoint=len(candidates),
            guess=parsed.guess or "",
            feedback=feedback_value,
            raw_information_efficiency=raw_efficiency,
            optimal_information_efficiency=optimum,
            normalized_information_efficiency=normalized_efficiency,
            rule_violations=violations,
            emotion_projections=emotion_projections,
            emotion_summary=emotion_summary,
            willingness_to_continue=willingness,
            code_version=self.code_version,
            dependency_versions=_dependency_versions(),
        )

    def run_pair(
        self, persona_id: str, config: RunConfig
    ) -> tuple[ExperimentRecord, ExperimentRecord]:
        checkpoint = self.make_checkpoint(persona_id, config)
        optimum = optimal_information_efficiency(checkpoint.state.candidates)
        return tuple(
            self.run_branch(checkpoint, condition, config, optimum=optimum)
            for condition in CONDITIONS
        )  # type: ignore[return-value]


class DryRunBackend:
    """Dependency-free deterministic backend for testing the whole pipeline."""

    def render_messages(
        self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = True
    ) -> str:
        return json.dumps(list(messages), ensure_ascii=False, sort_keys=True)

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int,
        sampling: SamplingConfig,
    ) -> str:
        if '"willingness"' in messages[-1]["content"]:
            return '{"willingness":7}'
        prior_guesses = sum(
            1
            for message in messages
            if message["role"] == "assistant" and '"guess"' in message["content"]
        )
        sequence = ("0123", "4567", "8901", "2345", "6789", "0246", "1357")
        return json.dumps({"guess": sequence[prior_guesses % len(sequence)]})

    def metadata(self) -> dict[str, Any]:
        return {"name": "dry-run", "model_type": "stub", "device": "none"}
