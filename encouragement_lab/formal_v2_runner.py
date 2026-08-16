"""Independent candidate runner for the three-arm formal-v2 design."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .experiment import ChatBackend, parse_guess
from .mastermind import AbsurdleState, information_efficiency
from .model import SamplingConfig, sampling_metadata
from .personas import PERSONA_KEYS, get_persona_spec
from .prompt_loader import load_prompts, validate_persona
from .records import file_checksum, prompt_checksum


ARMS = ("feedback_only", "supportive", "neutral")
TARGET_FAILURES = 5


class PromptBoundaryProbe(Protocol):
    def score_text(self, rendered_text: str) -> dict[str, Any]: ...


def derive_generation_seed(seed: int, attempt_index: int) -> int:
    """Return a stable v2 seed shared across arms within a seed block."""

    payload = f"formal-v2:{seed}:guess:{attempt_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**63 - 1
    )


def _score_prompt_boundary(
    backend: ChatBackend,
    probe: PromptBoundaryProbe | None,
    messages: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Score the user-ending boundary immediately before an assistant reply."""

    if probe is None:
        return {}
    rendered = backend.render_messages(messages, add_generation_prompt=True)
    return probe.score_text(rendered)


def _candidate_state(codes: Sequence[str]) -> dict[str, Any]:
    """Store a compact, independently verifiable candidate-set identity."""

    digest = hashlib.sha256()
    for code in sorted(codes):
        digest.update(code.encode("ascii"))
        digest.update(b"\n")
    return {
        "candidate_count": len(codes),
        "candidate_set_sha256": digest.hexdigest(),
    }


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "torch", "transformers", "repeng"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


class FormalV2Runner:
    """Run one arm from the initial game state through five failures."""

    def __init__(
        self,
        backend: ChatBackend,
        prompts: str | Path,
        *,
        probe: PromptBoundaryProbe | None = None,
        probe_metadata: Mapping[str, Any] | None = None,
        code_version: str = "uncommitted",
    ):
        self.backend = backend
        self.path = Path(prompts)
        self.prompts = load_prompts(self.path)
        self.probe = probe
        self.probe_metadata = probe_metadata
        self.code_version = code_version
        self._validate_prompts()

    def _validate_prompts(self) -> None:
        required = {
            "system.base",
            "game.intro",
            "game.feedback",
            "game.invalid_guess",
            "game.counter",
            "game.final_instruction",
            *PERSONA_KEYS,
            *(
                f"filler.{arm}.{round_id}"
                for arm in ("supportive", "neutral")
                for round_id in range(1, TARGET_FAILURES + 1)
            ),
        }
        missing = required.difference(self.prompts)
        if missing:
            raise ValueError(
                "formal_v2_prompts.md missing keys: "
                + ", ".join(sorted(missing))
            )
        for key in PERSONA_KEYS:
            validate_persona(self.prompts[key], name=key)

    def _feedback_user(
        self,
        *,
        arm: str,
        failure_count: int,
        feedback: str,
    ) -> tuple[str, str | None, str | None]:
        parts = [
            feedback,
            self.prompts["game.counter"]
            .format(failure_count=failure_count)
            .strip(),
        ]
        filler_id: str | None = None
        filler_text: str | None = None
        if arm != "feedback_only":
            filler_id = f"filler.{arm}.{failure_count}"
            filler_text = self.prompts[filler_id].strip()
            parts.append(filler_text)
        parts.append(self.prompts["game.final_instruction"].strip())
        return "\n".join(parts), filler_id, filler_text

    def provenance(self, sampling: SamplingConfig) -> dict[str, Any]:
        """Return provenance shared by every run in one scheduled invocation."""

        return {
            "prompt_sha256": file_checksum(self.path),
            "model": self.backend.metadata(),
            "sampling": sampling_metadata(sampling),
            "emotion_probe": self.probe_metadata,
            "code_version": self.code_version,
            "dependency_versions": _dependency_versions(),
            "final_instruction": self.prompts["game.final_instruction"].strip(),
            "target_failure_rounds": TARGET_FAILURES,
            "engine_sha256": file_checksum(
                Path(__file__).with_name("mastermind.py")
            ),
        }

    def run(
        self,
        persona_id: str,
        seed: int,
        arm: str,
        sampling: SamplingConfig,
    ) -> dict[str, Any]:
        if arm not in ARMS:
            raise ValueError(f"unknown formal-v2 arm: {arm}")
        persona_spec = get_persona_spec(persona_id)
        persona_text = self.prompts[persona_id].strip()
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.prompts["system.base"]
                .format(persona=persona_text)
                .strip(),
            },
            {"role": "user", "content": self.prompts["game.intro"].strip()},
        ]
        state = AbsurdleState()
        baseline = _score_prompt_boundary(self.backend, self.probe, messages)
        trajectory: list[dict[str, Any]] = []
        completed_failures = 0
        status = "complete_five_failures"

        for attempt_index in range(1, TARGET_FAILURES + 1):
            generation_seed = derive_generation_seed(seed, attempt_index)
            output = self.backend.generate(
                messages,
                seed=generation_seed,
                sampling=sampling,
            )
            messages.append({"role": "assistant", "content": output})
            parsed = parse_guess(output)
            candidates_before = state.candidates

            if parsed.guess is None:
                pattern = None
                raw_efficiency = 0.0
                feedback_text = self.prompts["game.invalid_guess"].strip()
                outcome = "invalid"
            else:
                raw_efficiency = information_efficiency(
                    candidates_before, parsed.guess
                )
                pattern, next_state = state.play(parsed.guess)
                state = next_state
                if pattern == (4, 0):
                    trajectory.append(
                        {
                            "attempt_index": attempt_index,
                            "failure_count_after": completed_failures,
                            "generation_seed": generation_seed,
                            "raw_response": output,
                            "guess": parsed.guess,
                            "rule_violations": list(parsed.violations),
                            "outcome": "win",
                            "feedback": [4, 0],
                            "candidate_state_before": _candidate_state(
                                candidates_before
                            ),
                            "candidate_state_after": _candidate_state(
                                state.candidates
                            ),
                            "raw_information_efficiency": raw_efficiency,
                            "feedback_frame": "win",
                            "filler_id": None,
                            "filler_text": None,
                            "user_prompt": None,
                            "readout": None,
                        }
                    )
                    status = "early_win"
                    break
                feedback_text = self.prompts["game.feedback"].format(
                    guess=parsed.guess,
                    exact=pattern[0],
                    misplaced=pattern[1],
                ).strip()
                outcome = "unsuccessful"

            completed_failures += 1
            user_prompt, filler_id, filler_text = self._feedback_user(
                arm=arm,
                failure_count=completed_failures,
                feedback=feedback_text,
            )
            messages.append({"role": "user", "content": user_prompt})
            readout = _score_prompt_boundary(self.backend, self.probe, messages)
            trajectory.append(
                {
                    "attempt_index": attempt_index,
                    "failure_count_after": completed_failures,
                    "generation_seed": generation_seed,
                    "raw_response": output,
                    "guess": parsed.guess,
                    "rule_violations": list(parsed.violations),
                    "outcome": outcome,
                    "feedback": list(pattern) if pattern is not None else None,
                    "candidate_state_before": _candidate_state(
                        candidates_before
                    ),
                    "candidate_state_after": _candidate_state(state.candidates),
                    "raw_information_efficiency": raw_efficiency,
                    "feedback_frame": (
                        "invalid_guess" if pattern is None else "valid_feedback"
                    ),
                    "filler_id": filler_id,
                    "filler_text": filler_text,
                    "user_prompt": user_prompt,
                    "readout": readout,
                }
            )

        return {
            "schema_version": 1,
            "record_kind": "formal_v2_run",
            "run_id": f"{persona_id}.seed-{seed}.{arm}",
            "seed": seed,
            "arm": arm,
            "persona_id": persona_id,
            "persona_quadrant": persona_spec.quadrant_id,
            "persona_template_id": persona_spec.template_id,
            "persona_prompt_sha256": prompt_checksum(persona_text),
            "status": status,
            "completed_failure_rounds": completed_failures,
            "baseline_readout": baseline,
            "trajectory": trajectory,
            "transcript": messages,
            "final_candidate_state": _candidate_state(state.candidates),
            "provenance": self.provenance(sampling),
        }
