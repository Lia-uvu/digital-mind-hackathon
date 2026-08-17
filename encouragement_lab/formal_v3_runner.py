"""Runner identity for the frozen discrete-emotion replication."""

from __future__ import annotations

from typing import Any

from .formal_v2_runner import ARMS, TARGET_FAILURES, FormalV2Runner
from .model import SamplingConfig


class FormalV3Runner(FormalV2Runner):
    """Reuse the unchanged three-arm game while emitting a new record kind."""

    def run(
        self, persona_id: str, seed: int, arm: str, sampling: SamplingConfig
    ) -> dict[str, Any]:
        record = super().run(persona_id, seed, arm, sampling)
        record["record_kind"] = "formal_v3_run"
        return record
