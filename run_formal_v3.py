#!/usr/bin/env python3
"""Collect the frozen three-concept discrete-emotion replication."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

from encouragement_lab.emotion_probe import EmotionDirections, EmotionProbe
from encouragement_lab.experiment import DryRunBackend
from encouragement_lab.formal_v3_records import append, resume_index
from encouragement_lab.formal_v3_runner import ARMS, FormalV3Runner
from encouragement_lab.model import LocalChatModel, SamplingConfig
from encouragement_lab.personas import PERSONA_KEYS
from encouragement_lab.records import file_checksum


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = ROOT / "formal_v2_prompts.md"
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_DIRECTIONS = (
    ROOT / "artifacts" / "qwen2.5-1.5b-emotion-vector-bench-layer17.npz"
)
FORMAL_SEEDS = tuple(range(3001, 3011))
FROZEN_PROMPT_SHA256 = "ff6008a741668b1c90a44740e0573f29fa793fd039b102e744e65c9e85fa4136"
FROZEN_MODEL_SHA256 = "58c7c8cabbfb8a71eef25c14860b07338eb7b689063cc0fc19f08f4247c39a7e"
FROZEN_DIRECTIONS_SHA256 = "aca2a4806c5cb475455c2f914b26fe1fe107ed90a1f62b44a59121c6a54d6fc0"
FROZEN_AXES = ("furious", "grief_stricken", "joyful")
FROZEN_LAYER = 17
FORMAL_V3_FREEZE_ID: str | None = "formal-v3-2026-08-17"


def source_snapshot_checksum() -> str:
    files = [
        ROOT / "run_formal_v3.py",
        *sorted((ROOT / "encouragement_lab").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def code_version() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return f"{commit}+formal-v3-source-sha256:{source_snapshot_checksum()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nonformal-smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--persona", choices=PERSONA_KEYS, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--emotion-directions", type=Path, default=DEFAULT_DIRECTIONS)
    return parser.parse_args()


def _probe_metadata(probe: EmotionProbe, artifact: Path) -> dict:
    directions = probe.directions
    return {
        "artifact_name": artifact.name,
        "artifact_sha256": file_checksum(artifact),
        "model_checksum": directions.model_checksum,
        "materials_checksum": directions.materials_checksum,
        "axes": sorted(directions.directions),
        "layers": sorted({layer for values in directions.directions.values() for layer in values}),
        "external_source": directions.validation,
    }


def _real_backend(args: argparse.Namespace) -> tuple[LocalChatModel, EmotionProbe, dict]:
    backend = LocalChatModel.load(args.model, args.device)
    directions = EmotionDirections.load(args.emotion_directions)
    if directions.model_checksum != backend.snapshot_checksum:
        raise SystemExit("external emotion vectors target a different model snapshot")
    if tuple(sorted(directions.directions)) != FROZEN_AXES:
        raise SystemExit("external emotion artifact has unexpected concepts")
    if {layer for values in directions.directions.values() for layer in values} != {FROZEN_LAYER}:
        raise SystemExit("external emotion artifact has unexpected layers")
    probe = EmotionProbe(backend.model, backend.tokenizer, directions)
    return backend, probe, _probe_metadata(probe, args.emotion_directions)


def main() -> None:
    args = parse_args()
    modes = sum((args.dry_run, args.nonformal_smoke, args.formal))
    if modes != 1:
        raise SystemExit("choose exactly one run mode")
    if args.formal and FORMAL_V3_FREEZE_ID is None:
        raise SystemExit("formal-v3 is not dated-frozen; formal collection is locked")
    if args.formal and args.device != "mps":
        raise SystemExit("formal-v3 frozen runtime requires explicit --device mps")
    if args.formal and file_checksum(args.prompts) != FROZEN_PROMPT_SHA256:
        raise SystemExit("formal-v3 prompt source differs from the dated freeze")
    if args.formal and file_checksum(args.emotion_directions) != FROZEN_DIRECTIONS_SHA256:
        raise SystemExit("formal-v3 direction artifact differs from the dated freeze")
    if args.output.exists() and args.output.stat().st_size and not args.resume:
        raise SystemExit("output already contains records; choose a new path or --resume")

    personas = args.persona or (["persona.high_e_high_n"] if not args.formal else list(PERSONA_KEYS))
    seeds = args.seed or ([9002] if not args.formal else list(FORMAL_SEEDS))
    if args.formal and (tuple(seeds) != FORMAL_SEEDS or personas != list(PERSONA_KEYS)):
        raise SystemExit("formal mode requires all 12 personas and seeds 3001-3010")
    if args.nonformal_smoke and any(seed in FORMAL_SEEDS for seed in seeds):
        raise SystemExit("nonformal smoke must not use reserved formal seeds")

    if args.dry_run:
        backend, probe, metadata = DryRunBackend(), None, None
    else:
        backend, probe, metadata = _real_backend(args)
        if args.formal and backend.snapshot_checksum != FROZEN_MODEL_SHA256:
            raise SystemExit("formal-v3 model snapshot differs from the dated freeze")
    sampling = SamplingConfig()
    runner = FormalV3Runner(
        backend, args.prompts, probe=probe, probe_metadata=metadata,
        code_version=code_version(), seed_namespace="formal-v3",
    )
    scheduled = {(persona, seed, arm) for persona in personas for seed in seeds for arm in ARMS}
    completed = resume_index(
        args.output, expected_provenance=runner.provenance(sampling), scheduled=scheduled
    ) if args.resume else set()
    for persona in personas:
        for seed in seeds:
            for arm in ARMS:
                key = (persona, seed, arm)
                if key in completed:
                    print(f"skipped complete {persona}.seed-{seed}.{arm}")
                    continue
                record = runner.run(persona, seed, arm, sampling)
                append(args.output, record)
                print(f"wrote {record['run_id']}: status={record['status']}")


if __name__ == "__main__":
    main()
