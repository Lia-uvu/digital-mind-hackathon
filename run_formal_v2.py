#!/usr/bin/env python3
"""CLI for formal-v2 dry runs, nonformal real-backend smoke, and collection."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

from encouragement_lab.emotion_probe import EmotionDirections, EmotionMaterials, EmotionProbe
from encouragement_lab.experiment import DryRunBackend
from encouragement_lab.formal_v2_records import append, resume_index
from encouragement_lab.formal_v2_runner import ARMS, FormalV2Runner
from encouragement_lab.model import LocalChatModel, SamplingConfig
from encouragement_lab.personas import PERSONA_KEYS
from encouragement_lab.prompt_loader import load_prompts
from encouragement_lab.records import file_checksum


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = ROOT / "formal_v2_prompts.md"
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_DIRECTIONS = ROOT / "artifacts" / "qwen2.5-1.5b-emotion-directions.npz"
FORMAL_SEEDS = tuple(range(2001, 2011))
FROZEN_PROMPT_SHA256 = "ff6008a741668b1c90a44740e0573f29fa793fd039b102e744e65c9e85fa4136"
FROZEN_MODEL_SHA256 = "58c7c8cabbfb8a71eef25c14860b07338eb7b689063cc0fc19f08f4247c39a7e"
FROZEN_DIRECTIONS_SHA256 = "8034145a877afb36dc4bf8bde8afc208da3268866d8335ee393bed543e5250f2"
# Set only in the dated freeze commit. Until then formal collection is impossible.
FORMAL_V2_FREEZE_ID: str | None = "formal-v2-2026-08-16"


def source_snapshot_checksum() -> str:
    """Hash the exact CLI and package sources that can generate a v2 record."""

    files = [
        ROOT / "run_formal_v2.py",
        *sorted((ROOT / "encouragement_lab").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def code_version() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return f"{commit}+formal-v2-source-sha256:{source_snapshot_checksum()}"


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
        "heldout_aggregate_accuracy": {
            axis: values.get("aggregate_accuracy")
            for axis, values in directions.validation.items()
        },
    }


def _real_backend(args: argparse.Namespace) -> tuple[LocalChatModel, EmotionProbe, dict]:
    backend = LocalChatModel.load(args.model, args.device)
    directions = EmotionDirections.load(args.emotion_directions)
    # The frozen directions were trained from the original probe materials,
    # not from the v2 runtime prompt document.
    materials = EmotionMaterials.from_prompts(load_prompts(ROOT / "prompts.md"))
    directions.assert_valid(materials, model_checksum=backend.snapshot_checksum)
    probe = EmotionProbe(backend.model, backend.tokenizer, directions)
    return backend, probe, _probe_metadata(probe, args.emotion_directions)


def main() -> None:
    args = parse_args()
    modes = sum((args.dry_run, args.nonformal_smoke, args.formal))
    if modes != 1:
        raise SystemExit("choose exactly one of --dry-run, --nonformal-smoke, or --formal")
    if args.formal and FORMAL_V2_FREEZE_ID is None:
        raise SystemExit("formal-v2 is not dated-frozen; formal collection is locked")
    if args.formal and args.device != "mps":
        raise SystemExit("formal-v2 frozen runtime requires explicit --device mps")
    if args.formal and file_checksum(args.prompts) != FROZEN_PROMPT_SHA256:
        raise SystemExit("formal-v2 prompt source differs from the dated freeze")
    if args.formal and file_checksum(args.emotion_directions) != FROZEN_DIRECTIONS_SHA256:
        raise SystemExit("formal-v2 direction artifact differs from the dated freeze")
    if args.output.exists() and args.output.stat().st_size and not args.resume:
        raise SystemExit(
            f"Output already contains records: {args.output}. "
            "Choose a new path or pass --resume for an exact matching dry run."
        )

    personas = args.persona or (["persona.high_e_high_n"] if not args.formal else list(PERSONA_KEYS))
    seeds = args.seed or ([9001] if not args.formal else list(FORMAL_SEEDS))
    if args.formal and (tuple(seeds) != FORMAL_SEEDS or personas != list(PERSONA_KEYS)):
        raise SystemExit("formal mode requires all 12 personas and exact seeds 2001-2010")
    if args.nonformal_smoke and any(seed in FORMAL_SEEDS for seed in seeds):
        raise SystemExit("nonformal smoke must not use reserved formal seeds 2001-2010")
    if args.dry_run:
        backend = DryRunBackend()
        probe = None
        current_probe_metadata = None
    else:
        backend, probe, current_probe_metadata = _real_backend(args)
        if args.formal and backend.snapshot_checksum != FROZEN_MODEL_SHA256:
            raise SystemExit("formal-v2 model snapshot differs from the dated freeze")
    sampling = SamplingConfig()
    runner = FormalV2Runner(
        backend,
        args.prompts,
        probe=probe,
        probe_metadata=current_probe_metadata,
        code_version=code_version(),
    )
    expected_provenance = runner.provenance(sampling)
    scheduled = {
        (persona, seed, arm)
        for persona in personas
        for seed in seeds
        for arm in ARMS
    }
    completed = (
        resume_index(
            args.output,
            expected_provenance=expected_provenance,
            scheduled=scheduled,
        )
        if args.resume
        else set()
    )

    for persona in personas:
        for seed in seeds:
            for arm in ARMS:
                key = (persona, seed, arm)
                if key in completed:
                    print(f"skipped complete {persona}.seed-{seed}.{arm}")
                    continue
                record = runner.run(persona, seed, arm, sampling)
                append(args.output, record)
                print(
                    f"wrote {record['run_id']}: "
                    f"status={record['status']} "
                    f"failures={record['completed_failure_rounds']}"
                )


if __name__ == "__main__":
    main()
