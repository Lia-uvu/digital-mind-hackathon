#!/usr/bin/env python3
"""Run dry, pilot, or full paired-branch experiments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

from encouragement_lab.emotion_probe import (
    EmotionDirections,
    EmotionMaterials,
    EmotionProbe,
)
from encouragement_lab.experiment import (
    CONDITIONS,
    DryRunBackend,
    ExperimentRunner,
    RunConfig,
)
from encouragement_lab.model import LocalChatModel, SamplingConfig, sampling_metadata
from encouragement_lab.personas import PERSONA_KEYS
from encouragement_lab.prompt_loader import load_prompts
from encouragement_lab.records import append_record, file_checksum, iter_records


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"


def source_snapshot_checksum() -> str:
    """Hash the exact runner and package sources used to generate records."""
    files = [ROOT / "run_experiment.py", *sorted((ROOT / "encouragement_lab").glob("*.py"))]
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
        return f"{commit}+source-sha256:{source_snapshot_checksum()}"
    except (OSError, subprocess.CalledProcessError):
        return f"unknown+source-sha256:{source_snapshot_checksum()}"


def probe_metadata(probe: EmotionProbe | None, artifact: Path) -> dict | None:
    """Return compact immutable provenance for the hidden-state probe."""
    if probe is None:
        return None
    directions = probe.directions
    return {
        "artifact_name": artifact.name,
        "artifact_sha256": file_checksum(artifact),
        "model_checksum": directions.model_checksum,
        "materials_checksum": directions.materials_checksum,
        "axes": sorted(directions.directions),
        "layers": sorted(
            {layer for layers in directions.directions.values() for layer in layers}
        ),
        "heldout_aggregate_accuracy": {
            axis: values.get("aggregate_accuracy")
            for axis, values in directions.validation.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts.md")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "runs.jsonl")
    parser.add_argument("--persona", choices=PERSONA_KEYS, action="append")
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--failure-rounds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an exactly matching partial output without duplicating branches",
    )
    parser.add_argument("--no-emotion", action="store_true")
    parser.add_argument(
        "--track-round-emotions",
        action="store_true",
        help="record read-only emotion projections after every pre-intervention feedback",
    )
    parser.add_argument(
        "--emotion-directions",
        type=Path,
        default=ROOT / "artifacts" / "qwen2.5-1.5b-emotion-directions.npz",
    )
    parser.add_argument("--train-emotion-directions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.failure_rounds < 1:
        raise SystemExit("--failure-rounds must be positive")
    sampling = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.output.exists() and args.output.stat().st_size and not args.resume:
        raise SystemExit(
            f"Output already contains records: {args.output}. "
            "Choose a new path or pass --resume for an exact matching run."
        )

    if args.dry_run:
        backend = DryRunBackend()
        probe = None
    else:
        backend = LocalChatModel.load(args.model, args.device)
        probe = None
        if not args.no_emotion:
            if args.train_emotion_directions:
                prompts = load_prompts(args.prompts)
                materials = EmotionMaterials.from_prompts(prompts)

                def render_user_prompt(text: str) -> str:
                    return backend.render_messages(
                        [{"role": "user", "content": text}]
                    )

                probe = EmotionProbe.train(
                    backend.model,
                    backend.tokenizer,
                    materials,
                    render_user_prompt,
                    model_checksum=backend.snapshot_checksum,
                )
                probe.directions.save(args.emotion_directions)
            elif args.emotion_directions.exists():
                directions = EmotionDirections.load(args.emotion_directions)
                materials = EmotionMaterials.from_prompts(load_prompts(args.prompts))
                directions.assert_valid(
                    materials, model_checksum=backend.snapshot_checksum
                )
                probe = EmotionProbe(backend.model, backend.tokenizer, directions)
            else:
                raise SystemExit(
                    "No validated emotion directions found. Pass "
                    "--train-emotion-directions once, or use --no-emotion for a model-only pilot."
                )

    version = code_version()
    current_probe_metadata = probe_metadata(probe, args.emotion_directions)
    runner = ExperimentRunner(
        backend,
        args.prompts,
        emotion_probe=probe,
        emotion_probe_metadata=current_probe_metadata,
        code_version=version,
    )
    personas = args.persona or list(PERSONA_KEYS)
    seeds = args.seed or [20260815]
    completed = _resume_state(
        args.output,
        enabled=args.resume,
        personas=personas,
        seeds=seeds,
        failure_rounds=args.failure_rounds,
        sampling=sampling_metadata(sampling),
        prompt_hash=file_checksum(args.prompts),
        model_metadata=backend.metadata(),
        probe=current_probe_metadata,
        version=version,
    )
    for persona in personas:
        for seed in seeds:
            key = (persona, seed)
            if completed.get(key) == set(CONDITIONS):
                print(f"skipped complete {persona}.seed-{seed}")
                continue
            config = RunConfig(
                seed=seed,
                failure_rounds=args.failure_rounds,
                sampling=sampling,
                track_round_emotions=args.track_round_emotions,
            )
            for record in runner.run_pair(persona, config):
                if record.condition in completed.get(key, set()):
                    print(f"skipped existing {record.run_id} {record.condition}")
                    continue
                append_record(args.output, record)
                print(
                    f"wrote {record.run_id} {record.condition}: "
                    f"guess={record.guess!r} I_norm={record.normalized_information_efficiency:.3f}"
                )


def _resume_state(
    output: Path,
    *,
    enabled: bool,
    personas: list[str],
    seeds: list[int],
    failure_rounds: int,
    sampling: dict,
    prompt_hash: str,
    model_metadata: dict,
    probe: dict | None,
    version: str,
) -> dict[tuple[str, int], set[str]]:
    """Validate and index an exact partial output before resuming it."""
    if not enabled or not output.exists() or not output.stat().st_size:
        return {}
    scheduled = {(persona, seed) for persona in personas for seed in seeds}
    completed: dict[tuple[str, int], set[str]] = {}
    for record in iter_records(output):
        key = (str(record["persona_id"]), int(record["seed"]))
        if key not in scheduled:
            raise SystemExit(f"Resume output contains unscheduled run: {key}")
        expected = {
            "failure_rounds": failure_rounds,
            "sampling": sampling,
            "prompt_checksum": prompt_hash,
            "model": model_metadata,
            "emotion_probe": probe,
            "code_version": version,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise SystemExit(
                    f"Resume output metadata mismatch for {key}: {field}"
                )
        condition = str(record["condition"])
        conditions = completed.setdefault(key, set())
        if condition in conditions:
            raise SystemExit(f"Resume output has duplicate branch: {key} {condition}")
        conditions.add(condition)
    return completed


if __name__ == "__main__":
    main()
