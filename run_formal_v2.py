#!/usr/bin/env python3
"""Dry-run-only CLI for the unfrozen formal-v2 candidate pipeline."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

from encouragement_lab.experiment import DryRunBackend
from encouragement_lab.formal_v2_records import append, resume_index
from encouragement_lab.formal_v2_runner import ARMS, FormalV2Runner
from encouragement_lab.model import SamplingConfig
from encouragement_lab.personas import PERSONA_KEYS


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = ROOT / "formal_v2_prompts.md"


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--persona", choices=PERSONA_KEYS, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit(
            "formal-v2 is not frozen: only the dependency-free --dry-run is permitted"
        )
    if args.output.exists() and args.output.stat().st_size and not args.resume:
        raise SystemExit(
            f"Output already contains records: {args.output}. "
            "Choose a new path or pass --resume for an exact matching dry run."
        )

    personas = args.persona or ["persona.high_e_high_n"]
    seeds = args.seed or [9001]
    backend = DryRunBackend()
    sampling = SamplingConfig()
    runner = FormalV2Runner(
        backend,
        args.prompts,
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
