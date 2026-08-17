#!/usr/bin/env python3
"""Create the frozen three-concept Qwen probe from emotion-vector-bench."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from encouragement_lab.emotion_probe import EmotionDirections


SOURCE_COMMIT = "f6c84d65832608b4c4457f3f4b248774a42df940"
SOURCE_SHA256 = "612f780909fe8f3e75b5a65882037a4b48f3372ac0ee96060133dc7947020475"
SOURCE_REPOSITORY = "https://github.com/mufxio/emotion-vector-bench"
SOURCE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SOURCE_LAYER = 17
CONCEPTS = {
    "joyful": "joyful",
    "grief_stricken": "grief-stricken",
    "furious": "furious",
}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-checksum", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observed = checksum(args.source)
    if observed != SOURCE_SHA256:
        raise SystemExit(
            f"source SHA-256 mismatch: expected {SOURCE_SHA256}, got {observed}"
        )
    source = np.load(args.source, allow_pickle=False)
    directions: dict[str, dict[int, np.ndarray]] = {}
    for axis, source_concept in CONCEPTS.items():
        key = f"layer_{SOURCE_LAYER}__{source_concept}"
        vector = source[key].astype(np.float32)
        if vector.shape != (1536,) or not np.isfinite(vector).all():
            raise SystemExit(f"invalid external vector {key}: {vector.shape}")
        directions[axis] = {SOURCE_LAYER: vector}

    provenance = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_artifact_sha256": SOURCE_SHA256,
        "source_model": SOURCE_MODEL,
        "source_layer": SOURCE_LAYER,
        "source_variant": "denoised_vectors.npz",
        "source_concepts": CONCEPTS,
        "selection_rule": "best reported 20-way probe accuracy layer",
    }
    artifact = EmotionDirections(
        model_type="qwen2",
        model_checksum=args.model_checksum,
        materials_checksum=f"external:{SOURCE_COMMIT}:{SOURCE_SHA256}",
        directions=directions,
        validation={axis: dict(provenance) for axis in directions},
    )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    artifact.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
