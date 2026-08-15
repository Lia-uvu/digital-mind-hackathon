#!/usr/bin/env python3
"""Reproducibly sample six records from a local Nemotron Personas Parquet file."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

SOURCE = "NVIDIA Nemotron-Personas-USA-Extended"
SOURCE_VERSION = "0.0.2"
SOURCE_URL = (
    "https://catalog.ngc.nvidia.com/orgs/nvidia/nemotron-personas/"
    "resources/nemotron-personas-dataset-en_us/-"
)
DEFAULT_INPUT = Path(__file__).parent / "data" / "en_US.parquet"
DEFAULT_OUTPUT = Path(__file__).parent / "nemotron_personas_6.json"
DEFAULT_MANIFEST = Path(__file__).parent / "selection_manifest.json"
SEED = 20260815
TARGET_COUNT = 6
MAX_CANDIDATES = 200

# Deliberately narrow: remove explicit response rules/direct concepts, while
# retaining indirect traits such as worried, competitive, or reserved.
FILTER_TERMS = (
    r"encourag",
    r"frustrat",
    r"give\s+up",
    r"giving\s+up",
    r"gave\s+up",
    r"confidence",
    r"confident",
    r"self[- ]?assur",
    r"鼓励",
    r"挫败",
    r"放弃",
    r"自信",
)
FILTER_RE = re.compile("|".join(FILTER_TERMS), re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def searchable_text(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


class IndexedParquet:
    """Read individual global rows without loading the 1.31 GB file at once."""

    def __init__(self, path: Path) -> None:
        self.file = pq.ParquetFile(path)
        totals: list[int] = []
        total = 0
        for group_index in range(self.file.num_row_groups):
            total += self.file.metadata.row_group(group_index).num_rows
            totals.append(total)
        self.group_ends = totals
        self.row_count = total
        self._cached_group_index: int | None = None
        self._cached_rows: list[dict] = []

    def row(self, global_index: int) -> dict:
        group_index = bisect.bisect_right(self.group_ends, global_index)
        group_start = 0 if group_index == 0 else self.group_ends[group_index - 1]
        if group_index != self._cached_group_index:
            self._cached_rows = self.file.read_row_group(group_index).to_pylist()
            self._cached_group_index = group_index
        return self._cached_rows[global_index - group_start]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    dataset = IndexedParquet(args.input)
    rng = random.Random(SEED)
    candidate_indices = rng.sample(range(dataset.row_count), MAX_CANDIDATES)
    selected: list[dict] = []
    audit: list[dict] = []

    for index in candidate_indices:
        row = dataset.row(index)
        hits = sorted(
            {match.group(0).lower() for match in FILTER_RE.finditer(searchable_text(row))}
        )
        if hits:
            audit.append(
                {"row_index": index, "status": "rejected", "matched_terms": hits}
            )
            continue

        selected.append({"row_index": index, **row})
        audit.append({"row_index": index, "status": "selected"})
        if len(selected) == TARGET_COUNT:
            break

    if len(selected) != TARGET_COUNT:
        raise RuntimeError(
            f"Selected {len(selected)} of {TARGET_COUNT} requested personas"
        )

    result = {
        "source": SOURCE,
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
        "input": {
            "filename": args.input.name,
            "sha256": sha256(args.input),
            "row_count": dataset.row_count,
        },
        "sampling": {
            "method": "first passing rows in a seeded random permutation",
            "seed": SEED,
            "target_count": TARGET_COUNT,
        },
        "filter": {
            "policy": (
                "Exclude explicit response rules/direct encouragement, frustration, "
                "giving-up, and confidence terms; retain indirect personality traits."
            ),
            "regex_terms": list(FILTER_TERMS),
            "scope": "all record fields",
        },
        "personas": selected,
        "audit": audit,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": SOURCE,
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
        "input_filename": args.input.name,
        "input_sha256": result["input"]["sha256"],
        "input_row_count": dataset.row_count,
        "seed": SEED,
        "filter_terms": list(FILTER_TERMS),
        "filter_scope": "all record fields",
        "selected_row_indices": [row["row_index"] for row in selected],
        "private_output_sha256": sha256(args.output),
        "note": (
            "The complete selected records remain local because the NVIDIA "
            "Dataset License Agreement prohibits redistribution in whole or in part."
        ),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(selected)} personas to {args.output}")
    print(f"Wrote public reproducibility manifest to {args.manifest}")


if __name__ == "__main__":
    main()
