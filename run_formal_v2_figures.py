#!/usr/bin/env python3
"""Render formal-v2 publication figures from a validated analysis bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from encouragement_lab.formal_v2_figures import write_figure_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_figure_bundle(args.analysis_dir, args.output_dir)
    print(
        f"wrote {len(manifest['artifacts_sha256'])} formal-v2 figure artifacts "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
