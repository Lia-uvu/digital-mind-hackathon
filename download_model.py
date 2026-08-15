#!/usr/bin/env python3
"""Download the official Qwen snapshot from ModelScope for local loading."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models" / "Qwen2.5-1.5B-Instruct"
    )
    args = parser.parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError as error:
        raise SystemExit("Install requirements.txt before downloading the model") from error

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(args.model, local_dir=str(destination))
    print(f"downloaded {args.model} to {resolved}")


if __name__ == "__main__":
    main()
