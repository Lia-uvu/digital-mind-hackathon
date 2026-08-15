"""Print preregistered pre-intervention frustration checks as JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from encouragement_lab.manipulation import analyze_frustration_jsonl


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="versioned experiment JSONL file")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (default: 2)")
    args = parser.parse_args(argv)

    result = analyze_frustration_jsonl(args.records)
    print(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False, indent=args.indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
