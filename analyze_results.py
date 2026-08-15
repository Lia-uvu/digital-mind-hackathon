#!/usr/bin/env python3
"""Print paired deltas plus prompt-template and balanced-quadrant summaries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from encouragement_lab.analysis import (
    analyze_jsonl,
    summarize_by_persona,
    summarize_by_quadrant,
)
from encouragement_lab.factorial import FACTORIAL_METRICS, analyze_factorial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--allow-incomplete-templates",
        action="store_true",
        help="emit template summaries without balanced quadrant summaries",
    )
    args = parser.parse_args()

    pairs = analyze_jsonl(args.input)
    persona_summaries = summarize_by_persona(pairs)
    quadrant_summaries = None
    factorial_analyses = None
    quadrant_note = None
    try:
        quadrant_summaries = summarize_by_quadrant(pairs)
        factorial_analyses = {
            metric: analyze_factorial(pairs, metric=metric)
            for metric in FACTORIAL_METRICS
        }
    except ValueError as error:
        if not args.allow_incomplete_templates:
            raise
        quadrant_note = str(error)
    print(
        json.dumps(
            {
                "paired_deltas": [asdict(pair) for pair in pairs],
                "persona_summaries": {
                    persona: asdict(summary)
                    for persona, summary in persona_summaries.items()
                },
                "quadrant_summaries": (
                    {
                        quadrant: asdict(summary)
                        for quadrant, summary in quadrant_summaries.items()
                    }
                    if quadrant_summaries is not None
                    else None
                ),
                "quadrant_summary_note": quadrant_note,
                "factorial_analyses": (
                    {
                        metric: asdict(analysis)
                        for metric, analysis in factorial_analyses.items()
                    }
                    if factorial_analyses is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
