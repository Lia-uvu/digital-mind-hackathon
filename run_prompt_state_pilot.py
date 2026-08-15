#!/usr/bin/env python3
"""Measure intervention prompt-end state without generating a response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from encouragement_lab.emotion_probe import EmotionDirections, EmotionMaterials, EmotionProbe
from encouragement_lab.experiment import CONDITIONS
from encouragement_lab.model import LocalChatModel
from encouragement_lab.prompt_loader import load_prompts
from encouragement_lab.records import file_checksum
from run_response_pilot import load_checkpoints


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "results/formal-v1.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pause-prompt-state-v1.jsonl")
    parser.add_argument("--prompts", type=Path, default=ROOT / "response_pilot_prompts.md")
    parser.add_argument("--emotion-materials", type=Path, default=ROOT / "prompts.md")
    parser.add_argument("--emotion-directions", type=Path, default=ROOT / "artifacts/qwen2.5-1.5b-emotion-directions.npz")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--persona", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    interventions = load_prompts(args.prompts)
    if set(interventions) != {f"condition.{name}" for name in CONDITIONS}:
        raise SystemExit("prompt-state file must contain exactly two conditions")

    backend = LocalChatModel.load(args.model, args.device)
    directions = EmotionDirections.load(args.emotion_directions)
    materials = EmotionMaterials.from_prompts(load_prompts(args.emotion_materials))
    directions.assert_valid(materials, model_checksum=backend.snapshot_checksum)
    probe = EmotionProbe(backend.model, backend.tokenizer, directions)

    checkpoints = load_checkpoints(args.source)
    if args.seed:
        checkpoints = [row for row in checkpoints if row["seed"] in args.seed]
    if args.persona:
        checkpoints = [row for row in checkpoints if row["persona_id"] in args.persona]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for checkpoint in checkpoints:
            for condition in CONDITIONS:
                intervention = interventions[f"condition.{condition}"].strip()
                messages = [
                    *checkpoint["messages"],
                    {"role": "user", "content": intervention},
                ]
                prompt_end = probe.score_text(backend.render_messages(messages))
                delta = {
                    axis: prompt_end[axis]["median"]
                    - checkpoint["pre_intervention_emotion"][axis]["median"]
                    for axis in ("positive", "negative", "frustration")
                }
                record = {
                    "schema_version": 1,
                    **{
                        key: checkpoint[key]
                        for key in (
                            "checkpoint_id",
                            "run_id",
                            "seed",
                            "persona_id",
                            "persona_quadrant",
                            "persona_template_id",
                        )
                    },
                    "condition": condition,
                    "model": backend.metadata(),
                    "source_file": str(args.source),
                    "source_sha256": file_checksum(args.source),
                    "intervention_prompt_sha256": file_checksum(args.prompts),
                    "direction_artifact_sha256": file_checksum(args.emotion_directions),
                    "intervention": intervention,
                    "generated_response": None,
                    "pre_intervention_emotion": checkpoint["pre_intervention_emotion"],
                    "prompt_end_emotion": prompt_end,
                    "prompt_end_delta": delta,
                }
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                handle.flush()
                print(
                    checkpoint["run_id"],
                    condition,
                    f"frustration_delta={delta['frustration']:+.6f}",
                )


if __name__ == "__main__":
    main()
