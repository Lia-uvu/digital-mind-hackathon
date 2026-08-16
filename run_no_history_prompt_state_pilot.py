#!/usr/bin/env python3
"""Read pause messages after a persona alone, without game history or generation.

The formal seed list is retained as a balanced record index.  This runner does
not sample or generate, so a seed is not a stochastic replicate here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from encouragement_lab.emotion_probe import EmotionDirections, EmotionMaterials, EmotionProbe
from encouragement_lab.experiment import CONDITIONS
from encouragement_lab.model import LocalChatModel
from encouragement_lab.personas import PERSONA_SPECS
from encouragement_lab.prompt_loader import load_prompts
from encouragement_lab.records import file_checksum


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
FORMAL_SEEDS = tuple(range(1001, 1011))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/no-history-pause-prompt-state-v1.jsonl",
    )
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts.md")
    parser.add_argument(
        "--interventions", type=Path, default=ROOT / "response_pilot_prompts.md"
    )
    parser.add_argument("--emotion-materials", type=Path, default=ROOT / "prompts.md")
    parser.add_argument(
        "--emotion-directions",
        type=Path,
        default=ROOT / "artifacts/qwen2.5-1.5b-emotion-directions.npz",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--persona", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    prompts = load_prompts(args.prompts)
    interventions = load_prompts(args.interventions)
    if set(interventions) != {f"condition.{name}" for name in CONDITIONS}:
        raise SystemExit("intervention file must contain exactly two conditions")

    specs = list(PERSONA_SPECS)
    if args.persona:
        specs = [spec for spec in specs if spec.prompt_key in args.persona]
    seeds = tuple(args.seed) if args.seed else FORMAL_SEEDS

    backend = LocalChatModel.load(args.model, args.device)
    directions = EmotionDirections.load(args.emotion_directions)
    materials = EmotionMaterials.from_prompts(load_prompts(args.emotion_materials))
    directions.assert_valid(materials, model_checksum=backend.snapshot_checksum)
    probe = EmotionProbe(backend.model, backend.tokenizer, directions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for spec in specs:
            persona = prompts[spec.prompt_key].strip()
            for seed in seeds:
                for condition in CONDITIONS:
                    intervention = interventions[f"condition.{condition}"].strip()
                    # Deliberately do not use system.base: it contains game instructions.
                    messages = [
                        {"role": "system", "content": persona},
                        {"role": "user", "content": intervention},
                    ]
                    prompt_end = probe.score_text(backend.render_messages(messages))
                    record = {
                        "schema_version": 1,
                        "run_id": f"{spec.prompt_key}.seed-{seed}",
                        "checkpoint_id": None,
                        "seed": seed,
                        "seed_is_record_index_only": True,
                        "persona_id": spec.prompt_key,
                        "persona_quadrant": spec.quadrant_id,
                        "persona_template_id": spec.template_id,
                        "condition": condition,
                        "model": backend.metadata(),
                        "persona_prompt_sha256": file_checksum(args.prompts),
                        "intervention_prompt_sha256": file_checksum(args.interventions),
                        "direction_artifact_sha256": file_checksum(args.emotion_directions),
                        "persona": persona,
                        "intervention": intervention,
                        "messages": messages,
                        "generated_response": None,
                        "prompt_end_emotion": prompt_end,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    print(spec.prompt_key, seed, condition, f"frustration={prompt_end['frustration']['median']:+.6f}")


if __name__ == "__main__":
    main()
