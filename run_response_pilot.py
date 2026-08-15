#!/usr/bin/env python3
"""Replay saved formal checkpoints and measure response-time token trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from encouragement_lab.emotion_probe import EmotionDirections, EmotionMaterials, EmotionProbe
from encouragement_lab.experiment import CONDITIONS, derive_generation_seed
from encouragement_lab.model import LocalChatModel, SamplingConfig, sampling_metadata
from encouragement_lab.prompt_loader import load_prompts
from encouragement_lab.records import file_checksum


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "results/formal-v1.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/response-pilot-v1.jsonl")
    parser.add_argument("--prompts", type=Path, default=ROOT / "response_pilot_prompts.md")
    parser.add_argument("--emotion-materials", type=Path, default=ROOT / "prompts.md")
    parser.add_argument("--emotion-directions", type=Path, default=ROOT / "artifacts/qwen2.5-1.5b-emotion-directions.npz")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--persona", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def load_checkpoints(path: Path) -> list[dict]:
    """Return one verified shared checkpoint per formal run."""
    formal_prompts = load_prompts(ROOT / "prompts.md")
    grouped: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(row["checkpoint_id"], []).append(row)
    checkpoints: list[dict] = []
    for checkpoint_id, rows in grouped.items():
        if {row["condition"] for row in rows} != set(CONDITIONS):
            raise ValueError(f"incomplete formal pair: {checkpoint_id}")
        histories = []
        for row in rows:
            transcript = row["transcript"]
            condition_index = next(
                index
                for index, message in enumerate(transcript)
                if message["role"] == "user"
                and message["content"]
                == formal_prompts[f"condition.{row['condition']}"].strip()
            )
            histories.append(transcript[:condition_index])
        if histories[0] != histories[1]:
            raise ValueError(f"formal branches do not share history: {checkpoint_id}")
        base = rows[0]
        checkpoints.append({
            "checkpoint_id": checkpoint_id,
            "run_id": base["run_id"],
            "seed": base["seed"],
            "persona_id": base["persona_id"],
            "persona_quadrant": base.get("persona_quadrant"),
            "persona_template_id": base.get("persona_template_id"),
            "messages": histories[0],
            "pre_intervention_emotion": base["emotion_projections"]["pre_intervention"],
        })
    return sorted(checkpoints, key=lambda row: (row["persona_id"], row["seed"]))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    interventions = load_prompts(args.prompts)
    if set(interventions) != {f"condition.{name}" for name in CONDITIONS}:
        raise SystemExit("response pilot prompt file must contain exactly two conditions")
    backend = LocalChatModel.load(args.model, args.device)
    directions = EmotionDirections.load(args.emotion_directions)
    materials = EmotionMaterials.from_prompts(load_prompts(args.emotion_materials))
    directions.assert_valid(materials, model_checksum=backend.snapshot_checksum)
    probe = EmotionProbe(backend.model, backend.tokenizer, directions)
    sampling = SamplingConfig(args.max_new_tokens, args.temperature, args.top_p)
    checkpoints = load_checkpoints(args.source)
    if args.seed:
        checkpoints = [row for row in checkpoints if row["seed"] in args.seed]
    if args.persona:
        checkpoints = [row for row in checkpoints if row["persona_id"] in args.persona]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for checkpoint in checkpoints:
            for condition in CONDITIONS:
                messages = [*checkpoint["messages"], {
                    "role": "user",
                    "content": interventions[f"condition.{condition}"].strip(),
                }]
                generation_seed = derive_generation_seed(
                    checkpoint["seed"], "response_only_reply"
                )
                generated = backend.generate_detailed(
                    messages, seed=generation_seed, sampling=sampling
                )
                all_ids = generated.prompt_token_ids + generated.generated_token_ids
                trajectory = probe.score_token_ids(
                    all_ids, token_start=len(generated.prompt_token_ids)
                )
                frustration = [
                    token["projections"]["frustration"]["median"]
                    for token in trajectory
                ]
                record = {
                    "schema_version": 1,
                    **{key: checkpoint[key] for key in (
                        "checkpoint_id", "run_id", "seed", "persona_id",
                        "persona_quadrant", "persona_template_id",
                    )},
                    "condition": condition,
                    "generation_seed": generation_seed,
                    "sampling": sampling_metadata(sampling),
                    "model": backend.metadata(),
                    "source_file": str(args.source),
                    "source_sha256": file_checksum(args.source),
                    "intervention_prompt_sha256": file_checksum(args.prompts),
                    "direction_artifact_sha256": file_checksum(args.emotion_directions),
                    "intervention": messages[-1]["content"],
                    "response": generated.text,
                    "generated_token_count": len(generated.generated_token_ids),
                    "pre_intervention_emotion": checkpoint["pre_intervention_emotion"],
                    "prompt_end_emotion": probe.score_text(backend.render_messages(messages)),
                    "token_trajectory": trajectory,
                    "frustration_summary": {
                        "first_token": frustration[0] if frustration else None,
                        "final_token": frustration[-1] if frustration else None,
                        "median_across_tokens": float(median(frustration)) if frustration else None,
                        "minimum": min(frustration) if frustration else None,
                        "maximum": max(frustration) if frustration else None,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                print(checkpoint["run_id"], condition, repr(generated.text), record["frustration_summary"])


if __name__ == "__main__":
    main()
