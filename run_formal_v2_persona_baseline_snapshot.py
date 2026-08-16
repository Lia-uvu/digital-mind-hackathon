#!/usr/bin/env python3
"""Save an immutable no-game, no-generation probe baseline for persona v3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from encouragement_lab.emotion_probe import EmotionDirections, EmotionMaterials
from encouragement_lab.model import LocalChatModel
from encouragement_lab.persona_baseline import baseline_rows, summarize_baseline, write_snapshot
from encouragement_lab.persona_calibration import (
    CALIBRATION_LAYERS,
    audit_persona_tokens,
    calibration_personas,
    calibration_suffixes,
    render_calibration_inputs,
    token_audit_json,
)
from encouragement_lab.prompt_loader import load_prompts, validate_persona
from encouragement_lab.records import file_checksum
from run_formal_v2_persona_calibration import _hidden_endpoints


ROOT = Path(__file__).resolve().parent
DEFAULT_PERSONAS = ROOT / "formal_v2_personas_v3.md"
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_DIRECTIONS = ROOT / "artifacts" / "qwen2.5-1.5b-emotion-directions.npz"
DEFAULT_SOURCE = ROOT / "results" / "formal-v2-persona-calibration-v3.json"
DEFAULT_JSON = ROOT / "results" / "formal-v2-persona-calibration-v3-baseline.json"
DEFAULT_CSV = ROOT / "results" / "formal-v2-persona-calibration-v3-baseline.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personas", type=Path, default=DEFAULT_PERSONAS)
    parser.add_argument("--calibration-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--emotion-materials", type=Path, default=ROOT / "prompts.md")
    parser.add_argument("--emotion-directions", type=Path, default=DEFAULT_DIRECTIONS)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def _source_checksum(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    def command(args: list[str]) -> str | None:
        try:
            return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"head": command(["git", "rev-parse", "HEAD"]), "status_porcelain": command(["git", "status", "--porcelain"])}


def main() -> None:
    args = parse_args()
    if args.json_output.exists() or args.csv_output.exists():
        raise SystemExit("refusing to overwrite baseline snapshot output")
    if not args.calibration_source.exists():
        raise SystemExit(f"missing v3 calibration source: {args.calibration_source}")

    prompts = load_prompts(args.personas)
    personas = calibration_personas(prompts)
    suffixes = calibration_suffixes(prompts)
    for key, persona in personas.items():
        validate_persona(persona, name=key)

    backend = LocalChatModel.load(args.model, args.device)
    directions = EmotionDirections.load(args.emotion_directions)
    materials = EmotionMaterials.from_prompts(load_prompts(args.emotion_materials))
    directions.assert_valid(materials, model_checksum=backend.snapshot_checksum)
    direction_layer_map = {
        layer: backend.model.config.num_hidden_layers + layer if layer < 0 else layer
        for layer in CALIBRATION_LAYERS
    }
    render = backend.render_messages
    audits = audit_persona_tokens(backend.tokenizer, personas, suffixes, render)
    rendered = render_calibration_inputs(personas, suffixes, render)
    hidden = _hidden_endpoints(backend, rendered)
    raw_rows = baseline_rows(hidden, directions.directions, direction_layer_map, suffixes)
    snapshot_id = "formal-v2-persona-calibration-v3-baseline"
    suffix_hashes = {key: hashlib.sha256(text.encode("utf-8")).hexdigest() for key, text in suffixes.items()}
    csv_rows = [
        {
            **row,
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "suffix_text_sha256": suffix_hashes[row["suffix_id"]],
        }
        for row in raw_rows
    ]
    source_files = [
        ROOT / "run_formal_v2_persona_baseline_snapshot.py",
        ROOT / "encouragement_lab" / "persona_baseline.py",
        ROOT / "encouragement_lab" / "persona_calibration.py",
        ROOT / "run_formal_v2_persona_calibration.py",
    ]
    payload = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "purpose": "paper snapshot; deterministic persona-system plus common neutral-user endpoint; no game and no generation",
        "stochastic_sampling": False,
        "model": backend.metadata(),
        "persona_prompt_path": str(args.personas),
        "persona_prompt_sha256": file_checksum(args.personas),
        "personas": personas,
        "suffixes": suffixes,
        "suffix_text_sha256": suffix_hashes,
        "token_audit": token_audit_json(audits),
        "layers_relative": list(CALIBRATION_LAYERS),
        "direction_layer_map": direction_layer_map,
        "direction_artifact_path": str(args.emotion_directions),
        "direction_artifact_sha256": file_checksum(args.emotion_directions),
        "emotion_materials_path": str(args.emotion_materials),
        "emotion_materials_sha256": file_checksum(args.emotion_materials),
        "calibration_source_path": str(args.calibration_source),
        "calibration_source_sha256": file_checksum(args.calibration_source),
        "runner_and_calibration_source_sha256": _source_checksum(source_files),
        "git": _git_metadata(),
        "raw_scores": raw_rows,
        "descriptive_summary": summarize_baseline(raw_rows),
    }
    write_snapshot(args.json_output, args.csv_output, payload, csv_rows)
    print(f"wrote immutable baseline snapshot: {args.json_output} and {args.csv_output}")


if __name__ == "__main__":
    main()
