#!/usr/bin/env python3
"""Write a non-overwritable, candidate-only Qwen filler token audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from encouragement_lab.filler_audit import audit_filler_pairs, audit_json, candidate_pairs
from encouragement_lab.persona_calibration import calibration_personas
from encouragement_lab.prompt_loader import load_prompts
from encouragement_lab.records import file_checksum


ROOT = Path(__file__).resolve().parent


def _tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, local_files_only=Path(model).is_dir())


def _render(tokenizer):
    return lambda messages: tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fillers", type=Path, default=ROOT / "formal_v2_filler_candidates.md")
    parser.add_argument("--personas", type=Path, default=ROOT / "formal_v2_personas_v3.md")
    parser.add_argument("--model", default=str(ROOT / "models" / "Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/formal-v2-filler-candidates-token-audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    tokenizer = _tokenizer(args.model)
    pairs = candidate_pairs(load_prompts(args.fillers))
    personas = calibration_personas(load_prompts(args.personas))
    audits = audit_filler_pairs(tokenizer, pairs, personas, _render(tokenizer))
    payload = {
        "schema_version": 2,
        "kind": "formal_v2_filler_candidate_token_audit",
        "candidate_only_not_frozen": True,
        "collection_not_authorized": True,
        "fillers_sha256": file_checksum(args.fillers),
        "personas_sha256": file_checksum(args.personas),
        "model": str(args.model),
        "full_turn_frames": {
            "valid_feedback": "exact protocol valid-feedback wording; each candidate uses its round id 1–5",
            "invalid_guess": "exact protocol invalid-guess wording; each candidate uses its round id 1–5",
        },
        "audit": audit_json(audits),
        "freeze_note": (
            "v1/v2 old full-turn audits used a representative non-protocol frame and cannot support freeze. "
            f"Re-run {args.fillers.name} against the final runner message construction and frozen persona file before collection."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote candidate-only filler audit: {args.output}")


if __name__ == "__main__":
    main()
