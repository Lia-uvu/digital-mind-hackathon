#!/usr/bin/env python3
"""Audit and score formal-v2 persona drafts without a game or generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from encouragement_lab.emotion_probe import EmotionDirections, EmotionMaterials
from encouragement_lab.model import LocalChatModel
from encouragement_lab.persona_calibration import (
    CALIBRATION_LAYERS,
    audit_persona_tokens,
    calibration_personas,
    calibration_suffixes,
    contamination_audit,
    evaluate_leave_one_template_out,
    render_calibration_inputs,
    token_audit_json,
)
from encouragement_lab.prompt_loader import load_prompts, validate_persona
from encouragement_lab.records import file_checksum


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_PERSONAS = ROOT / "formal_v2_personas.md"
DEFAULT_DIRECTIONS = ROOT / "artifacts" / "qwen2.5-1.5b-emotion-directions.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personas", type=Path, default=DEFAULT_PERSONAS)
    parser.add_argument("--emotion-materials", type=Path, default=ROOT / "prompts.md")
    parser.add_argument("--emotion-directions", type=Path, default=DEFAULT_DIRECTIONS)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/formal-v2-persona-calibration.json")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="run the Qwen tokenizer/render audit without loading weights or hidden states",
    )
    return parser.parse_args()


def _load_qwen_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment integration
        raise SystemExit("transformers is required for the tokenizer audit") from error
    return AutoTokenizer.from_pretrained(model, local_files_only=Path(model).is_dir())


def _render_with_tokenizer(tokenizer):
    def render(messages, *, add_generation_prompt: bool = True) -> str:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
        )

    return render


def _hidden_endpoints(backend: LocalChatModel, rendered: dict[str, dict[str, str]]) -> dict[str, dict[str, dict[int, Any]]]:
    from repeng.extract import batched_get_hiddens

    result: dict[str, dict[str, dict[int, Any]]] = {}
    for suffix, by_persona in rendered.items():
        ordered = sorted(by_persona.items())
        hidden = batched_get_hiddens(
            backend.model,
            backend.tokenizer,
            [text for _, text in ordered],
            list(CALIBRATION_LAYERS),
            batch_size=2,
        )
        result[suffix] = {
            persona: {layer: hidden[layer][index] for layer in CALIBRATION_LAYERS}
            for index, (persona, _) in enumerate(ordered)
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    prompts = load_prompts(args.personas)
    personas = calibration_personas(prompts)
    suffixes = calibration_suffixes(prompts)
    for key, persona in personas.items():
        validate_persona(persona, name=key)

    tokenizer = _load_qwen_tokenizer(args.model)
    render = _render_with_tokenizer(tokenizer)
    audits = audit_persona_tokens(tokenizer, personas, suffixes, render)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "formal_v2_persona_activation_calibration",
        "prompt_only_no_generation": True,
        "seed_is_not_used": True,
        "personas_sha256": file_checksum(args.personas),
        "model": str(args.model),
        "layers": list(CALIBRATION_LAYERS),
        "suffixes": suffixes,
        "token_audit": token_audit_json(audits),
    }
    if not args.audit_only:
        backend = LocalChatModel.load(args.model, args.device)
        rendered = render_calibration_inputs(personas, suffixes, backend.render_messages)
        hidden = _hidden_endpoints(backend, rendered)
        payload["model_metadata"] = backend.metadata()
        payload["activation_calibration"] = evaluate_leave_one_template_out(hidden)

        directions = EmotionDirections.load(args.emotion_directions)
        materials = EmotionMaterials.from_prompts(load_prompts(args.emotion_materials))
        directions.assert_valid(materials, model_checksum=backend.snapshot_checksum)
        direction_layer_map = {
            layer: backend.model.config.num_hidden_layers + layer
            if layer < 0
            else layer
            for layer in CALIBRATION_LAYERS
        }
        if any(
            direction_layer_map[layer] not in directions.directions[axis]
            for axis in directions.directions
            for layer in CALIBRATION_LAYERS
        ):
            raise SystemExit("existing emotion-direction artifact does not cover calibration layers")
        payload["existing_probe_contamination_audit"] = contamination_audit(
            hidden, directions.directions, direction_layer_map=direction_layer_map
        )
        payload["existing_probe_direction_layer_map"] = direction_layer_map
        payload["emotion_direction_artifact_sha256"] = file_checksum(args.emotion_directions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.audit_only:
        print(f"wrote tokenizer audit: {args.output}")
    else:
        decision = payload["activation_calibration"]["pass_rule"]
        print(
            f"wrote activation calibration: {args.output}; "
            f"passed={decision['passed']} robust={decision['robust']}"
        )


if __name__ == "__main__":
    main()
