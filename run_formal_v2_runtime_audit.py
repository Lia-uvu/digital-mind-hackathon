#!/usr/bin/env python3
"""Audit the exact formal-v2 runtime prompts with the local Qwen tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from encouragement_lab.formal_v2_runner import FormalV2Runner
from encouragement_lab.personas import PERSONA_KEYS
from encouragement_lab.records import file_checksum


ROOT = Path(__file__).resolve().parent


class TokenizerBackend:
    def __init__(self, tokenizer, model_name: str):
        self.tokenizer = tokenizer
        self.model_name = model_name

    def render_messages(self, messages, *, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def metadata(self):
        return {"name": self.model_name, "audit_only": True}

    def generate(self, messages, *, seed, sampling):  # pragma: no cover
        raise RuntimeError("audit backend cannot generate")


def token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def difference_spans(left: list[int], right: list[int]) -> list[list[int]]:
    if len(left) != len(right):
        raise ValueError("cannot compare token positions for unequal sequences")
    spans, start = [], None
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b and start is None:
            start = index
        elif a == b and start is not None:
            spans.append([start, index])
            start = None
    if start is not None:
        spans.append([start, len(left)])
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=ROOT / "formal_v2_prompts.md")
    parser.add_argument("--model", default=str(ROOT / "models/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=Path(args.model).is_dir()
    )
    backend = TokenizerBackend(tokenizer, str(args.model))
    runner = FormalV2Runner(backend, args.prompts, code_version="runtime-audit")
    audit: dict = {
        "schema_version": 1,
        "kind": "formal_v2_consolidated_runtime_token_render_audit",
        "collection_not_authorized_by_this_artifact": True,
        "prompts_sha256": file_checksum(args.prompts),
        "model_snapshot_sha256": hashlib.sha256(
            b"".join(
                path.relative_to(Path(args.model)).as_posix().encode() + b"\0" + path.read_bytes()
                for path in sorted(Path(args.model).glob("*"))
                if path.is_file() and path.name in {"tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"}
            )
        ).hexdigest(),
        "personas": {},
        "rounds": [],
    }

    for template in ("v1", "v2", "v3"):
        keys = [key for key in PERSONA_KEYS if (key.endswith(f".{template}") if template != "v1" else ".v" not in key)]
        rows = {}
        for key in keys:
            persona = runner.prompts[key].strip()
            system = runner.prompts["system.base"].format(persona=persona).strip()
            rendered = backend.render_messages([
                {"role": "system", "content": system},
                {"role": "user", "content": runner.prompts["game.intro"].strip()},
            ])
            rows[key] = {
                "sentence_token_counts": [len(token_ids(tokenizer, sentence.strip())) for sentence in persona.split(".") if sentence.strip()],
                "persona_tokens": len(token_ids(tokenizer, persona)),
                "system_tokens": len(token_ids(tokenizer, system)),
                "rendered_intro_tokens": len(token_ids(tokenizer, rendered)),
            }
        for field in ("sentence_token_counts", "persona_tokens", "system_tokens", "rendered_intro_tokens"):
            if len({json.dumps(row[field]) for row in rows.values()}) != 1:
                raise ValueError(f"persona {template} is not matched for {field}")
        audit["personas"][template] = rows

    for persona_id in PERSONA_KEYS:
        persona = runner.prompts[persona_id].strip()
        system = runner.prompts["system.base"].format(persona=persona).strip()
        histories = {
            arm: [
                {"role": "system", "content": system},
                {"role": "user", "content": runner.prompts["game.intro"].strip()},
            ]
            for arm in ("feedback_only", "supportive", "neutral")
        }
        for round_id in range(1, 6):
            for frame, feedback in (
                ("valid_feedback", runner.prompts["game.feedback"].format(guess="0123", exact=0, misplaced=0).strip()),
                ("invalid_guess", runner.prompts["game.invalid_guess"].strip()),
            ):
                rendered = {}
                raw_turn = {}
                for arm in histories:
                    prompt, _, _ = runner._feedback_user(arm=arm, failure_count=round_id, feedback=feedback)
                    trial = histories[arm] + [
                        {"role": "assistant", "content": '{"guess":"0123"}'},
                        {"role": "user", "content": prompt},
                    ]
                    raw_turn[arm] = token_ids(tokenizer, prompt)
                    rendered[arm] = token_ids(tokenizer, backend.render_messages(trial))
                if len(raw_turn["supportive"]) != len(raw_turn["neutral"]):
                    raise ValueError(f"round {round_id} {frame} full-turn mismatch")
                if len(rendered["supportive"]) != len(rendered["neutral"]):
                    raise ValueError(f"round {round_id} {frame} cumulative render mismatch for {persona_id}")
                audit["rounds"].append({
                    "persona_id": persona_id,
                    "round": round_id,
                    "frame": frame,
                    "full_turn_token_counts": {arm: len(ids) for arm, ids in raw_turn.items()},
                    "cumulative_render_token_counts": {arm: len(ids) for arm, ids in rendered.items()},
                    "supportive_neutral_full_turn_difference_spans": difference_spans(raw_turn["supportive"], raw_turn["neutral"]),
                    "supportive_neutral_render_difference_spans": difference_spans(rendered["supportive"], rendered["neutral"]),
                })
            # Advance the exact valid-feedback history; invalid is audited as an
            # alternative frame at every position without fabricating mixed paths.
            for arm in histories:
                feedback = runner.prompts["game.feedback"].format(guess="0123", exact=0, misplaced=0).strip()
                prompt, _, _ = runner._feedback_user(arm=arm, failure_count=round_id, feedback=feedback)
                histories[arm].extend([
                    {"role": "assistant", "content": '{"guess":"0123"}'},
                    {"role": "user", "content": prompt},
                ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote runtime audit: {args.output}")


if __name__ == "__main__":
    main()
