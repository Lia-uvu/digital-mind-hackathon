from __future__ import annotations

from pathlib import Path

import pytest

from encouragement_lab.filler_audit import audit_filler_pairs, candidate_pairs, feedback_turn
from encouragement_lab.persona_calibration import calibration_personas
from encouragement_lab.prompt_loader import load_prompts


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(range(len(text.replace(".", " .").replace(",", " ,").split()))) }


def _render(messages):
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def test_candidate_pairs_are_five_ordered_and_have_no_direct_response_cues():
    pairs = candidate_pairs(load_prompts(ROOT / "formal_v2_filler_candidates.md"))
    personas = calibration_personas(load_prompts(ROOT / "formal_v2_personas_v3.md"))
    audits = audit_filler_pairs(_Tokenizer(), pairs, personas, _render)
    assert [audit.round_id for audit in audits] == ["1", "2", "3", "4", "5"]
    assert all(audit.supportive_tokens == audit.neutral_tokens for audit in audits)
    assert all(audit.valid_feedback_turn_supportive_tokens == audit.valid_feedback_turn_neutral_tokens for audit in audits)
    assert all(audit.invalid_guess_turn_supportive_tokens == audit.invalid_guess_turn_neutral_tokens for audit in audits)
    assert all(counts[0] == counts[1] for audit in audits for counts in audit.rendered_valid_token_counts.values())
    assert all(counts[0] == counts[1] for audit in audits for counts in audit.rendered_invalid_token_counts.values())
    valid = feedback_turn(pairs["1"][0], round_id="1", frame="valid_feedback")
    invalid = feedback_turn(pairs["5"][0], round_id="5", frame="invalid_guess")
    assert "0 digit(s) are in the correct position" in valid
    assert "Consecutive unsuccessful guesses: 1." in valid
    assert "did not contain one valid four-digit guess" in invalid
    assert "Consecutive unsuccessful guesses: 5." in invalid
    assert "To continue, Reply with exactly one JSON object" in valid


def test_candidate_audit_rejects_a_direct_continue_cue():
    pairs = candidate_pairs(load_prompts(ROOT / "formal_v2_filler_candidates.md"))
    pairs["1"] = ("You can continue.", pairs["1"][1])
    personas = {"persona.high_e_high_n": "A general persona."}
    with pytest.raises(ValueError, match="prohibited cue"):
        audit_filler_pairs(_Tokenizer(), pairs, personas, _render)


def test_v2_candidates_add_reassurance_to_each_effort_affirmation():
    pairs = candidate_pairs(load_prompts(ROOT / "formal_v2_filler_candidates_v2.md"))
    support = [pair[0].lower() for pair in pairs.values()]
    assert all("no need" in text or "it is okay" in text for text in support)
    assert all(any(word in text for word in ("effort", "work", "attention")) for text in support)
    neutral = " ".join(pair[1].lower() for pair in pairs.values())
    assert "attention" not in neutral
    assert "information" not in neutral


def test_v3_rejects_v2_history_terms_and_keeps_nonbehavioral_reassurance():
    v2 = " ".join(load_prompts(ROOT / "formal_v2_filler_candidates_v2.md").values()).lower()
    assert any(term in v2 for term in ("timestamped", "slowly", "rush", "hurry"))
    pairs = candidate_pairs(load_prompts(ROOT / "formal_v2_filler_candidates_v3.md"))
    v3 = " ".join(text for pair in pairs.values() for text in pair).lower()
    for term in (
        "timestamp", "one entry", "slow", "rush", "hurry", "continue", "strategy", "guess", "next",
        "increasingly", "still", "again", "so far",
    ):
        assert term not in v3
    tokenizer = _Tokenizer()
    personas = calibration_personas(load_prompts(ROOT / "formal_v2_personas_v3.md"))
    audits = audit_filler_pairs(tokenizer, pairs, personas, _render)
    lengths = [audit.supportive_tokens for audit in audits]
    assert max(lengths) - min(lengths) <= 2


def test_v4_uses_only_the_two_flat_reassurance_carriers_and_natural_record_language():
    v3 = " ".join(load_prompts(ROOT / "formal_v2_filler_candidates_v3.md").values()).lower()
    assert "perfect result" in v3 or "in sequence" in v3
    pairs = candidate_pairs(load_prompts(ROOT / "formal_v2_filler_candidates_v4.md"))
    support = [pair[0].lower() for pair in pairs.values()]
    assert all(text.startswith(("it is okay if this feels difficult", "it is understandable if this feels difficult")) for text in support)
    neutral = [pair[1].lower() for pair in pairs.values()]
    assert all("completed round is recorded" in text and "entry" in text for text in neutral)
    assert not any(term in " ".join(neutral) for term in ("sequence", "places", "timestamp", "one entry"))
