import pytest

from encouragement_lab.prompt_loader import (
    PersonaLeakError,
    PromptFormatError,
    find_persona_leaks,
    load_personas,
    load_prompts,
    validate_persona,
)


def write_prompts(tmp_path, content: str):
    path = tmp_path / "prompts.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_named_text_blocks_without_normalizing_their_body(tmp_path) -> None:
    source = write_prompts(
        tmp_path,
        "# Experiment prompts\n\n## persona_low_e_low_n\n```text\n安静内敛，情绪通常平稳。\n```\n\n## encouragement\n```text\n你已经认真尝试了。\n```\n",
    )

    assert load_prompts(source) == {
        "persona_low_e_low_n": "安静内敛，情绪通常平稳。\n",
        "encouragement": "你已经认真尝试了。\n",
    }


def test_loader_allows_normal_markdown_spacing_before_fence(tmp_path) -> None:
    source = write_prompts(
        tmp_path,
        "## persona.high_e_low_n\n\n```text\ncalm and outgoing\n```\n",
    )
    assert load_personas(source) == {
        "persona.high_e_low_n": "calm and outgoing\n"
    }


@pytest.mark.parametrize(
    "content, message",
    [
        ("## one\ntext\n", "immediately followed"),
        ("## one\n```text\nhello\n", "unclosed"),
        ("## one\n```text\n \n```\n", "must not be empty"),
        ("## \n```text\nhello\n```\n", "key must not be empty"),
        ("## one\n```text\na\n```\n## one\n```text\nb\n```\n", "duplicate"),
    ],
)
def test_loader_rejects_malformed_named_blocks(tmp_path, content: str, message: str) -> None:
    with pytest.raises(PromptFormatError, match=message):
        load_prompts(write_prompts(tmp_path, content))


def test_persona_lint_flags_obvious_stimulus_or_outcome_leaks() -> None:
    hits = find_persona_leaks("Encouragement after failure makes you continue with the next guess in Mastermind.")

    assert {"encouragement", "failure", "continue", "next guess", "Mastermind"}.issubset(hits)
    assert {"鼓励", "受挫或失败", "放弃", "继续", "下一猜"}.issubset(
        find_persona_leaks("被鼓励后失败时不放弃，会继续进行下一次猜测。")
    )


def test_persona_lint_allows_general_traits_and_load_personas_applies_it(tmp_path) -> None:
    text = "容易担忧，但安静内敛；平时也可以活跃外向。"
    assert find_persona_leaks(text) == ()
    validate_persona(text)

    source = write_prompts(
        tmp_path,
        f"## persona_high_n\n```text\n{text}\n```\n\n## neutral\n```text\n请作答。\n```\n",
    )
    assert load_personas(source) == {"persona_high_n": text + "\n"}

    leaking = write_prompts(tmp_path, "## persona_high_n\n```text\n被鼓励后会继续。\n```\n")
    with pytest.raises(PersonaLeakError, match="possible experimental leakage"):
        load_personas(leaking)
