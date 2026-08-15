"""Load editable experiment prompts and lint persona text for obvious leaks.

``prompts.md`` is intentionally a tiny, human-editable format rather than a
general Markdown parser.  Each named prompt is a level-two heading immediately
followed by a ``text`` fenced code block.  The body is returned exactly as it
appears inside that block so checksums and experiment records remain traceable
to the editable source.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping


class PromptFormatError(ValueError):
    """Raised when a prompts document does not follow the project format."""


class PersonaLeakError(ValueError):
    """Raised when a persona appears to prescribe an experimental reaction."""


_HEADING = re.compile(r"^##[ \t]+(?P<key>.*?)[ \t]*$")
_OPEN_FENCE = re.compile(r"^(?P<fence>`{3,})text[ \t]*$", re.IGNORECASE)

# This is deliberately a narrow guardrail, not a semantic review.  In
# particular it does not flag ordinary traits such as "容易担忧" or "安静内敛".
_PERSONA_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("encouragement", re.compile(r"\bencourag\w*\b", re.IGNORECASE)),
    ("frustration", re.compile(r"\bfrustrat\w*\b", re.IGNORECASE)),
    ("failure", re.compile(r"\bfail(?:ed|ing|ure|ures)?\b", re.IGNORECASE)),
    ("give up", re.compile(r"\bgive[\s-]*up\b", re.IGNORECASE)),
    ("continue", re.compile(r"\bcontinu\w*\b", re.IGNORECASE)),
    ("next guess", re.compile(r"\bnext[\s-]+guess\w*\b", re.IGNORECASE)),
    ("Mastermind", re.compile(r"\bmastermind\b", re.IGNORECASE)),
    ("鼓励", re.compile(r"鼓励|安慰")),
    ("受挫或失败", re.compile(r"受挫|挫败|失败|连败")),
    ("放弃", re.compile(r"放弃|不再玩")),
    ("继续", re.compile(r"继续(?:玩|游戏|猜测)?|愿意继续")),
    ("下一猜", re.compile(r"下一[次轮]?猜(?:测)?|下次猜(?:测)?")),
    ("Mastermind", re.compile(r"密码大师|猜码游戏|数字猜谜")),
)


def load_prompts(path: str | Path) -> dict[str, str]:
    """Return ``key -> exact code-block body`` from a ``prompts.md`` file.

    Every level-two heading is treated as a prompt key.  A heading must have a
    nonempty key and be followed, allowing blank Markdown spacing, by a
    `````text`` fence;
    duplicate keys, missing closing fences, and whitespace-only bodies are
    rejected rather than silently changing an experiment's stimuli.
    """
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    prompts: dict[str, str] = {}
    index = 0

    while index < len(lines):
        heading = _HEADING.match(lines[index].rstrip("\r\n"))
        if heading is None:
            index += 1
            continue

        key = heading.group("key").strip()
        line_number = index + 1
        if not key:
            raise PromptFormatError(f"{source}:{line_number}: prompt key must not be empty")
        if key in prompts:
            raise PromptFormatError(f"{source}:{line_number}: duplicate prompt key {key!r}")
        fence_index = index + 1
        while fence_index < len(lines) and not lines[fence_index].strip():
            fence_index += 1
        if fence_index >= len(lines):
            raise PromptFormatError(f"{source}:{line_number}: missing text block for {key!r}")

        opening = _OPEN_FENCE.match(lines[fence_index].rstrip("\r\n"))
        if opening is None:
            raise PromptFormatError(
                f"{source}:{line_number}: {key!r} must be immediately followed by a ```text block"
            )
        closing_fence = opening.group("fence")
        body_start = fence_index + 1
        cursor = body_start
        while cursor < len(lines) and lines[cursor].rstrip("\r\n") != closing_fence:
            cursor += 1
        if cursor == len(lines):
            raise PromptFormatError(f"{source}:{line_number}: unclosed text block for {key!r}")

        body = "".join(lines[body_start:cursor])
        if not body.strip():
            raise PromptFormatError(f"{source}:{line_number}: prompt {key!r} must not be empty")
        prompts[key] = body
        index = cursor + 1

    return prompts


def find_persona_leaks(persona: str) -> tuple[str, ...]:
    """Return labels for obvious stimulus/outcome terms found in *persona*.

    The result is suitable for a runner to surface before an experiment.  It
    cannot replace the required human semantic review of persona wording.
    """
    if not isinstance(persona, str):
        raise TypeError("persona must be a string")
    return tuple(label for label, pattern in _PERSONA_LEAK_PATTERNS if pattern.search(persona))


def validate_persona(persona: str, *, name: str = "persona") -> None:
    """Raise :class:`PersonaLeakError` when the obvious-leak lint finds terms."""
    hits = find_persona_leaks(persona)
    if hits:
        raise PersonaLeakError(f"{name} contains possible experimental leakage: {', '.join(hits)}")


def load_personas(path: str | Path) -> dict[str, str]:
    """Load and lint prompt entries whose keys begin with ``persona_``.

    Keeping this separate from :func:`load_prompts` allows the same document to
    contain the deliberately encouraging intervention text without false
    positives.
    """
    prompts = load_prompts(path)
    personas = {
        key: text
        for key, text in prompts.items()
        if key.startswith(("persona.", "persona_"))
    }
    for key, text in personas.items():
        validate_persona(text, name=key)
    return personas


def persona_lint(persona: str) -> tuple[str, ...]:
    """Compatibility-friendly short name for :func:`find_persona_leaks`."""
    return find_persona_leaks(persona)
