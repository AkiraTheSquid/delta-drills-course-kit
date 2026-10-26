"""A free question must set the scene before it asks (2026-10-25).

A free question is the probe a learner meets COLD, before any lesson: the
diagnostic asks it first. ai-cells-cancer shipped 68 of them as bare one-liners
("注意力汇聚是怎样把多个输入合成一个输出的？", i.e. "how does attention pooling
combine inputs?") and Seth could not tell what was being asked. So a
contributor course's free prompt needs a lead-in: at least one sentence of
context (which paper, which setting, what the parts are) before the ask, and
enough words in all to carry it.

Words are counted as the grader counts them (`spans.word_count`: two Chinese
characters = one word), sentences as the grader splits them (`spans.sentences`:
。！？ end a sentence, "e.g." and "0.8" do not), so the rule is the same in
every course language.
"""

from __future__ import annotations

from app.concept_tutor.grade.spans import sentences, word_count

MIN_WORDS = 20      # ~40 Chinese characters
_CLOSERS = "\"')]”’」』）"


def _asks(sentence: str) -> bool:
    return sentence.rstrip().rstrip(_CLOSERS).endswith(("?", "？"))


def _lead_in(prompt: str) -> bool:
    """A statement comes before the ask. The ask = the first question
    sentence, or the last sentence when none ends in "?" ("请解释……。")."""
    parts = [s["text"] for s in sentences(prompt) if word_count(s["text"]) >= 1]
    ask = next((i for i, t in enumerate(parts) if _asks(t)), len(parts) - 1)
    return ask >= 1


def bare(prompt: str) -> str:
    """Why `prompt` is too bare to ask cold, or "" when it is fine."""
    words = word_count(prompt)
    lacks = []
    if not _lead_in(prompt):
        lacks.append("no lead-in sentence before the ask")
    if words < MIN_WORDS:
        lacks.append(f"{words:g} words, need {MIN_WORDS} (2 Chinese characters = 1 word)")
    return "; ".join(lacks)


def problems(spec: dict) -> list[str]:
    """One error per free question that would reach a learner with no context."""
    out = []
    for q in spec.get("questions", []):
        if q.get("format", "free") != "free":
            continue
        why = bare(q.get("prompt", ""))
        if why:
            out.append(f"question {q.get('id')}: prompt too bare to ask cold ({why}). Open with a sentence "
                       "naming the paper or setting and what the parts are, then ask.")
    return out
