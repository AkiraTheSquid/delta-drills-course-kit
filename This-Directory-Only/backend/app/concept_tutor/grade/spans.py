"""Sentences and auto-links over the learner's own text (deterministic first).

1. `sentences(text)` splits with character offsets, protecting `$…$` maths,
   "e.g."/"i.e." and decimals, so Jev can be asked "which sentence uses X?"
   by sentence id (S1, S2, …) — ids, never positions inside a list of items.
2. `alias_hits(text, pack)` fuzzy-matches concept aliases on word boundaries:
   multi-word aliases at similarity ≥ 0.9 over the same number of words,
   one-word or ≤ 4-character aliases only exactly (so "SAE" never matches
   "say"). The longest non-overlapping match wins. Stdlib difflib, no new
   dependency. A Chinese alias (no spaces to bound it) matches as an exact
   substring.
3. `segments(...)` cuts the text into `[{text, link?}]` for the client, which
   builds DOM with textContent only. The server never returns HTML built from
   learner text. A concept Jev judged used (correct/partial/misused) with no
   alias hit links its `where` sentence, styled "inferred".
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

_PROTECT = re.compile(r"\$[^$]*\$|\b(?:e\.g|i\.e|etc|vs|cf)\.|\d\.\d")
# Chinese full stops end a sentence with no space after them (。！？ and closers).
_END = re.compile(r"[.!?]+[\"')\]]*(?=\s+|$)|[。！？]+[”’」』）)\]]*")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD = re.compile(r"[A-Za-z0-9$\\^_{}]+(?:['’-][A-Za-z0-9]+)*")


def word_count(text: str) -> float:
    """Words in `text`. Chinese has no spaces: two characters count as one word.
    Lives here (stdlib only) so question_shape, and the course kit's study_check,
    need not import the grader."""
    text = text or ""
    # a bare punctuation token ("。", "-") is not a word
    latin = [t for t in _CJK.sub(" ", text).split() if any(ch.isalnum() for ch in t)]
    return len(latin) + len(_CJK.findall(text)) / 2


def sentences(text: str) -> list[dict]:
    """[{id: "S1", start, end, text}] covering every non-blank stretch of `text`."""
    masked = list(text)
    for m in _PROTECT.finditer(text):
        for i in range(m.start(), m.end()):
            if masked[i] in ".!?":
                masked[i] = "·"
    flat = "".join(masked)
    cuts, start = [], 0
    for m in _END.finditer(flat):
        cuts.append((start, m.end()))
        start = m.end()
    if start < len(text):
        cuts.append((start, len(text)))
    out = []
    for a, b in cuts:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            out.append({"id": f"S{len(out) + 1}", "start": a, "end": b, "text": text[a:b]})
    return out


def _norm(w: str) -> str:
    w = w.casefold().replace("’", "'").replace("-", " ").replace("_", " ")
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _words(text: str) -> list[tuple[str, int, int]]:
    out = []
    for m in _WORD.finditer(text):
        for part in _norm(m.group()).split():
            out.append((part, m.start(), m.end()))
    return out


def alias_hits(text: str, aliases: dict[str, list[str]]) -> list[dict]:
    """[{concept, start, end}] for alias matches, longest first, no overlaps."""
    words = _words(text)
    cands = []
    wins: dict[int, list[str]] = {}   # the text's n-word windows, built once per n, not once per alias
    for cid, names in aliases.items():
        for name in names:
            if _CJK.search(name):   # Chinese: no word boundaries, so an exact substring
                for m in re.finditer(re.escape(name), text, re.IGNORECASE):   # offsets on text itself
                    cands.append({"concept": cid, "start": m.start(), "end": m.end()})
                continue
            target = [_norm(w) for w in name.split()]
            target = [t for part in target for t in part.split()]
            n = len(target)
            if not n:
                continue
            exact = n == 1 or len(name) <= 4
            joined = " ".join(target)
            lb = len(joined)
            sm = None if exact else SequenceMatcher(None, "", joined)   # b indexed once per alias, not per window
            if n not in wins:
                wins[n] = [" ".join(w for w, _, _ in words[i:i + n]) for i in range(len(words) - n + 1)]
            for i, window in enumerate(wins[n]):
                if exact:
                    ok = window == joined
                elif 2.0 * min(len(window), lb) / (len(window) + lb) < 0.9:
                    ok = False   # = real_quick_ratio(), inlined: lengths too far apart
                else:   # the quick ratio bounds ratio() from above: same matches, most windows cut early
                    sm.set_seq1(window)
                    ok = sm.quick_ratio() >= 0.9 and sm.ratio() >= 0.9
                if ok:
                    cands.append({"concept": cid, "start": words[i][1], "end": words[i + n - 1][2]})
    cands.sort(key=lambda h: (-(h["end"] - h["start"]), h["start"]))
    taken: list[tuple[int, int]] = []
    hits = []
    for h in cands:
        if all(h["end"] <= a or h["start"] >= b for a, b in taken):
            taken.append((h["start"], h["end"]))
            hits.append(h)
    return sorted(hits, key=lambda h: h["start"])


def segments(text: str, links: list[dict], lo: int = 0, hi: Optional[int] = None) -> list[dict]:
    """Cut text[lo:hi] into plain and linked segments. `links` are
    {start, end, link: {...}} on the full text, non-overlapping."""
    hi = len(text) if hi is None else hi
    out, pos = [], lo
    for ln in sorted(links, key=lambda x: x["start"]):
        a, b = max(ln["start"], lo), min(ln["end"], hi)
        if a >= b or a < pos:
            continue
        if a > pos:
            out.append({"text": text[pos:a]})
        out.append({"text": text[a:b], "link": ln["link"]})
        pos = b
    if pos < hi:
        out.append({"text": text[pos:hi]})
    return out
