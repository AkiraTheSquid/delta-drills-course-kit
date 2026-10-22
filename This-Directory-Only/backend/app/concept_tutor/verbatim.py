"""The rule that makes a passage a quotation: it is VERBATIM in its source.

Moved here from scripts/concept_packs/check_pack.py (2026-10-22) so the server
can hold a contributor's course to it at publish time (contrib.py). The
offline gate re-exports these names, so it checks exactly what the server does.

Both sides get one normalisation: Unicode NFKC; curly quotes -> straight;
en/em/minus dashes -> "-"; a hyphen before a line break joined, other hyphens
dropped, every run of whitespace collapsed to one space. A changed, merged or
split word still fails, and a match may not start or end mid-word.
"""

from __future__ import annotations

import html
import re
import unicodedata

MAX_WORDS = 220
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
                         "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
                         "¬": "-",   # ¬ = an old scanned JAMA's line-break hyphen (Donabedian 1988)
                         "\x96": "-", "\x97": "-"})   # cp1252 en/em dash left as C1 controls (PSNet 2007 print)


def norm(s: str) -> str:
    """Forgive layout, never wording. Word boundaries SURVIVE: "not able" and
    "notable" stay different (codex critic 2026-10-03; the first version dropped
    all whitespace and let a merged word through)."""
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES)
    s = re.sub(r"-\s+", "", s)   # line-break hyphenation: "reten-\ntion" -> "retention"
    s = s.replace("-", "")       # so "test-enhanced" matches "test-\nenhanced" either way
    return re.sub(r"\s+", " ", s).strip()


def contains(src: str, passage: str) -> bool:
    """Normalised substring match that cannot start or end mid-word. `src` is
    already normalised (norm_source / norm)."""
    p = norm(passage)
    pat = ("(?<!\\w)" if p[:1].isalnum() else "") + re.escape(p) + ("(?!\\w)" if p[-1:].isalnum() else "")
    return re.search(pat, src) is not None


def text_of(raw: str, suffix: str) -> str:
    """A text source's words: HTML loses its tags, scripts and styles; Markdown
    and plain text are read as they are. (PDFs are the offline gate's alone.)"""
    if suffix.lower() in (".html", ".htm"):
        raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
        raw = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return raw


def near(passage: str, src: str) -> str:
    """How far the passage matches before it diverges — for the error message."""
    p = norm(passage)
    lo, hi = 0, len(p)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if p[:mid] in src:
            lo = mid
        else:
            hi = mid - 1
    return f"first {lo}/{len(p)} normalised chars match; diverges near {p[lo:lo + 40]!r}"
