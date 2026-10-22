#!/usr/bin/env python3
"""Offline gate for a concept pack: every passage is VERBATIM from its source.

    This-Directory-Only/backend/.venv/bin/python3 \
        This-Directory-Only/scripts/concept_packs/check_pack.py [pack_id ...]

The tutor shows passages as exact quotations, so this is the check that makes
that true. For each passage it extracts the source file's text (pdftotext for
a PDF, tags stripped for HTML, raw for code and Markdown) and requires the
passage to be a substring of it after one normalisation applied to BOTH sides:

  * Unicode NFKC; curly quotes -> straight; en/em/minus dashes -> "-"
  * a hyphen before a line break joined, other hyphens dropped, every run of
    whitespace collapsed to one space

That lets a PDF's line-break hyphenation ("reten-\\ntion") and column wrapping
match the passage's running text while word boundaries survive: a changed,
merged or split word fails, and a match may not start or end mid-word. A passage that
crosses a PDF page break picks up the running header ("253 Test-Enhanced
Learning") and fails: quote within a page.

Also runs the pack's structural checks (pack.problems) and a length cap.
Exit 1 on any failure. It never edits the pack and never sets `vetted`: that
flag is a person's sign-off.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "This-Directory-Only" / "backend"))

from app.concept_tutor.pack import PACKS_DIR, load, problems  # noqa: E402

from app.concept_tutor.verbatim import MAX_WORDS, contains, norm, text_of  # noqa: E402,F401
from app.concept_tutor.verbatim import near as _near  # noqa: E402

PDF_CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "delta-drills" / "pdftotext"


def norm_source(path: str) -> str:
    """norm(source_text(path)), once per source (a pack quotes each many times)."""
    return _norm_source(path)


@lru_cache(maxsize=None)
def _norm_source(path: str) -> str:
    return norm(source_text(path))


@lru_cache(maxsize=None)
def source_text(path: str) -> str:
    f = REPO / path
    if not f.exists():
        raise FileNotFoundError(f"source file missing: {path}")
    if f.suffix.lower() == ".pdf":
        # pdftotext is ~0.15 s a PDF and the pack cites 40+; the text is cached
        # on disk by (path, size, mtime) so `mod watch` fits its 10 s budget.
        st = f.stat()
        key = hashlib.sha256(f"{f.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
        cached = PDF_CACHE / f"{key}.txt"
        if cached.exists():
            return cached.read_text(encoding="utf-8")
        out = subprocess.run(["pdftotext", "-enc", "UTF-8", str(f), "-"], capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"pdftotext failed on {path}: {out.stderr.strip()}")
        if len(out.stdout.split()) < 50:
            raise RuntimeError(f"{path} has no text layer (scanned?) — it cannot back a verbatim passage")
        try:
            PDF_CACHE.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            tmp.write_text(out.stdout, encoding="utf-8")
            tmp.replace(cached)
        except OSError:
            pass  # an unwritable cache only costs speed
        return out.stdout
    return text_of(f.read_text(encoding="utf-8", errors="replace"), f.suffix)


def check(pack_id: str) -> list[str]:
    return check_loaded(pack_id)[0]


def check_loaded(pack_id: str):
    """(failures, pack or None). The pack is None when it did not load."""
    load.cache_clear()
    try:
        pack = load(pack_id)
    except ValueError as exc:
        return [str(exc)], None
    bad = list(problems(pack))
    for x in pack.passages:
        s = pack.source(x.source)
        n = len(x.text.split())
        if n > MAX_WORDS:
            bad.append(f"passage {x.id}: {n} words (cap {MAX_WORDS})")
        try:
            src = norm_source(s.path)
        except (FileNotFoundError, RuntimeError) as exc:
            bad.append(f"passage {x.id}: {exc}")
            continue
        if not contains(src, x.text):
            bad.append(f"passage {x.id} is NOT verbatim in {s.path}: {_near(x.text, src)}")
    used = {pid for c in pack.concepts for pid in c.passages}
    bad += [f"passage {x.id} is used by no concept" for x in pack.passages if x.id not in used]
    return bad, pack


def main(argv: list[str]) -> int:
    ids = argv or sorted(p.name for p in PACKS_DIR.iterdir() if (p / "pack.json").exists())
    failed = 0
    for pid in ids:
        bad, pack = check_loaded(pid)
        n = len(pack.passages) if pack else 0
        if bad:
            failed += 1
            print(f"FAIL {pid}")
            for b in bad:
                print(f"  - {b}")
        else:
            print(f"ok   {pid}: {n} passages verbatim, structure clean"
                  + ("" if pack.vetted else " (not yet vetted by a person)"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
