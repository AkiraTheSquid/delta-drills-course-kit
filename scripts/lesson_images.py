"""Pictures in content: every `![alt](src)` a learner can be shown is checked.

Authored through content-mcp `image_add`, which copies the file to
`lessons/<course>/images/` and hands back the markdown. This is the gate side.

- The src must be ROOT-ABSOLUTE `/lessons/<course>/images/<file>`. The app has
  deep links, and a relative src resolves against `/practice/...`, where the
  SPA rewrite answers index.html with a 200 — a broken picture nothing reports.
- No SVG. Served same-origin, an SVG opened directly runs its script with the
  app's cookies; raster formats cannot.
- `https://` is allowed (the renderers whitelist it); any other scheme is not.
- Alt text is required: it is what a screen reader says, and what shows when
  the file fails to load.
"""
from __future__ import annotations

import re

from lesson_lib import LESSONS_DIR

IMAGE_MD = re.compile(r"!\[([^\]]*)\]\(([^)\s]*)\)")
EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def check_text(text, label, errors):
    for alt, src in IMAGE_MD.findall(text or ""):
        where = f"{label}: image {src or '(empty)'}"
        if not alt.strip():
            errors.append(f"{where}: no alt text — say what the picture shows")
        if src.startswith("https://"):
            continue
        if src.startswith(("http://", "//")) or re.match(r"^[a-zA-Z][\w+.-]*:", src):
            errors.append(f"{where}: only https:// or a /lessons/<course>/images/ file may be shown")
            continue
        if not src.startswith("/lessons/") or "/images/" not in src or ".." in src:
            errors.append(f"{where}: a local picture must be /lessons/<course>/images/<file> "
                          "(root-absolute; a relative src breaks on deep links) — add it with image_add")
            continue
        if not src.lower().endswith(EXTS):
            errors.append(f"{where}: not a PNG/JPEG/GIF/WebP (SVG is refused — export it as PNG)")
        elif not (LESSONS_DIR / src[len("/lessons/"):]).is_file():
            errors.append(f"{where}: file not found under lessons/")


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def check_all(kp_paths, bank, errors):
    """Every KP page, the MC bank beside it, and every drill in the bank."""
    for path in kp_paths:
        check_text(path.read_text(encoding="utf-8"), path.name, errors)
        problems = path.with_name(path.stem + ".problems.json")
        if problems.exists():
            check_text(problems.read_text(encoding="utf-8"), problems.name, errors)
    for qid, question in bank.items():
        for text in _strings(question):
            if "![" in text:
                check_text(text, f"q{qid}", errors)
