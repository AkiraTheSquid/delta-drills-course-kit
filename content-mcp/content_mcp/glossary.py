"""Keep `lessons/glossary.js`'s `kcLesson` map in step with the registry.

That map is a hand-held COPY of the registry (kc -> [unit title, kc title]) so
a hover popup costs no network request, and `watch_jargon.py` fails the
lessons watcher the moment a KC is missing from it ("kcLesson is missing …").
Every registry write here therefore writes the copy too; without it, adding a
concept through `graph_add_kc` could never pass `pipeline_watchers`.
"""

from __future__ import annotations

import json
import re

from . import paths

HEAD = "window.DD_GLOSSARY.kcLesson = {"
ENTRY = re.compile(r'^(\s*)"([\w.-]+)":\s*\[.*\],?\s*$')


def _line(kc: str, unit_title: str, kc_title: str) -> str:
    return f'  {json.dumps(kc)}: [{json.dumps(unit_title, ensure_ascii=False)}, ' \
           f'{json.dumps(kc_title, ensure_ascii=False)}]'


def sync(kc: str, registry: dict) -> dict:
    """Insert or rewrite the one entry for `kc`. A no-op if glossary.js has no map."""
    if not paths.GLOSSARY.exists():
        return {"glossary": "absent"}
    text = paths.GLOSSARY.read_text(encoding="utf-8")
    start = text.find(HEAD)
    if start < 0:
        return {"glossary": "no kcLesson map"}
    end = text.find("\n};", start)
    if end < 0:
        raise ValueError("glossary.js: kcLesson map has no closing '};' — refusing to edit it.")
    node = next(k for k in registry["kcs"] if k["id"] == kc)
    unit = next(les for les in registry["lessons"] if les["id"] == node["lesson"])
    body = text[start + len(HEAD):end].strip("\n").split("\n")
    fresh = _line(kc, unit["title"], node["title"])
    for i, line in enumerate(body):
        match = ENTRY.match(line)
        if match and match.group(2) == kc:
            body[i] = fresh + ("," if line.rstrip().endswith(",") else "")
            action = "updated"
            break
    else:
        if body and body[-1].strip() and not body[-1].rstrip().endswith(","):
            body[-1] = body[-1].rstrip() + ","
        body.append(fresh)
        action = "added"
    new_text = text[:start + len(HEAD)] + "\n" + "\n".join(body) + text[end:]
    paths.GLOSSARY.write_text(new_text, encoding="utf-8")
    return {"glossary": action}
