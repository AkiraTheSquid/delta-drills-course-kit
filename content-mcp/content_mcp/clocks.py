"""The answer clock every concept needs: `lessons/placement_time_caps.json`.

One clock per registry KC, in seconds (1..1200), read by the placement
diagnostic and by ordinary practice. The lessons watcher fails the moment a
KC has none ("no placement clock for …"), so `graph_add_kc` writes one too.
The number is judged from the drills' shape (see the file's own `_comment`):
the short py-0/np-1 answers get 180-300 s, the ARENA exercises up to 1200.
"""

from __future__ import annotations

import json

from . import paths

CAPS = paths.LESSONS / "placement_time_caps.json"
DEFAULT_SECS = 600


def ensure(kc: str, secs: int | None) -> dict:
    """Set `kc`'s clock (or leave an existing one when `secs` is None)."""
    if not CAPS.exists():
        return {"answer_clock": "absent"}
    caps = json.loads(CAPS.read_text(encoding="utf-8"))
    table = caps.setdefault("kcs", {})
    if secs is None and kc in table:
        return {"answer_secs": table[kc]}
    value = DEFAULT_SECS if secs is None else secs
    validate(value)
    table[kc] = value
    indent = paths.json_indent_of(CAPS, default=2)
    CAPS.write_text(json.dumps(caps, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"answer_secs": value}


def validate(secs: int | None) -> None:
    """Refuse a bad clock BEFORE the registry is written."""
    if secs is not None and (not isinstance(secs, int) or isinstance(secs, bool) or not 0 < secs <= 1200):
        raise ValueError("answer_secs must be a whole number of seconds, 1-1200.")
