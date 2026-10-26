"""Fork a Study course: copy it under a NEW course id that is yours to change
(2026-10-26, Seth: "they can copy over a course and fork it however they want").

The source is either a folder in this checkout (`study-courses/<id>/`) or,
with `live=True`, any PUBLIC course published live on Delta Drills (fetched
from the server's /contrib/<id>/export). Every id in a Study course starts
with `<course id>.` and every file path with `study-courses/<course id>/`, so
the copy is a rename of those prefixes; prose is left alone.

The fork is a separate course: publishing it never touches the original, and
the original's owner never sees your changes. Built-in courses (shipped in the
app image, not published live) cannot be forked this way.
"""

from __future__ import annotations

import base64
import re
import shutil

from . import paths, study, study_publish


def _overlaps(a: str, b: str) -> bool:
    # Same rule as the backend's contrib_merge.owned, which a publish refuses on.
    return a == b or b.startswith(a + ".") or b.startswith(a + "-")


def _rename(value, old: str, new: str):
    if isinstance(value, dict):   # keys too: `prereqs` is {concept id: weight}
        return {_rename(k, old, new): _rename(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename(v, old, new) for v in value]
    if isinstance(value, str):
        if value == old:
            return new
        if value.startswith(old + "."):
            return new + value[len(old):]
        prefix = f"study-courses/{old}/"
        if value.startswith(prefix):
            return f"study-courses/{new}/" + value[len(prefix):]
    return value


def _local(source: str) -> tuple[dict, dict[str, bytes]]:
    spec = study.load(source)
    home = study.folder(source)
    files = {}
    for top in ("sources", "images"):
        d = home / top
        for f in sorted(d.iterdir()) if d.is_dir() else []:
            if f.is_file() and not f.is_symlink() and not f.name.startswith(".") \
                    and f.name not in ("README.md", "watch.py"):
                files[f"{top}/{f.name}"] = f.read_bytes()
    return spec, files


def _live(source: str, channel: str) -> tuple[dict, dict[str, bytes]]:
    got = study_publish.export(source, channel)
    return got["spec"], {rel: base64.b64decode(data) for rel, data in got.get("files", {}).items()}


def fork(source: str, new: str, title: str | None = None, live: bool = False,
         channel: str = "stable") -> dict:
    study.folder(source)   # validates the id
    home = study.folder(new)
    if len(new) > 40:
        raise ValueError("new course id: 40 characters at most.")
    if _overlaps(source, new) or _overlaps(new, source):
        raise ValueError(f"'{new}' overlaps '{source}' (one id is a prefix of the other at '.' or '-'): "
                         f"pick an id that does not start with '{source}-', e.g. 'my-{source}'.")
    if home.exists():
        raise ValueError(f"study-courses/{new}/ already exists.")

    spec, files = _live(source, channel) if live else _local(source)
    spec = _rename(spec, source, new)
    course = spec["course"]
    course["title"] = (title or f"{course.get('title', source)} (fork)").strip()
    course.pop("private_to", None)   # the original's audience is not the fork's
    course.pop("default_on", None)

    try:
        for top in ("sources", "images"):
            (home / top).mkdir(parents=True, exist_ok=True)
        for rel, data in files.items():
            top, _, name = rel.partition("/")
            if top not in ("sources", "images") or not name or "/" in name or name.startswith("."):
                continue
            (home / top / name).write_bytes(data)
        study.save(new, spec)
    except Exception:
        shutil.rmtree(home, ignore_errors=True)
        raise

    no_derivs = [s["id"] for s in spec.get("sources", [])
                 if re.search(r"\bND\b|NO.?DERIV", str(s.get("license", "")).upper())]
    return {
        "course": new, "forked_from": source, "from": f"live ({channel})" if live else "this checkout",
        "folder": str(home.relative_to(paths.REPO)),
        "counts": {k: len(spec.get(k, [])) for k in ("sections", "concepts", "passages", "questions", "diagrams")},
        "files": len(files),
        **({"verbatim_only_sources": no_derivs,
            "license_note": "These sources are No-Derivatives: keep their passages verbatim; change "
                            "questions and your own text freely."} if no_derivs else {}),
        "next_steps": ["Edit it with the study_* tools (it is yours; the original is untouched).",
                       f"study_check course={new} until ok.",
                       f"study_publish course={new} to put your version live."],
    }
