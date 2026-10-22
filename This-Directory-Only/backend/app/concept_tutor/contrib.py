"""Contributor Study courses published LIVE, without a deploy (2026-10-22).

A contributor (Ethan, from the public course kit) runs `study_publish` in the
content MCP; it POSTs the course folder to contrib_router.py, which lands here.
The course is checked by the same rules as the kit's `study_check`
(contrib_merge.check, against the REAL pack this time: private courses
included), its diagrams painted, and it is stored on the volume:

    <STORAGE_DIR>/study-contrib/<pack_id>/<course>/
        course.json  sources/  images/  diagrams/<id>.webp + .masked.webp
        owner.json   {"contributor": name, "published_at": iso}
    <STORAGE_DIR>/study-contrib/<pack_id>/.stamp

pack.load merges every stored course into the baked pack (`build`) and
rebuilds when `.stamp` moves, so a publish is live on the next request. The
volume survives deploys and pack rebuilds; the image does not need to change.

A stored course that stops merging cleanly (say a rebuilt pack took one of its
aliases) is SKIPPED at load and logged — contributed content can never take
the Study pack down.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.concept_tutor import contrib_merge as cm

log = logging.getLogger(__name__)

COURSE_RE = re.compile(r"(?=.{1,60}$)[a-z][a-z0-9]*(-[a-z0-9]+)*")
FILE_RE = re.compile(r"(sources|images)/[a-z0-9][a-z0-9._-]{0,80}")
SOURCE_EXT = {".md", ".txt", ".html", ".htm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_FILES = 80
MAX_BYTES = 25 * 1024 * 1024        # the whole upload, decoded
MAX_SIDE, MAX_PIXELS = 4000, 16_000_000
_publishing = threading.Lock()      # one publish at a time: each holds a raw pack copy (~400 MB)
_live: dict[str, set[str]] = {}     # pack id -> stored courses its last build merged


def root() -> Path | None:
    env = os.environ.get("STUDY_CONTRIB_DIR")
    if env:
        return Path(env)
    try:
        from app.config import settings
    except Exception:   # the course kit ships pack.py without the app's settings
        return None
    return Path(settings.storage_dir) / "study-contrib"


def _pack_dir(pack_id: str) -> Path | None:
    r = root()
    return r / pack_id if r is not None else None


def stamp(pack_id: str) -> int:
    d = _pack_dir(pack_id)
    try:
        return (d / ".stamp").stat().st_mtime_ns if d is not None else 0
    except OSError:
        return 0


def _touch(pack_id: str) -> None:
    f = _pack_dir(pack_id) / ".stamp"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(str(time.time_ns()))
    os.utime(f, ns=(time.time_ns(), time.time_ns()))


def stored(pack_id: str) -> list[tuple[str, Path, dict]]:
    """(course id, folder, spec) for every course stored for this pack."""
    d = _pack_dir(pack_id)
    if d is None or not d.is_dir():
        return []
    out = []
    for sub in sorted(d.iterdir()):
        if not sub.is_dir() or not COURSE_RE.fullmatch(sub.name):
            continue
        try:
            spec = json.loads((sub / "course.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("study-contrib %s: unreadable course.json (%s); skipped", sub.name, exc)
            continue
        if isinstance(spec, dict) and isinstance(spec.get("course"), dict) and spec["course"].get("id") == sub.name:
            out.append((sub.name, sub, spec))
    return out


def build(pack_id: str, files: dict, make: Callable[[dict], object]):
    """The baked pack plus every stored course. `make(files)` builds and checks
    a Pack (raises on any fault). All courses at once first — one build in the
    normal case — and course by course only when that fails."""
    courses = stored(pack_id)
    _live[pack_id] = set()
    if courses:
        try:   # merge too: a malformed course.json raises here, and must reach the fallback
            merged = files
            for _, _, spec in courses:
                merged = cm.merge(merged, spec)
            pack = make(merged)
            _live[pack_id] = {cid for cid, _, _ in courses}
            return pack
        except Exception as exc:
            log.warning("study-contrib: merging all %d courses failed (%s); trying one by one", len(courses), exc)
    pack = make(files)              # a broken BAKED pack still raises, as before
    live = set()
    for cid, _, spec in courses:
        try:
            cand = cm.merge(files, spec)
            pack, files = make(cand), cand
            live.add(cid)
        except Exception as exc:
            log.warning("study-contrib %s: does not merge (%s); skipped", cid, exc)
    _live[pack_id] = live
    return pack


def diagram_file(pack_id: str, did: str, masked: bool) -> Path | None:
    """A stored course's diagram picture, or None (then the baked one is used)."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", did or ""):
        return None
    d = _pack_dir(pack_id)
    if d is None or not d.is_dir():
        return None
    for sub in d.iterdir():
        # only a course the current build MERGED: a skipped one never serves pictures
        if sub.name in _live.get(pack_id, ()) and cm.owned(sub.name, did):
            f = sub / "diagrams" / f"{did}{'.masked' if masked else ''}.webp"
            if f.is_file():
                return f
    return None


@contextmanager
def _locked(home: Path):
    """The thread lock, plus a file lock on the store, so a second server
    process (uvicorn --workers) cannot interleave a publish either."""
    import fcntl
    home.mkdir(parents=True, exist_ok=True)
    with _publishing, open(home / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield


class Refused(ValueError):
    """A publish the contributor must fix (400), not a server fault."""


def _owner(folder: Path) -> dict:
    try:
        return json.loads((folder / "owner.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _check_files(files: dict[str, bytes]) -> None:
    if len(files) > MAX_FILES:
        raise Refused(f"{len(files)} files; at most {MAX_FILES}.")
    if sum(len(b) for b in files.values()) > MAX_BYTES:
        raise Refused(f"upload is over {MAX_BYTES // (1024 * 1024)} MB.")
    for rel in files:
        if not FILE_RE.fullmatch(rel) or ".." in rel:
            raise Refused(f"file {rel!r}: only sources/<name> and images/<name> (lowercase, no folders).")
        ext = Path(rel).suffix.lower()
        if ext not in (SOURCE_EXT if rel.startswith("sources/") else IMAGE_EXT):
            raise Refused(f"file {rel!r}: a source is .md/.txt/.html, a picture .png/.jpg/.webp/.gif.")


def _check_images(stage: Path) -> None:
    from PIL import Image
    for f in sorted((stage / "images").glob("*")):
        try:
            with Image.open(f) as img:
                if img.format not in ("PNG", "JPEG", "WEBP", "GIF"):
                    raise Refused(f"{f.name}: not a PNG, JPEG, WebP or GIF.")
                w, h = img.size
                if max(w, h) > MAX_SIDE or w * h > MAX_PIXELS:
                    raise Refused(f"{f.name}: {w}x{h} is too large (longest side {MAX_SIDE}).")
                img.verify()
        except Refused:
            raise
        except Exception as exc:
            raise Refused(f"{f.name}: not a readable picture ({exc}).") from None


def _prefix_clash(cid: str, others: set[str]) -> str | None:
    for o in sorted(others):
        if o != cid and (cm.owned(o, cid) or cm.owned(cid, o)):
            return o
    return None


def publish(pack_id: str, contributor: str, spec: dict, files: dict[str, bytes],
            packs_dir: Path, applied: dict, loader: Callable[[str], object]) -> dict:
    """Check, store and go live. `applied` = the baked pack's .contrib record
    ({course: ids its private-repo apply wrote}), so a course first shipped by
    pull request can move to live publishing. Returns check's result plus
    `live`; a Refused is the contributor's to fix."""
    cid = str(spec.get("course", {}).get("id", ""))
    if not COURSE_RE.fullmatch(cid):
        raise Refused(f"course id {cid!r}: lowercase letters/digits and dashes.")
    # default_on would switch the course on for every learner who never chose
    # their courses — a contributor may list a course, never push it on anyone.
    spec = {**spec, "course": {**spec["course"], "default_on": False}}
    _check_files(files)
    home = _pack_dir(pack_id)
    if home is None:
        raise RuntimeError("no storage directory for contributor courses")
    with _locked(home):
        final = home / cid
        owner = _owner(final).get("contributor") if final.exists() else contributor
        if owner != contributor:   # a stored course with no readable owner fails CLOSED
            raise Refused(f"course {cid!r} was published by another contributor."
                          + ("" if owner else " (its ownership record is unreadable; ask Seth)"))
        baked = cm.read_pack(packs_dir / pack_id)
        others = [(o, d, s) for o, d, s in stored(pack_id) if o != cid]
        # A baked course with THIS id (one first shipped by pull request) is
        # not a clash: check()'s collision guard decides, from `applied`.
        clash = _prefix_clash(cid, {k["id"] for k in baked["pack"].get("courses", [])}
                              | {o for o, _, _ in others})
        if clash:
            raise Refused(f"course id {cid!r} overlaps course {clash!r} (one id is a prefix of the other) — pick another id.")
        base = baked
        for _, _, s in others:
            base = cm.merge(base, s)

        stage = home / ".incoming" / f"{cid}-{uuid.uuid4().hex[:8]}"
        stage.mkdir(parents=True)
        try:
            (stage / "course.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
            for rel, data in files.items():
                (stage / rel).parent.mkdir(exist_ok=True)
                (stage / rel).write_bytes(data)
            _check_images(stage)
            result = cm.check(spec, stage, base, set(applied.get(cid, [])))
            del base, baked
            if not result["ok"]:
                return {**result, "live": False}
            cm.paint(spec, stage, stage / "diagrams")
            (stage / "owner.json").write_text(json.dumps({
                "contributor": contributor, "published_at": datetime.now(timezone.utc).isoformat()}))
            if not _swap_in(pack_id, final, stage, loader):
                return {**result, "ok": False, "live": False,
                        "errors": ["the course passed the check but did not load; the live copy is unchanged (see the server log)"]}
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
        return {**result, "live": True}


def _swap_in(pack_id: str, final: Path, new: Path | None, loader: Callable[[str], object]) -> bool:
    """Replace `final` with `new` (None = remove it) and reload. On ANY failure,
    or when `new` does not come out of the load merged, the previous copy is put
    back and reloaded, so the store never ends half-changed. True = done."""
    old = final.parent / ".incoming" / f"{final.name}-old-{uuid.uuid4().hex[:8]}"
    old.parent.mkdir(parents=True, exist_ok=True)
    had = final.exists()
    if had:
        final.rename(old)
    try:
        if new is not None:
            new.rename(final)
        _touch(pack_id)
        loader(pack_id)
        ok = new is None or final.name in _live.get(pack_id, set())
    except Exception:
        log.exception("study-contrib %s: swap failed; restoring the previous copy", final.name)
        ok = False
    if not ok:
        shutil.rmtree(final, ignore_errors=True)
        if had:
            old.rename(final)
        _touch(pack_id)
        try:
            loader(pack_id)
        except Exception:
            log.exception("study-contrib: reload after restoring %s failed", final.name)
        return False
    shutil.rmtree(old, ignore_errors=True)
    return True


def unpublish(pack_id: str, contributor: str, cid: str, loader: Callable[[str], object]) -> dict:
    if not COURSE_RE.fullmatch(cid or ""):
        raise Refused(f"course id {cid!r}: lowercase letters/digits and dashes.")
    home = _pack_dir(pack_id)
    if home is None:
        raise RuntimeError("no storage directory for contributor courses")
    with _locked(home):
        final = home / cid
        if not final.is_dir():
            raise Refused(f"course {cid!r} is not published.")
        if _owner(final).get("contributor") != contributor:
            raise Refused(f"course {cid!r} was published by another contributor.")
        if not _swap_in(pack_id, final, None, loader):
            raise RuntimeError(f"could not take {cid!r} down; it is still live")
        # A course first shipped in the image (pull request + apply) is still
        # served from there: say so instead of claiming it is gone.
        live = any(c.id == cid for c in loader(pack_id).courses)
        return {"ok": True, "course": cid, "live": live,
                **({"note": "a copy of this course is built into the app; ask Seth to remove it"} if live else {})}


def status(pack_id: str, cid: str, pack) -> dict:
    final = (_pack_dir(pack_id) or Path("/nonexistent")) / cid
    course = next((c for c in pack.courses if c.id == cid), None)
    return {"course": cid, "stored": final.is_dir(), "live": course is not None,
            "questions": sum(1 for q in pack.questions if cm.owned(cid, q.id)),
            "public": bool(course and course.public), **_owner(final)}
