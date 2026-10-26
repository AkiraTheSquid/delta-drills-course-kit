"""One contributor Study course merged into a pack, and the rules it must pass.

A contributor course is the content MCP's `study-courses/<course>/` folder:
`course.json` (course, sections, sources, concepts, passages, questions,
diagrams, edges), its `sources/` (the text every passage quotes verbatim) and
its `images/` (the pictures its image-occlusion cards hide labels on).

Two callers, one rule set (moved here from scripts/study_course.py 2026-10-22):
  * scripts/study_course.py — `check` in the kit / private repo, `apply` into
    the private repo's live pack;
  * contrib.py — the server, when a contributor PUBLISHES a course live.

Every id a course writes is namespaced by it (`<course>.<id>`, diagrams
`<course>-<id>`), so merging replaces exactly what the course wrote before and
never a concept somebody else did.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.concept_tutor import question_quality, verbatim

PACK_FILES = ("concepts", "edges", "passages", "questions")
META_LISTS = ("courses", "sections", "sources", "diagrams")
MASK_FILL, MASK_LINE = "#FFEBA2", "#212121"   # Anki image occlusion's colours, as bio_label_questions


def owned(course: str, item_id: str) -> bool:
    return item_id == course or item_id.startswith(course + ".") or item_id.startswith(course + "-")


def owned_ids(files: dict, course: str) -> set:
    return ({x["id"] for name in PACK_FILES for x in files[name] if owned(course, x.get("id", ""))}
            | {x["id"] for key in META_LISTS for x in files["pack"].get(key, []) if owned(course, x["id"])})


def read_pack(folder: Path) -> dict:
    files = {"pack": json.loads((folder / "pack.json").read_text(encoding="utf-8"))}
    for name in PACK_FILES:
        f = folder / f"{name}.json"
        files[name] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    return files


def merge(files: dict, spec: dict) -> dict:
    """The pack with this course's previous copy removed and the spec added.
    `files` is not modified."""
    cid = spec["course"]["id"]
    meta = dict(files["pack"])
    meta["courses"] = [k for k in meta.get("courses", []) if k["id"] != cid]
    for key in ("sections", "sources", "diagrams"):
        meta[key] = [x for x in meta.get(key, []) if not owned(cid, x["id"])]
    out = {name: [x for x in files[name] if not owned(cid, x.get("id", ""))] for name in PACK_FILES}
    out["edges"] = [e for e in files["edges"] if not (owned(cid, e["a"]) or owned(cid, e["b"]))]

    sections = spec.get("sections", [])
    meta["sections"] += sections
    k = spec["course"]
    meta["courses"].append({kk: v for kk, v in {
        "id": cid, "title": k["title"], "eyebrow": k.get("eyebrow", ""),
        "sections": [s["id"] for s in sections], "public": bool(k.get("public", False)),
        "default_on": bool(k.get("default_on", False)), "language": k.get("language", ""),
        # private_to = ONLY these emails see it (access.sees), invites or not
        "private_to": [str(e) for e in k.get("private_to") or []],
    }.items() if v not in ("", [])})
    meta["sources"] += spec.get("sources", [])
    meta["diagrams"] += [{kk: v for kk, v in d.items() if kk != "image"} for d in spec.get("diagrams", [])]
    top = max([c.get("order", 0) for c in out["concepts"]] or [0])
    out["concepts"] += [{**c, "order": top + 1 + i, "aliases": _free_aliases(c, cid, out["concepts"])}
                        for i, c in enumerate(spec.get("concepts", []))]
    out["passages"] += spec.get("passages", [])
    out["questions"] += spec.get("questions", [])
    out["edges"] += spec.get("edges", [])
    out["pack"] = meta
    return out


def _free_aliases(concept: dict, course: str, others: list) -> list:
    """An alias names ONE concept pack-wide (pack.problems), and a contributor
    cannot see the private courses: "Nucleus" is already AP Biology's. So an
    alias another concept owns is dropped here, at merge time — the kit's check
    and the server agree — and a concept left with none is called
    "<title> (<course>)"."""
    taken = {a.casefold() for c in others for a in c.get("aliases", [])}
    keep = [a for a in concept.get("aliases", []) if a.casefold() not in taken]
    return keep or [f"{concept['title']} ({course})"]


def model(files: dict):
    from app.concept_tutor.pack import Pack
    return Pack(**files["pack"], **{name: files[name] for name in PACK_FILES})


def source_file(course_dir: Path, cid: str, path: str) -> Path:
    """A source is a file in the course's OWN sources/ folder — the path
    study_source_add records — and nothing else (no '..', no other folder)."""
    prefix = f"study-courses/{cid}/sources/"
    name = path[len(prefix):] if path.startswith(prefix) else ""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise FileNotFoundError(f"source path {path!r} must be {prefix}<file>")
    return course_dir / "sources" / name


def _foreign_ids(spec: dict, cid: str) -> list[str]:
    """Ids the spec writes OUTSIDE its own namespace. merge() only removes the
    course's own ids, so a foreign one would sit beside (or shadow) someone
    else's concept, question or section."""
    bad = [x.get("id", "?") for name in ("concepts", "passages", "questions", "sections", "sources", "diagrams")
           for x in spec.get(name, []) if not owned(cid, str(x.get("id", "")))]
    # BOTH ends: merge() drops every edge touching the course, so an edge to
    # another course would be erased whenever that course is merged again.
    bad += [f"edge {e.get('a')}→{e.get('b')}" for e in spec.get("edges", [])
            if not (owned(cid, str(e.get("a", ""))) and owned(cid, str(e.get("b", ""))))]
    return bad


def check(spec: dict, course_dir: Path, base: dict, prior: set, strict: bool = False) -> dict:
    """The course merged into `base` against the pack's own rules. `prior` =
    ids under the course's prefix that `base` holds AS this course's (its last
    apply); any other id there belongs to someone else, and merging would
    delete it — so that is refused. `strict` (the server's publish): the Jev
    question rules must run, so a missing TypeSafe key refuses the course."""
    from pydantic import ValidationError
    from app.concept_tutor.pack import problems

    cid = spec["course"]["id"]
    taken = sorted(owned_ids(base, cid) - set(prior))
    if taken:
        return {"ok": False, "course": cid, "errors": [
            f"course id {cid!r} collides with content already in the pack ({', '.join(taken[:5])}"
            f"{' …' if len(taken) > 5 else ''}) — pick another course id"]}
    foreign = _foreign_ids(spec, cid)
    if foreign:
        return {"ok": False, "course": cid, "errors": [
            f"ids outside course {cid!r}: {', '.join(foreign[:5])}{' …' if len(foreign) > 5 else ''}"]}
    errors: list[str] = []
    try:
        before = set(problems(model(base)))
    except ValidationError:
        before = set()
    merged = merge(base, spec)
    try:
        pack = model(merged)
    except ValidationError as exc:
        for e in exc.errors()[:40]:
            loc = e["loc"]
            # loc = (field, index, ...): name the course's own item, not a pack-wide index
            item = ""
            if len(loc) >= 2 and isinstance(loc[1], int) and loc[0] in PACK_FILES:
                item = merged[loc[0]][loc[1]].get("id", "")
            errors.append(f"{item or '.'.join(map(str, loc[:2]))}: {'.'.join(map(str, loc[2:]))} {e['msg']}")
        return {"ok": False, "course": cid, "errors": errors}
    errors += [p for p in problems(pack) if p not in before]
    # A course's questions grade only its own concepts: a target in another
    # course would move mastery in a course the contributor cannot see.
    for q in spec.get("questions", []):
        for t in q.get("targets", []):
            if not owned(cid, t):
                errors.append(f"question {q['id']}: targets {t!r}, which is not in course {cid!r}")
    # Every question rule (lead-in, no flashcards, expectations, the Jev
    # judgements) lives in question_rules.json; one judge for MCP, CLI and publish.
    quality = question_quality.judge(spec, strict=strict)
    errors += quality["errors"]

    ids = {x["id"] for x in spec.get("passages", [])}
    texts: dict[str, str] = {}
    for x in (x for x in pack.passages if x.id in ids):
        src = pack.source(x.source)
        n = len(x.text.split())
        if n > verbatim.MAX_WORDS:
            errors.append(f"passage {x.id}: {n} words (cap {verbatim.MAX_WORDS})")
        try:
            if src.path not in texts:
                f = source_file(course_dir, cid, src.path)
                if not f.is_file():
                    raise FileNotFoundError(f"source file missing: {src.path}")
                texts[src.path] = verbatim.norm(verbatim.text_of(f.read_text(encoding="utf-8", errors="replace"), f.suffix))
        except FileNotFoundError as exc:
            errors.append(f"passage {x.id}: {exc}")
            continue
        if not verbatim.contains(texts[src.path], x.text):
            errors.append(f"passage {x.id} is NOT verbatim in {src.path}: {verbatim.near(x.text, texts[src.path])}")
    used = {pid for c in spec.get("concepts", []) for pid in c.get("passages", [])}
    errors += [f"passage {pid} is used by no concept" for pid in sorted(ids - used)]

    from PIL import Image
    for d in spec.get("diagrams", []):
        f = image_file(course_dir, d)
        if f is None:
            errors.append(f"diagram {d['id']}: picture {d.get('image')} is missing")
            continue
        try:
            with Image.open(f) as img:
                size = img.size
        except Exception as exc:
            errors.append(f"diagram {d['id']}: picture {d.get('image')} is unreadable ({exc})")
            continue
        if size != (d["w"], d["h"]):
            errors.append(f"diagram {d['id']}: picture is {size}, boxes were drawn on {d['w']}x{d['h']}")
    by_format: dict[str, int] = {}
    for q in spec.get("questions", []):
        by_format[q.get("format", "free")] = by_format.get(q.get("format", "free"), 0) + 1
    # Not an error (a private course may hold the alias), but say so: a dropped
    # alias stops matching answers in this course.
    kept = {c["id"]: {a.casefold() for a in c["aliases"]} for c in merged["concepts"] if owned(cid, c["id"])}
    warnings = [f"concept {c['id']}: alias {a!r} is taken elsewhere in the pack and is dropped"
                for c in spec.get("concepts", []) for a in c.get("aliases", [])
                if a.casefold() not in kept.get(c["id"], set())] + quality["warnings"]
    return {"ok": not errors, "course": cid, "errors": errors, "warnings": warnings,
            "concepts": len(spec.get("concepts", [])), "questions": len(spec.get("questions", [])),
            "by_format": by_format, "quality": quality["scores"]}


def image_file(course_dir: Path, diagram: dict) -> Path | None:
    """A diagram's picture: `images/<file>` inside the course folder, nowhere else."""
    rel = str(diagram.get("image", ""))
    name = rel[len("images/"):] if rel.startswith("images/") else ""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    f = course_dir / "images" / name
    return f if f.is_file() else None


def paint(spec: dict, course_dir: Path, out: Path) -> None:
    """`<id>.webp` and `<id>.masked.webp` (every box painted over) per diagram."""
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    for d in spec.get("diagrams", []):
        with Image.open(image_file(course_dir, d)) as src:
            img = src.convert("RGB")
        masked = img.copy()
        draw = ImageDraw.Draw(masked)
        for b in d["boxes"]:
            draw.rectangle(b, fill=MASK_FILL, outline=MASK_LINE, width=1)
        img.save(out / f"{d['id']}.webp", quality=88, method=6)
        masked.save(out / f"{d['id']}.masked.webp", quality=88, method=6)
