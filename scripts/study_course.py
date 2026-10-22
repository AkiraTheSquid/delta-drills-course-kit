#!/usr/bin/env python3
"""A contributor's Study course (study-courses/<course>/) against the real pack rules.

The content MCP keeps a Study course as ONE tracked source set —
`study-courses/<course>/course.json` plus its `sources/` (the files every
passage is quoted from, verbatim) and `images/` (the pictures its image-
occlusion cards hide labels on). The MCP itself is stdlib-only; everything
here needs pydantic (the pack model) or Pillow (the pictures), so the MCP
runs this file under the content python and reads one JSON object back.

    prepare-image SRC DEST           copy a picture in, RGB, longest side <= 1600 px
    preview COURSE_DIR DIAGRAM OUT   the picture with a coordinate grid and every box numbered
    check COURSE_DIR BASE_PACK       merge into a copy of the base pack: pack.problems()
                                     (only faults this course adds), verbatim passages,
                                     pictures present and the size the boxes were drawn on
    apply COURSE_DIR PACK_DIR        the same merge, WRITTEN into the live pack, diagrams painted

`apply` replaces everything the course owned before (ids under `<course>.`,
diagrams under `<course>-`), so it is re-runnable: a pack builder rebuild
drops contributor courses (and bio_label_questions.write_images deletes their
pictures), and running apply again puts them back unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "This-Directory-Only" / "backend"))
sys.path.insert(0, str(REPO / "This-Directory-Only" / "scripts" / "concept_packs"))

MAX_SIDE = 1600
MASK_FILL, MASK_LINE = "#FFEBA2", "#212121"   # Anki image occlusion's colours, as bio_label_questions
PACK_FILES = ("concepts", "edges", "passages", "questions")
# Which contributor courses `apply` has written into a pack. Only those may be
# replaced: a course id that is merely a PREFIX the pack already uses
# ("bioprok") would otherwise delete that course's concepts on apply. A
# subfolder, not a *.json beside the pack files, so publish.py never ships it.
APPLIED = Path(".contrib") / "applied.json"


def _out(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def prepare_image(src: str, dest: str) -> dict:
    from PIL import Image
    img = Image.open(src)
    img.load()
    img = img.convert("RGB")
    scale = min(1.0, MAX_SIDE / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG", optimize=True)
    return {"w": img.width, "h": img.height, "scaled_by": round(scale, 4)}


def _course(course_dir: Path) -> dict:
    return json.loads((course_dir / "course.json").read_text(encoding="utf-8"))


def preview(course_dir: Path, did: str, out: str) -> dict:
    from PIL import Image, ImageDraw
    spec = _course(course_dir)
    d = next((x for x in spec.get("diagrams", []) if x["id"] == did), None)
    if d is None:
        raise SystemExit(f"no diagram {did}")
    img = Image.open(course_dir / d["image"]).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    step = 50 if max(img.size) <= 800 else 100
    for x in range(0, img.width, step):
        draw.line([(x, 0), (x, img.height)], fill=(0, 120, 255, 70))
        draw.text((x + 2, 2), str(x), fill=(0, 80, 200, 255))
    for y in range(0, img.height, step):
        draw.line([(0, y), (img.width, y)], fill=(0, 120, 255, 70))
        draw.text((2, y + 2), str(y), fill=(0, 80, 200, 255))
    for i, b in enumerate(d["boxes"]):
        draw.rectangle(b, outline=(230, 0, 0, 255), width=3)
        ty = b[1] - 16 if b[1] >= 16 else b[3] + 2   # the number sits OUTSIDE the box, never on its text
        draw.rectangle([b[0], ty, b[0] + 26, ty + 14], fill=(230, 0, 0, 255))
        draw.text((b[0] + 3, ty + 1), str(i), fill=(255, 255, 255, 255))
    img.save(out, format="PNG")
    return {"preview": out, "w": img.width, "h": img.height,
            "boxes": [{"box": i, "label": lab, "xyxy": b} for i, (b, lab) in enumerate(zip(d["boxes"], d["labels"]))]}


def _read_pack(folder: Path) -> dict:
    files = {"pack": json.loads((folder / "pack.json").read_text(encoding="utf-8"))}
    for name in PACK_FILES:
        f = folder / f"{name}.json"
        files[name] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    return files


def _owned(course: str, item_id: str) -> bool:
    return item_id == course or item_id.startswith(course + ".") or item_id.startswith(course + "-")


def _owned_ids(files: dict, course: str) -> set:
    return ({x["id"] for name in PACK_FILES for x in files[name] if _owned(course, x.get("id", ""))}
            | {x["id"] for key in ("courses", "sections", "sources", "diagrams")
               for x in files["pack"].get(key, []) if _owned(course, x["id"])})


def _applied(pack_dir: Path) -> dict:
    """{course: [every id its last apply wrote]}. An id under the course's
    prefix that is NOT in that list belongs to a builder, and apply would
    delete it — so check refuses instead."""
    f = pack_dir / APPLIED
    data = json.loads(f.read_text()) if f.exists() else {}
    return data if isinstance(data, dict) else {}


def merge(files: dict, spec: dict) -> dict:
    """The base pack with this course's previous copy removed and the spec added."""
    cid = spec["course"]["id"]
    meta = dict(files["pack"])
    meta["courses"] = [k for k in meta.get("courses", []) if k["id"] != cid]
    meta["sections"] = [s for s in meta.get("sections", []) if not _owned(cid, s["id"])]
    meta["sources"] = [s for s in meta.get("sources", []) if not _owned(cid, s["id"])]
    meta["diagrams"] = [d for d in meta.get("diagrams", []) if not _owned(cid, d["id"])]
    out = {name: [x for x in files[name] if not _owned(cid, x.get("id", ""))] for name in PACK_FILES}
    out["edges"] = [e for e in files["edges"] if not (_owned(cid, e["a"]) or _owned(cid, e["b"]))]

    sections = spec.get("sections", [])
    meta["sections"] += sections
    k = spec["course"]
    meta["courses"].append({kk: v for kk, v in {
        "id": cid, "title": k["title"], "eyebrow": k.get("eyebrow", ""),
        "sections": [s["id"] for s in sections], "public": bool(k.get("public", False)),
        "default_on": bool(k.get("default_on", False)),
    }.items() if v not in ("",)})
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
    and the private repo's apply then agree — and a concept left with none is
    called "<title> (<course>)"."""
    taken = {a.casefold() for c in others for a in c.get("aliases", [])}
    keep = [a for a in concept.get("aliases", []) if a.casefold() not in taken]
    return keep or [f"{concept['title']} ({course})"]


def _model(files: dict):
    from app.concept_tutor.pack import Pack
    return Pack(**files["pack"], **{name: files[name] for name in PACK_FILES})


def check(course_dir: Path, base: Path) -> dict:
    from pydantic import ValidationError
    from app.concept_tutor.pack import problems
    import check_pack

    spec = _course(course_dir)
    cid = spec["course"]["id"]
    base_files = _read_pack(base)
    errors: list[str] = []
    taken = sorted(_owned_ids(base_files, cid) - set(_applied(base).get(cid, [])))
    if taken:
        return {"ok": False, "errors": [
            f"course id {cid!r} collides with content already in the pack ({', '.join(taken[:5])}"
            f"{' …' if len(taken) > 5 else ''}) — pick another course id"]}
    try:
        before = set(problems(_model(base_files)))
    except ValidationError:
        before = set()
    merged = merge(base_files, spec)
    try:
        pack = _model(merged)
    except ValidationError as exc:
        for e in exc.errors()[:40]:
            loc = e["loc"]
            # loc = (field, index, ...): name the course's own item, not a pack-wide index
            item = ""
            if len(loc) >= 2 and isinstance(loc[1], int) and loc[0] in PACK_FILES:
                item = merged[loc[0]][loc[1]].get("id", "")
            errors.append(f"{item or '.'.join(map(str, loc[:2]))}: {'.'.join(map(str, loc[2:]))} {e['msg']}")
        return {"ok": False, "errors": errors}
    errors += [p for p in problems(pack) if p not in before]
    # A course's questions grade only its own concepts: a target in another
    # course would move mastery in a course the contributor cannot see.
    for q in spec.get("questions", []):
        for t in q.get("targets", []):
            if not _owned(cid, t):
                errors.append(f"question {q['id']}: targets {t!r}, which is not in course {cid!r}")

    ids = {x["id"] for x in spec.get("passages", [])}
    for x in (x for x in pack.passages if x.id in ids):
        src = pack.source(x.source)
        n = len(x.text.split())
        if n > check_pack.MAX_WORDS:
            errors.append(f"passage {x.id}: {n} words (cap {check_pack.MAX_WORDS})")
        try:
            text = check_pack.norm_source(src.path)
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(f"passage {x.id}: {exc}")
            continue
        if not check_pack.contains(text, x.text):
            errors.append(f"passage {x.id} is NOT verbatim in {src.path}: {check_pack._near(x.text, text)}")
    used = {pid for c in spec.get("concepts", []) for pid in c.get("passages", [])}
    errors += [f"passage {pid} is used by no concept" for pid in sorted(ids - used)]

    from PIL import Image
    for d in spec.get("diagrams", []):
        f = course_dir / d.get("image", "")
        if not f.is_file():
            errors.append(f"diagram {d['id']}: picture {d.get('image')} is missing")
            continue
        with Image.open(f) as img:
            if img.size != (d["w"], d["h"]):
                errors.append(f"diagram {d['id']}: picture is {img.size}, boxes were drawn on {d['w']}x{d['h']}")
    counts = {"concepts": len(spec.get("concepts", [])), "questions": len(spec.get("questions", []))}
    by_format: dict[str, int] = {}
    for q in spec.get("questions", []):
        by_format[q.get("format", "free")] = by_format.get(q.get("format", "free"), 0) + 1
    return {"ok": not errors, "course": cid, "errors": errors, **counts, "by_format": by_format}


def apply(course_dir: Path, pack_dir: Path) -> dict:
    from PIL import Image, ImageDraw
    result = check(course_dir, pack_dir)
    if not result["ok"]:
        return {**result, "applied": False}
    spec = _course(course_dir)
    merged = merge(_read_pack(pack_dir), spec)

    def put(target: Path, text: str) -> None:
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)

    # Pictures FIRST: a failure here leaves the pack JSON untouched, so the pack
    # never names a diagram whose files were not written.
    out = pack_dir / "diagrams"
    out.mkdir(exist_ok=True)
    for d in spec.get("diagrams", []):
        img = Image.open(course_dir / d["image"]).convert("RGB")
        masked = img.copy()
        draw = ImageDraw.Draw(masked)
        for b in d["boxes"]:
            draw.rectangle(b, fill=MASK_FILL, outline=MASK_LINE, width=1)
        img.save(out / f"{d['id']}.webp", quality=88, method=6)
        masked.save(out / f"{d['id']}.masked.webp", quality=88, method=6)
    # Ownership record before the pack: if the JSON writes stop half way, the
    # next apply still knows these ids are this course's and can finish.
    record = pack_dir / APPLIED
    record.parent.mkdir(exist_ok=True)
    applied = _applied(pack_dir)
    cid = spec["course"]["id"]
    applied[cid] = sorted(set(applied.get(cid, [])) | _owned_ids(merged, cid))
    put(record, json.dumps(applied, indent=1, sort_keys=True) + "\n")
    for name in ("pack", *PACK_FILES):
        put(pack_dir / f"{name}.json", json.dumps(merged[name], ensure_ascii=False, indent=2) + "\n")
    return {**result, "applied": True, "pack": str(pack_dir)}


def main(argv: list[str]) -> int:
    cmd, *args = argv
    try:
        if cmd == "prepare-image":
            _out(prepare_image(*args))
        elif cmd == "preview":
            _out(preview(Path(args[0]), args[1], args[2]))
        elif cmd == "check":
            _out(check(Path(args[0]), Path(args[1])))
        elif cmd == "apply":
            _out(apply(Path(args[0]), Path(args[1])))
        else:
            raise SystemExit(f"unknown command {cmd}")
    except Exception as exc:  # one JSON line back to the MCP, whatever broke
        _out({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
