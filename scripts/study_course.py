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

MAX_SIDE = 1600
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


def _applied(pack_dir: Path) -> dict:
    """{course: [every id its last apply wrote]}. An id under the course's
    prefix that is NOT in that list belongs to a builder, and apply would
    delete it — so check refuses instead."""
    f = pack_dir / APPLIED
    data = json.loads(f.read_text()) if f.exists() else {}
    return data if isinstance(data, dict) else {}


def check(course_dir: Path, base: Path) -> dict:
    """The rules live in the backend (contrib_merge.check), so this check and
    the server's publish check are the same code."""
    from app.concept_tutor import contrib_merge
    spec = _course(course_dir)
    return contrib_merge.check(spec, course_dir, contrib_merge.read_pack(base),
                               set(_applied(base).get(spec["course"]["id"], [])))


def apply(course_dir: Path, pack_dir: Path) -> dict:
    from app.concept_tutor import contrib_merge
    result = check(course_dir, pack_dir)
    if not result["ok"]:
        return {**result, "applied": False}
    spec = _course(course_dir)
    merged = contrib_merge.merge(contrib_merge.read_pack(pack_dir), spec)

    def put(target: Path, text: str) -> None:
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)

    # Pictures FIRST: a failure here leaves the pack JSON untouched, so the pack
    # never names a diagram whose files were not written.
    contrib_merge.paint(spec, course_dir, pack_dir / "diagrams")
    # Ownership record before the pack: if the JSON writes stop half way, the
    # next apply still knows these ids are this course's and can finish.
    record = pack_dir / APPLIED
    record.parent.mkdir(exist_ok=True)
    applied = _applied(pack_dir)
    cid = spec["course"]["id"]
    applied[cid] = sorted(set(applied.get(cid, [])) | contrib_merge.owned_ids(merged, cid))
    put(record, json.dumps(applied, indent=1, sort_keys=True) + "\n")
    for name in ("pack", *contrib_merge.PACK_FILES):
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
