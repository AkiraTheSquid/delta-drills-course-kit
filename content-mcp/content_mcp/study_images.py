"""Image occlusion for a Study course, and the runs that need pydantic / Pillow.

An image-occlusion diagram is a picture plus one pixel box [x0, y0, x1, y1]
per label. The app shows it with EVERY box painted over and one highlighted,
and the learner types what that box hides (Anki's "hide all, guess one";
graded without a model by backend grade/label.py). `study_diagram_add` copies
the picture in (longest side <= 1600 px, so the boxes are drawn on the size
that ships), stores the boxes, and writes one `label` card per box.

Boxes are where an author goes wrong, so `study_diagram_preview` draws the
picture with a coordinate grid and every box numbered: look at it, fix the
numbers with `study_diagram_update`, look again.

Everything here that touches pixels or the pack model runs
scripts/study_course.py under the content python and reads one JSON line back.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from . import paths, study


def _helper(*args: str, timeout: int = 300) -> dict:
    proc = subprocess.run([paths.python_for_content(), str(paths.STUDY_HELPER), *args],
                          capture_output=True, text=True, timeout=timeout, cwd=str(paths.REPO))
    lines = [x for x in proc.stdout.splitlines() if x.strip().startswith("{")]
    if not lines:
        raise RuntimeError(
            f"study helper failed (exit {proc.returncode}): {proc.stderr.strip()[-1500:]}\n"
            "The content python needs pydantic and Pillow: .venv/bin/pip install -r requirements.txt")
    return json.loads(lines[-1])


def _cards(course: str, d: dict, concept: str, answers: dict) -> list[dict]:
    """One label card per box. A box's card counts for the concept named in
    `concepts[box]` when given, else for the diagram's concept."""
    out = []
    for i, label in enumerate(d["labels"]):
        out.append({"id": f"{course}.io-{d['id'][len(course) + 1:]}-{i}", "kind": "content", "format": "label",
                    "demand": "recall", "targets": [d.get("_targets", {}).get(i, concept)],
                    "prompt": f"{d['title']}: what does the highlighted box hide?",
                    "reference": label, "diagram": d["id"], "box": i,
                    **({"answers": answers[str(i)]} if answers.get(str(i)) else {})})
    return out


def _check_boxes(boxes, labels, w, h) -> list:
    if len(boxes) != len(labels) or not boxes:
        raise ValueError(f"{len(labels)} labels for {len(boxes)} boxes — one label per box.")
    clean = []
    for i, b in enumerate(boxes):
        if len(b) != 4 or not all(isinstance(v, (int, float)) for v in b):
            raise ValueError(f"box {i} must be [x0, y0, x1, y1] in pixels.")
        x0, y0, x1, y1 = (round(v) for v in b)
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"box {i} {b} is outside the {w}x{h} picture (or has no area).")
        clean.append([x0, y0, x1, y1])
    return clean


def diagram_add(course: str, diagram: str, title: str, source_path: str, labels: list, boxes: list,
                concept: str, box_concepts: dict | None = None, answers: dict | None = None,
                boxes_on: str = "prepared", replace: bool = False) -> dict:
    """`boxes_on`: 'prepared' = pixels of the picture as stored (<= 1600 px; see
    the returned w/h — a picture already that small is unchanged), 'original' =
    pixels of the file you passed, scaled here."""
    spec = study.load(course)
    did = study._diagram_id(course, diagram)
    cid = study._ref(course, concept)
    if any(d["id"] == did for d in spec["diagrams"]) and not replace:
        raise ValueError(f"diagram '{did}' already exists — pass replace=true, or study_diagram_update.")
    src = Path(source_path).expanduser()
    if not src.is_file():
        raise ValueError(f"No file at {source_path}.")
    if src.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        raise ValueError("A diagram is a PNG, JPEG, WebP or GIF (no SVG).")
    if boxes_on not in ("prepared", "original"):
        raise ValueError("boxes_on: 'prepared' or 'original'.")
    rel = f"images/{did[len(course) + 1:]}.png"
    final = study.folder(course) / rel
    staged = final.with_name(final.stem + ".incoming.png")
    # Prepared beside the old picture and moved in only once everything checks,
    # so a failed replace leaves the previous diagram intact.
    size = _helper("prepare-image", str(src), str(staged))
    try:
        if "w" not in size:
            raise ValueError("; ".join(size.get("errors", ["could not read the picture"])))
        scale = size["scaled_by"] if boxes_on == "original" else 1.0
        clean = _check_boxes([[v * scale for v in b] for b in boxes], labels, size["w"], size["h"])
        targets = {int(k): study._ref(course, v) for k, v in (box_concepts or {}).items()}
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    staged.replace(final)
    d = {"id": did, "title": title.strip(), "w": size["w"], "h": size["h"], "boxes": clean,
         "labels": [str(x).strip() for x in labels], "image": rel}
    spec["diagrams"] = [x for x in spec["diagrams"] if x["id"] != did] + [d]
    cards = _cards(course, {**d, "_targets": targets}, cid, {str(k): v for k, v in (answers or {}).items()})
    spec["questions"] = [q for q in spec["questions"] if not (q["format"] == "label" and q.get("diagram") == did)] + cards
    for c in spec["concepts"]:
        if c["id"] == cid and did not in c.get("diagrams", []):
            c.setdefault("diagrams", []).append(did)
    study.save(course, spec)
    return {"diagram": did, "w": size["w"], "h": size["h"], "scaled_by": size["scaled_by"],
            "cards": [q["id"] for q in cards],
            "next_step": "study_diagram_preview, LOOK at the picture, fix boxes with study_diagram_update."}


def diagram_update(course: str, diagram: str, boxes: list | None = None, labels: list | None = None,
                   title: str | None = None) -> dict:
    spec = study.load(course)
    did = study._diagram_id(course, diagram)
    d = next((x for x in spec["diagrams"] if x["id"] == did), None)
    if d is None:
        raise ValueError(f"No diagram '{did}'.")
    old_cards = {q["box"]: q for q in spec["questions"] if q["format"] == "label" and q.get("diagram") == did}
    labels = labels if labels is not None else d["labels"]
    d["boxes"] = _check_boxes(boxes if boxes is not None else d["boxes"], labels, d["w"], d["h"])
    d["labels"] = [str(x).strip() for x in labels]
    if title:
        d["title"] = title.strip()
    concept = next((c["id"] for c in spec["concepts"] if did in c.get("diagrams", [])), None)
    # Keyed by LABEL, not box index: inserting or deleting a box must not hand
    # one label's concept or accepted answers to its neighbour.
    by_label = {q["reference"]: q for q in old_cards.values()}
    targets = {i: by_label[lab]["targets"][0] for i, lab in enumerate(d["labels"]) if lab in by_label}
    answers = {str(i): by_label[lab]["answers"] for i, lab in enumerate(d["labels"])
               if lab in by_label and by_label[lab].get("answers")}
    cards = _cards(course, {**d, "_targets": targets}, concept or next(iter(targets.values()), ""), answers)
    spec["questions"] = [q for q in spec["questions"] if not (q["format"] == "label" and q.get("diagram") == did)] + cards
    study.save(course, spec)
    return {"diagram": did, "boxes": len(d["boxes"]), "cards": [q["id"] for q in cards]}


def diagram_preview(course: str, diagram: str) -> dict:
    study.load(course)
    did = study._diagram_id(course, diagram)
    out = Path(tempfile.gettempdir()) / f"study-preview-{did}.png"
    result = _helper("preview", str(study.folder(course)), did, str(out))
    result["next_step"] = ("Open `preview` (Read the PNG): blue grid lines every 50/100 px with their "
                           "coordinates, red numbered boxes. Each box must cover its label's TEXT fully.")
    return result


def check(course: str) -> dict:
    study.load(course)
    base = paths.study_pack()
    if base is None:
        raise RuntimeError("No Study pack to check against (study-packs/ in the kit, the live pack here).")
    return _helper("check", str(study.folder(course)), str(base))


def apply(course: str) -> dict:
    study.load(course)
    live = paths.LIVE_STUDY_PACK
    if not (live / "pack.json").exists():
        raise RuntimeError(
            "This checkout has no live Study pack (the course kit never does). study_check is the "
            f"gate here: when it is ok, send study-courses/{course}/ as a pull request — it is "
            "applied and deployed from the private repo.")
    return _helper("apply", str(study.folder(course)), str(live), timeout=900)
