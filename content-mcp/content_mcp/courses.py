"""Courses and their units: the two things a new course needs before any page.

A COURSE is a folder under `lessons/` that holds its KP pages (and its
`images/`). A UNIT is a row in `kc_registry.json`'s `lessons` list — the
`lesson` every concept names (`py-0`, `tr-1`) with the topic its drills are
filed under. A course has one unit or several (`pytorch/` holds `tr-1` and the
`ar-*` units), and the two never had a tool of their own: `graph_add_kc`
refused any unit the registry did not already list, and `lesson_create`
refused any folder that did not already hold a page. So "build me a course"
could not get past its first call.

`course_create` makes the folder (with its `course.json` marker — git does not
keep an empty directory) and the first unit; `course_add_unit` adds another.
"""

from __future__ import annotations

import json
import re

from . import paths

COURSE_RE = re.compile(r"^[a-z][a-z0-9-]{1,39}$")
UNIT_RE = re.compile(r"^[a-z][a-z0-9]{0,5}-\d{1,3}$")
# The math lane (scripts/validate_math.py MATH_TOPICS): a `kind: math` page —
# prose + multiple choice, no code — must sit in a unit with this topic.
MATH_TOPIC = "Mathematics"
# Folders under lessons/ that are not courses, so a course may not take their name.
RESERVED = {"notes", "notebooks", "images", "leetcode", "__pycache__"}


def _registry() -> dict:
    return json.loads(paths.KC_REGISTRY.read_text())


def _save_registry(registry: dict) -> None:
    indent = paths.json_indent_of(paths.KC_REGISTRY)
    paths.KC_REGISTRY.write_text(json.dumps(registry, indent=indent) + "\n")


def _marker(course: str) -> dict:
    path = paths.LESSONS / course / paths.COURSE_MARKER
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _course_of_kc() -> dict[str, str]:
    """kc id -> course folder, read off where each KC's page lives."""
    from . import lessons
    return {kc: path.parent.name for kc, path in lessons.index().items()}


def list_courses() -> dict:
    registry = _registry()
    where = _course_of_kc()
    units_by_course: dict[str, set[str]] = {}
    for kc in registry["kcs"]:
        if kc["id"] in where:
            units_by_course.setdefault(where[kc["id"]], set()).add(kc.get("lesson"))
    out = []
    for course in paths.courses():
        marker = _marker(course)
        units = set(units_by_course.get(course, set())) | set(marker.get("units", []))
        folder = paths.LESSONS / course
        out.append({
            "course": course,
            "title": marker.get("title"),
            "description": marker.get("description"),
            "units": [u for u in (les["id"] for les in registry["lessons"]) if u in units],
            "pages": len(list(folder.glob("kp-*.md"))),
            "images": len(list((folder / "images").glob("*"))) if (folder / "images").is_dir() else 0,
        })
    return {"courses": out, "units": registry["lessons"]}


def _unit_row(unit: str, topic: str, title: str, subtopic_key: str | None) -> dict:
    if not UNIT_RE.fullmatch(unit):
        raise ValueError(
            f"Unit id '{unit}' must look like the existing ones: a short lowercase "
            "prefix, a dash, a number (py-0, tr-1, pb-1)."
        )
    if not str(topic).strip() or not str(title).strip():
        raise ValueError("A unit needs a topic and a title.")
    return {
        "id": unit,
        "topic": topic.strip(),
        "title": title.strip(),
        "subtopic_key": (subtopic_key or f"{topic.strip()}: {title.strip()}").strip(),
    }


def _add_unit_row(registry: dict, row: dict) -> None:
    if row["id"] in {les["id"] for les in registry["lessons"]}:
        raise ValueError(f"Unit '{row['id']}' already exists — pick another id.")
    titles = {les["title"] for les in registry["lessons"]}
    if row["title"] in titles:
        # glossary.js and the KG panel name a unit by its title; two units with
        # one title would read as one place to a learner.
        raise ValueError(f"A unit is already titled '{row['title']}'.")
    registry["lessons"].append(row)


def create_course(course: str, unit: str, title: str, topic: str,
                  unit_title: str | None = None, description: str = "",
                  kind: str = "code", subtopic_key: str | None = None) -> dict:
    """A new course folder plus its first unit."""
    if not COURSE_RE.fullmatch(course or ""):
        raise ValueError(
            f"Course folder '{course}' must be lowercase letters, digits and dashes "
            "(2-40 chars, starting with a letter), e.g. 'probability'."
        )
    if course in RESERVED:
        raise ValueError(f"'{course}' is a reserved folder name under lessons/.")
    if course in paths.courses() or (paths.LESSONS / course).exists():
        raise ValueError(f"Course '{course}' already exists — add a unit with course_add_unit.")
    if kind not in ("code", "math"):
        raise ValueError("kind must be 'code' (runnable Python drills) or 'math' (prose + multiple choice).")
    if not str(title or "").strip():
        raise ValueError("A course needs a title.")
    if kind == "math" and topic != MATH_TOPIC:
        raise ValueError(
            f"A math course's pages are `kind: math`, and the validator holds those to the "
            f"'{MATH_TOPIC}' topic (scripts/validate_math.py). Pass topic='{MATH_TOPIC}'."
        )
    registry = _registry()
    row = _unit_row(unit, topic, unit_title or title, subtopic_key)
    _add_unit_row(registry, row)

    folder = paths.LESSONS / course
    folder.mkdir(parents=True)
    marker = {"title": title.strip(), "description": description.strip(), "kind": kind,
              "units": [unit]}
    (folder / paths.COURSE_MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    _save_registry(registry)
    return {
        "course": course,
        "folder": str(folder.relative_to(paths.REPO)),
        "unit": row,
        "next_steps": [
            f"graph_add_kc with lesson='{unit}' for each concept (prerequisites first).",
            f"lesson_create with course='{course}' for each concept's page"
            + (" (frontmatter `kind: math`), then problems_add for its multiple-choice problems." if kind == "math" else "."),
            "image_add to put a picture in the course; it returns the markdown to paste into a page.",
            "pipeline_check, then pipeline_watchers.",
        ],
    }


def add_unit(course: str, unit: str, title: str, topic: str,
             subtopic_key: str | None = None) -> dict:
    """Another unit inside an existing course folder."""
    if course not in paths.courses():
        raise ValueError(f"Unknown course '{course}'. Known: {', '.join(paths.courses())}")
    marker_path = paths.LESSONS / course / paths.COURSE_MARKER
    if marker_path.exists() and _marker(course).get("kind") == "math" and topic != MATH_TOPIC:
        raise ValueError(f"'{course}' is a math course: its units take topic='{MATH_TOPIC}' (scripts/validate_math.py).")
    registry = _registry()
    row = _unit_row(unit, topic, title, subtopic_key)
    _add_unit_row(registry, row)
    _save_registry(registry)
    if marker_path.exists():
        marker = _marker(course)
        marker.setdefault("units", [])
        if unit not in marker["units"]:
            marker["units"].append(unit)
        marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    return {"course": course, "unit": row}
