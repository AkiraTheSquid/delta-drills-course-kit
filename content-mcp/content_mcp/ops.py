"""The tool registry — ONE definition, two front ends.

`server.py` (MCP over stdio) and `cli.py` (the terminal command) both build
themselves from this table. A tool added here appears in both without further
work, and neither front end can drift from the other, because neither one
knows anything the table does not say.

Every op declares:
  writes — True if it changes a file. Write ops take a daily snapshot first.
           No password: anyone with this checkout can edit it (removed 2026-10-26).
           Going LIVE is gated server-side by the publish key (study_publish).
  params — JSON Schema properties; the CLI derives its flags from the same dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from . import problems as mc
from . import atoms as atoms_mod
from . import backup, courses, drills, graph, images, lessons, paths, pipeline


@dataclass
class Op:
    name: str
    summary: str
    handler: Callable[..., Any]
    params: dict = field(default_factory=dict)
    required: list = field(default_factory=list)
    writes: bool = False
    # Whether `ops.call` takes the daily snapshot before running this op.
    # 🔴 False for the backup ops themselves. `backup_restore` must NEVER
    # snapshot first: when the existing snapshot is stale, rotating it captures
    # the very breakage the caller is trying to escape and destroys the good
    # copy in the same breath. `backup_now` manages its own rotation.
    snapshot: bool = True

    def schema(self) -> dict:
        return {"type": "object", "properties": dict(self.params), "required": list(self.required)}


REGISTRY: dict[str, Op] = {}


def op(name, summary, params=None, required=None, writes=False, snapshot=True):
    def wrap(fn):
        REGISTRY[name] = Op(name, summary, fn, params or {}, required or [],
                            writes, snapshot)
        return fn
    return wrap


def call(name: str, arguments: dict | None = None) -> Any:
    """Invoke an op by name. The single entry point both front ends use."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown tool '{name}'. Known: {', '.join(sorted(REGISTRY))}")
    spec = REGISTRY[name]
    args = dict(arguments or {})
    args.pop("token", None)  # callers from the password era still send one
    result: dict[str, Any] = {}
    if spec.writes and spec.snapshot:
        # Fails closed: if the safety copy cannot be written, the write does not
        # happen. A mutation with no snapshot behind it is the one state this
        # whole design exists to prevent.
        snap = backup.ensure()
        result["_backup"] = {
            "rotated": snap.get("rotated"),
            "snapshot_age_hours": snap.get("age_hours"),
            "path": snap.get("path"),
        }
    missing = [key for key in spec.required if key not in args]
    if missing:
        raise ValueError(f"{name}: missing required argument(s): {', '.join(missing)}")
    payload = spec.handler(**args)
    if isinstance(payload, dict):
        result.update(payload)
        return result
    return payload if not result else {"_backup": result["_backup"], "result": payload}


STR = {"type": "string"}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}
STRS = {"type": "array", "items": {"type": "string"}}
INTS = {"type": "array", "items": {"type": "integer"}}


# ── status ─────────────────────────────────────────────────────────────────
@op("content_status", "Where everything stands: backup age, content counts, pipeline interpreter.")
def _status():
    registry = graph.load()
    return {
        "repo": str(paths.REPO),
        "backup": backup.status(),
        "python": pipeline.python_in_use(),
        "content": {
            "courses": paths.courses(),
            "kp_pages": len(paths.kp_files()),
            "kcs": len(registry["kcs"]),
            "registry_problems": graph.check(registry),
            "drills_in_bank": len(drills.bank()),
            "next_drill_id": drills.next_id(),
        },
    }


# ── courses ────────────────────────────────────────────────────────────────
@op("course_list", "Every course folder with its units, page count and image count, plus every unit in the registry.")
def _course_list():
    return courses.list_courses()


@op("course_create",
    "Start a NEW course: its folder under lessons/ and its first unit (the `lesson` id concepts name). "
    "kind 'code' = runnable Python drills; kind 'math' = prose + multiple-choice pages (topic must be Mathematics). "
    "For free-response or image-occlusion questions use the study_* tools instead.",
    {"course": {**STR, "description": "Folder name, lowercase-dashes, e.g. 'probability'."},
     "unit": {**STR, "description": "New unit id like 'pb-1' (short prefix, dash, number)."},
     "title": {**STR, "description": "The course's title."},
     "topic": {**STR, "description": "Topic the unit's drills are filed under, e.g. 'Probability'."},
     "unit_title": {**STR, "description": "The first unit's title (defaults to the course title)."},
     "description": STR,
     "kind": {**STR, "description": "'code' (default) or 'math'."},
     "subtopic_key": STR},
    ["course", "unit", "title", "topic"], writes=True)
def _course_create(course, unit, title, topic, unit_title=None, description="", kind="code",
                   subtopic_key=None):
    return courses.create_course(course, unit, title, topic, unit_title, description, kind, subtopic_key)


@op("course_add_unit", "Add another unit (registry lesson) to an existing course folder.",
    {"course": STR, "unit": STR, "title": STR, "topic": STR, "subtopic_key": STR},
    ["course", "unit", "title", "topic"], writes=True)
def _course_add_unit(course, unit, title, topic, subtopic_key=None):
    return courses.add_unit(course, unit, title, topic, subtopic_key)


# ── images ─────────────────────────────────────────────────────────────────
@op("image_add",
    "Put a picture (PNG/JPEG/GIF/WebP, <=2 MB; no SVG) into a course and get back the markdown that shows it. "
    "Works in lesson pages, drill prompts and multiple-choice prompts/solutions.",
    {"course": STR,
     "alt": {**STR, "description": "What the picture shows (required; read aloud by screen readers)."},
     "source_path": {**STR, "description": "A local image file to copy in."},
     "data_base64": {**STR, "description": "The image bytes, base64, instead of source_path."},
     "name": {**STR, "description": "File name to store it under (default: from the source)."},
     "replace": {**BOOL, "description": "Overwrite an existing file of that name."}},
    ["course", "alt"], writes=True)
def _image_add(course, alt, source_path=None, data_base64=None, name=None, replace=False):
    return images.add(course, alt, source_path, data_base64, name, replace)


@op("image_list", "Every picture in a course (or all courses) and which pages/drills show it.",
    {"course": STR})
def _image_list(course=None):
    return images.list_images(course)


@op("image_remove", "Delete a picture nothing shows any more.",
    {"course": STR, "name": STR}, ["course", "name"], writes=True)
def _image_remove(course, name):
    return images.remove(course, name)


# ── lessons ────────────────────────────────────────────────────────────────
@op("lesson_list", "List every KP page with its title, rung drill ids and drill count.",
    {"course": {**STR, "description": "Restrict to one course folder (python, numpy, einops)."}})
def _lesson_list(course: str | None = None):
    return {"pages": lessons.list_kps(course)}


@op("lesson_read", "Read one KP page: parsed metadata plus the full markdown.",
    {"kc": {**STR, "description": "KC id, e.g. torch.random-seeding."},
     "include_body": {**BOOL, "description": "Set false for metadata only."}},
    ["kc"])
def _lesson_read(kc: str, include_body: bool = True):
    return lessons.read_kp(kc, include_body)


@op("lesson_outline", "Section headings and segment structure of one page.",
    {"kc": STR}, ["kc"])
def _lesson_outline(kc: str):
    return lessons.outline(kc)


@op("lesson_authoring_guide", "The authoring contract every page must satisfy — read before writing one.")
def _guide():
    return {"markdown": lessons.authoring_guide()}


@op("lesson_edit", "Replace an exact string inside a KP page (the safe, surgical edit).",
    {"kc": STR, "old": {**STR, "description": "Exact text to replace."},
     "new": STR,
     "expect_count": {**INT, "description": "Refuse unless `old` appears exactly this many times (default 1)."}},
    ["kc", "old", "new"], writes=True)
def _lesson_edit(kc: str, old: str, new: str, expect_count: int = 1):
    return lessons.edit_kp(kc, old, new, expect_count)


@op("lesson_write", "Replace a whole KP page. Frontmatter must still name the same KC.",
    {"kc": STR, "markdown": STR}, ["kc", "markdown"], writes=True)
def _lesson_write(kc: str, markdown: str):
    return lessons.write_kp(kc, markdown)


@op("lesson_create", "Create a new KP page for a KC that already exists in the registry.",
    {"course": STR, "kc": STR, "markdown": STR}, ["course", "kc", "markdown"], writes=True)
def _lesson_create(course: str, kc: str, markdown: str):
    return lessons.create_kp(course, kc, markdown)


# ── multiple-choice problems (math pages, or MC items on a code page) ────────
@op("problems_read", "The multiple-choice problems file beside a KC's page.",
    {"kc": STR}, ["kc"])
def _problems_read(kc):
    return mc.read(kc)


@op("problems_add",
    "Add multiple-choice problems to a KC (kp-<slug>.problems.json). Ids are minted when omitted. "
    "Schema: {kind: compute|derivation-step|statement, prompt, choices:[{key:A,text,value?}], answer, solution, "
    "difficulty 10-100, verify:{sympy:{truth}} for compute/derivation-step}.",
    {"kc": STR, "problems": {"type": "array", "items": {"type": "object"}}},
    ["kc", "problems"], writes=True)
def _problems_add(kc, problems: list):
    return mc.add(kc, problems)


@op("problems_update", "Change fields of one multiple-choice problem in place (its id stays).",
    {"kc": STR, "id": INT, "fields": {"type": "object"}}, ["kc", "id", "fields"], writes=True)
def _problems_update(kc, id: int, fields: dict):
    return mc.update(kc, id, fields)


# ── concept graph ──────────────────────────────────────────────────────────
@op("graph_list", "The concept graph: every KC with its prereqs, teaching position and drill count.",
    {"lesson": {**STR, "description": "Restrict to one lesson id, e.g. np-1."}})
def _graph_list(lesson: str | None = None):
    return graph.list_graph(lesson)


@op("graph_read", "One concept: prereqs, dependents, position, whether it has a page and drills.",
    {"kc": STR}, ["kc"])
def _graph_read(kc: str):
    return graph.read_node(kc)


@op("graph_add_kc",
    "Add a concept node. Inserted after its last prereq so teaching order stays a linear extension.",
    {"kc": STR, "lesson": STR, "title": STR, "prereqs": STRS, "syntax": BOOL,
     "after": {**STR, "description": "Insert directly after this KC instead of after the last prereq."},
     "answer_secs": {**INT, "description": "Answer clock per drill, 1-1200 s (default 600). Short "
                     "one-call answers 180-300; ARENA-sized exercises up to 1200."}},
    ["kc", "lesson", "title"], writes=True)
def _graph_add(kc: str, lesson: str, title: str, prereqs=None, syntax: bool = True, after=None,
               answer_secs=None):
    return graph.add_kc(kc, lesson, title, prereqs, syntax, after, answer_secs)


@op("graph_update_kc", "Change a concept's title, prereqs, lesson, or its position in the sequence.",
    {"kc": STR, "title": STR, "prereqs": STRS, "lesson": STR, "after": STR,
     "answer_secs": {**INT, "description": "Answer clock per drill, 1-1200 s."}},
    ["kc"], writes=True)
def _graph_update(kc: str, title=None, prereqs=None, lesson=None, after=None, answer_secs=None):
    return graph.update_kc(kc, title, prereqs, lesson, after, answer_secs)


# ── drills ─────────────────────────────────────────────────────────────────
@op("drill_search", "Search the drill bank by free text, by KC, or by id.",
    {"query": STR, "kc": STR, "ids": INTS, "limit": INT})
def _drill_search(query=None, kc=None, ids=None, limit: int = 25):
    return drills.search(query, kc, ids, limit)


@op("drill_read", "One drill in full: prompt, starter, answer, test cases, tags, override layers.",
    {"id": INT}, ["id"])
def _drill_read(id: int):
    return drills.read(id)


@op("drill_next_id", "The id the next authored drill will be served under. Ids are positional.")
def _drill_next_id():
    return {"next_id": drills.next_id()}


@op("drill_add",
    "Author a new drill: appends a row to curated_additions.csv (ids mint above every existing one) "
    "plus an optional function-mode override payload.",
    {"topic": STR, "subtopic": STR, "question_text": STR, "answer_code": STR,
     "difficulty": {**INT, "description": "0-100; <=35 easy, <=65 medium, else hard."},
     "expected_output": STR,
     "override": {"type": "object",
                  "description": "function_name, starter_code, test_cases, wrong_examples, ..."},
     "atoms": {"type": "array", "items": {}, "description":
               "Mastery atoms this drill trains (ids from atom_list / atom_add, or "
               "{atom_id, confidence}). Without them a correct answer moves no mastery "
               "and the bank gate fails the drill (atom_tag_missing)."}},
    ["topic", "subtopic", "question_text", "answer_code"], writes=True)
def _drill_add(topic, subtopic, question_text, answer_code, difficulty: int = 50,
               expected_output: str = "", override=None, atoms=None):
    if atoms:
        tags_ok = atoms_mod.resolve(atoms)  # refuse a bad atom before the CSV row exists
    result = drills.add(topic, subtopic, question_text, answer_code, difficulty,
                        expected_output, override)
    if atoms:
        result["atoms"] = atoms_mod.tag(result["id"], tags_ok)["atoms"]
    else:
        result["atom_warning"] = "No atoms given — drill_tag it before pipeline_check."
    return result


@op("drill_update", "Change an existing drill by appending an override record (last layer wins).",
    {"id": INT, "fields": {"type": "object", "description": "Whitelisted override fields only."}},
    ["id", "fields"], writes=True)
def _drill_update(id: int, fields: dict):
    return drills.update(id, fields)


@op("drill_tag",
    "Set the mastery atoms a drill trains (appends its tag row; the last row per drill wins).",
    {"id": INT, "atoms": {"type": "array", "items": {},
                          "description": "Atom ids, or {atom_id, confidence} objects."}},
    ["id", "atoms"], writes=True)
def _drill_tag(id: int, atoms: list):
    return atoms_mod.tag(id, atoms)


@op("atom_list", "Search the mastery-atom graph (id, title, topic, how many drills train each).",
    {"query": STR, "limit": INT})
def _atom_list(query: str = "", limit: int = 40):
    return atoms_mod.list_atoms(query, limit)


@op("atom_add",
    "Add a mastery atom to the atom graph: prereqs (existing atoms) OR root=true. A new "
    "course's drills need an atom of their own to tag.",
    {"atom": STR, "title": STR, "description": STR, "topic": STR,
     "subtopic": {**STR, "description": "Usually the unit's subtopic_key, e.g. 'Probability: Probability basics'."},
     "prereqs": {"type": "array", "items": STR}, "root": {"type": "boolean"}},
    ["atom", "title", "description", "topic", "subtopic"], writes=True)
def _atom_add(atom, title, description, topic, subtopic, prereqs=None, root: bool = False):
    return atoms_mod.add(atom, title, description, topic, subtopic, prereqs, root)


@op("drill_retire", "Stop serving a drill without renumbering anything behind it.",
    {"id": INT, "reason": STR}, ["id"], writes=True)
def _drill_retire(id: int, reason: str = ""):
    return drills.retire(id, reason)


# ── pipeline ───────────────────────────────────────────────────────────────
@op("pipeline_check", "Run the whole content gate in order, stopping at the first failure.",
    {"fast": {**BOOL, "description": "Skip fence execution and the bank export. Never enough to ship."}},
    writes=True)
def _check(fast: bool = False):
    return pipeline.check_all(fast)


@op("pipeline_step", "Run one pipeline step: validate, compile, qmatrix, export, audit_bank, notebooks.",
    {"step": STR}, ["step"], writes=True)
def _step(step: str):
    return pipeline.step(step)


@op("pipeline_audit", "Run one standing audit and return its report.",
    {"audit": {**STR, "description": "One of: " + ", ".join(pipeline.AUDITS)},
     "args": STRS},
    ["audit"])
def _audit(audit: str, args=None):
    return pipeline.audit(audit, args)


@op("pipeline_watchers", "Run the folder health watchers that carry the standing content guards.",
    {"folders": STRS})
def _watchers(folders=None):
    return pipeline.folder_watchers(folders)


# ── backup ─────────────────────────────────────────────────────────────────
@op("backup_status", "Age, size and file count of the single content snapshot.")
def _backup_status():
    return backup.status()


@op("backup_now", "Refresh the snapshot. Without force, a snapshot younger than 24h is kept.",
    {"force": BOOL}, writes=True, snapshot=False)
def _backup_now(force: bool = False):
    return backup.snapshot(force)


@op("backup_list", "Every file the current snapshot carries.")
def _backup_list():
    files = backup.contents()
    return {"file_count": len(files), "files": files[:200], "truncated": len(files) > 200}


@op("backup_restore",
    "Put the snapshot back. The current tree is parked beside it first, so a restore is undoable.",
    {"confirm": {**BOOL, "description": "Must be true. Restoring overwrites current content."},
     "paths": {**STRS, "description": "Repo-relative paths to restore. Omit for everything."}},
    ["confirm"], writes=True, snapshot=False)
def _backup_restore(confirm: bool, paths: list | None = None):
    if not confirm:
        raise ValueError("Refusing to restore without confirm=true.")
    return backup.restore(paths, keep_current=True)


def catalogue() -> list[dict]:
    return [
        {"name": spec.name, "summary": spec.summary, "writes": spec.writes,
         "params": sorted(spec.params), "required": spec.required}
        for spec in sorted(REGISTRY.values(), key=lambda s: s.name)
    ]


def as_json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


# The Study-course tools register themselves into REGISTRY on import.
from . import ops_study  # noqa: E402,F401
