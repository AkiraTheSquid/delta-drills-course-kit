"""The Study-course tools (content_mcp/study.py, study_images.py), registered
into ops.REGISTRY. A module of their own so ops.py stays one screen per family."""

from __future__ import annotations

from . import study, study_images
from .ops import INT, STR, op

STRS = {"type": "array", "items": STR}
OBJ = {"type": "object"}
ARR = {"type": "array", "items": {}}
BOOL = {"type": "boolean"}


@op("study_list", "List the Study courses written through the MCP (concepts, questions by format, diagrams).")
def _study_list():
    return study.list_courses()


@op("study_read", "Read a Study course's whole spec (course.json).", {"course": STR}, ["course"])
def _study_read(course: str):
    return study.load(course)


@op("study_course_create",
    "Start a Study course: free-answer (Jev-graded), multiple choice, typed cards and image "
    "occlusion. sections = [{id, title}] or titles; ids are namespaced by the course.",
    {"course": {**STR, "description": "lowercase-dashed id, e.g. 'ap-bio-genetics'"}, "title": STR,
     "sections": ARR, "eyebrow": STR, "public": {**BOOL, "description": "true = every signed-in learner sees it"}},
    ["course", "title", "sections"], needs_auth=True)
def _study_course_create(course, title, sections, eyebrow: str = "", public: bool = False):
    return study.create(course, title, sections, eyebrow, public)


@op("study_source_add",
    "Add the text passages are quoted from (verbatim). text= the words, or source_path= a .md/.txt/.html "
    "file. license is required — the course kit is public.",
    {"course": STR, "source": STR, "title": STR, "license": STR, "text": STR, "source_path": STR,
     "authors": STR, "year": INT, "url": STR, "replace": BOOL},
    ["course", "source", "title", "license"], needs_auth=True)
def _study_source_add(course, source, title, license, text=None, source_path=None, authors="", year=0,
                      url=None, replace=False):
    return study.source_add(course, source, title, license, text, source_path, authors, year, url,
                            replace=replace)


@op("study_concept_add",
    "Add one concept (one idea). concept = {id, title, section, definition, intro?, aliases?, "
    "kind: conceptual|math, prereqs: [ids], constraints: [2+ things a right answer must say], "
    "misconceptions: [{label, description}], passages: [{source, text (EXACT copy), locator?}], "
    "diagrams?: [ids]}.",
    {"course": STR, "concept": OBJ, "replace": BOOL}, ["course", "concept"], needs_auth=True)
def _study_concept_add(course, concept, replace: bool = False):
    return study.concept_add(course, concept, replace)


@op("study_question_add",
    "Add questions. Each = {format: free|mc|card, kind: content|evaluative|relation|check, targets: "
    "[concept ids], prompt, reference (the model answer), options (mc: [{text, correct, why}]), "
    "answers (card: other accepted spellings), demand?, id?}. Image-occlusion cards come from "
    "study_diagram_add.",
    {"course": STR, "questions": ARR, "replace": BOOL}, ["course", "questions"], needs_auth=True)
def _study_question_add(course, questions, replace: bool = False):
    return study.question_add(course, questions, replace)


@op("study_diagram_add",
    "Image occlusion: copy a picture in and hide its labels. labels[i] is what box i hides; boxes = "
    "[[x0, y0, x1, y1], ...] in pixels. One label card per box, for `concept` (or box_concepts "
    "{box: concept}). Then study_diagram_preview to check the boxes by eye.",
    {"course": STR, "diagram": STR, "title": STR, "source_path": STR, "labels": STRS, "boxes": ARR,
     "concept": STR, "box_concepts": OBJ,
     "answers": {**OBJ, "description": "{box: [other accepted spellings]}"},
     "boxes_on": {**STR, "enum": ["prepared", "original"], "description": "'prepared' (default: pixels of the stored <=1600 px picture) "
                  "or 'original' (pixels of the file passed)"},
     "replace": BOOL},
    ["course", "diagram", "title", "source_path", "labels", "boxes", "concept"], needs_auth=True)
def _study_diagram_add(course, diagram, title, source_path, labels, boxes, concept, box_concepts=None,
                       answers=None, boxes_on: str = "prepared", replace: bool = False):
    return study_images.diagram_add(course, diagram, title, source_path, labels, boxes, concept,
                                    box_concepts, answers, boxes_on, replace)


@op("study_diagram_update", "Move a diagram's boxes or rename its labels; its cards are rewritten.",
    {"course": STR, "diagram": STR, "boxes": ARR, "labels": STRS, "title": STR},
    ["course", "diagram"], needs_auth=True)
def _study_diagram_update(course, diagram, boxes=None, labels=None, title=None):
    return study_images.diagram_update(course, diagram, boxes, labels, title)


@op("study_diagram_preview",
    "Draw a diagram with a coordinate grid and its numbered boxes; returns a PNG path to look at.",
    {"course": STR, "diagram": STR}, ["course", "diagram"])
def _study_diagram_preview(course, diagram):
    return study_images.diagram_preview(course, diagram)


@op("study_remove", "Remove a concept, question, source, diagram (and its cards) or section.",
    {"course": STR, "kind": STR, "id": STR}, ["course", "kind", "id"], needs_auth=True)
def _study_remove(course, kind, id):
    return study.remove(course, kind, id)


@op("study_check",
    "Check a Study course against the real pack rules (merged into a copy of the pack): structure, "
    "verbatim passages, pictures. Writes nothing.",
    {"course": STR}, ["course"])
def _study_check(course):
    return study_images.check(course)


@op("study_apply",
    "Private repo only: merge a checked Study course into the live pack the app serves (re-runnable). "
    "In the course kit, send study-courses/<course>/ as a pull request instead.",
    {"course": STR}, ["course"], needs_auth=True)
def _study_apply(course):
    return study_images.apply(course)
