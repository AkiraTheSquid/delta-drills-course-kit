"""Content packs: the concepts, verbatim passages and banked questions the tutor teaches.

A pack is a folder under `app/data/concept_packs/<pack_id>/` (server-only, because
it holds reference answers):

    pack.json        {id, title, version, vetted, course, sources: [Source], sections: [Section]}
    concepts.json    [Concept]
    edges.json       [Edge]
    passages.json    [Passage]
    questions.json   [Question]

Two kinds of concept node share one graph:
  * `conceptual`: free-text questions, graded by Jev (grade/).
  * `math`: multiple-choice questions, graded deterministically; Jev is never
    called for them.

A `label` question is an image-occlusion card (Anki's "hide all, guess one"):
a pack `Diagram` with every label painted over and one box highlighted; the
learner types what it hides, graded deterministically against `reference` +
`answers` (grade/label.py), never by Jev. The images are server-only files in
`<pack>/diagrams/` (`<id>.webp`, and `<id>.masked.webp` with every box painted).

Concept ids that also exist in `lessons/kc_registry.json` (the Delta Drills
course's `deltadrills.*` chain) are the SAME concept: the pack extends that
graph instead of forking it. The procedural pipeline never reads a pack, so a
pack-only concept cannot leak into the frontier, XP or course mix
(docs/spec-conceptual-tutor.md §5).

Passages are verbatim. The runtime trusts the pack; the offline gate
`scripts/concept_packs/check_pack.py` proves each passage is an exact
substring of its source file before a pack is marked vetted.
"""

from __future__ import annotations

import math
import json
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

PACKS_DIR = Path(__file__).resolve().parent.parent / "data" / "concept_packs"
# Broad diagnostic questions (Seth, 2026-10-06: "a lot of the concepts at once");
# explore.ranked scores every right/wrong outcome, 2^6 = 64 at most.
MAX_COMBINED = 6
# A CASE (a combined question with its own clock, `Question.seconds`): one
# clinical case written up in full, graded on every concept it touches
# (Seth, 2026-10-10: the CCS cases, "one question connects to a crap ton of
# different concepts"). explore.ranked samples its outcomes past MAX_COMBINED.
MAX_CASE = 40

Demand = Literal["recall", "explain", "contrast", "apply", "evaluate", "relate"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(_M):
    id: str
    title: str
    authors: str
    year: int
    venue: str = ""
    url: Optional[str] = None
    # Repo-relative file the passages are checked against (PDF, HTML, .py, .md).
    path: str
    kind: Literal["paper", "post", "app-source", "doc"] = "paper"
    license: str = ""     # reuse terms, shown with the citation (a public course must say)


class Section(_M):
    """A colour-coded part of the course. Concepts name theirs in `section`;
    the Courses tab, the Study cards and the course graph all paint by it."""
    id: str
    title: str
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")


class Unit(_M):
    """A part of a course, switched on or off for practice on its own
    (planner.active_concepts; Seth, 2026-10-07: the biology course's class unit
    and its DAT unit). A set of the course's sections; the graph and what the
    learner may see stay the whole course."""
    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    title: str
    eyebrow: str = ""
    sections: list[str] = Field(min_length=1)


class Course(_M):
    """A course the Courses tab lists: a set of the pack's sections. A pack may
    hold several (Seth, 2026-10-04), each enabled for practice on its own.
    `private_to` non-empty = ONLY those emails may see it, invites or not
    (access.py); empty = the owner and the invite list."""
    id: str
    title: str
    eyebrow: str = ""
    sections: list[str] = Field(min_length=1)
    private_to: list[str] = []
    public: bool = False  # True = every signed-in learner sees it, no invite (access.py)
    # False = off for a learner who never chose their courses: they switch it
    # on, and then it is the only one on (access_router.set_course).
    default_on: bool = True
    # A page of its own: deltadrills.com/<link> opens this course alone
    # (Local_Deployed_Shared/focus-route.js). The page finds its course by this
    # slug through /access, so no static file names a private course.
    link: str = Field(default="", pattern=r"^([a-z0-9]+(-[a-z0-9]+)*)?$", max_length=60)
    # Study's review clock: seconds to read the feedback before the next
    # question, by format ({"mc": 60, "free": 120}); empty = page defaults. In the
    # pack, not the page, so no static file names a private course (study_meter.js).
    review: dict[Literal["mc", "free"], int] = {}
    # Study's ANSWER clock: seconds to answer a question, by format; empty = the
    # page's default (study_meter.js ALLOW, MC 120 s). A case keeps its own (Question.seconds).
    answer: dict[Literal["mc", "free"], int] = {}
    # Parts switched on or off for practice on their own (Unit); [] = one whole.
    units: list[Unit] = []
    # CRAM mode (cram.py, Seth 2026-10-08: an exam tomorrow, two hours tonight):
    # its sections most important first, and the clocks while a learner crams,
    # {"answer": {"mc": s, "free": s}, "review": {...}}. Empty = no cram order.
    cram_order: list[str] = []
    cram_clock: dict[Literal["answer", "review"], dict[Literal["mc", "free"], int]] = {}
    # QUESTION BANK mode (qbank.py; Seth 2026-10-10, the Step 3 MVP):
    # each concept is an AREA, the item scheduler picks bank MC questions:
    # area by weight x weakness and the question nearest qbank.TARGET correct.
    # `area_weights` = concept id -> exam relevance (AMBOSS's % per system).
    # Model-selected lesson support is added by bank_support.py (2026-10-12).
    mode: Literal["tutor", "qbank"] = "tutor"
    area_weights: dict[str, float] = {}
    # CASES of another course this one asks too (Seth 2026-10-11: the cardiology
    # CCS cases are part of the Step 3 course, and still the CCS course's on
    # /step3): their targets join course_concepts, so its Study plans them and
    # its graph shows them. Their sections stay the owner's (own_concepts), so
    # a learner who may not see the owner never sees them (seen_concepts).
    cases: list[str] = []
    # False = NO LESSONS (Seth, 2026-10-12, the AISC proposal course: "just the
    # question and free answer"): a new concept opens with its question, and
    # being asked counts as taught, so its FSRS reviews still come (model.build).
    lessons: bool = True
    # True = answers are SPOKEN and marked for delivery (oral.py, Seth
    # 2026-10-21, Paula's QI rotation): a dictation mic in the answer box,
    # a verbatim transcript, amber marks for wordy / tangent / vague / repeat.
    oral: bool = False


class Misconception(_M):
    id: str
    label: str            # short, shown in feedback
    description: str      # the wrong belief, fed to Jev


class Constraint(_M):
    id: str
    text: str             # what a correct account must respect
    demand: Demand = "explain"
    required: bool = False   # True = the core; a violation caps the grade


class LessonPart(_M):
    """A run of lesson text. `quote` = the source's own words, VERBATIM from the
    block's passage (problems() checks it); the rest is written to join them."""
    text: str = Field(min_length=1)
    quote: bool = False


class LessonBlock(_M):
    """One block of an authored lesson (Seth, 2026-10-11): mostly the source's
    words, joined by connecting text so it reads as one lesson, not quotes.
    `passage` = the passage the quoted parts come from; the lesson shows it as
    a source pill that opens the original text."""
    kind: Literal["heading", "para", "list", "image"]
    parts: list[LessonPart] = []          # heading, para, and an image's caption
    items: list[list[LessonPart]] = []    # list: one bullet each
    passage: str = ""
    src: str = ""                         # image: an https URL


class ExampleStep(_M):
    """One move in a worked example and the reason it is the right move."""
    text: str = Field(min_length=1)
    why: str = Field(min_length=1)


class WorkedExample(_M):
    """A worked example after the lesson: a problem, the plan (which idea applies
    and why), each step with its reason, the answer. A bare answer teaches nothing
    (Seth, 2026-10-11), so plan and every why are required."""
    title: str
    problem: str
    plan: str = Field(min_length=1)
    steps: list[ExampleStep] = Field(min_length=1)
    answer: str


class Concept(_M):
    id: str
    title: str
    kind: Literal["conceptual", "math"] = "conceptual"
    aliases: list[str] = Field(min_length=1)
    definition: str
    intro: str            # 1-2 sentences shown above the lesson's passages
    constraints: list[Constraint] = []
    misconceptions: list[Misconception] = []
    passages: list[str] = Field(min_length=1)
    prereqs: dict[str, float] = {}
    order: int
    procedural_kcs: list[str] = []
    section: str = ""     # a Pack.sections id; "" only in a pack with no sections
    figures: list[str] = []   # Pack.figures ids, shown in the lesson above the passages
    diagrams: list[str] = []  # Pack.diagrams ids: the labelled picture, shown in the lesson
    # An authored lesson (LessonBlock): when present it is the lesson, and the
    # passages are its sources (shown from its pills, not as cards).
    lesson: list[LessonBlock] = []
    examples: list[WorkedExample] = []    # shown after the lesson


class Edge(_M):
    a: str
    b: str
    type: Literal["contrasts", "is-a", "causes", "mitigates", "evidence-for",
                  "example-of", "explains", "depends-on"]
    relation_constraints: list[str] = []
    note: str = ""


class Passage(_M):
    id: str
    source: str
    locator: str          # "§General Discussion, ¶1"
    text: str             # VERBATIM, ≤ ~220 words


class Option(_M):
    id: str
    text: str
    correct: bool = False
    why: str              # shown after the answer
    misconception: Optional[str] = None
    images: list[str] = []  # a bank option's explanation pictures (https), shown with `why` after the answer


class CaseRow(_M):
    """One line of a case's answer key as the CCS review shows it (Seth 2026-10-11,
    "that format"): its section, level and kind. DISPLAY ONLY: grade/case.py and
    case_key read `reference`; voice.case_feedback colours these by the grade."""
    section: Literal["diagnostic", "treatment", "preventive"]
    kind: Literal["order", "avoid", "optional"]
    level: Literal["Critical", "High", "Moderate", "Low", "Minor", ""] = ""
    order: str
    reason: str
    target: str = ""    # kind "order": the target it grades


class SettingStep(_M):
    at: str = ""        # the key's simulated time, "0 Days, 06 Hours, 00 Minutes"
    places: str         # "Intensive Care Unit or Inpatient Unit"
    reason: str = ""


class CaseSetting(_M):
    """Where a case's patient belongs: where it opens, the key's correct
    location(s) at each graded time, and its sequencing advice. Display only."""
    start: str = ""
    steps: list[SettingStep] = []
    sequence: list[str] = []


class Question(_M):
    id: str
    kind: Literal["content", "evaluative", "relation", "check", "combined"]  # combined: explore probes only
    format: Literal["free", "mc", "label", "card"] = "free"
    demand: Demand
    targets: list[str] = Field(min_length=1)
    expected: list[str] = []
    required: list[str] = []
    prompt: str
    reference: str
    rephrasings: list[str] = []
    options: list[Option] = []
    # Combined only. listed = separable sub-questions bundled together (a testlet);
    # integrated = a full answer must RELATE the targets (explore.py weights,
    # gates and grades the two differently; scripts/concept_packs/combined_integration.py).
    integration: Optional[Literal["listed", "integrated"]] = None
    # Free only: an author-written question ASKED ONE PART PER TURN (broad_steps.py),
    # each part a single ask, understanding before opinion (Seth, 2026-10-18, the
    # BlueDot governance assignments: avoid "multiple independent questions within
    # a single question"). The prompt lists them after its colon, for the grader.
    parts: list[str] = []
    # Label only: the Pack.diagrams id, which of its boxes is asked, and the other
    # spellings accepted besides `reference` ("Rough ER").
    # Card (a typed flashcard, grade/card.py): `answers` likewise; `diagram`
    # and `box` (optional) = the picture its FEEDBACK shows, never the question;
    # `group` = cards easily mixed up (the four scientists): typing another
    # card's answer of the group is wrong without asking Jev.
    diagram: str = ""
    box: int = -1
    answers: list[str] = []
    group: str = ""
    # > 0 = this many seconds on the answer clock instead of the format's default
    # (study_meter.js reads it off the body). On a combined question it makes it
    # a CASE (is_case, up to MAX_CASE targets): a CCS case's own 10 or 20 minutes.
    # A CCS node's own short questions carry their case's clock too (Seth,
    # 2026-10-12: "10 or 20 depending on whether ccscases.com marked it").
    seconds: int = Field(default=0, ge=0)
    # Case only, display only: the answer key line by line and the setting.
    key_rows: list[CaseRow] = []
    setting: Optional[CaseSetting] = None
    # Case only: the course concepts it applies (a CCS cardiology case -> the
    # Step 3 cardiology concepts); a case that missed orders teaches their
    # lessons first (planner._case_lessons).
    links: list[str] = []
    # A question-bank item (Course.mode "qbank"): where it comes from, the share
    # of that bank's users who answered it right (its difficulty for qbank.py;
    # None = unknown) and pictures shown under the prompt (https URLs).
    bank: str = ""
    peer_correct: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    images: list[str] = []
    # The stem's video (AMBOSS "Cardiac examination is shown": heart sounds on
    # YouTube), played under the prompt; https://www.youtube.com/watch?v=<id> only.
    videos: list[str] = []
    # Oral course only (Course.oral): the target speaking time, [lo, hi]
    # seconds, shown beside how long the answer took. [] = no target.
    say_in: list[int] = Field(default=[], max_length=2)


def is_case(q: Question) -> bool:
    """A case question: combined, on its own clock (Question.seconds)."""
    return q.kind == "combined" and q.seconds > 0


def case_key(q: Question) -> list[tuple[str, str]]:
    """A case reference's answer-key lines, one per target in target order:
    "- <order> — <reason>" -> (order, reason). "- Do not order" lines are not
    targets. pack.problems checks there is one per target."""
    rows = [x[2:] for x in q.reference.split("\n") if x.startswith("- ") and not x.startswith("- Do not order")]
    return [tuple(x.split(" — ", 1)) if " — " in x else (x, "") for x in rows]


class Figure(_M):
    """An interactive figure a lesson embeds: a vendored widget (`demo`, rendered
    with `props` by the figure host in a sandboxed iframe), with its title and
    caption in the words of the course it comes from, and a credit link."""
    id: str
    demo: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9]*$")
    props: dict = {}
    title: str = ""
    caption: str = ""
    credit: str
    url: str


class Diagram(_M):
    """A labelled picture for image-occlusion cards. `boxes` are pixel boxes
    [x0, y0, x1, y1] on the w x h image, one per label; the files are
    `<pack>/diagrams/<id>.webp` and `<id>.masked.webp` (Pack.diagram_file)."""
    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    title: str
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    boxes: list[tuple[int, int, int, int]] = Field(min_length=1)
    labels: list[str] = []   # what each box hides: grading refuses another box's label


class Pack(_M):
    id: str
    title: str
    version: str
    vetted: bool = False
    course: str = ""
    sources: list[Source]
    sections: list[Section] = []
    courses: list[Course] = []
    concepts: list[Concept]
    edges: list[Edge] = []
    passages: list[Passage]
    questions: list[Question]
    figures: list[Figure] = []
    diagrams: list[Diagram] = []
    _ix: dict = PrivateAttr(default_factory=dict)

    def model_post_init(self, _ctx) -> None:
        self._ix.update(
            concepts={c.id: c for c in self.concepts},
            passages={x.id: x for x in self.passages},
            questions={q.id: q for q in self.questions},
            sources={s.id: s for s in self.sources},
            figures={f.id: f for f in self.figures},
            diagrams={d.id: d for d in self.diagrams},
        )

    def concept(self, cid: str) -> Concept:
        return self._ix["concepts"][cid]

    def passage(self, pid: str) -> Passage:
        return self._ix["passages"][pid]

    def question(self, qid: str) -> Question:
        return self._ix["questions"][qid]

    def source(self, sid: str) -> Source:
        return self._ix["sources"][sid]

    def figure(self, fid: str) -> Figure:
        return self._ix["figures"][fid]

    def diagram(self, did: str) -> Diagram:
        return self._ix["diagrams"][did]

    def diagram_file(self, did: str, masked: bool) -> Path:
        from app.concept_tutor import contrib   # a live-published course's pictures are on the volume
        return (contrib.diagram_file(self.id, did, masked)
                or PACKS_DIR / self.id / "diagrams" / f"{did}{'.masked' if masked else ''}.webp")

    def questions_for(self, cid: str) -> list[Question]:
        # Indexed by target, once per questions LIST: a voice pack (speech.py)
        # is a model_copy with its own list but this same _ix, and must see
        # only its own questions. A linear scan here made problems() and every
        # cold load() O(concepts x questions), which jammed prod on 2026-10-11.
        by = self._ix.setdefault("by_target", {})
        hit = by.get(id(self.questions))
        if hit is None or hit[0] is not self.questions:
            ix: dict[str, list[Question]] = {}
            for q in self.questions:
                for t in dict.fromkeys(q.targets):
                    ix.setdefault(t, []).append(q)
            hit = by[id(self.questions)] = (self.questions, ix)
        return list(hit[1].get(cid, ()))

    def ordered(self) -> list[Concept]:
        return sorted(self.concepts, key=lambda c: c.order)

    def _course_sets(self, course_id: str) -> tuple[frozenset, frozenset]:
        """(own, all): the concepts of a course's sections, and those plus the
        targets of the cases it borrows (Course.cases)."""
        memo = self._ix.setdefault("course_concepts", {})   # shared with model_copy()s, so keyed by list identity too
        hit = memo.get(course_id)
        if hit is None or hit[0] is not self.concepts or hit[1] is not self.courses or hit[2] is not self.questions:
            k = next((k for k in self.courses if k.id == course_id), None)
            secs = set(k.sections) if k else set()
            own = frozenset(c.id for c in self.concepts if c.section in secs)
            qs = {q.id: q for q in self.questions} if k and k.cases else {}
            more = {t for x in (k.cases if k else []) if x in qs for t in qs[x].targets}
            hit = memo[course_id] = (self.concepts, self.courses, self.questions, own, own | more)
        return hit[3], hit[4]

    def course_concepts(self, course_id: str) -> set[str]:
        """The concept ids of one course, its borrowed cases' targets included;
        the whole pack for a pack with no courses."""
        if not self.courses:
            return {c.id for c in self.concepts}
        return set(self._course_sets(course_id)[1])

    def own_concepts(self, course_id: str) -> set[str]:
        """The concept ids of one course's own sections: who may SEE a concept,
        and which course it belongs to (a borrowed case's stay its owner's)."""
        if not self.courses:
            return {c.id for c in self.concepts}
        return set(self._course_sets(course_id)[0])

    def lessonless(self, cid: str) -> bool:
        """`cid` belongs to a course with no lessons (Course.lessons False)."""
        return any(not k.lessons and cid in self._course_sets(k.id)[0] for k in self.courses)



def lesson_text(c: Concept) -> str:
    """An authored lesson (Concept.lesson) and its worked examples as plain text,
    for the Ask prompt: the chat under a lesson answers from what is on screen."""
    out = []
    for b in c.lesson:
        if b.kind == "list":
            out += ["- " + "".join(x.text for x in item) for item in b.items]
        else:
            t = "".join(x.text for x in b.parts)
            out.append(f"[Figure] {t}" if b.kind == "image" else t)
    for x in c.examples:
        steps = "\n".join(f"{s.text} (Why: {s.why})" for s in x.steps)
        out.append(f"Worked example, {x.title}: {x.problem}\nPlan: {x.plan}\n{steps}\nAnswer: {x.answer}")
    return "\n".join(out)


def _flat(s: str) -> str:
    """Whitespace collapsed and `**` dropped: a quote may cross a source's line
    break or bold, never change a word."""
    return " ".join(s.replace("**", "").split())


YOUTUBE = re.compile(r"https://www\.youtube\.com/watch\?v=[A-Za-z0-9_-]{11}")
_BARE_MATHS = re.compile(r"\d\s*[-−+×*/÷]\s*\d|[=<>≤≥±]")


def bare_maths(s: str) -> bool:
    """Arithmetic or a relation written outside $...$: it shows as plain text,
    not KaTeX (Seth, 2026-10-11: lessons and examples write their maths in LaTeX)."""
    return bool(_BARE_MATHS.search(re.sub(r"\$[^$]*\$", "", s)))


def lesson_problems(p: Pack) -> list[str]:
    """An authored lesson (Concept.lesson): every quoted part is VERBATIM from its
    block's passage, that passage is one of the concept's own (the pill shows
    it), and an image is an https URL."""
    out: list[str] = []
    pas = {x.id: x for x in p.passages}
    for c in (c for c in p.concepts if c.lesson):
        for i, b in enumerate(c.lesson):
            where = f"{c.id}: lesson block {i}"
            parts = b.parts + [x for item in b.items for x in item]
            if b.passage and b.passage not in c.passages:
                out.append(f"{where}: passage {b.passage} is not one of the concept's passages")
                continue
            quoted = [x for x in parts if x.quote]
            if quoted and not b.passage:
                out.append(f"{where}: quotes without a passage")
                continue
            src = _flat(pas[b.passage].text) if b.passage in pas else ""
            out += [f"{where}: quote not verbatim in {b.passage}: {x.text[:60]!r}" for x in quoted
                    if _flat(x.text) not in src]
            if b.kind == "image" and not b.src.startswith("https://"):
                out.append(f"{where}: image src is not an https URL")
            if b.kind == "list" and (not b.items or b.parts):
                out.append(f"{where}: a list holds items only, and at least one")
            if b.kind != "list" and (not b.parts or b.items):
                out.append(f"{where}: no text, or items outside a list")
            if b.src and b.kind != "image":
                out.append(f"{where}: src outside an image")
            # A heading carries no pill: the block under it credits its quote.
            nxt = c.lesson[i + 1] if i + 1 < len(c.lesson) else None
            if b.kind == "heading" and quoted and (nxt is None or nxt.passage != b.passage):
                out.append(f"{where}: a quoted heading must share the next block's passage")
        glue = [x.text for b in c.lesson for x in b.parts + [y for it in b.items for y in it] if not x.quote]
        ex = [t for x in c.examples for t in (x.problem, x.plan, x.answer, *(f for st in x.steps for f in (st.text, st.why)))]
        out += [f"{c.id}: maths outside $...$: {t[:60]!r}" for t in glue + ex if bare_maths(t)]
    out += [f"{c.id}: worked examples without a lesson" for c in p.concepts if c.examples and not c.lesson]
    return out


def problems(p: Pack) -> list[str]:
    """Structural faults a pack must not ship with. Empty = loadable. The
    verbatim-source check is offline (check_pack.py): it needs the source files."""
    out: list[str] = []
    # Lookups index by id and keep the LAST duplicate while lists keep all, so a
    # duplicate id would resolve to different content than the planner iterates.
    for kind, ids in (("concept", [c.id for c in p.concepts]), ("passage", [x.id for x in p.passages]),
                      ("question", [q.id for q in p.questions]), ("source", [s.id for s in p.sources])):
        # Counter, not ids.count per id: quadratic over ~12k questions cost 43 s of a cold load on
        # one shared CPU (2026-10-11, pack 0.17.0)
        out += [f"duplicate {kind} id {i}" for i, n in sorted(Counter(ids).items()) if n > 1]
    concepts = {c.id for c in p.concepts}
    section_ids = [s.id for s in p.sections]
    out += [f"duplicate section id {i}" for i in sorted({i for i in section_ids if section_ids.count(i) > 1})]
    if p.sections:
        out += [f"{c.id}: unknown section {c.section!r}" for c in p.concepts if c.section not in section_ids]
    course_ids = [k.id for k in p.courses]
    out += [f"duplicate course id {i}" for i in sorted({i for i in course_ids if course_ids.count(i) > 1})]
    if p.courses:
        owned = [s for k in p.courses for s in k.sections]
        out += [f"section {s} is in no course" for s in section_ids if s not in owned]
        out += [f"section {s} is in {owned.count(s)} courses" for s in sorted(set(owned)) if owned.count(s) > 1]
        out += [f"course section {s} is unknown" for s in sorted(set(owned)) if s not in section_ids]
    passages = {x.id for x in p.passages}
    sources = {s.id for s in p.sources}
    fig_ids = [f.id for f in p.figures]
    out += [f"duplicate figure id {i}" for i in sorted({i for i in fig_ids if fig_ids.count(i) > 1})]
    out += [f"{c.id}: unknown figure {f}" for c in p.concepts for f in c.figures if f not in set(fig_ids)]
    out += [f"course {k.id}: public and private_to at once" for k in p.courses if k.public and k.private_to]
    dia = {d.id: d for d in p.diagrams}
    dia_ids = [d.id for d in p.diagrams]
    out += [f"duplicate diagram id {i}" for i in sorted({i for i in dia_ids if dia_ids.count(i) > 1})]
    out += [f"{c.id}: unknown diagram {d}" for c in p.concepts for d in c.diagrams if d not in dia]
    out += [f"diagram {d.id}: {len(d.labels)} labels for {len(d.boxes)} boxes" for d in p.diagrams
            if len(d.labels) != len(d.boxes)]
    out += [f"diagram {d.id}: box {b} is outside the {d.w}x{d.h} image" for d in p.diagrams for b in d.boxes
            if not (0 <= b[0] < b[2] <= d.w and 0 <= b[1] < b[3] <= d.h)]
    for k in p.courses:
        uids = [u.id for u in k.units]
        out += [f"course {k.id}: duplicate unit id {i}" for i in sorted({i for i in uids if uids.count(i) > 1})]
        owned = [s for u in k.units for s in u.sections]
        out += [f"course {k.id}, unit {u.id}: section {s} is not the course's" for u in k.units
                for s in u.sections if s not in k.sections]
        out += [f"course {k.id}: section {s} in two units" for s in sorted({s for s in owned if owned.count(s) > 1})]
        if k.units:   # a section in no unit could never be switched off
            out += [f"course {k.id}: section {s} in no unit" for s in k.sections if s not in owned]
    for k in (k for k in p.courses if k.mode == "qbank"):
        areas = p.course_concepts(k.id)
        out += [f"course {k.id}: area weight on unknown area {a}" for a in k.area_weights if a not in areas]
        out += [f"course {k.id}: area {a} is not a math (MC) concept" for a in sorted(areas)
                if p.concept(a).kind != "math"]
        out += [f"course {k.id}: area weight {a}={w} is not a finite number >= 0" for a, w in k.area_weights.items()
                if not (math.isfinite(w) and w >= 0)]
        out += [f"{q.id}: a bank question must be one-target MC with its bank named" for q in p.questions
                if set(q.targets) & areas and (q.format != "mc" or len(q.targets) != 1 or q.kind != "content" or not q.bank)]
    out += [f"{q.id}: image {u!r} is not an https URL" for q in p.questions
            for u in q.images + [u for o in q.options for u in o.images] if not u.startswith("https://")]
    out += [f"{q.id}: video {u!r} is not a YouTube watch URL" for q in p.questions for u in q.videos
            if not YOUTUBE.fullmatch(u)]
    out += lesson_problems(p)
    for k in p.courses:   # Course.cases: another course's cases, asked here too
        qs = {q.id: q for q in p.questions}
        own = p.own_concepts(k.id)
        out += [f"course {k.id}: borrowed case {x} is not a case" for x in k.cases if x not in qs or not is_case(qs[x])]
        out += [f"course {k.id}: borrowed case {x} is its own" for x in k.cases
                if x in qs and set(qs[x].targets) <= own]
        # Study plans by concept, so the cases are the allowlist only while no other
        # combined question fits the borrowed concepts too.
        wide = p.course_concepts(k.id)
        out += [f"course {k.id}: {q.id} fits its borrowed concepts but is not one of its cases" for q in p.questions
                if k.cases and q.kind == "combined" and q.id not in k.cases and set(q.targets) <= wide
                and not set(q.targets) <= own]
    out += [f"{q.id}: links on a question that is not a case" for q in p.questions if q.links and not is_case(q)]
    out += [f"{q.id}: unknown link {x}" for q in p.questions for x in q.links if x not in concepts]
    links = [k.link for k in p.courses if k.link]
    out += [f"duplicate course link {x}" for x in sorted({x for x in links if links.count(x) > 1})]
    # A public course republishes its quotations, so every source it quotes states its reuse terms.
    pas, src = {x.id: x for x in p.passages}, {x.id: x for x in p.sources}
    for k in (k for k in p.courses if k.public):
        quoted = {pas[x].source for c in p.concepts if c.section in k.sections for x in c.passages if x in pas}
        out += [f"course {k.id}: source {sid} has no license" for sid in sorted(quoted) if sid in src and not src[sid].license]
    seen_alias: dict[str, str] = {}
    for c in p.concepts:
        for a in c.aliases:
            k = a.casefold()
            if k in seen_alias and seen_alias[k] != c.id:
                out.append(f"alias {a!r} on both {seen_alias[k]} and {c.id}")
            seen_alias[k] = c.id
            if len(a.split()) > 6:
                out.append(f"{c.id}: alias {a!r} is longer than 6 words")
        out += [f"{c.id}: unknown passage {x}" for x in c.passages if x not in passages]
        out += [f"{c.id}: unknown prereq {x}" for x in c.prereqs if x not in concepts]
        qs = p.questions_for(c.id)
        if c.kind == "conceptual" and len(c.constraints) < 2:
            out.append(f"{c.id}: a conceptual concept needs >= 2 constraints")
        if not any(q.kind == "content" for q in qs):
            out.append(f"{c.id}: no content question")
        if c.kind == "conceptual" and not any(q.kind in ("evaluative", "relation") for q in qs):
            out.append(f"{c.id}: no evaluative/relation question")
    out += [f"passage {x.id}: unknown source {x.source}" for x in p.passages if x.source not in sources]
    for q in p.questions:
        for cid in q.targets + q.expected + q.required:
            if cid not in concepts:
                out.append(f"{q.id}: unknown concept {cid}")
        if not set(q.required) <= set(q.expected) | set(q.targets):
            out.append(f"{q.id}: required must be among targets/expected")
        if q.format == "mc":
            if sum(o.correct for o in q.options) != 1 or len(q.options) < 3:
                out.append(f"{q.id}: an mc question needs >= 3 options and exactly one correct")
            ids = [o.id for o in q.options]
            if len(set(ids)) != len(ids):
                out.append(f"{q.id}: duplicate option ids")  # the client submits only the id
        elif q.options:
            out.append(f"{q.id}: options on a {q.format} question")
        if q.format == "label":
            d = dia.get(q.diagram)
            if d is None or not 0 <= q.box < len(d.boxes):
                out.append(f"{q.id}: a label question needs a known diagram and one of its boxes")
            if q.kind not in ("content", "check") or len(q.targets) != 1 or not q.reference.strip():
                out.append(f"{q.id}: a label question is a content/check card on one concept with a reference")
        elif q.format == "card":
            d = dia.get(q.diagram) if q.diagram else None
            if q.diagram and (d is None or not -1 <= q.box < len(d.boxes)):
                out.append(f"{q.id}: a card's diagram must be known, its box -1 or one of its boxes")
            if not q.diagram and q.box != -1:
                out.append(f"{q.id}: a box without a diagram")
            if q.kind not in ("content", "check") or len(q.targets) != 1 or not q.reference.strip():
                out.append(f"{q.id}: a card is a content/check question on one concept with a reference")
        elif q.diagram or q.box != -1 or q.answers:
            out.append(f"{q.id}: diagram/box/answers on a {q.format} question")
        if q.group and q.format != "card":
            out.append(f"{q.id}: a group on a {q.format} question")
        if q.reference and q.reference.strip() and q.reference.strip()[:60] in q.prompt:
            out.append(f"{q.id}: the prompt contains its own reference")
        if q.kind == "combined":
            out += _combined_problems(p, q)
        elif q.integration is not None:
            out.append(f"{q.id}: only a combined question is listed or integrated")
        if (q.key_rows or q.setting) and not is_case(q):
            out.append(f"{q.id}: only a case has key rows and a setting")
    for e in p.edges:
        out += [f"edge {e.a}|{e.b}: unknown concept {x}" for x in (e.a, e.b) if x not in concepts]
    out += _strong_prereq_cycles(p)
    return out


def _combined_problems(p: Pack, q: Question) -> list[str]:
    """A combined probe (explore.py): free text on 2-MAX_COMBINED distinct conceptual
    targets of one course (2-MAX_CASE for a case), so one answer grades each of them."""
    out = []
    cs = {c.id: c for c in p.concepts}
    cap = MAX_CASE if is_case(q) else MAX_COMBINED
    if q.format != "free" or not 2 <= len(set(q.targets)) <= cap or len(set(q.targets)) != len(q.targets):
        out.append(f"{q.id}: a {'case' if is_case(q) else 'combined question'} is free text on 2-{cap} distinct targets")
    if is_case(q) and q.integration != "listed":
        out.append(f"{q.id}: a case is listed (each target graded on its own), not integrated")
    key = case_key(q) if is_case(q) else []
    if is_case(q) and (not q.reference.startswith("Diagnosis:") or len(key) != len(q.targets)
                       or not all(o.strip() and why for o, why in key)):   # case.py grades + case_feedback quotes these
        out.append(f"{q.id}: a case reference is 'Diagnosis: ...' then one '- <order> — <reason>' line per target")
    if q.key_rows:   # voice.case_feedback colours each order row by its target's grade
        rows = {r.target for r in q.key_rows if r.kind == "order"}
        if (rows != set(q.targets) or any(bool(r.target) != (r.kind == "order") for r in q.key_rows)
                or not all(r.order.strip() for r in q.key_rows)):
            out.append(f"{q.id}: case key rows: one or more 'order' rows per target, only those name one, none blank")
    if is_case(q):   # the fallback order grade/case.py asks about when a key row names none
        out += [f"{q.id}: case target {t} needs a 'decision' constraint (the order it grades)"
                for t in q.targets if t in cs and not any(k.id == "decision" for k in cs[t].constraints)]
    if any(t in cs and cs[t].kind != "conceptual" for t in q.targets):
        out.append(f"{q.id}: combined targets must be conceptual (math is probed one concept at a time)")
    if p.courses and not any(set(q.targets) <= p.own_concepts(k.id) for k in p.courses):   # a borrowed case too
        out.append(f"{q.id}: combined targets span courses")
    if q.integration is None:
        out.append(f"{q.id}: a combined question must say whether it is listed or integrated")
    return out


def target_pairs(q: Question) -> list[tuple[str, str]]:
    """Every pair of an INTEGRATED question's targets, in target order: the
    relations its answer must state (grader.py grades each with pass C)."""
    if q.kind != "combined" or q.integration != "integrated":
        return []
    t = q.targets
    return [(t[i], t[j]) for i in range(len(t)) for j in range(i + 1, len(t))]


def _strong_prereq_cycles(p: Pack) -> list[str]:
    graph = {c.id: [x for x, w in c.prereqs.items() if w >= 0.5] for c in p.concepts}
    state: dict[str, int] = {}

    def visit(n: str) -> bool:
        state[n] = 1
        for m in graph.get(n, []):
            if state.get(m) == 1 or (state.get(m) is None and visit(m)):
                return True
        state[n] = 2
        return False

    return [f"prereq cycle through {n}" for n in graph if state.get(n) is None and visit(n)]


def _read(folder: Path, name: str, default):
    f = folder / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else default


_loading = threading.Lock()
_cache: dict[str, tuple[int, "Pack"]] = {}


def load(pack_id: str) -> Pack:
    """The pack, built and checked once. One thread builds it while the rest
    wait: lru_cache alone let every request of a cold server build its own
    copy at once (2026-10-11: eleven at a time on one shared CPU, each holding
    a DB connection, until the pool ran out and the backend stopped answering).
    Rebuilt when a contributor publishes a course live (contrib.stamp moves);
    only the newest build is kept, so an old pack is not held in memory."""
    from app.concept_tutor import contrib
    with _loading:
        key = contrib.stamp(pack_id)
        hit = _cache.get(pack_id)
        if hit is not None and hit[0] == key:
            return hit[1]
        pack = _load(pack_id)
        _cache.pop(pack_id, None)
        _cache[pack_id] = (key, pack)
        while len(_cache) > 8:          # the bound lru_cache(maxsize=8) had
            _cache.pop(next(iter(_cache)))
        return pack


def _build(files: dict) -> Pack:
    pack = Pack(**files["pack"], concepts=files["concepts"], edges=files["edges"],
                passages=files["passages"], questions=files["questions"])
    bad = problems(pack)
    if bad:
        raise ValueError(f"concept pack {pack.id} is invalid: " + "; ".join(bad[:8]))
    return pack


def _load(pack_id: str) -> Pack:
    from app.concept_tutor import contrib
    folder = PACKS_DIR / pack_id
    meta = _read(folder, "pack.json", None)
    if meta is None:
        raise FileNotFoundError(f"no concept pack {pack_id!r} in {PACKS_DIR}")
    files = {"pack": meta, **{name: _read(folder, f"{name}.json", [])
                              for name in ("concepts", "edges", "passages", "questions")}}
    return contrib.build(pack_id, files, _build)


load.cache_clear = _cache.clear   # check_pack.py and tests reset it
