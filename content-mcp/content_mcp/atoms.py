"""Mastery atoms: the atom graph a drill's correct answer actually moves.

A KC's mastery is the weighted mean of per-atom mastery readings over its ATOMS, and a
drill reaches an atom only through a row in `question_atom_tags.jsonl`. A drill
with no row is graded, recorded, and moves nothing — `audit_question_bank.py
--gate` blocks it as `atom_tag_missing`. `drill_add` used to append to the CSV
and the overrides but never to this file, so every drill an AI authored failed
the bank gate, and a new course has no atom of its own to tag them with.

`atom_add` puts a node (and its prerequisite edges) into the atom graph
`arena_drillable_v1.json`; `drill_tag` writes a drill's tag row. The loader
(`backend/app/questions.py::_apply_atom_tags`) keeps the LAST row per question,
so a re-tag is an append, never an edit.
"""

from __future__ import annotations

import json
import re

from . import drills, paths

ATOM_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


def _graph() -> dict:
    return json.loads(paths.ATOM_GRAPH.read_text(encoding="utf-8"))


def _tag_rows() -> list[dict]:
    if not paths.ATOM_TAGS.exists():
        return []
    rows = []
    for line in paths.ATOM_TAGS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _trained() -> dict[str, int]:
    """atom -> how many drills' current (last) tag row names it."""
    last: dict[int, list] = {}
    for row in _tag_rows():
        last[int(row["question_id"])] = row.get("atoms", [])
    counts: dict[str, int] = {}
    for atoms in last.values():
        for tag in atoms:
            counts[tag["atom_id"]] = counts.get(tag["atom_id"], 0) + 1
    return counts


def list_atoms(query: str = "", limit: int = 40) -> dict:
    graph = _graph()
    trained = _trained()
    needle = query.lower().strip()
    hits = [
        {"id": c["id"], "title": c.get("title"), "topic": c.get("topic"),
         "subtopic": c.get("subtopic"), "drills": trained.get(c["id"], 0)}
        for c in graph["concepts"]
        if not needle or needle in c["id"] or needle in str(c.get("title", "")).lower()
        or needle in str(c.get("topic", "")).lower()
    ]
    return {"total_matches": len(hits), "returned": hits[:limit]}


def add(atom: str, title: str, description: str, topic: str, subtopic: str,
        prereqs: list[str] | None = None, root: bool = False) -> dict:
    """One new atom node, wired by its prerequisite edges (or declared a root)."""
    if not ATOM_RE.fullmatch(atom or ""):
        raise ValueError(f"Atom id '{atom}' must be lowercase words joined by dashes, e.g. 'pb-complement'.")
    prereqs = list(prereqs or [])
    if bool(prereqs) == bool(root):
        raise ValueError(
            "Pass prereqs (the atoms a learner must hold first) OR root=true, not both and not "
            "neither — the bank gate refuses an atom with no prerequisite edge that is not "
            "declared an intentional root (atom_tag_unwired)."
        )
    if not str(title).strip() or not str(description).strip():
        raise ValueError("An atom needs a title and a one-line description.")
    graph = _graph()
    ids = {c["id"] for c in graph["concepts"]}
    if atom in ids:
        raise ValueError(f"Atom '{atom}' already exists.")
    unknown = [p for p in prereqs if p not in ids]
    if unknown:
        raise ValueError(f"Unknown prerequisite atom(s): {unknown}. See atom_list.")
    graph["concepts"].append({
        "id": atom, "title": title.strip(), "topic": topic, "subtopic": subtopic,
        "node_type": "concept", "kind": "skill", "tier": "core",
        "description": description.strip(), "tags": ["drill-atom"],
    })
    for prereq in prereqs:
        graph["prerequisite_edges"].append({
            "prerequisite_id": prereq, "dependent_id": atom, "weight": 1.0,
            "confidence": 0.8, "is_hard_gate": True, "is_encompassing": False,
            "propagation_weight": 0.0, "rationale": "[builds-on] added through the content MCP",
        })
    if root:
        graph.setdefault("intentional_root_atoms", []).append(atom)
    indent = paths.json_indent_of(paths.ATOM_GRAPH, default=2)
    paths.ATOM_GRAPH.write_text(json.dumps(graph, indent=indent, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    trained = _trained()
    untrained = [p for p in prereqs if not trained.get(p)]
    return {
        "atom": atom, "prereqs": prereqs, "root": root,
        **({"warning": f"prerequisite(s) {untrained} have no tagged drill yet — the bank gate "
                       "fails (atom_prereq_untrainable) until each does."} if untrained else {}),
        "next_step": "drill_tag each of the concept's drills with this atom (or pass atoms= to drill_add).",
    }


def resolve(atoms: list) -> list[dict]:
    """Check tags BEFORE anything is written. Each item is an atom id or {atom_id, confidence}."""
    if not atoms:
        raise ValueError("Pass at least one atom.")
    ids = {c["id"] for c in _graph()["concepts"]}
    tags = []
    for item in atoms:
        atom_id = item if isinstance(item, str) else item.get("atom_id")
        confidence = 1.0 if isinstance(item, str) else float(item.get("confidence", 1.0))
        if atom_id not in ids:
            raise ValueError(f"Unknown atom '{atom_id}'. Add it with atom_add, or see atom_list.")
        if not 0.0 < confidence <= 1.0:
            raise ValueError(f"confidence for {atom_id} must be in (0, 1].")
        if any(t["atom_id"] == atom_id for t in tags):
            raise ValueError(f"atom '{atom_id}' is listed twice.")
        tags.append({"atom_id": atom_id, "confidence": confidence})
    return tags


def tag(qid: int, atoms: list) -> dict:
    """Write a drill's atom tags (the drill's whole tag set — the last row wins)."""
    qid = int(qid)
    # Ids are positional: a drill added since the last export is not in the
    # compiled bank yet but already owns its slot below next_id().
    if qid < 1 or not (drills.exists(qid) or qid < drills.next_id()):
        raise KeyError(f"No drill {qid}. Check the id with drill_search.")
    tags = resolve(atoms)
    with open(paths.ATOM_TAGS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"question_id": qid, "atoms": tags}) + "\n")
    return {"id": qid, "atoms": [t["atom_id"] for t in tags],
            "file": str(paths.ATOM_TAGS.relative_to(paths.REPO))}
