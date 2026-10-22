#!/usr/bin/env python3
"""Check (and fill) lessons/arena_groups.json — ARENA's exercise GROUPS.

A group is the third level under a section (0.1 -> 0.1.2): a run of ARENA's own
exercises whose code builds on each other, worked in order as one small
codebase. The groups and their parts' `fn` are authored; this script fills
each part's `cell` (the exercise's heading cell, which ArenaNotebook.open
scrolls to), `kc` and `title`, and refuses a file where

  * a part is not in lessons/arena_exercise_kcs.json, or its concept is not
    in the registry (no concept = nothing to judge readiness or progress by);
  * a part has no stub in its notebook (given code is not an exercise);
  * a part sits in two groups, or parts are out of notebook order (within a
    group or across groups);
  * a part has no reference solution under its exercise (a miss is finished
    for the learner with it);
  * a part's code uses a function that a LATER part (or a later group)
    defines — the order would make the learner call code they haven't
    written yet.

Usage:
    python3 scripts/check_arena_groups.py            # check, exit 1 on a finding
    python3 scripts/check_arena_groups.py --write    # fill cell/kc/title, then check
    python3 scripts/check_arena_groups.py --deps     # print every part's uses

The compiled notebooks (lessons/notebooks/arena-*.json) are gitignored build
output (scripts/compile_arena_notebooks.py); a section whose notebook is
missing is reported, not guessed at.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARENA_ROOT") or Path(__file__).resolve().parents[1])
LESSONS = ROOT / "Local_Deployed_Shared" / "lessons"
GROUPS = Path(os.environ.get("ARENA_GROUPS") or LESSONS / "arena_groups.json")
MAP = LESSONS / "arena_exercise_kcs.json"
REGISTRY = LESSONS / "kc_registry.json"
NOTEBOOK = LESSONS / "notebooks" / "arena-{}.json"

EXERCISE = re.compile(r"^#{1,6}\s+(?:Bonus\s+)?Exercise\b", re.M | re.I)
STUB = re.compile(r"your code here|YOUR_CODE_HERE|NotImplementedError|#\s*TODO|^\s*pass\s*$", re.I | re.M)


def _defines(src: str, fn: str) -> bool:
    return re.search(r"^\s*(?:def|class)\s+%s\b" % re.escape(fn), src, re.M) is not None


def _block(src: str, fn: str) -> str:
    """The `def fn`/`class fn` block in `src`, docstrings and comments removed —
    the code a learner writes for this part, not the prose around it."""
    lines = src.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"(?:def|class)\s+%s\b" % re.escape(fn), l)), None)
    if start is None:
        return ""
    end = next((i for i in range(start + 1, len(lines))
                if lines[i] and not lines[i][0].isspace() and not lines[i].startswith((")", "#"))), len(lines))
    body = "\n".join(lines[start + 1:end])
    body = re.sub(r'("""|\'\'\')[\s\S]*?\1', "", body)
    return re.sub(r"#.*", "", body)


def _uses(code: str, fn: str) -> bool:
    """`code` names `fn` itself — not `F.fn` / `t.nn.functional.fn`, which is
    torch's function of the same name (ARENA's Conv2d calls F.conv2d)."""
    return re.search(r"(?<![\w.])%s\b" % re.escape(fn), code) is not None


def locate(cells: list[dict], fn: str) -> dict | None:
    """The stub cell that defines `fn`, its exercise heading, and the code of
    the part: the stub's block plus the reference solution's block."""
    stub = next((i for i, c in enumerate(cells)
                 if c["t"] != "md" and c.get("role") != "details"
                 and _defines(c["src"], fn) and STUB.search(c["src"])), None)
    if stub is None:
        return None
    heads = [i for i, c in enumerate(cells)
             if c["t"] == "md" and c.get("role") != "details" and EXERCISE.search(c["src"])]
    head = max((i for i in heads if i <= stub), default=None)
    if head is None:
        return None
    end = min((i for i in heads if i > head), default=len(cells))
    details = [c["src"] for c in cells[stub + 1:end] if c.get("role") == "details"]
    solution = next((src for src in details if _defines(src, fn)), "")
    # A class whose solution is one of its stubbed methods (BatchNorm2d's is
    # only `forward`): practice/arena-groups.js splices it into the class.
    methods = set(re.findall(r"^\s+def\s+(\w+)\s*\(\s*self\b", _block(cells[stub]["src"], fn), re.M))
    method = next((src for src in details
                   if any(re.search(r"^def\s+%s\s*\(\s*self\b" % m, src, re.M) for m in methods)), "")
    code = _block(cells[stub]["src"], fn) + "\n" + (_block(solution, fn) or method)
    return {"stub": stub, "head": head, "cell": cells[head]["id"], "code": code,
            "solved": bool(solution or method)}


def check(write: bool = False, show_deps: bool = False) -> list[str]:
    data = json.loads(GROUPS.read_text())
    emap = json.loads(MAP.read_text())
    reg = {k["id"]: k for k in json.loads(REGISTRY.read_text())["kcs"]}
    problems: list[str] = []
    seen_ids: set[str] = set()

    for slug, groups in data["sections"].items():
        path = Path(str(NOTEBOOK).format(slug))
        if not path.exists():
            problems.append(f"{slug}: compiled notebook missing ({path.name}) — run scripts/compile_arena_notebooks.py")
            continue
        cells = json.loads(path.read_text())["cells"]
        table = emap.get(slug, {})
        order: list[tuple[str, str, int]] = []  # (group id, fn, stub index) in authored order
        placed: dict[str, str] = {}
        expect = slug.replace("-", ".")
        for n, group in enumerate(groups, 1):
            gid = group.get("id")
            if gid != f"{expect}.{n}":
                problems.append(f"{slug}: group #{n} is numbered {gid!r}, expected {expect}.{n}")
            if gid in seen_ids:
                problems.append(f"{gid}: duplicate group id")
            seen_ids.add(gid)
            if not group.get("title") or not group.get("parts"):
                problems.append(f"{gid}: needs a title and at least one part")
            for part in group.get("parts", []):
                fn = part.get("fn")
                if fn in placed:
                    problems.append(f"{gid}: {fn} is already a part of {placed[fn]}")
                    continue
                placed[fn] = gid
                row = table.get(fn)
                if not row or not row.get("kc"):
                    problems.append(f"{gid}: {fn} is not in arena_exercise_kcs.json[{slug}] with a kc")
                    continue
                if row["kc"] not in reg:
                    problems.append(f"{gid}: {fn}'s concept {row['kc']} is not in kc_registry.json")
                spot = locate(cells, fn)
                if not spot:
                    problems.append(f"{gid}: {fn} has no stub under an Exercise heading in {path.name}")
                    continue
                if not spot["solved"]:
                    # A missed part is finished FOR the learner with this code
                    # (practice/arena-groups.js) — without it the next part
                    # would call a function that was never written.
                    problems.append(f"{gid}: {fn} has no reference solution (a details cell defining it) under its exercise")
                want = {"cell": spot["cell"], "kc": row["kc"], "title": row.get("title") or fn}
                if write:
                    part.update(want)
                else:
                    for key, value in want.items():
                        if part.get(key) != value:
                            problems.append(f"{gid}: {fn}.{key} is {part.get(key)!r}, notebook says {value!r} (run --write)")
                order.append((gid, fn, spot["stub"]))
                part["_code"] = spot["code"]

        # Notebook order, within a group and across groups.
        for (g1, f1, s1), (g2, f2, s2) in zip(order, order[1:]):
            if s2 < s1:
                where = g1 if g1 == g2 else f"{g1} → {g2}"
                problems.append(f"{where}: {f2} comes before {f1} in the notebook but after it in the groups")
        # No part may lean on a later part's code.
        flat = [p for g in groups for p in g.get("parts", []) if "_code" in p]
        for i, part in enumerate(flat):
            later = [q["fn"] for q in flat[i + 1:] if _uses(part["_code"], q["fn"])]
            earlier = [q["fn"] for q in flat[:i] if _uses(part["_code"], q["fn"])]
            if later:
                problems.append(f"{placed[part['fn']]}: {part['fn']} uses {', '.join(later)}, which come later")
            if show_deps:
                print(f"{placed[part['fn']]:7s} {part['fn']:34s} uses {', '.join(earlier) or '-'}")
        for g in groups:
            for p in g.get("parts", []):
                p.pop("_code", None)

    if write:
        GROUPS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return problems


if __name__ == "__main__":
    found = check(write="--write" in sys.argv, show_deps="--deps" in sys.argv)
    for line in found:
        print("✗", line)
    if found:
        sys.exit(1)
    print(f"✓ {GROUPS.name}: every group checks out")
