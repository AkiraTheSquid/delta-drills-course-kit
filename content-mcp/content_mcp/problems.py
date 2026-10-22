"""Multiple-choice problems: `lessons/<course>/kp-<slug>.problems.json`.

The math lane (and any code page that owns MC items) keeps its problems in a
JSON bank beside the page, NOT in the CSV drill bank. The schema and every
rule live in `lessons/math_bank.py` and `scripts/validate_math.py`; this module
only reads and writes the file, mints ids, and refuses the shapes the exporter
would choke on. The SymPy proof (`truth == value[answer]`, every distractor
different) is the validator's job — run pipeline_check after adding problems.

Ids are EXPLICIT (>= 50000), unlike the positional CSV ids: a new id is one
above the largest id any problems file holds, so two pages never share one.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import lessons, paths

ID_FLOOR = 50000
KINDS = ("compute", "derivation-step", "statement")
SYMPY_KINDS = ("compute", "derivation-step")


def _bank_path(kc: str) -> Path:
    page = lessons.resolve(kc)
    return page.with_name(page.stem + ".problems.json")


def _all_ids() -> dict[int, str]:
    found: dict[int, str] = {}
    for path in sorted(paths.LESSONS.glob("*/kp-*.problems.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for problem in data.get("problems", []):
            if isinstance(problem, dict) and isinstance(problem.get("id"), int):
                found[problem["id"]] = str(path.relative_to(paths.REPO))
    return found


def _save(path: Path, data: dict) -> None:
    # The file's own indent (the existing banks use 2), so an append is a small diff.
    indent = paths.json_indent_of(path, default=2) if path.exists() else 2
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False) + "\n")


def next_id() -> int:
    ids = _all_ids()
    return max([ID_FLOOR - 1, *ids]) + 1


def read(kc: str) -> dict:
    path = _bank_path(kc)
    if not path.exists():
        return {"kc": kc, "file": str(path.relative_to(paths.REPO)), "exists": False, "problems": []}
    data = json.loads(path.read_text())
    return {"kc": kc, "file": str(path.relative_to(paths.REPO)), "exists": True, **data}


def _shape(problem: dict, label: str) -> list[str]:
    """The structural rules — enough that a write can never produce a file the
    loader cannot read. The proof itself is validate_math.py's."""
    errs = []
    if problem.get("kind") not in KINDS:
        errs.append(f"{label}: kind must be one of {KINDS}")
    for field in ("prompt", "choices", "answer", "solution"):
        if field not in problem:
            errs.append(f"{label}: missing `{field}`")
    choices = problem.get("choices") or []
    if not isinstance(choices, list) or not 2 <= len(choices) <= 6:
        errs.append(f"{label}: 2-6 choices")
    else:
        keys = [c.get("key") for c in choices if isinstance(c, dict)]
        if keys != [chr(ord("A") + i) for i in range(len(choices))]:
            errs.append(f"{label}: choice keys must be A, B, C… in order (got {keys})")
        if problem.get("answer") not in keys:
            errs.append(f"{label}: answer {problem.get('answer')!r} is not one of the keys")
        if problem.get("kind") in SYMPY_KINDS and not all("value" in c for c in choices):
            errs.append(f"{label}: a {problem.get('kind')} problem needs a SymPy `value` on every choice")
    sympy = (problem.get("verify") or {}).get("sympy")
    if problem.get("kind") in SYMPY_KINDS and not (sympy and sympy.get("truth")):
        errs.append(f"{label}: a {problem.get('kind')} problem needs verify.sympy.truth")
    if problem.get("kind") == "statement" and sympy:
        errs.append(f"{label}: a statement problem must not carry verify.sympy")
    difficulty = problem.get("difficulty", 50)
    if not isinstance(difficulty, int) or not 10 <= difficulty <= 100:
        errs.append(f"{label}: difficulty must be an integer 10-100")
    return errs


def add(kc: str, problems: list[dict]) -> dict:
    """Append problems to the KC's bank, minting an id for any that has none."""
    if not problems:
        raise ValueError("No problems given.")
    path = _bank_path(kc)
    data = json.loads(path.read_text()) if path.exists() else {"kc": kc, "problems": []}
    if data.get("kc") != kc:
        raise ValueError(f"{path.name} says kc {data.get('kc')!r}, not {kc!r}.")
    taken = _all_ids()
    explicit = [p["id"] for p in problems if isinstance(p.get("id"), int)]
    minted = max([ID_FLOOR - 1, *taken, *explicit]) + 1
    added, errors = [], []
    for i, raw in enumerate(problems):
        problem = dict(raw)
        if "id" not in problem:
            problem["id"] = minted
            minted += 1
        pid = problem["id"]
        if not isinstance(pid, int) or pid < ID_FLOOR:
            errors.append(f"problem #{i}: id must be an integer >= {ID_FLOOR}")
            continue
        if pid in taken:
            errors.append(f"problem #{i}: id {pid} is already used in {taken[pid]}")
            continue
        problem.setdefault("difficulty", 50)
        errors += _shape(problem, f"problem #{i} (id {pid})")
        taken[pid] = str(path.relative_to(paths.REPO))
        added.append(problem)
    if errors:
        raise ValueError("Nothing written: " + "; ".join(errors))
    data["problems"].extend(added)
    _save(path, data)
    return {
        "kc": kc,
        "file": str(path.relative_to(paths.REPO)),
        "added_ids": [p["id"] for p in added],
        "next_step": "List the ids under the page's practice sections (frontmatter + `### q<id>`), "
                     "then pipeline_check — validate_math proves every answer key in SymPy.",
    }


def update(kc: str, problem_id: int, fields: dict) -> dict:
    """Change fields of one existing problem in place (its id stays)."""
    if "id" in fields:
        raise ValueError("A problem's id never changes.")
    path = _bank_path(kc)
    if not path.exists():
        raise ValueError(f"{kc} has no problems file yet — use problems_add.")
    data = json.loads(path.read_text())
    for problem in data.get("problems", []):
        if problem.get("id") == problem_id:
            merged = {**problem, **fields}
            errors = _shape(merged, f"id {problem_id}")
            if errors:
                raise ValueError("Nothing written: " + "; ".join(errors))
            problem.clear()
            problem.update(merged)
            _save(path, data)
            return {"kc": kc, "id": problem_id, "fields": sorted(fields)}
    raise KeyError(f"No problem {problem_id} in {path.name}.")
