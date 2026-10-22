"""Running the repo's own content pipeline.

Nothing here reimplements a check. Every entry shells out to the script the
repo already gates its deploys on, under the backend venv interpreter —
`python3` on this machine has no torch, and without torch validate_lessons.py
reports every torch drill broken and export_questions_json.py leaves
`expected_output` stale instead of recomputing it.

`check_all` is the order a content change has to survive — the one
lessons/AUTHORING.md lists: re-export the bank, compile the pages, rebuild the
Q-matrix, validate, then audit the bank.

🔴 EXPORT FIRST. It used to run validate first, and validate reads the bank
from the last export — so a drill just added with `drill_add` was "not in
bank" and the page that lists it could never pass. Validate also reads
`qmatrix_tags.json` for its coverage check, so qmatrix precedes it too.

The two notebook steps follow compile (both read lessons_structured.json): the
lessons watcher fails any unit without a Colab entry and a web notebook, so a
unit made by `course_create` could never pass until something generated them.
`symbols` rebuilds question_symbol_kcs.json from the bank + qmatrix (the scripts
watcher fails a stale one), so it follows qmatrix.
"""

from __future__ import annotations

import subprocess
import time

from . import paths

DEFAULT_TIMEOUT = 900


def _run(argv: list[str], cwd=None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd or paths.REPO),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "command": " ".join(argv), "timed_out": True,
                "seconds": round(time.time() - started, 1)}
    except FileNotFoundError as err:
        return {"ok": False, "command": " ".join(argv), "error": str(err)}
    return {
        "ok": proc.returncode == 0,
        "command": " ".join(argv),
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 1),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _script(path, *args: str) -> dict:
    return _run([paths.python_for_content(), str(path), *args])


STEPS = {
    "validate": lambda: _script(paths.VALIDATE_SCRIPT, "--coverage"),
    "compile": lambda: _script(paths.COMPILE_SCRIPT),
    "qmatrix": lambda: _script(paths.QMATRIX_SCRIPT),
    "export": lambda: _script(paths.EXPORT_SCRIPT),
    "audit_bank": lambda: _script(paths.BANK_AUDIT, "--gate"),
    "colab": lambda: _script(paths.COLAB_NOTEBOOKS_SCRIPT),
    "symbols": lambda: _script(paths.SCRIPTS / "build_symbol_prereqs.py"),
    "notebooks": lambda: _script(paths.WEB_NOTEBOOKS_SCRIPT),
}

AUDITS = {
    "symbol_coverage": "audit_symbol_coverage.py",
    "solution_prereqs": "audit_solution_prereqs.py",
    "arena_grounding": "audit_arena_grounding.py",
    "graph_structure": "audit_graph_structure.py",
    "lesson_syntax": "audit_lesson_syntax.py",
    "question_syntax": "audit_question_syntax.py",
    "prose_prereqs": "audit_prose_prereqs.py",
    "ladder_pairing": "audit_ladder_pairing.py",
}


def step(name: str) -> dict:
    if name not in STEPS:
        raise KeyError(f"Unknown step '{name}'. Known: {', '.join(STEPS)}")
    return STEPS[name]()


def audit(name: str, extra_args: list[str] | None = None) -> dict:
    if name not in AUDITS:
        raise KeyError(f"Unknown audit '{name}'. Known: {', '.join(AUDITS)}")
    return _script(paths.SCRIPTS / AUDITS[name], *(extra_args or []))


def check_all(fast: bool = False) -> dict:
    """The whole gate, in order, stopping at the first failure.

    `fast` skips the two slow steps (fence execution and the bank export) —
    useful while iterating on registry structure, never sufficient to ship.
    """
    order = ["compile", "qmatrix"] if fast else ["export", "compile", "colab", "notebooks", "qmatrix", "symbols", "validate", "audit_bank"]
    results = []
    for name in order:
        result = step(name)
        results.append({"step": name, **result})
        if not result.get("ok"):
            return {"ok": False, "failed_at": name, "steps": results}
    return {"ok": True, "steps": results}


def folder_watchers(folders: list[str] | None = None) -> dict:
    """Run the Modulario watch.py health checks for the content folders.

    These carry the standing content guards (prerequisite order, ARENA
    grounding, symbol coverage), so a drill reaching for an untaught function
    is refused where it was written rather than found at deploy time.
    """
    targets = folders or [
        "Local_Deployed_Shared/lessons",
        "scripts",
        "Local_Deployed_Shared/pipeline",
    ]
    results = []
    for folder in targets:
        watcher = paths.REPO / folder / "watch.py"
        if not watcher.exists():
            results.append({"folder": folder, "ok": True, "skipped": "no watch.py"})
            continue
        results.append({"folder": folder, **_run([paths.python_for_content(), str(watcher)])})
    return {"ok": all(r.get("ok", True) for r in results), "watchers": results}


def python_in_use() -> dict:
    interpreter = paths.python_for_content()
    has_torch = _run([interpreter, "-c", "import torch"], timeout=120).get("ok", False)
    return {
        "interpreter": interpreter,
        "is_backend_venv": paths.BACKEND_PY.exists(),
        "has_torch": has_torch,
        "warning": None if has_torch else (
            f"{interpreter} cannot import torch. Torch drills will validate as broken and "
            "expected_output will not be recomputed — install torch into it (see README)."
        ),
    }
