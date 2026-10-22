"""watch.py — health checks for study-courses

Each folder is a Study course written through the content MCP
(content_mcp/study.py). The full gate — the pack model, verbatim passages,
pictures — is `study_check` (scripts/study_course.py, needs pydantic); these
are the cheap structural invariants that keep a course loadable by the MCP.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("course", "sections", "sources", "concepts", "passages", "questions", "diagrams")


def _courses():
    for name in sorted(os.listdir(HERE)):
        f = os.path.join(HERE, name, "course.json")
        if os.path.isfile(f):
            with open(f, encoding="utf-8") as handle:
                yield name, json.load(handle)


def check_every_course_parses():
    for name, spec in _courses():
        missing = [k for k in KEYS if k not in spec]
        assert not missing, f"{name}/course.json lacks {missing}"
        assert spec["course"]["id"] == name, f"{name}/course.json names course {spec['course']['id']!r}"


def check_ids_are_namespaced():
    """Every id carries the course prefix, so applying a course can only ever
    replace its own content in the shared pack (study_course.py _owned)."""
    for name, spec in _courses():
        for key in ("sections", "sources", "concepts", "passages", "questions"):
            bad = [x["id"] for x in spec[key] if not x["id"].startswith(name + ".")]
            assert not bad, f"{name}: {key} not under '{name}.': {bad[:3]}"
        bad = [d["id"] for d in spec["diagrams"] if not d["id"].startswith(name + "-")]
        assert not bad, f"{name}: diagrams not under '{name}-': {bad[:3]}"


def check_files_exist():
    for name, spec in _courses():
        repo = os.path.dirname(HERE)
        gone = [s["path"] for s in spec["sources"] if not os.path.isfile(os.path.join(repo, s["path"]))]
        gone += [d["image"] for d in spec["diagrams"] if not os.path.isfile(os.path.join(HERE, name, d["image"]))]
        assert not gone, f"{name}: missing file(s) {gone}"


if __name__ == '__main__':
    checks = [check_every_course_parses, check_ids_are_namespaced, check_files_exist]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"PASS study-courses ({len(checks)} checks)")
