"""Drills that must be solved with a particular tool, or without one.

Seth, 2026-10-05, on einops.pooling: the coding exercises should be "set up in
such a way that you essentially are required to utilize the einops in order to
train that particular concept. Otherwise, a learner wouldn't learn" it. A
`reshape` plus `.mean` passes every pooling case and practises nothing the
einops page teaches. The reverse holds for the plain-PyTorch twin of a concept:
an einops one-liner there skips the reshape it exists to drill.

`tool_requirements.json` (beside this file) lists the rules, one group per
concept:

    {"kc": "einops.pooling", "ids": [324, ...], "require": ["einops.reduce"]}
    {"kc": "torch.window-pooling", "ids": [...], "forbid": ["einops"],
     "require_method": ["reshape", "view"]}

`require_method` is ANY-of: the answer must call one of those tensor methods.

`apply(qid, test_cases, function_name)` appends a check to every case's
`assert_code`, the hook all four graders already run after the submitted call
(backend `code_runner`, the browser's `test-check.js`, `colab_grader.py`,
`validate_lessons.py`). So the rule needs no grader change, and it reaches the
learner as the failing case's message.

The check is STATIC: it reads the names the submitted function's bytecode
refers to (and those of helper functions it calls from the same module), then
resolves them against the function's globals. That accepts every honest
spelling (`einops.reduce(...)`, `from einops import reduce`, `import einops as
E`, an import inside the function) without running anything twice.

ONE module, two callers: `export_questions_json.py` imports it and
`backend/app/questions.py` loads it by path (the same trick `math_questions.py`
uses for `lessons/math_bank.py`), so the exported bank and the served bank
cannot disagree about what a drill requires.

Known limit (codex review 2026-10-05): a NAME counts, not a call, so
`_ = einops.reduce` beside a reshape answer passes; and a tool bound as a
default argument (`def solve(x, f=einops.reduce)`) is not seen at all. That takes deliberate
effort; the check exists to steer an honest learner onto the skill, not to
stop someone gaming it.
"""

from __future__ import annotations

import json
from pathlib import Path

RULES_PATH = Path(__file__).with_name("tool_requirements.json")

# Marks a case that already carries the check, so apply() is idempotent.
MARKER = "# dd-tool-requirement"

# Runs in the grader's assert namespace: the learner's globals plus `result`.
_HELPER = '''\
{marker}
import importlib as _dd_il
import types as _dd_types
def _dd_names(_fn, _seen_fns):
    _names = set()
    def _walk(_c):
        _names.update(_c.co_names)
        for _k in _c.co_consts:
            if isinstance(_k, _dd_types.CodeType):
                _walk(_k)
    _walk(_fn.__code__)
    for _n in list(_names):
        _v = _fn.__globals__.get(_n)
        if isinstance(_v, _dd_types.FunctionType) and _v not in _seen_fns \\
                and _v.__globals__ is _fn.__globals__:
            _seen_fns.add(_v)
            _names |= _dd_names(_v, _seen_fns)
    return _names
_dd_fn = {fn}
_dd_g = _dd_fn.__globals__
_dd_seen = _dd_names(_dd_fn, {{_dd_fn}})
def _dd_is_mod(_n, _mod):
    _v = _dd_g.get(_n)
    return _n == _mod or (isinstance(_v, _dd_types.ModuleType) and _v.__name__.split('.')[0] == _mod)
def _dd_calls(_mod, _attr):
    _target = getattr(_dd_il.import_module(_mod), _attr)
    if any(_dd_g.get(_n) is _target for _n in _dd_seen):
        return True
    return _attr in _dd_seen and any(_dd_is_mod(_n, _mod) for _n in _dd_seen)
def _dd_touches(_mod):
    for _n in _dd_seen:
        _v = _dd_g.get(_n)
        if _dd_is_mod(_n, _mod) or str(getattr(_v, '__module__', '') or '').split('.')[0] == _mod:
            return True
    return False
'''


def require_message(dotted: str) -> str:
    return (f"This drill practises {dotted}: your answer has to call {dotted}. "
            "Other code can produce the same numbers, but it trains a different skill.")


def method_message(methods: list[str]) -> str:
    spelled = " or ".join(f".{m}" for m in methods)
    return (f"This drill practises splitting an axis with {spelled}: your answer has to use it. "
            "A loop can produce the same numbers, but it trains a different skill.")


def forbid_message(module: str) -> str:
    return (f"This drill practises plain PyTorch: solve it without {module}. "
            f"The {module} version of this skill is its own lesson.")


def build_assert(rule: dict, function_name: str) -> str:
    """The assert_code text for one rule against `function_name`."""
    lines = [_HELPER.format(marker=MARKER, fn=function_name)]
    for dotted in rule.get("require", []):
        module, attr = dotted.rsplit(".", 1)
        lines.append(f"assert _dd_calls({module!r}, {attr!r}), {require_message(dotted)!r}")
    for module in rule.get("forbid", []):
        lines.append(f"assert not _dd_touches({module!r}), {forbid_message(module)!r}")
    if rule.get("require_method"):
        methods = list(rule["require_method"])
        lines.append(f"assert any(_m in _dd_seen for _m in {methods!r}), {method_message(methods)!r}")
    return "\n".join(lines) + "\n"


def load_rules(path: Path = RULES_PATH) -> dict[int, dict]:
    """qid -> rule. A qid in two groups is an authoring error."""
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id: dict[int, dict] = {}
    for group in data["groups"]:
        rule = {k: group[k] for k in ("require", "forbid", "require_method") if group.get(k)}
        if not rule:
            raise ValueError(f"tool_requirements: group {group.get('kc')} requires and forbids nothing")
        for qid in group["ids"]:
            if int(qid) in by_id:
                raise ValueError(f"tool_requirements: q{qid} is listed twice")
            by_id[int(qid)] = {**rule, "kc": group.get("kc")}
    return by_id


_RULES: dict[int, dict] | None = None


def apply(qid: int, test_cases: list[dict], function_name: str | None,
          rules: dict[int, dict] | None = None) -> list[dict]:
    """Return test_cases with the rule's check appended to each case's assert_code.

    `rules` defaults to tool_requirements.json, read once per process."""
    global _RULES
    if rules is None:
        if _RULES is None:
            _RULES = load_rules()
        rules = _RULES
    rule = rules.get(int(qid))
    if not rule or not test_cases or not function_name:
        return test_cases
    check = build_assert(rule, function_name)
    out = []
    for case in test_cases:
        prior = case.get("assert_code") or ""
        if MARKER in prior:
            out.append(case)
            continue
        joined = (prior.rstrip("\n") + "\n" + check) if prior.strip() else check
        out.append({**case, "assert_code": joined})
    return out
