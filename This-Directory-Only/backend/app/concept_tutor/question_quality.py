"""Question-quality rules for contributed Study courses, partly judged by Jev (2026-10-26).

Seth, 2026-10-26: ai-cells-cancer shipped flashcards ("that's dumb as hell")
and questions answerable by paraphrasing one sentence of the paper. The fix is
not "remember to write better questions" but a check that refuses them, so
the rules live in ONE file, `question_rules.json`, and three gates load it:
`contrib_merge.check` (= the MCP/CLI `study_check` AND the server's publish)
and the course `watch.py`. A rule is either code (format, demand, expectation
count, lead-in) or a Jev noul asked once per question. Item design follows
AutoTutor (3-6 expectation propositions per question), Graesser & Person's
deep-reasoning questions, Kintsch's textbase vs situation model, and
Carpenter's moderate cues (the reading Seth pasted 2026-10-26).

Stdlib only: the public course kit ships this file next to contrib_merge and
has no backend, so the Jev call is plain urllib (same wire format as
grade/jev.py) and the backend settings are optional.

The key: TYPESAFE_API_KEY env, else the backend setting, else
This-Directory-Only/typesafe_difficulty/.env (local). With NO key the Jev rules
are skipped with a warning (a kit contributor's check), except under
`strict` (the server's publish), where a missing key is an ERROR. With a key,
a Jev failure is an ERROR: a rule that could not run never passes a question.
The first unreachable call opens a breaker so the rest fail at once instead of
each waiting out its own timeout.

Answers are cached by fingerprint (DD_QQ_CACHE, default ~/.cache/dd-question-
quality): only the raw p is stored, so a threshold change re-decides from the
cache without asking again.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from app.concept_tutor import question_shape

RULES_FILE = Path(__file__).with_name("question_rules.json")
URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"
_LOCAL_ENV = Path(__file__).resolve().parents[3] / "typesafe_difficulty" / ".env"
WORKERS = 8

# ask(state, questions) -> {name: p_true}; tests swap in a fake.
Ask = Callable[[dict, dict], dict]


_override: Optional[Ask] = None


def set_ask(fn: Optional[Ask]) -> None:
    """Swap the Jev call (tests use a scripted fake; None = the real one)."""
    global _override
    _override = fn


def rules() -> list[dict]:
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))["rules"]


def _settings():
    try:
        from app.config import settings
        return settings
    except Exception:  # the kit has no app.config
        return None


def api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    s = _settings()
    if not key and s is not None:
        key = (getattr(s, "typesafe_api_key", "") or "").strip()
    if not key and _LOCAL_ENV.exists():
        for line in _LOCAL_ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    return key


def _model() -> str:
    s = _settings()
    return os.environ.get("TYPESAFE_MODEL") or (getattr(s, "typesafe_model", "") if s else "") or DEFAULT_MODEL


def _post(url: str, key: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def http_ask(key: str) -> Ask:
    """Jev, falling back to Kev (settings.kev_base_url) like grade/jev.py."""
    model = _model()
    s = _settings()
    kev = (getattr(s, "kev_base_url", "") or "").rstrip("/") if s else ""

    def ask(state: dict, questions: dict) -> dict:
        payload = {"model": model, "state": state, "questions": questions}
        try:
            body = _post(URL, key, payload, 30)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if not kev:
                raise RuntimeError(f"Jev unavailable: {exc}") from exc
            body = _post(kev + "/v1/systemone", getattr(s, "kev_api_key", ""), payload, 60)
        answers = body.get("answers") or {}
        return {n: float(a["noul"]) for n, a in answers.items() if isinstance(a, dict) and "noul" in a}
    return ask


def _cache_dir() -> Path:
    return Path(os.environ.get("DD_QQ_CACHE") or Path.home() / ".cache" / "dd-question-quality")


def _cached(ask: Ask, model: str) -> Ask:
    def wrapped(state: dict, questions: dict) -> dict:
        blob = json.dumps({"m": model, "s": state, "q": questions}, sort_keys=True, ensure_ascii=False)
        f = _cache_dir() / (hashlib.sha1(blob.encode("utf-8")).hexdigest() + ".json")
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        got = ask(state, questions)
        if set(got) >= set(questions):  # never cache a partial answer
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(got), encoding="utf-8")
            except OSError:
                pass
        return got
    return wrapped


# --- code rules ---------------------------------------------------------------

def _fmt(q: dict) -> str:
    return q.get("format", "free")


def _code(rule: dict, q: dict) -> str:
    """Why `q` breaks a code rule ("" = it does not)."""
    check = rule["check"]
    if check == "format":
        return f"format {_fmt(q)!r}" if _fmt(q) in rule["refused"] else ""
    if check == "demand":
        return f"demand {q.get('demand')!r}" if q.get("demand") in rule["refused"] else ""
    if check == "expectations":
        n = len([e for e in q.get("expectations", []) if str(e).strip()])
        return f"{n} expectations" if not rule["min"] <= n <= rule["max"] else ""
    if check == "bare":
        return question_shape.bare(q.get("prompt", ""))
    raise ValueError(f"question_rules.json: rule {rule['id']} has unknown check {check!r}")


def _applies(rule: dict, q: dict) -> bool:
    return "formats" not in rule or _fmt(q) in rule["formats"]


# --- Jev rules ----------------------------------------------------------------

def _state(q: dict, spec: dict, passages: bool) -> dict:
    state = {"course_language": spec.get("course", {}).get("language") or "en",
             "question_format": _fmt(q), "demand": q.get("demand", ""),
             "prompt": q.get("prompt", ""), "reference_answer": q.get("reference", ""),
             "expectations": q.get("expectations", [])}
    if _fmt(q) == "mc":
        state["options"] = [{"text": o.get("text", ""), "correct": bool(o.get("correct"))} for o in q.get("options", [])]
    if passages:
        texts = {p["id"]: p["text"] for p in spec.get("passages", [])}
        concepts = {c["id"]: c for c in spec.get("concepts", [])}
        pids = [pid for t in q.get("targets", []) for pid in concepts.get(t, {}).get("passages", [])]
        # every passage of every target: a cut would judge textbase without the passage it paraphrases
        state["source_passages"] = [texts[p] for p in dict.fromkeys(pids) if p in texts]
    return state


def _noul(rule: dict) -> dict:
    return {"type": "noul", "instructions": rule["ask"], "criteria": {"true": rule["true"], "false": rule["false"]}}


def _fails(rule: dict, p: float) -> bool:
    return p >= rule["threshold"] if rule["fail_if"] == "true" else p < rule["threshold"]


def judge(spec: dict, ask: Optional[Ask] = None, only: Optional[set] = None,
          strict: bool = False) -> dict:
    """{"errors": [...], "warnings": [...], "scores": {qid: {rule: p}}} for the
    course's questions (`only` = these question ids). `strict` = no key is an
    error, not a warning (publish: the server must never skip the Jev rules)."""
    all_rules = rules()
    qs = [q for q in spec.get("questions", []) if only is None or q.get("id") in only]
    errors: list[str] = []
    for q in qs:
        for r in (r for r in all_rules if r["kind"] == "code" and _applies(r, q)):
            why = _code(r, q)
            if why:
                errors.append(f"question {q.get('id')}: [{r['id']}] {why}: {r['message']}")

    jev_rules = [r for r in all_rules if r["kind"] == "jev"]
    warnings: list[str] = []
    scores: dict[str, dict[str, float]] = {}
    # One Jev request per (question, passages-or-not) state; every rule names
    # its own judgement in its instructions (the positional trap, grade/jev.py).
    jobs = []
    for q in qs:
        for with_passages in (False, True):
            group = {r["id"]: _noul(r) for r in jev_rules if _applies(r, q) and bool(r.get("passages")) == with_passages}
            if group:
                jobs.append((q, _state(q, spec, with_passages), group))

    ask = ask or _override
    if ask is None and jobs:
        key = api_key()
        if not key:
            why = (f"Jev question-quality rules ({', '.join(r['id'] for r in jev_rules)}) did not run: "
                   "no TypeSafe key here.")
            if strict:
                return {"errors": errors + [why + " Publish refuses a course whose rules did not run."],
                        "scores": scores, "warnings": []}
            return {"errors": errors, "scores": scores, "warnings": [why + " The server runs them on publish."]}
        ask = _cached(http_ask(key), _model())

    down = threading.Event()  # breaker: once Jev is unreachable, skip the wait

    def run(job):
        q, state, group = job
        if down.is_set():
            return q, group, {}, "Jev unreachable earlier in this check"
        try:
            return q, group, ask(state, group), ""
        except Exception as exc:
            down.set()
            return q, group, {}, f"{type(exc).__name__}: {exc}"

    by_id = {r["id"]: r for r in jev_rules}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for q, group, got, err in pool.map(run, jobs):
            qid = q.get("id")
            for name in group:
                if name not in got:
                    errors.append(f"question {qid}: [{name}] Jev did not answer ({err or 'missing'}); "
                                  "a rule that did not run never passes a question")
                    continue
                p = got[name]
                if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
                    errors.append(f"question {qid}: [{name}] Jev answered {p!r}, not a probability; "
                                  "a rule that did not run never passes a question")
                    continue
                scores.setdefault(qid, {})[name] = round(p, 3)
                if _fails(by_id[name], p):
                    errors.append(f"question {qid}: [{name}] p={p:.2f}: {by_id[name]['message']}")
    return {"errors": errors, "warnings": warnings, "scores": scores}
