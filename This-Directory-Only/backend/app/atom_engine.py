"""atom_engine.py — per-atom logistic knowledge tracing: THE mastery + gating signal.

Replaced `bkt_mastery.py` on 2026-10-23. Same job, same callers, same graph:
every concept-graph atom carries a belief about the learner, the belief moves
on graded evidence, encompassing edges pass FIRe credit down to simpler atoms,
and the prerequisite lattice gates what can be served. What changed is the
model behind the belief.

Why the swap
------------
An offline replay of every graded prod answer (270 answers, 18 learners,
2026-08-06..10-18; fitted on the first 60% by time, scored on the last 40%)
put the two models side by side on the same answers:

    model                     test Brier   AUC
    base rate                   0.222      0.50
    BKT per atom (this job)     0.302      0.48
    LKT per atom (this file)    0.214      0.64

BKT did worse than predicting the average for everyone, fitted or not; its
P(known) had no ranking signal on recent answers. The logistic engine keyed by
atom matched the per-KC engine (0.218) and is what this module runs.

Model
-----
One Gaussian `ability` posterior per atom, in logits, updated by the SAME
assumed-density filter as `logistic_engine` (Glicko-shaped, self-damping), with
the same fixed features for the item an answer was given on: difficulty, rung,
worked example, lesson page, and forgetting (1 − FSRS recall). Prerequisite and
encompassing mastery are NOT features here: they are KC-level graph terms, and
dropping them changed nothing in the replay (0.2142 either way).

  * `confidence` (how sure a question→atom tag is) and FIRe credit are
    TEMPERED observations: the likelihood is raised to a power w ∈ [0, 1], so
    w = 1 is a full answer, w = 0 is nothing, and in between the step and the
    variance reduction both shrink together. BKT soft-applied these by moving
    a fraction of the way to its posterior; tempering is the Bayesian version.
  * FIRe — a CORRECT answer on an advanced atom is a tempered success on each
    simpler atom it encompasses, at w = confidence × propagation_weight.
    Encompassing edges remain the only channel between atoms.
  * forgetting — the `recency` feature (FSRS recall of the concepts that
    exercise the atom), applied only by readers that ask for it. Gates never
    do: a cleared prerequisite is a milestone, and time away does not re-lock it.

Mastery readout
---------------
`mastery()` is the LOWER edge of the central ~68% credible band (z = 1, one
standard deviation below the mean, so ~84% of the posterior lies above it) on
P(correct) for an unaided (`solo`), median-difficulty item — the same quantity the ladder's
`logistic_engine.mastered()` tests, so there is ONE definition of "knows it"
across the app. An atom is mastered (and so unlocks its dependents) at
THRESHOLD = logistic_engine.PROMOTE_P = 0.55.

The scale lines up with the BKT number it replaces, which is why every 0-to-1
consumer (frontend included) keeps working with only the threshold changed: a
fresh atom reads ~0.11 (BKT: 0.10), and the migration is anchored so a BKT
atom at exactly the old 0.85 unlock bar reads exactly THRESHOLD.

Migration
---------
`migrate_bkt()` runs once per saved state on load: each BKT posterior L seeds
the atom's mean from BKT's own P(correct) = g + (1 − s − g)·L, at the
prior's full width and shifted so the old 0.85 bar lands exactly on THRESHOLD, and every atom BKT had at or above its 0.85 unlock bar is
recorded in `atom_legacy_unlocked` and stays mastered for good (Seth,
2026-10-23: nothing already unlocked re-locks at the switch).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from app import logistic_engine as E

ABILITY = E.ABILITY.name
THRESHOLD = E.PROMOTE_P      # mastered / unlocked at a lower bound >= this
Z = E.LADDER_Z               # z = 1: lower edge of the central ~68% band, the ladder's

# Self-reported level → prior ability mean, in logits. None/unknown keeps the
# engine's cold prior. Chosen to keep BKT's implied first-answer P(correct)
# (g + (1 − s − g)·p_init with BKT's g = 0.2, s = 0.1): beginner p_init 0.02 →
# 0.214, default 0.10 → 0.27 (= sigmoid(-1), the engine's own prior), strong
# 0.45 → 0.515. The prior positions the difficulty search; it is not evidence,
# so the gates read the same prior width either way and nothing unlocks from
# self-report alone (strong reads ~0.26 fresh, far under THRESHOLD).
PRIOR_MEAN_BY_LEVEL: Dict[str, float] = {
    "beginner": -1.30,
    "strong": 0.06,
}

# The BKT parameters the migration reads old posteriors with. Not a model:
# only `migrate_bkt` uses them, to turn a stored L back into the P(correct)
# BKT itself predicted from it.
_BKT_GUESS, _BKT_SLIP, _BKT_UNLOCK = 0.20, 0.10, 0.85


def _bkt_logit(L: float) -> float:
    p = _BKT_GUESS + (1.0 - _BKT_SLIP - _BKT_GUESS) * L
    return math.log(p / (1.0 - p))


# Anchor: an atom at exactly BKT's unlock bar must read exactly THRESHOLD, or
# atoms just under the old bar would unlock at the switch (unshifted, L = 0.84
# read 0.554) and atoms just over it would sit on the new bar by luck.
_SEED_SHIFT = _bkt_logit(_BKT_UNLOCK) - (
    math.log(THRESHOLD / (1.0 - THRESHOLD)) + Z * math.sqrt(E.ABILITY.prior_var)
)

GRAPH_PATH = (
    Path(__file__).resolve().parent
    / "data" / "concept_graphs" / "arena_drillable_v1.json"
)


# --- graph indexes ------------------------------------------------------------

# Atoms that are infrastructure / tooling / dataset / GPU / visualization noise,
# NOT real conceptual prerequisites. They have no exercises and must never gate
# learning. Skipped when computing an atom's gating prerequisites. (Demoted
# 2026-05-30 from the untrainable-prereq audit.)
NON_GATING_ATOMS = frozenset({
    "device-to-cuda", "cuda-availability-check", "plotly-imshow", "loss-landscape",
    "tqdm-progress-bar", "mnist-dataset", "imagenet-1k-labels", "wandb-watch-gradients",
    "imagenet-normalization", "torchvision-models-pretrained", "trainer-class-pattern",
    "cifar10-dataset",
    # 2026-07-08 orphan-atom fix: untrainable prereqs (zero bank questions) that
    # permanently locked their dependents; no trainable twin exists to rewrite to.
    "tensor-item-scalar", "singular-matrix-mask-trick", "unbroadcast-pattern",
    "validation-no-grad",
    # 2026-08-30, same failure and same remedy after the ARENA content cut. Every
    # question training these two went with the concepts that taught them, and a
    # gating prereq with no trainer locks its dependents forever: as-strided
    # windowing was the whole of numpy.sliding-windows / window-stencil (Tensor.
    # unfold appears in ZERO of the 458 ARENA notebooks — einops does this job in
    # the corpus), and inf-masking was carried only by retired triangle drills.
    # The two atoms whose concepts SURVIVED the cut are not here: q103 and q198
    # were kept in the bank instead, which is the better fix where it is available.
    "as-strided-windowing", "inf-masking",
})


@lru_cache(maxsize=1)
def _graph(graph_path: str = str(GRAPH_PATH)) -> dict:
    return json.loads(Path(graph_path).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _encompassing_index() -> Dict[str, Tuple[Tuple[str, float], ...]]:
    """advanced_atom -> ((simpler_atom, propagation_weight), ...), single hop.

    Edge orientation: prerequisite_id = simpler, dependent_id = advanced
    (mastering the advanced atom implies the simpler one).
    """
    idx: Dict[str, List[Tuple[str, float]]] = {}
    for e in _graph().get("prerequisite_edges", []):
        if not e.get("is_encompassing"):
            continue
        w = float(e.get("propagation_weight", 0.0))
        if w <= 0.0:
            continue
        idx.setdefault(e["dependent_id"], []).append((e["prerequisite_id"], w))
    return {k: tuple(v) for k, v in idx.items()}


def encompassed_by(atom_id: str) -> Tuple[Tuple[str, float], ...]:
    """Simpler atoms credited when `atom_id` (advanced) is answered correctly."""
    return _encompassing_index().get(atom_id, ())


@lru_cache(maxsize=1)
def _prereq_index() -> Dict[str, frozenset]:
    """dependent_atom -> frozenset(prerequisite_atoms), single hop, with
    NON_GATING_ATOMS stripped so they never gate."""
    idx: Dict[str, set] = {}
    for e in _graph().get("prerequisite_edges", []):
        if e["prerequisite_id"] in NON_GATING_ATOMS:
            continue
        idx.setdefault(e["dependent_id"], set()).add(e["prerequisite_id"])
    return {k: frozenset(v) for k, v in idx.items()}


def prerequisites(atom_id: str) -> frozenset:
    """The gating prerequisite atoms of `atom_id`. Empty for roots / unknown ids."""
    return _prereq_index().get(atom_id, frozenset())


@lru_cache(maxsize=1)
def all_atoms() -> Tuple[str, ...]:
    return tuple(c["id"] for c in _graph().get("concepts", []))


# --- state access -------------------------------------------------------------
#
# getattr everywhere: the watch runners and several readers hand in ladder-only
# or SimpleNamespace states with none of these fields, and an absent store is a
# learner who has never been scored, not an error.

def _store(state) -> Dict[str, dict]:
    store = getattr(state, "atom_posteriors", None)
    if not isinstance(store, dict):
        store = {}
        try:
            state.atom_posteriors = store
        except Exception:
            pass
    return store


def _legacy(state) -> frozenset:
    return frozenset(getattr(state, "atom_legacy_unlocked", None) or ())


def prior(level: Optional[str] = None) -> E.Posterior:
    """The never-scored belief for a learner of this self-reported level."""
    base = E.initial_posterior(E.ABILITY)
    mean = PRIOR_MEAN_BY_LEVEL.get(level or "", base.mean)
    return E.Posterior(mean=mean, var=base.var, n=0, last_seen=None)


def prior_p(level: Optional[str] = None) -> float:
    """P(correct) on a median solo item before any evidence, for this level."""
    return E.sigmoid(prior(level).mean)


def posterior(state, atom_id: str) -> E.Posterior:
    """The atom's stored belief, or the learner's prior if it was never moved."""
    raw = (getattr(state, "atom_posteriors", None) or {}).get(atom_id)
    post = E.Posterior.from_dict(raw)
    return post if post is not None else prior(getattr(state, "self_reported_level", None))


def posteriors_of(state, atoms) -> Dict[str, Dict[str, float]]:
    """{atom: {mean, var}} for the given atoms — the exact beliefs, for audit
    rows that must let history be replayed without re-running the model."""
    return {a: {"mean": round(p.mean, 6), "var": round(p.var, 6)}
            for a in atoms for p in (posterior(state, a),)}


def has_evidence(state, atom_id: str) -> bool:
    """True once anything (an answer, FIRe credit, or the BKT migration) has
    written a belief for this atom."""
    return atom_id in (getattr(state, "atom_posteriors", None) or {})


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _days_between(earlier: Optional[str], later: datetime) -> float:
    t = _parse(earlier)
    if later.tzinfo is None:  # stored stamps are UTC-aware; a naive `now` is read as UTC
        later = later.replace(tzinfo=timezone.utc)
    return max(0.0, (later - t).total_seconds() / 86400.0) if t else 0.0


# --- readout ------------------------------------------------------------------

def mastery_of(post: E.Posterior, recall: Optional[float] = None, days_idle: float = 0.0) -> float:
    """Lower credible bound on P(correct) for an unaided median item.

    `recall` (FSRS R of the concepts exercising the atom) adds the engine's
    forgetting term; `days_idle` widens the posterior the way the engine does
    between attempts. Both default to off, which is the gate's reading.
    """
    if days_idle > 0:
        post = E.inflate(post, E.ABILITY, days_idle)
    values = {ABILITY: 1.0}
    if recall is not None:
        values[E.RECENCY.name] = 1.0 - min(1.0, max(0.0, float(recall)))
    lo, _ = E.predict(values, {ABILITY: post}).interval(Z)
    return lo


def belief_reading(m: float, var: float = 0.05, last_seen: Optional[str] = None) -> dict:
    """A stored posterior (dict) whose gate reading `mastery_of` is `m` — for
    fixtures and tools that need "an atom at mastery m" without replaying
    answers. Narrow by default, as a well-practised atom would be."""
    m = min(0.999, max(0.001, float(m)))
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mastery_of(E.Posterior(mean=mid, var=var)) < m:
            lo = mid
        else:
            hi = mid
    return E.Posterior(mean=(lo + hi) / 2, var=var, n=1, last_seen=last_seen).to_dict()


def p_solve(state, atom_id: str) -> float:
    """Point estimate of P(correct) on an unaided median solo item (no
    credible bound, no floor) — for consumers that want a probability, not a
    gate reading (placement priors)."""
    return E.predict({ABILITY: 1.0}, {ABILITY: posterior(state, atom_id)}).p


def mastery(state, atom_id: str, *, recall: Optional[float] = None,
            now: Optional[datetime] = None, decay: bool = False) -> float:
    """The atom's mastery in [0, 1] (module docstring, "Mastery readout").

    `decay=False` (default) is the evidence-only reading every gate uses. An
    atom grandfathered by the BKT migration never reads below THRESHOLD there,
    so number-reading consumers (the frontend) agree with the gates.

    `decay=True` is for display: applies `recall` (if given) and widens the
    posterior by the days since it last moved, and does not floor.
    """
    post = posterior(state, atom_id)
    if not decay:
        m = mastery_of(post)
        return max(m, THRESHOLD) if atom_id in _legacy(state) else m
    now = now or datetime.now(timezone.utc)
    return mastery_of(post, recall=recall, days_idle=_days_between(post.last_seen, now))


def readouts(state) -> Dict[str, float]:
    """{atom: mastery} for every atom with a belief — the frontend's
    `atom_mastery` map (evidence-only, same number the gates read)."""
    return {a: round(mastery(state, a), 4) for a in (getattr(state, "atom_posteriors", None) or {})}


def last_seen(state) -> Dict[str, str]:
    """{atom: ISO timestamp of the last time its belief moved}."""
    return {a: raw["last_seen"] for a, raw in (getattr(state, "atom_posteriors", None) or {}).items()
            if isinstance(raw, dict) and raw.get("last_seen")}


def is_mastered(state, atom_id: str) -> bool:
    return atom_id in _legacy(state) or mastery(state, atom_id) >= THRESHOLD


def atom_is_ready(state, atom_id: str) -> bool:
    """READY-TO-LEARN gate for a teaching item targeting `atom_id`: every gating
    prerequisite is mastered. Roots are always ready; the atom itself is not
    required (non-circular)."""
    return all(is_mastered(state, p) for p in prerequisites(atom_id))


def item_is_unlocked(state, required_atom_ids: Iterable[str]) -> bool:
    """Unlock gate for a composite / ARENA item: EVERY atom it composes is
    mastered. No required atoms → unlocked (nothing to gate on)."""
    return all(is_mastered(state, a) for a in required_atom_ids)


def gate_sets(state) -> Tuple[List[str], List[str]]:
    """(ready_atoms, mastered_atoms) over every graph atom."""
    ready: List[str] = []
    mastered: List[str] = []
    for a in all_atoms():
        if is_mastered(state, a):
            mastered.append(a)
        if atom_is_ready(state, a):
            ready.append(a)
    return ready, mastered


# --- evidence -----------------------------------------------------------------

def item_values(*, difficulty_score: Optional[float] = None, stage: Optional[str] = None,
                example: bool = False, lesson_days: Optional[float] = None,
                recall: Optional[float] = None) -> Dict[str, float]:
    """The fixed-feature design entries for the item an answer was given on.
    Missing facts contribute 0, i.e. a median, unaided, solo, fresh item."""
    return {
        E.DIFFICULTY.name: E.difficulty_to_logits(difficulty_score),
        E.STAGE.name: E.stage_offset(stage),
        E.EXAMPLE.name: E.example_offset(example),
        E.LESSON.name: E.lesson_value(lesson_days),
        E.RECENCY.name: 0.0 if recall is None else 1.0 - min(1.0, max(0.0, float(recall))),
    }


def with_recall(values: Mapping[str, float], recall: Optional[float]) -> Dict[str, float]:
    """`values` with only the recency entry set from `recall`; the item's own
    difficulty/stage/example/lesson entries are kept as they are."""
    out = dict(values)
    if recall is not None:
        out[E.RECENCY.name] = item_values(recall=recall)[E.RECENCY.name]
    return out


def observe(post: E.Posterior, values: Mapping[str, float], correct: bool,
            weight: float = 1.0, *, days_elapsed: float = 0.0,
            timestamp: Optional[str] = None) -> E.Posterior:
    """One tempered ADF step on an atom's ability posterior.

    The likelihood is raised to `weight` ∈ [0, 1]. At weight 1 this is exactly
    `logistic_engine.update` for a single learned feature at x = 1:

        mu  += w·s²·(y − p) / (1 + w·v·s²)
        s²  -= w·s⁴·v       / (1 + w·v·s²)      v = p(1 − p)

    Inflation for `days_elapsed` comes first, as in `logistic_engine.step`.
    """
    w = min(1.0, max(0.0, float(weight)))
    if days_elapsed > 0:
        post = E.inflate(post, E.ABILITY, days_elapsed)
    if w <= 0.0:
        return post
    vals = dict(values)
    vals[ABILITY] = 1.0
    p = E.predict(vals, {ABILITY: post}).p
    v = p * (1.0 - p)
    s2 = max(post.var, 0.0)
    denom = 1.0 + w * v * s2
    mean = post.mean + w * s2 * ((1.0 if correct else 0.0) - p) / denom
    var = s2 - w * s2 * s2 * v / denom
    var = max(var, E.ABILITY.min_var)
    if E.ABILITY.max_var is not None:
        var = min(var, E.ABILITY.max_var)
    return E.Posterior(mean=mean, var=var, n=post.n + 1, last_seen=timestamp or post.last_seen)


def apply_attempt(state, atom_id: str, correct: bool, *,
                  values: Optional[Mapping[str, float]] = None,
                  confidence: float = 1.0,
                  repeats: int = 1,
                  now: Optional[datetime] = None) -> Dict[str, float]:
    """Fold one graded answer into `atom_id`, then FIRe-credit the simpler
    atoms it encompasses (correct answers only). Mutates the state's store;
    returns {atom: new mastery} for every atom it moved, for logging and the
    frontend.

    `values` are the item's fixed features (`item_values`); None is a median
    unaided solo item. `confidence` ∈ [0, 1] tempers the observation (a tag
    the q-matrix is unsure of says less); 0 is a no-op. FIRe credit is a
    tempered SUCCESS at w = confidence × propagation_weight on the simpler
    atom's own baseline (ability only): the simpler skill was exercised inside
    the advanced item, so the item's difficulty and rung are not its own.

    A failed advanced attempt credits nothing downward: it did not show the
    simpler skill was exercised.

    `repeats` folds the same outcome in that many times on the practiced atom
    (a multi-try MC answer with k wrong picks is k misses on one item); FIRe
    still fires once.
    """
    c = min(1.0, max(0.0, float(confidence)))
    changed: Dict[str, float] = {}
    if c <= 0.0:
        return changed
    now = now or datetime.now(timezone.utc)
    ts = now.isoformat()
    store = _store(state)

    post = posterior(state, atom_id)
    days = _days_between(post.last_seen, now)
    for _ in range(max(1, int(repeats))):
        post = observe(post, values or {}, correct, c, days_elapsed=days, timestamp=ts)
        days = 0.0
    store[atom_id] = post.to_dict()
    changed[atom_id] = mastery(state, atom_id)

    if correct:
        for simpler, w in encompassed_by(atom_id):
            sp = posterior(state, simpler)
            store[simpler] = observe(
                sp, {}, True, c * w,
                days_elapsed=_days_between(sp.last_seen, now), timestamp=ts,
            ).to_dict()
            changed[simpler] = mastery(state, simpler)
    return changed


# --- one-time migration from BKT ---------------------------------------------

def seed_from_bkt(L: float, last_seen: Optional[str] = None) -> E.Posterior:
    """The posterior a BKT mastery L maps to (see `migrate_bkt`). Also replays
    pre-switch `bkt_update` log rows into group-chart history."""
    L = min(1.0, max(0.0, float(L)))
    mean = max(-4.0, min(4.0, _bkt_logit(L) - _SEED_SHIFT))
    return E.Posterior(mean=mean, var=E.ABILITY.prior_var, n=0, last_seen=last_seen)


def bkt_unlocked(L: float) -> bool:
    """Whether BKT had this atom at or above its unlock bar — the migration's
    grandfather test, also used when history replays `bkt_update` rows."""
    return float(L) >= _BKT_UNLOCK


def migrate_bkt(data: dict) -> dict:
    """Seed `atom_posteriors` / `atom_legacy_unlocked` from a save's BKT fields.

    Runs on load (adaptive._load_user_state) and does nothing once the save has
    a non-empty `atom_posteriors` (or one with no BKT fields beside it). An
    empty store next to BKT data is a save some writer round-tripped before it
    was migrated (the offline twin), so it is seeded then. The BKT fields are left in the dict untouched, so a
    rollback to a BKT build reads the learner exactly as they were at the switch.

    Each L seeds mean = logit(g + (1 − s − g)·L) − _SEED_SHIFT (BKT's own
    predicted P(correct), anchored so L = 0.85 reads exactly THRESHOLD),
    clipped to ±4, at the prior's full width — BKT's number is a starting
    point, not a measurement the new engine should be confident in.
    """
    if not isinstance(data, dict) or data.get("atom_posteriors") or (
            "atom_posteriors" in data and not data.get("atom_mastery")):
        return data
    bkt = data.get("atom_mastery") or {}
    stamps = data.get("atom_last_ts") or {}
    posts: Dict[str, dict] = {}
    legacy: List[str] = []
    for atom, raw in bkt.items():
        try:
            L = min(1.0, max(0.0, float(raw)))
        except (TypeError, ValueError):
            continue
        posts[atom] = seed_from_bkt(L, stamps.get(atom)).to_dict()
        if bkt_unlocked(L):
            legacy.append(atom)
    data["atom_posteriors"] = posts
    data["atom_legacy_unlocked"] = sorted(legacy)
    return data
