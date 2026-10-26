#!/usr/bin/env python3
"""Deep verification of the atom prereq graph — beginner lock state, deadlock
sweep, diagnostic-seed reachability. Run from This-Directory-Only/backend:
  .venv/bin/python ../../Local_Deployed_Shared/pipeline/verify_atom_graph.py
(The fast structural checks also run in audit_question_bank.py --gate.)"""
import json
import sys
from graphlib import TopologicalSorter, CycleError

sys.path.insert(0, ".")
from app import atom_engine
from app.questions import get_all_questions
from app.prioritization import question_is_unlocked
from app.adaptive import UserPracticeState

fail = 0

def check(name, ok, detail=""):
    global fail
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fail += 1

g = json.loads(open("app/data/concept_graphs/arena_drillable_v1.json").read())
edges = g["prerequisite_edges"]
node_ids = {c["id"] for c in g["concepts"]}
questions = get_all_questions()
tagged = {}
for q in questions:
    for t in (q.atom_tags or []):
        tagged.setdefault(t["atom_id"], []).append(q.id)

# 1. every tagged atom is a node
missing_nodes = sorted(set(tagged) - node_ids)
check("all tagged atoms are graph nodes", not missing_nodes, str(missing_nodes))

# 2. every tagged atom is gated or an intentional root
incoming = {e["dependent_id"] for e in edges}
roots = set(g.get("intentional_root_atoms", []))
unwired = sorted(a for a in tagged if a not in incoming and a not in roots)
check("all tagged atoms gated-or-intentional-root", not unwired, str(unwired))

# 3. every gating prereq of a tagged atom is trainable
bad = {a: sorted(p for p in atom_engine.prerequisites(a) if p not in tagged)
       for a in tagged}
bad = {a: v for a, v in bad.items() if v}
check("no untrainable gating prereqs on tagged atoms", not bad, str(bad))

# 4. gating graph is acyclic
ts = TopologicalSorter()
for e in edges:
    ts.add(e["dependent_id"], e["prerequisite_id"])
try:
    order = list(ts.static_order())
    check("prereq graph acyclic", True, f"{len(order)} atoms in topo order")
except CycleError as exc:
    check("prereq graph acyclic", False, str(exc))
    order = []

# 5. fresh beginner: advanced atoms locked, entry points open
state = UserPracticeState(user_id="__verify__")
state.self_reported_level = "beginner"
unlocked = [q for q in questions if question_is_unlocked(state, q)]
locked = [q for q in questions if not question_is_unlocked(state, q)]
locked_ids = {q.id for q in locked}
check("fresh beginner has servable questions", len(unlocked) > 50, f"{len(unlocked)} unlocked / {len(locked)} locked")
for qid in (459, 460, 461):  # BatchNorm broadcasting — Seth's incident
    check(f"q{qid} (BatchNorm broadcasting) locked for beginner", qid in locked_ids)
adv = [q.id for q in unlocked
       for t in (q.atom_tags or [])
       if t["atom_id"] in ("conv2d-module", "training-loop-skeleton", "cross-entropy-loss",
                            "nn-module-subclass", "batchnorm-running-stats-ema")]
check("flagship advanced atoms locked for beginner", not adv, f"unlocked advanced qids: {adv}")

# 6. no deadlock: mastering atoms in topo order unlocks every tagged question
MASTERED = atom_engine.belief_reading(0.95)
sweep = UserPracticeState(user_id="__verify_sweep__")
for a in order + sorted(tagged):  # pure roots (no edges) are trivially masterable
    if a not in sweep.atom_posteriors and atom_engine.atom_is_ready(sweep, a):
        sweep.atom_posteriors[a] = dict(MASTERED)
still_locked = [a for a in tagged if not atom_engine.atom_is_ready(sweep, a)]
check("no permanently-locked tagged atom (full-mastery sweep)", not still_locked, str(still_locked))

# 7a. unlocked entry set spans several topics (queue has somewhere to start)
entry_topics = sorted({q.topic for q in unlocked})
check("entry set spans >=3 topics", len(entry_topics) >= 3, str(entry_topics))

# 7b. every tagged atom mastered → every question unlocks (no gate keys on
# anything but atom mastery, so locked topics are reachable, not walled off)
strong = UserPracticeState(user_id="__verify_strong__")
strong.atom_posteriors = {a: dict(MASTERED) for a in tagged}
unlocked_strong = [q for q in questions if question_is_unlocked(strong, q)]
check("all tagged atoms mastered unlocks everything", len(unlocked_strong) == len(questions),
      f"{len(unlocked_strong)}/{len(questions)}")

print()
sys.exit(1 if fail else 0)
