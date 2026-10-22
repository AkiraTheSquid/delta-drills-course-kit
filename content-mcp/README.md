# content-mcp

## Purpose
- Lets the course content be edited from OUTSIDE the app: a contributor points
  Claude Code at this repo, says "add a lesson on einops.einsum" or "fix the
  wrong example on q713", and the edit lands in the real source files under the
  same guards a hand edit would face.
- The domain concern is letting people who are not Seth author content without
  letting them break it — a password on writes, a snapshot to roll back to, and
  the repo's own validators run over every change.

## Owns
- The tool registry (`content_mcp/ops.py`) that defines what "editing the
  content" means: 28 operations over lessons, the concept graph, and drills.
- The two front ends over that registry: an MCP stdio server and the
  `dd-content` CLI.
- The write gate: the shared editing password and its session, and the single
  rolling content snapshot with its restore.

## Does NOT own
- The content itself — `Local_Deployed_Shared/lessons/`, the drill CSVs in
  `This-Directory-Only/csv files of problems/`, the override layers in
  `This-Directory-Only/chatgpt/`.
- Any validation rule. Every check shells out to the script the repo already
  gates deploys on (`scripts/validate_lessons.py`, `pipeline/audit_question_bank.py`,
  the folder `watch.py` guards). Nothing here reimplements a check, so nothing
  here can disagree with one.
- The KP markdown format — that is `lessons/AUTHORING.md`, parsed through the
  repo's own `scripts/lesson_lib.py`.
- Serving, grading, deploying. This is authoring-time only.

## Key Files
- `content_mcp/ops.py`: the registry. One `@op` per tool, declaring its summary,
  JSON-Schema params and whether it writes. **Both front ends are generated from
  it**, so a tool added here appears in the MCP and the CLI at once and the two
  cannot drift.
- `content_mcp/server.py`: MCP over stdio, hand-rolled and dependency-free so a
  contributor needs nothing but a Python 3.9+ interpreter.
- `content_mcp/cli.py`: `dd-content`. Subcommands and flags are derived from the
  same registry.
- `content_mcp/auth.py`: the password gate. Reads are open; every write goes
  through `require()`.
- `content_mcp/backup.py`: the one-day-old safety net.
- `content_mcp/lessons.py` / `graph.py` / `drills.py`: the three content layers.
- `content_mcp/courses.py` / `images.py` / `problems.py` / `glossary.py`: course
  rows + units, lesson pictures (`lessons/images/`), multiple-choice problems,
  the glossary kept in step with `graph_add_kc`.
- `content_mcp/atoms.py`: the mastery-atom graph (`arena_drillable_v1.json`) and
  drill tags (`question_atom_tags.jsonl`); every new drill needs a tag or the
  bank audit blocks it (`atom_tag_missing`).
- `content_mcp/clocks.py`: the per-KC placement answer clock
  (`lessons/placement_time_caps.json`), set by `graph_add_kc answer_secs`.
- `content_mcp/study.py` / `study_images.py` / `ops_study.py`: Study courses
  (free / mc / card / image-occlusion `label`) written to `study-courses/<id>/`.
  Anything needing pydantic or Pillow runs `scripts/study_course.py` under the
  content python, so this package stays stdlib-only.
- `content_mcp/pipeline.py`: runs the repo's validators under the backend venv.
- `bin/dd-content`, `bin/dd-content-mcp`: entry points. Both prefer the backend
  venv interpreter, because bare `python3` on this machine has no torch.
- `auth.json`: the salted PBKDF2 digest of the editing password. Committed on
  purpose — it is a digest, and a contributor cloning the repo needs the gate to
  already exist. The password itself is shared out of band.

## Data & External Dependencies
- Reads: `lessons/kc_registry.json`, `lessons/qmatrix_tags.json`, the KP
  markdown, `questions_structured.json`, the override JSONLs.
- Writes: KP markdown, `kc_registry.json`, `curated_additions.csv`,
  `curated_overrides.jsonl`, `pipeline/retired_question_ids.json`.
- Imports two repo modules rather than copying them: `scripts/lesson_lib.py`
  (the KP parser) and `pipeline/export_questions_json.py` (the authority on id
  assignment).
- Local state lives in `.content-mcp/` at the repo root — gitignored, holds the
  snapshot and the session token.

## How It Works (Flow)
1. A client (Claude Code, or a person at a terminal) calls a tool.
2. `ops.call` looks it up. If the op writes, it demands a live session and then
   calls `backup.ensure()` — which refreshes the snapshot only if the one on
   disk is at least 24h old.
3. The handler edits the real source file, refusing anything structurally wrong
   (a registry write that breaks teaching order, an override field outside the
   export whitelist, a page whose frontmatter names a different KC).
4. The caller runs `pipeline_check`, which is the repo's real gate: validate,
   compile, qmatrix, export, audit.
5. If it went wrong, `backup_restore` puts yesterday's content back.

## Invariants & Constraints
- 🔴 **stdout in `server.py` carries the JSON-RPC stream and nothing else.** One
  stray `print` and the client sees a parse error instead of a tool list. All
  diagnostics go to stderr.
- 🔴 **Question ids are POSITIONAL** — id N is the Nth CSV row across the
  export's sources in order. New drills are APPENDED to `curated_additions.csv`
  (the last source) and never inserted; a retired drill goes in
  `retired_question_ids.json` rather than having its row deleted, because
  deleting a row renumbers every question below it and silently re-points every
  qmatrix tag and every stored `served_question_id`.
- 🔴 **`questions.json` is generated.** Nothing here writes to it. Drill changes
  are override records; a hand edit to the artifact is erased by the next export.
- **The override-layer ORDER list is not duplicated here.** It already exists
  twice (exporter and `backend/app/questions.py`) and must stay in sync as an
  ordered sequence. `drills._override_layers_for` reports where an id is
  mentioned by scanning the directory, and leaves "which layer wins" to the
  exporter.
- **Registry order is a linear extension of the prereq lattice.** `graph.check`
  enforces it on every write, so a concept can never be sequenced before
  something it needs.
- **Reads never need the password; writes always do.** A tool that changes a
  file and is registered without `needs_auth=True` is a bug — `watch.py` checks
  for it.
- 🔴 **Within ONE override layer the exporter keeps the LAST record per id, no
  merge.** `drill_update` therefore writes the full merged record; a bare
  `{id, field}` once dropped a drill's test cases (watch.py guards it).
- 🔴 **Study ids are namespaced by course** (`<course>.<local>`, diagrams
  `<course>-<local>`). `study_apply` replaces only what the course owns in the
  shared pack and refuses a prefix that a builder already owns
  (`.contrib/applied.json`). A pack rebuild drops applied courses → re-run
  `study_apply`.
- 🔴 **`study_publish` sends only the files the course references** (each
  source's file, each diagram's picture), and the publish key lives in
  `.content-mcp/publish-key` (600), written by `set-publish-key --key -` from
  the contributor's own terminal — never an op argument, so never the AI's.
- The pipeline must run under `This-Directory-Only/backend/.venv/bin/python3`.
  Bare `python3` has no torch: every torch drill validates as broken and
  `expected_output` is left stale instead of recomputed.

## Extension Points
- New tool → one `@op(...)` in `ops.py` with a handler. It shows up in the MCP
  tool list and as a `dd-content` subcommand automatically; add nothing to
  `server.py` or `cli.py`.
- New content layer (say, the `notes/` metadata files) → a module beside
  `lessons.py`, then ops that call it.
- New gate to run after edits → an entry in `pipeline.STEPS` or `pipeline.AUDITS`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **The snapshot can capture a mess** — `ACTIVE (by design)`
  - When it happens: content is broken, nobody notices, and a write the next day
    rotates the snapshot over the last good copy.
  - Symptom: `backup_restore` restores the same breakage.
  - Root cause: Seth asked for exactly one backup about a day old, and one
    snapshot cannot be both "yesterday" and "the last known-good".
  - Prevention/fix: run `pipeline_check` before ending a session — a change that
    has not passed it has not landed. `backup_restore` also parks the current
    tree at `.content-mcp/pre-restore.tar.gz`, so a restore is itself undoable,
    and git remains the real history.
  - Status: `ACTIVE` — accepted trade for a backup an author can reason about.

- **A write that passes here can still fail the deploy** — `ACTIVE`
  - When it happens: a new drill uses a function no lesson at or before its
    concept teaches, or one that appears in zero ARENA notebooks.
  - Symptom: the prerequisite or ARENA-grounding ratchet fails later.
  - Root cause: those guards live in the folder watchers, not in the file write.
  - Prevention/fix: run `pipeline_watchers` after authoring, not just
    `pipeline_check`.
  - Status: `ACTIVE`.

## Recent Changes
- 2026-10-22: `study_publish` / `study_unpublish` / `study_publish_status` (`content_mcp/study_publish.py`, stdlib urllib) + CLI `set-publish-key`: a Study course goes LIVE on deltadrills.com (default `channel: stable`; `beta` for tests) through the backend's `/api/conceptual/tutor/contrib/*` — no PR, no deploy. 55 tools.
- 2026-10-21: Whole-course authoring tested end to end in a scratch kit. New: `course_create`/`course_add_unit`, `image_add/list/remove`, `problems_*` (MC), `atom_list`/`atom_add`/`drill_tag` (+ `drill_add atoms=`), `graph_add_kc answer_secs`, and 12 `study_*` tools for Study courses incl. image occlusion (preview PNG with grid + numbered boxes). `pipeline_check` order is now export → compile → colab → notebooks → qmatrix → symbols → validate → audit_bank. Fixed: `drill_update` wiped every other override field; `drill_add`/`drill_update` warn on a starter that does the work.
- 2026-10-07: Public course kit (`ops/course-kit/`). Interpreter order is now backend venv -> `DELTA_DRILLS_PYTHON` -> `<repo>/.venv` (the kit's venv) -> `python3`, in `paths.python_for_content()` and both `bin/` scripts. `set-password --password -` reads stdin. The kit ships NO `auth.json`, so its users set their own password. `CONTRIBUTING-CONTENT.md` covers both repos.
- 2026-09-02: Created. 28 tools over lessons/graph/drills/pipeline/backup, an
  MCP stdio server and the `dd-content` CLI generated from one registry,
  password-gated writes, and a single rolling ~24h content snapshot. Registered
  for Claude Code in the repo's `.mcp.json`.
