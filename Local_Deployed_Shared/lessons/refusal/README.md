# refusal

## Purpose
- Generated KP pages for the refusal-direction branch of the ARENA graph.

## Owns
- `kp-*.md` pages for diff-in-means, pick-layer, direction-ablation, activation-addition, read-intervention-table.

## Does NOT own
- Their source: `scripts/refusal_course/` generates them. Hand edits are overwritten.

## Key Files
- `kp-diff-in-means.md` .. `kp-read-intervention-table.md`.

## Data & External Dependencies
- Registry rows in `../kc_registry.json`; compiled by `../compile_lessons.py`.

## How It Works (Flow)
1. `scripts/refusal_course/build.py --apply` renders pages.
2. Run `compile_lessons.py` after any page change.

## Invariants & Constraints
- Lines <= 54 columns, prose between code blocks, asserting blocks print.
- Every runnable cell asserts via `# Hidden checks`.

## Extension Points
- Change the source module, never the page.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **"teaching source changed" compile error** — `ACTIVE`
  - Symptom: compile refuses after a page edit.
  - Prevention/fix: run `compile_lessons.py`.

## Recent Changes
- 2026-10-17: Five pages generated.
