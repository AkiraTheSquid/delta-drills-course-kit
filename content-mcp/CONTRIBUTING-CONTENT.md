# Editing Delta Drills content with Claude Code

You do not need to know this repo. You need the editing password, a terminal,
and Claude Code. Everything below works on a fresh clone with nothing installed
but Python 3.9+.

## 1. Point Claude Code at the repo

```bash
git clone https://github.com/AkiraTheSquid/delta-drills-course-kit
cd delta-drills-course-kit
claude
```

That is the public course kit. Seth's private app repo works the same way.

The repo ships a `.mcp.json`, so Claude Code offers to enable the
`delta-drills-content` MCP server on first launch. Say yes. Check it took with
`/mcp` — the `delta-drills-content` tools should be listed.

If you would rather register it by hand, or you use a different MCP client:

```json
{ "mcpServers": {
    "delta-drills-content": { "command": "/abs/path/to/content-mcp/bin/dd-content-mcp" }
} }
```

## 2. Unlock editing

Reading is open to anyone. Writing needs a password, once per session.

- **Public course kit:** it ships no password. Your clone is yours, so set one
  first. `--password -` reads it from stdin, which keeps it out of shell history:
  `read -rs PW && printf '%s' "$PW" | ./content-mcp/bin/dd-content set-password --password - ; unset PW`.
- **Seth's private repo:** ask Seth for the shared password. It is shared out of
  band and never written into the repo. Only its salted digest lives in
  `content-mcp/auth.json`.

> Log in to the content server with the content password

or, in a terminal:

```bash
./content-mcp/bin/dd-content login --password '<the shared password>'
```

The session lasts 12 hours. For scripts and CI, set
`DELTA_DRILLS_CONTENT_PASSWORD` in the environment instead.

## 3. Ask for what you want

The point of the MCP is that you describe the change and Claude Code makes it
against the real files:

> Read the lesson on einops.merge-axes and tell me what it covers.
>
> The worked example on torch.dtype-astype never prints anything. Fix it so it
> shows the contrast it is teaching.
>
> Add three Solo drills to python.indexing — negative indices, and one that
> shows an out-of-range index raising.
>
> Add a concept node for einops.einsum after einops.merge-axes, then write its
> page.

Claude Code will use the tools below. You can also drive them yourself with
`./content-mcp/bin/dd-content <tool> --help`.

## The layers of content

There are two kinds of course. A **code course** (Practice) is pages and drills
the learner runs. A **Study course** is ideas, with free-answer, multiple-choice,
flashcard and image-occlusion questions (see "Study courses" below).

| Layer | What it is | Tools |
|---|---|---|
| **Courses** | A folder under `lessons/` and its units (rows in the registry) | `course_list`, `course_create`, `course_add_unit` |
| **Images** | Pictures in a course: PNG/JPEG/GIF/WebP, alt text required, no SVG | `image_add` (returns the markdown to paste), `image_list`, `image_remove` |
| **Pages** | One markdown file per concept, four rungs: Lesson → Faded → Solo → Integrated | `lesson_list`, `lesson_read`, `lesson_outline`, `lesson_edit`, `lesson_write`, `lesson_create` |
| **Graph** | `kc_registry.json` — the concepts, their prerequisites, and the teaching order | `graph_list`, `graph_read`, `graph_add_kc`, `graph_update_kc` |
| **Drills** | The question bank the app actually serves | `drill_search`, `drill_read`, `drill_add`, `drill_update`, `drill_tag`, `drill_retire` |
| **Atoms** | The mastery graph a correct answer moves. Every drill needs one (`atom_tag_missing` otherwise) | `atom_list`, `atom_add`, `drill_tag` (or `atoms=` on `drill_add`) |
| **Multiple choice** | A math page's problems, `kp-<slug>.problems.json`, every answer proven in SymPy | `problems_read`, `problems_add`, `problems_update` |
| **Study** | Study courses, in `study-courses/<course>/` | `study_*` (below) |

Plus `pipeline_*` to check your work and `backup_*` to undo it.

## Building a new code course, in order

1. `course_create`: the folder and its first unit (`kind: math` for a prose + multiple-choice course).
2. `graph_add_kc` for each concept, prerequisites first. It also writes the glossary entry and the answer clock.
3. `atom_add` for the course's mastery atom(s) (`root=true` for the first one).
4. `image_add` for any picture. Paste the markdown it returns.
5. `drill_add` with `atoms=[...]` for each drill. Starters are an empty `def` with
   `pass`, never `return None`.
6. `lesson_create` for each page, listing those drill ids. For a math page, use `problems_add` too.
7. `pipeline_check`, then `pipeline_watchers`.

## Study courses: free answer, multiple choice, image occlusion

Needs only `pydantic` and `Pillow` in the content python (`.venv`), no torch.

1. `study_course_create`: the id, the title and the sections (the colour-coded parts).
2. `study_source_add`: the text the course quotes. Use your own notes or openly
   licensed text, and state the license. The course kit is public.
3. `study_concept_add`, one per idea. Give it a `definition` and two or more
   `constraints` (things a right answer must say). Its `passages` are
   `{source, text}` with the text copied **exactly** from that source.
4. `study_question_add`. `free` questions are graded by a model against the
   `reference`. `mc` has three or more options, one of them correct. `card` is a
   typed flashcard with other accepted spellings in `answers`. Each concept needs
   a `content` question, and a conceptual one also needs an `evaluative`
   question (judge or explain a claim).
5. `study_diagram_add` for image occlusion: the picture, `labels`, and one pixel
   box `[x0, y0, x1, y1]` per label. It writes one card per box. Then run
   `study_diagram_preview` and **look at the PNG it returns**. It shows a
   coordinate grid and the numbered boxes. Fix any box that misses its label
   with `study_diagram_update`.
6. `study_check` until it says `ok`. It merges the course into a copy of the
   app's Study pack and runs the app's own rules on it.
7. `study_publish`: it is live on deltadrills.com within seconds (the server
   re-checks it against every course, private ones included). Republish after
   any change; `study_unpublish` takes it down; `study_publish_status` says
   what is live. It needs your publish key from Seth, stored once by YOU in a
   terminal (`read -rs K && printf '%s' "$K" | ./content-mcp/bin/dd-content
   set-publish-key --key -`), never pasted to the AI. In the private repo,
   `study_apply` + a deploy still works for a course that should ship in the
   image.

Every id is namespaced by the course (`<course>.<id>`), so a Study course can
only ever replace its own content in the shared pack. An alias that another
course already uses ("Nucleus") is dropped when the course is merged.

## 4. Check your work — this is not optional

```bash
./content-mcp/bin/dd-content pipeline_check
```

That runs the repo's real gate, in order: every code fence in every page is
executed, every faded solution is graded against its drill's test cases, the
pages are compiled, the Q-matrix is rebuilt, the bank is re-exported, and the
bank is audited for drills that can be gamed. It stops at the first failure and
hands you the output.

Then:

```bash
./content-mcp/bin/dd-content pipeline_watchers
```

Those carry two standing guards that `pipeline_check` does not: a drill may not
use a function that no lesson at or before its concept has taught, and a
function that appears in zero ARENA notebooks is attention spent on something no
learner needs.

**A change that has not passed both has not landed.**

## 5. If you break something

```bash
./content-mcp/bin/dd-content backup_status     # how old is the snapshot
./content-mcp/bin/dd-content backup_list       # what it holds
./content-mcp/bin/dd-content backup_restore --confirm
```

One snapshot of all content is kept, refreshed automatically before the first
write of any day (`backup_restore` deliberately never rotates it first, so
recovering cannot destroy the copy you are recovering from) — so it is up to about 24 hours old and always predates the
session you are in. Restore overwrites every content file with what the snapshot holds, and parks
the current tree at `.content-mcp/pre-restore.tar.gz` first, so a restore is
itself undoable. It does not DELETE — a file created after the snapshot is left
in place and named in the result, because deleting other people's new work
would be worse than leaving it.

To roll back one file rather than everything:

```bash
./content-mcp/bin/dd-content backup_restore --confirm \
  --paths 'Local_Deployed_Shared/lessons/tensors/kp-tensor-model.md'
```

Git is still the real history. The snapshot is for the case where you have made
a mess across several files and want out of it in one command.

## Things that will bite you

- **Read `lesson_authoring_guide` before writing a page.** The four-rung format
  is a contract the validator enforces, not a style preference. One concept per
  segment, exactly one worked example per segment, and every segment must stand
  alone because the learner sees one segment per page.
- **A code fence gets a Run button if and only if CI runs it.** Plain
  ` ```python ` is executed by the validator and rendered as a runnable cell;
  ` ```python no-run ` is illustrative and never executed. There is no third
  state.
- **Question ids are positional.** `drill_add` appends and mints the next id;
  `drill_retire` stops a drill being served. Never delete a row from a CSV — it
  renumbers every question below it and silently re-points every tag and every
  stored attempt.
- **Never hand-edit `questions.json`, `questions_structured.json`, or
  `lessons_structured.json`.** They are generated on every export and your edit
  will vanish. Change the source and re-run the pipeline.
- **Adding a KC takes more than the registry.** `graph_add_kc` writes the
  glossary entry and the answer clock with it. Its drills still need an atom
  (`atom_add` + `drill_tag`), and `pipeline_check` builds the unit's notebooks.
- **Only whitelisted fields work in a drill override.** `drill_update` refuses
  anything else and names the allowed set, because the exporter would ignore it
  silently and your edit would look like it worked.
- **`drill_update` merges.** Within one override file the exporter keeps only
  the last record per drill, so `drill_update` writes the drill's whole record
  with your fields changed, and nothing else is lost.

## Changing the password

```bash
./content-mcp/bin/dd-content set-password --password '<new one>' --current '<the old one>'
```

Rewrites the salted digest in `content-mcp/auth.json`. Changing it requires the
current password, so holding the CLI is not enough to seize the gate. Existing
sessions keep working until they expire; share the new password out of band.
