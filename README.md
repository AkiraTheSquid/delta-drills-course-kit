# Delta Drills course kit

The course content behind [Delta Drills](https://deltadrills.com), an adaptive
practice app, and the MCP server that edits it. This includes the concept graph,
the lesson pages and the drills. Point Claude Code (or any MCP client) at this
repo and describe the change you want, for example "add a lesson on
einops.einsum", "fix the wrong example on q713" or "build me a course on
Bayesian statistics". The edit lands in the real source files, checked by the
same validators the app's own deploys use.

## Quick start

You need Python 3.9+, git and [Claude Code](https://claude.com/claude-code).

```bash
git clone https://github.com/AkiraTheSquid/delta-drills-course-kit
cd delta-drills-course-kit

# The validators run every lesson's code, so they need torch (CPU is enough).
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r requirements.txt

# Writes are password-gated so an AI can't edit content until you unlock it.
# This is your own clone, so pick any password:
read -rs -p "New editing password: " PW && printf '%s' "$PW" | ./content-mcp/bin/dd-content set-password --password - ; unset PW

claude
```

Claude Code finds `.mcp.json` and offers to enable the `delta-drills-content`
server. Say yes, then check with `/mcp` that its tools are listed. Unlock
writes for 12 hours from your own terminal, so the password never goes through
the chat:

```bash
read -rs -p "Editing password: " PW && printf '%s' "$PW" | ./content-mcp/bin/dd-content login --password - ; unset PW
```

The full guide (tools, the order to build a course in, and what to do when a
check fails) is in [`content-mcp/CONTRIBUTING-CONTENT.md`](content-mcp/CONTRIBUTING-CONTENT.md).
The authoring rules for a lesson page are in
[`Local_Deployed_Shared/lessons/AUTHORING.md`](Local_Deployed_Shared/lessons/AUTHORING.md).

## Study courses: free answer, multiple choice, image occlusion

A Study course is the other kind of course in the app: closed-book questions
on ideas rather than code. Answers are typed in your own words and graded by a
model, or they are multiple choice, typed flashcards, or **image occlusion**
(a picture with every label covered; the learner names the highlighted one).
It needs no torch, only `pydantic` and `Pillow`:

```bash
python3 -m venv .venv && .venv/bin/pip install pydantic Pillow
```

Then ask Claude, for example: "Make a Study course on cell organelles from my
notes in ~/bio-notes.md, with free-answer and multiple-choice questions, and an
image-occlusion card for each label on ~/cell-diagram.png." The `study_*` tools
keep the course in `study-courses/<course>/`. Every quote must be copied
exactly from a source you added, and every concept needs two or more things a
right answer must say. `study_check` runs the app's own rules on the course and
tells you what to fix. For image occlusion, `study_diagram_preview` draws the
picture with a grid and the numbered boxes, so you can see whether a box
covers its label.

🔴 This repo is public. A source is published with your course, so use your own
notes or openly licensed text (OpenStax is CC BY), never a teacher's slides or
a copyrighted textbook.

When `study_check` says `ok`, open a pull request with your
`study-courses/<course>/` folder. It is applied to the app and deployed from
there.

## Sending your work back

Open a pull request. A change has landed only when `pipeline_check` and
`pipeline_watchers` both pass. Run them before you push. An accepted pull
request is applied to the app's source and then published back here, so it
reaches both the app and this repo.

## What's in here

| Path | What it is |
|---|---|
| `content-mcp/` | The MCP server and its `dd-content` CLI. Both are generated from one tool registry. |
| `Local_Deployed_Shared/lessons/` | Lesson pages (`kp-*.md`) per course, the concept graph (`kc_registry.json`), compiled ARENA notebooks, and LeetCode problems |
| `Local_Deployed_Shared/pipeline/` | Exports and audits the drill bank |
| `This-Directory-Only/csv files of problems/`, `This-Directory-Only/chatgpt/` | Drill sources and their override layers. Question ids are positional, so append and never delete. |
| `study-packs/` | The app's published Study courses. Read-only here: `study_check` checks a new course against them. |
| `study-courses/` | Study courses written through the MCP (`study_*` tools), one folder each |
| `scripts/` | Validators, compilers and the ratchet audits the watchers run |

The odd folder names come from the app repo. The layout is kept the same so
content moves between the two without a translation step.

## Notes

- `questions.json`, `questions_structured.json` and `lessons_structured.json`
  are generated. Never hand-edit them, because the next export overwrites them.
- This repo is published from the app's private repository, and each
  publish replaces the tree. Your fork's history is unaffected.
- Third-party material is credited where it is used. ARENA notebooks are MIT
  licensed by their authors. If you hold rights to something here and want it
  removed, open an issue.
