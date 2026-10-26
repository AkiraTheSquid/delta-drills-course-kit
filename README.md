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

claude
```

Claude Code finds `.mcp.json` and offers to enable the `delta-drills-content`
server. Say yes, then check with `/mcp` that its tools are listed. There is
no password: reads and writes are open, and every write takes a snapshot
first (`backup_restore` rolls a bad session back).

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

### Publishing it live (no pull request, no waiting)

When `study_check` says `ok`, ask Claude to **publish** it (`study_publish`).
The app's server checks the course again against every course it has
(including private ones this repo cannot see), paints the image-occlusion
pictures, and serves it within seconds on deltadrills.com: Study tab, switch
the course on in Courses. Change anything, then publish again; that replaces
the live copy. `study_unpublish` takes it down. `public=true` shows it to
every signed-in learner (otherwise Seth and the learners he has invited to
Study).

Publishing needs no key or password. The first publish makes a random owner
id in `.content-mcp/publish-key`; only that clone can replace or remove the
course afterwards, so keep that file.

### Changing it later, forking someone else's, admins

- **Your own course:** edit it with the same tools and publish again, any
  time, from the clone that published it.
- **Fork any course:** `study_fork source=<course> new=my-copy` copies
  a course to a new id that is yours (`live=true` fetches a public course
  published on deltadrills.com). It is a separate course: publishing it never
  touches the original. Sources marked No-Derivatives must stay word for word;
  the fork tells you which.
- **Admins:** `study_whoami` shows your contributor id. Send it to Seth; once
  he makes it an admin you may edit, republish or unpublish any contributed
  course (the original owner stays on record).

🔴 Use your own notes or text you may share (OpenStax is CC BY), never a
teacher's slides or a copyrighted textbook. A published course's sources sit
on the server; if you also send a pull request, they become public here.

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
