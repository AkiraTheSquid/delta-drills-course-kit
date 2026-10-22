# Study courses written through the content MCP

Each folder is one Study course (the Study tab: free-answer questions graded by
Jev, multiple choice, typed cards, image-occlusion labels), kept as a tracked
source set because the pack the app serves is generated and gitignored.

    <course>/course.json   course, sections, sources, concepts, passages, questions, diagrams
    <course>/sources/      files every passage is quoted from, VERBATIM
    <course>/images/       pictures of the image-occlusion cards (<= 1600 px)

Write them only through the MCP's `study_*` tools: they fill in the ids and
boilerplate the pack model needs, and `study_check` runs the pack's own rules.
In the private repo `study_apply` merges a course into the live pack
(re-runnable: rerun it after a pack builder rebuild, which drops contributor
courses). In the course kit, a passing `study_check` is the gate and the
folder goes in as a pull request.

🔴 The course kit is PUBLIC. A source here is republished: only text you may
share (your own notes, OpenStax / CC BY material), never a teacher's slides or
a copyrighted textbook. `study_source_add` makes you state the license.
