"""Publish a Study course LIVE to Delta Drills — no pull request, no deploy.

`study_publish` sends `study-courses/<course>/` (course.json, sources/,
images/) to the app's server, which checks it against the REAL pack (private
courses included — the kit's `study_check` cannot see those), paints its
image-occlusion pictures, and serves it from the next request on
(backend app/concept_tutor/contrib.py). Republishing replaces the course;
`study_unpublish` takes it down. Only the contributor who first published a
course may replace or remove it.

No auth since 2026-10-26 (Seth): anyone may publish. The key sent in
X-Contrib-Key is only an OWNER ID, so nobody else can replace or remove your
course. The first publish makes a random one, unasked, in
`.content-mcp/publish-key` (mode 600; never committed — the whole
`.content-mcp/` is machine-local state). Lose it and you can no longer update
courses it published. Admins (ids Seth lists in the server's
admins.json; see `study_whoami`) may update or remove ANY contributed course. A named key from Seth (`dd-content set-publish-key`)
still works and shows his name for you instead of `anon-<hash>`.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import urllib.error
import urllib.request
from pathlib import Path

from . import paths, study

# The backends moved Fly → Oracle VM on 2026-10-25; the Fly apps are stopped,
# so a publish sent there would land on a volume nothing serves.
SERVERS = {"stable": "https://dd-api.163-192-116-252.sslip.io",
           "beta": "https://dd-api-beta.163-192-116-252.sslip.io"}
ROUTE = "/api/conceptual/tutor/contrib"
APP = {"stable": "https://deltadrills.com", "beta": "https://delta-drills-beta.vercel.app"}
TIMEOUT = 240   # the server rebuilds the whole Study pack before it answers


def _key_file() -> Path:
    return paths.state_dir() / "publish-key"


def write_key(key: str) -> Path:
    key = key.strip()
    if len(key) < 20:
        raise ValueError("That does not look like a publish key (20+ characters).")
    f = _key_file()
    f.write_text(key + "\n")
    f.chmod(0o600)
    return f


def _key() -> str:
    key = os.environ.get("DELTA_DRILLS_PUBLISH_KEY", "").strip()
    if not key and _key_file().exists():
        key = _key_file().read_text().strip()
    if not key:
        key = secrets.token_urlsafe(32)   # this clone's owner id, made once
        write_key(key)
    return key


def _server(channel: str) -> str:
    override = os.environ.get("DELTA_DRILLS_PUBLISH_URL", "").strip()
    if override:
        return override.rstrip("/")
    if channel not in SERVERS:
        raise ValueError(f"channel must be one of {sorted(SERVERS)}")
    return SERVERS[channel]


def _call(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "X-Contrib-Key": _key(), "Content-Type": "application/json", "User-Agent": "dd-content"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
            return json.loads(res.read().decode() or "{}")
    except urllib.error.HTTPError as err:
        try:
            detail = json.loads(err.read().decode()).get("detail")
        except Exception:
            detail = err.reason
        if err.code == 401:
            raise PermissionError(f"The server refused the owner id: {detail}") from None
        if err.code == 404:
            raise RuntimeError("That server does not have live publishing yet (it predates 2026-10-26).") from None
        raise ValueError(f"Server said {err.code}: {detail}") from None
    except urllib.error.URLError as err:
        raise RuntimeError(f"Could not reach {url}: {err.reason}") from None


def _files(course: str, spec: dict) -> dict[str, str]:
    """Only what the course REFERENCES: each source's file and each diagram's
    picture. Anything else in the folder (a README, an editor's backup) stays
    home, and so does anything a crafted path points OUTSIDE the folder (only
    `sources/<file>` / `images/<file>` that resolve inside it are read)."""
    home = study.folder(course).resolve()
    prefix = f"study-courses/{course}/"
    rels = [s["path"][len(prefix):] for s in spec.get("sources", []) if str(s.get("path", "")).startswith(prefix)]
    rels += [str(d["image"]) for d in spec.get("diagrams", []) if d.get("image")]
    out = {}
    for rel in rels:
        top, _, name = rel.partition("/")
        if top not in ("sources", "images") or not name or "/" in name or "\\" in name or name.startswith("."):
            continue
        f = (home / top / name)
        if f.is_symlink() or not f.is_file() or f.resolve().parent != home / top:
            continue
        out[rel] = base64.b64encode(f.read_bytes()).decode()
    return out


def publish(course: str, channel: str = "stable", public: bool | None = None) -> dict:
    spec = study.load(course)
    if public is not None and bool(public) != bool(spec["course"].get("public", False)):
        spec["course"]["public"] = bool(public)
        study.save(course, spec)
    result = _call("POST", _server(channel) + ROUTE + "/publish", {"spec": spec, "files": _files(course, spec)})
    if result.get("live"):
        who = ("every signed-in learner" if spec["course"].get("public")
               else "Seth and the learners he has invited to Study (publish with public=true for everyone)")
        where = (os.environ.get("DELTA_DRILLS_PUBLISH_URL", "").strip() or APP.get(channel, channel))
        result["next_step"] = (f"LIVE on {where} now — Study tab, course "
                               f"'{spec['course']['title']}' (Courses: switch it on). Visible to {who}. "
                               "Republish after any change; nothing else to do.")
    else:
        result["next_step"] = "Not published (the live course, if any, is unchanged). Fix the errors and publish again."
    return result


def export(course: str, channel: str = "stable") -> dict:
    """{spec, files} of a course published live: public ones for anyone, a
    private one for its owner or an admin. study_fork's live source."""
    study.folder(course)
    return _call("GET", _server(channel) + ROUTE + "/" + course + "/export")


def whoami(channel: str = "stable") -> dict:
    """The name the server knows this clone by (`anon-<hash>` unless Seth gave
    you a named key) and whether it is an admin. Safe to share: it is a hash,
    not the key. Send it to Seth to be made an admin."""
    me = _call("GET", _server(channel) + ROUTE + "/whoami/me")
    if not me.get("admin"):
        me["how_to_become_admin"] = ("Send this contributor id to Seth. Admins can edit or take "
                                     "down ANY contributed course, not only their own.")
    return me


def unpublish(course: str, channel: str = "stable") -> dict:
    study.folder(course)   # validates the id
    return _call("DELETE", _server(channel) + ROUTE + "/" + course)


def status(course: str, channel: str = "stable") -> dict:
    study.folder(course)
    return _call("GET", _server(channel) + ROUTE + "/" + course)
