"""Pictures in a course: `lessons/<course>/images/<name>`.

The app already renders `![alt](src)` in a lesson (practice/lessons.js, with a
scheme whitelist), and since this module it renders one in the KG lesson panel,
the standalone viewer and a drill prompt too. What it never had was a way to
GET a picture into the content: an author could only hotlink.

`image_add` copies a local file (or base64 bytes) into the course's `images/`
folder and hands back the exact markdown to paste. The src is ROOT-ABSOLUTE
(`/lessons/<course>/images/x.png`) on purpose: the app has deep links
(`/practice/...`, `/course/...`), and a relative src resolves against the deep
path, where the SPA rewrite answers with index.html — a broken image that
looks like a 200. Vercel serves real files before rewrites, so the absolute
path reaches the file from any route.

🔴 No SVG. An SVG is a document that can carry script; served same-origin and
opened directly, it runs with the app's cookies. Raster formats cannot.
Bytes are sniffed, not trusted by extension, so a renamed file is refused too.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from . import paths

MAX_BYTES = 2 * 1024 * 1024
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
# extension -> (canonical extension, magic-byte test)
FORMATS = {
    "png": ("png", lambda b: b.startswith(b"\x89PNG\r\n\x1a\n")),
    "jpg": ("jpg", lambda b: b.startswith(b"\xff\xd8\xff")),
    "jpeg": ("jpg", lambda b: b.startswith(b"\xff\xd8\xff")),
    "gif": ("gif", lambda b: b[:6] in (b"GIF87a", b"GIF89a")),
    "webp": ("webp", lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP"),
}


def _sniff(data: bytes) -> str | None:
    for ext, (canonical, test) in FORMATS.items():
        if test(data):
            return canonical
    return None


def folder(course: str) -> Path:
    if course not in paths.courses():
        raise ValueError(f"Unknown course '{course}'. Known: {', '.join(paths.courses())}")
    return paths.LESSONS / course / "images"


def src_for(course: str, name: str) -> str:
    return f"/lessons/{course}/images/{name}"


def _slug(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:60] or "image"


def add(course: str, alt: str, source_path: str | None = None, data_base64: str | None = None,
        name: str | None = None, replace: bool = False) -> dict:
    """Copy one picture into the course and return the markdown that shows it."""
    if bool(source_path) == bool(data_base64):
        raise ValueError("Pass exactly one of source_path (a local file) or data_base64.")
    if not str(alt or "").strip():
        raise ValueError(
            "alt text is required — it is what a screen reader says and what shows if the "
            "picture fails to load. Describe what the picture shows."
        )
    if "]" in alt or "\n" in alt:
        raise ValueError("alt text may not contain ']' or a newline (it would end the markdown).")
    if source_path:
        src = Path(source_path).expanduser()
        if not src.is_file():
            raise ValueError(f"No file at {source_path}.")
        if src.stat().st_size > MAX_BYTES:
            raise ValueError(f"{src.name} is {src.stat().st_size} bytes; the limit is {MAX_BYTES}. Resize it first.")
        data = src.read_bytes()
        stem = src.stem
    else:
        try:
            data = base64.b64decode(data_base64, validate=True)
        except Exception as err:  # binascii.Error is not exported under a stable name
            raise ValueError(f"data_base64 is not valid base64: {err}") from err
        stem = "image"
    if len(data) > MAX_BYTES:
        raise ValueError(f"The picture is {len(data)} bytes; the limit is {MAX_BYTES}. Resize it first.")
    kind = _sniff(data)
    if not kind:
        raise ValueError(
            "Not a PNG, JPEG, GIF or WebP (checked by its bytes, not its name). "
            "SVG is refused on purpose — export it as PNG."
        )

    if name:
        base, dot, ext = name.lower().rpartition(".")
        if not dot or FORMATS.get(ext, (None,))[0] != kind:
            raise ValueError(f"name '{name}' must end in .{kind} — the file's bytes are {kind.upper()}.")
        final = f"{_slug(base)}.{kind}"
    else:
        final = f"{_slug(stem)}.{kind}"
    if not NAME_RE.fullmatch(final):
        raise ValueError(f"Unusable file name '{final}'.")

    target_dir = folder(course)
    target = target_dir / final
    if target.exists() and not replace:
        if target.read_bytes() == data:
            return _result(course, final, alt, len(data), kind, unchanged=True)
        raise ValueError(
            f"{target.relative_to(paths.REPO)} already exists with different bytes. "
            "Pass a different name, or replace=true to overwrite it."
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return _result(course, final, alt, len(data), kind)


def _result(course, name, alt, size, kind, unchanged=False) -> dict:
    src = src_for(course, name)
    return {
        "file": str((folder(course) / name).relative_to(paths.REPO)),
        "src": src,
        "markdown": f"![{alt.strip()}]({src})",
        "bytes": size,
        "format": kind,
        "unchanged": unchanged,
        "next_step": "Paste `markdown` on its own line in a page's prose (lesson_edit / lesson_write), "
                     "a drill's question_text, or a problem's prompt/solution.",
    }


def list_images(course: str | None = None) -> dict:
    courses = [course] if course else paths.courses()
    refs = referenced()
    out = []
    for name in courses:
        d = folder(name)
        if not d.is_dir():
            continue
        for path in sorted(d.iterdir()):
            if path.is_file():
                src = src_for(name, path.name)
                out.append({"course": name, "src": src, "bytes": path.stat().st_size,
                            "used_by": sorted(refs.get(src, []))})
    return {"images": out}


IMAGE_MD = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def referenced() -> dict[str, set[str]]:
    """src -> the KP files (and problem banks) that show it."""
    found: dict[str, set[str]] = {}
    sources = [p for p in sorted(paths.LESSONS.glob("*/kp-*")) if p.suffix in (".md", ".json")]
    # A drill prompt lives in its CSV row or an override record, not a page.
    sources += sorted(paths.CSV_DIR.glob("*.csv")) + sorted(paths.CHATGPT.glob("*.jsonl"))
    for path in sources:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for src in IMAGE_MD.findall(text):
            found.setdefault(src, set()).add(str(path.relative_to(paths.REPO)))
    return found


def remove(course: str, name: str) -> dict:
    # 🔴 a bare file name only: "../kp-x.md" would otherwise unlink a lesson page.
    if not NAME_RE.fullmatch(str(name or "")):
        raise ValueError(f"'{name}' is not an image file name (see image_list).")
    target = folder(course) / name
    if not target.is_file():
        raise ValueError(f"No image {src_for(course, name)}.")
    users = referenced().get(src_for(course, name))
    if users:
        raise ValueError(f"Still shown by {sorted(users)} — take it out of those first.")
    target.unlink()
    return {"removed": src_for(course, name)}
