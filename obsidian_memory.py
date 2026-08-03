"""Obsidian vault memory interface -- ties Concept Shakedown / Root Cause
Copilot / the Spikeling integrations into the user's REAL, existing
Obsidian vault (C:\\Users\\gbran\\OneDrive\\Documents\\Spikeling\\vault)
instead of inventing a separate, disconnected memory system for this
tool ecosystem.

This is not a new format -- it matches the vault's OWN established
conventions exactly (read from its README.md and Lessons/log-direct-
edits.md, not guessed): `Project Work/<YYYYMMDD_HHMMSS>_<slug>.md` with
`date`/`kind`/`status` frontmatter and a first-person past-tense `##
Output` section; `Lessons/<slug>.md` with `kind: lesson`/`scope:`/
`severity:` frontmatter, short and tagged. The vault's own README states
its purpose plainly: "Everything the assistant/agents do gets logged
here as markdown, and the pipeline reads it back so past work and past
mistakes inform new work" -- this module is the same idea, applied to
this session's tool ecosystem rather than the Tribe/Spikeling pipeline
that already writes here.

Three real capabilities, not a mock of them:
  log_project_work() -- writes a real ledger entry for a shakedown run,
    a Root Cause Copilot diagnosis, or any other direct-edit-style
    finding worth persisting past this session.
  log_lesson()       -- writes a real, reusable gotcha (e.g. a Spikeling
    DSL edge case found this session) into Lessons/, in the vault's own
    short/tagged/actionable style.
  search_vault()     -- real keyword search across Lessons/ + Project
    Work/ (grep-based, no embedding index -- the vault is markdown text,
    not a database), so a caller can check "did we already learn this"
    before re-deriving a finding by hand.
  get_active_lessons() -- parses Home.md's own curated "Active lessons"
    bullet list, the vault's highest-signal must-read set, rather than
    re-deriving importance from a full-text search.
"""
import datetime
import glob as globmod
import os
import re

DEFAULT_VAULT_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling\vault"


def _slugify(title, max_words=8):
    words = re.findall(r"[a-z0-9]+", title.lower())
    return "-".join(words[:max_words]) or "entry"


def log_project_work(title, project_tag, status, output_text,
                      kind="direct_edit", vault_root=DEFAULT_VAULT_ROOT, timestamp=None):
    """title: short human title (no project prefix -- added automatically).
    project_tag: e.g. "observe-api", matching the vault's existing
    "[tribe] ..." bracket-prefix convention on Project Work titles.
    status: e.g. "done_direct_awaiting_f5", "completed_verified" -- the
    vault's own honest status vocabulary (see README.md's Conventions).
    output_text: the first-person past-tense narrative -- what happened,
    root cause/what was built, how it was verified. Written as-is inside
    a fenced code block, matching every existing direct_edit entry's
    format exactly (see Lessons/log-direct-edits.md's example).

    Returns the written file's path."""
    ts = timestamp or datetime.datetime.now()
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    slug = _slugify(title)
    filename = f"{stamp}_{slug}.md"
    work_dir = os.path.join(vault_root, "Project Work")
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, filename)

    content = (
        f"---\n"
        f"date: {ts.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"kind: {kind}\n"
        f"status: {status}\n"
        f"---\n\n"
        f"# [{project_tag}] {title}\n\n"
        f"## Output\n"
        f"```\n{output_text.strip()}\n```\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def log_lesson(title, scope_tags, body_markdown, severity="process", vault_root=DEFAULT_VAULT_ROOT):
    """title: short lesson name (becomes the filename slug and the [[wiki-
    link]] name other notes/Home.md would reference). scope_tags: list of
    keyword strings (e.g. ["spikeling", "stdp", "dsl"]) matching the
    vault's `scope:` tagging convention for keyword search. severity: the
    vault's existing vocabulary is loose ("process", or a plain
    technical category) -- pass whatever fits, this doesn't enforce an
    enum since the vault itself doesn't. body_markdown: the lesson body
    (symptom, cause, fix, rule) -- kept to the vault's own "short and
    actionable" convention, not enforced here but worth the caller
    respecting.

    Returns (path, wikilink_name) -- wikilink_name is what Home.md's
    Active-lessons list would reference as [[wikilink_name]]."""
    slug = _slugify(title, max_words=10)
    lessons_dir = os.path.join(vault_root, "Lessons")
    os.makedirs(lessons_dir, exist_ok=True)
    path = os.path.join(lessons_dir, f"{slug}.md")

    content = (
        f"---\n"
        f"kind: lesson\n"
        f"scope: {', '.join(scope_tags)}\n"
        f"severity: {severity}\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"{body_markdown.strip()}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path, slug


def search_vault(query, vault_root=DEFAULT_VAULT_ROOT, folders=("Lessons", "Project Work"), max_results=10):
    """Real keyword search (case-insensitive substring, no embeddings --
    the vault is plain markdown, this is a grep) across the given
    folders. Returns [{"path", "title", "snippet"}], most-recently-
    modified first, capped at max_results -- mirrors the vault README's
    own stated purpose for search_vault(): 'did we do X before' recall,
    not semantic ranking."""
    query_lower = query.lower()
    matches = []
    for folder in folders:
        folder_path = os.path.join(vault_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for path in globmod.glob(os.path.join(folder_path, "*.md")):
            try:
                text = open(path, encoding="utf-8").read()
            except OSError:
                continue
            if query_lower not in text.lower():
                continue
            title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            title = title_match.group(1) if title_match else os.path.basename(path)
            idx = text.lower().find(query_lower)
            start = max(0, idx - 60)
            snippet = text[start:idx + len(query) + 60].replace("\n", " ").strip()
            matches.append({"path": path, "title": title, "snippet": snippet,
                             "mtime": os.path.getmtime(path)})

    matches.sort(key=lambda m: -m["mtime"])
    for m in matches:
        del m["mtime"]
    return matches[:max_results]


def get_active_lessons(vault_root=DEFAULT_VAULT_ROOT):
    """Parses Home.md's own curated '## \U0001f4a1 Active lessons' bullet
    list (each line `- [[wikilink]] -- summary`) -- the vault's own
    highest-signal must-read set, real parsing of the actual file rather
    than re-deriving importance from a full-text search. Returns
    [{"name": wikilink, "summary": text}]. Empty list (not an error) if
    Home.md is missing or the section isn't found -- a missing curated
    list shouldn't be conflated with "no lessons exist" (search_vault()
    still works independently)."""
    home_path = os.path.join(vault_root, "Home.md")
    if not os.path.exists(home_path):
        return []
    text = open(home_path, encoding="utf-8").read()
    section = re.search(r"Active lessons.*?\n(.*?)(?:\n##|\Z)", text, re.DOTALL)
    if not section:
        return []
    lessons = []
    for line in section.group(1).splitlines():
        # Real lines are Obsidian callout bullets: "> - [[name]] -- summary"
        # (found by testing against the actual file, not assumed) -- the
        # leading "> " blockquote marker must be stripped before the "-".
        m = re.match(r"\s*>?\s*-\s*\[\[([^\]]+)\]\]\s*(?:--|—|-)?\s*(.*)", line)
        if m:
            lessons.append({"name": m.group(1), "summary": m.group(2).strip()})
    return lessons
