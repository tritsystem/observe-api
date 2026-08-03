#!/usr/bin/env python3
"""Concept Shakedown -- generalized, config-driven version of the ad-hoc
audit that found real gaps in observe-api's own codebase
(FINDING_concept_shakedown.md, root_cause_copilot's "use it on real data"
extension). Reusable: point it at ANY local codebase + ANY evidence
source (a SQLite database, on-disk artifacts, and/or the test suite) via
a config file -- nothing here is hardcoded to observe-api.

  python concept_shakedown.py --config shakedown_config.example.json

What stays generic vs. what needs a config per target system:
  - Codebase scanning + a local OBSERVE self-index: fully generic, works
    on any local directory, no git clone or company-specific setup
    needed.
  - Concept extraction: config-driven regex patterns. Different
    frameworks/languages declare things differently (FastAPI route
    decorators vs Flask's @app.route vs Express's app.get(...) vs a Go
    net/http mux) -- there's no one pattern that covers every codebase,
    so the config supplies the patterns. There is no way to auto-detect
    an arbitrary company's framework conventions; this boundary is
    disclosed; not silently assumed away.
  - Evidence checking: three generic, broadly-applicable mechanisms --
    (1) SQLite table/column match or "does this table have any rows at
    all" (a coarser, still-real signal when there's no per-concept
    marker column), (2) on-disk glob existence, (3) literal string
    search across the test suite (the most broadly reliable of the
    three, since most real test suites reference the concept by its
    literal name -- an endpoint path, an env var name). A company using
    CloudWatch/Datadog/a different database instead of SQLite would need
    to export or mirror relevant evidence into one of these three shapes
    first -- that boundary is disclosed too, not hidden behind a claim
    of universal compatibility.
"""
import argparse
import glob as globmod
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np


def log(msg):
    """Progress/status output goes to stderr, always -- real bug, caught
    live (again; the same class of bug root_cause_copilot.py's --json
    mode had): printing progress straight to stdout landed ahead of the
    final json.dumps() on the same stream, so --json wasn't actually
    clean, parseable JSON. Confirmed by actually piping a real run
    through json.load() and watching it fail."""
    print(msg, file=sys.stderr, flush=True)

DEFAULT_EXTS = {".py", ".js", ".ts", ".go", ".rb", ".java", ".rs", ".php",
                ".md", ".html", ".txt", ".json", ".yaml", ".yml"}
DEFAULT_SKIP = {"__pycache__", ".git", ".pytest_cache", ".venv", "venv",
                "node_modules", "dist", "build", "target"}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


# ---------------- Step 1: generic local self-index ----------------

def gather_files(repo_path, extra_skip_dirs, exts):
    skip = DEFAULT_SKIP | set(extra_skip_dirs or [])
    files = []
    for root, dirs, fnames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for fname in fnames:
            if Path(fname).suffix.lower() in exts:
                files.append(os.path.join(root, fname))
    return files


def build_self_index(repo_path, extra_skip_dirs, exts, out_dir):
    """Same shape as observe-api's own _build_self_index.py, generalized:
    any local repo_path, any skip-list/extensions, not this project's
    specific directories. Not using search_engine.SearchEngine.build_index()
    directly because its default skip-list is hardcoded and doesn't accept
    extra exclusions -- a real, separate gap noted in the observe-api repo
    itself, not re-solved here to keep this tool a single, portable file."""
    from sentence_transformers import SentenceTransformer

    files = gather_files(repo_path, extra_skip_dirs, exts)
    log(f"[shakedown] {len(files)} source files found under {repo_path}")

    chunks = []
    for fpath in files:
        rel = os.path.relpath(fpath, repo_path)
        try:
            text = open(fpath, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if len(text.strip()) < 50:
            continue
        for i in range(0, len(text), 700):
            chunk = text[i:i + 800]
            if len(chunk.strip()) > 50:
                chunks.append({"text": f"file:{rel}\n{chunk}", "rel_path": rel, "offset": i})

    log(f"[shakedown] {len(chunks):,} chunks from {len(files)} files")
    log("[shakedown] loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    vecs = []
    bs = 128
    for i in range(0, len(texts), bs):
        v = model.encode(texts[i:i + bs], normalize_embeddings=True, show_progress_bar=False)
        vecs.append(v)
    vectors = np.vstack(vecs).astype("float32") if vecs else np.zeros((0, 384), dtype="float32")

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "vectors_float32.npy"), vectors)

    path_index, path_table, chunk_rows = {}, [], []
    for c in chunks:
        key = c["rel_path"]
        if key not in path_index:
            path_index[key] = len(path_table)
            path_table.append({"base_dir": repo_path, "rel_path": key})
        chunk_rows.append([path_index[key], c["offset"]])
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"paths": path_table, "chunks": chunk_rows}, f)

    log(f"[shakedown] wrote {len(chunks):,} chunks, {len(path_table)} files -> {out_dir}")
    return files


# ---------------- Step 2: config-driven concept extraction ----------------

def extract_concepts(repo_path, files, concept_defs):
    """concept_defs: list of {"type": str, "pattern": regex, "name_group": int}.
    Scans real source files directly with the caller-supplied patterns --
    concept extraction needs exact text matching, not semantic search, and
    there's no framework-agnostic pattern that works for every codebase,
    so this is deliberately config-driven rather than hardcoded."""
    concepts = []
    seen = set()
    for fpath in files:
        try:
            text = open(fpath, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        rel = os.path.relpath(fpath, repo_path)
        for cdef in concept_defs:
            pattern = re.compile(cdef["pattern"])
            group = cdef.get("name_group", 1)
            for m in pattern.finditer(text):
                try:
                    name = m.group(group)
                except IndexError:
                    continue
                key = (cdef["type"], name)
                if key in seen:
                    continue
                seen.add(key)
                line = text[:m.start()].count("\n") + 1
                concepts.append({"type": cdef["type"], "name": name, "file": rel, "line": line})
    return concepts


# ---------------- Step 3: generic evidence checks ----------------

def check_sqlite_evidence(db_path, table, column=None, value=None, match_mode="substring"):
    """match_mode: "any_rows" (table has ANY rows at all -- a coarser
    signal for when there's no per-concept marker column), "exact", or
    "substring". Returns True/False/None -- None means "couldn't check"
    (missing db/table/column), which must never be silently treated the
    same as a real "no evidence found" (a missing table is a config
    problem, not proof the concept was never exercised)."""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        if match_mode == "any_rows":
            cur.execute(f"SELECT COUNT(*) FROM {table}")
        elif match_mode == "exact":
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (value,))
        else:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} LIKE ?", (f"%{value}%",))
        count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return None


def check_disk_glob_evidence(repo_path, glob_pattern, concept_name):
    pattern = glob_pattern.format(concept=concept_name)
    if not os.path.isabs(pattern):
        pattern = os.path.join(repo_path, pattern)
    return len(globmod.glob(pattern, recursive=True)) > 0


_file_grep_cache = {}  # (repo_path, glob) -> list of (abs_path, text) for every matched file


def check_test_grep_evidence(repo_path, test_glob, concept_name, exclude_file=None):
    """Real performance fix, caught before it became a problem rather than
    after: the original version re-globbed AND re-read every matching file
    from disk for EVERY concept, so a large codebase (hundreds/thousands of
    concepts against a test/log tree with hundreds/thousands of files)
    would redo the same I/O over and over -- O(concepts x files) file
    reads instead of O(files). Reading each matched file exactly once per
    glob pattern and reusing that for every concept checked against it
    turns this back into what it should be: read once, then just an
    in-memory substring check per concept.

    exclude_file: real correctness fix, caught on a live run against React
    -- unlike a concept declared once in one canonical file (e.g. a
    feature-flag file), something like `process.env.X` is read from many
    scattered files, so broadening an evidence glob to cover more of the
    codebase risks it accidentally including the concept's OWN declaring
    file, which would trivially "confirm" every single concept for a
    meaningless reason (it's present where it was extracted from, not
    because anything else references it). Excluding that one file keeps
    the check honest: evidence means "referenced somewhere ELSE."
    """
    key = (repo_path, test_glob)
    if key not in _file_grep_cache:
        matches = globmod.glob(os.path.join(repo_path, test_glob), recursive=True)
        entries = []
        for fpath in matches:
            try:
                entries.append((os.path.abspath(fpath), open(fpath, encoding="utf-8", errors="ignore").read()))
            except Exception:
                continue
        _file_grep_cache[key] = entries

    exclude_abs = os.path.abspath(os.path.join(repo_path, exclude_file)) if exclude_file else None
    for fpath, text in _file_grep_cache[key]:
        if exclude_abs and fpath == exclude_abs:
            continue
        if concept_name in text:
            return True
    return False


_jsonl_cache = {}  # (repo_path, glob) -> list of parsed dicts from every matched file


def check_jsonl_field_evidence(repo_path, glob_pattern, field, concept_name, match_mode="exact"):
    """Structured-log evidence: real gateway/access logs and most modern
    incident/audit logs are JSON Lines (one JSON object per line), not
    plain text -- checking a SPECIFIC FIELD (e.g. "path", "endpoint",
    "flag_key") is a much more precise signal than a raw substring search,
    which could false-positive on the concept name appearing incidentally
    in an unrelated field (a user-agent string, a request body). Silently
    skips lines that aren't valid JSON (a log file often has a header/
    footer line or two) rather than failing the whole check."""
    key = (repo_path, glob_pattern)
    if key not in _jsonl_cache:
        matches = globmod.glob(os.path.join(repo_path, glob_pattern), recursive=True)
        records = []
        for fpath in matches:
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
        _jsonl_cache[key] = records

    for record in _jsonl_cache[key]:
        value = record.get(field)
        if value is None:
            continue
        value = str(value)
        if match_mode == "exact" and value == concept_name:
            return True
        if match_mode == "substring" and concept_name in value:
            return True
    return False


def check_http_api_evidence(url_template, concept_name, headers=None, timeout=10,
                             success_field=None, success_value=None):
    """Real evidence adapter for a service reachable over HTTP -- the
    natural fit for a feature-flag service's own evaluation/exposure API
    (LaunchDarkly, Split, a custom internal flag service, etc.): does
    calling GET <url_template with {concept} substituted> show this flag
    was ever actually evaluated for a real user, not just declared in
    code? Returns None (not False) on any connection failure -- "the flag
    service is unreachable" must never be silently read as "no evidence
    this flag is used." success_field/success_value: optional JSON-
    response check (e.g. a response like {"evaluations": 42} counts as
    evidence only if "evaluations" > 0) -- without them, any 200 response
    counts as evidence, the honest default for a service whose exact
    response shape isn't known in advance.

    Real bug, caught by actually testing against a live local server
    rather than assuming: urllib.request.urlopen() RAISES HTTPError for
    any non-2xx status -- it never returns a response object with
    resp.status set to, say, 404, so a literal "if resp.status != 200"
    check inside the try block is unreachable dead code. A 404
    specifically is a meaningfully DIFFERENT, real signal from a
    connection failure -- "this flag doesn't exist in the service at
    all" (a genuine, confirmable negative) vs. "couldn't reach the
    service" (genuinely unknown) -- so they're handled as two separate
    cases below instead of collapsed into one broad except clause."""
    import urllib.error
    import urllib.request

    url = url_template.format(concept=concept_name)
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if success_field is None:
                return True
            try:
                body = json.loads(resp.read())
            except json.JSONDecodeError:
                return None
            value = body.get(success_field)
            if success_value is None:
                return bool(value)
            return value == success_value
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False  # confirmed: this concept isn't registered in the service at all
        return None  # any other HTTP error (5xx, auth failure, etc.) -- genuinely unknown
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None


def evaluate_concept(repo_path, concept, evidence_config):
    signals = []
    rules = evidence_config.get("rules", {}).get(concept["type"], [])
    any_true = False
    any_checked = False

    for rule in rules:
        kind = rule["kind"]
        if kind == "sqlite":
            db_path = rule["db"]
            if not os.path.isabs(db_path):
                db_path = os.path.join(repo_path, db_path)
            result = check_sqlite_evidence(
                db_path, rule["table"], rule.get("column"), concept["name"],
                rule.get("match_mode", "substring"),
            )
            signals.append({"kind": "sqlite", "table": rule["table"],
                             "match_mode": rule.get("match_mode", "substring"), "result": result})
        elif kind == "disk_glob":
            result = check_disk_glob_evidence(repo_path, rule["pattern"], concept["name"])
            signals.append({"kind": "disk_glob", "pattern": rule["pattern"], "result": result})
        elif kind in ("test_grep", "file_grep"):
            # Same mechanism (glob + literal substring search) under two
            # names: "test_grep" for a test suite, "file_grep" for any
            # other file set -- real server/application logs, a ledger
            # directory, whatever this target system actually has. Not
            # every target has a test suite (a SourceMod plugin doesn't),
            # but most have SOME real file-based trace worth grepping.
            result = check_test_grep_evidence(repo_path, rule["glob"], concept["name"],
                                               exclude_file=concept.get("file"))
            signals.append({"kind": kind, "glob": rule["glob"], "result": result})
        elif kind == "jsonl_field":
            glob_pattern = rule["glob"]
            abs_glob = glob_pattern if os.path.isabs(glob_pattern) else glob_pattern
            result = check_jsonl_field_evidence(
                repo_path, abs_glob, rule["field"], concept["name"], rule.get("match_mode", "exact")
            )
            signals.append({"kind": "jsonl_field", "glob": glob_pattern, "field": rule["field"], "result": result})
        elif kind == "http_api":
            result = check_http_api_evidence(
                rule["url_template"], concept["name"], headers=rule.get("headers"),
                timeout=rule.get("timeout", 10), success_field=rule.get("success_field"),
                success_value=rule.get("success_value"),
            )
            signals.append({"kind": "http_api", "url_template": rule["url_template"], "result": result})
        else:
            continue
        if result is not None:
            any_checked = True
            any_true = any_true or result

    if not rules:
        verdict = "UNOBSERVABLE (no evidence rule configured for this concept type)"
    elif not any_checked:
        verdict = "UNKNOWN (evidence source(s) unreachable/missing -- not the same as no evidence)"
    elif any_true:
        verdict = "CONFIRMED (real evidence found)"
    else:
        verdict = "NO EVIDENCE (checked, found nothing)"

    return {"verdict": verdict, "signals": signals}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--json", action="store_true", help="print the full structured report as JSON")
    ap.add_argument("--build-index", action="store_true",
                     help="also build a real OBSERVE embedding index of the codebase (for "
                          "semantic search afterward) -- OFF by default: concept extraction and "
                          "evidence-checking are plain regex/grep over the real files and never "
                          "consume the embeddings, so building them is real, avoidable cost "
                          "(minutes, not seconds, on a large codebase) unless you actually want "
                          "the codebase searchable afterward too.")
    args = ap.parse_args()

    config = load_config(args.config)
    repo_path = os.path.abspath(config["repo"]["path"])
    extra_skip = config["repo"].get("extra_skip_dirs", [])
    exts = set(config["repo"].get("extensions", DEFAULT_EXTS))

    if args.build_index:
        out_dir = config.get("index_out_dir", os.path.join(repo_path, ".shakedown-index"))
        files = build_self_index(repo_path, extra_skip, exts, out_dir)
    else:
        files = gather_files(repo_path, extra_skip, exts)
        log(f"[shakedown] {len(files)} source files found under {repo_path} "
              f"(skipping embedding index -- not needed for concept extraction; pass "
              f"--build-index to also build one)")

    concepts = extract_concepts(repo_path, files, config["concepts"])
    log(f"[shakedown] {len(concepts)} unique concepts extracted")

    results = []
    for concept in concepts:
        outcome = evaluate_concept(repo_path, concept, config["evidence"])
        results.append({**concept, **outcome})

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        by_type = {}
        for r in results:
            by_type.setdefault(r["type"], []).append(r)
        for ctype, items in by_type.items():
            print(f"\n=== {ctype} ({len(items)}) ===")
            for r in items:
                print(f"  {r['name']} ({r['file']}:{r['line']}) -> {r['verdict']}")


if __name__ == "__main__":
    main()
