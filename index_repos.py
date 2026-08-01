"""
Clones the curated v1 repo list and builds one shared search index over all
of them via SearchEngine.build_index() -- matching OBSERVE's existing
single-shared-index design (see the scoping report: base_dir_filter scopes
a query to one repo within that shared index, it's not per-tenant
isolation, which is fine here since there's only one "tenant": this
product's own curated corpus).

Writes repo_manifest.json (repo name -> local clone path) that server.py's
`repo` query filter reads at request time.

Run once to build the index, and again (safe to re-run) to refresh it with
upstream changes -- each run does a fresh shallow clone, so it's always
indexing the latest default branch, not whatever was checked out last time.
"""
import json
import os
import shutil
import subprocess
import sys

from search_engine import SearchEngine

# Curated for v1: real, widely-used, substantial-but-not-astronomically-huge
# repos (skips things like the full Linux kernel, CPython, or PyTorch --
# those are orders of magnitude bigger and a v2 concern, not a "let's get
# something real shipped" v1 concern). Spans a few ecosystems on purpose,
# since agents searching code aren't all working in one language.
REPOS = {
    "react":    "https://github.com/facebook/react.git",
    "django":   "https://github.com/django/django.git",
    "flask":    "https://github.com/pallets/flask.git",
    "fastapi":  "https://github.com/tiangolo/fastapi.git",
    "express":  "https://github.com/expressjs/express.git",
    "redis":    "https://github.com/redis/redis.git",
    "numpy":    "https://github.com/numpy/numpy.git",
    "pandas":   "https://github.com/pandas-dev/pandas.git",
    "svelte":   "https://github.com/sveltejs/svelte.git",
    "vue":      "https://github.com/vuejs/core.git",
    "axios":    "https://github.com/axios/axios.git",
    "gin":      "https://github.com/gin-gonic/gin.git",
    "cargo":    "https://github.com/rust-lang/cargo.git",
    "tokio":    "https://github.com/tokio-rs/tokio.git",
    "laravel":  "https://github.com/laravel/laravel.git",
}

CLONE_ROOT = os.environ.get("OBSERVE_CLONE_ROOT", "./repos")
INDEX_DIR = os.environ.get("OBSERVE_INDEX_DIR", "/data/observe-index")
MODEL_PATH = os.environ.get("OBSERVE_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")


def clone_all() -> dict:
    os.makedirs(CLONE_ROOT, exist_ok=True)
    manifest = {}
    for name, url in REPOS.items():
        dest = os.path.abspath(os.path.join(CLONE_ROOT, name))
        if os.path.exists(dest):
            print(f"[clone] {name}: removing stale clone before re-cloning")
            shutil.rmtree(dest)
        print(f"[clone] {name} <- {url}")
        # --depth 1: only the latest commit -- we're indexing current code,
        # not history, and full history on repos like react/django/numpy
        # would be a large, pointless download for this purpose.
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, dest],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[clone] {name}: FAILED -- {result.stderr.strip()}", file=sys.stderr)
            continue
        manifest[name] = dest
    return manifest


def build_index(manifest: dict):
    engine = SearchEngine()
    done = {"flag": False}
    statuses = []

    def on_status(msg):
        statuses.append(msg)
        print(f"[index] {msg}")

    def on_done():
        done["flag"] = True

    engine.build_index(list(manifest.values()), INDEX_DIR, on_status, on_done)

    import time
    start = time.time()
    while not done["flag"]:
        if time.time() - start > 3600:
            raise TimeoutError(f"build_index did not finish within 1 hour -- last status: {statuses[-1] if statuses else '(none)'}")
        time.sleep(2)


def main():
    manifest = clone_all()
    if not manifest:
        print("[index_repos] no repos cloned successfully -- aborting", file=sys.stderr)
        sys.exit(1)

    with open("repo_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[index_repos] wrote repo_manifest.json with {len(manifest)} repos")

    build_index(manifest)
    print("[index_repos] done -- index ready at", INDEX_DIR)


if __name__ == "__main__":
    main()
