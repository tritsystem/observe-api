"""
Private per-tenant repo indexing -- lets an API key index their own git
repo, isolated from the shared 15-repo corpus and from every other tenant.

Isolation model: every tenant's clone + index lives under a directory
named by key_hash (sha256 of their raw API key -- same hash db.py already
uses, never the raw key itself). Isolation is enforced by WHERE data lives
on disk, not just by an app-level filter a future bug could bypass -- a
caller can never even construct another tenant's path without already
knowing that tenant's raw API key (at which point they already have full
access to that account regardless of this feature).

Indexing runs in a background thread (real minutes for a real repo --
see this project's own 15-repo corpus build) with a status you poll, the
same async-status-callback shape SearchEngine itself already uses.

Reuses SearchEngine directly (one instance per tenant) rather than
duplicating its scan/chunk/embed logic -- the class was already written
generically enough (no global-singleton assumption) to support this
without modification, aside from the shared_model addition so N tenants
don't each load their own copy of the embedding model.

Security: git_url is restricted to https:// URLs on a small host
allowlist (github.com/gitlab.com/bitbucket.org). Accepting an arbitrary
server-side `git clone` target from user input is a real SSRF / local-file
risk otherwise (file://, ssh://, internal-network http:// targets, or a
string starting with "-" being parsed as a git option instead of a URL) --
not a hypothetical, a documented class of vulnerability. Self-hosted git
servers are out of scope for v1, disclosed not silently unsupported.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time

from search_engine import SearchEngine

PRIVATE_ROOT = os.environ.get("OBSERVE_PRIVATE_ROOT", "./private")
MAX_CACHED_ENGINES = int(os.environ.get("OBSERVE_PRIVATE_MAX_CACHED", "20"))
MODEL_PATH = os.environ.get("OBSERVE_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
CLONE_TIMEOUT_SECONDS = 300

ALLOWED_GIT_HOSTS = {"github.com", "gitlab.com", "bitbucket.org"}
_GIT_URL_RE = re.compile(
    r"^https://(?P<host>[a-zA-Z0-9.-]+)/[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+(?:\.git)?/?$"
)


class InvalidGitUrl(Exception):
    pass


def validate_git_url(git_url: str) -> str:
    """Raises InvalidGitUrl unless git_url is a plain https:// URL on the
    host allowlist, shaped like https://host/owner/repo[.git]. Rejects
    anything else outright -- including strings starting with '-' (which
    `_GIT_URL_RE` already can't match, since it requires the https://
    scheme literally, closing the git-option-injection angle too)."""
    m = _GIT_URL_RE.match(git_url.strip())
    if not m:
        raise InvalidGitUrl(
            "git_url must look like https://github.com/<owner>/<repo> "
            "(or gitlab.com / bitbucket.org) -- other schemes and hosts "
            "aren't supported in v1"
        )
    if m.group("host") not in ALLOWED_GIT_HOSTS:
        raise InvalidGitUrl(
            f"host '{m.group('host')}' isn't on the v1 allowlist "
            f"({', '.join(sorted(ALLOWED_GIT_HOSTS))})"
        )
    return git_url.strip()


def _tenant_dir(key_hash: str) -> str:
    # key_hash is already a hex sha256 digest (from db.hash_key) -- safe as
    # a path component as-is, but double-guard against path traversal
    # regardless of what ever calls this.
    safe = re.sub(r"[^a-fA-F0-9]", "", key_hash)
    return os.path.join(PRIVATE_ROOT, safe)


def _status_path(key_hash: str) -> str:
    return os.path.join(_tenant_dir(key_hash), "status.json")


def _write_status(key_hash: str, status: dict):
    # Real bug this fixes: writing directly to status.json (truncate then
    # write) leaves a window where a concurrent GET /v1/private/status can
    # read a half-written/empty file and 500 on JSONDecodeError -- caught
    # live during testing, not hypothetical. Write to a temp file in the
    # same directory, then os.replace() (atomic on both POSIX and Windows)
    # so a reader only ever sees the fully-old or fully-new content.
    d = _tenant_dir(key_hash)
    os.makedirs(d, exist_ok=True)
    final_path = _status_path(key_hash)
    tmp_path = final_path + f".tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(status, f)
    os.replace(tmp_path, final_path)


def get_status(key_hash: str) -> dict:
    path = _status_path(key_hash)
    if not os.path.exists(path):
        return {"state": "none"}
    try:
        return json.load(open(path))
    except json.JSONDecodeError:
        # Shouldn't happen anymore with the atomic write above, but a
        # transient read-during-replace on some filesystems is cheaper to
        # tolerate with one retry than to let surface as a 500.
        time.sleep(0.05)
        return json.load(open(path))


class TenantIndexManager:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self._shared_model = None  # lazy: only load once a private index is actually used
        self._engines = {}    # key_hash -> SearchEngine
        self._lru = []        # key_hash access order, most-recent last
        self._lock = threading.Lock()

    def _get_shared_model(self):
        if self._shared_model is None:
            from sentence_transformers import SentenceTransformer
            self._shared_model = SentenceTransformer(self.model_path)
        return self._shared_model

    def start_indexing(self, key_hash: str, git_url: str):
        git_url = validate_git_url(git_url)  # raises InvalidGitUrl -- let the caller 400 on it
        existing = get_status(key_hash)
        if existing.get("state") == "indexing":
            raise RuntimeError("already indexing -- check status before starting another")

        _write_status(key_hash, {"state": "indexing", "started_at": time.time(), "git_url": git_url})

        def _run():
            tenant_dir = _tenant_dir(key_hash)
            clone_dir = os.path.join(tenant_dir, "repo")
            index_dir = os.path.join(tenant_dir, "index")
            try:
                if os.path.exists(clone_dir):
                    shutil.rmtree(clone_dir)
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", git_url, clone_dir],
                    capture_output=True, text=True, timeout=CLONE_TIMEOUT_SECONDS,
                )
                if result.returncode != 0:
                    _write_status(key_hash, {"state": "error", "error": f"clone failed: {result.stderr.strip()[:500]}"})
                    return

                engine = SearchEngine(shared_model=self._get_shared_model())
                done = {"flag": False}

                def on_status(msg):
                    _write_status(key_hash, {"state": "indexing", "detail": msg})

                def on_done():
                    done["flag"] = True

                engine.build_index([clone_dir], index_dir, on_status, on_done,
                                    model_path=self.model_path, quantize=False)

                start = time.time()
                while not done["flag"]:
                    if time.time() - start > 3600:
                        _write_status(key_hash, {"state": "error", "error": "indexing timed out after 1 hour"})
                        return
                    time.sleep(1)

                with self._lock:
                    self._cache_engine(key_hash, engine)

                _write_status(key_hash, {
                    "state": "ready",
                    "chunks": len(engine.metadata),
                    "git_url": git_url,
                    "finished_at": time.time(),
                })
            except subprocess.TimeoutExpired:
                _write_status(key_hash, {"state": "error", "error": f"clone timed out after {CLONE_TIMEOUT_SECONDS}s"})
            except Exception as e:
                _write_status(key_hash, {"state": "error", "error": str(e)[:500]})

        threading.Thread(target=_run, daemon=True).start()

    def _cache_engine(self, key_hash, engine):
        self._engines[key_hash] = engine
        if key_hash in self._lru:
            self._lru.remove(key_hash)
        self._lru.append(key_hash)
        while len(self._lru) > MAX_CACHED_ENGINES:
            evict = self._lru.pop(0)
            self._engines.pop(evict, None)

    def _get_engine(self, key_hash: str):
        with self._lock:
            if key_hash in self._engines:
                self._lru.remove(key_hash)
                self._lru.append(key_hash)
                return self._engines[key_hash]

        status = get_status(key_hash)
        if status.get("state") != "ready":
            return None

        engine = SearchEngine(shared_model=self._get_shared_model())
        index_dir = os.path.join(_tenant_dir(key_hash), "index")
        done_statuses = []
        engine.load(index_dir, self.model_path, done_statuses.append)
        start = time.time()
        while not engine.ready:
            if time.time() - start > 60:
                return None
            time.sleep(0.1)

        with self._lock:
            self._cache_engine(key_hash, engine)
        return engine

    def search(self, key_hash: str, query: str, k: int = 10):
        """Returns None if this tenant has no ready private index (caller
        should surface that as a clear 404, not an empty-results 200 --
        those mean two different things to an API consumer)."""
        engine = self._get_engine(key_hash)
        if engine is None:
            return None
        return engine.search(query, k=k)


manager = TenantIndexManager()
