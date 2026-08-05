"""
OBSERVE hosted search API -- pay-per-query semantic code search over a
curated set of indexed open source repos. Built for AI agents to call
directly: an API key (Authorization: Bearer obs_...) gates every /v1/search
call, prepaid credits (via Stripe Checkout, see billing.py) fund usage,
fully automated end to end -- no human review in the signup/payment/search
loop.
"""
import asyncio
import faulthandler
import os
import signal
from contextlib import asynccontextmanager
from typing import Optional

# Debug-only: SIGUSR1 dumps every thread's real Python stack trace to
# server.log. Added to actually end a real, repeated (4 different
# structural fixes tried, all failed) startup deadlock this session
# instead of continuing to infer what's stuck from CPU/IO/FD proxies --
# `kill -USR1 <pid>` on the next hang gets the real answer directly.
faulthandler.register(signal.SIGUSR1)

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, EmailStr

import a2a_adapter
import billing
import db
import rate_limit
import tenant_index
from search_engine import SearchEngine

INDEX_DIR = os.environ.get("OBSERVE_INDEX_DIR", "/data/observe-index")
MODEL_PATH = os.environ.get("OBSERVE_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
CREDITS_PER_SEARCH = int(os.environ.get("OBSERVE_CREDITS_PER_SEARCH", "1"))
# Indexing a private repo is a real, one-time compute cost (clone + embed
# every chunk, real minutes -- see this project's own corpus build), unlike
# a search's near-zero marginal cost. Priced separately, not folded into
# the per-search price. Not refunded on a failed index (e.g. bad git_url) --
# a disclosed v1 simplification, not an oversight.
CREDITS_PER_PRIVATE_INDEX = int(os.environ.get("OBSERVE_CREDITS_PER_PRIVATE_INDEX", "2000"))
# Free trial credits granted at signup, before any payment -- lets a new
# caller try a handful of real searches without hitting Stripe first. v1 has
# no email verification (see db.create_api_key), so this is exploitable via
# repeated signups; kept small deliberately for that reason, not an oversight.
SIGNUP_BONUS_CREDITS = int(os.environ.get("OBSERVE_SIGNUP_BONUS_CREDITS", "100"))

_LANDING_PAGE_PATH = os.path.join(os.path.dirname(__file__), "landing", "index.html")
_LEGAL_DIR = os.path.join(os.path.dirname(__file__), "legal")

engine = SearchEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Blocking on purpose -- the process shouldn't accept traffic before the
    # model + index are actually loaded and ready to serve real results.
    # Timeout configurable since load time scales with corpus size (759k
    # chunks now vs. whatever this was originally tuned against) and with
    # whatever else is competing for CPU/GPU on the host at startup time --
    # the 180s default undershot both of those in practice.
    #
    # Run via asyncio.to_thread, NOT called directly -- load_blocking()
    # itself spawns a second thread (SearchEngine.load()'s own
    # threading.Thread, shared with the GUI desktop app's non-blocking
    # use case) and busy-polls it with a synchronous time.sleep(0.2) loop.
    # Calling that directly from this coroutine froze uvicorn's asyncio
    # event loop for the entire load (a synchronous blocking call inside
    # an async function doesn't yield), and repeatedly, reproducibly
    # deadlocked real startups this session (40min-7.5hrs) even though the
    # identical load logic completed in ~60s every time when reproduced
    # standalone outside a running event loop. asyncio.to_thread moves the
    # whole blocking call to a worker thread so the event loop stays live
    # throughout -- root cause not fully isolated beyond "the frozen event
    # loop was a real, necessary ingredient," but this is also just
    # correct practice for blocking work inside an async function
    # regardless.
    load_timeout = int(os.environ.get("OBSERVE_LOAD_TIMEOUT_SECONDS", "900"))
    status = await asyncio.to_thread(engine.load_blocking, INDEX_DIR, MODEL_PATH, load_timeout)
    # flush=True: stdout is block-buffered (not line-buffered) when
    # redirected to a file, not a TTY -- without this, a real completion
    # print sat unflushed for the entire process lifetime, making a
    # genuinely-finished, genuinely-working server look permanently stuck
    # to anything grepping server.log for this line. Real, costly bug this
    # session: repeatedly killed servers that had actually already loaded
    # successfully and were serving real search results, based on a log
    # file that just hadn't received the bytes yet. PYTHONUNBUFFERED=1 on
    # the process env is the more robust fix (catches this for every
    # print, not just this one) -- see restart script -- flush=True here
    # too since relying only on an external env var for a status line this
    # important isn't enough on its own.
    print(f"[startup] engine ready: {status}", flush=True)
    yield


app = FastAPI(title="OBSERVE Search API", version="1.0.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page():
    # Read from disk on every request (not cached at import time) -- the
    # page is a handful of KB, and this lets it be edited/redeployed without
    # a server restart, same reasoning as _RepoRegistry.get() below.
    with open(_LANDING_PAGE_PATH) as f:
        return f.read()


# Stripe's account activation flow requires these three URLs to actually
# resolve (Terms of service, Privacy policy, Refund and return policy) --
# not placeholders. Same "read from disk every request" reasoning as the
# landing page above.
@app.get("/terms", response_class=HTMLResponse, include_in_schema=False)
def terms_page():
    with open(os.path.join(_LEGAL_DIR, "terms.html")) as f:
        return f.read()


@app.get("/privacy", response_class=HTMLResponse, include_in_schema=False)
def privacy_page():
    with open(os.path.join(_LEGAL_DIR, "privacy.html")) as f:
        return f.read()


@app.get("/refund-policy", response_class=HTMLResponse, include_in_schema=False)
def refund_policy_page():
    with open(os.path.join(_LEGAL_DIR, "refund-policy.html")) as f:
        return f.read()


# Required (non-conditionally) by Stripe's product feed spec: a resolvable
# image_link, >= 800x800px JPEG/PNG. See legal/product-image.png.
@app.get("/product-image.png", include_in_schema=False)
def product_image():
    return FileResponse(os.path.join(_LEGAL_DIR, "product-image.png"), media_type="image/png")


def _require_key(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header -- expected 'Bearer obs_...'")
    raw_key = authorization.removeprefix("Bearer ").strip()
    record = db.get_key_record(raw_key)
    if record is None:
        raise HTTPException(status_code=401, detail="invalid API key")
    return raw_key


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    repo: Optional[str] = None  # optional: scope to one indexed repo's base_dir
    # Reframes results as unverified candidates instead of ranked answers --
    # a semantic match is correlation with the query, not proof a given
    # location causes whatever behavior the agent is actually investigating
    # (root-causing a bug, explaining a regression, etc.). Template-only in
    # v1 (no LLM call, no added cost) -- see verification_hint below.
    investigate: bool = False


class SearchResult(BaseModel):
    score: float
    path: str
    preview: str
    repo: Optional[str] = None
    verification_hint: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    credits_remaining: int
    note: Optional[str] = None


@app.post("/v1/search", response_model=SearchResponse)
def search(req: SearchRequest, authorization: Optional[str] = Header(None)):
    raw_key = _require_key(authorization)

    # Rate limit BEFORE the credit deduct/search work below -- credits alone
    # bound cost, not request rate, so a key with a large balance could
    # otherwise blast it in a tight loop and starve other concurrent callers
    # on this process (see rate_limit.py).
    if not rate_limit.allow(raw_key):
        raise HTTPException(status_code=429, detail="rate limit exceeded -- slow down and retry shortly")

    if req.k < 1 or req.k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    # Validate the repo filter BEFORE spending a credit -- a typo'd repo
    # name shouldn't cost the caller anything.
    base_dir_filter = None
    if req.repo:
        repos = REPO_REGISTRY.get()
        base_dir_filter = repos.get(req.repo)
        if base_dir_filter is None:
            raise HTTPException(status_code=400, detail=f"unknown repo '{req.repo}' -- see /v1/repos for valid names")

    # Atomic pre-deduct (not deduct-after-success) is what actually
    # prevents a balance race under concurrent requests on the same key --
    # see db.deduct_credit's own comment. The tradeoff is refunding on a
    # search failure below, rather than losing atomicity to check-then-act.
    if not db.deduct_credit(raw_key, CREDITS_PER_SEARCH):
        raise HTTPException(status_code=402, detail="insufficient credits -- purchase more via /v1/signup")

    try:
        raw_results = engine.search(req.query, k=req.k, base_dir_filter=base_dir_filter)
    except Exception:
        db.deduct_credit(raw_key, -CREDITS_PER_SEARCH)  # refund -- don't charge for a failed search
        raise HTTPException(status_code=500, detail="search failed -- credit refunded, please retry")

    db.log_usage(raw_key, req.query, req.repo, len(raw_results))

    record = db.get_key_record(raw_key)
    return SearchResponse(
        results=[
            SearchResult(
                score=r["score"],
                path=r["path"],
                preview=r["preview"],
                repo=req.repo,
                verification_hint=_verification_hint(r["path"]) if req.investigate else None,
            )
            for r in raw_results
        ],
        credits_remaining=record["credits"],
        note=(
            "investigate mode: results below are unverified candidates ranked by semantic "
            "similarity to your query, not confirmed causes. Each has a suggested falsification "
            "test -- run it before treating a candidate as the actual cause."
        ) if req.investigate else None,
    )


def _verification_hint(path: str) -> str:
    return (
        f"Unverified candidate -- {path} matched semantically but hasn't been tested as the "
        f"actual cause. Before treating it as the cause: isolate {path} (stub it out, disable "
        f"it, or add instrumentation around it) and re-run the behavior you're investigating "
        f"to see whether it persists without this code path."
    )


@app.get("/v1/repos")
def list_repos():
    """Which repos are actually indexed and searchable right now -- an agent
    should call this before assuming a `repo` filter name is valid."""
    return {"repos": sorted(REPO_REGISTRY.get().keys())}


class PrivateIndexRequest(BaseModel):
    git_url: str  # https://github.com|gitlab.com|bitbucket.org/<owner>/<repo> -- see tenant_index.py


class PrivateIndexResponse(BaseModel):
    status: str
    credits_remaining: int


@app.post("/v1/private/index", response_model=PrivateIndexResponse)
def private_index(req: PrivateIndexRequest, authorization: Optional[str] = Header(None)):
    raw_key = _require_key(authorization)
    key_hash = db.hash_key(raw_key)

    try:
        tenant_index.validate_git_url(req.git_url)
    except tenant_index.InvalidGitUrl as e:
        raise HTTPException(status_code=400, detail=str(e))

    existing = tenant_index.get_status(key_hash)
    if existing.get("state") == "indexing":
        raise HTTPException(status_code=409, detail="already indexing a repo -- check /v1/private/status first")

    # Charged on start, not refunded on failure (see CREDITS_PER_PRIVATE_INDEX
    # comment) -- this is a real compute cost regardless of whether the
    # repo the caller pointed at turns out valid.
    if not db.deduct_credit(raw_key, CREDITS_PER_PRIVATE_INDEX):
        raise HTTPException(status_code=402, detail=f"insufficient credits -- indexing costs {CREDITS_PER_PRIVATE_INDEX}")

    try:
        tenant_index.manager.start_indexing(key_hash, req.git_url)
    except RuntimeError as e:
        db.deduct_credit(raw_key, -CREDITS_PER_PRIVATE_INDEX)  # refund -- didn't actually start
        raise HTTPException(status_code=409, detail=str(e))

    record = db.get_key_record(raw_key)
    return PrivateIndexResponse(status="indexing", credits_remaining=record["credits"])


@app.get("/v1/private/status")
def private_status(authorization: Optional[str] = Header(None)):
    raw_key = _require_key(authorization)
    return tenant_index.get_status(db.hash_key(raw_key))


class PrivateSearchRequest(BaseModel):
    query: str
    k: int = 10


@app.post("/v1/private/search", response_model=SearchResponse)
def private_search(req: PrivateSearchRequest, authorization: Optional[str] = Header(None)):
    raw_key = _require_key(authorization)
    key_hash = db.hash_key(raw_key)

    if not rate_limit.allow(raw_key):
        raise HTTPException(status_code=429, detail="rate limit exceeded -- slow down and retry shortly")
    if req.k < 1 or req.k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    # 404, not an empty 200 -- "no private index" and "your index has no
    # matches for this query" are different facts, and an agent deciding
    # whether to call /v1/private/index needs to be able to tell them apart.
    status = tenant_index.get_status(key_hash)
    if status.get("state") != "ready":
        raise HTTPException(status_code=404, detail=f"no ready private index for this key (state: {status.get('state', 'none')}) -- see /v1/private/index")

    if not db.deduct_credit(raw_key, CREDITS_PER_SEARCH):
        raise HTTPException(status_code=402, detail="insufficient credits -- purchase more via /v1/signup")

    try:
        raw_results = tenant_index.manager.search(key_hash, req.query, k=req.k)
    except Exception:
        db.deduct_credit(raw_key, -CREDITS_PER_SEARCH)
        raise HTTPException(status_code=500, detail="search failed -- credit refunded, please retry")

    # Real bug, caught live by actually running a private search and then
    # querying usage_log for it: this call was missing entirely, so a
    # successful, credit-charged private search left zero trace in the
    # table meant to record API usage -- undercounting real activity with
    # no error or warning anywhere. "__private__" (not a real shared-repo
    # name, so it can't collide with one) marks these rows as private-index
    # searches, distinguishable from shared-search rows in the same table
    # without a schema migration.
    db.log_usage(raw_key, req.query, "__private__", len(raw_results or []))

    record = db.get_key_record(raw_key)
    return SearchResponse(
        results=[
            SearchResult(score=r["score"], path=r["path"], preview=r["preview"], repo=None)
            for r in (raw_results or [])
        ],
        credits_remaining=record["credits"],
    )


class SignupRequest(BaseModel):
    email: EmailStr


class SignupResponse(BaseModel):
    api_key: str
    checkout_url: str
    note: str


@app.post("/v1/signup", response_model=SignupResponse)
def signup(req: SignupRequest):
    raw_key = db.create_api_key(req.email, initial_credits=SIGNUP_BONUS_CREDITS)
    checkout_url = billing.create_checkout_session(req.email, db.hash_key(raw_key))
    return SignupResponse(
        api_key=raw_key,
        checkout_url=checkout_url,
        note=f"Save this API key now -- it is only ever shown once and is not recoverable. "
             f"Balance starts at {SIGNUP_BONUS_CREDITS} free trial credits; complete checkout_url to add more.",
    )


@app.get("/v1/balance")
def balance(authorization: Optional[str] = Header(None)):
    raw_key = _require_key(authorization)
    record = db.get_key_record(raw_key)
    return {"credits": record["credits"]}


@app.post("/v1/webhook/stripe")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature")):
    payload = await request.body()
    billing.handle_webhook(payload, stripe_signature)
    return {"received": True}


class _RepoRegistry:
    """Lazily loaded map of repo name -> base_dir (the same string
    build_index() recorded as each chunk's base_dir), read from a JSON
    manifest written by the indexing script (see index_repos.py). Cached
    in memory, refreshed on each call to pick up a re-index without a
    server restart -- cheap since it's a tiny file."""
    def __init__(self, path="repo_manifest.json"):
        self.path = path

    def get(self) -> dict:
        import json
        if not os.path.exists(self.path):
            return {}
        with open(self.path) as f:
            return json.load(f)


REPO_REGISTRY = _RepoRegistry()

# Real A2A protocol support, not just a discovery manifest -- see
# a2a_adapter.py's module docstring for what's actually implemented (v1:
# message:send with a plain-text query only) vs. deferred.
a2a_adapter.register_a2a_routes(app, engine, db, rate_limit, _require_key, CREDITS_PER_SEARCH)
