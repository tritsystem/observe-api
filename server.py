"""
OBSERVE hosted search API -- pay-per-query semantic code search over a
curated set of indexed open source repos. Built for AI agents to call
directly: an API key (Authorization: Bearer obs_...) gates every /v1/search
call, prepaid credits (via Stripe Checkout, see billing.py) fund usage,
fully automated end to end -- no human review in the signup/payment/search
loop.
"""
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, EmailStr

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

engine = SearchEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Blocking on purpose -- the process shouldn't accept traffic before the
    # model + index are actually loaded and ready to serve real results.
    status = engine.load_blocking(INDEX_DIR, MODEL_PATH)
    print(f"[startup] engine ready: {status}")
    yield


app = FastAPI(title="OBSERVE Search API", version="1.0.0", lifespan=lifespan)


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


class SearchResult(BaseModel):
    score: float
    path: str
    preview: str
    repo: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    credits_remaining: int


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
            SearchResult(score=r["score"], path=r["path"], preview=r["preview"], repo=req.repo)
            for r in raw_results
        ],
        credits_remaining=record["credits"],
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
