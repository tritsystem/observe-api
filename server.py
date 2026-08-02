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
from search_engine import SearchEngine

INDEX_DIR = os.environ.get("OBSERVE_INDEX_DIR", "/data/observe-index")
MODEL_PATH = os.environ.get("OBSERVE_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
CREDITS_PER_SEARCH = int(os.environ.get("OBSERVE_CREDITS_PER_SEARCH", "1"))

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


class SignupRequest(BaseModel):
    email: EmailStr


class SignupResponse(BaseModel):
    api_key: str
    checkout_url: str
    note: str


@app.post("/v1/signup", response_model=SignupResponse)
def signup(req: SignupRequest):
    raw_key = db.create_api_key(req.email)
    checkout_url = billing.create_checkout_session(req.email, db.hash_key(raw_key))
    return SignupResponse(
        api_key=raw_key,
        checkout_url=checkout_url,
        note="Save this API key now -- it is only ever shown once and is not recoverable. "
             "Balance starts at 0 credits; complete checkout_url to fund it.",
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
