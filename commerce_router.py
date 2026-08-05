"""
commerce_router.py -- an ACP-compatible buyer/seller discovery and
routing layer.

Real, disclosed scope, established by reading the actual spec rather
than assuming: the Agentic Commerce Protocol
(github.com/agentic-commerce-protocol/agentic-commerce-protocol,
maintained by OpenAI + Stripe) defines the checkout/payment flow
(POST /checkout_sessions, etc.) between ONE buyer-agent and ONE
already-known merchant. Its own 2026-04-17 OpenAPI spec has no
seller_id/merchant_id field at all -- a session is scoped entirely by
the caller's Bearer token, and multi-seller discovery is explicitly
left to "the marketplace layer above this API." This module IS that
marketplace layer, not a reimplementation of ACP checkout itself.

What this does: a seller registers a real ACP `checkout_session_url`
(their own merchant endpoint) plus a listing feed (short text
descriptions of what they sell). A buyer-agent describes what it wants
in natural language. Matching reuses OBSERVE's own embedding model
(the same SentenceTransformer already resident in memory for code
search -- see `engine.model` in server.py -- not a second model load)
to rank listings against the buyer's intent, and returns each match's
seller `checkout_session_url` + the seller's own `item_id`. The
buyer's agent then calls that URL directly, following the real ACP
spec (POST .../checkout_sessions with an Item{id, name, unit_amount}),
to actually complete the purchase.

What this explicitly, permanently does NOT do: handle payment
credentials, proxy the checkout call, hold funds, or act as a payment
provider or merchant of record. It returns a pointer to a seller's own
real ACP endpoint, the same trust boundary a search engine or directory
has with a business it lists -- never a payment processor's.

Real, confirmed purchases (POST /v1/commerce/feedback with
outcome="purchased") also archive to the user's real Obsidian vault via
obsidian_memory.py -- the same vault interface already dogfooded for
this session's own work, applied here to commerce events instead of
dev-session narration. Scoped deliberately to confirmed purchases only,
not every search or every feedback call: that's the rare, real
ground-truth event worth a durable, human-readable archive entry, not
routine API traffic that would flood the vault. A vault-write failure
never breaks the actual feedback response -- archiving is a best-effort
side effect of a real event, not a dependency of the API's own
correctness (see commerce_feedback()'s try/except).
"""
import os
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import billing
import commerce_search_index
import commerce_spiking_memory
import obsidian_memory

CREDITS_PER_COMMERCE_SEARCH_DEFAULT = 1

# Spam/abuse bounds -- judgment calls, not measurements (no real
# marketplace traffic exists yet to size these against). Real gap this
# closes: registration/listing endpoints had no rate limiting at all
# (only /v1/commerce/search did), so a single key could otherwise spam
# unbounded sellers/listings with no cost or throttle.
MAX_LISTINGS_PER_CALL = 200
MAX_LISTINGS_PER_SELLER = 2000

# How much a listing's learned affinity heat can nudge its rank, relative
# to cosine similarity (which stays the primary signal -- see
# commerce_spiking_memory.py's module docstring for why this is additive,
# not a replacement). A judgment call, not a measurement: heat is
# normalized to roughly [0, 1] (membrane_potential / neuron threshold,
# capped) then scaled by this weight before adding to the [~0, 1] cosine
# score, so even a maximally "hot" listing can only outrank a
# meaningfully-more-relevant cosine match by this much.
MEMORY_BLEND_WEIGHT = 0.15


class SellerRegisterRequest(BaseModel):
    name: str
    checkout_session_url: str


class SellerRegisterResponse(BaseModel):
    seller_id: int


class ListingIn(BaseModel):
    item_id: str
    name: str
    description: str
    unit_amount: Optional[int] = None  # minor currency units, matches ACP's Item.unit_amount
    currency: str = "usd"
    category: Optional[str] = None


class ListingsAddRequest(BaseModel):
    listings: List[ListingIn]


class ListingsAddResponse(BaseModel):
    added: int


class CommerceSearchRequest(BaseModel):
    intent: Optional[str] = None
    max_price: Optional[int] = None  # minor currency units
    category: Optional[str] = None
    k: int = 10
    # References a saved commerce_buyer_agents row (see the dashboard/
    # import models below) -- when given, any of intent/max_price/category
    # left unset here falls back to that profile's saved default. intent
    # must come from one or the other; a call with neither is a 400.
    buyer_agent_id: Optional[int] = None


class CommerceMatch(BaseModel):
    score: float
    seller_id: int  # real bug found live: an independent agent had to GUESS this to call /v1/commerce/feedback, since only seller_name was ever returned here
    seller_name: str
    checkout_session_url: str
    item_id: str
    name: str
    description: str
    unit_amount: Optional[int]
    currency: str
    memory_boost: float = 0.0  # how much learned affinity (see commerce_spiking_memory.py) nudged this match's score -- 0.0 if never matched together with anything before for this buyer key
    match_id: str = ""  # real correlation id for the two-sided reputation system -- pass through to /v1/commerce/feedback and (if your checkout flow supports a client reference) to the seller, so their /v1/commerce/seller-feedback report links to the same real event


class CommerceSearchResponse(BaseModel):
    matches: List[CommerceMatch]
    credits_remaining: int
    note: str = (
        "Each match's checkout_session_url is the seller's own real ACP "
        "endpoint (POST .../checkout_sessions per the Agentic Commerce "
        "Protocol). Call it directly with the given item_id to complete "
        "a purchase -- this API never handles payment or credentials."
    )


_VALID_OUTCOMES = {"purchased", "not_purchased", "irrelevant"}

# A confirmed real purchase is stronger ground truth than merely
# appearing in a search result (see commerce_spiking_memory.py's
# DEFAULT_TOP_HIT_DRIVE=80.0, used for ordinary matches) -- reinforced
# with more drive so the learned network actually distinguishes "this
# got shown" from "this got bought." MUST stay strictly below the
# network's neuron threshold (DEFAULT_THRESHOLD=100.0) -- found by
# testing, not assumed: core/runtime/runtime.py's real _fire() resets
# membrane_potential to exactly 0.0 the instant a neuron crosses
# threshold, so a drive AT or ABOVE threshold makes the neuron fire and
# immediately erases the very "heat" this reinforcement is meant to
# create -- a stronger signal that self-defeats via the LIF fire/reset
# semantics, not a bug in this module. 95.0 leaves real headroom above
# the ordinary 80.0 drive while staying safely under the 100.0
# threshold. A judgment call, not a measurement (no real conversion
# data exists yet to tune this against).
CONFIRMED_PURCHASE_DRIVE = 95.0


class FeedbackRequest(BaseModel):
    seller_id: int
    item_id: str
    outcome: str  # "purchased" | "not_purchased" | "irrelevant"
    match_id: Optional[str] = None  # optional for backward compat with callers built before this existed


class FeedbackResponse(BaseModel):
    recorded: bool
    reinforced: bool
    note: str


_VALID_SELLER_OUTCOMES = {"fulfilled", "buyer_never_completed", "disputed"}


class SellerFeedbackRequest(BaseModel):
    match_id: str
    outcome: str  # "fulfilled" | "buyer_never_completed" | "disputed"
    rating: Optional[int] = None  # 1-5, optional


class SellerFeedbackResponse(BaseModel):
    recorded: bool
    note: str


class ReputationSummary(BaseModel):
    tier: str  # "new" | "trusted" | "verified"
    total_matches: int
    buyer_confirmed_purchases: int
    seller_confirmed_fulfillments: int
    disputes: int
    note: str = (
        "Self-reported by both sides, not independently audited -- OBSERVE "
        "never sees the actual checkout (see commerce_router.py's module "
        "docstring). A tier reflects agreement between two disconnected "
        "parties over real time, not a cryptographic guarantee."
    )


class VerifyMatchResponse(BaseModel):
    found: bool
    tier: Optional[str] = None
    buyer_confirmed_purchases: Optional[int] = None
    seller_confirmed_fulfillments: Optional[int] = None
    disputes: Optional[int] = None


class NetworkStats(BaseModel):
    total_agents: int
    verified_agents: int
    trusted_agents: int
    total_matches: int
    total_confirmed_transactions: int
    total_disputes: int
    note: str = (
        "Aggregate, anonymized -- no individual buyer key or seller "
        "identity is exposed here. See /v1/commerce/my-reputation for "
        "your own key's detail."
    )


class CheckoutSessionRequest(BaseModel):
    item_id: str
    email: str


class CheckoutSessionResponse(BaseModel):
    id: str
    status: str
    checkout_url: str
    api_key: str
    note: str


# OBSERVE's own real ACP catalog -- exactly one item, its own credit
# package, priced identically to /v1/signup's checkout (billing.py is the
# single source of truth for that price, not duplicated here). This is
# what a buyer-agent gets back from a commerce_search match if OBSERVE is
# itself registered as a seller in its own marketplace.
_OBSERVE_CATALOG = {
    "observe-credits": {
        "name": f"OBSERVE API credits ({billing.PACKAGE_CREDITS:,})",
        "unit_amount": billing.PACKAGE_PRICE_CENTS,
    },
}


# Reputation tier thresholds -- judgment calls, not measurements (no real
# transaction volume exists yet to tune these against). Disclosed here,
# not hidden: "verified" requires BOTH sides' agreement on a real
# transaction, not just the buyer's own claim -- that's the entire point
# of the seller-feedback half of this system existing at all.
TRUSTED_MIN_CONFIRMED = 3
VERIFIED_MIN_SELLER_CONFIRMED = 5
VERIFIED_MAX_DISPUTES = 0


def _compute_tier(buyer_confirmed: int, seller_confirmed: int, disputes: int) -> str:
    if disputes > VERIFIED_MAX_DISPUTES:
        return "new"  # a real dispute resets trust rather than averaging it away
    if seller_confirmed >= VERIFIED_MIN_SELLER_CONFIRMED:
        return "verified"
    if buyer_confirmed >= TRUSTED_MIN_CONFIRMED or seller_confirmed >= 1:
        return "trusted"
    return "new"


# ---------- Dashboard / universal setup: sellers-with-listings, buyer-agent
# configs, and bulk import. All of this is CONFIGURATION only -- it creates
# and lists rows in the tables above, nothing here runs an agent loop. An
# external caller (a hand-written script, a LangChain/CrewAI tool, the
# dashboard's own "test search" button) still has to make the actual
# /v1/commerce/search or /v1/commerce/feedback call itself.

class ListingOut(BaseModel):
    item_id: str
    name: str
    description: str
    unit_amount: Optional[int]
    currency: str
    category: Optional[str]


class SellerWithListings(BaseModel):
    seller_id: int
    name: str
    checkout_session_url: str
    listings: List[ListingOut]


class BuyerAgentIn(BaseModel):
    name: str
    default_intent: str
    max_price: Optional[int] = None
    category: Optional[str] = None


class BuyerAgentOut(BaseModel):
    id: int
    name: str
    default_intent: str
    max_price: Optional[int]
    category: Optional[str]


class DeleteResponse(BaseModel):
    deleted: bool


class SellerImport(BaseModel):
    name: str
    checkout_session_url: str
    listings: List[ListingIn] = []


class ImportRequest(BaseModel):
    sellers: List[SellerImport] = []
    buyer_agents: List[BuyerAgentIn] = []


class ImportResponse(BaseModel):
    sellers_created: int
    listings_created: int
    buyer_agents_created: int


def register_commerce_routes(app: FastAPI, engine, db, rate_limit, require_key_fn, credits_per_search: int = CREDITS_PER_COMMERCE_SEARCH_DEFAULT):
    # Per-buyer-key learned listing-affinity memory (real Spikeling STDP,
    # see commerce_spiking_memory.py) -- one ListingAffinityMemory per
    # key_hash, in-process cache for the life of this process, backed by
    # commerce_memory_weights so a restart doesn't discard real learned
    # affinity (see _get_memory/_save_memory below). membrane_potential/
    # heat itself is NOT persisted (short-term signal by design), only
    # the learned synapse weights.
    _key_memories: Dict[str, commerce_spiking_memory.ListingAffinityMemory] = {}

    def _get_memory(key_hash: str) -> commerce_spiking_memory.ListingAffinityMemory:
        if key_hash in _key_memories:
            return _key_memories[key_hash]
        memory = commerce_spiking_memory.ListingAffinityMemory()
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT src_item, dst_item, weight FROM commerce_memory_weights WHERE key_hash = ?",
                (key_hash,),
            ).fetchall()
        if rows:
            memory.load_rows([(r["src_item"], r["dst_item"], r["weight"]) for r in rows])
        _key_memories[key_hash] = memory
        return memory

    def _save_memory(key_hash: str, memory: commerce_spiking_memory.ListingAffinityMemory) -> None:
        rows = memory.to_rows()
        if not rows:
            return
        with db.get_conn() as conn:
            # UPSERT, not DELETE-then-reinsert-everything -- a real
            # efficiency fix found while checking this for scale: the
            # original version deleted and rewrote the WHOLE weight set
            # (up to MAX_TRACKED_LISTINGS^2 rows) on every single search
            # or purchase, real write amplification under concurrent
            # traffic. Same end state, no full-table churn.
            conn.executemany(
                "INSERT INTO commerce_memory_weights (key_hash, src_item, dst_item, weight) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key_hash, src_item, dst_item) DO UPDATE SET weight = excluded.weight",
                [(key_hash, src, dst, weight) for src, dst, weight in rows],
            )

    # FAISS-backed vector index, one per real db.DB_PATH (not a single
    # flat cache) -- see commerce_search_index.py's module docstring for
    # the real ~32.7s/search benchmark this replaces. Keyed by DB_PATH
    # rather than a single instance so each test's isolated temp DB gets
    # its own isolated index (matching how SQLite's own db.DB_PATH
    # monkeypatch already isolates tests) instead of leaking listing IDs
    # across tests that happen to share the same process.
    _commerce_indices: Dict[str, commerce_search_index.CommerceVectorIndex] = {}

    def _get_commerce_index(dim: int) -> commerce_search_index.CommerceVectorIndex:
        db_path = db.DB_PATH
        if db_path not in _commerce_indices:
            index_path = str(db_path) + ".commerce_index.faiss"
            _commerce_indices[db_path] = commerce_search_index.CommerceVectorIndex(dim, index_path)
        return _commerce_indices[db_path]

    def _create_seller(key_hash: str, name: str, checkout_session_url: str) -> int:
        """Shared by the single-seller route and /v1/commerce/import so
        both go through identical validation -- no second copy to drift."""
        if not checkout_session_url.startswith("https://"):
            raise HTTPException(status_code=400, detail="checkout_session_url must be https -- ACP checkout sessions carry payment intent, never serve this over plain http")
        if not name.strip():
            raise HTTPException(status_code=400, detail="name must not be empty")
        with db.get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO commerce_sellers (key_hash, name, checkout_session_url, created_at) VALUES (?, ?, ?, ?)",
                (key_hash, name, checkout_session_url, time.time()),
            )
            return cur.lastrowid

    def _create_listings(key_hash: str, seller_id: int, listings: List[ListingIn]) -> int:
        """Shared by the single-seller route and /v1/commerce/import."""
        if not listings:
            raise HTTPException(status_code=400, detail="listings must not be empty")
        if len(listings) > MAX_LISTINGS_PER_CALL:
            raise HTTPException(status_code=400, detail=f"at most {MAX_LISTINGS_PER_CALL} listings per call")
        for listing in listings:
            if listing.unit_amount is not None and listing.unit_amount <= 0:
                raise HTTPException(status_code=400, detail=f"unit_amount must be positive if set (item_id={listing.item_id})")
            if not listing.item_id.strip() or not listing.name.strip():
                raise HTTPException(status_code=400, detail="item_id and name must not be empty")

        with db.get_conn() as conn:
            seller = conn.execute(
                "SELECT * FROM commerce_sellers WHERE id = ? AND key_hash = ?", (seller_id, key_hash)
            ).fetchone()
        if seller is None:
            raise HTTPException(status_code=404, detail="no seller with that id owned by this API key")
        if not engine.ready or engine.model is None:
            raise HTTPException(status_code=503, detail="search engine not ready yet")

        with db.get_conn() as conn:
            existing_count = conn.execute(
                "SELECT COUNT(*) AS n FROM commerce_listings WHERE seller_id = ?", (seller_id,)
            ).fetchone()["n"]
        if existing_count + len(listings) > MAX_LISTINGS_PER_SELLER:
            raise HTTPException(
                status_code=400,
                detail=f"this would exceed the {MAX_LISTINGS_PER_SELLER}-listing cap per seller ({existing_count} existing)",
            )

        texts = [f"{l.name}. {l.description}" for l in listings]
        vecs = engine.model.encode(texts, normalize_embeddings=True).astype("float32")
        row_ids = []
        with db.get_conn() as conn:
            for listing, vec in zip(listings, vecs):
                # embedding column kept for schema compatibility (NOT
                # NULL) but no longer written or read as a real vector --
                # the FAISS index below is now the single source of
                # truth for search. Real scale bug this replaces: search
                # used to re-parse this column's comma-joined floats for
                # EVERY listing on EVERY request; see
                # commerce_search_index.py's module docstring for the
                # measured ~32.7s/search cost that caused.
                cur = conn.execute(
                    "INSERT INTO commerce_listings (seller_id, item_id, name, description, unit_amount, currency, category, embedding, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (seller_id, listing.item_id, listing.name, listing.description, listing.unit_amount,
                     listing.currency, listing.category, "", time.time()),
                )
                row_ids.append(cur.lastrowid)

        commerce_index = _get_commerce_index(vecs.shape[1])
        commerce_index.add(row_ids, vecs)
        return len(listings)

    @app.post("/v1/commerce/sellers", response_model=SellerRegisterResponse)
    def register_seller(req: SellerRegisterRequest, authorization: Optional[str] = Header(None)):
        raw_key = require_key_fn(authorization)
        if not rate_limit.allow(raw_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        seller_id = _create_seller(db.hash_key(raw_key), req.name, req.checkout_session_url)
        return SellerRegisterResponse(seller_id=seller_id)

    @app.post("/v1/commerce/sellers/{seller_id}/listings", response_model=ListingsAddResponse)
    def add_listings(seller_id: int, req: ListingsAddRequest, authorization: Optional[str] = Header(None)):
        raw_key = require_key_fn(authorization)
        if not rate_limit.allow(raw_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        added = _create_listings(db.hash_key(raw_key), seller_id, req.listings)
        return ListingsAddResponse(added=added)

    @app.get("/v1/commerce/my-sellers", response_model=List[SellerWithListings])
    def my_sellers(authorization: Optional[str] = Header(None)):
        """For a dashboard: everything this API key has registered as a
        seller, with its listings nested -- there was previously no way
        to see this without remembering seller_id from the original
        register_seller response."""
        raw_key = require_key_fn(authorization)
        key_hash = db.hash_key(raw_key)
        with db.get_conn() as conn:
            sellers = conn.execute(
                "SELECT * FROM commerce_sellers WHERE key_hash = ? ORDER BY created_at DESC", (key_hash,)
            ).fetchall()
            out = []
            for s in sellers:
                listings = conn.execute(
                    "SELECT * FROM commerce_listings WHERE seller_id = ? ORDER BY created_at DESC", (s["id"],)
                ).fetchall()
                out.append(SellerWithListings(
                    seller_id=s["id"], name=s["name"], checkout_session_url=s["checkout_session_url"],
                    listings=[
                        ListingOut(item_id=l["item_id"], name=l["name"], description=l["description"],
                                   unit_amount=l["unit_amount"], currency=l["currency"], category=l["category"])
                        for l in listings
                    ],
                ))
        return out

    @app.post("/v1/commerce/buyer-agents", response_model=BuyerAgentOut)
    def create_buyer_agent(req: BuyerAgentIn, authorization: Optional[str] = Header(None)):
        """A saved buyer-agent CONFIGURATION, not a running process --
        see db.py's commerce_buyer_agents comment. Lets an external agent
        (any framework) reference buyer_agent_id in /v1/commerce/search
        instead of repeating the same intent/filters on every call."""
        raw_key = require_key_fn(authorization)
        if not rate_limit.allow(raw_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        if not req.name.strip():
            raise HTTPException(status_code=400, detail="name must not be empty")
        if not req.default_intent.strip():
            raise HTTPException(status_code=400, detail="default_intent must not be empty")
        if req.max_price is not None and req.max_price <= 0:
            raise HTTPException(status_code=400, detail="max_price must be positive if set")
        key_hash = db.hash_key(raw_key)
        with db.get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO commerce_buyer_agents (key_hash, name, default_intent, max_price, category, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key_hash, req.name, req.default_intent, req.max_price, req.category, time.time()),
            )
            agent_id = cur.lastrowid
        return BuyerAgentOut(id=agent_id, name=req.name, default_intent=req.default_intent,
                              max_price=req.max_price, category=req.category)

    @app.get("/v1/commerce/buyer-agents", response_model=List[BuyerAgentOut])
    def list_buyer_agents(authorization: Optional[str] = Header(None)):
        raw_key = require_key_fn(authorization)
        key_hash = db.hash_key(raw_key)
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM commerce_buyer_agents WHERE key_hash = ? ORDER BY created_at DESC", (key_hash,)
            ).fetchall()
        return [
            BuyerAgentOut(id=r["id"], name=r["name"], default_intent=r["default_intent"],
                          max_price=r["max_price"], category=r["category"])
            for r in rows
        ]

    @app.delete("/v1/commerce/buyer-agents/{agent_id}", response_model=DeleteResponse)
    def delete_buyer_agent(agent_id: int, authorization: Optional[str] = Header(None)):
        raw_key = require_key_fn(authorization)
        key_hash = db.hash_key(raw_key)
        with db.get_conn() as conn:
            cur = conn.execute(
                "DELETE FROM commerce_buyer_agents WHERE id = ? AND key_hash = ?", (agent_id, key_hash)
            )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="no buyer agent with that id owned by this API key")
        return DeleteResponse(deleted=True)

    @app.post("/v1/commerce/import", response_model=ImportResponse)
    def commerce_import(req: ImportRequest, authorization: Optional[str] = Header(None)):
        """Universal onboarding path: paste a JSON description of sellers
        (with their listings) and/or buyer-agent profiles instead of
        making one call per seller and one per listing batch. Same
        validation, same underlying tables, as the individual endpoints
        above -- this is a convenience wrapper, not a second data model.
        Partial-batch semantics: sellers are created one at a time and a
        failure on one (e.g. bad checkout_session_url) stops the import
        with the count of what succeeded before it, rather than either
        silently skipping bad entries or rolling back ones that already
        committed -- SQLite autocommit per statement here means a
        already-created seller from earlier in the same request really
        did get created, and hiding that would be a real inconsistency
        between what this response says and what /v1/commerce/my-sellers
        would show right after."""
        raw_key = require_key_fn(authorization)
        if not rate_limit.allow(raw_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        key_hash = db.hash_key(raw_key)

        sellers_created = 0
        listings_created = 0
        buyer_agents_created = 0

        for s in req.sellers:
            seller_id = _create_seller(key_hash, s.name, s.checkout_session_url)
            sellers_created += 1
            if s.listings:
                listings_created += _create_listings(key_hash, seller_id, s.listings)

        for ba in req.buyer_agents:
            if not ba.name.strip() or not ba.default_intent.strip():
                raise HTTPException(status_code=400, detail="buyer agent name and default_intent must not be empty")
            if ba.max_price is not None and ba.max_price <= 0:
                raise HTTPException(status_code=400, detail="buyer agent max_price must be positive if set")
            with db.get_conn() as conn:
                conn.execute(
                    "INSERT INTO commerce_buyer_agents (key_hash, name, default_intent, max_price, category, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key_hash, ba.name, ba.default_intent, ba.max_price, ba.category, time.time()),
                )
            buyer_agents_created += 1

        return ImportResponse(
            sellers_created=sellers_created, listings_created=listings_created,
            buyer_agents_created=buyer_agents_created,
        )

    def _do_commerce_search(intent: str, max_price: Optional[int], category: Optional[str], k: int, raw_key: str) -> List[CommerceMatch]:
        """The actual search/rank/learn logic, factored out of the
        /v1/commerce/search REST handler so ucp_adapter.py's UCP-shaped
        catalog search can reuse it exactly instead of a second
        implementation that could drift -- same reasoning as unifying the
        cost guard into core.search() earlier this session. Callers own
        their own auth/credit/rate-limit handling first; this only does
        the search itself."""
        qvec = engine.model.encode([intent], normalize_embeddings=True).astype("float32")[0]
        commerce_index = _get_commerce_index(qvec.shape[0])

        if commerce_index.ntotal == 0:
            return []

        # Real scale fix, measured before assuming it mattered: this used
        # to SELECT every listing matching category/price into Python and
        # score each one in a loop -- ~32.7s/search at 20,000 real
        # listings (see commerce_search_index.py's module docstring).
        # FAISS now generates a small candidate set directly (exact
        # cosine similarity, not approximate -- see that module for why),
        # and only THOSE rows get fetched/filtered. Over-fetches beyond
        # k because category/max_price filtering happens AFTER
        # retrieval, on the candidate set, not inside the vector search
        # itself -- a real, disclosed tradeoff: a very restrictive filter
        # combined with a small candidate pool can under-return relative
        # to what technically exists in the full catalog. 10x/min-100 is
        # a judgment call, not a measurement.
        k_candidates = max(k * 10, 100)
        candidates = commerce_index.search(qvec, k_candidates)
        if not candidates:
            return []

        candidate_listing_ids = [listing_id for listing_id, _score in candidates]
        score_by_id = {listing_id: score for listing_id, score in candidates}

        with db.get_conn() as conn:
            placeholders = ",".join("?" * len(candidate_listing_ids))
            sql = (
                f"SELECT l.*, s.name AS seller_name, s.checkout_session_url AS checkout_session_url "
                f"FROM commerce_listings l JOIN commerce_sellers s ON l.seller_id = s.id "
                f"WHERE l.id IN ({placeholders})"
            )
            params = list(candidate_listing_ids)
            if category:
                sql += " AND l.category = ?"
                params.append(category)
            if max_price is not None:
                sql += " AND (l.unit_amount IS NULL OR l.unit_amount <= ?)"
                params.append(max_price)
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        # score_by_id's value IS the real cosine similarity already
        # (IndexFlatIP over L2-normalized vectors == cosine, exactly the
        # same equivalence the old code relied on for its manual dot
        # product) -- no re-computation needed.
        scored = [(score_by_id[row["id"]], row) for row in rows]

        # Listings are keyed by (seller_id, item_id) for the memory, not
        # bare item_id -- item_id is only unique WITHIN one seller's own
        # catalog (it's their own SKU), so two different sellers reusing
        # "sku-1" must not be treated as the same listing by the learned
        # affinity network.
        key_hash = db.hash_key(raw_key)
        memory = _get_memory(key_hash)
        now_ms = time.time() * 1000.0
        memory.decay(now_ms)

        candidate_ids = [f"{row['seller_id']}:{row['item_id']}" for _cosine, row in scored]

        blended = []
        for cosine, row in scored:
            composite_id = f"{row['seller_id']}:{row['item_id']}"
            # Two real signals read here, both reflecting PAST searches
            # only (read before this search's own observe_search call
            # below, so a listing can never boost itself from the very
            # search computing its rank right now):
            #
            # 1. heat -- short-term/decaying (membrane_potential), "has
            #    this been recently active." Deliberately NOT persisted
            #    across a restart (see commerce_spiking_memory.py).
            # 2. learned connection strength to any OTHER candidate in
            #    THIS SAME result batch -- the long-term, persisted STDP
            #    signal (see _save_memory/_get_memory above). Real bug
            #    this fixes, found by testing: memory_boost used to read
            #    heat only, so persisting synapse weights across a
            #    restart had zero visible effect on ranking -- the one
            #    thing persistence is actually for.
            heat = memory.heat(composite_id)
            heat_component = min(1.0, heat / commerce_spiking_memory.DEFAULT_THRESHOLD)

            best_connection = 0.0
            for other_id in candidate_ids:
                if other_id == composite_id:
                    continue
                conn = memory.learned_connection(other_id, composite_id)
                if conn is not None:
                    best_connection = max(best_connection, conn)
            # Learned weights start at DEFAULT_SEED_WEIGHT and grow from
            # there -- normalized relative to the seed, not to threshold
            # (a totally different scale). 4x seed is a judgment call
            # for "meaningfully learned," not a measurement (no real
            # usage data exists yet to tune this against).
            seed = commerce_spiking_memory.DEFAULT_SEED_WEIGHT
            connection_component = min(1.0, max(0.0, best_connection - seed) / (3 * seed))

            boost = MEMORY_BLEND_WEIGHT * min(1.0, heat_component + connection_component)
            blended.append((cosine + boost, cosine, boost, row, composite_id))
        blended.sort(key=lambda t: -t[0])

        top = blended[:k]

        # A real match_id per returned match -- the shared correlation
        # point the two-sided reputation system needs (see
        # commerce_matches' table comment in db.py). Logged for every
        # match actually shown, not just ones that convert -- a buyer or
        # seller can only reference a match_id that genuinely happened.
        match_ids = [str(uuid.uuid4()) for _ in top]
        now = time.time()
        with db.get_conn() as conn:
            conn.executemany(
                "INSERT INTO commerce_matches (match_id, buyer_key_hash, seller_id, item_id, created_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (mid, key_hash, row["seller_id"], row["item_id"], now)
                    for mid, (final_score, cosine, boost, row, composite_id) in zip(match_ids, top)
                ],
            )

        matches = [
            CommerceMatch(
                score=final_score, seller_id=row["seller_id"], seller_name=row["seller_name"], checkout_session_url=row["checkout_session_url"],
                item_id=row["item_id"], name=row["name"], description=row["description"],
                unit_amount=row["unit_amount"], currency=row["currency"], memory_boost=boost, match_id=mid,
            )
            for mid, (final_score, cosine, boost, row, composite_id) in zip(match_ids, top)
        ]

        # Now that ranking is decided, teach the memory what this search
        # actually returned -- affects FUTURE searches for this key, not
        # this one (see the heat-read-before-observe ordering above).
        memory.observe_search([composite_id for *_rest, composite_id in top], now_ms)
        _save_memory(key_hash, memory)

        return matches

    @app.post("/v1/commerce/search", response_model=CommerceSearchResponse)
    def commerce_search(req: CommerceSearchRequest, authorization: Optional[str] = Header(None)):
        raw_key = require_key_fn(authorization)
        if not rate_limit.allow(raw_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")

        intent, max_price, category = req.intent, req.max_price, req.category
        if req.buyer_agent_id is not None:
            key_hash = db.hash_key(raw_key)
            with db.get_conn() as conn:
                agent = conn.execute(
                    "SELECT * FROM commerce_buyer_agents WHERE id = ? AND key_hash = ?",
                    (req.buyer_agent_id, key_hash),
                ).fetchone()
            if agent is None:
                raise HTTPException(status_code=404, detail="no buyer agent with that id owned by this API key")
            # Explicit fields in this call win; a profile only fills in
            # what the caller didn't already specify.
            intent = intent if intent is not None else agent["default_intent"]
            max_price = max_price if max_price is not None else agent["max_price"]
            category = category if category is not None else agent["category"]
        if not intent:
            raise HTTPException(status_code=400, detail="intent is required, either directly or via a buyer_agent_id with a saved default_intent")

        if not db.deduct_credit(raw_key, credits_per_search):
            raise HTTPException(status_code=402, detail="insufficient credits")
        if not engine.ready or engine.model is None:
            db.deduct_credit(raw_key, -credits_per_search)
            raise HTTPException(status_code=503, detail="search engine not ready yet")

        matches = _do_commerce_search(intent, max_price, category, req.k, raw_key)
        record = db.get_key_record(raw_key)
        return CommerceSearchResponse(matches=matches, credits_remaining=record["credits"])

    @app.post("/v1/commerce/feedback", response_model=FeedbackResponse)
    def commerce_feedback(req: FeedbackRequest, authorization: Optional[str] = Header(None)):
        """The self-adjusting feedback loop: a buyer's agent reports back
        what actually happened after calling a match's checkout_session_url
        directly (this API never sees that call -- see the module
        docstring's trust boundary). "purchased" feeds real ground truth
        into that buyer's ListingAffinityMemory via a stronger-than-normal
        STDP drive (CONFIRMED_PURCHASE_DRIVE), the same "ground-truth-
        confirmed signal reinforced more than a mere occurrence" pattern
        spiking_adaptive_weights.py already established for evidence-kind
        learning, applied here to listings instead.

        Disclosed limitation, inherited from that same established
        pattern (see spikeling-stdp-dt-0 lesson + spiking_adaptive_
        weights.py's own docstring): replayed in forward-chronological
        order, the real STDPLearner rule's `dt` is never negative, so it
        can only push a listing's learned weight UP, never down --
        "not_purchased"/"irrelevant" feedback is honestly recorded (for
        future analysis, and so this endpoint doesn't silently drop real
        signal) but does NOT currently suppress that listing's ranking.
        Not a workaround for this; disclosed as a real, accepted
        characteristic of the mechanism, consistent with how it's already
        treated elsewhere in this codebase.

        Also disclosed: this is SELF-REPORTED by the buyer's own key, not
        independently verified against the seller's real checkout record
        (this API never sees that transaction, by design). A malicious or
        careless caller could report a false "purchased" to inflate a
        listing's ranking for their own key's future searches -- the
        blast radius is scoped to that one caller's own learned memory,
        not global, but this is a real, not-yet-hardened trust gap worth
        stating plainly rather than leaving implicit."""
        raw_key = require_key_fn(authorization)
        if req.outcome not in _VALID_OUTCOMES:
            raise HTTPException(status_code=400, detail=f"outcome must be one of {sorted(_VALID_OUTCOMES)}")

        key_hash = db.hash_key(raw_key)

        # If match_id is given, it's the real correlation point -- verify
        # it actually belongs to THIS buyer key (closes a real trust gap
        # the seller_id/item_id-only path below still has: without a
        # match_id, any authenticated key could report feedback for a
        # seller/item combo it never actually searched for). Legacy
        # callers built before match_id existed (the CLI/MCP tools from
        # before this) still work via the old path.
        if req.match_id:
            with db.get_conn() as conn:
                match = conn.execute(
                    "SELECT * FROM commerce_matches WHERE match_id = ? AND buyer_key_hash = ?",
                    (req.match_id, key_hash),
                ).fetchone()
            if match is None:
                raise HTTPException(status_code=404, detail="no match_id found for this key -- it must be one returned by your own earlier /v1/commerce/search call")
            req.seller_id = match["seller_id"]
            req.item_id = match["item_id"]

        with db.get_conn() as conn:
            seller = conn.execute("SELECT * FROM commerce_sellers WHERE id = ?", (req.seller_id,)).fetchone()
        if seller is None:
            raise HTTPException(status_code=404, detail="no seller with that id")

        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO commerce_feedback (key_hash, seller_id, item_id, outcome, created_at) VALUES (?, ?, ?, ?, ?)",
                (key_hash, req.seller_id, req.item_id, req.outcome, time.time()),
            )

        reinforced = False
        if req.outcome == "purchased":
            memory = _get_memory(key_hash)
            now_ms = time.time() * 1000.0
            memory.decay(now_ms)
            composite_id = f"{req.seller_id}:{req.item_id}"
            # Real bug found by testing this live, not in isolation: a
            # FIXED drive (even one already kept below threshold, see
            # CONFIRMED_PURCHASE_DRIVE's own comment) can still push the
            # neuron over threshold and self-erase via the same LIF
            # fire/reset mechanism -- IF the neuron already has residual
            # heat from a real prior search, which is the realistic case
            # (a buyer searches before they buy). Reproduced live: search
            # for boots, then report a purchase, then search again --
            # the purchased item's boost read 0.0, the opposite of
            # intended. Fixed by computing the ACTUAL safe drive against
            # CURRENT state (post-decay) instead of a fixed constant,
            # guaranteeing this can only raise heat, never reset it.
            current_heat = memory.heat(composite_id)
            safety_margin = 1.0
            safe_drive = max(0.0, min(
                CONFIRMED_PURCHASE_DRIVE,
                commerce_spiking_memory.DEFAULT_THRESHOLD - safety_margin - current_heat,
            ))
            memory.observe_search([composite_id], now_ms, top_drive=safe_drive)
            _save_memory(key_hash, memory)
            reinforced = True

            with db.get_conn() as conn:
                listing = conn.execute(
                    "SELECT * FROM commerce_listings WHERE seller_id = ? AND item_id = ?",
                    (req.seller_id, req.item_id),
                ).fetchone()
            listing_name = listing["name"] if listing else req.item_id
            try:
                obsidian_memory.log_project_work(
                    title=f"Real confirmed purchase: {listing_name} via {seller['name']}",
                    project_tag="observe-api-commerce",
                    status="real_event",
                    output_text=(
                        f"Buyer key {key_hash[:12]}... reported outcome=purchased for "
                        f"seller_id={req.seller_id} ({seller['name']}), item_id={req.item_id} "
                        f"({listing_name}). Self-reported by the buyer's own agent, not "
                        f"independently verified against the seller's real ACP checkout record "
                        f"(this API never sees that transaction by design -- see "
                        f"commerce_router.py's module docstring). Reinforced this buyer's "
                        f"listing-affinity memory with drive={CONFIRMED_PURCHASE_DRIVE} via real "
                        f"Spikeling STDP (commerce_spiking_memory.py)."
                    ),
                )
            except Exception:
                # Archiving to the vault is a best-effort side effect of a
                # real event, never a dependency of this endpoint's own
                # correctness -- see module docstring. A vault-write
                # failure (disk full, path missing, permissions) must not
                # turn a real, successfully-recorded purchase into a
                # failed API response.
                pass

        note = (
            "Recorded and reinforced -- this listing's learned affinity for your key just grew from real ground truth."
            if reinforced else
            "Recorded. Not currently used to suppress ranking (the real STDP mechanism here can only "
            "strengthen, never weaken, a learned connection) -- see this endpoint's docstring."
        )
        return FeedbackResponse(recorded=True, reinforced=reinforced, note=note)

    def _reputation_for_key(key_hash: str) -> dict:
        with db.get_conn() as conn:
            total_matches = conn.execute(
                "SELECT COUNT(*) AS n FROM commerce_matches WHERE buyer_key_hash = ?", (key_hash,)
            ).fetchone()["n"]
            buyer_confirmed = conn.execute(
                "SELECT COUNT(*) AS n FROM commerce_feedback WHERE key_hash = ? AND outcome = 'purchased'", (key_hash,)
            ).fetchone()["n"]
            seller_confirmed = conn.execute(
                "SELECT COUNT(*) AS n FROM commerce_seller_feedback sf "
                "JOIN commerce_matches m ON sf.match_id = m.match_id "
                "WHERE m.buyer_key_hash = ? AND sf.outcome = 'fulfilled'",
                (key_hash,),
            ).fetchone()["n"]
            disputes = conn.execute(
                "SELECT COUNT(*) AS n FROM commerce_seller_feedback sf "
                "JOIN commerce_matches m ON sf.match_id = m.match_id "
                "WHERE m.buyer_key_hash = ? AND sf.outcome = 'disputed'",
                (key_hash,),
            ).fetchone()["n"]
        return {
            "tier": _compute_tier(buyer_confirmed, seller_confirmed, disputes),
            "total_matches": total_matches,
            "buyer_confirmed_purchases": buyer_confirmed,
            "seller_confirmed_fulfillments": seller_confirmed,
            "disputes": disputes,
        }

    @app.post("/v1/commerce/seller-feedback", response_model=SellerFeedbackResponse)
    def commerce_seller_feedback(req: SellerFeedbackRequest, authorization: Optional[str] = Header(None)):
        """The seller-side half of the two-sided reputation loop. Only the
        seller who OWNS the seller_id referenced by match_id can report on
        it -- authenticated by their own API key, verified against
        commerce_sellers.key_hash, not just trusted from the request body.
        "fulfilled" is real ground truth that a buyer-agent's key can
        eventually reach "verified" tier from; "disputed" resets that
        key's tier back to "new" rather than being averaged away (see
        _compute_tier) -- a real, deliberate asymmetry: trust should be
        harder to keep than to lose."""
        raw_key = require_key_fn(authorization)
        if req.outcome not in _VALID_SELLER_OUTCOMES:
            raise HTTPException(status_code=400, detail=f"outcome must be one of {sorted(_VALID_SELLER_OUTCOMES)}")
        if req.rating is not None and not (1 <= req.rating <= 5):
            raise HTTPException(status_code=400, detail="rating must be 1-5 if given")

        seller_key_hash = db.hash_key(raw_key)
        with db.get_conn() as conn:
            match = conn.execute("SELECT * FROM commerce_matches WHERE match_id = ?", (req.match_id,)).fetchone()
        if match is None:
            raise HTTPException(status_code=404, detail="no match with that match_id")
        with db.get_conn() as conn:
            seller = conn.execute(
                "SELECT * FROM commerce_sellers WHERE id = ? AND key_hash = ?",
                (match["seller_id"], seller_key_hash),
            ).fetchone()
        if seller is None:
            raise HTTPException(status_code=403, detail="this match's seller is not owned by your API key")

        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO commerce_seller_feedback (match_id, seller_key_hash, outcome, rating, created_at) VALUES (?, ?, ?, ?, ?)",
                (req.match_id, seller_key_hash, req.outcome, req.rating, time.time()),
            )
        return SellerFeedbackResponse(recorded=True, note="Recorded. Contributes to the buyer key's reputation tier.")

    @app.get("/v1/commerce/my-reputation", response_model=ReputationSummary)
    def commerce_my_reputation(authorization: Optional[str] = Header(None)):
        raw_key = require_key_fn(authorization)
        return ReputationSummary(**_reputation_for_key(db.hash_key(raw_key)))

    @app.get("/v1/commerce/verify-match", response_model=VerifyMatchResponse)
    def commerce_verify_match(match_id: str, authorization: Optional[str] = Header(None)):
        """For a seller deciding whether to trust an incoming buyer before
        fulfilling: pass the match_id you'd have from a buyer referencing
        it in their real checkout call. Only the owning seller can check
        it -- returns the buyer's tier and counts, never the buyer's raw
        key or unrelated transaction history (privacy boundary: a
        seller learns "is this buyer trustworthy," not "who is this
        buyer, what else have they bought")."""
        raw_key = require_key_fn(authorization)
        seller_key_hash = db.hash_key(raw_key)
        with db.get_conn() as conn:
            match = conn.execute("SELECT * FROM commerce_matches WHERE match_id = ?", (match_id,)).fetchone()
        if match is None:
            return VerifyMatchResponse(found=False)
        with db.get_conn() as conn:
            seller = conn.execute(
                "SELECT * FROM commerce_sellers WHERE id = ? AND key_hash = ?",
                (match["seller_id"], seller_key_hash),
            ).fetchone()
        if seller is None:
            raise HTTPException(status_code=403, detail="this match's seller is not owned by your API key")
        rep = _reputation_for_key(match["buyer_key_hash"])
        return VerifyMatchResponse(
            found=True, tier=rep["tier"],
            buyer_confirmed_purchases=rep["buyer_confirmed_purchases"],
            seller_confirmed_fulfillments=rep["seller_confirmed_fulfillments"],
            disputes=rep["disputes"],
        )

    @app.get("/v1/commerce/network-stats", response_model=NetworkStats)
    def commerce_network_stats():
        """Public, no API key needed -- aggregate and anonymized, the
        "platform" view of the trust network's real size. No individual
        buyer/seller identity is derivable from this.

        Disclosed scale limitation, not hidden: this recomputes each
        distinct buyer's tier with its own DB round-trip -- fine at
        today's real scale (zero production transactions), a real O(N)
        cost once there are many thousands of distinct agents. The same
        class of thing the FAISS rewrite fixed for search; not fixed
        here yet since there's no real traffic to measure against, matching
        this project's own "measure before assuming it matters" discipline
        rather than optimizing a number nobody has hit yet."""
        with db.get_conn() as conn:
            buyer_hashes = [r["buyer_key_hash"] for r in conn.execute("SELECT DISTINCT buyer_key_hash FROM commerce_matches").fetchall()]
            total_matches = conn.execute("SELECT COUNT(*) AS n FROM commerce_matches").fetchone()["n"]
            total_confirmed = conn.execute(
                "SELECT COUNT(*) AS n FROM commerce_seller_feedback WHERE outcome = 'fulfilled'"
            ).fetchone()["n"]
            total_disputes = conn.execute(
                "SELECT COUNT(*) AS n FROM commerce_seller_feedback WHERE outcome = 'disputed'"
            ).fetchone()["n"]

        verified = trusted = 0
        for bh in buyer_hashes:
            tier = _reputation_for_key(bh)["tier"]
            if tier == "verified":
                verified += 1
            elif tier == "trusted":
                trusted += 1

        return NetworkStats(
            total_agents=len(buyer_hashes), verified_agents=verified, trusted_agents=trusted,
            total_matches=total_matches, total_confirmed_transactions=total_confirmed,
            total_disputes=total_disputes,
        )

    @app.post("/v1/commerce/checkout_sessions", response_model=CheckoutSessionResponse)
    def observe_checkout_session(req: CheckoutSessionRequest):
        """OBSERVE's own real ACP checkout endpoint -- this is the exact
        checkout_session_url OBSERVE registers for itself as a seller in
        its own marketplace, so any buyer-agent that discovers OBSERVE via
        commerce_search (searching for something like "semantic code
        search API for agents") can complete a real purchase the same way
        it would with any other listed seller, no separate integration.

        Deliberately simplified relative to full ACP: one real item (the
        credit package /v1/signup already sells), not an arbitrary
        catalog, and a caller with no existing OBSERVE account gets one
        created inline rather than needing a pre-existing key -- "wants
        to buy OBSERVE credits" and "doesn't have an OBSERVE key yet" are
        the same caller by definition here. Reuses billing.py's real
        Stripe integration, not a second payment path.
        """
        item = _OBSERVE_CATALOG.get(req.item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"unknown item_id -- must be one of {sorted(_OBSERVE_CATALOG)}")
        if not req.email.strip():
            raise HTTPException(status_code=400, detail="email must not be empty")
        signup_bonus = int(os.environ.get("OBSERVE_SIGNUP_BONUS_CREDITS", "100"))
        raw_key = db.create_api_key(req.email, initial_credits=signup_bonus)
        key_hash = db.hash_key(raw_key)
        checkout_url = billing.create_checkout_session(req.email, key_hash)
        return CheckoutSessionResponse(
            id=key_hash[:16],
            status="requires_payment",
            checkout_url=checkout_url,
            api_key=raw_key,
            note=(
                f"New OBSERVE account created with {signup_bonus} free trial "
                "credits -- save this api_key now, it is only ever shown "
                "once. Complete checkout_url to add the paid package."
            ),
        )

    # Returned so ucp_adapter.py can reuse the exact same search/rank/learn
    # logic for its UCP-shaped catalog endpoint, instead of a second
    # implementation that could drift from this one.
    return _do_commerce_search
