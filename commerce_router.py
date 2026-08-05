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
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import commerce_spiking_memory
import obsidian_memory

CREDITS_PER_COMMERCE_SEARCH_DEFAULT = 1

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
    intent: str
    max_price: Optional[int] = None  # minor currency units
    category: Optional[str] = None
    k: int = 10


class CommerceMatch(BaseModel):
    score: float
    seller_name: str
    checkout_session_url: str
    item_id: str
    name: str
    description: str
    unit_amount: Optional[int]
    currency: str
    memory_boost: float = 0.0  # how much learned affinity (see commerce_spiking_memory.py) nudged this match's score -- 0.0 if never matched together with anything before for this buyer key


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


class FeedbackResponse(BaseModel):
    recorded: bool
    reinforced: bool
    note: str


def register_commerce_routes(app: FastAPI, engine, db, rate_limit, require_key_fn, credits_per_search: int = CREDITS_PER_COMMERCE_SEARCH_DEFAULT):
    # Per-buyer-key learned listing-affinity memory (real Spikeling STDP,
    # see commerce_spiking_memory.py) -- one ListingAffinityMemory per
    # key_hash, created lazily on that key's first search and kept for
    # the life of this process. A restart resets learned affinity to
    # cold; that's a disclosed v1 limitation (no persistence layer for
    # the network's synapse weights yet), not silent data loss of
    # anything else -- the underlying listings/cosine ranking are
    # unaffected, only the additive memory_boost signal resets.
    _key_memories: Dict[str, commerce_spiking_memory.ListingAffinityMemory] = {}

    @app.post("/v1/commerce/sellers", response_model=SellerRegisterResponse)
    def register_seller(req: SellerRegisterRequest, authorization: Optional[str] = Header(None)):
        raw_key = require_key_fn(authorization)
        if not req.checkout_session_url.startswith("https://"):
            raise HTTPException(status_code=400, detail="checkout_session_url must be https -- ACP checkout sessions carry payment intent, never serve this over plain http")
        key_hash = db.hash_key(raw_key)
        with db.get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO commerce_sellers (key_hash, name, checkout_session_url, created_at) VALUES (?, ?, ?, ?)",
                (key_hash, req.name, req.checkout_session_url, time.time()),
            )
            seller_id = cur.lastrowid
        return SellerRegisterResponse(seller_id=seller_id)

    @app.post("/v1/commerce/sellers/{seller_id}/listings", response_model=ListingsAddResponse)
    def add_listings(seller_id: int, req: ListingsAddRequest, authorization: Optional[str] = Header(None)):
        raw_key = require_key_fn(authorization)
        key_hash = db.hash_key(raw_key)
        with db.get_conn() as conn:
            seller = conn.execute(
                "SELECT * FROM commerce_sellers WHERE id = ? AND key_hash = ?", (seller_id, key_hash)
            ).fetchone()
        if seller is None:
            raise HTTPException(status_code=404, detail="no seller with that id owned by this API key")
        if not engine.ready or engine.model is None:
            raise HTTPException(status_code=503, detail="search engine not ready yet")

        texts = [f"{l.name}. {l.description}" for l in req.listings]
        vecs = engine.model.encode(texts, normalize_embeddings=True).astype("float32")
        with db.get_conn() as conn:
            for listing, vec in zip(req.listings, vecs):
                conn.execute(
                    "INSERT INTO commerce_listings (seller_id, item_id, name, description, unit_amount, currency, category, embedding, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (seller_id, listing.item_id, listing.name, listing.description, listing.unit_amount,
                     listing.currency, listing.category, ",".join(str(x) for x in vec.tolist()), time.time()),
                )
        return ListingsAddResponse(added=len(req.listings))

    @app.post("/v1/commerce/search", response_model=CommerceSearchResponse)
    def commerce_search(req: CommerceSearchRequest, authorization: Optional[str] = Header(None)):
        raw_key = require_key_fn(authorization)
        if not rate_limit.allow(raw_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        if not db.deduct_credit(raw_key, credits_per_search):
            raise HTTPException(status_code=402, detail="insufficient credits")
        if not engine.ready or engine.model is None:
            db.deduct_credit(raw_key, -credits_per_search)
            raise HTTPException(status_code=503, detail="search engine not ready yet")

        import numpy as np

        with db.get_conn() as conn:
            sql = (
                "SELECT l.*, s.name AS seller_name, s.checkout_session_url AS checkout_session_url "
                "FROM commerce_listings l JOIN commerce_sellers s ON l.seller_id = s.id"
            )
            params = []
            clauses = []
            if req.category:
                clauses.append("l.category = ?")
                params.append(req.category)
            if req.max_price is not None:
                clauses.append("(l.unit_amount IS NULL OR l.unit_amount <= ?)")
                params.append(req.max_price)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            record = db.get_key_record(raw_key)
            return CommerceSearchResponse(matches=[], credits_remaining=record["credits"])

        qvec = engine.model.encode([req.intent], normalize_embeddings=True).astype("float32")[0]
        scored = []
        for row in rows:
            vec = np.array([float(x) for x in row["embedding"].split(",")], dtype="float32")
            cosine = float(np.dot(qvec, vec))  # both sides are normalize_embeddings=True unit vectors, so dot == cosine similarity
            scored.append((cosine, row))

        # Listings are keyed by (seller_id, item_id) for the memory, not
        # bare item_id -- item_id is only unique WITHIN one seller's own
        # catalog (it's their own SKU), so two different sellers reusing
        # "sku-1" must not be treated as the same listing by the learned
        # affinity network.
        key_hash = db.hash_key(raw_key)
        memory = _key_memories.setdefault(key_hash, commerce_spiking_memory.ListingAffinityMemory())
        now_ms = time.time() * 1000.0
        memory.decay(now_ms)

        blended = []
        for cosine, row in scored:
            composite_id = f"{row['seller_id']}:{row['item_id']}"
            # Heat reflects PAST searches only -- read before this
            # search's own observe_search call below, so a listing can
            # never boost itself from the very search that's computing
            # its rank right now.
            heat = memory.heat(composite_id)
            boost = MEMORY_BLEND_WEIGHT * min(1.0, heat / commerce_spiking_memory.DEFAULT_THRESHOLD)
            blended.append((cosine + boost, cosine, boost, row, composite_id))
        blended.sort(key=lambda t: -t[0])

        top = blended[:req.k]
        matches = [
            CommerceMatch(
                score=final_score, seller_name=row["seller_name"], checkout_session_url=row["checkout_session_url"],
                item_id=row["item_id"], name=row["name"], description=row["description"],
                unit_amount=row["unit_amount"], currency=row["currency"], memory_boost=boost,
            )
            for final_score, cosine, boost, row, composite_id in top
        ]

        # Now that ranking is decided, teach the memory what this search
        # actually returned -- affects FUTURE searches for this key, not
        # this one (see the heat-read-before-observe ordering above).
        memory.observe_search([composite_id for *_rest, composite_id in top], now_ms)

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

        with db.get_conn() as conn:
            seller = conn.execute("SELECT * FROM commerce_sellers WHERE id = ?", (req.seller_id,)).fetchone()
        if seller is None:
            raise HTTPException(status_code=404, detail="no seller with that id")

        key_hash = db.hash_key(raw_key)
        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO commerce_feedback (key_hash, seller_id, item_id, outcome, created_at) VALUES (?, ?, ?, ?, ?)",
                (key_hash, req.seller_id, req.item_id, req.outcome, time.time()),
            )

        reinforced = False
        if req.outcome == "purchased":
            memory = _key_memories.setdefault(key_hash, commerce_spiking_memory.ListingAffinityMemory())
            now_ms = time.time() * 1000.0
            memory.decay(now_ms)
            composite_id = f"{req.seller_id}:{req.item_id}"
            memory.observe_search([composite_id], now_ms, top_drive=CONFIRMED_PURCHASE_DRIVE)
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
