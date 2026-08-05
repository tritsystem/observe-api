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
"""
import time
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

CREDITS_PER_COMMERCE_SEARCH_DEFAULT = 1


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


class CommerceSearchResponse(BaseModel):
    matches: List[CommerceMatch]
    credits_remaining: int
    note: str = (
        "Each match's checkout_session_url is the seller's own real ACP "
        "endpoint (POST .../checkout_sessions per the Agentic Commerce "
        "Protocol). Call it directly with the given item_id to complete "
        "a purchase -- this API never handles payment or credentials."
    )


def register_commerce_routes(app: FastAPI, engine, db, rate_limit, require_key_fn, credits_per_search: int = CREDITS_PER_COMMERCE_SEARCH_DEFAULT):
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
            score = float(np.dot(qvec, vec))  # both sides are normalize_embeddings=True unit vectors, so dot == cosine similarity
            scored.append((score, row))
        scored.sort(key=lambda t: -t[0])

        matches = [
            CommerceMatch(
                score=score, seller_name=row["seller_name"], checkout_session_url=row["checkout_session_url"],
                item_id=row["item_id"], name=row["name"], description=row["description"],
                unit_amount=row["unit_amount"], currency=row["currency"],
            )
            for score, row in scored[:req.k]
        ]
        record = db.get_key_record(raw_key)
        return CommerceSearchResponse(matches=matches, credits_remaining=record["credits"])
