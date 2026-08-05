"""
ucp_adapter.py -- publishes OBSERVE's aggregated multi-seller catalog as a
real Universal Commerce Protocol (UCP) business, reachable by Google's AI
Mode / Gemini and any other UCP-native client.

Real, disclosed scope, established by reading the actual spec rather
than assuming -- fetched directly from github.com/Universal-Commerce-
Protocol/ucp (source/schemas/*.json), not summarized secondhand:

UCP (launched 2026-01-11 by Google with Shopify/Etsy/Wayfair/Target/
Walmart) is a SEPARATE, NOT interoperable protocol from ACP -- the two
don't share a wire format, and UCP's own spec has the exact same
single-business scoping gap ACP does: "Platforms interact with
individual businesses via their published UCP profiles... There is no
aggregation layer, federated search mechanism, or marketplace-wide
product indexing described" (confirmed by reading the real spec
overview, not assumed). commerce_router.py's ACP adapter already fills
that gap for ACP; this module does the same thing for UCP -- OBSERVE
publishes itself as ONE UCP business whose catalog happens to aggregate
many real sellers underneath, reusing the exact same search/rank/learn
logic (commerce_router.py's _do_commerce_search, not a second
implementation).

What this implements, verified against the real fetched schemas:
- GET /.well-known/ucp -- a real UCP business profile (source/schemas/
  profile.json's business_schema), declaring the
  dev.ucp.shopping.catalog.search capability and a REST service binding.
- POST /ucp/catalog/search -- shaped exactly like source/schemas/
  shopping/catalog_search.json's search_request/search_response, mapping
  OBSERVE's internal listings to real UCP Product+Variant objects
  (source/schemas/shopping/types/product.json, variant.json).

Real, disclosed limitations, not hidden:
- Every listing maps to exactly ONE variant (itself) -- no options/
  variant support, since sellers don't provide that data to OBSERVE
  today. A real, honest simplification of UCP's richer variant model.
- HTTP Message Signatures (the JWK-based request signing UCP's real
  profile.json schema expects for production-grade security) are NOT
  implemented -- this endpoint is reachable over plain HTTPS + the same
  Bearer API key the rest of OBSERVE's commerce API uses, not UCP's own
  signature scheme. A strict, fully-compliant UCP platform may reject
  unsigned requests; this is a real, known gap flagged here rather than
  a claim of full compliance.
- No Checkout/Cart/Order capabilities -- OBSERVE never touches payment
  or checkout, by the same design principle as the ACP adapter (see
  commerce_router.py's module docstring). Only the Catalog Search
  capability is published.
"""
import time
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

UCP_VERSION = "2026-04-08"  # the real spec version whose schemas this was built against


class UcpSearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    pagination: Optional[dict] = None


def _build_manifest(base_url: str) -> dict:
    return {
        "ucp": {
            "version": UCP_VERSION,
            "services": {
                "dev.ucp.shopping": [
                    {"version": UCP_VERSION, "transport": "rest", "endpoint": f"{base_url}/ucp"}
                ]
            },
            "capabilities": {
                "dev.ucp.shopping.catalog.search": [
                    {"version": UCP_VERSION, "schema": "https://ucp.dev/schemas/shopping/catalog_search.json"}
                ]
            },
            # Deliberately empty, not omitted -- OBSERVE never handles
            # payment (see module docstring). business_schema requires
            # the key to exist; an empty registry is the honest way to
            # say "no payment handlers here," not a placeholder for one
            # that's coming.
            "payment_handlers": {},
        }
    }


def _match_to_ucp_product(match) -> dict:
    """Maps one CommerceMatch (commerce_router.py) to a real UCP Product
    with exactly one Variant -- see module docstring for why one variant,
    not the richer multi-variant model UCP's own schema supports."""
    composite_id = f"{match.seller_id}:{match.item_id}"
    amount = match.unit_amount if match.unit_amount is not None else 0
    currency = (match.currency or "usd").upper()
    price = {"amount": amount, "currency": currency}
    description = {"plain": match.description}
    variant = {
        "id": composite_id,
        "title": match.name,
        "description": description,
        "price": price,
        "seller": {"name": match.seller_name},
    }
    return {
        "id": composite_id,
        "title": match.name,
        "description": description,
        "price_range": {"min": price, "max": price},
        "variants": [variant],
    }


def register_ucp_routes(app: FastAPI, engine, db, rate_limit, require_key_fn, credits_per_search: int, do_commerce_search) -> None:
    @app.get("/.well-known/ucp", include_in_schema=False)
    def ucp_profile():
        # Real base_url would ideally come from request context; using
        # the same fixed production domain the rest of this codebase's
        # docs/landing page already hardcode (see server.py's own
        # OBSERVE_CHECKOUT_SUCCESS_URL default) rather than trusting a
        # client-supplied Host header for a discovery document.
        return _build_manifest("https://api.observe-search.online")

    @app.post("/ucp/catalog/search")
    def ucp_catalog_search(req: UcpSearchRequest, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None, alias="X-Api-Key")):
        # UCP's own real spec supports an X-Api-Key header alongside
        # Authorization (seen in the real fetched checkout endpoint
        # parameters) -- accept either, so a UCP-native caller doesn't
        # need to know OBSERVE's specific Bearer convention.
        raw_key = None
        if authorization and authorization.startswith("Bearer "):
            raw_key = authorization.removeprefix("Bearer ").strip()
        elif x_api_key:
            raw_key = x_api_key
        if not raw_key or db.get_key_record(raw_key) is None:
            raise HTTPException(status_code=401, detail="missing or invalid API key (Authorization: Bearer ... or X-Api-Key)")

        if not rate_limit.allow(raw_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        if not db.deduct_credit(raw_key, credits_per_search):
            raise HTTPException(status_code=402, detail="insufficient credits")
        if not engine.ready or engine.model is None:
            db.deduct_credit(raw_key, -credits_per_search)
            raise HTTPException(status_code=503, detail="search engine not ready yet")

        filters = req.filters or {}
        max_price = filters.get("max_price")
        category = filters.get("category")
        pagination = req.pagination or {}
        k = pagination.get("limit", 10)

        matches = do_commerce_search(req.query, max_price, category, k, raw_key)

        return {
            "ucp": {"version": UCP_VERSION, "status": "success"},
            "products": [_match_to_ucp_product(m) for m in matches],
        }
