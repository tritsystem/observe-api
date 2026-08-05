"""
Framework-agnostic HTTP client for /v1/commerce/* -- same shared-core
pattern as core.py's code-search client, extended to today's ACP
buyer/seller discovery work so it's reachable from the MCP server, the
CLI, and (later, if wanted) LangChain/CrewAI wrappers without duplicating
request/error-handling logic three times.
"""
import os
from typing import List, Optional

import httpx

API_BASE = os.environ.get("OBSERVE_API_BASE", "https://api.observe-search.online")


def _resolve_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.environ.get("OBSERVE_API_KEY")


def register_seller(name: str, checkout_session_url: str, api_key: Optional[str] = None) -> str:
    key = _resolve_key(api_key)
    if not key:
        return "Error: no API key. Pass api_key= or set OBSERVE_API_KEY."
    try:
        resp = httpx.post(
            f"{API_BASE}/v1/commerce/sellers",
            headers={"Authorization": f"Bearer {key}"},
            json={"name": name, "checkout_session_url": checkout_session_url},
            timeout=30,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"
    return f"Registered seller_id={resp.json()['seller_id']}"


def add_listings(seller_id: int, listings: List[dict], api_key: Optional[str] = None) -> str:
    """listings: [{"item_id", "name", "description", "unit_amount"?, "currency"?, "category"?}, ...]"""
    key = _resolve_key(api_key)
    if not key:
        return "Error: no API key. Pass api_key= or set OBSERVE_API_KEY."
    try:
        resp = httpx.post(
            f"{API_BASE}/v1/commerce/sellers/{seller_id}/listings",
            headers={"Authorization": f"Bearer {key}"},
            json={"listings": listings},
            timeout=30,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"
    return f"Added {resp.json()['added']} listing(s)."


def commerce_search(intent: str, max_price: Optional[int] = None, category: Optional[str] = None,
                     k: int = 10, api_key: Optional[str] = None) -> str:
    key = _resolve_key(api_key)
    if not key:
        return "Error: no API key. Pass api_key= or set OBSERVE_API_KEY."
    try:
        resp = httpx.post(
            f"{API_BASE}/v1/commerce/search",
            headers={"Authorization": f"Bearer {key}"},
            json={"intent": intent, "max_price": max_price, "category": category, "k": k},
            timeout=30,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    if resp.status_code == 402:
        return "Error: insufficient credits."
    if resp.status_code == 401:
        return "Error: invalid API key."
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"

    data = resp.json()
    if not data["matches"]:
        return f"No matches found. ({data['credits_remaining']} credits remaining.)"
    lines = [f"{len(data['matches'])} match(es), {data['credits_remaining']} credits remaining:\n"]
    for m in data["matches"]:
        price = f"{m['unit_amount'] / 100:.2f} {m['currency']}" if m.get("unit_amount") else "price not listed"
        lines.append(
            f"[{m['seller_name']} (seller_id={m['seller_id']})] {m['name']} — {price} (score {m['score']:.3f})\n"
            f"  checkout: {m['checkout_session_url']}  item_id={m['item_id']}  match_id={m.get('match_id', '')}\n"
        )
    return "\n".join(lines)


def report_purchase_feedback(seller_id: int, item_id: str, outcome: str, api_key: Optional[str] = None,
                              match_id: Optional[str] = None) -> str:
    """outcome: "purchased" | "not_purchased" | "irrelevant" -- see
    commerce_router.py's POST /v1/commerce/feedback docstring for the
    real, disclosed trust boundary (self-reported, not independently
    verified). Pass match_id (from a real commerce_search result) when
    you have it -- ties this report to a real, verifiable prior match
    instead of a bare seller_id/item_id claim, and is required for it to
    count toward your key's real reputation tier."""
    key = _resolve_key(api_key)
    if not key:
        return "Error: no API key. Pass api_key= or set OBSERVE_API_KEY."
    try:
        resp = httpx.post(
            f"{API_BASE}/v1/commerce/feedback",
            headers={"Authorization": f"Bearer {key}"},
            json={"seller_id": seller_id, "item_id": item_id, "outcome": outcome, "match_id": match_id},
            timeout=30,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"
    return resp.json()["note"]


def report_seller_feedback(match_id: str, outcome: str, rating: Optional[int] = None,
                            api_key: Optional[str] = None) -> str:
    """outcome: "fulfilled" | "buyer_never_completed" | "disputed". Only
    the seller who owns the match's seller_id can report -- authenticated
    by your own API key, not the request body."""
    key = _resolve_key(api_key)
    if not key:
        return "Error: no API key. Pass api_key= or set OBSERVE_API_KEY."
    try:
        resp = httpx.post(
            f"{API_BASE}/v1/commerce/seller-feedback",
            headers={"Authorization": f"Bearer {key}"},
            json={"match_id": match_id, "outcome": outcome, "rating": rating},
            timeout=30,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"
    return resp.json()["note"]


def get_my_reputation(api_key: Optional[str] = None) -> str:
    key = _resolve_key(api_key)
    if not key:
        return "Error: no API key. Pass api_key= or set OBSERVE_API_KEY."
    try:
        resp = httpx.get(f"{API_BASE}/v1/commerce/my-reputation", headers={"Authorization": f"Bearer {key}"}, timeout=15)
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"
    d = resp.json()
    return (
        f"Tier: {d['tier']}\n"
        f"Total matches: {d['total_matches']}\n"
        f"Buyer-confirmed purchases: {d['buyer_confirmed_purchases']}\n"
        f"Seller-confirmed fulfillments: {d['seller_confirmed_fulfillments']}\n"
        f"Disputes: {d['disputes']}"
    )


def verify_match(match_id: str, api_key: Optional[str] = None) -> str:
    """For a seller deciding whether to trust an incoming buyer -- returns
    the buyer's reputation tier without exposing their identity or
    unrelated history."""
    key = _resolve_key(api_key)
    if not key:
        return "Error: no API key. Pass api_key= or set OBSERVE_API_KEY."
    try:
        resp = httpx.get(
            f"{API_BASE}/v1/commerce/verify-match", params={"match_id": match_id},
            headers={"Authorization": f"Bearer {key}"}, timeout=15,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"
    d = resp.json()
    if not d["found"]:
        return "No match with that match_id."
    return (
        f"Tier: {d['tier']}\n"
        f"Buyer-confirmed purchases: {d['buyer_confirmed_purchases']}\n"
        f"Seller-confirmed fulfillments: {d['seller_confirmed_fulfillments']}\n"
        f"Disputes: {d['disputes']}"
    )


def get_network_stats() -> str:
    """Public, no API key needed -- the aggregate size/health of the
    trust network."""
    try:
        resp = httpx.get(f"{API_BASE}/v1/commerce/network-stats", timeout=15)
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"
    d = resp.json()
    return (
        f"Total agents seen: {d['total_agents']} "
        f"({d['verified_agents']} verified, {d['trusted_agents']} trusted)\n"
        f"Total matches: {d['total_matches']}\n"
        f"Total confirmed transactions: {d['total_confirmed_transactions']}\n"
        f"Total disputes: {d['total_disputes']}"
    )


COMMERCE_SEARCH_DESCRIPTION = (
    "ACP-compatible buyer/seller discovery -- given a natural-language "
    "description of what you want to buy, returns matched real sellers' "
    "checkout_session_url + item_id + match_id for you (the calling agent) "
    "to complete a purchase directly against, following the real Agentic "
    "Commerce Protocol. This tool never handles payment or credentials, "
    "and never sees whether a purchase actually completes -- report back "
    "with report_purchase_feedback (pass the match_id) so future searches "
    "learn from real outcomes and your key's reputation tier can grow. "
    "Costs API credits per call."
)
