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
            f"[{m['seller_name']}] {m['name']} — {price} (score {m['score']:.3f})\n"
            f"  checkout: {m['checkout_session_url']}  item_id={m['item_id']}\n"
        )
    return "\n".join(lines)


def report_purchase_feedback(seller_id: int, item_id: str, outcome: str, api_key: Optional[str] = None) -> str:
    """outcome: "purchased" | "not_purchased" | "irrelevant" -- see
    commerce_router.py's POST /v1/commerce/feedback docstring for the
    real, disclosed trust boundary (self-reported, not independently
    verified)."""
    key = _resolve_key(api_key)
    if not key:
        return "Error: no API key. Pass api_key= or set OBSERVE_API_KEY."
    try:
        resp = httpx.post(
            f"{API_BASE}/v1/commerce/feedback",
            headers={"Authorization": f"Bearer {key}"},
            json={"seller_id": seller_id, "item_id": item_id, "outcome": outcome},
            timeout=30,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"
    return resp.json()["note"]


COMMERCE_SEARCH_DESCRIPTION = (
    "ACP-compatible buyer/seller discovery -- given a natural-language "
    "description of what you want to buy, returns matched real sellers' "
    "checkout_session_url + item_id for you (the calling agent) to complete "
    "a purchase directly against, following the real Agentic Commerce "
    "Protocol. This tool never handles payment or credentials, and never "
    "sees whether a purchase actually completes -- report back with "
    "report_purchase_feedback so future searches learn from real outcomes. "
    "Costs API credits per call."
)
