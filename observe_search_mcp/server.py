"""
Thin MCP client for the hosted OBSERVE Search API -- distinct from
OBSERVE's own local trit_mcp_server.py, which searches a LOCAL index. This
one calls the paid hosted API over HTTP with a configured API key, so
Claude Code / Claude Desktop / Cursor / any MCP client can search the
curated remote corpus (React, Django, NumPy, etc.) without running OBSERVE
locally at all.

Setup:
  pip install observe-search-mcp
  export OBSERVE_API_KEY=obs_...   # from POST /v1/signup
  claude mcp add observe-hosted -- python -m observe_search_mcp.server
"""
import os
import sys

import httpx

# Real bug found by actually installing and importing this, not assumed:
# mcp.server.fastmcp.FastMCP existed in the SDK's 1.x line (what pyproject
# .toml's loose `mcp>=1.0` pin allows) but was renamed/moved to
# mcp.server.mcpserver.MCPServer in the SDK's 2.0.0 release -- a fresh
# `pip install observe-search-mcp` today pulls 2.0.0 and the old import
# raises ModuleNotFoundError immediately, silently breaking every MCP
# client that tried to use this. Same decorator/run() API in both
# generations (verified directly against the installed 2.0.0 package's
# real constructor signature), so a try/except covers both without a
# behavior change for whichever one a caller actually has installed.
try:
    from mcp.server.fastmcp import FastMCP as _MCPServerClass
except ImportError:
    from mcp.server.mcpserver import MCPServer as _MCPServerClass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe_search_tools import commerce as _commerce  # noqa: E402
from observe_search_tools import core as _core  # noqa: E402

API_BASE = os.environ.get("OBSERVE_API_BASE", "https://api.observe-search.online")
API_KEY = os.environ.get("OBSERVE_API_KEY")

mcp = _MCPServerClass("observe-hosted-search")


@mcp.tool()
def search_code_hosted(query: str, k: int = 10, repo: str | None = None, force: bool = False) -> str:
    """Semantic code search over a curated set of popular open source repos
    (React, Django, NumPy, FastAPI, Tokio, and more -- call list_repos_hosted
    for the current list). Costs API credits per call (see check_balance).

    Use this when you can only DESCRIBE what you're looking for, not name
    the exact identifier -- e.g. "where does this handle retrying a failed
    upload." If you already know the exact function/class/file name, grep
    or a direct file read will be faster and free; this tool is for
    vocabulary-mismatch and concept-only queries, not exact-name lookups.
    A real, automated check (not just this description) refuses an
    exact-identifier-shaped query by default -- pass force=True if it's
    genuinely a concept-only query despite looking like one token.

    Args:
        query: natural-language description of what you're looking for.
        k: number of results to return (1-50, default 10).
        repo: optional -- scope the search to one indexed repo (see
            list_repos_hosted). Omit to search across all indexed repos.
        force: bypass the exact-identifier cost guard (default False).
    """
    # core.search() checks the cost guard BEFORE the API-key check -- no
    # redundant pre-check here, so that ordering stays correct instead of
    # this function silently short-circuiting it.
    return _core.search(query, k=k, repo=repo, api_key=API_KEY, force=force)


@mcp.tool()
def list_repos_hosted() -> str:
    """Lists which repositories are currently indexed and searchable via
    search_code_hosted's `repo` filter."""
    try:
        resp = httpx.get(f"{API_BASE}/v1/repos", timeout=10)
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    return ", ".join(resp.json()["repos"])


@mcp.tool()
def check_balance() -> str:
    """Checks the remaining API credit balance for the configured
    OBSERVE_API_KEY."""
    if not API_KEY:
        return "Error: OBSERVE_API_KEY is not set."
    try:
        resp = httpx.get(
            f"{API_BASE}/v1/balance",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"
    return f"{resp.json()['credits']} credits remaining."


@mcp.tool()
def register_seller_hosted(name: str, checkout_session_url: str) -> str:
    """Registers a seller for the ACP-compatible commerce discovery API --
    free (no credits charged). checkout_session_url must be the seller's
    own real ACP endpoint (https). Returns the new seller_id, needed for
    add_listings_hosted."""
    return _commerce.register_seller(name, checkout_session_url, api_key=API_KEY)


@mcp.tool()
def add_listings_hosted(seller_id: int, listings: list) -> str:
    """Adds product listings to a seller registered via
    register_seller_hosted -- free (no credits charged). Each listing:
    {"item_id": str, "name": str, "description": str, "unit_amount": int
    (minor currency units, optional), "currency": str (default "usd"),
    "category": str (optional)}."""
    return _commerce.add_listings(seller_id, listings, api_key=API_KEY)


@mcp.tool()
def commerce_search_hosted(intent: str, max_price: int | None = None, category: str | None = None, k: int = 10) -> str:
    """ACP-compatible buyer/seller discovery: describe what you want to buy
    in plain language, get back matched real sellers' checkout_session_url
    + item_id to complete a purchase directly against. Never handles
    payment or credentials, and never sees whether a purchase completes --
    call report_purchase_feedback_hosted afterward so future searches
    learn from real outcomes. Costs API credits per call."""
    return _commerce.commerce_search(intent, max_price=max_price, category=category, k=k, api_key=API_KEY)


@mcp.tool()
def report_purchase_feedback_hosted(seller_id: int, item_id: str, outcome: str, match_id: str | None = None) -> str:
    """Reports a real outcome after calling a match's checkout_session_url
    directly (this API never sees that transaction). outcome: "purchased",
    "not_purchased", or "irrelevant". A confirmed purchase teaches the
    ranking for future searches with this key; self-reported, not
    independently verified. Pass match_id (from commerce_search_hosted's
    result) so this counts toward your key's real reputation tier -- see
    check_my_reputation_hosted."""
    return _commerce.report_purchase_feedback(seller_id, item_id, outcome, api_key=API_KEY, match_id=match_id)


@mcp.tool()
def report_seller_feedback_hosted(match_id: str, outcome: str, rating: int | None = None) -> str:
    """As a SELLER: reports what really happened for a match_id a buyer
    referenced. outcome: "fulfilled", "buyer_never_completed", or
    "disputed". Only works if your API key owns the match's seller_id.
    This is the independent, seller-side half of the reputation system --
    a buyer's own "purchased" claim alone can't reach "verified" tier,
    only real seller confirmation can."""
    return _commerce.report_seller_feedback(match_id, outcome, rating=rating, api_key=API_KEY)


@mcp.tool()
def check_my_reputation_hosted() -> str:
    """Checks your own API key's real reputation tier ("new", "trusted",
    or "verified") in the commerce trust network, built from real
    buyer-reported purchases and independent seller-confirmed
    fulfillments over time."""
    return _commerce.get_my_reputation(api_key=API_KEY)


@mcp.tool()
def verify_match_hosted(match_id: str) -> str:
    """As a SELLER deciding whether to trust an incoming buyer: checks the
    buyer's reputation tier for one of your own matches, without exposing
    their identity or unrelated transaction history."""
    return _commerce.verify_match(match_id, api_key=API_KEY)


@mcp.tool()
def commerce_network_stats_hosted() -> str:
    """Public aggregate stats for the whole ACP commerce trust network
    (total agents, how many are trusted/verified, total confirmed
    transactions and disputes) -- no API key needed, no individual
    identity exposed."""
    return _commerce.get_network_stats()


def main():
    mcp.run()


if __name__ == "__main__":
    main()
