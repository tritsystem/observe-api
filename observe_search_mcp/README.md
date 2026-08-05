# observe-search-mcp

MCP client for the hosted [OBSERVE Search API](https://api.observe-search.online)
-- pay-per-query semantic code search over curated open source repos,
plus ACP-compatible agentic commerce discovery, for AI agents.

This is a thin HTTP client, not a local search engine or a payment
processor -- it calls the hosted API, which does the actual
embedding/search work and never sees payment credentials. No torch, no
FAISS, no local model download; this package is tiny.

## Setup

1. Get an API key (free to create, prepaid credits required to search):
   ```
   curl -X POST https://api.observe-search.online/v1/signup -d '{"email":"you@example.com"}' -H 'Content-Type: application/json'
   ```
   Save the returned `api_key` -- it's shown once. Follow `checkout_url` to
   buy credits.

2. Install and register (the PyPI package is `observe-search-tools`; the
   installed command is `observe-search-mcp`):
   ```
   pip install observe-search-tools[mcp]
   export OBSERVE_API_KEY=obs_...
   claude mcp add observe-hosted -- observe-search-mcp
   ```
   Or add directly to a `.mcp.json` / client config:
   ```json
   {
     "mcpServers": {
       "observe-hosted": {
         "command": "observe-search-mcp",
         "env": { "OBSERVE_API_KEY": "obs_..." }
       }
     }
   }
   ```

## Tools

- `search_code_hosted(query, k=10, repo=None, force=False)` -- semantic
  code search. A real automated cost guard checks the query before
  spending a credit: an exact-identifier or file-path-shaped query (the
  case grep already wins, measured 5/5 vs. 3/5) is refused by default --
  pass `force=True` if it's genuinely a concept-only query despite
  looking like one token.
- `list_repos_hosted()` -- which repos are currently indexed.
- `check_balance()` -- remaining credit balance for the configured key.
- `register_seller_hosted(name, checkout_session_url)` -- register as a
  seller in the ACP-compatible commerce discovery layer. Free.
- `add_listings_hosted(seller_id, listings)` -- add product listings.
  Free.
- `commerce_search_hosted(intent, max_price=None, category=None, k=10)`
  -- buyer-side discovery: describe what you want, get back real
  sellers' `checkout_session_url` + `item_id` to complete a purchase
  directly against. Costs credits.
- `report_purchase_feedback_hosted(seller_id, item_id, outcome)` --
  report a real outcome after calling a match's checkout endpoint
  directly. Teaches future rankings for your key.

## A real compatibility note

The `mcp` SDK renamed its high-level server class between major
versions (`mcp.server.fastmcp.FastMCP` in the 1.x line,
`mcp.server.mcpserver.MCPServer` in 2.0.0) -- found by actually
installing and running this package, not assumed. `server.py` tries
both imports, so it works against whichever `mcp` version is actually
installed.

## Not just MCP

The same client core also ships a CLI (`observe` command) and
LangChain/CrewAI tool wrappers -- see the main README.md for
all of them; they share one HTTP client so behavior is identical
across every integration path, not four separate implementations.

## Pricing

Prepaid credits, priced to clearly undercut the token cost a search saves
(OBSERVE's own benchmark: ~66% fewer tokens than plain search on the same
queries). Commerce search is priced separately (its own env var,
`OBSERVE_CREDITS_PER_COMMERCE_SEARCH`, decoupled from code-search
pricing) -- see the main API's README for the full reasoning and the
real cost comparison against Algolia's published pricing.
