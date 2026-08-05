# observe-search-mcp

MCP client for the hosted [OBSERVE Search API](https://api.observe-search.online)
-- pay-per-query semantic code search over curated open source repos
(React, Django, NumPy, FastAPI, Tokio, and more), for AI agents.

This is a thin HTTP client, not a local search engine -- it calls the
hosted API, which does the actual embedding/search work. No torch, no
FAISS, no local model download; this package is tiny.

## Setup

1. Get an API key (free to create, prepaid credits required to search):
   ```
   curl -X POST https://api.observe-search.online/v1/signup -d '{"email":"you@example.com"}' -H 'Content-Type: application/json'
   ```
   Save the returned `api_key` -- it's shown once. Follow `checkout_url` to
   buy credits.

2. Install and register:
   ```
   pip install observe-search-mcp
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

- `search_code_hosted(query, k=10, repo=None)` -- semantic search. Use for
  "describe what you want" queries, not exact-name lookups (grep still
  wins there -- see the tool's own description for the full guidance an
  agent reads).
- `list_repos_hosted()` -- which repos are currently indexed.
- `check_balance()` -- remaining credit balance for the configured key.

## Pricing

Prepaid credits, priced to clearly undercut the token cost a search saves
(OBSERVE's own benchmark: ~66% fewer tokens than plain search on the same
queries) -- see the main API's README for the full reasoning.
