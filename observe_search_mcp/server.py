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
from mcp.server.fastmcp import FastMCP

API_BASE = os.environ.get("OBSERVE_API_BASE", "https://api.observe-search.dev")
API_KEY = os.environ.get("OBSERVE_API_KEY")

mcp = FastMCP("observe-hosted-search")


@mcp.tool()
def search_code_hosted(query: str, k: int = 10, repo: str | None = None) -> str:
    """Semantic code search over a curated set of popular open source repos
    (React, Django, NumPy, FastAPI, Tokio, and more -- call list_repos_hosted
    for the current list). Costs API credits per call (see check_balance).

    Use this when you can only DESCRIBE what you're looking for, not name
    the exact identifier -- e.g. "where does this handle retrying a failed
    upload." If you already know the exact function/class/file name, grep
    or a direct file read will be faster and free; this tool is for
    vocabulary-mismatch and concept-only queries, not exact-name lookups.

    Args:
        query: natural-language description of what you're looking for.
        k: number of results to return (1-50, default 10).
        repo: optional -- scope the search to one indexed repo (see
            list_repos_hosted). Omit to search across all indexed repos.
    """
    if not API_KEY:
        return "Error: OBSERVE_API_KEY is not set. Get one from POST /v1/signup at " + API_BASE
    try:
        resp = httpx.post(
            f"{API_BASE}/v1/search",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"query": query, "k": k, "repo": repo},
            timeout=30,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"

    if resp.status_code == 402:
        return "Error: insufficient credits. Buy more at the checkout_url from your original /v1/signup response."
    if resp.status_code == 401:
        return "Error: invalid API key. Check OBSERVE_API_KEY."
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"

    data = resp.json()
    if not data["results"]:
        return f"No results found. ({data['credits_remaining']} credits remaining.)"

    lines = [f"{len(data['results'])} result(s), {data['credits_remaining']} credits remaining:\n"]
    for r in data["results"]:
        repo_tag = f"[{r['repo']}] " if r.get("repo") else ""
        lines.append(f"{repo_tag}{r['path']} (score {r['score']:.3f})\n  {r['preview']}\n")
    return "\n".join(lines)


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


def main():
    mcp.run()


if __name__ == "__main__":
    main()
