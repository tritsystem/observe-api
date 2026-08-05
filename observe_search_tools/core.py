"""
Framework-agnostic HTTP client for the hosted OBSERVE Search API -- shared
by the LangChain and CrewAI tool wrappers so neither reimplements the same
request/error-handling logic.
"""
import os
from typing import Optional

import httpx

API_BASE = os.environ.get("OBSERVE_API_BASE", "https://api.observe-search.online")


def search(query: str, k: int = 10, repo: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """Calls POST /v1/search and returns a human/LLM-readable string --
    formatted output (not raw JSON) since both LangChain and CrewAI tools
    are meant to return text an agent reads directly, not a structure it
    has to parse itself."""
    key = api_key or os.environ.get("OBSERVE_API_KEY")
    if not key:
        return "Error: no API key. Pass api_key= or set OBSERVE_API_KEY. Get one from POST /v1/signup at " + API_BASE

    try:
        resp = httpx.post(
            f"{API_BASE}/v1/search",
            headers={"Authorization": f"Bearer {key}"},
            json={"query": query, "k": k, "repo": repo},
            timeout=30,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"

    if resp.status_code == 402:
        return "Error: insufficient credits. Buy more via the checkout_url from your original signup response."
    if resp.status_code == 401:
        return "Error: invalid API key."
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


def private_search(query: str, k: int = 10, api_key: Optional[str] = None) -> str:
    """Calls POST /v1/private/search -- the caller's OWN indexed repo, not
    the shared curated corpus search() hits. Requires a private index
    already built for this key via POST /v1/private/index (see
    tenant_index.py); this function only searches, it doesn't index."""
    key = api_key or os.environ.get("OBSERVE_API_KEY")
    if not key:
        return "Error: no API key. Pass api_key= or set OBSERVE_API_KEY."

    try:
        resp = httpx.post(
            f"{API_BASE}/v1/private/search",
            headers={"Authorization": f"Bearer {key}"},
            json={"query": query, "k": k},
            timeout=30,
        )
    except httpx.RequestError as e:
        return f"Error: could not reach {API_BASE}: {e}"

    if resp.status_code == 404:
        return "Error: no ready private index for this key. POST /v1/private/index first."
    if resp.status_code == 402:
        return "Error: insufficient credits. Buy more via the checkout_url from your original signup response."
    if resp.status_code == 401:
        return "Error: invalid API key."
    if resp.status_code != 200:
        return f"Error: API returned {resp.status_code}: {resp.text}"

    data = resp.json()
    if not data["results"]:
        return f"No results found. ({data['credits_remaining']} credits remaining.)"

    lines = [f"{len(data['results'])} result(s), {data['credits_remaining']} credits remaining:\n"]
    for r in data["results"]:
        lines.append(f"{r['path']} (score {r['score']:.3f})\n  {r['preview']}\n")
    return "\n".join(lines)


TOOL_DESCRIPTION = (
    "Semantic code search over a curated set of popular open source repos "
    "(React, Django, NumPy, FastAPI, Tokio, and more). Use this when you can "
    "only DESCRIBE what you're looking for, not name the exact identifier -- "
    "e.g. 'where does this handle retrying a failed upload'. If you already "
    "know the exact function/class/file name, grep or a direct file read "
    "will be faster and free; this tool is for vocabulary-mismatch and "
    "concept-only queries. Costs API credits per call."
)
