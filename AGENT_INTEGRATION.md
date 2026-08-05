# Wiring OBSERVE into an agent's toolset

Four ways to give an agent the motive to call OBSERVE, in order of how
much of that motive you get for free. As of 2026-08-05, the motive isn't
just a tool description an agent might ignore -- `core.search()` (the one
shared client every path below calls into) refuses an exact-identifier
or file-path-shaped query BEFORE it ever reaches the network, for every
integration path at once, not duplicated per-wrapper. `force=True` (or
`--force` on the CLI) overrides it.

## 1. MCP (Claude Code, Claude Desktop, Cursor) -- motive comes for free

Once `observe-search-mcp` is registered (see
[`observe_search_mcp/README.md`](observe_search_mcp/README.md)), the agent
reads the tool's own description on every turn -- no system prompt wiring
needed:

> "Semantic code search over a curated set of popular open source repos...
> Use this when you can only DESCRIBE what you're looking for, not name the
> exact identifier... If you already know the exact function/class/file
> name, grep or a direct file read will be faster and free."

That last sentence matters as much as the first: it's what stops the agent
from calling this API (and spending credits) on queries grep already wins.
Verified real vs. imagined -- see the main README's grep-vs-OBSERVE
benchmark for where each actually wins.

**If you want to reinforce it anyway** (e.g. your project's system prompt
already talks about tool selection), one line is enough:

```
When you need to find code by describing its behavior rather than its
name, use observe_search_hosted before falling back to a broad grep.
```

## 2. LangChain / CrewAI -- same free motive, different wiring

Both wrap the identical `TOOL_DESCRIPTION` from
[`observe_search_tools/core.py`](observe_search_tools/core.py), so the
agent gets the same "describe, don't name" guidance automatically. The
actual integration is genuinely this small:

```python
# LangChain
from observe_search_tools.langchain_tool import observe_search_tool
agent = create_react_agent(llm, [observe_search_tool, ...])
```

```python
# CrewAI
from observe_search_tools.crewai_tool import ObserveSearchTool
agent = Agent(..., tools=[ObserveSearchTool()])
```

`pip install observe-search-tools`, set `OBSERVE_API_KEY`, done -- no
description to write, no prompt engineering, the tool carries its own
motive with it.

## 3. CLI (shell-based agents, CI scripts, anything that can exec a command)

For an agent harness that shells out rather than speaking MCP or a
Python framework directly -- the `observe` command, installed by the
same `observe-search-tools` package:

```bash
pip install observe-search-tools
export OBSERVE_API_KEY=obs_...
observe search "where does this handle retrying a failed upload"
observe search "retryUpload"              # refused by the cost guard, no credit spent
observe search "retryUpload" --force      # explicit override
observe commerce-search "waterproof boots for a muddy trail"
```

Same `core.py`/`commerce.py` client as every other path above, so
behavior (including the cost guard) is identical -- this isn't a fourth
reimplementation with its own bugs, it's the same one wrapped in
`argparse` instead of a framework decorator. `observe --help` lists
every subcommand, including the commerce ones
(`commerce-register-seller`, `commerce-add-listings`,
`commerce-feedback`).

## 4. Raw HTTP / a custom agent framework -- you write the motive

If you're not using MCP or one of the two wrapped frameworks, there's no
tool description doing this for you -- your agent's system prompt has to
carry the same guidance directly. A paragraph that's worked in the
frameworks above, adapted to prose:

```
You have access to a semantic code search API (POST /v1/search, curated
open source repos -- React, Django, NumPy, FastAPI, Tokio, and more).
Reach for it when you can only describe what you're looking for, not name
it exactly -- e.g. "where does this retry a failed upload" or "how does
this library handle timezone conversion." If you already know the exact
function, class, or file name, use grep or a direct file read instead --
it's faster, free, and this API costs credits per call. Full API shape:
https://github.com/gbranaa4-hue/observe-api#api-shape
```

Minimal wiring, no SDK:

```python
import httpx

def observe_search(query: str, k: int = 10, api_key: str = None) -> dict:
    return httpx.post(
        "https://api.observe-search.online/v1/search",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "k": k},
        timeout=30,
    ).json()
```

Register `observe_search` as a tool with whatever schema your framework
expects, paste the system-prompt paragraph above (or your own version of
it) into the agent's instructions, and the motive exists the same way it
does in the MCP/LangChain/CrewAI paths -- just written explicitly instead
of inherited from a tool description.

## The one principle underneath all four paths

The motive isn't "search more" -- it's "search when a name-based lookup
would fail." Every wiring above (MCP, LangChain/CrewAI, CLI, raw HTTP) says the same thing: grep/file-read first
if you know the identifier, this API when you only know the behavior.
Skipping that distinction in your own system prompt is the single most
likely way to turn this into an expensive, slower grep replacement instead
of what it's actually for.
