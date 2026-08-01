# Registry submissions

Real, currently-active MCP/agent-tool directories, verified before listing
here (not guessed). Submission itself needs your own GitHub/account
identity -- I'm not going to open PRs or fill out forms under your name
without you doing that action yourself, same reasoning as not posting to
HN on your behalf. Everything below is the content ready to paste in.

## 1. awesome-mcp-servers (github.com/punkpeye/awesome-mcp-servers)

The largest community-curated MCP list (91.7k stars) -- also auto-indexed
into glama.ai/mcp/servers, so one submission here covers two directories.

**How**: fork the repo, add an entry following its `CONTRIBUTING.md`
format, open a PR.

**Entry content**:
```
- [OBSERVE Search API](https://github.com/gbranaa4-hue/012-trit-search) -
  Pay-per-query semantic code search over curated open source repos
  (React, Django, NumPy, and more). API-key auth, prepaid credits, priced
  under the token cost it saves.
```

## 2. Smithery (smithery.ai)

**How**: `smithery mcp publish "https://api.observe-search.dev" -n <yourorg>/observe-search-mcp`
via the Smithery CLI, or the web dashboard at smithery.ai.

**Metadata**:
- Name: `observe-search-mcp`
- One-sentence description: "Pay-per-query semantic code search over curated open source repos, priced under the tokens it saves."
- Tool count: 3 (`search_code_hosted`, `list_repos_hosted`, `check_balance`)
- Transport: stdio
- GitHub: the observe-api repo (once pushed to a public remote)
- Homepage: the landing page

## 3. mcp.so

Submission form on the site itself (mcp.so) -- same metadata as above.

## 4. Official MCP Registry

Accepts PRs against the registry's server list. Check the current
contribution docs at the time you submit -- registry submission formats
change more often than the community lists above, worth a fresh check
rather than assuming this doc's format guidance is still current.

## 5. LangChain community integrations

python.langchain.com documents community tool integrations -- worth a PR
to their integrations docs once `observe-search-tools` is actually
published to PyPI (the docs typically link an installable package, not a
GitHub-only repo).

## 6. PyPI itself

Publishing `observe-search-tools` to PyPI (`pip install
observe-search-tools`) is its own real discoverability channel --
developers searching PyPI or scanning `requirements.txt`/`pyproject.toml`
dependency graphs for LangChain/CrewAI tools will find it there without
any registry submission at all. This needs a PyPI account (yours) and
`twine upload` or a CI publish step -- not something to skip in favor of
just the registries above.

## Before submitting anywhere

All of these are more valuable once the API is actually live (real
domain, real `/v1/repos` response, a working signup flow) -- submitting
registry entries that 404 or point at an unfinished API does more harm
than good to credibility. Sequence this after deployment (see the main
README's TODO list), not before.
