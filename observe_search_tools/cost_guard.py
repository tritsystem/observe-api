"""
A real, automated cost-saving check -- not just descriptive text in a tool
docstring hoping the calling model reads and follows it (that's what
TOOL_DESCRIPTION in core.py already does, and it's real, but it only works
if the calling model actually complies). This module is the harder
guarantee: both the CLI and the MCP server call looks_like_exact_identifier()
before spending a credit on a search, and refuse by default when it fires.

Grounded in this project's own measured benchmark (see README/landing
page's positioning section): grep wins 5/5 vs. 3/5 when the caller already
knows the exact identifier; semantic search wins on vocabulary-mismatch,
concept-only queries (66% fewer tokens vs. plain search). A query that's a
single bare identifier or a file path is exactly the case grep already
wins -- paying a credit for it is avoidable, not a judgment call.

Deliberately conservative: only fires on UNAMBIGUOUS single-token queries.
Never blocks a real natural-language query, and never blocks outright --
every caller can override with force=True (or --force on the CLI), since
the heuristic can be wrong (a single real word can be a legitimate
concept-only query too, e.g. "authentication").
"""
import re

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PATH_LIKE_RE = re.compile(r"^[\w./\\-]+\.[A-Za-z0-9]{1,5}$")


def looks_like_exact_identifier(query: str) -> bool:
    """True for a single bare identifier (snake_case, CamelCase, a plain
    word with no spaces) or something file-path-shaped -- the case this
    project's own benchmark measured grep winning outright. False for
    anything with a space (a real natural-language description), and
    False for an empty/whitespace-only query (not this function's job to
    validate that)."""
    q = query.strip()
    if not q or " " in q:
        return False
    return bool(_IDENTIFIER_RE.match(q) or _PATH_LIKE_RE.match(q))


COST_GUARD_MESSAGE = (
    "This looks like an exact identifier or file path, not a description -- "
    "grep or a direct file read is faster and free for this (measured 5/5 "
    "vs. 3/5 for semantic search on exact-name queries). Skipped the paid "
    "search call. Pass force=True (CLI: --force) if this really is a "
    "concept-only query, not an exact name."
)
