"""
LangChain tool wrapper for the hosted OBSERVE Search API.

Usage:
    from observe_search_tools.langchain_tool import observe_search_tool
    agent = create_react_agent(llm, [observe_search_tool, ...])
"""
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .core import search, TOOL_DESCRIPTION


class ObserveSearchInput(BaseModel):
    query: str = Field(description="Natural-language description of what you're looking for.")
    k: int = Field(default=10, description="Number of results to return (1-50).")
    repo: Optional[str] = Field(default=None, description="Optional: scope to one indexed repo. Omit to search all.")


def _run(query: str, k: int = 10, repo: Optional[str] = None) -> str:
    return search(query, k=k, repo=repo)


observe_search_tool = StructuredTool.from_function(
    func=_run,
    name="observe_search_hosted",
    description=TOOL_DESCRIPTION,
    args_schema=ObserveSearchInput,
)
