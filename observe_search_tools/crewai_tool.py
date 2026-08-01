"""
CrewAI tool wrapper for the hosted OBSERVE Search API.

Usage:
    from observe_search_tools.crewai_tool import ObserveSearchTool
    agent = Agent(..., tools=[ObserveSearchTool()])
"""
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .core import search, TOOL_DESCRIPTION


class ObserveSearchArgs(BaseModel):
    query: str = Field(description="Natural-language description of what you're looking for.")
    k: int = Field(default=10, description="Number of results to return (1-50).")
    repo: Optional[str] = Field(default=None, description="Optional: scope to one indexed repo. Omit to search all.")


class ObserveSearchTool(BaseTool):
    name: str = "observe_search_hosted"
    description: str = TOOL_DESCRIPTION
    args_schema: Type[BaseModel] = ObserveSearchArgs

    def _run(self, query: str, k: int = 10, repo: Optional[str] = None) -> str:
        return search(query, k=k, repo=repo)
