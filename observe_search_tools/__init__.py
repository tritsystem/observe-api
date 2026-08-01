"""
observe_search_tools -- agent-framework tool wrappers for the hosted
OBSERVE Search API.

Deliberately NOT importing langchain_tool/crewai_tool here: they depend on
optional, mutually-independent framework packages (langchain-core,
crewai), and importing both eagerly would force a LangChain-only user to
also have crewai installed (and vice versa). Import directly from the
submodule you need:

    from observe_search_tools.langchain_tool import observe_search_tool
    from observe_search_tools.crewai_tool import ObserveSearchTool

core.search() itself has no framework dependency (just httpx) and is
always safe to import.
"""
from .core import search  # noqa: F401
