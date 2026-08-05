"""
Tests for observe_search_mcp/server.py. Requires the real `mcp` SDK
package to be installed -- this is what actually caught the real bug
(mcp 2.0.0 renamed FastMCP -> MCPServer, moved modules) that no amount
of reading the code would have found, since it only breaks at import
time against a real installed SDK.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

mcp_pkg = pytest.importorskip("mcp", reason="mcp SDK not installed in this environment")

import observe_search_mcp.server as server  # noqa: E402


def test_all_tools_actually_registered_with_the_real_server():
    """Not just 'the function exists in the module' -- confirms the
    decorator actually registered each tool with the live server
    instance, the thing an MCP client actually queries."""
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "search_code_hosted", "list_repos_hosted", "check_balance",
        "register_seller_hosted", "add_listings_hosted",
        "commerce_search_hosted", "report_purchase_feedback_hosted",
        "report_seller_feedback_hosted", "check_my_reputation_hosted",
        "verify_match_hosted", "commerce_network_stats_hosted",
    }


def test_cost_guard_blocks_identifier_shaped_query_without_any_network_call():
    """The real point of the cost guard: an exact-identifier query never
    even reaches httpx.post, so it can never spend a credit -- verified
    by patching httpx.post to raise if called at all."""
    with patch("httpx.post", side_effect=AssertionError("should never be called")):
        result = asyncio.run(server.mcp.call_tool("search_code_hosted", {"query": "retryUpload"}))
    text = result.structured_content["result"]
    assert "grep" in text.lower()
    assert "skipped" in text.lower()


def test_cost_guard_does_not_block_a_real_natural_language_query(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "obs_test")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"results": [], "credits_remaining": 3}
    with patch("httpx.post", return_value=fake_resp) as mock_post:
        result = asyncio.run(server.mcp.call_tool(
            "search_code_hosted", {"query": "where does this handle retrying a failed upload"}
        ))
    mock_post.assert_called_once()
    assert "3 credits remaining" in result.structured_content["result"]


def test_force_bypasses_the_cost_guard(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "obs_test")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"results": [], "credits_remaining": 3}
    with patch("httpx.post", return_value=fake_resp) as mock_post:
        asyncio.run(server.mcp.call_tool("search_code_hosted", {"query": "retryUpload", "force": True}))
    mock_post.assert_called_once()
