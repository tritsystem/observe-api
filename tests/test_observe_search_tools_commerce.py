"""
Tests for observe_search_tools/commerce.py -- mocks httpx.post to verify
correct request construction and response formatting, matching how the
underlying /v1/commerce/* endpoints were already verified for real
end-to-end (this session's live server testing) -- this layer's own job
is just "does the thin client build the right request and format the
right response," not re-proving the API itself works.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from observe_search_tools import commerce  # noqa: E402


def _fake_response(status_code, json_data, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = text
    return resp


def test_register_seller_without_api_key_returns_error_not_exception():
    result = commerce.register_seller("Store", "https://x.example.com/checkout_sessions", api_key=None)
    assert "Error" in result
    assert "API key" in result


def test_register_seller_success():
    with patch("httpx.post", return_value=_fake_response(200, {"seller_id": 42})) as mock_post:
        result = commerce.register_seller("Store", "https://x.example.com/checkout_sessions", api_key="obs_test")
    assert "42" in result
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer obs_test"
    assert call_kwargs["json"]["checkout_session_url"] == "https://x.example.com/checkout_sessions"


def test_commerce_search_formats_real_matches():
    fake_body = {
        "matches": [{
            "seller_name": "Trailhead", "name": "Boots", "unit_amount": 12000, "currency": "usd",
            "score": 0.933, "checkout_session_url": "https://x.example.com/checkout_sessions", "item_id": "sku-1",
        }],
        "credits_remaining": 42,
    }
    with patch("httpx.post", return_value=_fake_response(200, fake_body)):
        result = commerce.commerce_search("waterproof boots", api_key="obs_test")
    assert "Trailhead" in result
    assert "sku-1" in result
    assert "120.00 usd" in result
    assert "42 credits remaining" in result


def test_commerce_search_empty_matches():
    with patch("httpx.post", return_value=_fake_response(200, {"matches": [], "credits_remaining": 5})):
        result = commerce.commerce_search("anything", api_key="obs_test")
    assert "No matches" in result
    assert "5 credits remaining" in result


def test_commerce_search_insufficient_credits():
    with patch("httpx.post", return_value=_fake_response(402, {})):
        result = commerce.commerce_search("anything", api_key="obs_test")
    assert "insufficient credits" in result.lower()


def test_report_purchase_feedback_returns_the_real_note():
    with patch("httpx.post", return_value=_fake_response(200, {"note": "Recorded and reinforced -- real ground truth."})):
        result = commerce.report_purchase_feedback(1, "sku-1", "purchased", api_key="obs_test")
    assert "Recorded and reinforced" in result
