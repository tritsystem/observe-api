"""
Tests for ucp_adapter.py -- verifies the manifest and catalog search are
actually shaped like the real UCP schemas (fetched from
github.com/Universal-Commerce-Protocol/ucp, not guessed), and that the
UCP endpoint returns the SAME underlying data as /v1/commerce/search
(proving it reuses commerce_router.py's real search logic, not a
second, potentially-drifted implementation).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import billing  # noqa: E402
import commerce_router  # noqa: E402
import server  # noqa: E402


_VOCAB = ["boot", "hiking", "waterproof", "laptop", "keyboard", "coffee", "bean"]


class _FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        vecs = []
        for t in texts:
            low = t.lower()
            v = np.array([low.count(w) for w in _VOCAB], dtype="float32")
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.array(vecs, dtype="float32")


@pytest.fixture
def client(fresh_db, monkeypatch):
    monkeypatch.setattr(server, "db", fresh_db)
    monkeypatch.setattr(server.engine, "model", _FakeModel())
    monkeypatch.setattr(server.engine, "ready", True)
    monkeypatch.setattr(billing, "create_checkout_session", lambda email, key_hash: "https://checkout.stripe.com/fake-session")
    monkeypatch.setattr(commerce_router, "obsidian_memory", MagicMock())
    return TestClient(server.app)


def _signup_and_fund(client, fresh_db, email="buyer@example.com", credits=50):
    resp = client.post("/v1/signup", json={"email": email})
    api_key = resp.json()["api_key"]
    key_hash = fresh_db.hash_key(api_key)
    current = fresh_db.get_key_record(api_key)["credits"]
    if credits != current:
        fresh_db.add_credits(key_hash, credits - current, f"test_sess_{email}", 0)
    return api_key


def _register_seller_and_listing(client, api_key):
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "Trailhead Outfitters", "checkout_session_url": "https://trailhead.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": "sku-boots", "name": "Waterproof Hiking Boots", "description": "hiking boot waterproof", "unit_amount": 12000, "currency": "usd"}]},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return seller_id


def test_manifest_matches_the_real_ucp_business_schema_required_fields(client, fresh_db):
    resp = client.get("/.well-known/ucp")
    assert resp.status_code == 200
    body = resp.json()
    ucp = body["ucp"]
    # business_schema (source/schemas/ucp.json) requires these two keys to exist
    assert "services" in ucp
    assert "payment_handlers" in ucp
    assert ucp["payment_handlers"] == {}, "OBSERVE never handles payment -- must be empty, not populated"
    service = ucp["services"]["dev.ucp.shopping"][0]
    assert service["transport"] == "rest"
    assert service["endpoint"].endswith("/ucp")
    capability = ucp["capabilities"]["dev.ucp.shopping.catalog.search"][0]
    assert "schema" in capability


def test_catalog_search_requires_auth(client, fresh_db):
    resp = client.post("/ucp/catalog/search", json={"query": "waterproof boots"})
    assert resp.status_code == 401


def test_catalog_search_accepts_x_api_key_header(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    _register_seller_and_listing(client, api_key)
    resp = client.post("/ucp/catalog/search", json={"query": "waterproof hiking boots"}, headers={"X-Api-Key": api_key})
    assert resp.status_code == 200


def test_catalog_search_returns_real_ucp_product_shape(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    _register_seller_and_listing(client, api_key)
    resp = client.post(
        "/ucp/catalog/search", json={"query": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ucp"]["version"]
    assert body["ucp"]["status"] == "success"
    products = body["products"]
    assert len(products) == 1
    p = products[0]
    # Real UCP Product schema (source/schemas/shopping/types/product.json)
    # required fields: id, title, description, price_range, variants
    assert p["id"]
    assert p["title"] == "Waterproof Hiking Boots"
    assert p["description"]["plain"]
    assert p["price_range"]["min"]["amount"] == 12000
    assert p["price_range"]["min"]["currency"] == "USD"
    assert len(p["variants"]) == 1
    v = p["variants"][0]
    assert v["price"]["amount"] == 12000
    assert v["price"]["currency"] == "USD"
    assert v["seller"]["name"] == "Trailhead Outfitters"


def test_catalog_search_deducts_the_same_credits_as_commerce_search(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=10)
    _register_seller_and_listing(client, api_key)
    client.post("/ucp/catalog/search", json={"query": "waterproof boots"}, headers={"Authorization": f"Bearer {api_key}"})
    balance = client.get("/v1/balance", headers={"Authorization": f"Bearer {api_key}"}).json()["credits"]
    assert balance == 9


def test_catalog_search_reuses_the_same_underlying_search_as_v1_commerce_search(client, fresh_db):
    """The actual architectural claim: UCP and the REST API return the
    SAME listing for the SAME query, proving one shared search
    implementation, not two that could drift."""
    api_key = _signup_and_fund(client, fresh_db, credits=20)
    _register_seller_and_listing(client, api_key)

    rest_result = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()
    ucp_result = client.post(
        "/ucp/catalog/search", json={"query": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()

    assert rest_result["matches"][0]["item_id"] == "sku-boots"
    assert ucp_result["products"][0]["id"].endswith(":sku-boots")


def test_catalog_search_respects_max_price_filter(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "Store", "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [
            {"item_id": "cheap", "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 5000},
            {"item_id": "pricey", "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 50000},
        ]},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp = client.post(
        "/ucp/catalog/search",
        json={"query": "waterproof hiking boots", "filters": {"max_price": 10000}},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    products = resp.json()["products"]
    assert len(products) == 1
    assert products[0]["id"].endswith(":cheap")
