"""
Tests for commerce_router.py's HTTP surface. Never loads the real
embedding model -- server.engine.model is monkeypatched with a small
deterministic fake whose vectors are literal keyword-count features, so
ranking behavior is actually verifiable (a boots-related intent should
outrank a laptop listing) rather than just checking the endpoint returns
200.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import billing  # noqa: E402
import server  # noqa: E402


_VOCAB = ["boot", "hiking", "waterproof", "laptop", "keyboard", "coffee", "bean"]


class _FakeModel:
    """Deterministic bag-of-words encoder over _VOCAB, normalized like a
    real SentenceTransformer call (normalize_embeddings=True) so cosine
    similarity via a plain dot product behaves the same way
    commerce_router.py actually computes it."""

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
    return TestClient(server.app)


def _signup_and_fund(client, fresh_db, email="seller@example.com", credits=50):
    resp = client.post("/v1/signup", json={"email": email})
    api_key = resp.json()["api_key"]
    key_hash = fresh_db.hash_key(api_key)
    current = fresh_db.get_key_record(api_key)["credits"]
    if credits != current:
        fresh_db.add_credits(key_hash, credits - current, f"test_sess_{email}", 0)
    return api_key


def test_register_seller_requires_https(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    resp = client.post(
        "/v1/commerce/sellers",
        json={"name": "Trailhead Outfitters", "checkout_session_url": "http://insecure.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400
    assert "https" in resp.json()["detail"]


def test_register_seller_and_add_listings(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    resp = client.post(
        "/v1/commerce/sellers",
        json={"name": "Trailhead Outfitters", "checkout_session_url": "https://trailhead.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    seller_id = resp.json()["seller_id"]

    resp = client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [
            {"item_id": "sku-1", "name": "Waterproof Hiking Boots", "description": "Rugged waterproof boots for hiking trails", "unit_amount": 12000, "currency": "usd", "category": "footwear"},
            {"item_id": "sku-2", "name": "Mechanical Keyboard", "description": "A laptop-friendly mechanical keyboard", "unit_amount": 8000, "currency": "usd", "category": "electronics"},
        ]},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    assert resp.json()["added"] == 2


def test_listings_scoped_to_owning_key(client, fresh_db):
    seller_key = _signup_and_fund(client, fresh_db, email="seller@example.com")
    other_key = _signup_and_fund(client, fresh_db, email="other@example.com")
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "Trailhead Outfitters", "checkout_session_url": "https://trailhead.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {seller_key}"},
    ).json()["seller_id"]

    resp = client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": "sku-1", "name": "Boots", "description": "boots", "unit_amount": 100}]},
        headers={"Authorization": f"Bearer {other_key}"},
    )
    assert resp.status_code == 404


def test_search_ranks_by_real_semantic_relevance_not_insertion_order(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "General Store", "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["seller_id"]

    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [
            {"item_id": "sku-laptop", "name": "Laptop Keyboard", "description": "keyboard for laptop", "unit_amount": 8000, "category": "electronics"},
            {"item_id": "sku-boots", "name": "Waterproof Hiking Boots", "description": "boot for hiking waterproof trail", "unit_amount": 12000, "category": "footwear"},
        ]},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    resp = client.post(
        "/v1/commerce/search",
        json={"intent": "I need waterproof boots for hiking", "k": 5},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["matches"]) == 2
    # The boots listing shares real vocabulary with the intent; the
    # keyboard listing shares none -- if this ever regresses to
    # insertion-order or an unnormalized/broken similarity computation,
    # this ordering assertion catches it.
    assert body["matches"][0]["item_id"] == "sku-boots"
    assert body["matches"][0]["score"] > body["matches"][1]["score"]
    assert "checkout_session_url" in body["matches"][0]
    assert body["matches"][0]["checkout_session_url"] == "https://store.example.com/checkout_sessions"


def test_search_respects_max_price_filter(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "General Store", "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [
            {"item_id": "cheap-boots", "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 5000},
            {"item_id": "pricey-boots", "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 50000},
        ]},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp = client.post(
        "/v1/commerce/search",
        json={"intent": "waterproof hiking boots", "max_price": 10000},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    matches = resp.json()["matches"]
    assert len(matches) == 1
    assert matches[0]["item_id"] == "cheap-boots"


def test_search_charges_a_credit_and_refuses_at_zero_balance(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=1)
    resp1 = client.post("/v1/commerce/search", json={"intent": "anything"}, headers={"Authorization": f"Bearer {api_key}"})
    assert resp1.status_code == 200
    assert resp1.json()["credits_remaining"] == 0

    resp2 = client.post("/v1/commerce/search", json={"intent": "anything"}, headers={"Authorization": f"Bearer {api_key}"})
    assert resp2.status_code == 402


def test_search_requires_auth(client, fresh_db):
    resp = client.post("/v1/commerce/search", json={"intent": "anything"})
    assert resp.status_code == 401


def test_repeated_searches_wire_real_spiking_memory_end_to_end(client, fresh_db):
    """Not a unit test of commerce_spiking_memory.py itself (see
    tests/test_commerce_spiking_memory.py for that, against the real
    Spikeling engine) -- this verifies the actual HTTP-level wiring:
    memory_boost starts at 0.0 (nothing learned yet) and becomes nonzero
    after a listing has genuinely been searched-and-returned before."""
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "General Store", "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": "sku-boots", "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 12000}]},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    resp1 = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp1.json()["matches"][0]["memory_boost"] == 0.0

    resp2 = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp2.json()["matches"][0]["memory_boost"] > 0.0
