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
    # Never touch the user's real Obsidian vault from an automated test
    # run -- tests use this mock as a spy (see
    # test_purchased_feedback_archives_to_obsidian) rather than the real
    # filesystem writer.
    monkeypatch.setattr(commerce_router, "obsidian_memory", MagicMock())
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


def test_register_seller_rejects_empty_name(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    resp = client.post(
        "/v1/commerce/sellers",
        json={"name": "   ", "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400


def test_add_listings_rejects_non_positive_unit_amount(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "General Store", "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["seller_id"]
    resp = client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": "sku-1", "name": "Free?!", "description": "suspicious", "unit_amount": 0}]},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400


def test_add_listings_rejects_empty_batch(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "General Store", "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["seller_id"]
    resp = client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": []},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400


def test_add_listings_enforces_per_seller_cap(client, fresh_db, monkeypatch):
    monkeypatch.setattr(commerce_router, "MAX_LISTINGS_PER_SELLER", 1)
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "General Store", "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": "sku-1", "name": "First", "description": "d"}]},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp = client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": "sku-2", "name": "Second", "description": "d"}]},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400
    assert "cap" in resp.json()["detail"]


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


def _make_seller_and_listing(client, api_key, item_id="sku-boots"):
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "General Store", "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": item_id, "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 12000}]},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return seller_id


def test_feedback_requires_auth(client, fresh_db):
    resp = client.post("/v1/commerce/feedback", json={"seller_id": 1, "item_id": "sku-boots", "outcome": "purchased"})
    assert resp.status_code == 401


def test_feedback_rejects_unknown_seller(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    resp = client.post(
        "/v1/commerce/feedback",
        json={"seller_id": 9999, "item_id": "sku-boots", "outcome": "purchased"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 404


def test_feedback_rejects_invalid_outcome(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = _make_seller_and_listing(client, api_key)
    resp = client.post(
        "/v1/commerce/feedback",
        json={"seller_id": seller_id, "item_id": "sku-boots", "outcome": "loved_it"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400


def test_purchased_feedback_reinforces_and_beats_plain_search_boost(client, fresh_db):
    """The actual self-adjustment claim: a real confirmed purchase should
    teach the network MORE than merely appearing in search results does
    (CONFIRMED_PURCHASE_DRIVE > the ordinary top-hit drive)."""
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = _make_seller_and_listing(client, api_key)

    resp = client.post(
        "/v1/commerce/feedback",
        json={"seller_id": seller_id, "item_id": "sku-boots", "outcome": "purchased"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["recorded"] is True
    assert body["reinforced"] is True

    search_after_purchase = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["matches"][0]["memory_boost"]

    # Compare against a second, independent key that only ever searched
    # (never reported a purchase) -- isolates the purchase-feedback
    # effect from ordinary search-driven learning.
    other_key = _signup_and_fund(client, fresh_db, email="other-buyer@example.com")
    client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {other_key}"},
    )
    search_only_boost = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {other_key}"},
    ).json()["matches"][0]["memory_boost"]

    assert search_after_purchase > search_only_boost


def test_not_purchased_feedback_is_recorded_but_not_reinforced(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = _make_seller_and_listing(client, api_key)
    resp = client.post(
        "/v1/commerce/feedback",
        json={"seller_id": seller_id, "item_id": "sku-boots", "outcome": "not_purchased"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["recorded"] is True
    assert body["reinforced"] is False


def test_purchased_feedback_archives_to_obsidian_vault(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = _make_seller_and_listing(client, api_key)

    resp = client.post(
        "/v1/commerce/feedback",
        json={"seller_id": seller_id, "item_id": "sku-boots", "outcome": "purchased"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    commerce_router.obsidian_memory.log_project_work.assert_called_once()
    call_kwargs = commerce_router.obsidian_memory.log_project_work.call_args.kwargs
    assert call_kwargs["project_tag"] == "observe-api-commerce"
    assert "sku-boots" in call_kwargs["output_text"]
    assert "purchased" in call_kwargs["output_text"]


def test_non_purchase_feedback_does_not_write_to_obsidian_vault(client, fresh_db):
    """Scoped deliberately to confirmed purchases only (see
    commerce_router.py's module docstring) -- routine non-purchase
    feedback shouldn't flood the vault with an entry per call."""
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = _make_seller_and_listing(client, api_key)
    client.post(
        "/v1/commerce/feedback",
        json={"seller_id": seller_id, "item_id": "sku-boots", "outcome": "not_purchased"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    commerce_router.obsidian_memory.log_project_work.assert_not_called()


def test_vault_write_failure_does_not_break_the_feedback_response(client, fresh_db):
    """Archiving is a best-effort side effect, never a dependency of this
    endpoint's own correctness (see module docstring + the try/except in
    commerce_feedback())."""
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = _make_seller_and_listing(client, api_key)
    commerce_router.obsidian_memory.log_project_work.side_effect = OSError("disk full")

    resp = client.post(
        "/v1/commerce/feedback",
        json={"seller_id": seller_id, "item_id": "sku-boots", "outcome": "purchased"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    assert resp.json()["reinforced"] is True


def test_learned_memory_survives_a_simulated_process_restart(client, fresh_db):
    """The actual robustness claim: register_commerce_routes() a second
    time on a brand-new FastAPI app (a fresh set of closures/in-memory
    caches, same DB file) simulates a real process restart. Real bug this
    guards against: an earlier version kept ListingAffinityMemory purely
    in-process, so a restart silently discarded all learned affinity."""
    from fastapi import FastAPI as _FastAPI
    from fastapi.testclient import TestClient as _TestClient

    # Two listings, both real matches for the same query -- a learned
    # CONNECTION (the persisted signal) needs a real pair to form
    # between; a single-listing catalog only ever exercises heat (never
    # persisted, by design), so it can't prove persistence actually
    # affects visible ranking.
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "General Store", "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [
            {"item_id": "sku-boots", "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 12000},
            {"item_id": "sku-socks", "name": "Socks", "description": "hiking boot waterproof wool socks", "unit_amount": 1500},
        ]},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    # "Process 1": search several times so both listings keep co-occurring
    # and a real learned connection actually forms between them.
    for _ in range(3):
        client.post("/v1/commerce/search", json={"intent": "waterproof hiking boots"}, headers={"Authorization": f"Bearer {api_key}"})

    # "Process 2": a fresh app, fresh closures/_key_memories cache, same
    # underlying fresh_db -- register_commerce_routes runs its own
    # module-level setup exactly like a real restart would.
    app2 = _FastAPI()
    commerce_router.register_commerce_routes(app2, server.engine, fresh_db, server.rate_limit, server._require_key)
    client2 = _TestClient(app2)

    resp = client2.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.json()["matches"][0]["memory_boost"] > 0.0, "learned affinity should have loaded from the DB, not started cold"
