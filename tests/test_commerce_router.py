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


def test_purchase_after_a_real_search_still_reinforces_not_resets(client, fresh_db):
    """Reproduces a real bug found by testing this live against the
    production server, not caught by the test above: that test fires
    feedback BEFORE ever searching, so the neuron starts at 0 heat and
    a fixed drive safely lands below threshold. The realistic order is
    the opposite -- a buyer searches, THEN buys -- which leaves the
    purchased item with real residual heat already on it. Adding a
    fixed CONFIRMED_PURCHASE_DRIVE on top of that residual heat pushed
    the neuron over threshold, causing a real LIF fire+reset that
    erased its own reinforcement (observed live: memory_boost read 0.0
    for the just-purchased item, the opposite of intended). Fixed by
    computing the actual safe drive against current state."""
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = _make_seller_and_listing(client, api_key)

    # Realistic order: search first (creates real residual heat on
    # sku-boots), THEN report the purchase.
    client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp = client.post(
        "/v1/commerce/feedback",
        json={"seller_id": seller_id, "item_id": "sku-boots", "outcome": "purchased"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    assert resp.json()["reinforced"] is True

    boost_after_purchase = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["matches"][0]["memory_boost"]

    assert boost_after_purchase > 0.0, "a real confirmed purchase must never read as zero boost, even with residual heat already on the neuron"


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


def test_search_returns_a_real_match_id(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    _make_seller_and_listing(client, api_key)
    resp = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    matches = resp.json()["matches"]
    assert len(matches) == 1
    assert matches[0]["match_id"]
    assert len(matches[0]["match_id"]) == 36  # real uuid4 string length


def test_feedback_with_match_id_from_a_different_key_is_rejected(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    _make_seller_and_listing(client, api_key)
    match_id = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["matches"][0]["match_id"]

    other_key = _signup_and_fund(client, fresh_db, email="other@example.com")
    resp = client.post(
        "/v1/commerce/feedback",
        json={"seller_id": 1, "item_id": "sku-boots", "outcome": "purchased", "match_id": match_id},
        headers={"Authorization": f"Bearer {other_key}"},
    )
    assert resp.status_code == 404


def test_seller_feedback_requires_owning_the_seller(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    seller_id = _make_seller_and_listing(client, api_key)
    match_id = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["matches"][0]["match_id"]

    not_the_seller = _signup_and_fund(client, fresh_db, email="not-the-seller@example.com")
    resp = client.post(
        "/v1/commerce/seller-feedback",
        json={"match_id": match_id, "outcome": "fulfilled"},
        headers={"Authorization": f"Bearer {not_the_seller}"},
    )
    assert resp.status_code == 403


def test_seller_feedback_rejects_bad_rating(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    _make_seller_and_listing(client, api_key)
    match_id = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["matches"][0]["match_id"]
    resp = client.post(
        "/v1/commerce/seller-feedback",
        json={"match_id": match_id, "outcome": "fulfilled", "rating": 9},
        headers={"Authorization": f"Bearer {api_key}"},  # this key IS the seller too, registered via _make_seller_and_listing
    )
    assert resp.status_code == 400


def test_reputation_starts_at_new_tier(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db)
    resp = client.get("/v1/commerce/my-reputation", headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tier"] == "new"
    assert body["total_matches"] == 0


def _register_seller_with_boots(client, seller_key, name):
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": name, "checkout_session_url": "https://store.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {seller_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": "sku-1", "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 12000}]},
        headers={"Authorization": f"Bearer {seller_key}"},
    )
    return seller_id


def _confirm_fulfillment(client, buyer_key, seller_key, seller_id):
    # Two sellers in the same test can have near-identical listings (both
    # sell "boots"), so don't assume matches[0] is the intended seller --
    # pick the match that actually belongs to it, same as a real buyer
    # agent would need to.
    matches = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots", "k": 10},
        headers={"Authorization": f"Bearer {buyer_key}"},
    ).json()["matches"]
    match_id = next(m["match_id"] for m in matches if m["seller_id"] == seller_id)
    resp = client.post(
        "/v1/commerce/seller-feedback",
        json={"match_id": match_id, "outcome": "fulfilled", "rating": 5},
        headers={"Authorization": f"Bearer {seller_key}"},
    )
    assert resp.status_code == 200


def test_reputation_reaches_verified_after_real_seller_confirmed_fulfillments(client, fresh_db):
    """The actual point of the two-sided system: a buyer's own
    self-reported 'purchased' claims alone should NOT be enough for
    verified -- it takes independent SELLERS confirming real fulfillments,
    spanning more than one seller (see
    test_single_seller_repeated_fulfillments_cannot_reach_verified_alone
    for why one seller alone isn't enough, even confirmed many times)."""
    buyer_key = _signup_and_fund(client, fresh_db, email="buyer@example.com")
    seller_a_key = _signup_and_fund(client, fresh_db, email="seller-a@example.com")
    seller_b_key = _signup_and_fund(client, fresh_db, email="seller-b@example.com")
    seller_a_id = _register_seller_with_boots(client, seller_a_key, "Trusted Store A")
    seller_b_id = _register_seller_with_boots(client, seller_b_key, "Trusted Store B")

    for _ in range(3):
        _confirm_fulfillment(client, buyer_key, seller_a_key, seller_a_id)
    for _ in range(2):
        _confirm_fulfillment(client, buyer_key, seller_b_key, seller_b_id)

    rep = client.get("/v1/commerce/my-reputation", headers={"Authorization": f"Bearer {buyer_key}"}).json()
    assert rep["tier"] == "verified"
    assert rep["seller_confirmed_fulfillments"] == 5
    assert rep["distinct_sellers_confirmed"] == 2


def test_single_seller_repeated_fulfillments_cannot_reach_verified_alone(client, fresh_db):
    """Collusion-resistance regression: one buyer key and one seller key
    (which could be the same real person controlling both) confirming
    fulfillments against EACH OTHER, no matter how many times, must not
    be enough to manufacture 'verified' -- that requires
    VERIFIED_MIN_DISTINCT_SELLERS distinct sellers, not just
    VERIFIED_MIN_SELLER_CONFIRMED confirmations from one relationship."""
    buyer_key = _signup_and_fund(client, fresh_db, email="colluding-buyer@example.com")
    seller_key = _signup_and_fund(client, fresh_db, email="colluding-seller@example.com")
    seller_id = _register_seller_with_boots(client, seller_key, "Solo Seller")

    for _ in range(10):  # well past VERIFIED_MIN_SELLER_CONFIRMED, still just one seller
        _confirm_fulfillment(client, buyer_key, seller_key, seller_id)

    rep = client.get("/v1/commerce/my-reputation", headers={"Authorization": f"Bearer {buyer_key}"}).json()
    assert rep["seller_confirmed_fulfillments"] == 10
    assert rep["distinct_sellers_confirmed"] == 1
    assert rep["tier"] == "trusted"  # real signal, just not the strongest one
    assert rep["tier"] != "verified"


def test_a_single_dispute_resets_tier_to_new(client, fresh_db):
    buyer_key = _signup_and_fund(client, fresh_db, email="buyer2@example.com")
    seller_key = _signup_and_fund(client, fresh_db, email="seller-owner2@example.com")
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "Store", "checkout_session_url": "https://store2.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {seller_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": "sku-1", "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 12000}]},
        headers={"Authorization": f"Bearer {seller_key}"},
    )
    for outcome in ["fulfilled", "fulfilled", "fulfilled", "fulfilled", "disputed"]:
        match_id = client.post(
            "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
            headers={"Authorization": f"Bearer {buyer_key}"},
        ).json()["matches"][0]["match_id"]
        client.post(
            "/v1/commerce/seller-feedback",
            json={"match_id": match_id, "outcome": outcome},
            headers={"Authorization": f"Bearer {seller_key}"},
        )
    rep = client.get("/v1/commerce/my-reputation", headers={"Authorization": f"Bearer {buyer_key}"}).json()
    assert rep["tier"] == "new"
    assert rep["disputes"] == 1


def test_verify_match_hides_buyer_identity_shows_only_tier(client, fresh_db):
    buyer_key = _signup_and_fund(client, fresh_db, email="buyer3@example.com")
    seller_key = _signup_and_fund(client, fresh_db, email="seller-owner3@example.com")
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "Store", "checkout_session_url": "https://store3.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {seller_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": "sku-1", "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 12000}]},
        headers={"Authorization": f"Bearer {seller_key}"},
    )
    match_id = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {buyer_key}"},
    ).json()["matches"][0]["match_id"]

    resp = client.get(f"/v1/commerce/verify-match?match_id={match_id}", headers={"Authorization": f"Bearer {seller_key}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["tier"] == "new"
    assert "buyer_key_hash" not in body
    assert "key_hash" not in body


def test_verify_match_rejects_a_seller_who_does_not_own_it(client, fresh_db):
    buyer_key = _signup_and_fund(client, fresh_db, email="buyer4@example.com")
    seller_key = _signup_and_fund(client, fresh_db, email="seller-owner4@example.com")
    other_seller_key = _signup_and_fund(client, fresh_db, email="other-seller@example.com")
    seller_id = client.post(
        "/v1/commerce/sellers",
        json={"name": "Store", "checkout_session_url": "https://store4.example.com/checkout_sessions"},
        headers={"Authorization": f"Bearer {seller_key}"},
    ).json()["seller_id"]
    client.post(
        f"/v1/commerce/sellers/{seller_id}/listings",
        json={"listings": [{"item_id": "sku-1", "name": "Boots", "description": "hiking boot waterproof", "unit_amount": 12000}]},
        headers={"Authorization": f"Bearer {seller_key}"},
    )
    match_id = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {buyer_key}"},
    ).json()["matches"][0]["match_id"]

    resp = client.get(f"/v1/commerce/verify-match?match_id={match_id}", headers={"Authorization": f"Bearer {other_seller_key}"})
    assert resp.status_code == 403


def test_network_stats_is_public_and_aggregate(client, fresh_db):
    buyer_key = _signup_and_fund(client, fresh_db, email="buyer5@example.com")
    _make_seller_and_listing(client, buyer_key)
    client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {buyer_key}"},
    )
    resp = client.get("/v1/commerce/network-stats")  # no Authorization header at all
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_agents"] >= 1
    assert body["total_matches"] >= 1


def test_checkout_sessions_rejects_unknown_item(client, fresh_db):
    resp = client.post("/v1/commerce/checkout_sessions", json={"item_id": "not-a-real-item", "email": "buyer@example.com"})
    assert resp.status_code == 404


def test_checkout_sessions_rejects_empty_email(client, fresh_db):
    resp = client.post("/v1/commerce/checkout_sessions", json={"item_id": "observe-credits", "email": "  "})
    assert resp.status_code == 400


def test_checkout_sessions_creates_a_real_working_account(client, fresh_db):
    resp = client.post("/v1/commerce/checkout_sessions", json={"item_id": "observe-credits", "email": "newbuyer@example.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["checkout_url"] == "https://checkout.stripe.com/fake-session"
    assert body["api_key"].startswith("obs_")
    # the returned key is real and usable, not a placeholder
    search_resp = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {body['api_key']}"},
    )
    assert search_resp.status_code == 200


def test_my_sellers_lists_own_sellers_with_nested_listings(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, email="dash1@example.com")
    _make_seller_and_listing(client, api_key, item_id="sku-a")
    other_key = _signup_and_fund(client, fresh_db, email="dash2@example.com")
    _make_seller_and_listing(client, other_key, item_id="sku-b")

    resp = client.get("/v1/commerce/my-sellers", headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1  # not the other key's seller
    assert body[0]["name"] == "General Store"
    assert len(body[0]["listings"]) == 1
    assert body[0]["listings"][0]["item_id"] == "sku-a"


def test_my_sellers_requires_auth(client, fresh_db):
    resp = client.get("/v1/commerce/my-sellers")
    assert resp.status_code == 401


def test_create_and_list_buyer_agent(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, email="buyeragent1@example.com")
    resp = client.post(
        "/v1/commerce/buyer-agents",
        json={"name": "Trail Shopper", "default_intent": "waterproof hiking boots", "max_price": 15000},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Trail Shopper"
    agent_id = body["id"]

    list_resp = client.get("/v1/commerce/buyer-agents", headers={"Authorization": f"Bearer {api_key}"})
    assert list_resp.status_code == 200
    ids = [a["id"] for a in list_resp.json()]
    assert agent_id in ids


def test_buyer_agent_rejects_empty_default_intent(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, email="buyeragent2@example.com")
    resp = client.post(
        "/v1/commerce/buyer-agents",
        json={"name": "Empty", "default_intent": "   "},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400


def test_delete_buyer_agent_requires_ownership(client, fresh_db):
    owner_key = _signup_and_fund(client, fresh_db, email="owner@example.com")
    other_key = _signup_and_fund(client, fresh_db, email="notowner@example.com")
    agent_id = client.post(
        "/v1/commerce/buyer-agents",
        json={"name": "Mine", "default_intent": "waterproof boots"},
        headers={"Authorization": f"Bearer {owner_key}"},
    ).json()["id"]

    steal_resp = client.delete(f"/v1/commerce/buyer-agents/{agent_id}", headers={"Authorization": f"Bearer {other_key}"})
    assert steal_resp.status_code == 404

    real_resp = client.delete(f"/v1/commerce/buyer-agents/{agent_id}", headers={"Authorization": f"Bearer {owner_key}"})
    assert real_resp.status_code == 200
    assert real_resp.json()["deleted"] is True


def test_search_via_buyer_agent_id_uses_saved_defaults(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, email="buyeragent3@example.com")
    _make_seller_and_listing(client, api_key, item_id="sku-boots")
    agent_id = client.post(
        "/v1/commerce/buyer-agents",
        json={"name": "Trail Shopper", "default_intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["id"]

    resp = client.post(
        "/v1/commerce/search", json={"buyer_agent_id": agent_id},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    matches = resp.json()["matches"]
    assert len(matches) >= 1
    assert matches[0]["item_id"] == "sku-boots"


def test_search_with_no_intent_and_no_buyer_agent_id_is_rejected(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, email="buyeragent4@example.com")
    resp = client.post("/v1/commerce/search", json={}, headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 400


def test_search_rejects_a_buyer_agent_id_owned_by_someone_else(client, fresh_db):
    owner_key = _signup_and_fund(client, fresh_db, email="owner2@example.com")
    other_key = _signup_and_fund(client, fresh_db, email="notowner2@example.com")
    agent_id = client.post(
        "/v1/commerce/buyer-agents",
        json={"name": "Mine", "default_intent": "waterproof boots"},
        headers={"Authorization": f"Bearer {owner_key}"},
    ).json()["id"]

    resp = client.post(
        "/v1/commerce/search", json={"buyer_agent_id": agent_id},
        headers={"Authorization": f"Bearer {other_key}"},
    )
    assert resp.status_code == 404


def test_import_creates_sellers_listings_and_buyer_agents_in_one_call(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, email="importer1@example.com")
    resp = client.post(
        "/v1/commerce/import",
        json={
            "sellers": [{
                "name": "Imported Outfitters",
                "checkout_session_url": "https://imported.example.com/checkout_sessions",
                "listings": [
                    {"item_id": "imp-1", "name": "Rain Jacket", "description": "waterproof rain jacket", "unit_amount": 9000},
                    {"item_id": "imp-2", "name": "Trail Poles", "description": "adjustable hiking poles", "unit_amount": 4000},
                ],
            }],
            "buyer_agents": [
                {"name": "Rain Shopper", "default_intent": "waterproof rain jacket", "max_price": 10000},
            ],
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sellers_created"] == 1
    assert body["listings_created"] == 2
    assert body["buyer_agents_created"] == 1

    my_sellers = client.get("/v1/commerce/my-sellers", headers={"Authorization": f"Bearer {api_key}"}).json()
    assert len(my_sellers) == 1
    assert len(my_sellers[0]["listings"]) == 2

    my_agents = client.get("/v1/commerce/buyer-agents", headers={"Authorization": f"Bearer {api_key}"}).json()
    assert len(my_agents) == 1
    assert my_agents[0]["name"] == "Rain Shopper"


def test_import_rejects_a_bad_checkout_url_and_reports_partial_progress(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, email="importer2@example.com")
    resp = client.post(
        "/v1/commerce/import",
        json={"sellers": [
            {"name": "Good Seller", "checkout_session_url": "https://good.example.com/checkout_sessions", "listings": []},
            {"name": "Bad Seller", "checkout_session_url": "http://insecure.example.com/checkout_sessions", "listings": []},
        ]},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400
    # the first, valid seller really was created before the second failed --
    # disclosed partial-progress semantics, not silently rolled back
    my_sellers = client.get("/v1/commerce/my-sellers", headers={"Authorization": f"Bearer {api_key}"}).json()
    assert len(my_sellers) == 1
    assert my_sellers[0]["name"] == "Good Seller"


def test_import_requires_auth(client, fresh_db):
    resp = client.post("/v1/commerce/import", json={"sellers": [], "buyer_agents": []})
    assert resp.status_code == 401


def _jwk_to_public_key(jwk):
    import base64
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    raw = base64.urlsafe_b64decode(jwk["x"] + "==")
    return Ed25519PublicKey.from_public_bytes(raw)


def _verify_receipt(client, jws):
    """Mimics what a real third party would do: fetch the public key from
    the well-known endpoint (not from any internal test fixture) and
    verify independently -- proves the receipt is checkable without
    trusting this API's live database, the actual point of signing it."""
    import jwt as pyjwt
    jwk = client.get("/.well-known/observe-commerce-signing-key").json()
    public_key = _jwk_to_public_key(jwk)
    return pyjwt.decode(jws, public_key, algorithms=["EdDSA"])


def test_well_known_signing_key_is_a_valid_jwk(client, fresh_db):
    resp = client.get("/.well-known/observe-commerce-signing-key")
    assert resp.status_code == 200
    jwk = resp.json()
    assert jwk["kty"] == "OKP"
    assert jwk["crv"] == "Ed25519"
    assert jwk["alg"] == "EdDSA"
    assert len(jwk["x"]) > 0
    # No Authorization header was sent -- public by design


def test_search_issues_a_verifiable_receipt_for_each_match(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, email="receipt-buyer@example.com")
    _make_seller_and_listing(client, api_key, item_id="sku-receipt")
    matches = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["matches"]
    match_id = matches[0]["match_id"]

    receipts = client.get(f"/v1/commerce/receipts/{match_id}").json()  # no auth header -- public
    assert len(receipts) == 1
    assert receipts[0]["event_type"] == "commerce.match"

    payload = _verify_receipt(client, receipts[0]["jws"])
    assert payload["type"] == "commerce.match"
    assert payload["match_id"] == match_id
    assert payload["item_id"] == "sku-receipt"


def test_buyer_and_seller_feedback_each_issue_a_verifiable_receipt(client, fresh_db):
    buyer_key = _signup_and_fund(client, fresh_db, email="receipt-buyer2@example.com")
    seller_key = _signup_and_fund(client, fresh_db, email="receipt-seller2@example.com")
    seller_id = _register_seller_with_boots(client, seller_key, "Receipt Test Store")

    match_id = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {buyer_key}"},
    ).json()["matches"][0]["match_id"]

    client.post(
        "/v1/commerce/feedback",
        json={"seller_id": seller_id, "item_id": "sku-1", "outcome": "purchased", "match_id": match_id},
        headers={"Authorization": f"Bearer {buyer_key}"},
    )
    client.post(
        "/v1/commerce/seller-feedback",
        json={"match_id": match_id, "outcome": "fulfilled", "rating": 5},
        headers={"Authorization": f"Bearer {seller_key}"},
    )

    receipts = client.get(f"/v1/commerce/receipts/{match_id}").json()
    event_types = {r["event_type"] for r in receipts}
    assert event_types == {"commerce.match", "commerce.buyer_feedback", "commerce.seller_feedback"}

    for r in receipts:
        payload = _verify_receipt(client, r["jws"])
        assert payload["match_id"] == match_id
        assert payload["type"] == r["event_type"]


def test_tampered_receipt_fails_verification(client, fresh_db):
    import jwt as pyjwt
    api_key = _signup_and_fund(client, fresh_db, email="receipt-tamper@example.com")
    _make_seller_and_listing(client, api_key, item_id="sku-tamper")
    match_id = client.post(
        "/v1/commerce/search", json={"intent": "waterproof hiking boots"},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["matches"][0]["match_id"]
    jws = client.get(f"/v1/commerce/receipts/{match_id}").json()[0]["jws"]

    # Flip one character in the signature segment -- must fail verification,
    # not silently pass.
    header, payload, sig = jws.split(".")
    tampered_sig = ("A" if sig[0] != "A" else "B") + sig[1:]
    tampered = f"{header}.{payload}.{tampered_sig}"

    jwk = client.get("/.well-known/observe-commerce-signing-key").json()
    public_key = _jwk_to_public_key(jwk)
    try:
        pyjwt.decode(tampered, public_key, algorithms=["EdDSA"])
        assert False, "tampered receipt should not verify"
    except pyjwt.InvalidSignatureError:
        pass


def test_receipts_for_unknown_match_id_returns_empty_list(client, fresh_db):
    resp = client.get("/v1/commerce/receipts/not-a-real-match-id")
    assert resp.status_code == 200
    assert resp.json() == []
