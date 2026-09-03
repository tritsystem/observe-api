"""
Tests for server.py's HTTP surface. Never triggers the real startup event
(which would try to load a real embedding model) -- TestClient is
instantiated WITHOUT the `with` context manager, so lifespan/startup
handlers never run; engine.search and billing.create_checkout_session are
monkeypatched directly instead of hitting a real model or Stripe.
"""
import json
import sys
from pathlib import Path

import pytest
import stripe
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import billing  # noqa: E402
import server  # noqa: E402


@pytest.fixture
def client(fresh_db, monkeypatch, tmp_path):
    # Redirect server.py's own `db` reference to the same patched module
    # fresh_db already pointed at a temp file (same module object, so this
    # is just making the intent explicit).
    monkeypatch.setattr(server, "db", fresh_db)

    # Never call the real embedding model.
    monkeypatch.setattr(
        server.engine, "search",
        lambda query, k=10, base_dir_filter=None: [
            {"score": 0.91, "path": "lib/retry.js", "preview": "function retryRequest(config) { ... }", "offset": 0},
        ],
    )

    # Never call real Stripe.
    monkeypatch.setattr(billing, "create_checkout_session", lambda email, key_hash: "https://checkout.stripe.com/fake-session")
    monkeypatch.setattr(billing, "create_pro_checkout_session", lambda email, key_hash, tier: "https://checkout.stripe.com/fake-pro-session")

    # A known, controlled repo registry instead of the real manifest file.
    manifest = tmp_path / "repo_manifest.json"
    manifest.write_text(json.dumps({"axios": "/repos/axios"}))
    monkeypatch.setattr(server.REPO_REGISTRY, "path", str(manifest))

    # No `with` block -- startup/shutdown lifespan events are intentionally
    # never triggered (they'd try to load a real model).
    return TestClient(server.app)


def _signup_and_fund(client, fresh_db, email="alice@example.com", credits=10):
    resp = client.post("/v1/signup", json={"email": email})
    assert resp.status_code == 200
    api_key = resp.json()["api_key"]
    # /v1/signup grants a free trial bonus (SIGNUP_BONUS_CREDITS) on top of
    # whatever this helper adds -- normalize to exactly `credits` regardless
    # of that bonus's configured value, so tests stay deterministic instead
    # of silently drifting by the bonus amount (real bug this caught: every
    # caller of this helper was off by exactly 100, the default bonus).
    key_hash = fresh_db.hash_key(api_key)
    current = fresh_db.get_key_record(api_key)["credits"]
    fresh_db.add_credits(key_hash, credits - current, f"sess_{email}", credits)
    return api_key


def test_signup_returns_key_and_checkout_url(client):
    resp = client.post("/v1/signup", json={"email": "alice@example.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["api_key"].startswith("obs_")
    assert body["checkout_url"] == "https://checkout.stripe.com/fake-session"


def test_signup_rejects_invalid_email(client):
    resp = client.post("/v1/signup", json={"email": "not-an-email"})
    assert resp.status_code == 422


def test_search_without_auth_header_is_401(client):
    resp = client.post("/v1/search", json={"query": "retry logic"})
    assert resp.status_code == 401


def test_search_with_unknown_key_is_401(client):
    resp = client.post(
        "/v1/search",
        json={"query": "retry logic"},
        headers={"Authorization": "Bearer obs_not_a_real_key"},
    )
    assert resp.status_code == 401


def test_search_with_zero_balance_is_402(client, fresh_db):
    api_key = client.post("/v1/signup", json={"email": "bob@example.com"}).json()["api_key"]
    # A fresh signup isn't actually zero-balance -- /v1/signup grants a free
    # trial bonus (SIGNUP_BONUS_CREDITS) by design, so this test drains it
    # first to reach the zero-balance precondition it's actually testing.
    key_hash = fresh_db.hash_key(api_key)
    bonus = fresh_db.get_key_record(api_key)["credits"]
    if bonus:
        fresh_db.add_credits(key_hash, -bonus, "sess_drain_bonus", 0)
    resp = client.post(
        "/v1/search",
        json={"query": "retry logic"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 402
    # A rejected-for-no-credits call must not itself have deducted anything.
    assert fresh_db.get_key_record(api_key)["credits"] == 0


def test_search_succeeds_and_deducts_exactly_one_credit(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=5)
    resp = client.post(
        "/v1/search",
        json={"query": "retry logic for a failed upload"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["credits_remaining"] == 4
    assert body["results"][0]["path"] == "lib/retry.js"


def test_search_logs_usage(client, fresh_db):
    # Regression test for a real bug: private_search() (below) was found,
    # by actually running it against a live server, to charge a credit and
    # return real results while never once calling db.log_usage -- silently
    # undercounting real API activity in the exact table meant to track it.
    # This test (shared search, which already worked) plus the private one
    # below make sure the fix holds and can't regress unnoticed again.
    api_key = _signup_and_fund(client, fresh_db, credits=5)
    client.post(
        "/v1/search",
        json={"query": "retry logic for a failed upload"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    key_hash = fresh_db.hash_key(api_key)
    with fresh_db.get_conn() as conn:
        rows = conn.execute(
            "SELECT query, repo_filter FROM usage_log WHERE key_hash = ?", (key_hash,)
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "retry logic for a failed upload"


def test_private_search_succeeds_and_logs_usage(client, fresh_db, monkeypatch):
    monkeypatch.setattr(server.tenant_index, "get_status", lambda key_hash: {"state": "ready"})
    monkeypatch.setattr(
        server.tenant_index.manager, "search",
        lambda key_hash, query, k=10: [
            {"score": 0.87, "path": "src/thing.py", "preview": "def thing(): ...", "offset": 0},
        ],
    )
    api_key = _signup_and_fund(client, fresh_db, credits=2005)
    resp = client.post(
        "/v1/private/search",
        json={"query": "where is thing defined"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["path"] == "src/thing.py"

    key_hash = fresh_db.hash_key(api_key)
    with fresh_db.get_conn() as conn:
        rows = conn.execute(
            "SELECT query, repo_filter FROM usage_log WHERE key_hash = ?", (key_hash,)
        ).fetchall()
    assert len(rows) == 1, "private_search() must call db.log_usage() same as shared search does"
    assert rows[0][0] == "where is thing defined"
    assert rows[0][1] == "__private__"


def test_search_with_unknown_repo_is_400_and_does_not_charge(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=5)
    resp = client.post(
        "/v1/search",
        json={"query": "retry logic", "repo": "not-a-real-repo"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400
    # The repo filter is validated BEFORE the credit is spent -- a typo'd
    # repo name must not cost the caller anything.
    assert fresh_db.get_key_record(api_key)["credits"] == 5


def test_search_with_known_repo_succeeds(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=5)
    resp = client.post(
        "/v1/search",
        json={"query": "retry logic", "repo": "axios"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    assert resp.json()["results"][0]["repo"] == "axios"


def test_search_out_of_range_k_is_400(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=5)
    resp = client.post(
        "/v1/search",
        json={"query": "retry logic", "k": 999},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400
    assert fresh_db.get_key_record(api_key)["credits"] == 5  # not charged


def test_search_failure_refunds_the_credit(client, fresh_db, monkeypatch):
    """If the engine itself throws, the caller must get their credit back
    -- not silently charged for a search that never produced a result."""
    api_key = _signup_and_fund(client, fresh_db, credits=5)

    def _boom(query, k=10, base_dir_filter=None):
        raise RuntimeError("simulated engine failure")

    monkeypatch.setattr(server.engine, "search", _boom)

    resp = client.post(
        "/v1/search",
        json={"query": "retry logic"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 500
    assert fresh_db.get_key_record(api_key)["credits"] == 5  # refunded, not 4


def test_balance_endpoint(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=42)
    resp = client.get("/v1/balance", headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 200
    assert resp.json()["credits"] == 42


def test_repos_endpoint_lists_the_registry(client):
    resp = client.get("/v1/repos")
    assert resp.status_code == 200
    assert resp.json()["repos"] == ["axios"]


# ------------------------------------------------------------------
# /v1/private/index, /v1/private/status, /v1/webhook/stripe -- previously
# had ZERO automated coverage (see FINDING_concept_shakedown.md): the only
# evidence any of these worked was manual, real-session curl calls, not
# anything that runs in CI. The real clone+embed itself isn't re-tested
# here (too slow/network-dependent for a unit test, and search_engine.py's
# own build_index has its own coverage) -- these test the HTTP endpoint's
# OWN logic: validation, credit charging, and status-conflict handling,
# same boundary the existing search tests already draw around engine.search.
# ------------------------------------------------------------------

def test_private_index_starts_indexing_and_deducts_credits(client, fresh_db, monkeypatch):
    calls = []
    monkeypatch.setattr(server.tenant_index, "get_status", lambda key_hash: {"state": "none"})
    monkeypatch.setattr(server.tenant_index.manager, "start_indexing",
                         lambda key_hash, git_url: calls.append((key_hash, git_url)))

    api_key = _signup_and_fund(client, fresh_db, credits=server.CREDITS_PER_PRIVATE_INDEX + 5)
    resp = client.post(
        "/v1/private/index",
        json={"git_url": "https://github.com/octocat/Hello-World"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "indexing"
    assert body["credits_remaining"] == 5
    assert len(calls) == 1 and calls[0][1] == "https://github.com/octocat/Hello-World"


def test_private_index_rejects_invalid_git_url_and_does_not_charge(client, fresh_db):
    # Real, unmocked validate_git_url() -- not a canned response.
    api_key = _signup_and_fund(client, fresh_db, credits=server.CREDITS_PER_PRIVATE_INDEX + 5)
    resp = client.post(
        "/v1/private/index",
        json={"git_url": "not-a-real-url"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 400
    assert fresh_db.get_key_record(api_key)["credits"] == server.CREDITS_PER_PRIVATE_INDEX + 5


def test_private_index_returns_409_when_already_indexing(client, fresh_db, monkeypatch):
    monkeypatch.setattr(server.tenant_index, "get_status", lambda key_hash: {"state": "indexing"})
    api_key = _signup_and_fund(client, fresh_db, credits=server.CREDITS_PER_PRIVATE_INDEX + 5)
    resp = client.post(
        "/v1/private/index",
        json={"git_url": "https://github.com/octocat/Hello-World"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 409
    # The 409 check happens before charging -- must not have been billed.
    assert fresh_db.get_key_record(api_key)["credits"] == server.CREDITS_PER_PRIVATE_INDEX + 5


def test_private_status_returns_current_state(client, fresh_db, monkeypatch):
    monkeypatch.setattr(server.tenant_index, "get_status",
                         lambda key_hash: {"state": "ready", "chunks": 324})
    api_key = _signup_and_fund(client, fresh_db, credits=1)
    resp = client.get("/v1/private/status", headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 200
    assert resp.json() == {"state": "ready", "chunks": 324}


def test_private_status_requires_auth(client):
    resp = client.get("/v1/private/status")
    assert resp.status_code == 401


def test_webhook_processes_checkout_completed_and_adds_credits(client, fresh_db, monkeypatch):
    api_key = _signup_and_fund(client, fresh_db, credits=0)
    key_hash = fresh_db.hash_key(api_key)

    monkeypatch.setattr(billing, "WEBHOOK_SECRET", "whsec_test")
    canned_event = {
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": "cs_test_real_shape",
            "metadata": {"key_hash": key_hash, "credits": "50000"},
            "amount_total": 500,
        }},
    }
    monkeypatch.setattr(stripe.Webhook, "construct_event", lambda payload, sig, secret: canned_event)

    resp = client.post(
        "/v1/webhook/stripe",
        content=b'{"fake": "payload"}',
        headers={"Stripe-Signature": "t=1,v1=fake"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"received": True}
    assert fresh_db.get_key_record(api_key)["credits"] == 50000


def test_webhook_rejects_invalid_signature(client, fresh_db, monkeypatch):
    monkeypatch.setattr(billing, "WEBHOOK_SECRET", "whsec_test")

    def _raise(payload, sig, secret):
        raise stripe.error.SignatureVerificationError("bad sig", sig)
    monkeypatch.setattr(stripe.Webhook, "construct_event", _raise)

    resp = client.post(
        "/v1/webhook/stripe",
        content=b'{"fake": "payload"}',
        headers={"Stripe-Signature": "t=1,v1=wrong"},
    )
    assert resp.status_code == 400



def test_subscribe_returns_checkout_url_for_starter(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=0)
    resp = client.post("/v1/subscribe", json={"tier": "starter"}, headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 200
    assert resp.json()["checkout_url"] == "https://checkout.stripe.com/fake-pro-session"


def test_subscribe_returns_checkout_url_for_compliance(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=0)
    resp = client.post("/v1/subscribe", json={"tier": "compliance"}, headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 200
    assert resp.json()["checkout_url"] == "https://checkout.stripe.com/fake-pro-session"


def test_subscribe_rejects_unknown_tier(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=0)
    resp = client.post("/v1/subscribe", json={"tier": "gold"}, headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 400


def test_subscribe_without_auth_is_401(client):
    resp = client.post("/v1/subscribe", json={"tier": "starter"})
    assert resp.status_code == 401


def test_balance_reports_not_pro_by_default(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=5)
    resp = client.get("/v1/balance", headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 200
    assert resp.json()["is_pro"] is False
    assert "pro_tier" not in resp.json()
    assert "pro_renews_at" not in resp.json()


def _canned_invoice_paid_event(invoice_id, customer, subscription, amount_paid=900):
    return {
        "type": "invoice.paid",
        "data": {"object": {
            "id": invoice_id,
            "customer": customer,
            "subscription": subscription,
            "amount_paid": amount_paid,
        }},
    }


def test_webhook_invoice_paid_activates_starter_and_grants_credits(client, fresh_db, monkeypatch):
    """The real subscription-activation path: checkout.session.completed
    for mode=subscription is a deliberate no-op (see billing.py), so
    invoice.paid is what actually has to do the work here."""
    api_key = _signup_and_fund(client, fresh_db, credits=0)
    key_hash = fresh_db.hash_key(api_key)

    monkeypatch.setattr(billing, "WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr(
        stripe.Subscription, "retrieve",
        lambda sub_id: {"metadata": {"key_hash": key_hash, "tier": "starter"}, "current_period_end": 1999999999},
    )
    canned_event = _canned_invoice_paid_event("in_test_1", "cus_test_1", "sub_test_1")
    monkeypatch.setattr(stripe.Webhook, "construct_event", lambda payload, sig, secret: canned_event)

    resp = client.post(
        "/v1/webhook/stripe",
        content=b'{"fake": "payload"}',
        headers={"Stripe-Signature": "t=1,v1=fake"},
    )
    assert resp.status_code == 200

    record = fresh_db.get_key_record(api_key)
    assert record["is_pro"] == 1
    assert record["pro_tier"] == "starter"
    assert record["credits"] == billing.STARTER_CREDITS_PER_PERIOD
    assert record["stripe_customer_id"] == "cus_test_1"


def test_webhook_invoice_paid_activates_compliance_with_its_own_credits(client, fresh_db, monkeypatch):
    api_key = _signup_and_fund(client, fresh_db, credits=0)
    key_hash = fresh_db.hash_key(api_key)

    monkeypatch.setattr(billing, "WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr(
        stripe.Subscription, "retrieve",
        lambda sub_id: {"metadata": {"key_hash": key_hash, "tier": "compliance"}, "current_period_end": 1999999999},
    )
    canned_event = _canned_invoice_paid_event("in_test_c1", "cus_test_c1", "sub_test_c1", amount_paid=4900)
    monkeypatch.setattr(stripe.Webhook, "construct_event", lambda payload, sig, secret: canned_event)

    resp = client.post("/v1/webhook/stripe", content=b'{}', headers={"Stripe-Signature": "t=1,v1=fake"})
    assert resp.status_code == 200

    record = fresh_db.get_key_record(api_key)
    assert record["pro_tier"] == "compliance"
    assert record["credits"] == billing.COMPLIANCE_CREDITS_PER_PERIOD


def test_webhook_invoice_paid_missing_tier_metadata_is_a_safe_noop(client, fresh_db, monkeypatch):
    """A subscription created before this feature existed (or outside
    create_pro_checkout_session) has no tier metadata -- must be skipped,
    not crash the webhook handler with a 500."""
    api_key = _signup_and_fund(client, fresh_db, credits=0)
    key_hash = fresh_db.hash_key(api_key)

    monkeypatch.setattr(billing, "WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr(
        stripe.Subscription, "retrieve",
        lambda sub_id: {"metadata": {"key_hash": key_hash}, "current_period_end": 1999999999},
    )
    canned_event = _canned_invoice_paid_event("in_test_notier", "cus_test_notier", "sub_test_notier")
    monkeypatch.setattr(stripe.Webhook, "construct_event", lambda payload, sig, secret: canned_event)

    resp = client.post("/v1/webhook/stripe", content=b'{}', headers={"Stripe-Signature": "t=1,v1=fake"})
    assert resp.status_code == 200
    assert fresh_db.get_key_record(api_key)["is_pro"] == 0


def test_webhook_invoice_paid_redelivered_is_a_safe_noop(client, fresh_db, monkeypatch):
    """The exact at-least-once-delivery scenario billing.py's try/except
    sqlite3.IntegrityError is built to survive -- the SAME invoice.paid
    event arriving twice must grant credits exactly once, and the second
    delivery must still return 200 (Stripe retries on non-2xx), not 500."""
    api_key = _signup_and_fund(client, fresh_db, credits=0)
    key_hash = fresh_db.hash_key(api_key)

    monkeypatch.setattr(billing, "WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr(
        stripe.Subscription, "retrieve",
        lambda sub_id: {"metadata": {"key_hash": key_hash, "tier": "starter"}, "current_period_end": 1999999999},
    )
    canned_event = _canned_invoice_paid_event("in_test_dupe", "cus_test_1", "sub_test_1")
    monkeypatch.setattr(stripe.Webhook, "construct_event", lambda payload, sig, secret: canned_event)

    resp1 = client.post("/v1/webhook/stripe", content=b'{}', headers={"Stripe-Signature": "t=1,v1=fake"})
    resp2 = client.post("/v1/webhook/stripe", content=b'{}', headers={"Stripe-Signature": "t=1,v1=fake"})
    assert resp1.status_code == 200
    assert resp2.status_code == 200

    # Exactly one period's credits granted, not two.
    assert fresh_db.get_key_record(api_key)["credits"] == billing.STARTER_CREDITS_PER_PERIOD


def test_webhook_subscription_deleted_deactivates_pro(client, fresh_db, monkeypatch):
    api_key = _signup_and_fund(client, fresh_db, credits=0)
    key_hash = fresh_db.hash_key(api_key)
    fresh_db.activate_pro(key_hash, "cus_test_2", "sub_test_2", 1999999999.0, 300000, "in_pre", "compliance")

    monkeypatch.setattr(billing, "WEBHOOK_SECRET", "whsec_test")
    canned_event = {
        "type": "customer.subscription.deleted",
        "data": {"object": {"customer": "cus_test_2"}},
    }
    monkeypatch.setattr(stripe.Webhook, "construct_event", lambda payload, sig, secret: canned_event)

    resp = client.post("/v1/webhook/stripe", content=b'{}', headers={"Stripe-Signature": "t=1,v1=fake"})
    assert resp.status_code == 200

    record = fresh_db.get_key_record(api_key)
    assert record["is_pro"] == 0
    assert record["pro_tier"] is None
    assert record["credits"] == 300000  # not clawed back


def test_starter_key_gets_starter_rate_limit_bucket(client, fresh_db, monkeypatch):
    """Regression guard for the actual point of the starter rate limit --
    a starter key must be able to burst past the standard tier's CAPACITY
    within a single test (no real time passing to refill either bucket)."""
    api_key = _signup_and_fund(client, fresh_db, credits=1000)
    key_hash = fresh_db.hash_key(api_key)
    fresh_db.activate_pro(key_hash, "cus_test_3", "sub_test_3", 1999999999.0, 0, "in_starter3", "starter")

    import rate_limit
    standard_capacity = int(rate_limit.CAPACITY)
    burst = standard_capacity + 5
    statuses = [
        client.post("/v1/search", json={"query": "retry logic"},
                     headers={"Authorization": f"Bearer {api_key}"}).status_code
        for _ in range(burst)
    ]
    assert 429 not in statuses, f"starter key got rate-limited within a {burst}-call burst -- not getting its own bucket"


def test_compliance_key_gets_a_bigger_bucket_than_starter(client, fresh_db, monkeypatch):
    api_key = _signup_and_fund(client, fresh_db, credits=1000)
    key_hash = fresh_db.hash_key(api_key)
    fresh_db.activate_pro(key_hash, "cus_test_4", "sub_test_4", 1999999999.0, 0, "in_compliance4", "compliance")

    burst = int(billing.STARTER_RATE_CAPACITY) + 5  # past starter's ceiling, within compliance's
    statuses = [
        client.post("/v1/search", json={"query": "retry logic"},
                     headers={"Authorization": f"Bearer {api_key}"}).status_code
        for _ in range(burst)
    ]
    assert 429 not in statuses, "compliance key should get a bucket bigger than starter's"


def test_private_index_is_free_for_compliance_tier(client, fresh_db, monkeypatch):
    monkeypatch.setattr(server.tenant_index, "get_status", lambda key_hash: {"state": "none"})
    monkeypatch.setattr(server.tenant_index.manager, "start_indexing", lambda key_hash, git_url: None)

    api_key = _signup_and_fund(client, fresh_db, credits=5)  # far less than CREDITS_PER_PRIVATE_INDEX
    key_hash = fresh_db.hash_key(api_key)
    fresh_db.activate_pro(key_hash, "cus_test_5", "sub_test_5", 1999999999.0, 0, "in_compliance5", "compliance")

    resp = client.post(
        "/v1/private/index",
        json={"git_url": "https://github.com/pallets/click"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    assert resp.json()["credits_remaining"] == 5  # untouched -- indexing was free


def test_private_index_still_costs_starter_tier(client, fresh_db, monkeypatch):
    monkeypatch.setattr(server.tenant_index, "get_status", lambda key_hash: {"state": "none"})
    monkeypatch.setattr(server.tenant_index.manager, "start_indexing", lambda key_hash, git_url: None)

    api_key = _signup_and_fund(client, fresh_db, credits=5)
    key_hash = fresh_db.hash_key(api_key)
    fresh_db.activate_pro(key_hash, "cus_test_6", "sub_test_6", 1999999999.0, 0, "in_starter6", "starter")

    resp = client.post(
        "/v1/private/index",
        json={"git_url": "https://github.com/pallets/click"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 402  # 5 credits is not enough at full price -- starter gets no discount


def test_audit_log_requires_compliance_tier(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=5)  # no subscription at all
    resp = client.get("/v1/audit-log", headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 403


def test_audit_log_rejects_starter_tier(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=5)
    key_hash = fresh_db.hash_key(api_key)
    fresh_db.activate_pro(key_hash, "cus_test_7", "sub_test_7", 1999999999.0, 0, "in_starter7", "starter")
    resp = client.get("/v1/audit-log", headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 403


def test_audit_log_returns_real_usage_history_for_compliance_tier(client, fresh_db):
    api_key = _signup_and_fund(client, fresh_db, credits=5)
    key_hash = fresh_db.hash_key(api_key)
    fresh_db.activate_pro(key_hash, "cus_test_8", "sub_test_8", 1999999999.0, 0, "in_compliance8", "compliance")

    client.post("/v1/search", json={"query": "retry logic for a failed upload"},
                headers={"Authorization": f"Bearer {api_key}"})

    resp = client.get("/v1/audit-log", headers={"Authorization": f"Bearer {api_key}"})
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["query"] == "retry logic for a failed upload"
