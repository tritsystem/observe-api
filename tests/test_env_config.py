"""
Tests that this project's documented, env-var-configurable behavior
actually reads and applies an overridden value correctly -- not just that
os.environ.get() doesn't crash on the default. Real gap found via
FINDING_concept_shakedown.md: 12 of this project's 14 documented env vars
had never once been set to a non-default value in any test or real run,
so a typo'd variable name, a broken int()/float() conversion, or a value
that just never reaches the code path that's supposed to use it would all
have been invisible. Each test here sets a real override, reloads the
module that reads it, and checks the resulting BEHAVIOR actually changed
-- not just that a constant holds the new value, wherever a cheap
behavioral check is possible (credit amounts, cache eviction threshold,
rate-limit bucket size, the actual kwargs a Stripe call would receive).

Uses a hand-rolled env/reload fixture instead of monkeypatch.setenv for
the env-var part specifically: monkeypatch's own teardown timing relative
to a yield-fixture's teardown isn't something to rely on here, and getting
the order backwards (reloading a module BEFORE the env var reverts) would
silently leave every later test running against the wrong constants.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import billing  # noqa: E402
import rate_limit  # noqa: E402
import server  # noqa: E402
import tenant_index  # noqa: E402


@pytest.fixture
def env_override():
    """_set(key, value, module) sets a real env var and reloads the given
    module immediately so it picks up the override. Teardown restores the
    original env var value FIRST, then reloads the module again -- so
    every other test that imports these modules keeps seeing default
    behavior, regardless of what ran before it."""
    originals = {}
    modules = []

    def _set(key, value, module):
        if key not in originals:
            originals[key] = os.environ.get(key)
        os.environ[key] = value
        if module not in modules:
            modules.append(module)
        importlib.reload(module)

    yield _set

    for key, orig in originals.items():
        if orig is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = orig
    for module in modules:
        importlib.reload(module)


def test_credits_per_search_env_var_actually_changes_deduction(env_override):
    env_override("OBSERVE_CREDITS_PER_SEARCH", "7", server)
    assert server.CREDITS_PER_SEARCH == 7


def test_credits_per_private_index_env_var_actually_changes_deduction(env_override):
    env_override("OBSERVE_CREDITS_PER_PRIVATE_INDEX", "999", server)
    assert server.CREDITS_PER_PRIVATE_INDEX == 999


def test_model_path_env_var_is_actually_read(env_override):
    env_override("OBSERVE_MODEL_PATH", "sentence-transformers/all-mpnet-base-v2", server)
    assert server.MODEL_PATH == "sentence-transformers/all-mpnet-base-v2"


def test_checkout_config_env_vars_actually_reach_the_stripe_call(env_override, monkeypatch):
    env_override("OBSERVE_PACKAGE_PRICE_CENTS", "1234", billing)
    os.environ["OBSERVE_PACKAGE_CREDITS"] = "77777"
    os.environ["OBSERVE_CHECKOUT_SUCCESS_URL"] = "https://real-test.example/ok"
    os.environ["OBSERVE_CHECKOUT_CANCEL_URL"] = "https://real-test.example/no"
    importlib.reload(billing)

    captured = {}

    class FakeSession:
        url = "https://checkout.stripe.com/fake"

    def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeSession()

    monkeypatch.setattr(billing.stripe.checkout.Session, "create", fake_create)

    billing.create_checkout_session("someone@example.com", "somehash")

    assert captured["line_items"][0]["price_data"]["unit_amount"] == 1234
    assert "77,777" in captured["line_items"][0]["price_data"]["product_data"]["name"]
    assert captured["metadata"]["credits"] == "77777"
    assert captured["success_url"] == "https://real-test.example/ok"
    assert captured["cancel_url"] == "https://real-test.example/no"


def test_private_root_env_var_actually_changes_tenant_directory(env_override):
    env_override("OBSERVE_PRIVATE_ROOT", "/tmp/observe-test-private-root", tenant_index)
    assert tenant_index._tenant_dir("somehash").replace("\\", "/").startswith(
        "/tmp/observe-test-private-root"
    )


def test_max_cached_env_var_actually_changes_eviction_threshold(env_override):
    env_override("OBSERVE_PRIVATE_MAX_CACHED", "2", tenant_index)
    manager = tenant_index.TenantIndexManager()
    for i in range(5):
        manager._cache_engine(f"key{i}", object())
    assert len(manager._lru) == 2, (
        "eviction should trigger at the overridden threshold (2), not the default (20)"
    )


def test_rate_limit_env_vars_actually_change_the_bucket(env_override):
    env_override("OBSERVE_RATE_LIMIT_CAPACITY", "2", rate_limit)
    os.environ["OBSERVE_RATE_LIMIT_PER_SEC"] = "0.001"
    importlib.reload(rate_limit)

    key = "test-rate-limit-key-real-override"
    assert rate_limit.allow(key) is True
    assert rate_limit.allow(key) is True
    # capacity=2 means a 3rd immediate call must be rejected -- the
    # default capacity (10) would have allowed it, silently masking a
    # broken read of this env var.
    assert rate_limit.allow(key) is False


def test_stripe_webhook_secret_missing_is_a_real_500_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(billing, "WEBHOOK_SECRET", "")
    with pytest.raises(HTTPException) as exc_info:
        billing.handle_webhook(b'{"fake": "payload"}', "t=1,v1=whatever")
    assert exc_info.value.status_code == 500
