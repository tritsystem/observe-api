"""
Tests for db.py -- particularly the atomic credit deduction (the thing
that actually prevents a balance race under concurrent requests) and the
purchase-attribution-by-key-hash fix (billing.py originally attributed
by email, which could collide across two signups -- see git history).
"""
import pytest


def test_create_api_key_starts_at_zero_credits(fresh_db):
    raw_key = fresh_db.create_api_key("alice@example.com")
    record = fresh_db.get_key_record(raw_key)
    assert record is not None
    assert record["credits"] == 0
    assert record["email"] == "alice@example.com"


def test_unknown_key_returns_none(fresh_db):
    assert fresh_db.get_key_record("obs_this_was_never_issued") is None


def test_deduct_credit_succeeds_with_sufficient_balance(fresh_db):
    raw_key = fresh_db.create_api_key("alice@example.com")
    fresh_db.add_credits(fresh_db.hash_key(raw_key), 10, "sess_1", 500)

    assert fresh_db.deduct_credit(raw_key, 1) is True
    assert fresh_db.get_key_record(raw_key)["credits"] == 9


def test_deduct_credit_fails_cleanly_with_insufficient_balance(fresh_db):
    raw_key = fresh_db.create_api_key("alice@example.com")
    # 0 credits by default -- deduction must be REFUSED, not go negative.
    assert fresh_db.deduct_credit(raw_key, 1) is False
    assert fresh_db.get_key_record(raw_key)["credits"] == 0


def test_deduct_credit_never_goes_negative_at_the_boundary(fresh_db):
    raw_key = fresh_db.create_api_key("alice@example.com")
    fresh_db.add_credits(fresh_db.hash_key(raw_key), 1, "sess_1", 5)

    assert fresh_db.deduct_credit(raw_key, 1) is True   # exactly 1 -> 0, allowed
    assert fresh_db.deduct_credit(raw_key, 1) is False  # 0 -> would be -1, refused
    assert fresh_db.get_key_record(raw_key)["credits"] == 0


def test_negative_amount_deduct_credit_acts_as_a_refund(fresh_db):
    """server.py's refund-on-search-failure path calls
    deduct_credit(key, -N) to add credits back -- this is the real
    mechanism it relies on, not a separate function."""
    raw_key = fresh_db.create_api_key("alice@example.com")
    fresh_db.add_credits(fresh_db.hash_key(raw_key), 5, "sess_1", 100)
    fresh_db.deduct_credit(raw_key, 1)
    assert fresh_db.get_key_record(raw_key)["credits"] == 4

    fresh_db.deduct_credit(raw_key, -1)  # refund
    assert fresh_db.get_key_record(raw_key)["credits"] == 5


def test_add_credits_attributes_to_the_exact_key_not_by_email(fresh_db):
    """Regression test for the real bug caught and fixed while building
    this: two accounts CAN share an email (nothing prevents duplicate
    signups), so attribution must be by key_hash, never by email, or a
    Stripe payment for one key could silently credit a different key
    that happens to share its owner's email address."""
    key_a = fresh_db.create_api_key("shared@example.com")
    key_b = fresh_db.create_api_key("shared@example.com")

    fresh_db.add_credits(fresh_db.hash_key(key_a), 100, "sess_a", 500)

    assert fresh_db.get_key_record(key_a)["credits"] == 100
    assert fresh_db.get_key_record(key_b)["credits"] == 0  # untouched


def test_add_credits_refuses_to_log_an_orphaned_purchase(fresh_db):
    with pytest.raises(ValueError):
        fresh_db.add_credits("not_a_real_key_hash", 100, "sess_x", 500)


def test_log_usage_records_query_and_repo(fresh_db):
    raw_key = fresh_db.create_api_key("alice@example.com")
    fresh_db.log_usage(raw_key, "retry logic", "axios", 3)

    with fresh_db.get_conn() as conn:
        row = conn.execute("SELECT * FROM usage_log").fetchone()
    assert row["query"] == "retry logic"
    assert row["repo_filter"] == "axios"
    assert row["result_count"] == 3
