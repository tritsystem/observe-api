"""
Shared fixtures. Tests never touch a real Stripe account or a real
embedding model -- db.DB_PATH is redirected to a temp file per test, and
server.engine.search / billing.create_checkout_session are monkeypatched
to canned behavior rather than making real network/model calls.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Points db.DB_PATH at a fresh temp file for this test only, so tests
    never share state with each other or with a real observe_api.db."""
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_file))
    db.init_db()
    return db
