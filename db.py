"""
SQLite storage for API keys and credit balances. Deliberately simple (raw
sqlite3, no ORM) for a v1 -- this is a single-process service with a single
writer, WAL mode makes concurrent reads safe alongside it.
"""
import hashlib
import secrets
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = "observe_api.db"


def hash_key(raw_key: str) -> str:
    # Store only the hash, never the raw key -- same practice as
    # GitHub/Stripe API tokens. A leaked DB doesn't leak usable keys.
    return hashlib.sha256(raw_key.encode()).hexdigest()


_hash_key = hash_key  # internal alias, kept for the calls below


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                credits INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                last_used_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL,
                query TEXT NOT NULL,
                repo_filter TEXT,
                result_count INTEGER,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS credit_purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL,
                stripe_session_id TEXT UNIQUE NOT NULL,
                credits_added INTEGER NOT NULL,
                amount_cents INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        # commerce_router.py's ACP-compatible buyer/seller discovery
        # tables -- defined here, not lazily in commerce_router.py, so
        # they're created by the same init_db() call every other table
        # already goes through (server.py's lifespan calls this on real
        # startup; tests' fresh_db fixture calls it against a temp DB).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commerce_sellers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                checkout_session_url TEXT NOT NULL,
                payment_rail TEXT NOT NULL DEFAULT 'acp',
                payment_uri TEXT,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commerce_listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seller_id INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                unit_amount INTEGER,
                currency TEXT NOT NULL DEFAULT 'usd',
                category TEXT,
                embedding TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        # Ground-truth feedback loop, self-reported by the buyer's own
        # agent -- see commerce_router.py's POST /v1/commerce/feedback
        # docstring for the real, disclosed trust boundary (OBSERVE never
        # sees the actual checkout, so this is self-reported, not
        # independently verified).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commerce_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL,
                seller_id INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        # Persists ListingAffinityMemory's real learned STDP weights per
        # buyer key, so a process restart doesn't silently discard
        # learned affinity -- see commerce_router.py's _get_memory() and
        # commerce_spiking_memory.py's to_rows()/load_rows(). Deliberately
        # does NOT persist membrane_potential/heat (short-term, decays
        # fast by design -- not worth the write volume).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commerce_memory_weights (
                key_hash TEXT NOT NULL,
                src_item TEXT NOT NULL,
                dst_item TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (key_hash, src_item, dst_item)
            )
        """)
        # A real, shared correlation point between the two disconnected
        # sides of a transaction (OBSERVE genuinely never sees the actual
        # checkout, by design -- see commerce_router.py's module
        # docstring). Every match a buyer sees in a real /v1/commerce/
        # search response gets a real match_id; the buyer's feedback and
        # the seller's feedback both reference it, so a reputation score
        # can be built from BOTH sides agreeing (or disagreeing) about
        # the same real event instead of trusting one side alone.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commerce_matches (
                match_id TEXT PRIMARY KEY,
                buyer_key_hash TEXT NOT NULL,
                seller_id INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        # The seller-side half of the two-sided feedback loop --
        # commerce_feedback already captures the buyer's self-report;
        # this is the seller confirming (or disputing) the same real
        # match_id. Both being real signals is what makes the reputation
        # system worth more than trusting either side alone.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commerce_seller_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                seller_key_hash TEXT NOT NULL,
                outcome TEXT NOT NULL,
                rating INTEGER,
                created_at REAL NOT NULL
            )
        """)
        # A saved, reusable buyer-agent CONFIGURATION -- not a running
        # process. commerce_search can reference one by id and fall back
        # to its default_intent/max_price/category for any field the
        # caller doesn't explicitly override, so a real external agent
        # (any framework) can be pointed at "buyer_agent_id=N" once
        # instead of repeating the same intent/filters on every call.
        # No execution loop lives here or anywhere else in this service --
        # something still has to actually call /v1/commerce/search.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commerce_buyer_agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                default_intent TEXT NOT NULL,
                max_price INTEGER,
                category TEXT,
                created_at REAL NOT NULL
            )
        """)
        # A quote LOCKS a listing's price/currency at a point in time, with
        # a real expiry -- closes a real gap: without this, a listing's
        # unit_amount is just whatever's live in commerce_listings at
        # query time, no protection for a buyer-agent against a seller
        # changing the price between when it saw a match and when it
        # actually acts on it (or vice versa -- a seller has no record of
        # what was actually offered if a buyer claims a different price
        # later). OBSERVE still never touches payment -- this is a signed
        # record of what price was observed being offered when, not an
        # escrow or a guarantee the seller will honor it.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commerce_quotes (
                quote_id TEXT PRIMARY KEY,
                buyer_key_hash TEXT NOT NULL,
                seller_id INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                unit_amount INTEGER,
                currency TEXT NOT NULL,
                match_id TEXT,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        # Ed25519/JWS-signed receipts (see commerce_receipts.py) for real
        # commerce events -- a match happening, a buyer/seller reporting an
        # outcome. Independently verifiable by any third party against
        # /.well-known/observe-commerce-signing-key, without needing to
        # trust this API's live database. Deliberately does NOT attest that
        # money moved (OBSERVE never touches payment) -- only that OBSERVE
        # recorded this exact claim, from this exact event, at this time.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commerce_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                jws TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        # Real migration, not just a CREATE TABLE change -- commerce_sellers
        # already has live production rows (CREATE TABLE IF NOT EXISTS does
        # NOT retroactively add columns to an existing table). Idempotent:
        # checks PRAGMA table_info first, so re-running init_db() on every
        # startup (as it already does) never double-applies or errors on an
        # already-migrated database.
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(commerce_sellers)").fetchall()}
        if "payment_rail" not in existing_cols:
            conn.execute("ALTER TABLE commerce_sellers ADD COLUMN payment_rail TEXT NOT NULL DEFAULT 'acp'")
        if "payment_uri" not in existing_cols:
            conn.execute("ALTER TABLE commerce_sellers ADD COLUMN payment_uri TEXT")


def create_api_key(email: str, initial_credits: int = 0) -> str:
    """Creates a new key, returns the RAW key (only ever returned once --
    caller must save it, we only ever store the hash from here on).
    initial_credits seeds the signup free-trial balance (see
    server.py's SIGNUP_BONUS_CREDITS) -- defaults to 0 for any other
    caller that doesn't want a bonus applied."""
    raw_key = "obs_" + secrets.token_urlsafe(32)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO api_keys (key_hash, email, credits, created_at) VALUES (?, ?, ?, ?)",
            (_hash_key(raw_key), email, initial_credits, time.time()),
        )
    return raw_key


def get_key_record(raw_key: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (_hash_key(raw_key),)
        ).fetchone()
        return dict(row) if row else None


def deduct_credit(raw_key: str, amount: int = 1) -> bool:
    """Atomically deducts `amount` credits if the balance covers it.
    Returns False (no deduction happens) if balance is insufficient --
    the UPDATE's WHERE clause makes this check-and-deduct atomic under
    SQLite's own transaction, no separate read-then-write race."""
    key_hash = _hash_key(raw_key)
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE api_keys SET credits = credits - ?, last_used_at = ? "
            "WHERE key_hash = ? AND credits >= ?",
            (amount, time.time(), key_hash, amount),
        )
        return cur.rowcount > 0


def add_credits(key_hash: str, credits: int, stripe_session_id: str, amount_cents: int):
    """Credits EXACTLY the one API key named by key_hash -- the checkout
    session's metadata carries this hash (set at signup time, see
    server.py), so a purchase is never ambiguous even if two accounts
    share an email address."""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE api_keys SET credits = credits + ? WHERE key_hash = ?",
            (credits, key_hash),
        )
        if cur.rowcount == 0:
            raise ValueError(f"add_credits: no api_key row for key_hash {key_hash!r} -- refusing to log an orphaned purchase")
        conn.execute(
            "INSERT INTO credit_purchases (key_hash, stripe_session_id, credits_added, amount_cents, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key_hash, stripe_session_id, credits, amount_cents, time.time()),
        )


def log_usage(raw_key: str, query: str, repo_filter: str | None, result_count: int):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO usage_log (key_hash, query, repo_filter, result_count, created_at) VALUES (?, ?, ?, ?, ?)",
            (_hash_key(raw_key), query, repo_filter, result_count, time.time()),
        )
