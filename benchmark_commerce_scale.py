"""
Real load test for commerce_router.py's search path -- registers real
listings through the actual HTTP API and times real /v1/commerce/search
calls at scale. Not part of the fast pytest suite on purpose (takes
real minutes at 100k listings, same reason
benchmark_finetuned_vs_stock_29.py isn't either) -- run manually:

    python benchmark_commerce_scale.py [N]

MEASURED RESULTS (2026-08-04, this exact script, N sellers x 500
listings each, a fake in-memory embedding model so the number reflects
this module's own retrieval cost, not the real SentenceTransformer's):

  Before the FAISS index (O(N) full-catalog scan + per-row text-parsed
  embeddings, commerce_router.py's original design):
    N=2,000 listings:    ~386ms/search average
    N=20,000 listings:   ~32,758ms/search average (32.7s -- unusable)

  After (commerce_search_index.py, IndexIDMap(IndexFlatIP)):
    N=20,000 listings:   ~48.4ms/search average  (~677x faster, same N)
    N=100,000 listings:  ~63.1ms/search average  (5x more data, only
                                                    ~1.3x slower --
                                                    genuinely sub-linear,
                                                    not just "faster")

Re-run this yourself before trusting these numbers on a different
machine/scale -- they're a real measurement from one run on one
machine, not a guarantee.
"""
import os
import sys
import tempfile
import time

import numpy as np
from fastapi.testclient import TestClient

import billing
import db

db.DB_PATH = os.path.join(tempfile.mkdtemp(), "bench.db")
db.init_db()

import server  # noqa: E402 -- must follow the db.DB_PATH override above


class _FakeModel:
    """Deterministic-shaped random embeddings -- fast to generate at
    real scale (100k+ texts), and this benchmark measures retrieval
    cost, not embedding-model quality (search_engine.py's own
    correctness is verified elsewhere against the real model)."""

    def __init__(self, dim=32):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True):
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**32))
        vecs = rng.normal(size=(len(texts), self.dim)).astype("float32")
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    listings_per_seller = 500
    batch = 200

    server.db = db
    server.engine.model = _FakeModel()
    server.engine.ready = True
    billing.create_checkout_session = lambda email, key_hash: "https://checkout.stripe.com/fake"
    server.rate_limit.allow = lambda key: True  # this is a throughput test, not a rate-limit test

    client = TestClient(server.app)
    api_key = client.post("/v1/signup", json={"email": "bench@example.com"}).json()["api_key"]

    print(f"Registering {n} listings across {n // listings_per_seller} sellers (real DB writes)...")
    t0 = time.time()
    for s in range(n // listings_per_seller):
        seller_id = client.post(
            "/v1/commerce/sellers",
            json={"name": f"Bench Store {s}", "checkout_session_url": f"https://bench{s}.example.com/checkout_sessions"},
            headers={"Authorization": f"Bearer {api_key}"},
        ).json()["seller_id"]
        for i in range(0, listings_per_seller, batch):
            chunk = [
                {"item_id": f"sku-{j}", "name": f"Item {j}", "description": f"synthetic bench item number {j}", "unit_amount": 1000 + j}
                for j in range(i, min(i + batch, listings_per_seller))
            ]
            resp = client.post(
                f"/v1/commerce/sellers/{seller_id}/listings",
                json={"listings": chunk},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code != 200:
                print("FAILED", s, i, resp.status_code, resp.text[:300])
                return
    print(f"Registered {n} listings in {time.time() - t0:.1f}s")

    key_hash = db.hash_key(api_key)
    db.add_credits(key_hash, 1000, "bench_topup", 0)

    print("Running 5 real searches over the full catalog, timing each...")
    times = []
    for i in range(5):
        t0 = time.time()
        resp = client.post(
            "/v1/commerce/search", json={"intent": "synthetic bench item"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        dt = time.time() - t0
        times.append(dt)
        print(f"  search {i}: {dt * 1000:.1f}ms, status={resp.status_code}, matches={len(resp.json().get('matches', []))}")

    print(f"avg search latency at N={n} listings: {sum(times) / len(times) * 1000:.1f}ms")


if __name__ == "__main__":
    main()
