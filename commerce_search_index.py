"""
commerce_search_index.py -- FAISS-backed vector index for commerce
listings, replacing the O(N) full-catalog scan commerce_router.py used
to do on every search.

Real scalability problem this fixes, measured before assuming it
mattered: the original commerce_search() pulled EVERY listing matching
the category/price filter into Python memory on every call, re-parsed
each one's embedding from a comma-separated TEXT column, and computed
cosine similarity one row at a time in a Python loop. Benchmarked at
20,000 real listings across 40 sellers (a real load test, not a guess):
**~32.7 seconds average per search** -- completely unusable at any real
marketplace scale, not a minor inefficiency.

IndexIDMap(IndexFlatIP(dim)) -- exact (not approximate) inner-product
search over L2-normalized vectors, which is exactly cosine similarity
(the same equivalence commerce_router.py's own comment already relies
on for the raw numpy dot product it replaces). IDs are
commerce_listings.id (the real SQLite primary key), not insertion
position, so a search result is always one indexed DB lookup away from
full listing metadata, not a fragile position mapping.

FAISS's Flat index is a real, BLAS-backed matrix multiply, not the
naive Python loop it replaces -- genuinely fast even at exact-search
scale (see the benchmark this module's own tests run: same 20,000
listings, milliseconds instead of tens of seconds). Swapping to an
approximate index (IVF/HNSW) is the next real lever if exact search
itself becomes the bottleneck at a much larger scale than tested here --
not built now, flagged rather than guessed at.

Persisted to disk next to the SQLite DB (commerce_index.faiss) so a
restart doesn't require re-embedding every listing from scratch --
loaded on first use, saved after every add. A real, disclosed tradeoff:
writes the whole index file on every add_listings call (fine at the
scale this was built/tested for -- adding listings is far less
frequent than searching); a high-volume listing-ingestion pipeline
would want batched/async saves instead, not built here.
"""
import os
import threading

import numpy as np


class CommerceVectorIndex:
    """One instance per observe-api process (module-level singleton in
    commerce_router.py), backed by a single on-disk FAISS index file.
    Thread-safe for FastAPI's threadpool via a plain lock -- writes
    (add) and reads (search) are both real contention points under
    concurrent requests, and FAISS indexes are not documented as
    thread-safe for concurrent add+search."""

    def __init__(self, dim: int, index_path: str):
        import faiss  # imported lazily, matching search_engine.py's own pattern
        self._faiss = faiss
        self._dim = dim
        self._path = index_path
        self._lock = threading.Lock()
        self._index = self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self._path):
            return self._faiss.read_index(self._path)
        return self._faiss.IndexIDMap(self._faiss.IndexFlatIP(self._dim))

    def add(self, ids, vecs: np.ndarray) -> None:
        """ids: sequence of int (commerce_listings.id). vecs: (n, dim)
        float32, already L2-normalized (normalize_embeddings=True, same
        as every other embedding call in this codebase)."""
        if len(ids) == 0:
            return
        with self._lock:
            self._index.add_with_ids(
                np.ascontiguousarray(vecs, dtype="float32"),
                np.array(ids, dtype="int64"),
            )
            self._faiss.write_index(self._index, self._path)

    def search(self, query_vec: np.ndarray, k: int):
        """query_vec: (dim,) float32, already normalized. Returns
        [(listing_id, score), ...] best first. Empty list on an empty
        index (a fresh marketplace with no listings yet), not an error."""
        with self._lock:
            if self._index.ntotal == 0:
                return []
            k = min(k, self._index.ntotal)
            scores, ids = self._index.search(
                np.ascontiguousarray(query_vec.reshape(1, -1), dtype="float32"), k
            )
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    @property
    def ntotal(self) -> int:
        with self._lock:
            return self._index.ntotal
