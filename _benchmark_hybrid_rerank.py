"""
Follow-up to _benchmark_hybrid_search.py: that experiment (full-corpus
dense + BM25 combined via equal-weight RRF) came back net negative (3
wins / 6 regressions / 11 ties) for two distinct reasons:

1. Cross-repo/cross-language contamination -- BM25 has zero notion of
   "this is the wrong project," so a shared word ("route", "expire") was
   enough to pull in an unrelated file from a completely different repo
   in this benchmark's 15-repos-in-one-index corpus (a real artifact of
   this benchmark's setup, not of OBSERVE's actual per-tenant indexes --
   but still worth designing around rather than assuming away).
2. BM25's term-frequency scoring favors verbose test/doc files that
   repeat a query word many times over terse, CORRECT implementation
   files that state it once (lost flask's scaffold.py, svelte's
   reactivity/set.js, pandas's merge.py this way).

This variant fixes both by restructuring the combination as
RETRIEVE-THEN-RERANK instead of two independent full-corpus rankings
fused together:
  1. Take the dense embedding's own top-N candidates (N=30) -- this is
     the actual production-realistic retrieval step, and it structurally
     can't pull in a wrong-repo file that dense similarity itself didn't
     already consider plausible.
  2. Re-rank ONLY those N candidates with a dense-dominant weighted blend
     (alpha=0.7 dense / 0.3 BM25, both min-max normalized within the pool)
     -- BM25 can now only reorder among candidates dense already vouched
     for, never introduce a new one from elsewhere in the corpus.

Reuses the same reconstructed chunk text + already-built BM25 index as
the previous script (same corpus, same embeddings) so this is testing the
COMBINATION STRATEGY as the only new variable.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

N_CANDIDATES = 30
ALPHA = 0.7  # weight on dense score; (1-ALPHA) on BM25
K = 5


def tokenize(text):
    raw = re.findall(r"[A-Za-z0-9]+", text.lower())
    tokens = list(raw)
    for tok in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        parts = [p for p in tok.split("_") if p]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


_file_cache = {}


def read_cached(base_dir, rel_path):
    key = (base_dir, rel_path)
    if key not in _file_cache:
        try:
            _file_cache[key] = Path(base_dir, rel_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            _file_cache[key] = ""
    return _file_cache[key]


def reconstruct_chunk_text(base_dir, rel_path, offset):
    text = read_cached(base_dir, rel_path)
    return f"file:{rel_path}\n{text[offset:offset+800]}"


QUERIES = [
    ("react",   "how does useState schedule a re-render"),
    ("react",   "where does react decide whether to skip a re-render for unchanged props"),
    ("react",   "where is the code that diffs two virtual dom trees"),
    ("django",  "how does django validate a model field before saving"),
    ("django",  "where does django route an incoming url to a view"),
    ("django",  "how does django's orm build a sql query from a queryset"),
    ("flask",   "how does flask register a route handler"),
    ("fastapi", "how does fastapi validate a request body against a pydantic model"),
    ("express", "how does express match a route with wildcard parameters"),
    ("redis",   "how does redis expire a key after a timeout"),
    ("redis",   "how does redis persist data to disk"),
    ("numpy",   "how does numpy broadcast arrays of different shapes"),
    ("pandas",  "how does pandas handle missing values when merging two dataframes"),
    ("svelte",  "how does svelte detect a reactive variable change"),
    ("vue",     "how does vue's reactivity system track dependencies"),
    ("axios",   "how does axios cancel an in-flight request"),
    ("gin",     "how does gin chain middleware for a request"),
    ("cargo",   "how does cargo resolve dependency version conflicts"),
    ("tokio",   "how does tokio schedule an async task onto a worker thread"),
    ("laravel", "how does laravel handle database migrations"),
]


def main():
    print("[bench] loading baseline index (vectors + metadata)...", flush=True)
    vecs = np.load("data/observe-index/vectors_float32.npy").astype("float32")
    meta = json.load(open("data/observe-index/metadata.json", encoding="utf-8"))
    paths = meta["paths"]
    rows = meta["chunks"]
    n = len(rows)
    print(f"[bench] {n:,} chunks", flush=True)

    print("[bench] reconstructing chunk text from disk for BM25 corpus...", flush=True)
    t0 = time.time()
    corpus_texts = []
    for i, (p_idx, offset) in enumerate(rows):
        p = paths[p_idx]
        corpus_texts.append(reconstruct_chunk_text(p["base_dir"], p["rel_path"], offset))
        if i % 50000 == 0:
            print(f"[bench] reconstructed {i:,}/{n:,}", flush=True)
    print(f"[bench] reconstruction done in {time.time()-t0:.1f}s ({len(_file_cache):,} unique files read)", flush=True)

    print("[bench] tokenizing + building BM25 index...", flush=True)
    t0 = time.time()
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"[bench] BM25 index built in {time.time()-t0:.1f}s", flush=True)

    print("[bench] loading embedding model for queries...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"\n[bench] {len(QUERIES)} queries -- dense-only (baseline) vs retrieve-then-rerank "
          f"(top-{N_CANDIDATES} dense, alpha={ALPHA}) top-{K}:\n", flush=True)
    for repo, q in QUERIES:
        qvec = model.encode([q], normalize_embeddings=True)[0].astype("float32")
        sims = vecs @ qvec
        dense_order = np.argsort(-sims)
        top_dense = dense_order[:K]

        candidates = dense_order[:N_CANDIDATES]
        cand_dense = sims[candidates]
        cand_bm25 = bm25.get_scores(tokenize(q))[candidates]

        d_min, d_max = cand_dense.min(), cand_dense.max()
        b_min, b_max = cand_bm25.min(), cand_bm25.max()
        dense_norm = (cand_dense - d_min) / (d_max - d_min + 1e-9)
        bm25_norm = (cand_bm25 - b_min) / (b_max - b_min + 1e-9)

        combined = ALPHA * dense_norm + (1 - ALPHA) * bm25_norm
        rerank_order = np.argsort(-combined)[:K]
        top_rerank = candidates[rerank_order]

        print(f"=== [{repo}] \"{q}\" ===", flush=True)
        print("  DENSE-ONLY (baseline):", flush=True)
        for idx in top_dense[:3]:
            p_idx, offset = rows[idx]
            p = paths[p_idx]
            print(f"    {p['rel_path']} (offset {offset}, score {sims[idx]:.3f})", flush=True)
        print("  RERANK (dense top-30, then dense+BM25 blend):", flush=True)
        for j, idx in enumerate(top_rerank[:3]):
            p_idx, offset = rows[idx]
            p = paths[p_idx]
            ci = list(candidates).index(idx)
            print(f"    {p['rel_path']} (offset {offset}, dense={sims[idx]:.3f}, "
                  f"bm25={cand_bm25[ci]:.2f}, combined={combined[rerank_order[j]]:.3f})", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
