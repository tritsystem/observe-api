"""
Does hybrid search (dense embedding + BM25 lexical, combined via
Reciprocal Rank Fusion) retrieve better than pure dense embedding search
alone?

Uses the EXACT same chunks as the deployed baseline index -- same
embedding vectors already on disk (data/observe-index/vectors_float32.npy),
same chunk boundaries (path + offset from metadata.json). Chunk TEXT isn't
stored in the index (only path+offset, regenerated lazily from disk), so
it's reconstructed here by re-reading each file and re-slicing the exact
same [offset:offset+800] window build_index() used. This isolates the
retrieval/ranking method as the ONLY variable -- chunking strategy is held
constant, since that question was already answered (and rejected) by
_benchmark_ast_chunking.py.

RRF (Reciprocal Rank Fusion): a standard, parameter-light way to combine
two differently-scaled rankings (cosine similarity vs BM25 score) without
needing to normalize or tune relative weights between them --
score(doc) = sum over each ranker of 1/(k + rank). k=60 is the standard
default from the original RRF literature (Cormack et al. 2009), not a
number tuned against this corpus.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

RRF_K = 60
K = 5


def tokenize(text):
    # Lowercase + split on non-alnum; also split snake_case boundaries
    # into sub-tokens (keeping the whole token too) so a natural-language
    # query has a better chance of lexically matching identifier pieces,
    # not just whole-identifier exact matches.
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

    print(f"\n[bench] {len(QUERIES)} queries -- dense-only (baseline) vs hybrid(RRF) top-{K}:\n", flush=True)
    for repo, q in QUERIES:
        qvec = model.encode([q], normalize_embeddings=True)[0].astype("float32")
        sims = vecs @ qvec
        dense_order = np.argsort(-sims)

        bm25_scores = bm25.get_scores(tokenize(q))
        bm25_order = np.argsort(-bm25_scores)

        dense_rank_pos = np.empty(n, dtype=np.int64)
        dense_rank_pos[dense_order] = np.arange(n)
        bm25_rank_pos = np.empty(n, dtype=np.int64)
        bm25_rank_pos[bm25_order] = np.arange(n)

        rrf_score = 1.0 / (RRF_K + dense_rank_pos + 1) + 1.0 / (RRF_K + bm25_rank_pos + 1)
        top_hybrid = np.argsort(-rrf_score)[:K]
        top_dense = dense_order[:K]

        print(f"=== [{repo}] \"{q}\" ===", flush=True)
        print("  DENSE-ONLY (baseline):", flush=True)
        for idx in top_dense[:3]:
            p_idx, offset = rows[idx]
            p = paths[p_idx]
            print(f"    {p['rel_path']} (offset {offset}, score {sims[idx]:.3f})", flush=True)
        print("  HYBRID (dense+BM25 RRF):", flush=True)
        for idx in top_hybrid[:3]:
            p_idx, offset = rows[idx]
            p = paths[p_idx]
            print(f"    {p['rel_path']} (offset {offset}, dense={sims[idx]:.3f}, bm25={bm25_scores[idx]:.2f})", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
