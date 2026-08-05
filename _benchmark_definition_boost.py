"""
Tests the one part of Semble's approach NOT already tried/rejected in this
project: a definition-vs-reference reranking boost, layered on top of the
CURRENTLY DEPLOYED retrieve-then-rerank pipeline (dense top-30, then
alpha-blended with BM25) -- no chunking change, no fusion-method change,
no corpus reindex. Same 20-query/15-repo harness as
_benchmark_hybrid_rerank.py and _benchmark_ast_chunking.py, run against the
real deployed index, for a directly comparable result.

Signal: for each of the dense top-30 candidates, scan the chunk text for a
definition line (def/class/function/func/fn/public function/etc.) whose
defined name's tokens overlap the query's tokens (both sides tokenized with
camelCase + snake_case splitting, since react's own vocabulary -- useState,
useEffect -- is camelCase and the existing _tokenize() in search_engine.py
only splits snake_case). A candidate that IS the definition site gets a
flat additive boost before the existing alpha-blended rerank.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

N_CANDIDATES = 30
ALPHA = 0.7
K = 5
DEF_BOOST = 0.15  # additive, same scale as the [0,1] normalized combined score

DEF_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:public\s+|private\s+|protected\s+|static\s+)*"
    r"(?:def|class|function|func|fn|void|const|let|var)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)

STOPWORDS = {
    "how", "does", "do", "the", "a", "an", "is", "are", "where", "when",
    "code", "that", "for", "to", "of", "in", "on", "with", "and", "or",
    "handle", "handles", "handling",
}


def split_camel_snake(token):
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", token)
    return [p.lower() for p in re.split(r"[_\W]+", parts) if p]


def tokenize(text):
    raw = re.findall(r"[A-Za-z0-9]+", text.lower())
    tokens = list(raw)
    for tok in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        parts = [p for p in tok.split("_") if p]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def query_symbol_tokens(query):
    out = set()
    for word in re.findall(r"[A-Za-z0-9_]+", query):
        for t in split_camel_snake(word):
            if t not in STOPWORDS and len(t) > 2:
                out.add(t)
    return out


def is_definition_of(chunk_text, query_tokens):
    for m in DEF_RE.finditer(chunk_text):
        name_tokens = set(split_camel_snake(m.group(1)))
        if name_tokens & query_tokens:
            return True, m.group(1)
    return False, None


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
    return f"file:{rel_path}\n{text[offset:offset + 800]}"


QUERIES = [
    ("react", "how does useState schedule a re-render"),
    ("react", "where does react decide whether to skip a re-render for unchanged props"),
    ("react", "where is the code that diffs two virtual dom trees"),
    ("django", "how does django validate a model field before saving"),
    ("django", "where does django route an incoming url to a view"),
    ("django", "how does django's orm build a sql query from a queryset"),
    ("flask", "how does flask register a route handler"),
    ("fastapi", "how does fastapi validate a request body against a pydantic model"),
    ("express", "how does express match a route with wildcard parameters"),
    ("redis", "how does redis expire a key after a timeout"),
    ("redis", "how does redis persist data to disk"),
    ("numpy", "how does numpy broadcast arrays of different shapes"),
    ("pandas", "how does pandas handle missing values when merging two dataframes"),
    ("svelte", "how does svelte detect a reactive variable change"),
    ("vue", "how does vue's reactivity system track dependencies"),
    ("axios", "how does axios cancel an in-flight request"),
    ("gin", "how does gin chain middleware for a request"),
    ("cargo", "how does cargo resolve dependency version conflicts"),
    ("tokio", "how does tokio schedule an async task onto a worker thread"),
    ("laravel", "how does laravel handle database migrations"),
]


def main():
    print("[bench] loading deployed baseline index...", flush=True)
    vecs = np.load("data/observe-index/vectors_float32.npy").astype("float32")
    meta = json.load(open("data/observe-index/metadata.json", encoding="utf-8"))
    paths = meta["paths"]
    rows = meta["chunks"]
    n = len(rows)
    print(f"[bench] {n:,} chunks", flush=True)

    print("[bench] reconstructing chunk text + building BM25 (same as prior benchmarks)...", flush=True)
    t0 = time.time()
    corpus_texts = []
    for i, (p_idx, offset) in enumerate(rows):
        p = paths[p_idx]
        corpus_texts.append(reconstruct_chunk_text(p["base_dir"], p["rel_path"], offset))
        if i % 100000 == 0:
            print(f"[bench] reconstructed {i:,}/{n:,}", flush=True)
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"[bench] ready in {time.time() - t0:.1f}s ({len(_file_cache):,} unique files)", flush=True)

    print("[bench] loading embedding model...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"\n[bench] {len(QUERIES)} queries -- CURRENT DEPLOYED rerank vs "
          f"+definition-boost (boost={DEF_BOOST}):\n", flush=True)

    for repo, q in QUERIES:
        qvec = model.encode([q], normalize_embeddings=True)[0].astype("float32")
        sims = vecs @ qvec
        dense_order = np.argsort(-sims)
        candidates = dense_order[:N_CANDIDATES]
        cand_dense = sims[candidates]
        cand_bm25 = np.asarray(bm25.get_batch_scores(tokenize(q), candidates.tolist()))

        d_min, d_max = cand_dense.min(), cand_dense.max()
        b_min, b_max = cand_bm25.min(), cand_bm25.max()
        dense_norm = (cand_dense - d_min) / (d_max - d_min + 1e-9)
        bm25_norm = (cand_bm25 - b_min) / (b_max - b_min + 1e-9)
        baseline_combined = ALPHA * dense_norm + (1 - ALPHA) * bm25_norm

        qtoks = query_symbol_tokens(q)
        boosted_combined = baseline_combined.copy()
        def_hits = {}
        for ci, idx in enumerate(candidates):
            is_def, name = is_definition_of(corpus_texts[idx], qtoks)
            if is_def:
                boosted_combined[ci] += DEF_BOOST
                def_hits[idx] = name

        base_order = np.argsort(-baseline_combined)[:K]
        boost_order = np.argsort(-boosted_combined)[:K]
        top_base = candidates[base_order]
        top_boost = candidates[boost_order]

        print(f"=== [{repo}] \"{q}\" ===  (query symbol tokens: {sorted(qtoks)})", flush=True)
        print("  BASELINE (deployed rerank):", flush=True)
        for idx in top_base[:3]:
            p_idx, offset = rows[idx]
            p = paths[p_idx]
            print(f"    {p['rel_path']} (offset {offset})", flush=True)
        print("  +DEFINITION BOOST:", flush=True)
        for idx in top_boost[:3]:
            p_idx, offset = rows[idx]
            p = paths[p_idx]
            tag = f"  [DEF: {def_hits[idx]}]" if idx in def_hits else ""
            print(f"    {p['rel_path']} (offset {offset}){tag}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
