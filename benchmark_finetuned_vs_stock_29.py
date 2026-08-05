"""
Does the fine-tuned code-minilm-v2 checkpoint (trained on the real 29-repo
corpus this deployment actually serves) retrieve better than the stock
all-MiniLM-L6-v2 that server.py's MODEL_PATH defaults to?

Same methodology as the original 15-repo _benchmark_finetuned_vs_stock.py --
both models embed the FULL real corpus and the same queries; top-10 overlap
and top-1 agreement are reported, plus each model's actual top-3 results per
query, since there's no labeled ground truth to compute recall/precision
against. Query set expanded from 15 to 29 -- one per repo, so every repo in
the actual deployed corpus gets exercised, not just the original subset.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

STOCK_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNED_MODEL_PATH = "./models/code-minilm-v2"

EXTS = {".py",".gd",".js",".ts",".cs",".rs",".go",
        ".c",".cpp",".h",".java",".lua",".rb",".php",
        ".swift",".kt",".dart",".zig",".md",".sh",".ps1"}
SKIP = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", "target", "models", "search_index",
}
SKIP_PATTERNS = ("_files", "_assets")


def scan_and_chunk(scan_dirs):
    chunks = []
    for base in scan_dirs:
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")
                       and not any(p in d for p in SKIP_PATTERNS)]
            for fname in fnames:
                if Path(fname).suffix.lower() not in EXTS:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    text = open(fpath, encoding="utf-8", errors="ignore").read()
                    if len(text.strip()) < 100:
                        continue
                    rel = os.path.relpath(fpath, base)
                    for i in range(0, len(text), 700):
                        chunk = text[i:i+800]
                        if len(chunk.strip()) > 50:
                            chunks.append({
                                "text": f"file:{rel}\n{chunk}",
                                "rel_path": rel,
                                "base_dir": base,
                                "offset": i,
                            })
                except Exception:
                    pass
    return chunks


def embed_all(model, texts, bs=256, label=""):
    vecs = []
    t0 = time.time()
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        v = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vecs.append(v)
        if (i // bs) % 100 == 0:
            print(f"[{label}] embedding... {min(i+bs, len(texts)):,}/{len(texts):,}", flush=True)
    print(f"[{label}] done in {time.time()-t0:.1f}s", flush=True)
    return np.vstack(vecs).astype("float32")


QUERIES = [
    ("react",        "how does useState schedule a re-render"),
    ("react",        "where does react decide whether to skip a re-render for unchanged props"),
    ("django",       "how does django validate a model field before saving"),
    ("django",       "where does django route an incoming url to a view"),
    ("flask",        "how does flask register a route handler"),
    ("fastapi",      "how does fastapi validate a request body against a pydantic model"),
    ("express",      "how does express match a route with wildcard parameters"),
    ("redis",        "how does redis expire a key after a timeout"),
    ("numpy",        "how does numpy broadcast arrays of different shapes"),
    ("pandas",       "how does pandas handle missing values when merging two dataframes"),
    ("svelte",       "how does svelte detect a reactive variable change"),
    ("vue",          "how does vue's reactivity system track dependencies"),
    ("axios",        "how does axios cancel an in-flight request"),
    ("gin",          "how does gin chain middleware for a request"),
    ("cargo",        "how does cargo resolve dependency version conflicts"),
    ("tokio",        "how does tokio schedule an async task onto a worker thread"),
    ("laravel",      "how does laravel handle database migrations"),
    ("rails",        "how does rails route an incoming request to a controller action"),
    ("spring-boot",  "how does spring boot autowire a dependency into a bean"),
    ("aspnetcore",   "how does aspnet core handle dependency injection for a service"),
    ("nestjs",       "how does nestjs apply a decorator to register a route handler"),
    ("nextjs",       "how does next.js handle server-side rendering for a page"),
    ("symfony",      "how does symfony resolve a service from the container"),
    ("phoenix",      "how does phoenix handle a websocket channel connection"),
    ("vapor",        "how does vapor register a route handler in swift"),
    ("ktor",         "how does ktor handle routing for an http request"),
    ("pytest",       "how does pytest discover and collect test functions"),
    ("actix-web",    "how does actix web handle an incoming http request asynchronously"),
    ("scikit-learn", "how does scikit-learn implement cross validation for a model"),
    ("langchain",    "how does langchain chain multiple llm calls together"),
    ("curl",         "how does curl handle following an http redirect"),
]


def main():
    manifest = json.load(open("repo_manifest.json"))
    scan_dirs = list(manifest.values())

    print("[bench] scanning + chunking...", flush=True)
    chunks = scan_and_chunk(scan_dirs)
    print(f"[bench] {len(chunks):,} chunks", flush=True)
    texts = [c["text"] for c in chunks]

    print("[bench] loading stock model...", flush=True)
    stock = SentenceTransformer(STOCK_MODEL_PATH)
    print("[bench] loading fine-tuned model...", flush=True)
    finetuned = SentenceTransformer(FINETUNED_MODEL_PATH)

    vecs_stock = embed_all(stock, texts, label="stock")
    vecs_ft    = embed_all(finetuned, texts, label="finetuned")

    K = 10
    overlaps = []
    top1_matches = 0
    print(f"\n[bench] {len(QUERIES)} queries, top-{K} overlap (stock vs fine-tuned):\n", flush=True)
    for repo, q in QUERIES:
        q_stock = stock.encode([q], normalize_embeddings=True)[0].astype("float32")
        q_ft    = finetuned.encode([q], normalize_embeddings=True)[0].astype("float32")

        sims_stock = vecs_stock @ q_stock
        top_stock  = np.argsort(-sims_stock)[:K]
        sims_ft    = vecs_ft @ q_ft
        top_ft     = np.argsort(-sims_ft)[:K]

        overlap = len(set(top_stock.tolist()) & set(top_ft.tolist()))
        overlaps.append(overlap)
        match = top_stock[0] == top_ft[0]
        top1_matches += match

        print(f"  [{repo:13s}] overlap@{K}: {overlap:2d}/{K}   top-1 match: {match}   \"{q}\"", flush=True)

        def _fmt(idx):
            c = chunks[idx]
            return f"{c['rel_path']}(@{c['offset']})"

        print(f"      stock    top1-3: {[_fmt(j) for j in top_stock[:3]]}", flush=True)
        print(f"      finetune top1-3: {[_fmt(j) for j in top_ft[:3]]}", flush=True)

    avg_overlap = sum(overlaps) / len(overlaps)
    print(f"\n[bench] RESULT: average top-{K} overlap = {avg_overlap:.1f}/{K} "
          f"({avg_overlap/K*100:.0f}%), top-1 exact match rate = {top1_matches}/{len(QUERIES)} "
          f"({top1_matches/len(QUERIES)*100:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
