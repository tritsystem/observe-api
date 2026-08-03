"""
Does real function/class-boundary chunking (ast_chunker.py) actually
retrieve better than the fixed-window chunking search_engine.py's
build_index() does today? Same corpus, same stock embedding model, same
20 queries as the earlier benchmarks -- only the chunking strategy differs.

Baseline: the ALREADY-DEPLOYED float32 index (data/observe-index) -- no
need to rebuild it, it's sitting on disk from the real deployment.
Candidate: freshly built here using ast_chunker.py, falling back to the
same fixed-window logic for unsupported file types / files with zero
detected definitions, so coverage doesn't regress.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ast_chunker import ast_chunks

MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

EXTS = {".py",".gd",".js",".ts",".cs",".rs",".go",
        ".c",".cpp",".h",".java",".lua",".rb",".php",
        ".swift",".kt",".dart",".zig",".md",".sh",".ps1"}
SKIP = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", "target", "models", "search_index",
    "ai_files", "addons",
    "AppData", "Temp", "Windows", "Program Files",
    "Program Files (x86)", "ProgramData",
    "$Recycle.Bin", "System Volume Information",
    "msys64", "mingw64", "mingw32", "Anaconda3", "miniconda3",
    "site-packages", ".cargo", ".rustup",
    ".nuget", ".gradle", ".m2",
    "Ableton", "Steam", "steamapps", "Epic Games",
    "Adobe", "Spotify",
}
SKIP_PATTERNS = ("_files", "_assets")


def chunk_file(text, ext):
    """AST-aware chunks where supported and non-empty, else the same
    fixed-window fallback build_index() already uses."""
    result = ast_chunks(text, ext)
    if result is not None:
        return result
    out = []
    for i in range(0, len(text), 700):
        chunk = text[i:i+800]
        if len(chunk.strip()) > 50:
            out.append((i, chunk))
    return out


def scan_and_chunk(scan_dirs):
    chunks = []
    ast_count = 0
    fallback_count = 0
    for base in scan_dirs:
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")
                       and not any(p in d for p in SKIP_PATTERNS)]
            for fname in fnames:
                ext = Path(fname).suffix.lower()
                if ext not in EXTS:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    text = open(fpath, encoding="utf-8", errors="ignore").read()
                    if len(text.strip()) < 100:
                        continue
                    rel = os.path.relpath(fpath, base)
                    used_ast = ast_chunks(text, ext) is not None
                    if used_ast:
                        ast_count += 1
                    else:
                        fallback_count += 1
                    for offset, chunk in chunk_file(text, ext):
                        chunks.append({
                            "text": f"file:{rel}\n{chunk}",
                            "rel_path": rel,
                            "base_dir": base,
                            "offset": offset,
                        })
                except Exception:
                    pass
    print(f"[chunk] {ast_count:,} files AST-chunked, {fallback_count:,} files fixed-window fallback", flush=True)
    return chunks


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
    manifest = json.load(open("repo_manifest.json"))
    scan_dirs = list(manifest.values())

    print("[bench] loading baseline (deployed fixed-window) index...", flush=True)
    baseline_vecs = np.load("data/observe-index/vectors_float32.npy").astype("float32")
    baseline_meta = json.load(open("data/observe-index/metadata.json", encoding="utf-8"))
    baseline_paths = baseline_meta["paths"]
    baseline_rows = baseline_meta["chunks"]
    print(f"[bench] baseline: {len(baseline_rows):,} chunks", flush=True)

    print("[bench] AST-chunking the same corpus...", flush=True)
    chunks = scan_and_chunk(scan_dirs)
    print(f"[bench] ast-chunked corpus: {len(chunks):,} chunks", flush=True)

    print("[bench] loading model...", flush=True)
    model = SentenceTransformer(MODEL_PATH)

    texts = [c["text"] for c in chunks]
    bs = 256
    vecs = []
    t0 = time.time()
    for i in range(0, len(texts), bs):
        v = model.encode(texts[i:i+bs], normalize_embeddings=True, show_progress_bar=False)
        vecs.append(v)
        if (i // bs) % 100 == 0:
            print(f"[bench] embedding... {min(i+bs, len(texts)):,}/{len(texts):,}", flush=True)
    ast_vecs = np.vstack(vecs).astype("float32")
    print(f"[bench] ast embedding done in {time.time()-t0:.1f}s", flush=True)

    K = 5
    print(f"\n[bench] {len(QUERIES)} queries -- baseline (fixed-window) vs ast-chunked top-{K}:\n", flush=True)
    for repo, q in QUERIES:
        qvec = model.encode([q], normalize_embeddings=True)[0].astype("float32")

        sims_base = baseline_vecs @ qvec
        top_base = np.argsort(-sims_base)[:K]

        sims_ast = ast_vecs @ qvec
        top_ast = np.argsort(-sims_ast)[:K]

        print(f"=== [{repo}] \"{q}\" ===", flush=True)
        print("  BASELINE (fixed-window):", flush=True)
        for idx in top_base[:3]:
            p_idx, offset = baseline_rows[idx]
            p = baseline_paths[p_idx]
            print(f"    {p['rel_path']} (offset {offset}, score {sims_base[idx]:.3f})", flush=True)
        print("  AST-CHUNKED:", flush=True)
        for idx in top_ast[:3]:
            c = chunks[idx]
            print(f"    {c['rel_path']} (offset {c['offset']}, score {sims_ast[idx]:.3f})", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
