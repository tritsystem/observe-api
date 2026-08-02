"""
Honest benchmark: does ternary quantization cost real retrieval quality, or is
the 19.9x compression effectively free? Builds BOTH representations from the
exact same embeddings (float32 exact vs. ternary-quantize-then-unpack, the
same round-trip used at real search time) over the full real 15-repo corpus,
then compares top-k overlap on a real, fixed set of vocabulary-mismatch-style
queries spanning most of the 15 repos.

Not a benchmark vs. grep (already done elsewhere, see launch/show-hn.md) --
this specifically tests the thing that hasn't been measured yet: does
quantizing the index change what gets returned, using the SAME chunking and
SAME embedding model as the real production index (search_engine.py), just
keeping the pre-quantization float32 vectors around instead of discarding
them.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from search_engine import pack_ternary, unpack_ternary

MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# Identical to search_engine.py's build_index() -- kept in sync on purpose so
# this is a faithful test of the real production chunking, not a strawman.
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


def main():
    manifest = json.load(open("repo_manifest.json"))
    scan_dirs = list(manifest.values())
    repo_by_dir = {v: k for k, v in manifest.items()}

    print("[bench] scanning + chunking (same logic as search_engine.py)...", flush=True)
    chunks = scan_and_chunk(scan_dirs)
    print(f"[bench] {len(chunks):,} chunks", flush=True)

    print("[bench] loading model...", flush=True)
    model = SentenceTransformer(MODEL_PATH)

    texts = [c["text"] for c in chunks]
    bs = 256
    vecs = []
    t0 = time.time()
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        v = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vecs.append(v)
        if (i // bs) % 50 == 0:
            print(f"[bench] embedding... {min(i+bs, len(texts)):,}/{len(texts):,}", flush=True)
    vecs = np.vstack(vecs).astype("float32")
    print(f"[bench] embedding done in {time.time()-t0:.1f}s, shape={vecs.shape}", flush=True)

    # Exact same quantization as search_engine.py's build_index()
    t = 0.7 * np.abs(vecs).mean()
    trit_vecs = np.where(vecs > t, 1, np.where(vecs < -t, -1, 0)).astype("int8")
    packed = pack_ternary(trit_vecs)
    ternary_roundtrip = unpack_ternary(packed, vecs.shape[1]).astype("float32")

    print(f"[bench] float32 index: {vecs.nbytes/1e6:.2f}MB, "
          f"ternary packed: {packed.nbytes/1e6:.2f}MB "
          f"({vecs.nbytes/packed.nbytes:.1f}x smaller)", flush=True)

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

    K = 10
    overlaps = []
    top1_matches = 0
    print(f"\n[bench] {len(QUERIES)} queries, top-{K} overlap (float32-exact vs ternary-roundtrip):\n", flush=True)
    for repo, q in QUERIES:
        qvec = model.encode([q], normalize_embeddings=True)[0].astype("float32")

        sims_f32 = vecs @ qvec
        top_f32 = set(np.argsort(-sims_f32)[:K].tolist())

        sims_tri = ternary_roundtrip @ qvec
        top_tri = set(np.argsort(-sims_tri)[:K].tolist())

        overlap = len(top_f32 & top_tri)
        overlaps.append(overlap)

        top1_f32 = np.argmax(sims_f32)
        top1_tri = np.argmax(sims_tri)
        if top1_f32 == top1_tri:
            top1_matches += 1

        print(f"  [{repo:8s}] overlap@{K}: {overlap:2d}/{K}   top-1 match: {top1_f32 == top1_tri}   \"{q}\"", flush=True)

    avg_overlap = sum(overlaps) / len(overlaps)
    print(f"\n[bench] RESULT: average top-{K} overlap = {avg_overlap:.1f}/{K} "
          f"({avg_overlap/K*100:.0f}%), top-1 exact match rate = {top1_matches}/{len(QUERIES)} "
          f"({top1_matches/len(QUERIES)*100:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
