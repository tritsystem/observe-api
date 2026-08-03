"""Build a small local OBSERVE index over observe-api's OWN codebase (not
the 15-repo shared corpus, not a git clone -- just the real source files
already on disk here), so OBSERVE can search itself. Same chunking as
build_index() (700-stride/800-window), same output format
(vectors_float32.npy + metadata.json) so SearchEngine.load() reads it
unchanged -- this is a scoping fix (build_index()'s default skip-list
doesn't exclude the local repos/ clone directory, so pointing it at "."
would re-embed the entire 232k-chunk shared corpus again), not a
different index format.
"""
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN_TARGETS = [
    ".",  # top-level loose .py files, handled specially (non-recursive)
    "observe_search_tools",
    "observe_search_mcp",
    "tests",
    "landing",
    "launch",
]
EXCLUDE_DIRS = {"repos", "data", "models", "checkpoints", "private",
                "__pycache__", ".pytest_cache", ".venv", ".git"}
EXTS = {".py", ".md", ".html", ".txt", ".json"}


def gather_files():
    files = []
    # top-level: non-recursive, just loose files directly in HERE
    for fname in os.listdir(HERE):
        fpath = os.path.join(HERE, fname)
        if os.path.isfile(fpath) and Path(fname).suffix.lower() in EXTS:
            files.append(fpath)
    for sub in SCAN_TARGETS[1:]:
        base = os.path.join(HERE, sub)
        if not os.path.isdir(base):
            continue
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fname in fnames:
                if Path(fname).suffix.lower() in EXTS:
                    files.append(os.path.join(root, fname))
    return files


def main():
    files = gather_files()
    print(f"[self-index] {len(files)} real observe-api source files found", flush=True)

    chunks = []
    for fpath in files:
        rel = os.path.relpath(fpath, HERE)
        try:
            text = open(fpath, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if len(text.strip()) < 50:
            continue
        for i in range(0, len(text), 700):
            chunk = text[i:i + 800]
            if len(chunk.strip()) > 50:
                chunks.append({"text": f"file:{rel}\n{chunk}", "rel_path": rel, "offset": i})

    print(f"[self-index] {len(chunks):,} chunks from {len(files)} files", flush=True)

    print("[self-index] loading model...", flush=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    vecs = []
    bs = 128
    for i in range(0, len(texts), bs):
        v = model.encode(texts[i:i + bs], normalize_embeddings=True, show_progress_bar=False)
        vecs.append(v)
    vectors = np.vstack(vecs).astype("float32")

    out_dir = os.path.join(HERE, "data", "self-index")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "vectors_float32.npy"), vectors)
    meta = {
        "paths": [{"base_dir": HERE, "rel_path": c["rel_path"]} for c in chunks],
        "chunks": [],
    }
    # dedupe paths into a path table, same compact [path_idx, offset] format
    # SearchEngine.load()/search() already expects.
    path_index = {}
    path_table = []
    chunk_rows = []
    for c in chunks:
        key = c["rel_path"]
        if key not in path_index:
            path_index[key] = len(path_table)
            path_table.append({"base_dir": HERE, "rel_path": key})
        chunk_rows.append([path_index[key], c["offset"]])
    meta = {"paths": path_table, "chunks": chunk_rows}
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"[self-index] wrote {len(chunks):,} chunks, {len(path_table)} files -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
