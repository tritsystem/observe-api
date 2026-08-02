"""
Standalone search engine, extracted from OBSERVE's trit_app.py (the desktop
GUI) so a headless server can import it without pulling in tkinter. Class
body is verbatim from the original -- it never actually depended on the GUI,
just lived in the same file. See the "extraction" note at the bottom for
what changed vs. the original.

Ternary bit-packing: packs 5 trits {-1,0,+1} into 1 byte (3^5=243 < 256) --
true ~1.6 bits/value, the real ~20x compression vs float32 (32 bits/value).
"""
import json
import os
import threading
from pathlib import Path


def pack_ternary(trits):
    """trits: int8 array (N, D) with values in {-1,0,1} -> packed uint8 (N, ceil(D/5))"""
    import numpy as np
    digits = (trits + 1).astype(np.int32)   # 0,1,2
    N, D = digits.shape
    pad = (-D) % 5
    if pad:
        digits = np.pad(digits, ((0, 0), (0, pad)), constant_values=1)  # pad with trit 0
    G = digits.shape[1] // 5
    digits = digits.reshape(N, G, 5)
    weights = np.array([1, 3, 9, 27, 81], dtype=np.int32)
    packed = (digits * weights).sum(axis=2).astype(np.uint8)
    return packed


def unpack_ternary(packed, orig_dim):
    """packed: uint8 array (N, G) -> trits int8 (N, orig_dim) with values in {-1,0,1}"""
    import numpy as np
    N, G = packed.shape
    vals = packed.astype(np.int32)
    out = np.zeros((N, G, 5), dtype=np.int8)
    for i in range(5):
        out[:, :, i] = (vals % 3) - 1
        vals //= 3
    return out.reshape(N, G * 5)[:, :orig_dim]


class SearchEngine:
    def __init__(self):
        self.index      = None
        self.metadata   = []
        self.path_table = []
        self.model      = None
        self.ready      = False
        self.status     = "Not initialized"

    def load(self, index_dir, model_path, on_status):
        def _load():
            try:
                import faiss
                import numpy as np
                from sentence_transformers import SentenceTransformer

                on_status("Loading model...")
                self.model = SentenceTransformer(model_path)

                f32_path  = os.path.join(index_dir, "vectors_float32.npy")
                trit_path = os.path.join(index_dir, "vectors_ternary.npy")
                trit_meta_path = os.path.join(index_dir, "vectors_meta.json")
                idx_path  = os.path.join(index_dir, "faiss.index")
                meta_path = os.path.join(index_dir, "metadata.json")

                if os.path.exists(f32_path) and os.path.exists(meta_path):
                    # Real measured tradeoff (see _benchmark_ternary_vs_float32.py):
                    # ternary quantization changed the top-1 result 40% of the
                    # time and top-10 overlap averaged only 70% vs. this exact
                    # float32 index, on 20 real queries across all 15 repos --
                    # not free. At this corpus's actual size (353MB), disk/RAM
                    # was never the real constraint, so this deployment serves
                    # the unquantized index instead of trading real result
                    # quality for a compression win it didn't need.
                    on_status("Loading float32 index...")
                    self.index    = np.load(f32_path).astype("float32")
                    raw = json.load(open(meta_path, encoding="utf-8"))
                    self.path_table = raw.get("paths", [])
                    self.metadata   = raw.get("chunks", [])
                    on_status(f"Ready - {len(self.metadata):,} chunks indexed "
                              f"({self.index.nbytes/1e6:.2f}MB on disk, full precision). Start searching!")
                elif os.path.exists(trit_path) and os.path.exists(meta_path):
                    on_status("Loading ternary-compressed index...")
                    packed = np.load(trit_path)
                    dim    = json.load(open(trit_meta_path)).get("dim", 384) if os.path.exists(trit_meta_path) else 384
                    disk_mb = packed.nbytes / 1e6
                    # Unpack ONCE here (disk stays 19.9x compressed) so every
                    # search afterward hits a plain float32 matmul, no per-query
                    # unpack cost -- balances disk savings with search speed.
                    self.index    = unpack_ternary(packed, dim).astype("float32")
                    raw = json.load(open(meta_path, encoding="utf-8"))
                    self.path_table = raw.get("paths", [])
                    self.metadata   = raw.get("chunks", [])
                    on_status(f"Ready - {len(self.metadata):,} chunks indexed "
                              f"({disk_mb:.2f}MB on disk, 19.9x compressed). Start searching!")
                elif os.path.exists(idx_path) and os.path.exists(meta_path):
                    on_status("Loading index...")
                    self.index    = faiss.read_index(idx_path)
                    self.metadata = json.load(open(meta_path, encoding="utf-8"))
                    on_status(f"Ready - {len(self.metadata):,} chunks indexed. Start searching!")
                else:
                    on_status("No index yet - add a directory then click INDEX CODEBASE")
            except Exception as e:
                on_status(f"Error: {e}")
            finally:
                # Loop waiting on self.ready must always terminate, even if
                # on_status() itself raises - a status-print failure must
                # never silently hang every caller forever.
                self.ready = True
        threading.Thread(target=_load, daemon=True).start()

    def search(self, query, k=10, base_dir_filter=None):
        if not self.ready or self.index is None or self.model is None:
            return []
        import numpy as np
        vec = self.model.encode([query], normalize_embeddings=True).astype("float32")

        # Filter to a specific project's chunks *before* taking top-k, not
        # after -- otherwise an unrelated, larger codebase in the same index
        # can crowd the target project out of the results entirely.
        chunk_mask = None
        if base_dir_filter and self.path_table:
            norm_filter = os.path.normcase(os.path.normpath(base_dir_filter))
            allowed_path_idx = {
                i for i, p in enumerate(self.path_table)
                if os.path.normcase(os.path.normpath(p["base_dir"])) == norm_filter
            }
            chunk_mask = np.array([
                isinstance(m, list) and m[0] in allowed_path_idx
                for m in self.metadata
            ])

        if isinstance(self.index, np.ndarray):
            # Ternary index, unpacked once at load time -- fast float32 matmul,
            # disk file is still 19.9x compressed (see load()/build_index()).
            sims = self.index @ vec[0]
            if chunk_mask is not None:
                sims = np.where(chunk_mask, sims, -np.inf)
            top  = np.argsort(-sims)[:min(k, len(sims))]
            scores, indices = sims[top], top
        else:
            scores0, indices0 = self.index.search(vec, min(k, self.index.ntotal))
            scores, indices = scores0[0], indices0[0]
        results = []
        for score, idx in zip(scores, indices):
            if score == -np.inf:
                continue
            if idx >= 0 and idx < len(self.metadata):
                m = self.metadata[idx]
                if isinstance(m, list) and self.path_table:
                    # Compact format: [path_idx, offset] -- regenerate
                    # preview lazily from the original file on disk.
                    p_idx, offset = m
                    p = self.path_table[p_idx]
                    rel_path = p["rel_path"]
                    preview = ""
                    try:
                        full = os.path.join(p["base_dir"], rel_path)
                        text = open(full, encoding="utf-8", errors="ignore").read()
                        preview = text[offset:offset+120].replace("\n", " ")
                    except Exception:
                        pass
                    results.append({"score": float(score), "path": rel_path, "preview": preview, "offset": offset})
                else:
                    # Legacy format fallback
                    results.append({
                        "score":   float(score),
                        "path":    m.get("rel_path", m.get("path", "?")),
                        "preview": m.get("preview", ""),
                        "offset":  m.get("offset"),
                    })
        return results

    def build_index(self, scan_dirs, index_dir, on_status, on_done, model_path="sentence-transformers/all-MiniLM-L6-v2", quantize=True):
        def _build():
            try:
                import faiss, numpy as np
                from sentence_transformers import SentenceTransformer

                if self.model is None:
                    on_status("Loading model...")
                    self.model = SentenceTransformer(model_path)

                EXTS = {".py",".gd",".js",".ts",".cs",".rs",".go",
                        ".c",".cpp",".h",".java",".lua",".rb",".php",
                        ".swift",".kt",".dart",".zig",".md",".sh",".ps1"}
                # Universal default skip-list -- folders that are virtually
                # never a user's own source code, regardless of whose
                # machine this runs on (installed software, OS internals,
                # package manager caches, build artifacts, web-page saves).
                SKIP = {
                    # Version control / build artifacts / common project conventions
                    ".git", "__pycache__", "node_modules", ".venv", "venv",
                    "dist", "build", "target", "models", "search_index",
                    "ai_files", "addons",
                    # Windows OS / system folders
                    "AppData", "Temp", "Windows", "Program Files",
                    "Program Files (x86)", "ProgramData",
                    "$Recycle.Bin", "System Volume Information",
                    # Dev toolchains / package manager caches (large, never
                    # the user's own code)
                    "msys64", "mingw64", "mingw32", "Anaconda3", "miniconda3",
                    "site-packages", ".cargo", ".rustup",
                    ".nuget", ".gradle", ".m2",
                    # Common large bundled creative/consumer software
                    "Ableton", "Steam", "steamapps", "Epic Games",
                    "Adobe", "Spotify",
                }

                chunks = []
                files  = 0
                on_status("Scanning files...")

                SKIP_PATTERNS = ("_files", "_assets")

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
                                # Chunk -- store only path + offset, not the
                                # full text/preview (regenerated lazily on
                                # display from the original file on disk).
                                for i in range(0, len(text), 700):
                                    chunk = text[i:i+800]
                                    if len(chunk.strip()) > 50:
                                        chunks.append({
                                            "text":     f"file:{rel}\n{chunk}",
                                            "rel_path": rel,
                                            "base_dir": base,
                                            "offset":   i,
                                        })
                                files += 1
                            except: pass

                on_status(f"Encoding {len(chunks):,} chunks from {files:,} files...")
                os.makedirs(index_dir, exist_ok=True)

                texts = [c["text"] for c in chunks]
                bs    = 128
                vecs  = []
                for i in range(0, len(texts), bs):
                    batch = texts[i:i+bs]
                    v = self.model.encode(batch, normalize_embeddings=True,
                                          show_progress_bar=False)
                    vecs.append(v)
                    pct = (i+bs) / len(texts) * 100
                    on_status(f"Encoding... {min(pct,100):.0f}%  ({min(i+bs, len(texts)):,}/{len(texts):,})")

                vecs = np.vstack(vecs).astype("float32")

                old_index = os.path.join(index_dir, "faiss.index")
                if os.path.exists(old_index):
                    os.remove(old_index)

                if quantize:
                    # Ternary compression: quantize to {-1,0,+1}, then bit-pack
                    # 5 trits/byte (3^5=243<256) for the real ~20x reduction vs
                    # float32. Query stays float32 at search time (asymmetric).
                    # Measured cost (_benchmark_ternary_vs_float32.py): only
                    # 70% top-10 overlap / 60% top-1 match vs. the exact
                    # float32 index on 20 real queries -- worth it only when
                    # disk/RAM at this corpus's size is the real constraint.
                    t = 0.7 * np.abs(vecs).mean()
                    trit_vecs = np.where(vecs > t, 1, np.where(vecs < -t, -1, 0)).astype("int8")
                    packed    = pack_ternary(trit_vecs)
                    stale_f32 = os.path.join(index_dir, "vectors_float32.npy")
                    if os.path.exists(stale_f32):
                        os.remove(stale_f32)
                    np.save(os.path.join(index_dir, "vectors_ternary.npy"), packed)
                    json.dump({"dim": int(vecs.shape[1])},
                              open(os.path.join(index_dir, "vectors_meta.json"), "w"))
                    self.index = trit_vecs.astype("float32")
                    size_mb  = packed.nbytes / 1e6
                    float_mb = vecs.nbytes / 1e6
                    size_note = f"{size_mb:.2f}MB packed-ternary vs {float_mb:.2f}MB float32, {float_mb/size_mb:.1f}x smaller"
                else:
                    stale_trit = os.path.join(index_dir, "vectors_ternary.npy")
                    stale_meta = os.path.join(index_dir, "vectors_meta.json")
                    if os.path.exists(stale_trit): os.remove(stale_trit)
                    if os.path.exists(stale_meta): os.remove(stale_meta)
                    np.save(os.path.join(index_dir, "vectors_float32.npy"), vecs)
                    self.index = vecs
                    size_note = f"{vecs.nbytes/1e6:.2f}MB float32, full precision (no quantization)"

                # Deduplicate paths into a small table; store only a path
                # index + byte offset per chunk (preview regenerated lazily
                # from the original file on disk, not cached here).
                path_table = []
                path_idx   = {}
                rows = []
                for c in chunks:
                    key = (c["base_dir"], c["rel_path"])
                    if key not in path_idx:
                        path_idx[key] = len(path_table)
                        path_table.append({"base_dir": c["base_dir"], "rel_path": c["rel_path"]})
                    rows.append([path_idx[key], c["offset"]])

                json.dump({"paths": path_table, "chunks": rows},
                          open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8"),
                          ensure_ascii=False, separators=(",", ":"))

                self.metadata   = rows
                self.path_table = path_table
                self.ready      = True
                on_status(f"Index complete -- {len(chunks):,} chunks, {files:,} files ({size_note})")
            except Exception as e:
                on_status(f"Index error: {e}")
            finally:
                # A caller polling on_done() (e.g. index_repos.py) must never
                # hang until its own timeout just because this thread failed --
                # on_done() fires exactly once whether _build succeeded or not.
                on_done()
        threading.Thread(target=_build, daemon=True).start()

    def load_blocking(self, index_dir, model_path, timeout=180):
        """Synchronous wrapper around load() for server startup -- blocks
        until self.ready or timeout. The class's own load()/build_index()
        are thread-based (built for a GUI event loop that can't block), but
        a server's startup path can just wait."""
        import time
        statuses = []
        self.load(index_dir, model_path, statuses.append)
        start = time.time()
        while not self.ready:
            if time.time() - start > timeout:
                raise TimeoutError(f"SearchEngine.load() did not finish within {timeout}s -- last status: {statuses[-1] if statuses else '(none)'}")
            time.sleep(0.2)
        return statuses[-1] if statuses else None


# Extraction notes (vs. the original trit_app.py):
#   - No behavior changes to SearchEngine itself -- verbatim class body.
#   - Dropped: tkinter GUI, MatrixRain, TritSearchApp, theme dicts, PyInstaller
#     packaging notes -- none of it is needed by a headless server.
#   - Added: load_blocking(), a synchronous convenience wrapper for server
#     startup. The original load()/build_index() stay thread-based and
#     unchanged (trit_mcp_server.py's own _ensure_loaded() already proved
#     this poll-until-ready pattern works fine for a server context).
