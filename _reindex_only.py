"""One-off: rebuild the index from the already-cloned repos in repo_manifest.json,
skipping clone_all() to avoid re-cloning 15 repos that are already on disk."""
import json
import os
import time

from search_engine import SearchEngine

INDEX_DIR = os.environ.get("OBSERVE_INDEX_DIR", "./data/observe-index")
MODEL_PATH = os.environ.get("OBSERVE_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")

manifest = json.load(open("repo_manifest.json"))
engine = SearchEngine()
done = {"flag": False}

def on_status(msg):
    print(f"[index] {msg}", flush=True)

def on_done():
    done["flag"] = True

engine.build_index(list(manifest.values()), INDEX_DIR, on_status, on_done, model_path=MODEL_PATH, quantize=False)

start = time.time()
while not done["flag"]:
    if time.time() - start > 7200:
        raise TimeoutError("did not finish within 2 hours")
    time.sleep(2)

print("done, ready=", engine.ready, "chunks=", len(engine.metadata))
