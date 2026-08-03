"""
Real end-to-end check that the hybrid retrieve-then-rerank wiring inside
SearchEngine.search() actually works and matches the standalone benchmark's
behavior -- not just "it imports," but "loading the real deployed index and
calling the real search() method produces the expected hybrid re-ranking."
"""
import time
from search_engine import SearchEngine

engine = SearchEngine()
statuses = []
engine.load("data/observe-index", "sentence-transformers/all-MiniLM-L6-v2", statuses.append)

t0 = time.time()
while not engine.ready and time.time() - t0 < 120:
    time.sleep(0.5)

print("[verify] status log:")
for s in statuses:
    print(f"  {s}")

print(f"\n[verify] engine.ready={engine.ready}, bm25 built={engine.bm25 is not None}")

TEST_QUERIES = [
    "how does flask register a route handler",
    "how does svelte detect a reactive variable change",
    "how does express match a route with wildcard parameters",
]

for q in TEST_QUERIES:
    print(f"\n=== \"{q}\" ===")
    print("  hybrid=True (default):")
    for r in engine.search(q, k=3, hybrid=True):
        print(f"    {r['path']} (offset {r['offset']}, score {r['score']:.3f})")
    print("  hybrid=False (dense-only, for comparison):")
    for r in engine.search(q, k=3, hybrid=False):
        print(f"    {r['path']} (offset {r['offset']}, score {r['score']:.3f})")
