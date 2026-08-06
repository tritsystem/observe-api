"""
Real, measured concurrent-load benchmark against /v1/commerce/search --
not the fast pytest suite (which never exercises real network/thread
concurrency), run manually against a live instance.

Measured results this session (isolated local instance, 32-core
machine, one seeded listing, 300 requests / 30-way concurrency, a pool
of 30 distinct API keys round-robined so the per-key rate limiter
(rate_limit.py) isn't what's being measured):

  baseline (fresh connection per db.get_conn() call, torch default
  thread count):
    p50=2024ms  p95=4325ms  p99=4792ms  14.0 req/s

  + db.py connection reuse (thread-local, keyed by DB_PATH):
    p50=2078ms  p95=2745ms  p99=3363ms  13.7 req/s
    -- barely moved the median; real win was tail latency (p95/p99),
    not throughput. Connection-open overhead was a real but secondary
    cost, not the dominant one.

  + torch.set_num_threads(1) / set_num_interop_threads(1) (server.py):
    p50=1639ms  p95=2375ms  p99=2730ms  17.2 req/s
    -- the real dominant cause: PyTorch's CPU backend fanning EACH
    model.encode() call across many BLAS threads, so N concurrent
    requests compete for N x torch.get_num_threads() threads on real
    physical cores -- severe oversubscription. ~23% throughput
    improvement, meaningfully lower tail latency.

Honest, disclosed limitation of even the fixed state: p50=1639ms at
concurrency=30 is still ~17x worse than p50=94ms at concurrency=1.
The remaining gap is very likely Python's GIL serializing pure-Python
work in the request path (score blending, the Spikeling STDP loop in
commerce_spiking_memory.py) across Starlette's thread pool -- threads
don't give true parallelism for CPU-bound pure-Python code the way
multiple PROCESSES would. Not fixed here: real horizontal scaling
(multiple uvicorn workers or multiple hosts) needs the rate limiter
(currently in-memory/per-process, see rate_limit.py) and the FAISS
commerce index / Spikeling memory caches (currently in-memory/per-
process, see commerce_router.py's _commerce_indices/_key_memories) to
move to shared, cross-process-safe state first -- a real, larger
architecture change, not attempted here given zero real production
traffic today to justify the added complexity before it's needed.
"""
import argparse
import concurrent.futures
import statistics
import time

import requests


def one_search(base_url, api_key):
    t0 = time.perf_counter()
    status = None
    try:
        r = requests.post(
            f"{base_url}/v1/commerce/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"intent": "waterproof hiking boots for a muddy trail", "k": 5},
            timeout=30,
        )
        status = r.status_code
        ok = r.status_code == 200
    except Exception:
        ok = False
    return time.perf_counter() - t0, ok, status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--key-pool-size", type=int, default=30)
    args = ap.parse_args()

    key_pool = []
    for i in range(args.key_pool_size):
        resp = requests.post(
            f"{args.base_url}/v1/signup",
            json={"email": f"loadtest-concurrent-{i}-{int(time.time())}@observe-search.online"},
        ).json()
        key_pool.append(resp["api_key"])
    print(f"[load] {args.requests} requests, concurrency={args.concurrency}, {len(key_pool)} distinct keys, target={args.base_url}", flush=True)

    latencies, oks, statuses = [], 0, {}
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_search, args.base_url, key_pool[i % len(key_pool)]) for i in range(args.requests)]
        for fut in concurrent.futures.as_completed(futures):
            lat, ok, status = fut.result()
            latencies.append(lat)
            oks += 1 if ok else 0
            statuses[status] = statuses.get(status, 0) + 1
    wall = time.perf_counter() - t_start

    latencies.sort()
    def pct(p):
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx]

    print(f"[load] status codes: {statuses}", flush=True)
    print(f"[load] wall time: {wall:.2f}s, {args.requests/wall:.1f} req/s achieved", flush=True)
    print(f"[load] success: {oks}/{args.requests}", flush=True)
    print(f"[load] p50={pct(0.50)*1000:.0f}ms p95={pct(0.95)*1000:.0f}ms p99={pct(0.99)*1000:.0f}ms max={latencies[-1]*1000:.0f}ms", flush=True)
    print(f"[load] mean={statistics.mean(latencies)*1000:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
