"""
Per-API-key token-bucket rate limiter -- an in-memory, single-process guard
against one key starving other concurrent callers.

The credit system already bounds total COST of abuse (a caller pays per
search), but nothing else bounds REQUEST RATE: a key with a large balance
could blast its whole balance in a tight loop, degrading /v1/search latency
for every other concurrent caller on this one process. This closes that gap
specifically -- it's a QoS/availability guard, not a cost control (credits
already cover cost).

Deliberately simple for v1: in-memory, per-process (fine for the current
single-process deployment; a multi-process/multi-host deployment would need
a shared store like Redis instead -- not built here, flagged rather than
guessed at).
"""
import os
import threading
import time

# Burst capacity and steady-state refill rate, tuned to allow normal agent
# usage (a few calls in quick succession) while capping a tight-loop caller
# well below what would meaningfully degrade other callers' latency on this
# process. Configurable via env vars without a code change.
CAPACITY = float(os.environ.get("OBSERVE_RATE_LIMIT_CAPACITY", "10"))
REFILL_PER_SEC = float(os.environ.get("OBSERVE_RATE_LIMIT_PER_SEC", "5"))

_buckets = {}
_lock = threading.Lock()


def allow(key: str) -> bool:
    """True if this call may proceed (consumes one token in that case).
    False if the caller is over rate and should get a 429."""
    now = time.monotonic()
    with _lock:
        tokens, last = _buckets.get(key, (CAPACITY, now))
        tokens = min(CAPACITY, tokens + (now - last) * REFILL_PER_SEC)
        if tokens < 1.0:
            _buckets[key] = (tokens, now)
            return False
        _buckets[key] = (tokens - 1.0, now)
        return True
