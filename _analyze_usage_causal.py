#!/usr/bin/env python3
"""Real causal analysis of observe-api's own usage_log, via methodlm's ADJUST
(backdoor adjustment + Cinelli-Hazlett robustness value). Re-run as more data
accumulates -- results below were computed on n=319 real usage_log rows,
2026-08-04.

Findings as of that run:
- query_length -> result_count: raw corr -0.46, adjusted partial -0.46
  (unchanged), RV=0.40 (robust -- well above the RV<0.10 fragile line).
  Real, adjustment-checked signal: longer queries return fewer results.
- has_repo_filter: zero variance across all 319 rows -- no real user has
  used the repo_filter parameter yet. Worth knowing for product decisions
  (is it discoverable? worth keeping?), not a methodlm finding per se.
- 0/319 searches returned zero results -- search isn\'t silently failing.

Requires methodlm on the path (METHODLM_PATH env var, or same-machine
default below) and a copy of observe_api.db (never point this at the live
file directly -- copy it first, e.g. via ata /tmp/methodlm_scratch/).
"""
import sqlite3, sys, os, datetime
import numpy as np

METHODLM_PATH = os.environ.get("METHODLM_PATH", "/mnt/c/Users/gbran/OneDrive/Documents/methodlm")
sys.path.insert(0, METHODLM_PATH)
import methodlm


def load_usage_features(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT query, repo_filter, result_count, created_at FROM usage_log")
    rows = cur.fetchall()

    query_length, has_repo_filter, hour, result_count = [], [], [], []
    for q, rf, rc, ca in rows:
        if rc is None or q is None:
            continue
        query_length.append(len(q))
        has_repo_filter.append(1.0 if rf else 0.0)
        result_count.append(float(rc))
        try:
            h = datetime.datetime.fromisoformat(ca.replace("Z", "+00:00")).hour
        except Exception:
            h = 12
        hour.append(float(h))

    return {
        "result_count": np.array(result_count),
        "query_length": np.array(query_length, dtype=float),
        "has_repo_filter": np.array(has_repo_filter, dtype=float),
        "hour": np.array(hour, dtype=float),
    }, len(rows)


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/methodlm_scratch/observe_api_copy.db"
    data, n_raw = load_usage_features(db_path)
    n = len(data["result_count"])
    print(f"Loaded {n_raw} usage_log rows, {n} usable after cleaning.")
    print("Zero-result searches:", int((data["result_count"] == 0).sum()), "/", n)
    print("has_repo_filter variance:", float(np.var(data["has_repo_filter"])),
          "(0 = no real user has used this parameter yet)")

    corr, run, strat, adjust = methodlm.make_tools(data, "result_count", False)
    print()
    print(adjust("query_length", ["has_repo_filter", "hour"]))
    print()
    print(adjust("has_repo_filter", ["query_length", "hour"]))


if __name__ == "__main__":
    main()
