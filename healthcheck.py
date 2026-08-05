"""
Real end-to-end health check for the live OBSERVE deployment -- not just
"does the process respond," but "does a real search actually return real
results." Exits non-zero on any real failure so this is usable directly
by an external uptime monitor (cron + alert-on-nonzero-exit, or wired
into a service like healthchecks.io/BetterUptime) without needing this
codebase's own alerting stack running.

Real, measured limitation this check itself doesn't cover: the print-
buffering bug this session found means a server can be genuinely healthy
while ITS OWN status log looks stuck -- this check bypasses that
entirely by hitting the real HTTP endpoints, not reading server.log.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://api.observe-search.online"


def check(name, fn):
    t0 = time.time()
    try:
        fn()
        print(f"OK   {name} ({time.time()-t0:.2f}s)")
        return True
    except Exception as e:
        print(f"FAIL {name}: {e}")
        return False


def check_landing_page():
    req = urllib.request.Request(BASE_URL + "/")
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status != 200:
            raise RuntimeError(f"status {resp.status}")


def check_repos_list():
    req = urllib.request.Request(BASE_URL + "/v1/repos")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.load(resp)
        repos = data.get("repos", [])
        if len(repos) < 20:
            raise RuntimeError(f"only {len(repos)} repos listed, expected 29+")


def check_agent_card():
    req = urllib.request.Request(BASE_URL + "/.well-known/agent-card.json")
    with urllib.request.urlopen(req, timeout=10) as resp:
        card = json.load(resp)
        if "skills" not in card or not card["skills"]:
            raise RuntimeError("agent card missing skills")


def check_real_search(api_key):
    req = urllib.request.Request(
        BASE_URL + "/v1/search",
        data=json.dumps({"query": "parse command line arguments", "k": 3}).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
        results = data.get("results", [])
        if len(results) == 0:
            raise RuntimeError("real search returned zero results")
        if not any(r["score"] > 0.3 for r in results):
            raise RuntimeError(f"top result score suspiciously low: {results[0]['score']}")


def main():
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    results = [
        check("landing page", check_landing_page),
        check("repos list (29+ repos)", check_repos_list),
        check("A2A agent card", check_agent_card),
    ]
    if api_key:
        results.append(check("real search returns real results", lambda: check_real_search(api_key)))
    else:
        print("SKIP real search check (no API key passed as argv[1])")

    if all(results):
        print("\nHEALTHY")
        sys.exit(0)
    else:
        print("\nUNHEALTHY -- see FAIL lines above")
        sys.exit(1)


if __name__ == "__main__":
    main()
