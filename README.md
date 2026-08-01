# OBSERVE Search API

Pay-per-query semantic code search over a curated set of popular open source
repos, built for AI agents to call directly. API key + prepaid credits,
fully automated signup-to-search loop -- no human review anywhere in that
path.

Built on top of [OBSERVE](https://github.com/gbranaa4-hue/012-trit-search)'s
existing `SearchEngine` (ternary-quantized embeddings, FAISS-backed,
function-boundary chunking) -- `search_engine.py` here is that class
extracted from OBSERVE's desktop GUI file into a standalone module with no
GUI dependency, so a headless server can import it cleanly. See the
extraction notes at the bottom of `search_engine.py` for exactly what
changed.

## What's real vs. what's still TODO

**Built and code-complete:**
- `search_engine.py` -- the actual search logic (unchanged from OBSERVE)
- `db.py` -- SQLite-backed API keys + credit balances, atomic deduct-on-search
- `billing.py` -- Stripe Checkout session creation + webhook handling for
  prepaid credit top-ups, unambiguously attributed per API key (not by
  email, which could collide)
- `server.py` -- FastAPI app: `/v1/signup`, `/v1/search`, `/v1/balance`,
  `/v1/repos`, `/v1/webhook/stripe`
- `index_repos.py` -- clones the curated repo list and builds the shared
  index

**Not done -- needs your action, not something I can do for you:**
1. A real Stripe account with live API keys (`STRIPE_SECRET_KEY`) and a
   webhook endpoint configured (`STRIPE_WEBHOOK_SECRET`) pointing at
   `https://YOUR_DOMAIN/v1/webhook/stripe`.
2. A cloud VM (or other always-on host) with a public IP/domain -- this
   needs the model resident in memory (several hundred MB-1GB+) for
   reasonable query latency, so it's not a fit for a cold-start serverless
   function.
3. Actually running `index_repos.py` once, which clones ~15 real repos and
   embeds every chunk -- expect real wall-clock time (the OBSERVE README's
   own benchmark of 58k chunks took minutes; this corpus is bigger).
4. A domain + TLS in front of the service (Caddy or nginx + Let's Encrypt
   is the standard, low-effort way to get both at once).

## Local dev (no Docker)

```
pip install -r requirements.txt
cp .env.example .env   # fill in real Stripe keys
python index_repos.py  # clones repos + builds the index (slow, run once)
uvicorn server:app --reload
```

## API shape

```
POST /v1/signup           {"email": "..."}  -> {"api_key": "obs_...", "checkout_url": "..."}
GET  /v1/balance          Authorization: Bearer obs_...  -> {"credits": N}
GET  /v1/repos            -> {"repos": ["react", "django", ...]}
POST /v1/search           Authorization: Bearer obs_...
                           {"query": "...", "k": 10, "repo": "react"}  (repo optional)
                           -> {"results": [{"score", "path", "preview", "repo"}], "credits_remaining": N}
```

## Honest limitations (v1, disclosed not hidden)

- Single shared index, not per-customer. Fine for "search our curated
  corpus," not a fit for "let customers index their own private repos"
  (that's a real multi-tenant isolation feature, not built here).
- No rate limiting beyond the credit balance itself -- a key with credits
  can call as fast as it wants. Fine for v1, revisit if abuse shows up.
- One fixed credit package ($10/1000 credits) -- no tiers, no
  subscriptions. Change the two env vars if you want a different single
  price point; multiple simultaneous tiers need real code changes.
- `index_repos.py` always re-clones from scratch (`--depth 1`) -- simple
  and always-fresh, but re-running it means downtime while the new index
  builds unless you add a blue/green swap (not built -- v1 assumes
  infrequent re-indexing, not a live-updating corpus).
