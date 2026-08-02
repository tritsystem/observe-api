# OBSERVE Search API

Pay-per-query semantic code search over a curated set of popular open source
repos, built for AI agents to call directly. API key + prepaid credits,
fully automated signup-to-search loop -- no human review anywhere in that
path.

Built on top of [OBSERVE](https://github.com/gbranaa4-hue/012-trit-search)'s
existing `SearchEngine` (embedding-based semantic search, function-boundary
chunking) -- `search_engine.py` here is that class extracted from OBSERVE's
desktop GUI file into a standalone module with no GUI dependency, so a
headless server can import it cleanly. See the extraction notes at the
bottom of `search_engine.py` for exactly what changed. This deployment
serves float32 (not ternary-quantized) embeddings -- see "Honest
limitations" below for the measured reason why.

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

POST /v1/private/index    Authorization: Bearer obs_...  {"git_url": "https://github.com/<owner>/<repo>"}
                           -> {"status": "indexing", "credits_remaining": N}  (github.com/gitlab.com/bitbucket.org only)
GET  /v1/private/status   Authorization: Bearer obs_...  -> {"state": "none"|"indexing"|"ready"|"error", ...}
POST /v1/private/search   Authorization: Bearer obs_...  {"query": "...", "k": 10}
                           -> same shape as /v1/search, scoped to ONLY this key's own indexed repo
                           -> 404 if this key has no ready private index yet
```

## Honest limitations (v1, disclosed not hidden)

- **Private per-tenant indexing is now built** (`tenant_index.py`,
  `POST /v1/private/index` + `GET /v1/private/status` +
  `POST /v1/private/search`) -- an API key can index their own repo,
  isolated from the shared 15-repo corpus and from every other tenant.
  Isolation is structural: each tenant's clone + index lives under a
  directory named by their key_hash, so a caller can't reach another
  tenant's data without already knowing that tenant's raw API key.
  Verified adversarially, not just by code review: two tenants indexed
  different real repos (pallets/itsdangerous, pallets/click), and each
  one's private search returned ONLY its own repo's files at k=10, even
  when explicitly querying for a concept that only exists in the other
  tenant's code. A key with no private index gets a real 404, not a
  misleading empty 200.
  - v1 scope, disclosed: `git_url` is restricted to https:// URLs on
    github.com/gitlab.com/bitbucket.org (a real SSRF/local-file-read risk
    otherwise -- accepting an arbitrary server-side `git clone` target
    from user input is a documented vulnerability class, not a
    hypothetical). Self-hosted git servers aren't supported yet.
  - Indexing is synchronous-triggered/background-executed (real minutes
    for a real repo) with a status you poll -- there's no webhook/push
    notification when it finishes.
  - Each tenant's SearchEngine instance is cached in memory after first
    use (LRU-capped, default 20), sharing one loaded embedding model
    across all of them -- not built for thousands of concurrent active
    tenants, fine for a v1 scale of usage.
  - Indexing costs credits up front (`OBSERVE_CREDITS_PER_PRIVATE_INDEX`,
    default 2000) and isn't refunded if the indexing job itself fails
    partway (a bad git_url is caught before charging; a crash mid-embed
    is not refunded) -- disclosed, not hidden.
- Per-key rate limiting (`rate_limit.py`, in-memory token bucket) now caps
  request RATE separately from the credit system's cost cap -- verified
  with a real 20-request concurrent burst (17 succeeded, 3 correctly got
  429, credit deduction matched exactly). Still in-memory/per-process,
  though -- a multi-process or multi-host deployment would need a shared
  store (Redis) instead, not built here.
- **Serves float32 embeddings, not ternary-quantized**, despite
  `search_engine.py` supporting both (`build_index(..., quantize=True)`).
  Measured via `_benchmark_ternary_vs_float32.py` on 20 real queries across
  all 15 repos: ternary quantization changed the top-1 result 40% of the
  time and averaged only 70% top-10 overlap vs. the exact float32 index --
  a real quality cost, not free. At this corpus's actual size (353MB
  float32), disk/RAM was never the real constraint, so the quality cost
  wasn't worth paying. Ternary quantization remains the right call for the
  desktop tool (a user's own local disk is a real constraint there); it
  wasn't re-verified as worthwhile for a server you control.
- Only benchmarked against plain grep (see `launch/show-hn.md`), not
  against other semantic/embedding-based code search tools (GitHub code
  search, Sourcegraph, or a vanilla embedding+vector-DB pipeline) -- those
  use the same fundamental approach and would likely also beat grep on
  vocabulary-mismatch queries. "Beats grep" isn't the same claim as "beats
  the actual competition," and that comparison hasn't been run.
- **Not actually using a fine-tuned embedding model, by design, after
  testing one.** `MODEL_PATH` defaults to stock
  `sentence-transformers/all-MiniLM-L6-v2`. The original
  012-trit-search fine-tune (`trit_embed_train.py`) turned out broken on
  inspection: both of its GitHub-streaming data sources are dead (one
  gated without auth, one uses a dataset-script format HF no longer
  supports), so every language's training pairs came back empty and it
  trained on local-repo-only data; its own validation also logged NaN for
  every epoch (`EmbeddingSimilarityEvaluator` fed a constant `1.0` label
  array -- correlation against a constant is undefined by definition, a
  real bug, not a training failure). Both were fixed (`_retrain_finetune.py`:
  real pairs mined from this repo's own 15-repo corpus, no external dataset
  dependency; a bug-free held-out self-retrieval check instead of the
  broken evaluator) and retrained -- the new model scored a clean 86%
  self-retrieval accuracy on held-out pairs, but then measured WORSE than
  stock on real open-domain retrieval (`_benchmark_finetuned_vs_stock.py`:
  5% top-10 overlap with stock, 0/20 top-1 matches, several wrong-repo
  top-1 results). Root cause: training anchors are function/class names
  and raw comments, not natural-language questions -- a real usage query
  like "how does useState schedule a re-render" doesn't resemble the
  training data's anchor style, so the fine-tune likely pulled the
  embedding space toward name-matching at the expense of the
  question-answering behavior this product actually needs. A properly
  targeted retrain (real natural-language-question anchors, e.g.
  LLM-generated per chunk) might work; this one, evidence in hand,
  doesn't. Don't re-enable `code-minilm-v2` (or the original checkpoint)
  without a benchmark proving it beats stock first.
- One fixed credit package ($5/50,000 credits, i.e. 0.01 cent/search) --
  no tiers, no subscriptions. Priced to clearly undercut the token cost a
  search saves (see billing.py's comment for the reasoning: OBSERVE's own
  benchmark is ~66% fewer tokens than plain search, and a search's real
  marginal compute cost is near-zero), not around infra cost. Change the
  two env vars if you want a different single price point; multiple
  simultaneous tiers need real code changes.
- `index_repos.py` always re-clones from scratch (`--depth 1`) -- simple
  and always-fresh, but re-running it means downtime while the new index
  builds unless you add a blue/green swap (not built -- v1 assumes
  infrequent re-indexing, not a live-updating corpus).
