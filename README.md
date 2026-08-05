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

**Testing search without real Stripe keys**: `/v1/signup` needs a live
`STRIPE_SECRET_KEY` (it creates a real Checkout session), but the search endpoints
themselves don't touch Stripe at all -- Stripe only lives in that one HTTP handler, not
in the underlying credit/key data model. To test `/v1/search` or the `/v1/private/*`
endpoints for real without any Stripe setup, create a key directly (same technique
`tests/test_server.py` already uses):
```python
import db
key = db.create_api_key("you@example.com")
db.add_credits(db.hash_key(key), 5000, "sess_manual_test", 0)
```
Verified for real this way: shared-index search, and a full private-index cycle
(`POST /v1/private/index` against a real GitHub repo -- real clone, real embed, real
chunk count -- followed by `POST /v1/private/search` returning genuinely relevant
results). Both matched exactly between raw `curl` and a client library's own request
function, confirming the actual request/response contract works end-to-end, not just
that it reads correctly on paper.

## API shape

```
POST /v1/signup           {"email": "..."}  -> {"api_key": "obs_...", "checkout_url": "..."}
GET  /v1/balance          Authorization: Bearer obs_...  -> {"credits": N}
GET  /v1/repos            -> {"repos": ["react", "django", ...]}
POST /v1/search           Authorization: Bearer obs_...
                           {"query": "...", "k": 10, "repo": "react", "investigate": false}  (repo, investigate optional)
                           -> {"results": [{"score", "path", "preview", "repo", "verification_hint"}], "credits_remaining": N, "note"}
                           -> investigate:true labels results as unverified candidates instead of ranked
                              answers, each with a suggested falsification test -- see "investigate mode"
                              below. Same credit cost as a normal search either way.

POST /v1/private/index    Authorization: Bearer obs_...  {"git_url": "https://github.com/<owner>/<repo>"}
                           -> {"status": "indexing", "credits_remaining": N}  (github.com/gitlab.com/bitbucket.org only)
GET  /v1/private/status   Authorization: Bearer obs_...  -> {"state": "none"|"indexing"|"ready"|"error", ...}
POST /v1/private/search   Authorization: Bearer obs_...  {"query": "...", "k": 10}
                           -> same shape as /v1/search, scoped to ONLY this key's own indexed repo
                           -> 404 if this key has no ready private index yet
```

See [`AGENT_INTEGRATION.md`](AGENT_INTEGRATION.md) for how to actually wire
this into an agent's toolset -- MCP/LangChain/CrewAI get the "when to call
this vs. grep" motive for free from the tool description; a raw HTTP/custom
agent integration needs that guidance written into its own system prompt,
with a real example.

## investigate mode (experimental)

`POST /v1/search` with `"investigate": true` reframes results as **unverified
candidates instead of ranked answers**, each with a `verification_hint`
suggesting a concrete falsification test (isolate the candidate, re-run the
behavior you're investigating, see if it persists). Template-only in v1 --
no LLM call, no added cost, same credit price as a normal search. Motivated
by [gbranaa4-hue/methodlm](https://github.com/gbranaa4-hue/methodlm)'s
causal-discipline framing (pre-register a hypothesis, falsify before
confirming) applied to code search instead of tabular data.

**The actual value isn't better ranking -- it's not acting on a wrong one.**
Measured against 10 realistic root-cause queries (memory leaks, race
conditions, deadlocks, N+1 queries, stale caches, buffer overflows, hangs,
timezone bugs -- top-3 each, 30 results total, manually assessed for
relevance, not an automated benchmark):
- **25/30 candidates genuinely on-topic** -- e.g. "off by one error in
  pagination" returned `django/core/paginator.py` for all 3, "buffer
  overflow in string parsing" surfaced `resp_parser.c`'s own comment "NOT
  SAFE FOR PARSING USER INPUT".
- **3/30 weak/tangential** (plausible-looking but not really the mechanism
  asked about).
- **2/30 clearly wrong**, both keyword-collision false positives: "race
  condition causing flaky test" pulled in a CI documentation file (matched
  on "test"/"CI", unrelated to race conditions); "N+1 query problem in ORM"
  pulled in FastAPI's `Query()` parameter docs (matched on "query", wrong
  meaning of the word entirely).

Those 2 wrong results are the actual point: without investigate mode, an
agent has no signal that rank #2 might be a keyword-collision rather than a
real answer. With it, the same wrong candidate gets the same "unverified,
here's how to falsify it" hint as a right one -- an agent that ran the
suggested test on the CI.md result would immediately see the race condition
persists, and correctly rule it out instead of citing it as the cause.

## Agentic Commerce Protocol (ACP) buyer/seller routing (new)

commerce_router.py adds an ACP-compatible discovery/matching layer, not
a reimplementation of ACP checkout itself. Read directly from the real
2026-04-17 ACP spec (github.com/agentic-commerce-protocol/agentic-commerce-protocol,
maintained by OpenAI + Stripe): a checkout session has no seller_id or
merchant_id field at all -- it's scoped entirely by the caller's Bearer
token to ONE already-known merchant, and multi-seller discovery is
explicitly left to "the marketplace layer above this API." This module
is that layer.

- `POST /v1/commerce/sellers` -- register a seller's real ACP
  `checkout_session_url` (must be https).
- `POST /v1/commerce/sellers/{id}/listings` -- add a listing feed
  (short text descriptions), embedded with the same SentenceTransformer
  already resident in memory for code search (`engine.model` -- no
  second model load).
- `POST /v1/commerce/search` -- a buyer-agent describes intent in
  natural language; results are ranked by real cosine similarity
  against listing embeddings and include each match's seller
  `checkout_session_url` + `item_id`.

**What this deliberately never does**: handle payment credentials,
proxy the checkout call, hold funds, or act as payment provider/merchant
of record. A match is a pointer to the seller's own real ACP endpoint --
the buyer's agent calls it directly (`POST .../checkout_sessions` with
the returned `item_id`, per the real spec) to actually transact. Same
trust boundary as a search engine listing a business, not a payment
processor's.

Tested with a deterministic fake embedding model (`tests/test_commerce_router.py`)
that verifies actual ranking behavior (a boots-related query outranks an
unrelated listing) and per-key listing isolation, not just that
endpoints return 200.

**Learned listing-affinity memory (real Spikeling STDP, not a counter)**:
commerce_spiking_memory.py gives each buyer key its own live network --
one neuron per listing that key has searched, real STDP learning (same
mechanism as spiking_search_heatmap.py's code-search heatmap, same
compiler.compiler/runtime.runtime engine, not a reimplementation) that
reinforces listings which keep getting returned to that buyer. A
listing's recent "heat" nudges its rank (`memory_boost` in each match,
capped at MEMORY_BLEND_WEIGHT=0.15 of the cosine score -- additive, not
a replacement for semantic relevance). Verified two ways: directly
against the real engine (tests/test_commerce_spiking_memory.py -- heat
grows from real firing, a never-searched listing stays at exactly 0.0,
the real STDP asymmetry already documented in spiking_search_heatmap.py
reproduces here too) and end-to-end through the actual HTTP API
(tests/test_commerce_router.py -- memory_boost is 0.0 on a listing's
first-ever search, nonzero on the next one). Found and fixed a real bug
building this: the sibling heatmap module hardcodes a bare Windows path
to locate the Spikeling engine, which silently fails
(ModuleNotFoundError) under WSL -- where this service's real production
process actually runs (see MIGRATE.md). commerce_spiking_memory.py
checks SPIKELING_CORE_PATH, then both the native-Windows and
WSL-mounted real paths, instead of assuming either. No persistence yet
for the learned network itself -- a restart resets memory to cold; a
disclosed v1 gap, not silent data loss (listings/cosine ranking are
unaffected).

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
- **Real coverage gaps found by using OBSERVE to shake down its own codebase, since
  fixed** -- see [`FINDING_concept_shakedown.md`](FINDING_concept_shakedown.md) for the
  full before/after. `/v1/private/search` was real, working, confirmed-executed code
  that left **zero trace** in `usage_log` (`db.log_usage()` was only ever called from
  the shared `/v1/search` handler) -- now wired in, verified against the real live
  server and database, not just a test. `/v1/private/index`, `/v1/private/status`,
  `/v1/private/search`, and `/v1/webhook/stripe` had zero automated test coverage --
  now 7 real tests cover all four. All 12 previously-unverified env vars (everything
  beyond `OBSERVE_INDEX_DIR`/`STRIPE_SECRET_KEY`) now have a test that sets a real
  override and checks the actual resulting behavior changed, not just that a constant
  holds the new value. Test suite: 38 passing, up from 21.
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
- **Hybrid search (dense embedding + BM25 lexical) is now live by default**
  in `search_engine.py`'s `search()` (`hybrid=True` param, on unless you pass
  `hybrid=False`) -- reached this after trying two other approaches and
  measuring both as net negative first, not guessed at:
  - Tried AST/function-boundary chunking (`ast_chunker.py`,
    `_benchmark_ast_chunking.py`) as a real replacement for the fixed-window
    chunking `build_index()` actually does (despite this project's own
    earlier docs claiming "function-boundary chunking" -- a real,
    now-corrected gap between documentation and implementation). Measured
    net negative on the same 20-query/15-repo benchmark: 1 clear win, 5
    regressions, 14 ties. Root cause: whole-function chunking nearly halves
    the real-code chunk count, so the SAME markdown/changelog chunks (still
    fixed-window, untouched by AST chunking) become a larger share of the
    candidate pool and started outranking legitimate code more often than
    the reverse. Not shipped.
  - Tried naive full-corpus dense+BM25 fusion via Reciprocal Rank Fusion
    (`_benchmark_hybrid_search.py`) -- also net negative (3 wins / 6
    regressions / 11 ties), for two distinct reasons: (a) BM25 has no
    notion of "wrong repo," so a shared word alone pulled in an unrelated
    file from a different project/language entirely (an artifact of this
    benchmark's 15-repos-in-one-index corpus specifically, not of
    OBSERVE's real per-tenant indexes, but a real risk in a shared/multi-
    project index); (b) BM25's term-frequency scoring favored verbose
    test/doc files that repeat a query word many times over terse, CORRECT
    implementation files that state it once -- lost flask's `scaffold.py`
    and svelte's `reactivity/set.js` (the actually-right answers) this way.
    Not shipped.
  - **What IS shipped**: retrieve-then-rerank (`_benchmark_hybrid_rerank.py`,
    now `SearchEngine._hybrid_rerank`) -- take the dense embedding's own
    top-30 candidates, then re-rank ONLY those with a dense-dominant
    weighted blend (0.7 dense / 0.3 BM25, both min-max normalized within
    the pool). This structurally can't introduce a candidate dense didn't
    already consider plausible, so it fixed both of naive fusion's failure
    modes: `flask`/`scaffold.py` and `svelte`/`reactivity/set.js` came back
    exactly right, and the wrong-language contamination case (an Express
    query pulling in a Go file) disappeared. Real, measured, disclosed
    tradeoff: it can't reach a correct answer dense missed entirely, so 2 of
    naive fusion's 3 genuine wins (a Django ORM query, a Vue reactivity
    query) are lost too -- safety over reach, a deliberate choice given the
    failure modes it avoids were the more damaging kind. One real
    regression remains even with this approach: a `redis` "expire a key"
    query still surfaces Django's own `redis.py` cache backend ahead of
    the real `expire.c`, because dense similarity itself (not BM25) ranked
    that wrong-repo file inside its own top-30 -- reranking can't fix a
    mistake dense already made upstream of it. Verified end-to-end against
    the real, deployed index (`_verify_hybrid_wiring.py`), not just the
    standalone benchmark script -- `SearchEngine.load()` now builds a BM25
    index automatically (via the new `rank-bm25` dependency), and any
    failure to do so (missing package, unreadable file) degrades silently
    to the previously-proven dense-only path rather than breaking search.
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
  **Re-tested at 2x scale (2026-08-04), same negative result confirmed:**
  after the corpus grew from 15 to 29 repos, retrained clean on the full
  759k-chunk corpus (55,091.9s, real wall-clock -- not a quick job),
  87.6% self-retrieval accuracy on held-out pairs (`438/500`) -- then
  re-ran the same overlap methodology against real open-domain queries,
  one per repo this time (`benchmark_finetuned_vs_stock_29.py`, 31
  queries): **0.8/10 average top-10 overlap (8%), 2/31 top-1 exact match
  (6%)** -- consistent with the original 15-repo result (5%/0%), not an
  artifact of too little training data. Manually inspected several
  results, not just the aggregate number: the fine-tune's failures are
  often actively worse than "different," not just non-overlapping --
  e.g. redis "expire a key" stock correctly surfaces the real `src/db.c`
  implementation, fine-tune returns unrelated Symfony PHP Redis lock/
  semaphore classes (wrong repo, wrong language); pytest "discover test
  functions" stock finds real `_pytest/main.py`, fine-tune finds
  scikit-learn's `discovery.py` (wrong repo). High self-retrieval
  accuracy and low real-query overlap coexisting twice now, at two
  different corpus sizes, is itself the finding: self-retrieval measures
  something this fine-tuning approach is good at that isn't the thing
  the product actually needs.
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
