# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This service is pre-1.0 — the API surface may still change.

## [Unreleased]

## [0.1.1] — 2026-09-23

CI was red on every run since the workflow was added (2026-09-03), and the
`v0.1.0` tag/release predates that workflow entirely — confirmed live via
`gh run list`/`gh run view`, not assumed from a prior session. This release
is the first one actually verified against a genuinely green CI run.

### Security

- **`/v1/webhook/stripe` now caps the request body (default 1 MB, `OBSERVE_WEBHOOK_MAX_BYTES`).** The route is public and
  the signature can only be verified once the whole body has been read, so previously an unauthenticated caller could make
  the process buffer an arbitrarily large body (measured: a 200 MB unsigned body grew the server's memory by ~212 MB before
  being rejected). The cap is enforced while streaming, so it also holds for chunked requests that carry no
  `Content-Length`; an oversized body gets `413`. Stripe events are a few KB, so real webhooks are unaffected.

### Fixed

- **`import server` crashed everywhere the Spikeling engine isn't checked out at a hardcoded personal path** (CI, or any
  machine other than the one that wrote this code) — `server.py` -> `a2a_adapter.py` -> `spiking_causal_relevance.py`,
  and separately `server.py` -> `commerce_router.py` -> `commerce_spiking_memory.py`, both imported the Spikeling
  compiler/runtime eagerly at module load and raised immediately if the sibling checkout wasn't found. This took the
  whole API down at process start, not just the specific causal-check / listing-affinity features that actually need
  Spikeling. Both now defer the failure to first real use, raising a clear `RuntimeError` only from the call site that
  needs it; the causal-check endpoint already had error handling around that call (refunds the credit, returns a clean
  JSON error), so this makes that the actual behavior instead of an import-time crash. Verified in a fresh Linux
  virtualenv matching CI (Spikeling genuinely absent): `import server` succeeds and
  `pytest tests/test_db.py tests/test_env_config.py tests/test_server.py` passes 65/65, up from 2 collection errors.
  Also verified the normal path (Spikeling present) is unchanged.

## [0.1.0] — 2026-09-03

First tagged release. Code-complete for the core search + billing loop; some
integration side-features are still TODO (see README).

### Added

- **Core API** — FastAPI service: `/v1/signup`, `/v1/search`, `/v1/balance`,
  `/v1/receipts`. Fully automated signup-to-search, no human review in that path.
- **Billing** — Stripe Checkout for prepaid credits + webhook handling,
  attributed per API key (not by email).
- **Storage** — SQLite-backed keys + credit balances, atomic deduct-on-search.
- **Rate limiting** — per-key, applied to signup and search.
- **Client wrappers** — one client core behind a CLI, an MCP server, and
  LangChain / CrewAI adapters, so behaviour is identical across all of them.
- **Agentic commerce discovery** — ACP- / Google-UCP-compatible discovery and a
  two-sided reputation layer from earned agreement between disconnected keys.
- **Search engine** — OBSERVE's `SearchEngine` extracted GUI-free into
  `search_engine.py`; serves float32 embeddings (the measured reason ternary
  isn't used here is in the README).
- **CI** — `tests/test_db.py`, `tests/test_env_config.py`, `tests/test_server.py`
  run on push/PR; tests never touch real Stripe / network / models.
- **Deployment** — `Dockerfile` + `Caddyfile` for a TLS reverse-proxied deploy;
  see `DEPLOY.md`.

### Known gaps

- `tests/test_concept_shakedown.py` and `tests/test_spiking_integrations.py`
  fail to collect (`ModuleNotFoundError: 'compiler'`) — a packaging gap in the
  Spikeling-integration side-features, scoped out of CI on purpose.

[Unreleased]: https://github.com/tritsystem/observe-api/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tritsystem/observe-api/releases/tag/v0.1.0
