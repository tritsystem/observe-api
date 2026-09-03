# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This service is pre-1.0 — the API surface may still change.

## [Unreleased]

_Nothing yet._

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
