# Contributing

This is the hosted **OBSERVE Search API** — a FastAPI service plus client
wrappers (CLI, MCP, LangChain, CrewAI), maintained by one person. Contributions
are welcome within that reality.

## Ground rules

1. **No fake numbers.** If you change a benchmark or a limitation claim, include
   the script and say what you ran it on. The README's "Honest limitations"
   section is load-bearing — keep it honest.
2. **The billing and key-isolation paths get the most scrutiny.** Anything that
   touches `billing.py`, `db.py`, or the `/v1/*` routes needs a test.
3. **Tests never make real network / Stripe / model calls** — see
   `tests/conftest.py`. Keep it that way.
4. **Disclose AI assistance** if you used it — a line in the PR is enough.

## Setup

```bash
git clone https://github.com/tritsystem/observe-api
cd observe-api
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before a PR

```bash
python -m pytest -q tests/test_db.py tests/test_env_config.py tests/test_server.py
```

Two suites (`test_concept_shakedown.py`, `test_spiking_integrations.py`) can't
collect yet — a known pre-existing packaging gap in the Spikeling-integration
side-features, not the core API. Don't let that block a core-API PR; do fix it if
that's what your PR is about.

- One logical change per PR.
- If it changes an endpoint's behaviour, update `README.md` and add/adjust a test.
- Deployment notes live in `DEPLOY.md`.

## Security

Don't open a public issue — see [SECURITY.md](SECURITY.md).
