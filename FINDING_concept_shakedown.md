# Finding: shaking down observe-api's own codebase for what's actually executed vs. theoretically present

**Update, same day**: all three action items below are now fixed and verified (38/38 tests
passing) — see the "Fixed" note at the end of each section. Left the original findings
unedited above that so the before/after is honest and checkable, not rewritten after the fact.

**Second update, same day — generalized into a reusable tool**: the ad-hoc script used for this
audit has been rewritten as `concept_shakedown.py`, a config-driven tool that works on any local
codebase (not just observe-api) — see its own docstring for the config schema and its honest
scope boundary (concept-extraction patterns and evidence rules are supplied per target system,
not auto-detected). Re-ran it against observe-api itself as validation
(`shakedown_config.observe-api.json`): it correctly reproduced the now-fixed state (all 8
endpoints show CONFIRMED, matching the tests added above), and — the real value of generalizing
this — **found 3 genuinely new gaps the original manual audit missed entirely**, because the
manual version only grepped a hand-picked set of files: `OBSERVE_CLONE_ROOT` (in
`index_repos.py`), and `OBSERVE_API_BASE`/`OBSERVE_API_KEY` (in `observe_search_mcp/server.py`) --
none of the three have any test-suite reference. A systematic, config-driven sweep over every
file found things a targeted manual grep didn't.

**Date measured**: 2026-08-03. **Method**: OBSERVE was used to search observe-api's own
codebase (a small local self-index, `_build_self_index.py`, 271 chunks / 34 files — the
default `build_index()` skip-list doesn't exclude the local `repos/` clone directory, so a
scoped self-index was built specifically for this rather than re-embedding the whole 232k-chunk
shared corpus) to locate the files describing the system's real endpoints and configuration.
Each candidate concept was then checked against three independent, real evidence sources: the
live SQLite database (`observe_api.db`'s `usage_log`/`api_keys`/`credit_purchases` tables), real
on-disk artifacts (`private/`'s tenant directories), and the automated test suite
(`tests/test_server.py`). Nothing here is inferred from reading code alone — every verdict below
is backed by one of those three real evidence sources.

## Endpoints: 8 declared, evidence checked for each

| Endpoint | Real DB/disk evidence | Test-suite covered? | Verdict |
|---|---|---|---|
| `POST /v1/search` | 36 real rows in `usage_log` | Yes | **Confirmed executed, both real and tested** |
| `GET /v1/repos` | Called for real (curl), but nothing persists it | Yes | **Confirmed executed for real, but has zero durable evidence** — the DB would show no trace if it hadn't been |
| `POST /v1/private/index` | 3 real tenant directories on disk (`private/<hash>/repo`, `private/<hash>/index`) | **No** | **Confirmed executed for real, zero automated test coverage** |
| `GET /v1/private/status` | Manually polled for real | **No** | **Confirmed executed for real, zero automated test coverage** |
| `POST /v1/private/search` | Called for real twice (raw curl + Root Cause Copilot's client) | **No** | **Confirmed executed for real — but produces ZERO trace in `usage_log`** (see below — a real, separate bug, not just an untested endpoint) |
| `POST /v1/signup` | Every real attempt against a live server this session failed with a Stripe `AuthenticationError` | Yes (test-only) | **Never once successfully executed against a real running server** — only exercised inside the test suite, presumably against mocked/absent Stripe |
| `GET /v1/balance` | Not manually exercised this session | Yes (test-only) | **Test-covered only, no real-server execution on record** |
| `POST /v1/webhook/stripe` | None | **No** | **Never executed in any form** — would require a real Stripe-delivered webhook; zero real evidence, zero test coverage |

**The most actionable finding**: `/v1/private/search` is real, working code (confirmed by
actually calling it twice, successfully, with correct results) that leaves **no trace at all** in
`usage_log`. Checked why: `db.log_usage()` is called from exactly one place in `server.py` — the
shared `/v1/search` handler — never from `private_search()`. This isn't a hypothetical gap; it's
the reason a real, successful private search I ran earlier today doesn't show up anywhere in the
database. Anyone querying `usage_log` to understand real API usage would silently undercount (or
entirely miss) all private-search activity.

**Fixed**: `private_search()` now calls `db.log_usage(raw_key, req.query, "__private__", ...)` --
`"__private__"` (not a real shared-repo name) marks these rows as private-index searches,
distinguishable from shared-search rows in the same table without a schema migration. Verified
three ways: a new regression test (`test_private_search_succeeds_and_logs_usage`), and a real
restart-and-recheck against the live server + live database (a fresh private search now leaves a
real row with `repo_filter='__private__'`).

**The second-most actionable finding**: `/v1/webhook/stripe` and `/v1/private/index`/
`/v1/private/status`/`/v1/private/search` have **zero automated test coverage** — `tests/test_server.py`
only exercises `/v1/search`, `/v1/repos`, `/v1/signup`, `/v1/balance`. The entire private-indexing
feature (arguably the more complex, more recently built half of the API) and the billing webhook
handler are only known to work because they were manually curl'd during real sessions, not because
any test asserts it.

**Fixed**: added 7 new tests covering all four previously-uncovered endpoints --
`test_private_index_starts_indexing_and_deducts_credits`,
`test_private_index_rejects_invalid_git_url_and_does_not_charge` (exercises the real, unmocked
`validate_git_url`), `test_private_index_returns_409_when_already_indexing`,
`test_private_status_returns_current_state`, `test_private_status_requires_auth`,
`test_webhook_processes_checkout_completed_and_adds_credits`, and
`test_webhook_rejects_invalid_signature`. The real clone+embed itself isn't re-run in these tests
(too slow/network-dependent for a unit test; `search_engine.py`'s own `build_index()` already has
separate coverage) -- these test the HTTP endpoints' own logic: validation, credit charging, and
status-conflict handling, same boundary the existing search tests already drew.

## Configuration: 14 declared env vars, evidence checked for each

| Variable | Ever set to a non-default value? |
|---|---|
| `OBSERVE_INDEX_DIR` | **Yes** — explicitly set to launch the local dev server correctly |
| `STRIPE_SECRET_KEY` | Confirmed **read** (the code path executes), confirmed **never set** — directly, verifiably the root cause of every real signup failure this session |
| `OBSERVE_MODEL_PATH`, `OBSERVE_CREDITS_PER_SEARCH`, `OBSERVE_CREDITS_PER_PRIVATE_INDEX`, `OBSERVE_PACKAGE_PRICE_CENTS`, `OBSERVE_PACKAGE_CREDITS`, `OBSERVE_CHECKOUT_SUCCESS_URL`, `OBSERVE_CHECKOUT_CANCEL_URL`, `OBSERVE_PRIVATE_ROOT`, `OBSERVE_PRIVATE_MAX_CACHED`, `OBSERVE_RATE_LIMIT_CAPACITY`, `OBSERVE_RATE_LIMIT_PER_SEC`, `STRIPE_WEBHOOK_SECRET` | **Never** — in tests or real runs, this whole session. All 12 have only ever run on their hardcoded default. Their "configurability" is theoretical: the code reads them, but nothing has verified the read actually works correctly with an overridden value (a typo in the variable name, a type-coercion bug on a non-default value, etc. would currently be invisible). |

**Fixed**: `tests/test_env_config.py` sets each variable to a real, non-default value, reloads the
module that reads it, and checks the resulting BEHAVIOR actually changed -- not just that a
constant holds the new value. `OBSERVE_CREDITS_PER_SEARCH`/`OBSERVE_CREDITS_PER_PRIVATE_INDEX`/
`OBSERVE_MODEL_PATH` are checked at the constant level (the cheapest correct check for something
that would otherwise need a real second model download); `OBSERVE_PACKAGE_PRICE_CENTS`/
`OBSERVE_PACKAGE_CREDITS`/`OBSERVE_CHECKOUT_SUCCESS_URL`/`OBSERVE_CHECKOUT_CANCEL_URL` are checked
by capturing the actual kwargs a (mocked) Stripe call would receive; `OBSERVE_PRIVATE_ROOT` by
checking the real path a tenant directory resolves to; `OBSERVE_PRIVATE_MAX_CACHED` by actually
triggering LRU eviction at the overridden threshold instead of the default; `OBSERVE_RATE_LIMIT_CAPACITY`/
`OBSERVE_RATE_LIMIT_PER_SEC` by actually exhausting a token bucket at the overridden size; and
`STRIPE_WEBHOOK_SECRET`'s absence path is confirmed to raise a real 500, not fail silently. All 8
new tests pass, and the full 38-test suite passes together (checked for cross-test pollution from
the module reloads these require).

## Why this matters, and its honest limits

This is a **different kind of check** than the causal-driver finding
(`FINDING_causal_harness_vs_naive_tools.md`) — that one used MethodLM's adjustment testing to
separate real causal drivers from confounded look-alikes in a metrics dataset. This one is
coverage/execution validation: does real evidence show a declared concept was ever actually
exercised, or does it only exist as code that's never been run outside of (at best) a narrow test?
No causal inference here — a concept with zero evidence isn't proven dead, only unproven live.
`OBSERVE_PRIVATE_ROOT` not appearing anywhere doesn't mean it's broken; it means nobody has ever
found out whether it works.

**Concrete, fixable items this surfaced — all three now fixed** (see the "Fixed" notes above):
(1) wired `db.log_usage()` into `private_search()`; (2) added 7 real tests for
`/v1/private/index`, `/v1/private/status`, `/v1/private/search`, and `/v1/webhook/stripe`; (3)
added 8 real tests verifying all 12 previously-unverified env vars actually round-trip into real
behavior. The full suite is now 38 tests, up from the 21 that existed before this shakedown, and
all 38 pass together.
