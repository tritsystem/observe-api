# Concept Shakedown — real use cases and their evidence adapters

Six use cases, each needing the same underlying pipeline (extract concepts from code,
check them against real evidence) but a different evidence adapter. All six adapters now
exist and are tested; connecting one to a specific real system (a company's actual flag
service, gateway, etc.) is the one remaining step, and needs real credentials/access this
repo doesn't have.

## 1. Feature-flag graveyard cleanup

Cross-reference flag names in code against a flag service's own real evaluation data to
find flags stuck at 100%/0% rollout that were never removed.

**Adapter**: `http_api` evidence kind (`check_http_api_evidence` in `concept_shakedown.py`)
— calls `GET <url_template with {concept}>`, reads a JSON field (e.g. `evaluations`) as the
evidence signal. Tested against a real local HTTP server simulating a flag-service's
evaluation-count API (not mocked-out logic — a genuine HTTP round-trip): a flag with real
evaluations confirms, a flag with zero evaluations or unregistered in the service (404)
comes back as a confirmed negative, and a genuinely unreachable service comes back as
unknown (never conflated with "confirmed unused").

**To point at a real system**: set `url_template` to the flag service's real API endpoint
and `headers` to a real auth token in the config.

## 2. API/endpoint deprecation audit

Check documented endpoints against real gateway/access logs for actual recent traffic;
endpoints with zero real hits are deprecation candidates.

**Adapter**: `jsonl_field` evidence kind (`check_jsonl_field_evidence`) — parses a glob of
JSON-Lines log files, checks a specific field (e.g. `path`) for an exact/substring match.
Tested against a synthetic gateway-log fixture: endpoints with real logged requests
confirm, endpoints with zero requests don't, malformed log lines are skipped without
crashing the check.

**To point at a real system**: point the glob at real exported access logs (most gateways
can export JSON-Lines) and set `field` to whatever key holds the request path.

## 3. Incident response — "was the safeguard actually on"

After an incident, check whether a specific kill-switch/circuit-breaker/rate-limit that
was supposed to prevent it has real evidence of ever firing.

**Adapter**: same `jsonl_field` mechanism as #2 — most incident/audit logs are also
JSON-structured. Already validated by the same test as #2 (the mechanism doesn't care
whether the log is a gateway log or an incident log; the concept type and field name in
the config are what change).

## 4. Pre-migration config audit

Before an infra migration, check which of a large env-var/config surface is real,
load-bearing vs. legacy cruft, using real evidence instead of asking around.

**No new adapter needed** — this is exactly `tests/test_env_config.py`'s pattern (real
override + reload + behavioral check) combined with the `test_grep`/`file_grep` evidence
kind already used against observe-api, Django, React, and Vue. Point the existing config
schema at the target repo before a migration; the "12 of 14 env vars were unverified"
finding on observe-api itself is a real instance of this exact use case.

## 5. Root Cause Copilot integration

Once MethodLM confirms a real causal driver of a metric, immediately check whether that
specific driver has any real test coverage in the codebase — turns "what caused this" into
"was this ever actually vetted" in one pass.

**Built and tested**: `root_cause_copilot.py`'s `diagnose()` now takes `local_repo_path` +
`test_glob`; each confirmed driver gets a `test_evidence` field via
`check_driver_has_test_evidence()` (a small, self-contained duplicate of the `test_grep`
mechanism — the two tools live in different repos on different drives, so a direct import
would be more fragile than a small copy). Verified against observe-api's real test suite:
a known-tested env var confirms, a made-up name doesn't, and no `local_repo_path` cleanly
returns "not checked" rather than a false negative.

## 6. Technical due diligence (M&A, inheriting a legacy codebase)

Which of the claimed features in the docs/README are backed by real evidence they work
vs. exist only as a claim.

**No new adapter needed** — this is precisely what the shakedown already did to
observe-api itself (`FINDING_concept_shakedown.md`): real endpoints, real env vars, real
test coverage, checked against the real database and the real test suite, with concrete
fixable findings (the `log_usage` gap) that came directly out of it. Point the tool at any
codebase you're evaluating, using whatever real evidence source (its own database, its own
logs, its own tests) you have access to during the diligence process.

## Summary table

| # | Use case | Adapter | Status |
|---|---|---|---|
| 1 | Feature-flag cleanup | `http_api` | Built, tested against a real local server |
| 2 | Endpoint deprecation | `jsonl_field` | Built, tested against a synthetic log fixture |
| 3 | Incident safeguard check | `jsonl_field` (same as #2) | Built, tested |
| 4 | Pre-migration config audit | `test_grep`/`file_grep` (existing) | Already validated on 4 real codebases |
| 5 | Root Cause Copilot integration | new `check_driver_has_test_evidence` | Built, tested against observe-api's real tests |
| 6 | Technical due diligence | full tool (existing) | Already validated on observe-api itself |
