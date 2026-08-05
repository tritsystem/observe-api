# OBSERVE roadmap — 50 milestones to self-hostable, enterprise-scalable

**Where this starts, honestly:** a single FastAPI process, SQLite for
billing/keys, a 759k-chunk FAISS index in RAM, hosted via a droplet
(public gateway) tunneled to a home WSL2 machine (actual compute). Real,
live, taking real payments -- but a solo-indie architecture, not an
enterprise one. This roadmap's end state: (a) **full self-independent
environments** -- a company can run their own OBSERVE instance, air-gapped
or in their own cloud, with no dependency on this hosted service, and (b)
**robust scalability** to the traffic/data/compliance bar big players in
this field (GitHub code search, Sourcegraph, Google/Bing code search)
already clear.

Five phases. Each milestone is meant to be independently shippable, not a
vague aspiration -- if a milestone can't be described as "done" or "not
done" against something testable, it's been rewritten until it can.

## Phase 1 — Harden what's live (M1-M10)
*Before building anything new: the current product needs to survive being
depended on.*

1. **Fix the single point of failure**: WSL2 host restart = full outage.
   Add a supervised restart (systemd-style watchdog or Windows service)
   so a host reboot doesn't require manual intervention.
2. **Move SQLite to WAL mode with a real backup cadence** (cron'd
   `sqlite3 .backup` to a separate disk/cloud target) -- right now a
   corrupted `observe_api.db` loses every customer's key and balance with
   no recovery path.
3. **Roll the Stripe secret key** that was exposed in plaintext during
   debugging (already flagged, not yet done).
4. **Structured logging + error tracking** (Sentry or equivalent) --
   right now failures are only visible via manual `tail server.log`.
5. **Rate-limit persistence across restarts** -- `rate_limit.py` is
   in-memory; a restart resets every caller's bucket, a real (if minor)
   abuse window.
6. **Automated health check + alerting** (the server-guard pattern
   already built for a different project, reused here) -- nobody gets
   paged today if the process dies at 3am.
7. **CI**: run the existing 38+ tests on every push, block merge on
   failure. None of this session's fixes were CI-gated.
8. **Dependency pinning + Dependabot/Renovate** -- `requirements.txt`
   currently has no version pins.
9. **A real staging environment** distinct from production -- every fix
   this session was tested directly against the live server.
10. **Document the actual on-call runbook**: what to check when search
    latency spikes, when a restart hangs, when Stripe webhooks stop
    firing -- this session's own debugging (BM25 full-corpus-scan bug,
    print-buffering false-hang, GPU contention) becomes the first three
    entries, not lost tribal knowledge.

## Phase 2 — Full self-independent environments (M11-M20)
*The actual literal goal: a company runs their OWN instance, no
dependency on this hosted service at all.*

11. **Extract `docker-compose.yml` into a real one-command self-host
    installer** -- already exists in skeleton form (`Dockerfile`,
    `docker-compose.yml`), needs to work with zero manual editing beyond
    a `.env` fill-in.
12. **Replace SQLite with a pluggable backend** (Postgres option) --
    self-hosters running at real scale need a real database, not a
    single file.
13. **Make Stripe billing optional/pluggable** -- a self-hosted,
    air-gapped deployment (the exact audience the landing page already
    targets: regulated industries, air-gapped teams) can't call out to
    Stripe at all. Needs a "no billing, just an internal API key allowlist"
    mode.
14. **Offline model bundling** -- `sentence-transformers/all-MiniLM-L6-v2`
    downloads from HuggingFace Hub on first run; an air-gapped deployment
    needs the weights bundled in the image, zero network calls at
    startup.
15. **Self-host indexing CLI** -- `index_repos.py` currently hardcodes
    open-source repos; a self-hoster needs to point it at their own
    private monorepo with one command, no code edit.
16. **Configurable auth backend** -- API keys work for the hosted SaaS;
    a self-hosted enterprise deployment needs to plug into their own
    identity provider (see Phase 3, SSO) instead of managing keys
    separately.
17. **A real "getting started self-hosted" doc**, tested by someone who
    isn't the author -- distinct from `DEPLOY.md` (which documents THIS
    deployment's specifics, not a generic self-host path).
18. **License clarity for self-hosting** -- decide and document what's
    actually licensed for a company to run internally vs. what stays
    hosted-SaaS-only (if anything). Currently undocumented.
19. **Self-host telemetry opt-in** (off by default) -- so the maintainer
    can learn real self-host adoption without violating the "fully
    independent" promise by phoning home without consent.
20. **A reference self-host deployment, actually run once** -- spin up a
    real self-hosted instance on a clean VM from the installer alone,
    verify it works end to end, the same live-verification discipline
    already used everywhere else in this codebase.

## Phase 3 — Enterprise readiness (M21-M30)
*What actually gets an enterprise security review to say yes.*

21. **SSO (SAML/OIDC)** -- required table stakes for any enterprise
    security review; API-key-only auth is a real, disclosed v1
    limitation (`db.create_api_key` has no email verification even
    today).
22. **Audit logging** -- who searched what, when, from where; currently
    `usage_log` exists but isn't exposed as a queryable audit trail an
    admin can review.
23. **Per-org multi-tenancy** (not just per-key) -- `tenant_index.py`
    already isolates by key_hash for private indexing; enterprise needs
    org-level grouping (multiple keys/seats under one billing entity).
24. **RBAC** -- today every key has identical permissions; enterprise
    needs read-only vs. admin vs. billing-owner roles.
25. **SOC 2 Type I** (the honest gap already named in server-guard's own
    "does this rival commercial software" analysis applies here too) --
    needs a real third-party audit, not more solo coding.
26. **Data residency options** -- self-hosting (Phase 2) already solves
    this for the extreme case; the hosted SaaS needs a documented data
    location story for customers who won't self-host but need one
    anyway.
27. **SLA + status page** -- a real uptime commitment and a public
    status page (already common infra: Better Uptime, Statuspage.io),
    currently nonexistent.
28. **Vulnerability disclosure policy + `security.txt`** -- currently no
    documented path for a researcher to report a real finding.
29. **Penetration test** (real, third-party) -- especially the private
    indexing path's git-URL SSRF mitigation (already scoped to
    github/gitlab/bitbucket, disclosed as v1) deserves independent
    verification before an enterprise trusts it with their private repo.
30. **Contract/procurement readiness** -- MSA template, DPA template,
    security questionnaire (SIG/CAIQ) pre-filled -- the unglamorous paperwork
    that actually gates enterprise deals.

## Phase 4 — Scale the infrastructure (M31-M40)
*Technical scalability: the actual "big companies" traffic/data bar.*

31. **Move off single-process FAISS-in-RAM** -- a real vector DB
    (Qdrant, Weaviate, or pgvector) that supports horizontal scaling and
    doesn't require reloading the entire index into one process's memory
    on every restart (the exact class of pain this session hit
    repeatedly).
32. **Distributed indexing pipeline** -- `index_repos.py` currently runs
    serially on one machine; a customer with a 10M-file monorepo needs
    parallelized, resumable indexing.
33. **Horizontal API scaling** -- multiple FastAPI replicas behind a
    real load balancer, not the current single droplet+tunnel setup.
34. **Move rate limiting to a shared store** (Redis) -- already flagged
    as a known v1 limitation in the README ("in-memory/per-process...
    a multi-process or multi-host deployment would need a shared store").
35. **Incremental re-indexing** -- `index_repos.py` always re-clones from
    scratch (a disclosed v1 simplification); real scale needs
    delta-indexing on file changes, not a full rebuild each time.
36. **Query result caching** -- a real cache layer (even simple LRU on
    hot queries) for high-QPS customers, not built today since v1 never
    needed it.
37. **Multi-region deployment** -- for latency and data-residency
    (overlaps Phase 3 M26) at real global scale.
38. **Load-tested, published capacity numbers** -- this session's own
    load test (100 concurrent searches, p50/p95 latency) becomes a
    published, repeatable benchmark instead of a one-off debugging
    artifact, re-run against the Phase 4 architecture to prove the
    scaling work actually worked.
39. **Chaos/failure testing** -- kill a replica mid-request, verify
    graceful degradation, not just happy-path testing.
40. **Cost model at scale** -- the current $5/50k-credits pricing was
    calibrated against near-zero marginal compute cost on a single
    process; verify the unit economics still hold under Phase 4's real
    infrastructure cost (vector DB hosting, multi-region, load
    balancers) before scaling traffic into a loss.

## Phase 5 — Compete with big players in this field (M41-M50)
*Market position: the actual "big companies in same field" framing.*

41. **Publish a real, adversarial benchmark vs. actual competitors** --
    the README already discloses this gap explicitly ("Only benchmarked
    against plain grep... not against other semantic/embedding-based
    code search tools... 'Beats grep' isn't the same claim as 'beats the
    actual competition'"). Close it for real: GitHub code search,
    Sourcegraph, Cursor's codebase search.
42. **IDE-native integrations** -- VS Code extension, JetBrains plugin --
    currently MCP/LangChain/CrewAI cover agent access, not direct
    human-in-editor search.
43. **A real free tier competitive with GitHub's native search** -- the
    honest pitch already identifies GitHub's coverage gaps (repos
    excluded from their index, keyword-only queries); converting that
    into a genuinely better free experience, not just a paid API.
44. **Team/org dashboards** -- usage analytics, seat management --
    the kind of admin surface Sourcegraph's enterprise tier has and
    OBSERVE currently doesn't.
45. **A partnerships/channel motion** -- the earlier agentic-commerce
    work (Stripe ACP, A2A protocol adapter) already positions OBSERVE
    for agent-to-agent discovery; extend that into human-facing
    marketplace listings (AWS/Azure/GCP marketplace) enterprises actually
    procure through.
46. **Case studies from real (not hypothetical) customers** -- needs
    Phase 1-4 to actually produce paying enterprise customers first;
    listed here as the milestone that closes the loop, not skippable.
47. **A dedicated enterprise sales motion** -- self-serve pricing (the
    current $5 flat package) doesn't fit a $50k/year enterprise deal;
    needs a real sales-assisted tier once Phase 3 makes that credible.
48. **Open-source the self-host core** (if not already fully done by
    Phase 2) as a distinct trust/adoption lever the way HashiCorp,
    Elastic, and GitLab's open-core models work -- competing on trust
    with entities like GitHub, not just features.
49. **A public transparency/trust page** -- uptime history, security
    posture, this-session's-own honesty-first pattern (disclosed
    negative results, real benchmarks) turned into a permanent public
    artifact, not just README prose.
50. **Re-run this exact roadmap's self-assessment against real
    competitors' public numbers once Phase 4 ships** -- the actual test
    of "robust scalability to big companies in same field" isn't a
    feature checklist, it's whether OBSERVE's real, measured numbers
    (latency, uptime, index freshness, benchmark accuracy) hold up next
    to GitHub/Sourcegraph's published ones. If they don't, that's the
    honest finding to report, not paper over -- consistent with every
    other benchmark in this codebase.

## What this roadmap deliberately does NOT promise

No fixed timeline -- Phase 1 alone (hardening a live, real, paying
product) is real work that hasn't started. No claim that 50 milestones
means 50 equal-sized units of effort -- M31 (moving off single-process
FAISS) is a bigger lift than M3 (rolling one API key). And no assumption
that competing head-to-head with GitHub/Sourcegraph on their own turf is
the only viable path -- Phase 2's self-hosted/air-gapped niche (the
landing page's actual current positioning: compliance-constrained teams)
may be the more honest, winnable wedge than Phase 5's direct competition,
and that's a real strategic choice to make consciously when this roadmap
is revisited, not an assumption baked in silently.
