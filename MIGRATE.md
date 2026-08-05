# Migrating observe-api off kimchi to a real cloud VM

This is for moving the *already-live* production service (currently running
on kimchi's own WSL2, tunneled through a too-small droplet) to a single,
properly-sized cloud VM -- stage 1 of the auto-scaling path discussed
2026-08-04. This is NOT the same as `DEPLOY.md`'s fresh-install steps 1-9:
this preserves real, live customer data (API keys, credit balances, usage
history) instead of starting empty.

**Real numbers this runbook is sized against** (measured 2026-08-04, kimchi):
- Live `uvicorn` process: **5.24GB RSS** with the full index loaded
- FAISS index on disk (`data/observe-index`): **1.5GB**
- Embedding model cache (`~/.cache/huggingface`): 88MB (small enough to just
  re-download fresh on the new VM rather than transfer -- not worth the
  extra migration step)
- `observe_api.db` (API keys, credits, usage log, purchases): 88KB as of
  this write-up -- trivially small to transfer, but it's the one file that
  MUST be copied exactly, not regenerated, since it holds real customer state

**Target**: an 8GB RAM droplet (see `DEPLOY.md` step 1's corrected sizing).

## Real risks, stated plainly, not glossed over
- **Cutover downtime**: DNS propagation is not instant. Between pointing the
  domain at the new VM's IP and the old kimchi-tunnel path going dark,
  there's a real window (minutes to low hours depending on TTL) where some
  requests may hit whichever side is currently resolving -- if both sides
  are serving from the same DB copy this is survivable, but if the old side
  keeps accepting writes after you've started migrating, you WILL diverge
  and lose data. The steps below are ordered specifically to avoid that.
- **The live DB is a single SQLite file with real credit balances in it.**
  Copying it while the live service is still writing to it risks a
  torn/inconsistent copy. Don't skip the "stop accepting writes first" step
  below to save time.

## Steps

1. **Provision the new 8GB droplet** per `DEPLOY.md` steps 1-5, but do NOT
   point DNS at it yet and do NOT run step 6 (index build) yet.
2. **Copy the index to the new droplet ahead of time**, while the old
   service is still fully live (this part is safe to do without downtime --
   the index is read-only at runtime): `scp` or `rsync` the 1.5GB
   `data/observe-index` directory from kimchi's WSL2
   (`~/observe-api/data/observe-index`) to the new droplet's
   `data/observe-index`. This is the slow part (1.5GB) -- get it done before
   the downtime window starts, not during it.
3. **Announce/schedule the real downtime window** (even if it's just to
   yourself) -- the next steps have a real, if short, availability gap.
4. **Stop the live service on kimchi** (`docker compose stop api` or however
   it's actually being run there -- confirm the real process name/method
   first, don't assume docker compose if it's running some other way) so no
   new writes land in the DB during the copy.
5. **Copy the live `observe_api.db`** from kimchi to the new droplet's
   `data/` directory (or wherever `.env`'s DB path points) -- this file
   is tiny (88KB as of this write-up) so the copy itself is fast; the
   downtime is really about steps 4-7 as a whole, not this copy.
6. **Bring the new droplet's service up** (`docker compose up -d`) pointed
   at the copied DB and index. Verify locally against the droplet's own IP
   (not the domain yet) that `/v1/balance` for a REAL existing key hash
   returns the correct, migrated credit balance -- this is the real proof
   the migration worked, not just "the process started."
7. **Cut DNS over** to the new droplet's IP.
8. **Decommission the old droplet's tunnel** and stop the kimchi-side
   service permanently (don't just leave it stopped-but-present -- a stale
   second copy of the DB sitting around is a real footgun for a future
   accidental restart).
9. **Verify for real** the same way `DEPLOY.md` step 8 describes, but against
   the real domain now pointing at the new VM, using a real pre-existing
   customer key, not just a fresh signup -- confirming migrated state
   actually works end-to-end, not just that a new install works.

## Rollback

If step 6's verification fails, DNS was never cut over (step 7 is the point
of no easy return) -- just leave the old kimchi service stopped-but-intact
and debug the new droplet without customer-facing impact. Once step 7 has
happened, rolling back means reversing DNS back to the old path AND
re-starting the kimchi service with the DB in whatever state it was left in
-- keep the old kimchi DB copy untouched (don't delete it) until the new
deployment has been stable for a real observation period, specifically so
this rollback path stays available.
