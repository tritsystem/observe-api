# Deploying observe-api

Real, step-by-step path from "code on GitHub" to "live at a real domain."
Nothing here is templated for illustration -- follow it in order.

## 1. Create the droplet
DigitalOcean -> Create -> Droplets -> Ubuntu 24.04 LTS -> Basic / Regular SSD
($6/mo is enough to start) -> add your SSH key -> Create. Note the public IP.

## 2. Point DNS at it
At your domain registrar (or DO's own DNS if you moved nameservers there),
add an A record: `yourdomain.com -> <droplet IP>`. Give it a few minutes to
propagate before step 5 (Caddy needs it resolvable to get a TLS cert).

## 3. Get the code onto the droplet
```
ssh root@<droplet IP>
git clone https://github.com/gbranaa4-hue/observe-api.git
cd observe-api
```

## 4. Fill in real secrets
```
cp .env.example .env
nano .env
```
Fill in:
- `STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` (from the Stripe dashboard --
  the webhook secret needs the endpoint URL, which needs the domain, so this
  is genuinely a step-4-after-step-2 dependency, not arbitrary ordering)
- `OBSERVE_CHECKOUT_SUCCESS_URL` / `OBSERVE_CHECKOUT_CANCEL_URL` -> your real domain
- Add one new line: `DOMAIN=yourdomain.com` (used by Caddy, not by the app itself)

## 5. Deploy
```
chmod +x deploy.sh
./deploy.sh
```
This installs Docker, opens only ports 22/80/443, and brings up two
containers: `api` (the FastAPI service) and `caddy` (reverse proxy +
automatic Let's Encrypt TLS for `$DOMAIN`).

## 6. Build the search index (one time, real wall-clock cost)
```
docker compose exec api python index_repos.py
```
Clones and embeds the ~15 curated repos. Expect this to take real minutes,
not seconds -- it's really cloning and really running the embedding model,
not a canned response.

## 7. Point the Stripe webhook at it
Stripe dashboard -> Developers -> Webhooks -> Add endpoint ->
`https://yourdomain.com/v1/webhook/stripe` -> copy the signing secret into
`.env`'s `STRIPE_WEBHOOK_SECRET` if you hadn't already, then:
```
docker compose restart api
```

## 8. Verify for real, not just "it's up"
```
curl https://yourdomain.com/v1/signup -X POST -H "Content-Type: application/json" -d '{}'
```
Should return a real API key. Then run one real search with it against
`/v1/search`, and do one real $5 Checkout purchase end-to-end (Stripe test
mode first if you want a dry run before going live) to confirm credits
actually land.

## 9. Only after 8 passes for real: update references + go public
- Done: live domain is `api.observe-search.online`, not the `.dev`
  placeholder that was still baked into `llms.txt`, `observe_search_mcp/`,
  and `observe_search_tools/` after the domain migration -- caught and
  fixed (would have silently broken anyone who installed the MCP/SDK
  packages and never overrode `OBSERVE_API_BASE`).
- Submit to the registries listed in `launch/registry-submissions.md`
  (requires your own GitHub/account identity on each PR/form).
- Post the drafted Show HN / blog content in `launch/`.

## 10. Ongoing operations (see `ROADMAP.md` Phase 1 for the full hardening list)

- **Backups**: `backup_db.py` uses SQLite's safe live-backup API (not a
  raw file copy, which can corrupt a WAL-mode DB mid-write), keeps 1
  week of history. A crontab entry (`0 */6 * * *`) is already registered
  on the host, but cron itself needs a one-time manual start:
  `sudo service cron start`, plus `[boot] command=service cron start` in
  `/etc/wsl.conf` so it survives a WSL restart. Verify it's actually
  running with `service cron status` after.
- **Health check**: `healthcheck.py [api_key]` hits real endpoints
  (landing page, repos list, agent card, and a real search if an API key
  is passed) -- deliberately does NOT read `server.log`, since a
  genuinely healthy server can look permanently stuck there (a real bug
  found this session: stdout is block-buffered when redirected to a
  file, so a completion message can sit unflushed indefinitely). Wire
  this into cron + an alerting channel, or an external uptime monitor
  (healthchecks.io, BetterUptime) via "alert if this script's exit code
  is nonzero."
- **CI**: `.github/workflows/ci.yml` runs the test suite on every push --
  exists locally in the repo checkout but isn't pushed yet (`gh` auth is
  missing the `workflow` scope needed to write files under
  `.github/workflows/`; run `gh auth refresh -h github.com -s workflow`
  once, then commit/push that file for real). Already proved its worth
  once even before being live: running the suite locally for the first
  time this session caught a real regression and 2 real pre-existing
  test/feature drift bugs.
