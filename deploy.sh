#!/usr/bin/env bash
# One-time setup for a fresh Ubuntu 24.04 droplet. Run as root (or with sudo).
# Usage: ./deploy.sh
#
# Before running: .env must already exist in this directory (copy .env.example
# and fill in real values) and DOMAIN must be set in the environment or in .env.
set -euo pipefail

if [ ! -f .env ]; then
  echo "Missing .env -- copy .env.example to .env and fill in real Stripe keys + URLs first." >&2
  exit 1
fi

if ! grep -q '^DOMAIN=' .env; then
  echo "Add DOMAIN=yourdomain.com to .env (used by Caddy for automatic HTTPS)." >&2
  exit 1
fi

# --- Docker + Compose plugin ---
if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# --- Firewall: only what's needed ---
if command -v ufw >/dev/null 2>&1; then
  ufw allow OpenSSH
  ufw allow 80/tcp
  ufw allow 443/tcp
  ufw --force enable
fi

# --- Bring the service up ---
export $(grep -v '^#' .env | grep '^DOMAIN=' | xargs)
docker compose up -d --build

echo
echo "Deployed. Check status with: docker compose ps"
echo "Caddy will obtain a Let's Encrypt cert for \$DOMAIN automatically the first"
echo "time it's reachable on port 80/443 -- DNS must already point here."
echo
echo "Next: run the one-time index build (real wall-clock time, embeds ~15 repos):"
echo "  docker compose exec api python index_repos.py"
