"""
Stripe Checkout integration for prepaid API credits. One fixed package for
v1 (configurable via env vars, not multiple tiers in code -- add tiers later
if there's real demand for them, not preemptively). Fully automated: no
human reviews or approves a purchase, the webhook is the only thing that
credits an account.

Needs real Stripe keys to actually run (STRIPE_SECRET_KEY,
STRIPE_WEBHOOK_SECRET) -- code-complete without them, but every call will
fail until they're set as real environment variables.
"""
import json
import os
import urllib.request

import stripe

from fastapi import HTTPException

import db

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# One package for v1: $5 -> 50,000 credits (0.01 cent/search). Priced
# deliberately far below the token cost it saves, not around infra cost --
# a search's real marginal cost is near-zero (milliseconds of CPU for one
# embedding + matmul against an already-loaded model/index), and OBSERVE's
# own benchmark is ~66% fewer tokens than plain search on the same query.
# At typical model API pricing (a few dollars per million tokens), a few
# thousand tokens saved is worth a fraction of a cent -- the price per
# search needs to clearly undercut that, or there's no reason for an agent
# operator to pay instead of just burning more tokens. Fixed costs
# (hosting) are covered by volume, not by pricing each query near its own
# compute cost. Change via env vars, not code, so pricing can be tuned
# without a redeploy.
PACKAGE_PRICE_CENTS = int(os.environ.get("OBSERVE_PACKAGE_PRICE_CENTS", "500"))
PACKAGE_CREDITS = int(os.environ.get("OBSERVE_PACKAGE_CREDITS", "50000"))

SUCCESS_URL = os.environ.get("OBSERVE_CHECKOUT_SUCCESS_URL", "https://example.com/success")
CANCEL_URL = os.environ.get("OBSERVE_CHECKOUT_CANCEL_URL", "https://example.com/cancel")

# Reuses the same Discord bot identity/token already set up for the
# Spikeling personal-assistant bot (see Documents/Spikeling/discord_bot.py)
# -- same env var names on purpose, so one bot token covers both without
# a second Discord application. This calls Discord's REST API directly
# (stdlib urllib, no new dependency) rather than going through that bot's
# live gateway client -- a sale notification is a one-shot DM, it doesn't
# need an open gateway connection to send it.
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_AUTHORIZED_USER_ID = os.environ.get("DISCORD_AUTHORIZED_USER_ID", "")


def _notify_discord_sale(amount_cents: int, credits: int):
    # Best-effort, never fatal -- a Discord outage or a missing token
    # shouldn't stop a real payment from crediting the buyer's account.
    if not DISCORD_BOT_TOKEN or not DISCORD_AUTHORIZED_USER_ID:
        return
    try:
        headers = {
            "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
            "Content-Type": "application/json",
            # Without a real User-Agent, urllib's default
            # ("Python-urllib/3.x") gets a Cloudflare error 1010 (bot-
            # signature block) in front of Discord's API before ever
            # reaching Discord's own auth/logic -- this was a real, latent
            # bug here (never fired by a real sale until now), found by
            # actually testing the identical pattern in a standalone
            # script and reading the real error body instead of assuming
            # a bare "403 Forbidden" meant something else.
            "User-Agent": "DiscordBot (https://github.com/gbranaa4-hue/observe-api, 1.0)",
        }
        # DM channels are opened by recipient_id, not addressed directly --
        # same two-call pattern (open channel, then post to it) Discord's
        # REST API requires for bot-initiated DMs.
        open_req = urllib.request.Request(
            "https://discord.com/api/v10/users/@me/channels",
            data=json.dumps({"recipient_id": DISCORD_AUTHORIZED_USER_ID}).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(open_req, timeout=10) as resp:
            channel_id = json.load(resp)["id"]

        text = (
            f"cha-ching, bro -- real sale on observe-api: "
            f"${amount_cents / 100:.2f} for {credits:,} credits."
        )
        send_req = urllib.request.Request(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            data=json.dumps({"content": text}).encode(),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(send_req, timeout=10).close()
    except Exception as e:
        print(f"(Discord sale notification failed, non-fatal: {e})", flush=True)


def create_checkout_session(email: str, key_hash: str) -> str:
    session = stripe.checkout.Session.create(
        mode="payment",
        customer_email=email,
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": f"OBSERVE API credits ({PACKAGE_CREDITS:,})",
                },
                "unit_amount": PACKAGE_PRICE_CENTS,
            },
            "quantity": 1,
        }],
        # key_hash (not email) is what the webhook uses to attribute the
        # payment -- unambiguous even if two accounts share an email.
        metadata={"key_hash": key_hash, "credits": str(PACKAGE_CREDITS)},
        success_url=SUCCESS_URL,
        cancel_url=CANCEL_URL,
    )
    return session.url


def handle_webhook(payload: bytes, sig_header: str | None):
    if not WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="STRIPE_WEBHOOK_SECRET not configured")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        raise HTTPException(status_code=400, detail=f"invalid webhook signature/payload: {e}")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        key_hash = session["metadata"]["key_hash"]
        credits = int(session["metadata"]["credits"])
        amount_cents = session["amount_total"]
        db.add_credits(key_hash, credits, session["id"], amount_cents)
        _notify_discord_sale(amount_cents, credits)
