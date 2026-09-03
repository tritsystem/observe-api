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
import sqlite3
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

# Two real, flat-fee recurring tiers, deliberately NOT priced by the same
# near-zero-marginal-cost logic as the pay-as-you-go package above. The
# strategic point isn't credit volume -- it's converting search volume's
# 1M-searches/day-for-$3k/mo problem (real math: at $0.0001/search, reaching
# meaningful revenue needs unrealistic anonymous traffic for a new product)
# into a subscriber-count problem instead.
#
# $9 Starter matches what Sourcegraph Cody's Pro tier used to cost -- flagged
# honestly, not hidden: Cody discontinued that tier in mid-2025 and is now
# Enterprise-only (~$59/user/mo), so this specific comp is stale by the time
# this was implemented. $9 stands on its own as a real, low-friction entry
# price regardless.
#
# $49 Compliance is the actual differentiated tier: bundles a real
# usage-audit export (see server.py's /v1/audit-log, compliance-only --
# reads this project's own existing usage_log table, not a reimplementation
# of mcp-gateway's separate hash-chained tamper-evident log, which solves a
# different problem: protecting an MCP tool-call trail, not exporting one
# customer's own usage history to them). Free (not just half-price) private
# indexing and the highest rate limit round out the real value gap between
# the two tiers, not just a bigger credit number.
STARTER_PRICE_CENTS = int(os.environ.get("OBSERVE_STARTER_PRICE_CENTS", "900"))
STARTER_CREDITS_PER_PERIOD = int(os.environ.get("OBSERVE_STARTER_CREDITS_PER_PERIOD", "100000"))
STARTER_RATE_CAPACITY = float(os.environ.get("OBSERVE_STARTER_RATE_LIMIT_CAPACITY", "20"))
STARTER_RATE_REFILL_PER_SEC = float(os.environ.get("OBSERVE_STARTER_RATE_LIMIT_PER_SEC", "10"))

COMPLIANCE_PRICE_CENTS = int(os.environ.get("OBSERVE_COMPLIANCE_PRICE_CENTS", "4900"))
COMPLIANCE_CREDITS_PER_PERIOD = int(os.environ.get("OBSERVE_COMPLIANCE_CREDITS_PER_PERIOD", "600000"))
COMPLIANCE_RATE_CAPACITY = float(os.environ.get("OBSERVE_COMPLIANCE_RATE_LIMIT_CAPACITY", "100"))
COMPLIANCE_RATE_REFILL_PER_SEC = float(os.environ.get("OBSERVE_COMPLIANCE_RATE_LIMIT_PER_SEC", "50"))

TIERS = {
    "starter": {
        "price_cents": STARTER_PRICE_CENTS,
        "credits_per_period": STARTER_CREDITS_PER_PERIOD,
        "rate_capacity": STARTER_RATE_CAPACITY,
        "rate_refill_per_sec": STARTER_RATE_REFILL_PER_SEC,
        "name": "OBSERVE API Starter",
        "description": f"{STARTER_CREDITS_PER_PERIOD:,} credits/mo, 2x rate limit",
    },
    "compliance": {
        "price_cents": COMPLIANCE_PRICE_CENTS,
        "credits_per_period": COMPLIANCE_CREDITS_PER_PERIOD,
        "rate_capacity": COMPLIANCE_RATE_CAPACITY,
        "rate_refill_per_sec": COMPLIANCE_RATE_REFILL_PER_SEC,
        "name": "OBSERVE API Compliance",
        "description": f"{COMPLIANCE_CREDITS_PER_PERIOD:,} credits/mo, 10x rate limit, "
                        f"free private indexing, usage audit-log export",
    },
}

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


def create_pro_checkout_session(email: str, key_hash: str, tier: str) -> str:
    """mode=subscription with an inline recurring price_data -- no
    pre-created Stripe Price object needed in the dashboard, same
    zero-dashboard-setup convenience the one-time package's price_data
    already has. subscription_data.metadata (not just the top-level
    session metadata) is the important part: it's copied onto the actual
    Subscription object Stripe creates, so every future invoice.paid /
    customer.subscription.deleted webhook for this subscription carries
    BOTH key_hash and tier directly -- those events reference the
    subscription/customer, not this checkout session, so without this
    they'd have no way back to the right api_keys row or to which tier's
    price/credits/limits to apply on renewal."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r} -- must be one of {list(TIERS)}")
    spec = TIERS[tier]
    session = stripe.checkout.Session.create(
        mode="subscription",
        customer_email=email,
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": spec["name"],
                    "description": spec["description"],
                },
                "unit_amount": spec["price_cents"],
                "recurring": {"interval": "month"},
            },
            "quantity": 1,
        }],
        metadata={"key_hash": key_hash, "tier": tier},
        subscription_data={"metadata": {"key_hash": key_hash, "tier": tier}},
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
        if session.get("mode") == "subscription":
            # The FIRST invoice for a new subscription is paid as part of
            # Checkout itself -- this event confirms the subscription
            # exists, but invoice.paid (below) is what actually carries the
            # invoice id + period_end needed to grant credits idempotently.
            # Stripe fires both for a new subscription; deliberately doing
            # the credit grant only in the invoice.paid branch means one
            # code path handles both first-payment and every renewal,
            # instead of two separate grant implementations that could drift.
            return
        key_hash = session["metadata"]["key_hash"]
        credits = int(session["metadata"]["credits"])
        amount_cents = session["amount_total"]
        db.add_credits(key_hash, credits, session["id"], amount_cents)
        _notify_discord_sale(amount_cents, credits)

    elif event["type"] == "invoice.paid":
        invoice = event["data"]["object"]
        subscription_id = invoice.get("subscription")
        if not subscription_id:
            return  # a one-time-purchase invoice, not a subscription -- nothing to do here
        sub = stripe.Subscription.retrieve(subscription_id)
        key_hash = sub["metadata"].get("key_hash")
        tier = sub["metadata"].get("tier")
        if not key_hash or tier not in TIERS:
            print(f"(pro invoice.paid for subscription {subscription_id} has no valid key_hash/tier "
                  f"metadata -- predates this feature or was created outside create_pro_checkout_session, "
                  f"skipping)", flush=True)
            return
        credits_granted = TIERS[tier]["credits_per_period"]
        try:
            db.activate_pro(
                key_hash=key_hash,
                stripe_customer_id=invoice["customer"],
                stripe_subscription_id=subscription_id,
                period_end=sub["current_period_end"],
                credits_granted=credits_granted,
                stripe_invoice_id=invoice["id"],
                tier=tier,
            )
            _notify_discord_sale(invoice["amount_paid"], credits_granted)
        except sqlite3.IntegrityError:
            # Redelivered webhook for an invoice already applied -- see
            # activate_pro's docstring. Correct, expected no-op, not an error.
            print(f"(invoice {invoice['id']} already applied, ignoring redelivered webhook)", flush=True)

    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        db.deactivate_pro(sub["customer"])
