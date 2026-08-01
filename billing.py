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
import os

import stripe

from fastapi import HTTPException

import db

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# One package for v1: $10 -> 1000 credits (1 cent/credit). Change via env
# vars, not code, so pricing can be tuned without a redeploy.
PACKAGE_PRICE_CENTS = int(os.environ.get("OBSERVE_PACKAGE_PRICE_CENTS", "1000"))
PACKAGE_CREDITS = int(os.environ.get("OBSERVE_PACKAGE_CREDITS", "1000"))

SUCCESS_URL = os.environ.get("OBSERVE_CHECKOUT_SUCCESS_URL", "https://example.com/success")
CANCEL_URL = os.environ.get("OBSERVE_CHECKOUT_CANCEL_URL", "https://example.com/cancel")


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
