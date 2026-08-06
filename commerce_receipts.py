"""
Cryptographically signed receipts for real commerce events -- the one
genuinely good idea from a researched comparison against ERC-8183/ICP-style
protocols, deliberately WITHOUT their other half (on-chain escrow, custody
of funds, a "Settler" that actually moves money). OBSERVE still never
touches payment -- see commerce_router.py's module docstring -- this only
makes the events it DOES observe (a match happening, a buyer/seller
reporting an outcome) independently verifiable by a third party who never
has to trust OBSERVE's live API or database, only its Ed25519 public key.

Real, disclosed scope: this is NOT the same class of guarantee as an
on-chain escrow receipt (ERC-8183) or ICP's settlement receipts, which
attest that money actually moved. A signed OBSERVE receipt attests only
"OBSERVE recorded this claim, from this caller, at this time, unmodified
since" -- the same trust boundary the rest of this system already has
(self-reported by both sides, see commerce_router.py), just now with
tamper-evidence and a timestamp neither side can quietly rewrite after
the fact. Worth having; not worth overselling as more than it is.

Uses PyJWT's EdDSA support (Ed25519) rather than hand-rolled signing --
JWS compact serialization (header.payload.signature, all base64url), the
same format ICP's own docs reference, so a receipt is verifiable with any
standard JWT/JOSE library, not just this codebase.
"""
import base64
import os

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_SEED_ENV_VAR = "OBSERVE_RECEIPT_SIGNING_KEY_SEED"


def _b64url_uint(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


class ReceiptSigner:
    def __init__(self, seed_hex: str | None = None):
        # Real, disclosed limitation: if OBSERVE_RECEIPT_SIGNING_KEY_SEED
        # isn't set, a fresh key is generated at process startup -- every
        # past receipt still verifies fine against ITS OWN embedded event
        # data, but the public key published at /.well-known/... changes
        # on every restart, breaking any third party that cached the old
        # one. Set the env var in production; ephemeral-by-default here
        # only so a fresh dev/test run never crashes on a missing secret.
        seed_hex = seed_hex if seed_hex is not None else os.environ.get(_SEED_ENV_VAR)
        if seed_hex:
            seed = bytes.fromhex(seed_hex)
            if len(seed) != 32:
                raise ValueError(f"{_SEED_ENV_VAR} must be 32 bytes (64 hex chars), got {len(seed)} bytes")
            self.private_key = Ed25519PrivateKey.from_private_bytes(seed)
            self.ephemeral = False
        else:
            self.private_key = Ed25519PrivateKey.generate()
            self.ephemeral = True
        self.public_key = self.private_key.public_key()
        self.kid = _b64url_uint(
            self.public_key.public_bytes_raw()
        )[:16]  # short, stable-per-key identifier, not a secret

    def sign(self, payload: dict) -> str:
        """Returns a compact JWS string. `payload` should be JSON-safe
        (str/int/float/bool/None/dict/list) -- caller's responsibility,
        same as any jwt.encode call."""
        return jwt.encode(payload, self.private_key, algorithm="EdDSA", headers={"kid": self.kid})

    def verify(self, token: str) -> dict:
        """Raises jwt.InvalidSignatureError (or other jwt exceptions) if
        the token doesn't verify against this signer's own public key --
        callers that just want a bool should catch and return False, this
        stays strict so tests can assert on the specific failure."""
        return jwt.decode(token, self.public_key, algorithms=["EdDSA"])

    def public_jwk(self) -> dict:
        """RFC 8037 JWK representation -- verifiable by any standard JOSE
        library, not just PyJWT, so a third party isn't locked into this
        codebase's specific tooling to check a receipt."""
        return {
            "kty": "OKP",
            "crv": "Ed25519",
            "x": _b64url_uint(self.public_key.public_bytes_raw()),
            "kid": self.kid,
            "use": "sig",
            "alg": "EdDSA",
        }


def generate_seed_hex() -> str:
    """For operators: run this once, put the result in
    OBSERVE_RECEIPT_SIGNING_KEY_SEED, and never regenerate it in
    production -- regenerating rotates the public key and orphans every
    previously-issued receipt's verifiability against the new one."""
    return Ed25519PrivateKey.generate().private_bytes_raw().hex()


if __name__ == "__main__":
    print(generate_seed_hex())
