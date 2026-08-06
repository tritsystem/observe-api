"""
Format validation for crypto payment endpoints a seller can register
alongside their real ACP checkout_session_url -- lets a listing point a
buyer-agent at Lightning, on-chain BTC, or an x402/stablecoin resource
as an alternative or supplement to an HTTPS checkout call, WITHOUT
OBSERVE touching payment in any way: no wallet, no node connection, no
balance check, no transaction ever submitted or observed. This module
only answers "is this syntactically a real address/invoice," the exact
same class of check checkout_session_url already gets (must start with
https://) -- a real, structural validation via actual checksum
algorithms (BIP-173/350 Bech32, Base58Check), not a cosmetic regex, so
a seller fat-fingering an address is caught before it's ever shown to a
buyer-agent as "real."

Real, disclosed limitation: a syntactically valid address can still be
the wrong one, already closed, or belong to someone else entirely --
checksum validation only rules out the "obviously broken" case (typos,
wrong network, truncation), the same way "this string parses as an
email" doesn't mean the address exists.
"""
import hashlib
import re

BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BECH32_CONST = 1
BECH32M_CONST = 0x2BC830A3


def _bech32_polymod(values):
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp):
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def bech32_encode(hrp, data, spec_const=BECH32_CONST):
    """Real encoder, kept alongside the decoder so tests can round-trip
    (encode -> decode -> verify) instead of depending on memorized
    external test vectors that could themselves be mistyped."""
    values = _bech32_hrp_expand(hrp) + data
    polymod = _bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ spec_const
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(BECH32_CHARSET[d] for d in data + checksum)


def bech32_decode(bech):
    """Returns (hrp, data, spec_const) if valid (spec_const tells the
    caller whether it verified as Bech32 or Bech32m -- BOLT11 invoices
    and legacy segwit v0 addresses use plain Bech32; segwit v1+ (taproot)
    addresses use Bech32m per BIP-350), or None if the checksum/format
    is invalid."""
    if any(ord(c) < 33 or ord(c) > 126 for c in bech):
        return None
    if bech.lower() != bech and bech.upper() != bech:
        return None  # mixed case is invalid per spec
    bech = bech.lower()
    pos = bech.rfind("1")
    if pos < 1 or pos + 7 > len(bech) or len(bech) > 90:
        return None
    hrp = bech[:pos]
    data_part = bech[pos + 1:]
    if any(c not in BECH32_CHARSET for c in data_part):
        return None
    data = [BECH32_CHARSET.index(c) for c in data_part]
    values = _bech32_hrp_expand(hrp) + data
    checksum_value = _bech32_polymod(values)
    if checksum_value == BECH32_CONST:
        return hrp, data[:-6], BECH32_CONST
    if checksum_value == BECH32M_CONST:
        return hrp, data[:-6], BECH32M_CONST
    return None


_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def base58check_encode(payload: bytes) -> str:
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    full = payload + checksum
    n = int.from_bytes(full, "big")
    out = ""
    while n > 0:
        n, rem = divmod(n, 58)
        out = _BASE58_ALPHABET[rem] + out
    n_leading_zero_bytes = len(full) - len(full.lstrip(b"\x00"))
    return "1" * n_leading_zero_bytes + out


def base58check_decode(s: str):
    """Returns the payload bytes (version byte + hash) if the checksum
    verifies, else None. Real double-SHA256 checksum check -- this is
    what actually catches a fat-fingered legacy address; a plain
    alphabet/length check would not."""
    if not s or any(c not in _BASE58_ALPHABET for c in s):
        return None
    n = 0
    for c in s:
        n = n * 58 + _BASE58_ALPHABET.index(c)
    n_leading_ones = len(s) - len(s.lstrip("1"))
    full = n.to_bytes((n.bit_length() + 7) // 8, "big") if n > 0 else b""
    full = b"\x00" * n_leading_ones + full
    if len(full) < 5:
        return None
    payload, checksum = full[:-4], full[-4:]
    real_checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    if checksum != real_checksum:
        return None
    return payload


def validate_onchain_btc_address(address: str, network: str = "mainnet") -> bool:
    """Structural validation only (see module docstring). Covers legacy
    P2PKH/P2SH (Base58Check, version bytes 0x00/0x05 mainnet or
    0x6F/0xC4 testnet) and native segwit v0/v1+ (Bech32/Bech32m, hrp
    'bc'/'tb')."""
    hrp_expected = "bc" if network == "mainnet" else "tb"
    if address.lower().startswith(hrp_expected + "1"):
        decoded = bech32_decode(address)
        return decoded is not None and decoded[0] == hrp_expected
    payload = base58check_decode(address)
    if payload is None:
        return False
    version = payload[0]
    valid_versions = {0x00, 0x05} if network == "mainnet" else {0x6F, 0xC4}
    return version in valid_versions and len(payload) == 21


def validate_lightning_invoice(invoice: str, network: str = "mainnet") -> bool:
    """Structural validation of a BOLT11 invoice's Bech32 envelope and
    human-readable prefix only -- does NOT decode/verify the embedded
    payment hash, amount, or signature (that needs real secp256k1
    recovery, out of scope for "is this well-formed"). A structurally
    valid-but-expired or already-paid invoice still passes this check,
    same disclosed limitation as the address validators above.

    Kept as a general-purpose utility -- a single BOLT11 invoice is a
    one-time, amount-specific payment request, so it doesn't fit a
    STANDING seller registration (see validate_lightning_address for
    that); this is here for a future per-quote invoice use case."""
    prefix = "lnbc" if network == "mainnet" else "lntb"
    if not invoice.lower().startswith(prefix):
        return False
    decoded = bech32_decode(invoice)
    return decoded is not None and decoded[0].startswith(prefix)


_LIGHTNING_ADDRESS_RE = re.compile(r"^[a-z0-9._-]+@[a-z0-9-]+(\.[a-z0-9-]+)+$", re.IGNORECASE)


def validate_lightning_address(address: str) -> bool:
    """Lightning Address (LUD-16) -- 'name@domain.com', the actual
    standing/reusable payment identifier real Lightning merchants
    publish (unlike a one-time BOLT11 invoice). Real resolution needs an
    HTTPS GET to https://domain/.well-known/lnurlp/name, which this
    deliberately does NOT perform (no network call from inside a
    synchronous registration handler, same reasoning checkout_session_url
    is never pinged either) -- format-only, disclosed."""
    return bool(_LIGHTNING_ADDRESS_RE.match(address)) and len(address) <= 255


_EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def validate_evm_address(address: str) -> bool:
    """0x + 40 hex chars -- the address shape stablecoins like USDC on
    Base/Ethereum/etc. use. Deliberately does NOT verify EIP-55 mixed-
    case checksums (a real, valid address can legitimately be all-
    lowercase) or that the address exists/holds a balance on any chain
    -- shape only."""
    return bool(_EVM_ADDRESS_RE.match(address))
