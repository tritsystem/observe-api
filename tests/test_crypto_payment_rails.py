"""
Tests crypto_payment_rails.py's real Bech32/Base58Check implementations.
Deliberately round-trips (encode -> decode -> verify) rather than
depending on memorized external test vectors that could themselves be
mistyped -- self-verifying, same rigor the module's own docstring
claims for itself.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import crypto_payment_rails as cpr  # noqa: E402


def test_bech32_round_trip():
    data = [0, 14, 20, 15, 7, 13, 26, 0, 25, 18, 6, 11, 13, 8, 21]  # arbitrary 5-bit values
    encoded = cpr.bech32_encode("bc", data, cpr.BECH32_CONST)
    decoded = cpr.bech32_decode(encoded)
    assert decoded is not None
    hrp, out_data, spec_const = decoded
    assert hrp == "bc"
    assert out_data == data
    assert spec_const == cpr.BECH32_CONST


def test_bech32m_round_trip():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    encoded = cpr.bech32_encode("bc", data, cpr.BECH32M_CONST)
    decoded = cpr.bech32_decode(encoded)
    assert decoded is not None
    assert decoded[2] == cpr.BECH32M_CONST


def test_bech32_rejects_corrupted_checksum():
    encoded = cpr.bech32_encode("bc", [1, 2, 3, 4, 5])
    # flip the last character (part of the checksum)
    last = encoded[-1]
    replacement = "q" if last != "q" else "p"
    corrupted = encoded[:-1] + replacement
    assert cpr.bech32_decode(corrupted) is None


def test_bech32_rejects_mixed_case():
    encoded = cpr.bech32_encode("bc", [1, 2, 3])
    mixed = encoded[:3] + encoded[3:].upper()
    assert cpr.bech32_decode(mixed) is None


def test_base58check_round_trip():
    payload = bytes([0x00]) + bytes(range(20))  # version byte 0x00 + a fake 20-byte hash
    encoded = cpr.base58check_encode(payload)
    decoded = cpr.base58check_decode(encoded)
    assert decoded == payload


def test_base58check_rejects_corrupted_checksum():
    payload = bytes([0x00]) + bytes(range(20))
    encoded = cpr.base58check_encode(payload)
    mutated = list(encoded)
    # mutate a middle character (payload region, not guaranteed-safe edge)
    mid = len(mutated) // 2
    original_char = mutated[mid]
    mutated[mid] = "1" if original_char != "1" else "2"
    corrupted = "".join(mutated)
    assert cpr.base58check_decode(corrupted) is None


def test_validate_onchain_btc_address_accepts_valid_bech32_p2wpkh():
    # a 20-byte witness program, version 0, is the real shape of a P2WPKH address
    witness_program = [0] + _convert_bits(list(range(20)))
    address = cpr.bech32_encode("bc", witness_program, cpr.BECH32_CONST)
    assert cpr.validate_onchain_btc_address(address, network="mainnet") is True


def test_validate_onchain_btc_address_rejects_wrong_network():
    witness_program = [0] + _convert_bits(list(range(20)))
    testnet_address = cpr.bech32_encode("tb", witness_program, cpr.BECH32_CONST)
    assert cpr.validate_onchain_btc_address(testnet_address, network="mainnet") is False
    assert cpr.validate_onchain_btc_address(testnet_address, network="testnet") is True


def test_validate_onchain_btc_address_accepts_valid_legacy_p2pkh():
    payload = bytes([0x00]) + bytes(range(20))
    address = cpr.base58check_encode(payload)
    assert cpr.validate_onchain_btc_address(address, network="mainnet") is True


def test_validate_onchain_btc_address_rejects_garbage():
    assert cpr.validate_onchain_btc_address("not-a-real-btc-address") is False
    assert cpr.validate_onchain_btc_address("") is False


def test_validate_lightning_invoice_accepts_valid_structure():
    invoice = cpr.bech32_encode("lnbc", [0, 1, 2, 3, 4, 5], cpr.BECH32_CONST)
    assert cpr.validate_lightning_invoice(invoice) is True


def test_validate_lightning_invoice_rejects_a_btc_address():
    witness_program = [0] + _convert_bits(list(range(20)))
    btc_address = cpr.bech32_encode("bc", witness_program, cpr.BECH32_CONST)
    assert cpr.validate_lightning_invoice(btc_address) is False


def test_validate_lightning_address_accepts_real_shape():
    assert cpr.validate_lightning_address("seller@example.com") is True
    assert cpr.validate_lightning_address("not-an-address") is False
    assert cpr.validate_lightning_address("@missing-local.com") is False


def test_validate_evm_address_accepts_real_shape():
    assert cpr.validate_evm_address("0x" + "a" * 40) is True
    assert cpr.validate_evm_address("0x" + "a" * 39) is False  # too short
    assert cpr.validate_evm_address("not-an-address") is False


def _convert_bits(data_8bit, from_bits=8, to_bits=5, pad=True):
    """Minimal bit-regrouping helper for building a real witness-program
    test fixture (8-bit bytes -> 5-bit bech32 groups) -- BIP-173's own
    conversion, needed here only to build valid segwit fixtures for the
    tests above."""
    acc, bits, ret = 0, 0, []
    maxv = (1 << to_bits) - 1
    for value in data_8bit:
        acc = (acc << from_bits) | value
        bits += from_bits
        while bits >= to_bits:
            bits -= to_bits
            ret.append((acc >> bits) & maxv)
    if pad and bits:
        ret.append((acc << (to_bits - bits)) & maxv)
    return ret
