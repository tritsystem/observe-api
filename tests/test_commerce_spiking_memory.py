"""
Tests for commerce_spiking_memory.py against the REAL Spikeling engine
(no fake/mock -- this module doesn't touch OBSERVE's embedding model at
all, so there's no reason to stub anything; if the real Spikeling core
fails to import, these tests fail loudly instead of silently passing
against a mock that could never have caught the WSL path bug this
module was fixed for).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commerce_spiking_memory import ListingAffinityMemory, MAX_TRACKED_LISTINGS  # noqa: E402


def test_never_seen_listing_has_zero_heat():
    mem = ListingAffinityMemory()
    assert mem.heat("1:anything") == 0.0


def test_repeated_co_occurrence_builds_real_heat_and_learned_connection():
    mem = ListingAffinityMemory()
    t = 1000.0
    for _ in range(5):
        mem.decay(t)
        mem.observe_search(["1:boots", "1:hat"], t)
        t += 60_000.0

    # Real firing happened -- both listings integrated real drive.
    assert mem.heat("1:boots") > 0.0
    assert mem.heat("1:hat") > 0.0
    # A listing that was never in any search result stays at zero --
    # heat is specific, not a global counter that drifts up regardless.
    assert mem.heat("1:never-searched") == 0.0

    # Real STDP learning happened, not just firing -- a connection
    # exists and grew past the seed weight for at least one direction
    # (see module docstring for the real, disclosed asymmetry: the
    # later-firing listing's synapse back to the earlier one is the one
    # that grows).
    forward = mem.learned_connection("1:boots", "1:hat")
    backward = mem.learned_connection("1:hat", "1:boots")
    assert forward is not None and backward is not None
    assert backward > forward, "expected the real STDP asymmetry (later-firing's synapse back to earlier grows more)"


def test_top_ranked_listing_gets_more_heat_than_lower_ranked():
    """Rank-decay drive (see observe_search's top_drive/rank_decay) means
    the #1 result in a search should end up with more heat than a
    lower-ranked one that co-appeared -- verifies the rank signal
    actually reaches the network, not just presence/absence."""
    mem = ListingAffinityMemory()
    mem.decay(0.0)
    mem.observe_search(["1:first", "1:second", "1:third"], 0.0)
    assert mem.heat("1:first") > mem.heat("1:second") > mem.heat("1:third")


def test_tracked_set_caps_at_max_without_crashing():
    mem = ListingAffinityMemory()
    t = 0.0
    # More distinct listings across repeated searches than the cap --
    # real behavior should be "stop tracking new ones," not an error.
    ids = [f"1:item-{i}" for i in range(MAX_TRACKED_LISTINGS + 10)]
    for i in range(0, len(ids), 2):
        mem.decay(t)
        mem.observe_search(ids[i:i + 2], t)
        t += 1000.0
    assert len(mem._name_map) <= MAX_TRACKED_LISTINGS
    # A listing that never made it under the cap is simply untracked
    # (heat 0), not a crash.
    assert mem.heat(ids[-1]) == 0.0
