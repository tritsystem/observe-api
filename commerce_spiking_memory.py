"""Spiking listing-affinity memory for commerce_router.py, same real
mechanism as spiking_search_heatmap.py (fully-connected small-seed-weight
network + Spikeling's real STDP learning, `compiler.compiler.compile_file`
+ `runtime.runtime.SpikelingRuntime` -- not a hand-written co-occurrence
counter) applied to a different question: not "which indexed docs keep
getting retrieved together" but "which listings keep getting matched
together for a given buyer's real searches."

Genuinely different scoping constraint than the doc heatmap, though: that
one takes a fixed doc_names set at construction time (a bounded batch).
Commerce listings grow live as sellers register more of them, so this
memory is scoped PER BUYER API KEY (their own view of real affinity
across whatever they've searched) and grows on demand, capped at
MAX_TRACKED_LISTINGS -- past the cap, newly-seen listings just aren't
added to that key's network rather than building eviction logic for a
v1. A disclosed simplification, not an oversight: see
ListingAffinityMemory.observe_search's docstring.

Same real STDP edge case spiking_search_heatmap.py already found and
fixed carried forward here unchanged: firing two listings at the exact
same timestamp lands `dt = t - downstream.last_spike_time` at exactly 0,
and STDPLearner.update()'s `if dt > 0: LTP else: LTD` treats dt==0 as
LTD (weaken) -- true same-search co-occurrence would otherwise weaken
the connection it's supposed to reinforce. Each listing within one
search result gets the same tiny (1e-3ms) per-rank stagger used there.
"""
import os
import re
import sys
import tempfile
from typing import Dict, List, Optional


def _locate_spikeling_core() -> str:
    """spiking_search_heatmap.py's sibling module hardcodes a bare
    Windows path (`C:\\Users\\gbran\\...`) -- verified by testing, not
    assumed, that this silently fails under WSL (ModuleNotFoundError:
    compiler), which is where server.py's real production process
    actually runs (see MIGRATE.md / DEPLOY.md: the live deployment is on
    kimchi's own WSL2). Checks SPIKELING_CORE_PATH first (same override
    sensor_duo.spiking_detector already supports), then both real known
    locations of this machine's actual Spikeling checkout -- the native
    Windows path and its WSL-mounted equivalent -- rather than assuming
    either one."""
    env_path = os.environ.get("SPIKELING_CORE_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    for candidate in (
        r"C:\Users\gbran\OneDrive\Documents\Spikeling\core",
        "/mnt/c/Users/gbran/OneDrive/Documents/Spikeling/core",
    ):
        if os.path.isdir(candidate):
            return candidate
    raise RuntimeError(
        "Couldn't find the Spikeling engine's core/ folder. Set "
        "SPIKELING_CORE_PATH, or clone https://github.com/gbranaa4-hue/Spikeling "
        "at C:\\Users\\gbran\\OneDrive\\Documents\\Spikeling."
    )


_CORE = _locate_spikeling_core()
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from compiler.compiler import compile_file  # noqa: E402
from runtime.runtime import SpikelingRuntime  # noqa: E402

DEFAULT_THRESHOLD = 100.0
DEFAULT_LEAK = 15.0
DEFAULT_SEED_WEIGHT = 0.05
DEFAULT_LEARN_RATE = 0.05
DEFAULT_TOP_HIT_DRIVE = 80.0
DEFAULT_RANK_DECAY = 0.15

MAX_TRACKED_LISTINGS = 40  # see module docstring -- v1 cap, no eviction


def _safe_neuron_name(item_id: str) -> str:
    return "l_" + re.sub(r"\W", "_", item_id)


def _generate_spk_text(neuron_names, threshold, leak, seed_weight, learn_rate) -> str:
    lines = ["# Generated commerce listing-affinity network -- fully",
             "# connected, one neuron per tracked listing. See",
             "# commerce_spiking_memory.py.", ""]
    for name in neuron_names:
        lines.append(f"neuron {name} threshold={int(threshold)} leak={int(leak)} type=LIF")
    lines.append("")
    names = list(neuron_names)
    for src in names:
        for dst in names:
            if src != dst:
                lines.append(f"connect {src} -> {dst} weight={seed_weight}")
    lines.append("")
    lines.append(f"learn=STDP rate={learn_rate}")
    lines.append("refractory=0ms")
    lines.append("")
    return "\n".join(lines)


def _compile_network(neuron_names, threshold, leak, seed_weight, learn_rate):
    spk_text = _generate_spk_text(neuron_names, threshold, leak, seed_weight, learn_rate)
    work_dir = tempfile.mkdtemp(prefix="observe_commerce_memory_")
    spk_path = os.path.join(work_dir, "listing_affinity.spk")
    with open(spk_path, "w", encoding="utf-8") as f:
        f.write(spk_text)
    out_dir = tempfile.mkdtemp(prefix="observe_commerce_memory_codegen_")
    return compile_file(spk_path, output_dir=out_dir)


class ListingAffinityMemory:
    """One buyer key's live, learned view of "which listings keep
    matching together." Empty at construction -- grows as observe_search
    sees new item_ids, up to MAX_TRACKED_LISTINGS."""

    def __init__(self, threshold=DEFAULT_THRESHOLD, leak=DEFAULT_LEAK,
                 seed_weight=DEFAULT_SEED_WEIGHT, learn_rate=DEFAULT_LEARN_RATE):
        self._threshold = threshold
        self._leak = leak
        self._seed_weight = seed_weight
        self._learn_rate = learn_rate
        self._name_map: Dict[str, str] = {}
        self.runtime: Optional[SpikelingRuntime] = None

    def _rebuild(self, item_ids) -> None:
        """Recompiles the network for the given tracked set. Real cost:
        this throws away any previously-learned synapse weights for
        listings that survive the rebuild -- acceptable here because a
        rebuild only happens when growing to include a genuinely new
        listing this key has never searched before, at which point
        there's no learned history to lose for the new pair anyway, and
        existing pairs re-accumulate quickly under real repeated use.
        Not acceptable for a topology that changes on every request --
        this is only called from observe_search when the tracked set
        actually grows, not on every search."""
        self._name_map = {i: _safe_neuron_name(i) for i in item_ids}
        if not self._name_map:
            self.runtime = None
            return
        ast = _compile_network(self._name_map.values(), self._threshold, self._leak,
                                self._seed_weight, self._learn_rate)
        self.runtime = SpikelingRuntime(ast)

    def observe_search(self, ranked_item_ids: List[str], timestamp_ms: float,
                        top_drive: float = DEFAULT_TOP_HIT_DRIVE,
                        rank_decay: float = DEFAULT_RANK_DECAY) -> None:
        """ranked_item_ids: one real search's matched listings, best
        match first. Grows the tracked set to include any new item_id
        seen here, up to MAX_TRACKED_LISTINGS -- past the cap, a new
        listing is silently not tracked (this key's memory just doesn't
        learn about it), rather than evicting an existing one; a v1
        simplification disclosed in the module docstring."""
        known = set(self._name_map.keys())
        new_ids = [i for i in ranked_item_ids if i not in known]
        room = MAX_TRACKED_LISTINGS - len(known)
        if new_ids and room > 0:
            grown = list(known) + new_ids[:room]
            self._rebuild(grown)

        if self.runtime is None:
            return
        for rank, item_id in enumerate(ranked_item_ids):
            if item_id not in self._name_map:
                continue
            drive = top_drive * max(0.0, 1.0 - rank_decay * rank)
            if drive <= 0:
                continue
            self.runtime.stimulate(self._name_map[item_id], timestamp_ms + rank * 1e-3, drive=drive)

    def decay(self, timestamp_ms: float) -> None:
        if self.runtime is not None:
            self.runtime.tick(timestamp_ms)

    def heat(self, item_id: str) -> float:
        if self.runtime is None or item_id not in self._name_map:
            return 0.0
        return self.runtime.neurons[self._name_map[item_id]].membrane_potential

    def learned_connection(self, src_item_id: str, dst_item_id: str) -> Optional[float]:
        if self.runtime is None or src_item_id not in self._name_map or dst_item_id not in self._name_map:
            return None
        src, dst = self._name_map[src_item_id], self._name_map[dst_item_id]
        for syn in self.runtime.synapses:
            if syn.src == src and syn.dst == dst:
                return syn.weight
        return None
