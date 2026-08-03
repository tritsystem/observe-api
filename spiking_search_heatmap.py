"""Spiking activity-propagation 'living heatmap' for observe-api search,
using a fully-connected small-seed-weight network with real STDP learning
(the same core/runtime/runtime.py mechanism as spiking_adaptive_weights.py,
applied here to a different question: not "which evidence kind predicts
real usage" but "which indexed docs keep getting retrieved together").

usage_log already records which query hit which docs. This turns that
into a live network: one neuron per indexed doc, all pairs seeded with a
small connection weight. Every real query drives the docs it actually
returned (top hit strongest, lower ranks weaker); docs that keep
co-appearing in the SAME result set fire close together in time, and
Spikeling's real STDP rule (learn=STDP) strengthens their connection --
so over many real queries, the network LEARNS which docs are actually
related from real co-retrieval, not a hand-written co-occurrence counter.
A doc's current membrane potential is its "heat" -- fires it was
directly hit for, plus secondhand warmth propagated from a strongly-
connected doc that just fired. Between queries, tick() lets heat decay,
so the map reflects recent real usage, not all-time totals.

Real asymmetry, found by testing (see observe_query()'s docstring for the
full story): the actual STDPLearner rule rewards the LATER-firing doc's
synapse back to the EARLIER-firing one, not the other direction -- so for
two docs that keep appearing in the same query, learned_connection(later,
earlier) grows, while learned_connection(earlier, later) stays at the
seed weight. Check both directions if a symmetric "these two are related"
signal is what's needed; don't assume the pair strengthens both ways.

This is a genuinely different use of STDP than spiking_adaptive_weights.py
(there: evidence-kind trustworthiness learned from ground-truth replay;
here: doc-relatedness learned from real co-retrieval), and a genuinely
different topology (there: star, N sources -> 1 sink; here: a fully-
connected population where any pair can influence any other).

Scoped to a bounded, explicit doc set (passed in, not auto-discovered) --
a fully-connected network is O(N^2) connections, fine for the tens of
docs a real usage pattern needs to track, not meant for observe-api's
entire multi-thousand-chunk index at once.
"""
import os
import re
import sys
import tempfile

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
_CORE = os.path.join(SPIKELING_ROOT, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from compiler.compiler import compile_file  # noqa: E402
from runtime.runtime import SpikelingRuntime  # noqa: E402

DEFAULT_THRESHOLD = 100.0
DEFAULT_LEAK = 15.0
DEFAULT_SEED_WEIGHT = 0.05   # small: one co-firing alone shouldn't confirm
                              # a relationship, only repeated real co-
                              # occurrence (via STDP reinforcement) should
DEFAULT_LEARN_RATE = 0.05
DEFAULT_TOP_HIT_DRIVE = 80.0
DEFAULT_RANK_DECAY = 0.15    # each rank position below the top hit gets
                              # this fraction less drive


def _safe_neuron_name(doc_name):
    return "d_" + re.sub(r"\W", "_", doc_name)


def _generate_spk_text(neuron_names, threshold, leak, seed_weight, learn_rate):
    lines = ["# Generated search-activity-heatmap network -- fully",
             "# connected, one neuron per tracked doc. See",
             "# spiking_search_heatmap.py.", ""]
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


def _compile_heatmap_network(neuron_names, threshold, leak, seed_weight, learn_rate):
    spk_text = _generate_spk_text(neuron_names, threshold, leak, seed_weight, learn_rate)
    work_dir = tempfile.mkdtemp(prefix="observe_search_heatmap_")
    spk_path = os.path.join(work_dir, "search_heatmap.spk")
    with open(spk_path, "w", encoding="utf-8") as f:
        f.write(spk_text)
    out_dir = tempfile.mkdtemp(prefix="observe_search_heatmap_codegen_")
    return compile_file(spk_path, output_dir=out_dir)


class SearchActivityHeatmap:
    def __init__(self, doc_names, threshold=DEFAULT_THRESHOLD, leak=DEFAULT_LEAK,
                 seed_weight=DEFAULT_SEED_WEIGHT, learn_rate=DEFAULT_LEARN_RATE):
        self._name_map = {d: _safe_neuron_name(d) for d in doc_names}
        ast = _compile_heatmap_network(self._name_map.values(), threshold, leak,
                                        seed_weight, learn_rate)
        self.runtime = SpikelingRuntime(ast)

    def observe_query(self, ranked_hit_docs, timestamp_ms,
                       top_drive=DEFAULT_TOP_HIT_DRIVE, rank_decay=DEFAULT_RANK_DECAY):
        """ranked_hit_docs: real search results for one query, most
        relevant first. Drives each hit directly (rank-decayed), letting
        real synapse propagation + STDP handle secondary warmth and
        learned relatedness.

        Real edge case, found by testing rather than assumed: firing two
        docs at the EXACT same timestamp lands STDPLearner.update()'s
        `dt = t - downstream.last_spike_time` at exactly 0, and the real
        code's `if dt > 0: LTP else: LTD` treats dt==0 as the LTD (weaken)
        branch, not LTP -- true same-query co-occurrence would otherwise
        WEAKEN the connection it's supposed to reinforce. Each doc within
        one query gets a tiny (1e-3ms) stagger so co-occurring docs land
        strictly in the LTP branch, far smaller than any real gap between
        separate queries."""
        for rank, doc in enumerate(ranked_hit_docs):
            if doc not in self._name_map:
                continue
            drive = top_drive * max(0.0, 1.0 - rank_decay * rank)
            if drive <= 0:
                continue
            self.runtime.stimulate(self._name_map[doc], timestamp_ms + rank * 1e-3, drive=drive)

    def decay(self, timestamp_ms):
        self.runtime.tick(timestamp_ms)

    def heat(self, doc_name):
        return self.runtime.neurons[self._name_map[doc_name]].membrane_potential

    def learned_connection(self, src_doc, dst_doc):
        src, dst = self._name_map[src_doc], self._name_map[dst_doc]
        for syn in self.runtime.synapses:
            if syn.src == src and syn.dst == dst:
                return syn.weight
        return None
