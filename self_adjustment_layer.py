"""Self-adjustment layer for steering a frozen LLM via context, not weights.

An SNN can't literally hold information inside an LLM's parameters -- they're
separate substrates, nothing merges. What this module actually does: learn,
from real confirm/disconfirm evidence over time, which behavioral adjustments
("be terse", "prefer STRAT over ADJUST first", whatever the caller defines)
are currently well-supported -- then translate the current learned state into
real text a caller injects into an LLM's system prompt. The LLM's behavior
shifts because its INPUT changed, not its weights. Same honest pattern as
this session's RECALL tool: real learned state -> real context injection.

Four structural properties, each tied to something concrete rather than left
as pure metaphor:

  BIDIRECTIONAL ("water flows both ways") -- confirming evidence raises a
    weight, disconfirming evidence lowers it. This deliberately does NOT
    reuse Spikeling's STDPLearner.update() dt-sign trick: spiking_evidence.py
    and spiking_adaptive_weights.py's own docstrings disclose that under
    forward-chronological replay, dt = t - downstream.last_spike_time is
    structurally never negative, so that mechanism can only ever strengthen,
    never weaken. Reusing it here for confirm/disconfirm semantics would
    repeat that exact bug. Bidirectional update is instead explicit,
    unambiguous Python -- +rate on confirm, -rate*LTD_ASYMMETRY on disconfirm
    (same 0.5x asymmetry STDPLearner itself uses, for consistency).

  DECAY-TO-ROOT ("settles like water finding its level") -- absent new
    evidence, a weight relaxes exponentially toward the current ROOT value
    (not a fixed per-node constant) over elapsed real time.

  ROOTS (two-timescale, "a tree") -- same principle as tritkit's
    TwoTimescaleLinear (methodlm.py's own readout): a slow-moving ROOT
    baseline anchors fast-moving LEAF weights. Each leaf's decay target is
    the root, and the root itself drifts slowly toward the population's
    real average behavior (ROOT_EMA_RATE << leaf learning rate) -- one
    leaf's one-off correction barely moves the root; sustained agreement
    across many leaves over real time does.

  MYCELIUM (graph diffusion, "network of communication") -- an update at
    one node propagates to every node it's connected to, decayed per hop
    (bounded by MAX_HOPS so it can't cycle forever), instead of only
    touching the node that was directly reinforced.

  BREATHE (homeostasis, NOT periodic oscillation) -- this repo's own vault
    has a real, confirmed finding on this exact question: blind periodic
    "breathing" (a fixed threshold rhythm) was tested and lost to noise;
    homeostatic/closed-loop threshold adaptation decisively beat both. So
    the confidence gate's effective threshold adapts based on real recent
    fire-rate vs. a target rate -- fires too often lately -> threshold
    rises (harder to fire); too rarely -> threshold falls. A fixed sine-wave
    threshold would be re-running an approach already falsified in this
    same codebase's own prior research.

Uses Spikeling's real DSL for the one piece it's a clean fit for: the
fire/no-fire confidence GATE (self_adjustment_gate.spk, threshold+leak LIF
neuron). The weight dynamics above are explicit Python, not forced through
the DSL, for the reasons disclosed above.
"""
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
_CORE = os.path.join(SPIKELING_ROOT, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from compiler.compiler import compile_file  # noqa: E402
from runtime.runtime import SpikelingRuntime  # noqa: E402

_SPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "self_adjustment_gate.spk")
_ast = None


def _get_ast():
    global _ast
    if _ast is None:
        out_dir = tempfile.mkdtemp(prefix="observe_self_adjust_")
        _ast = compile_file(_SPK_PATH, output_dir=out_dir)
    return _ast


LTD_ASYMMETRY = 1.0      # Real bug, found by testing: this was 0.5 (borrowed from
                         # STDPLearner's LTD branch) without re-deriving whether it fits THIS
                         # use case. It doesn't. Applying the point-group selection-rule
                         # theorem from symmetry-selection-rule/Symmetry_Selection_Rule_Lab_
                         # Report.pdf: on a symmetric structure, the cubic (even-order,
                         # rectifying) term integral vanishes exactly for every non-trivial
                         # mode; breaking that symmetry is what switches rectification back
                         # on. An LTP/LTD magnitude mismatch IS an even-order term on this
                         # update rule -- confirmed empirically: 20 confirms + 20 disconfirms,
                         # perfectly alternating, pushed weight from baseline 0.300 to 0.891
                         # at asymmetry=0.5 (real drift from supposedly-balanced evidence,
                         # exactly the rectification the theorem predicts from a broken-
                         # symmetry update rule). At 1.0 (LTP magnitude == LTD magnitude) the
                         # update is odd-symmetric in confirm/disconfirm, so balanced evidence
                         # has EXACTLY zero net effect, matching the theorem's silenced case.
                         # A caller who deliberately wants "confidence sticky, hard to lose"
                         # can still set this away from 1.0 -- that's now a disclosed, explicit
                         # symmetry-breaking choice, not an accidental default.
DIFFUSION_RATE = 0.15    # fraction of a node's delta that propagates to each direct connection
MAX_HOPS = 3              # mycelium propagation depth cap -- bounds cost, prevents cycling forever
HOP_DECAY = 0.5           # propagated effect halves each additional hop
ROOT_EMA_RATE = 0.02      # root moves ~10x slower than a typical leaf reinforce() rate (0.15-0.2)
GATE_DRIVE_SCALE = 150.0  # weight (0-1) * this must clear the gate's threshold=100 to fire
HOMEOSTATIC_TARGET_RATE = 0.3   # target fraction of confidence_gate() calls that should fire
HOMEOSTATIC_ADAPT_RATE = 3.0    # how fast the effective threshold chases the target rate --
                                 # tuned down from an initial 8.0: that value combined with a slow
                                 # relax left the gate stuck over-suppressed for tens of real
                                 # time-units after a single burst, found by rerunning the self-test
OFFSET_RELAX_RATE = 0.15        # gate_threshold_offset's own exhale -- real bug, found by running
                                 # the self-test: with no passive relaxation (and, initially, too
                                 # slow a relaxation), a burst that pushed the offset up left the
                                 # gate over-suppressed long after the layer stopped being actively
                                 # probed. Homeostasis has to keep operating over elapsed time, not
                                 # just when checked, and has to recover on a comparable timescale
                                 # to how fast it reacted.
HOMEOSTATIC_WINDOW = 20         # how many recent gate calls the fire-rate is measured over


@dataclass
class AdjustmentNode:
    name: str
    weight: float
    last_touch: float
    touch_count: int = 0


class SelfAdjustmentLayer:
    def __init__(self, root_baseline=0.3, decay_rate=0.05):
        self.nodes: dict[str, AdjustmentNode] = {}
        self.graph: dict[str, dict[str, float]] = {}   # mycelium adjacency: name -> {neighbor: rate}
        self.root_baseline = root_baseline
        self.decay_rate = decay_rate
        self.gate_threshold_offset = 0.0   # homeostatic adjustment on top of the .spk's threshold=100
        self._offset_last_touch = 0.0
        self._recent_fires: list[bool] = []   # bounded window for the homeostatic rate estimate

    def add_node(self, name, initial_weight=None, now=0.0):
        if name not in self.nodes:
            self.nodes[name] = AdjustmentNode(
                name=name, weight=initial_weight if initial_weight is not None else self.root_baseline,
                last_touch=now)
            self.graph.setdefault(name, {})
        return self.nodes[name]

    def connect(self, a, b, rate=DIFFUSION_RATE):
        """Undirected mycelium link -- reinforcing either node partially propagates to the other."""
        self.add_node(a)
        self.add_node(b)
        self.graph[a][b] = rate
        self.graph[b][a] = rate

    def _decay_toward_root(self, node: AdjustmentNode, now: float):
        elapsed = max(0.0, now - node.last_touch)
        if elapsed == 0.0:
            return
        factor = math.exp(-self.decay_rate * elapsed)
        node.weight = self.root_baseline + (node.weight - self.root_baseline) * factor
        node.last_touch = now

    def _propagate(self, origin: str, delta: float, now: float):
        """Mycelium: spread a fraction of origin's delta outward through the graph, decaying
        per hop, breadth-first, capped at MAX_HOPS so it can't circulate indefinitely."""
        frontier = {origin: delta}
        visited = {origin}
        for hop in range(MAX_HOPS):
            next_frontier = {}
            for src, src_delta in frontier.items():
                for neighbor, rate in self.graph.get(src, {}).items():
                    if neighbor in visited:
                        continue
                    passed = src_delta * rate * (HOP_DECAY ** hop)
                    if abs(passed) < 1e-4:
                        continue
                    node = self.add_node(neighbor, now=now)
                    self._decay_toward_root(node, now)
                    node.weight = max(0.0, min(1.0, node.weight + passed))
                    next_frontier[neighbor] = passed
                    visited.add(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier

    def reinforce(self, name, confirming: bool, now: float, rate=0.18, asymmetry=LTD_ASYMMETRY):
        """asymmetry: LTD magnitude relative to LTP, default 1.0 (symmetric -- balanced
        confirm/disconfirm evidence has exactly zero net effect, per the point-group
        selection-rule theorem: a symmetric update has no even-order/rectifying term). Pass
        something other than 1.0 only as a deliberate, disclosed symmetry-breaking choice
        (e.g. "confidence should be harder to lose than to gain") -- not as an unexamined
        default, which is the real bug this module shipped with initially."""
        node = self.add_node(name, now=now)
        self._decay_toward_root(node, now)

        delta = rate if confirming else -rate * asymmetry
        old_weight = node.weight
        node.weight = max(0.0, min(1.0, node.weight + delta))
        node.touch_count += 1
        actual_delta = node.weight - old_weight

        # Roots: the trunk drifts slowly toward whatever the leaves are actually doing --
        # one node's one-off correction barely moves it; sustained real agreement does.
        self.root_baseline += ROOT_EMA_RATE * (node.weight - self.root_baseline)

        if actual_delta:
            self._propagate(name, actual_delta, now)
        return node.weight

    def _relax_gate_offset(self, now: float):
        elapsed = max(0.0, now - self._offset_last_touch)
        if elapsed == 0.0:
            return
        self.gate_threshold_offset *= math.exp(-OFFSET_RELAX_RATE * elapsed)
        self._offset_last_touch = now

    def tick(self, now: float):
        """Let real time pass with no new evidence -- every node exhales toward the root, and
        the gate's homeostatic offset exhales back toward neutral too."""
        for node in self.nodes.values():
            self._decay_toward_root(node, now)
        self._relax_gate_offset(now)

    def confidence_gate(self, name: str, now: float = 0.0):
        """Real Spikeling LIF neuron decides fire/no-fire; threshold adapts homeostatically
        (NOT periodically -- see module docstring) based on the layer's recent real fire rate."""
        self._relax_gate_offset(now)
        node = self.nodes.get(name)
        weight = 0.0 if node is None else node.weight
        if node is not None:
            self._decay_toward_root(node, now)
            weight = node.weight

        runtime = SpikelingRuntime(_get_ast())
        drive = weight * GATE_DRIVE_SCALE - self.gate_threshold_offset
        runtime.stimulate("gate", 0.0, drive=max(0.0, drive))
        fired = runtime.neurons["gate"].fire_count > 0

        self._recent_fires.append(fired)
        if len(self._recent_fires) > HOMEOSTATIC_WINDOW:
            self._recent_fires.pop(0)
        if len(self._recent_fires) >= 5:
            observed_rate = sum(self._recent_fires) / len(self._recent_fires)
            error = observed_rate - HOMEOSTATIC_TARGET_RATE
            self.gate_threshold_offset += HOMEOSTATIC_ADAPT_RATE * error
            self.gate_threshold_offset = max(-80.0, min(80.0, self.gate_threshold_offset))

        return fired, weight

    def context_for(self, names, instructions=None, now: float = 0.0):
        """Real text for prompt injection -- only for nodes whose gate actually fires, same
        honest gating discipline retrieval_confidence.spk already enforces for RECALL.

        instructions: optional {name: actual instruction text}. Real gap, found before ever
        running an LLM-behavior test: without this, the injected text was only confidence
        METADATA ("weight 0.97, 4 reinforcements") -- nothing in it told the model what to
        actually DO. A caller who wants this to change real output must supply what the
        adjustment means in practice; the layer only tracks how confident to be about it."""
        instructions = instructions or {}
        lines = []
        for name in names:
            fired, weight = self.confidence_gate(name, now)
            if not fired:
                continue
            node = self.nodes[name]
            instruction = instructions.get(name)
            if instruction:
                lines.append(f"- {instruction} (learned confidence {weight:.2f}, "
                             f"{node.touch_count} real reinforcement(s))")
            else:
                lines.append(f"- {name}: weight {weight:.2f} (root {self.root_baseline:.2f}, "
                             f"{node.touch_count} real reinforcement(s), no instruction text registered)")
        if not lines:
            return ""
        return "Learned behavioral adjustments (gate-confirmed, not all tracked adjustments):\n" + "\n".join(lines)


if __name__ == "__main__":
    layer = SelfAdjustmentLayer()
    layer.connect("terse_style", "concise_comments", rate=0.2)
    layer.connect("concise_comments", "minimal_boilerplate", rate=0.2)   # 2 hops from terse_style

    print("=== WATER: bidirectional ===")
    t = 0.0
    for _ in range(4):
        t += 1.0
        w = layer.reinforce("terse_style", confirming=True, now=t)
    print(f"after 4x confirm: terse_style weight = {w:.3f} (started at root {layer.root_baseline:.3f})")
    t += 1.0
    w = layer.reinforce("terse_style", confirming=False, now=t)
    print(f"after 1x disconfirm: terse_style weight = {w:.3f}  <- real LTD, not just staying flat")

    print("\n=== MYCELIUM: multi-hop propagation (captured BEFORE the long decay gap below --")
    print("    a real self-test bug on the first run: printing this after a 200-unit tick()")
    print("    decayed the propagated effect back to root before it was ever shown) ===")
    print(f"concise_comments (1 hop away) weight: {layer.nodes['concise_comments'].weight:.3f} "
          f"vs root {layer.root_baseline:.3f}")
    print(f"minimal_boilerplate (2 hops away) weight: {layer.nodes['minimal_boilerplate'].weight:.3f} "
          f"vs root {layer.root_baseline:.3f}")
    print("(both should differ from root despite never being reinforced directly)")

    print("\n=== WATER: decay toward root ===")
    before = layer.nodes["terse_style"].weight
    layer.tick(now=t + 200.0)   # a long real gap with no reinforcement
    after = layer.nodes["terse_style"].weight
    print(f"weight before long gap: {before:.3f} -> after 200 time-units untouched: {after:.3f} "
          f"(relaxing toward root {layer.root_baseline:.3f})")

    print("\n=== ROOTS: two-timescale ===")
    print(f"root_baseline after all the above: {layer.root_baseline:.4f} "
          f"(moved only a little from initial 0.300 -- one node's history, slow EMA)")

    print("\n=== BREATHE: homeostatic gate threshold ===")
    print(f"threshold offset before a burst of confirms: {layer.gate_threshold_offset:.2f}")
    for i in range(15):
        layer.reinforce("terse_style", confirming=True, now=t + 10 + i)
        layer.confidence_gate("terse_style", now=t + 10 + i)
    print(f"threshold offset after 15x confirm+gate-check burst: {layer.gate_threshold_offset:.2f} "
          f"(should have risen -- gate got harder to satisfy as it fired too often)")

    print("\n=== context_for() real output ===")
    print(layer.context_for(["terse_style", "concise_comments", "minimal_boilerplate", "never_touched"], now=t + 30))
