"""Adaptive evidence-weight learner for Concept Shakedown, using
Spikeling's real STDP (Spike-Timing Dependent Plasticity) learning rule
(`learn=STDP rate=...` in the .spk DSL, see core/runtime/runtime.py's
STDPLearner) instead of the hand-tuned DEFAULT_WEIGHTS constant in
spiking_evidence.py.

spiking_evidence.py's weights (disk_glob=30, http_api=70, etc.) are a
disclosed, reasonable JUDGMENT CALL, not a measurement. This module
replaces that judgment call with a real learning signal: feed it
historical cases where (a) a specific evidence kind fired for a concept,
and (b) ground truth (a human review, a later incident, an actual audit)
confirmed whether the concept was really used. The synapse from that
evidence kind to a shared `verdict` neuron gets strengthened via real
STDP whenever the evidence kind's firing lands close in time to a real
positive-verdict firing -- an evidence kind that keeps showing up right
before genuine confirmations earns a higher learned weight than one that
mostly co-occurs with false positives.

HONEST LIMITATION, discovered by reading (not assuming) the real
STDPLearner.update() code: `dt = t - downstream.last_spike_time` is
computed at the presynaptic neuron's OWN fire time relative to the
downstream's MOST RECENT prior spike. Replayed in strict forward-
chronological order (t only increases), dt is never negative, so the
`dt < 0` (LTD / weaken) branch never triggers here -- this rule can only
ever push a weight UP from its initial value, never down. A kind that's
frequently a false positive just stays near its initial weight rather
than being actively suppressed. This is disclosed, not hidden: don't
read "learned weight == initial weight" as "neutral," read it as "never
saw evidence this kind predicts real usage."
"""
import os
import sys
import tempfile

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
_CORE = os.path.join(SPIKELING_ROOT, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from compiler.compiler import compile_file  # noqa: E402
from runtime.runtime import SpikelingRuntime  # noqa: E402

_SPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_weights.spk")
_ast = None

EVIDENCE_KINDS = ["disk_glob", "test_grep", "sqlite", "jsonl_field", "http_api"]
DEFAULT_FIRE_DRIVE = 150.0   # comfortably clears threshold=100 in one stimulate() call
CASE_SPACING = 100.0         # real time-units between historical cases, so one
                             # case's dt bookkeeping never bleeds into the next


def _get_ast():
    global _ast
    if _ast is None:
        out_dir = tempfile.mkdtemp(prefix="observe_spiking_adaptive_")
        _ast = compile_file(_SPK_PATH, output_dir=out_dir)
    return _ast


def learn_weights(historical_cases, fire_drive=DEFAULT_FIRE_DRIVE):
    """historical_cases: list of {"evidence_kind": one of EVIDENCE_KINDS,
    "ground_truth_positive": bool} -- one entry per real past instance
    where that evidence kind fired for some concept, and it's since been
    verified whether the concept was actually used.

    Returns {"weights": {kind: learned_weight}, "n_cases": int,
    "n_positive": {kind: int}, "n_negative": {kind: int}} -- learned
    weights start at 0.3 (the .spk file's initial synapse weight) and
    only move upward, per the module's disclosed STDP-direction
    limitation."""
    runtime = SpikelingRuntime(_get_ast())
    n_positive = {k: 0 for k in EVIDENCE_KINDS}
    n_negative = {k: 0 for k in EVIDENCE_KINDS}

    t = 0.0
    for case in historical_cases:
        kind = case["evidence_kind"]
        if kind not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind: {kind}")
        t += CASE_SPACING
        neuron_name = f"ev_{kind}"

        if case["ground_truth_positive"]:
            n_positive[kind] += 1
            runtime.stimulate("verdict", t, drive=fire_drive)
            runtime.stimulate(neuron_name, t + 1.0, drive=fire_drive)
        else:
            n_negative[kind] += 1
            runtime.stimulate(neuron_name, t + 1.0, drive=fire_drive)

    weights = {}
    for syn in runtime.synapses:
        if syn.dst == "verdict":
            kind = syn.src[len("ev_"):]
            weights[kind] = syn.weight

    return {"weights": weights, "n_cases": len(historical_cases),
            "n_positive": n_positive, "n_negative": n_negative}
