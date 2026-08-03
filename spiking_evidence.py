"""Spiking multi-signal evidence integrator for Concept Shakedown.

concept_shakedown.py's evaluate_concept() combines evidence signals with
crude boolean OR: any single True signal confirms the concept, regardless
of how many other checks came back False or unknown. A real leaky-
integrate-and-fire (LIF) neuron -- Spikeling's actual runtime, not a
from-scratch reimplementation -- gives a principled alternative: each
evidence signal becomes a weighted stimulus, the neuron's membrane
potential integrates them (with leak between stimuli), and it only
"fires" (confirms) once the combined, weighted evidence clears a real
threshold. This lets several weak, individually-inconclusive signals add
up to a real confirmation, or a single very strong signal (a real HTTP
API hit) confirm on its own -- a genuinely different answer than OR logic
for the case that actually matters: 2-3 weak signals that individually
wouldn't clear the bar.

Uses Spikeling's real DSL end to end -- the network lives in
evidence_integrator.spk (real .spk text, not a hand-built AST), compiled
via the actual compiler.compile_file()/SpikelingParser and run on the real
core/runtime/runtime.py, same as pond-health's pond_brain.spk. threshold=65
and leak=5 are the ones baked into that file; DEFAULT_THRESHOLD/DEFAULT_LEAK
below only document those same numbers for callers inspecting this module,
they aren't fed back into the DSL.
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

_SPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence_integrator.spk")
_ast = None


def _get_ast():
    """Compile evidence_integrator.spk once and cache the AST. compile_file()
    also emits C codegen as a side effect (real Spikeling behavior, same as
    pond-health) -- that output is thrown away in a temp dir since only the
    Python-runtime AST is needed here."""
    global _ast
    if _ast is None:
        out_dir = tempfile.mkdtemp(prefix="observe_spiking_")
        _ast = compile_file(_SPK_PATH, output_dir=out_dir)
    return _ast

# Weight per evidence-signal kind -- reflects how directly each one
# implies real execution, not an arbitrary scale. An HTTP hit against a
# real service (a flag actually evaluated) is stronger, more direct
# evidence than a name appearing in a grep of test-file text, which is
# itself stronger than a bare disk-glob existence check (a file existing
# doesn't mean the specific concept inside it ran). These are a real,
# disclosed judgment call -- tune per target system if a different
# ordering of confidence makes more sense there.
DEFAULT_WEIGHTS = {
    "disk_glob":   30.0,
    "test_grep":   40.0,
    "file_grep":   40.0,
    "sqlite":      50.0,
    "jsonl_field": 55.0,
    "http_api":    70.0,
}

DEFAULT_THRESHOLD = 65.0  # clears on one strong signal (http_api=70) OR two
                          # moderate ones (e.g. test_grep 40 + disk_glob 30 = 70
                          # minus a small leak between stimuli)
DEFAULT_LEAK = 5.0


def spiking_combine(signals, weights=None, threshold=DEFAULT_THRESHOLD, leak=DEFAULT_LEAK):
    """signals: list of {"kind": str, "result": True/False/None} (the same
    shape concept_shakedown.evaluate_concept() already builds). Returns
    (fired: bool, trace: list of membrane-potential readings after each
    stimulus) -- the trace is real, inspectable data for understanding WHY
    a borderline case did or didn't fire, not just a final boolean.

    Signals with result=None (couldn't check -- missing DB, unreachable
    service) contribute NO drive, same as concept_shakedown's own "None
    means unknown, never treated as a no" rule -- an unreachable evidence
    source must not accidentally suppress or inflate the reading.

    threshold/leak here are informational only -- the real network comes
    from compiling evidence_integrator.spk, so the effective values are
    whatever that file declares (65/5 by default). Pass a different
    threshold/leak to point at a different .spk file with those values
    baked in, rather than expecting them to override the compiled network."""
    weights = weights or DEFAULT_WEIGHTS

    runtime = SpikelingRuntime(_get_ast())

    trace = []
    t = 0.0
    for sig in signals:
        if sig.get("result") is not True:
            continue  # False or None (unknown) -- no drive either way
        drive = weights.get(sig["kind"], 40.0)
        runtime.stimulate("evidence", t, drive=drive)
        trace.append({"kind": sig["kind"], "drive": drive,
                       "membrane_potential": runtime.neurons["evidence"].membrane_potential,
                       "fired": runtime.neurons["evidence"].fire_count > 0})
        t += 1.0

    fired = runtime.neurons["evidence"].fire_count > 0
    return fired, trace


def spiking_verdict(signals, weights=None, threshold=DEFAULT_THRESHOLD, leak=DEFAULT_LEAK):
    """Drop-in alternative to concept_shakedown's own OR-based verdict
    string, same three-way shape (CONFIRMED / NO EVIDENCE / UNKNOWN) so it
    can be swapped in without changing anything downstream that reads the
    verdict string."""
    any_checked = any(s.get("result") is not None for s in signals)
    if not signals:
        return "UNOBSERVABLE (no evidence rule configured for this concept type)", []
    if not any_checked:
        return "UNKNOWN (evidence source(s) unreachable/missing -- not the same as no evidence)", []
    fired, trace = spiking_combine(signals, weights, threshold, leak)
    verdict = ("CONFIRMED (spiking integrator fired: weighted evidence cleared threshold)"
               if fired else
               "NO EVIDENCE (spiking integrator: weighted evidence did not clear threshold)")
    return verdict, trace
