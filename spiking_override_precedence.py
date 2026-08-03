"""Precedence/override checker for Concept Shakedown, using a real
inhibitory synapse (a negative-weight `connect` edge -- Spikeling's real
DSL supports this, see core/runtime/runtime.py's veto/hyperpolarization
mechanics) instead of hand-written if/else precedence logic.

Some concept pairs make an explicit claim of PRECEDENCE, not just
existence: "the OBSERVE_MODEL_PATH env var overrides the default model
path," "a feature-flag kill-switch suppresses the flagged code path,"
"an explicit test skip overrides the 'should run' expectation." Concept
Shakedown's own evidence checks can confirm each half exists, but not
whether the override half ACTUALLY takes precedence over the base half
in practice. This models the claimed precedence rule as a real inhibitory
synapse and drives it with real per-observation evidence pairs (e.g. one
row per request/log entry where both the override's and the base's real
evidence are recorded) -- if override evidence is present and base still
independently fires anyway, that's a genuine, disclosed inconsistency
between the claimed precedence and what really happens, not a guess.

Uses the real .spk file override_precedence.spk (two LIF neurons, one
inhibitory connection) compiled via the real compiler, run on the real
core/runtime/runtime.py.
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

_SPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "override_precedence.spk")
_ast = None

DEFAULT_OVERRIDE_DRIVE = 60.0   # clears the 50-threshold on its own -- a
                                 # real, confirmed override observation
                                 # should reliably fire and veto
DEFAULT_BASE_DRIVE = 60.0       # same magnitude as override's drive --
                                 # deliberately NOT weaker, so a "base
                                 # fires anyway" result reflects the
                                 # synapse's real veto strength, not a
                                 # rigged comparison


def _get_ast():
    global _ast
    if _ast is None:
        out_dir = tempfile.mkdtemp(prefix="observe_spiking_override_")
        _ast = compile_file(_SPK_PATH, output_dir=out_dir)
    return _ast


def check_override_precedence(observations, override_drive=DEFAULT_OVERRIDE_DRIVE,
                               base_drive=DEFAULT_BASE_DRIVE):
    """observations: list of (override_present, base_present) bool pairs,
    one per real observed instance. A fresh network is built per
    observation (each instance is independent -- a shared runtime would
    let one observation's membrane state leak into the next).

    Returns {"consistent": bool, "n_observations": int,
    "override_fired": int, "base_suppressed": int,
    "base_fired_despite_override": int, "violations": [indices]} --
    "violations" lists which real observations contradict the claimed
    precedence (override present, base fired anyway)."""
    override_fired = 0
    base_suppressed = 0
    base_fired_despite_override = 0
    violations = []

    for i, (override_present, base_present) in enumerate(observations):
        runtime = SpikelingRuntime(_get_ast())
        did_override_fire = False
        if override_present:
            runtime.stimulate("override_neuron", 0.0, drive=override_drive)
            did_override_fire = runtime.neurons["override_neuron"].fire_count > 0
            if did_override_fire:
                override_fired += 1

        did_base_fire = False
        if base_present:
            runtime.stimulate("base_neuron", 1.0, drive=base_drive)
            did_base_fire = runtime.neurons["base_neuron"].fire_count > 0

        if did_override_fire and base_present:
            if did_base_fire:
                base_fired_despite_override += 1
                violations.append(i)
            else:
                base_suppressed += 1

    return {
        "consistent": len(violations) == 0,
        "n_observations": len(observations),
        "override_fired": override_fired,
        "base_suppressed": base_suppressed,
        "base_fired_despite_override": base_fired_despite_override,
        "violations": violations,
    }
