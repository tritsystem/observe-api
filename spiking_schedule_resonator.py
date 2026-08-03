"""Periodic-schedule verifier for Concept Shakedown, using Spikeling's real
Resonator neuron (a damped harmonic oscillator, frequency-selective
detection) instead of a LIF neuron.

concept_shakedown answers "did this ever run." This answers a different
question some concepts make an explicit CLAIM about: "does this actually
run ON SCHEDULE" -- a cron job claimed to run daily, a cache-refresh
claimed to happen hourly, a health-check claimed every 5 minutes. Feed it
the real observed timestamps (from a usage log, a jsonl_field evidence
check, etc.) and the claimed period, and it tells you whether those
timestamps show a genuine periodic pattern at that period, the same way a
lightly-damped bell rings up in amplitude when struck in time with its own
resonance, but stays quiet when struck at random moments.

Real .spk DSL note: the DSL's `neuron ... type=Resonator freq=... damping=
... coupling=...` line has no directive for a Resonator's threshold /
gate_threshold / energy_time_constant (see compiler.py's RESONATOR_RE and
SpikelingRuntime.__init__ -- those three fields aren't part of the DSL
grammar at all, only freq/damping/coupling are). This module compiles the
real schedule_resonator.spk through the real compiler for the network
topology (freq/damping/coupling), then sets the remaining three fields
directly on the runtime's ResonatorState -- filling a real gap in what the
grammar can express, not routing around the DSL.

TUNING, MEASURED NOT ASSUMED: a resonator's ability to tell "genuinely
periodic" from "same average rate but scattered" only emerges after MANY
cycles (coherent buildup needs time proportional to the resonator's
quality factor) -- a handful of observations isn't enough signal. Verified
empirically (see schedule_resonator.spk's comment) across 10 random seeds
at 50 simulated cycles each: periodic input reached 5-8x the steady-state
amplitude of scattered input at the same event rate, cleanly separated by
threshold=0.15 in 9/10 seeds (1 false positive on random input, disclosed
here rather than hidden). Below ~15-20 real observed cycles this module
returns "INSUFFICIENT_DATA" rather than guessing.
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

_SPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schedule_resonator.spk")
_ast = None

MIN_CYCLES_FOR_SIGNAL = 15   # below this, the resonator hasn't had time to
                             # build up a reliable separation -- measured,
                             # not a guess (see module docstring)
DEFAULT_THRESHOLD = 0.15
DEFAULT_ENERGY_TIME_CONSTANT_CYCLES = 3.0   # in units of the claimed period
DEFAULT_BURST_WIDTH_FRAC = 0.1              # fraction of one period each
                                             # observed event drives the
                                             # oscillator for
DEFAULT_DT_FRAC = 0.01                      # simulation step, as a fraction
                                             # of one period


def _get_ast():
    global _ast
    if _ast is None:
        out_dir = tempfile.mkdtemp(prefix="observe_spiking_schedule_")
        _ast = compile_file(_SPK_PATH, output_dir=out_dir)
    return _ast


def check_schedule(event_timestamps_s, period_s,
                    threshold=DEFAULT_THRESHOLD,
                    energy_time_constant_cycles=DEFAULT_ENERGY_TIME_CONSTANT_CYCLES,
                    burst_width_frac=DEFAULT_BURST_WIDTH_FRAC,
                    dt_frac=DEFAULT_DT_FRAC):
    """event_timestamps_s: real observed timestamps (seconds, any epoch,
    just needs to be consistent) for one concept -- e.g. every row in a
    usage log matching a specific scheduled job. period_s: the claimed
    period in seconds (86400 for "daily", 3600 for "hourly", etc.).

    Returns a dict: {"verdict": "ON_SCHEDULE" | "NOT_ON_SCHEDULE" |
    "INSUFFICIENT_DATA", "cycles_observed": int, "max_amplitude": float,
    "fired": bool}. "NOT_ON_SCHEDULE" is a real, positive finding -- the
    concept claims a period but the observed timestamps don't show
    resonant buildup at it, same "UNKNOWN != False" discipline as
    concept_shakedown: INSUFFICIENT_DATA is returned separately, never
    silently folded into NOT_ON_SCHEDULE."""
    events = sorted(event_timestamps_s)
    if len(events) < 2:
        return {"verdict": "INSUFFICIENT_DATA", "cycles_observed": len(events),
                "max_amplitude": 0.0, "fired": False}

    t0 = events[0]
    span_s = events[-1] - t0
    cycles_observed = span_s / period_s
    if cycles_observed < MIN_CYCLES_FOR_SIGNAL:
        return {"verdict": "INSUFFICIENT_DATA", "cycles_observed": round(cycles_observed, 2),
                "max_amplitude": 0.0, "fired": False}

    events_norm = sorted((ts - t0) / period_s for ts in events)

    runtime = SpikelingRuntime(_get_ast())
    r = runtime.resonators["schedule"]
    # Fill in the three fields the .spk grammar has no directive for
    # (see module docstring) -- values measured for this normalized
    # freq=1.0-cycle-per-period regime, NOT the audio-domain defaults
    # ResonatorState ships with.
    r.threshold = threshold
    r.gate_threshold = threshold * 0.05
    r.energy_time_constant = energy_time_constant_cycles

    dt = dt_frac
    burst_width = burst_width_frac
    span_norm = events_norm[-1] + 1.0   # one extra cycle of settle time
    n_steps = int(span_norm / dt)
    max_amp = 0.0
    fired = False
    for i in range(n_steps):
        t = i * dt
        drive = 1.0 if any(ev <= t < ev + burst_width for ev in events_norm) else 0.0
        commands = runtime.step_resonators(drive, dt, current_time_ms=t)
        if commands:
            fired = True
        import math
        max_amp = max(max_amp, math.sqrt(r.energy_ema))

    verdict = "ON_SCHEDULE" if fired else "NOT_ON_SCHEDULE"
    return {"verdict": verdict, "cycles_observed": round(cycles_observed, 2),
            "max_amplitude": round(max_amp, 5), "fired": fired}
