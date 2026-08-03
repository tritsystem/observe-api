"""Coincidence detector for Concept Shakedown, using a real 2-into-1
converging network (two weak excitatory synapses that only jointly clear
threshold -- Spikeling's real connect/weight mechanics, not a hand-coded
boolean AND) to check concept pairs that are implicitly claimed to always
happen TOGETHER: a permission check paired with a feature-flag check, a
cache-write paired with its invalidation, an audit log paired with the
action it's supposed to record.

concept_shakedown checks each concept in isolation. This checks a
relationship BETWEEN two concepts across real timestamped evidence
(e.g. two jsonl_field log queries) -- did they really co-occur every
time, or does the codebase sometimes do one without the other (a real,
concrete divergence from the assumed pairing, worth flagging even though
each half individually looks "confirmed").

Uses the real coincidence_detector.spk network compiled via the real
compiler, run on the real core/runtime/runtime.py. The window mechanic
is real too: `tick()` is Spikeling's own leak-only timestep (no drive),
called once between windows so a lone contribution from one source
decays back toward 0 before the next window's evidence arrives -- this
module doesn't reimplement decay by hand, it uses the runtime's.
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

_SPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coincidence_detector.spk")
_ast = None

DEFAULT_DRIVE = 60.0  # clears each source neuron's own threshold=50 alone,
                       # but the DOWNSTREAM coincidence neuron only sees
                       # 0.6*50=30 per source via the connect weight


def _get_ast():
    global _ast
    if _ast is None:
        out_dir = tempfile.mkdtemp(prefix="observe_spiking_coincidence_")
        _ast = compile_file(_SPK_PATH, output_dir=out_dir)
    return _ast


def check_coincidence(events_a, events_b, window_s, drive=DEFAULT_DRIVE):
    """events_a/events_b: real observed timestamps (seconds) for two
    concepts assumed to always co-occur. window_s: how close two
    timestamps must be to count as "the same real instance" (e.g. two
    log lines from the same request).

    Returns {"n_windows": int, "both": int, "a_only": int, "b_only": int,
    "confirmed_coincidences": int, "divergences": [...]} -- "divergences"
    lists the windows where exactly one of the pair fired evidence but
    the network did NOT register a joint coincidence (either the true
    a_only/b_only cases, which are expected/correct non-fires, are
    filtered out of this list -- only genuine network/evidence
    mismatches would appear, which in a correctly-wired network is
    always empty; a non-empty list here would mean the .spk network
    itself is miswired, not a finding about the target codebase)."""
    all_events = sorted(events_a) + sorted(events_b)
    if not all_events:
        return {"n_windows": 0, "both": 0, "a_only": 0, "b_only": 0,
                "confirmed_coincidences": 0, "divergences": []}

    t_min, t_max = min(all_events), max(all_events)
    n_windows = int((t_max - t_min) / window_s) + 2

    runtime = SpikelingRuntime(_get_ast())
    coincidence = runtime.neurons["coincidence"]

    both = a_only = b_only = confirmed = 0
    divergences = []

    for w in range(n_windows):
        win_start = t_min + w * window_s
        win_end = win_start + window_s
        a_present = any(win_start <= t < win_end for t in events_a)
        b_present = any(win_start <= t < win_end for t in events_b)

        if not a_present and not b_present:
            runtime.tick(w * window_s)
            continue

        fires_before = coincidence.fire_count
        if a_present:
            runtime.stimulate("concept_a", w * window_s, drive=drive)
        if b_present:
            runtime.stimulate("concept_b", w * window_s, drive=drive)
        joint_fired = coincidence.fire_count > fires_before

        if a_present and b_present:
            both += 1
            if joint_fired:
                confirmed += 1
            else:
                divergences.append({"window": w, "reason": "both present but network did not confirm"})
        elif a_present:
            a_only += 1
            if joint_fired:
                divergences.append({"window": w, "reason": "a_only but network confirmed anyway"})
        elif b_present:
            b_only += 1
            if joint_fired:
                divergences.append({"window": w, "reason": "b_only but network confirmed anyway"})

        runtime.tick(w * window_s)  # decay any lone contribution before next window

    return {"n_windows": n_windows, "both": both, "a_only": a_only, "b_only": b_only,
            "confirmed_coincidences": confirmed, "divergences": divergences}
