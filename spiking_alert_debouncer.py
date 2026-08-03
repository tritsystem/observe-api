"""Alert debouncer for Concept Shakedown findings, using Spikeling's real
refractory-period mechanic (`refractory=<ms>ms` in the .spk DSL --
core/runtime/runtime.py's stimulate() hard-blocks re-firing within that
window after a neuron's last spike) instead of hand-written cooldown
bookkeeping.

A shakedown run repeated on a schedule (or a jsonl_field/http_api check
re-polled frequently) can flap: the same finding ("X went NO EVIDENCE",
"Y's schedule check failed") gets re-reported every run even though
nothing new happened. Left unfiltered that's alert fatigue -- the same
real problem shows up in a notification channel dozens of times. This
routes each finding through a real refractory-locked neuron per tracked
concept: the FIRST occurrence within a cooldown window fires (a real
alert), repeats within that window are refractory-blocked (Spikeling's
own mechanism, not a manual "have I seen this in the last N minutes"
dict), and once the cooldown elapses a genuinely NEW occurrence fires
again.

Real .spk DSL note: refractory_ms is a single network-wide AST field
(SpikelingRuntime.refractory_ms), not per-neuron -- every tracked concept
in one AlertDebouncer shares the same cooldown window, which matches how
a real alerting policy is usually configured (one debounce window per
channel/severity tier, not a bespoke one per concept).
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

DEFAULT_THRESHOLD = 50.0
DEFAULT_LEAK = 100.0     # larger than threshold -- membrane never carries
                          # over between separate stimulate() calls, so
                          # the ONLY thing that can suppress a firing is
                          # the real refractory lock, not leaky summation
DEFAULT_DRIVE = 100.0    # clears threshold in one call, always


def _safe_neuron_name(concept_name):
    return "a_" + re.sub(r"\W", "_", concept_name)


def _generate_spk_text(neuron_names, threshold, leak, cooldown_ms):
    lines = ["# Generated alert-debouncer network -- one refractory-locked",
             "# LIF neuron per tracked finding. See spiking_alert_debouncer.py.", ""]
    for name in neuron_names:
        lines.append(f"neuron {name} threshold={int(threshold)} leak={int(leak)} type=LIF")
    lines.append("")
    lines.append(f"refractory={int(cooldown_ms)}ms")
    lines.append("")
    return "\n".join(lines)


def _compile_debounce_network(neuron_names, threshold, leak, cooldown_ms):
    spk_text = _generate_spk_text(neuron_names, threshold, leak, cooldown_ms)
    work_dir = tempfile.mkdtemp(prefix="observe_alert_debounce_")
    spk_path = os.path.join(work_dir, "alert_debouncer.spk")
    with open(spk_path, "w", encoding="utf-8") as f:
        f.write(spk_text)
    out_dir = tempfile.mkdtemp(prefix="observe_alert_debounce_codegen_")
    return compile_file(spk_path, output_dir=out_dir)


class AlertDebouncer:
    def __init__(self, concept_names, cooldown_ms=3_600_000.0,
                 threshold=DEFAULT_THRESHOLD, leak=DEFAULT_LEAK):
        self._name_map = {c: _safe_neuron_name(c) for c in concept_names}
        ast = _compile_debounce_network(self._name_map.values(), threshold, leak, cooldown_ms)
        self.runtime = SpikelingRuntime(ast)
        self.drive = DEFAULT_DRIVE
        self._suppressed_count = {c: 0 for c in concept_names}

    def maybe_alert(self, concept_name, timestamp_ms):
        """Call every time a finding re-occurs for `concept_name`. Returns
        True if this is a genuine new alert (first occurrence, or the
        cooldown window has elapsed since the last real alert), False if
        it's a debounced repeat within the window."""
        neuron_name = self._name_map[concept_name]
        neuron = self.runtime.neurons[neuron_name]
        before = neuron.fire_count
        self.runtime.stimulate(neuron_name, timestamp_ms, drive=self.drive)
        fired = neuron.fire_count > before
        if not fired:
            self._suppressed_count[concept_name] += 1
        return fired

    def suppressed_count(self, concept_name):
        return self._suppressed_count[concept_name]
