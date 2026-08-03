"""Continuous concept-activity monitor -- reuses pond-health's real,
already-validated pattern (prediction/spiking_predictor.py:
SpikingAnomalyDetector) applied to code-concept activity instead of pond
sensor readings.

A one-shot Concept Shakedown run answers "was this concept EVER used."
This answers a genuinely different, continuous question: "is it STILL
being used, or did it just go quiet" -- and the inverse, "did a
previously-dead concept just come back to life." Feed it real evidence
events as they happen (a new usage_log row, a new gateway-log line) and
it tracks, per concept, whether a real LIF neuron is still firing
recently or has gone silent.

Same engine as spiking_evidence.py (Spikeling's real
core/runtime/runtime.py), different use of it: there, multiple evidence
KINDS are combined for one concept at one point in time; here, one
neuron per concept is driven repeatedly over TIME, and what matters is
the neuron's own last_spike_time / fire_count, not a single-pass sum.

Uses Spikeling's real DSL too, same as spiking_evidence.py -- but unlike
that module's single fixed evidence_integrator.spk, the neuron set here
is genuinely one-per-tracked-concept, and the concept list is only known
at runtime (a config-driven list that varies per target codebase). A
single static checked-in .spk file can't cover an arbitrary concept set,
so this writes real .spk DSL TEXT to a temp file per monitor instance --
one `neuron <safe_name> threshold=... leak=... type=LIF` line per
concept -- then compiles it through the same real compile_file()/parser
pond-health's static pond_brain.spk goes through. It's generated rather
than hand-authored because the network shape is a runtime parameter, not
because it skips the real DSL.
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

DEFAULT_DRIVE = 110.0     # fires on a single real observation, same as
                          # pond-health's DRIVE_CRITICAL -- one real usage
                          # event is enough to count as "alive right now"
DEFAULT_THRESHOLD = 100.0
DEFAULT_LEAK = 8.0
DEFAULT_STALE_AFTER_S = 3600.0  # how long since the last real observation
                                 # before a concept counts as "gone quiet"


def _safe_neuron_name(concept_name):
    """.spk neuron names must be identifier-like (\\w+); a real concept
    name (an endpoint path like "/v1/private/search", an env var, a flag
    key) often isn't, so map to a safe synthetic name and keep the
    mapping -- callers still refer to concepts by their real name."""
    return "n_" + re.sub(r"\W", "_", concept_name)


def _generate_spk_text(neuron_names, threshold, leak):
    """Real .spk DSL text, one `neuron` line per tracked concept -- same
    syntax as evidence_integrator.spk / pond-health's pond_brain.spk,
    just written per-instance since the concept set is a runtime
    parameter rather than a fixed set of names known ahead of time."""
    lines = ["# Generated activity-monitor network -- one LIF neuron per",
             "# tracked concept. See spiking_activity_monitor.py.", ""]
    for name in neuron_names:
        lines.append(f"neuron {name} threshold={int(threshold)} leak={int(leak)} type=LIF")
    lines.append("")
    lines.append("refractory=0ms")
    lines.append("")
    return "\n".join(lines)


def _compile_activity_network(neuron_names, threshold, leak):
    """Write the generated .spk text to a real temp file and compile it
    through the actual Spikeling compiler, same compile_file() entry
    point used for the static evidence_integrator.spk. compile_file()
    also emits C codegen as a side effect (real Spikeling behavior) --
    thrown away in its own temp dir since only the Python-runtime AST
    is used here, matching pond-health's own tempfile.mkdtemp() use for
    the same reason."""
    spk_text = _generate_spk_text(neuron_names, threshold, leak)
    work_dir = tempfile.mkdtemp(prefix="observe_activity_monitor_")
    spk_path = os.path.join(work_dir, "activity_monitor.spk")
    with open(spk_path, "w", encoding="utf-8") as f:
        f.write(spk_text)
    out_dir = tempfile.mkdtemp(prefix="observe_activity_monitor_codegen_")
    return compile_file(spk_path, output_dir=out_dir)


class ConceptActivityMonitor:
    def __init__(self, concept_names, drive=DEFAULT_DRIVE, leak=DEFAULT_LEAK,
                 threshold=DEFAULT_THRESHOLD, stale_after_s=DEFAULT_STALE_AFTER_S):
        self._name_map = {c: _safe_neuron_name(c) for c in concept_names}
        ast = _compile_activity_network(self._name_map.values(), threshold, leak)
        self.runtime = SpikelingRuntime(ast)
        self.drive = drive
        self.stale_after_s = stale_after_s
        self._ever_fired = {c: False for c in concept_names}
        self._is_stale = {c: False for c in concept_names}
        self._revival_events = []   # (concept, timestamp_s) -- fired AFTER having gone stale
        self._silence_events = []   # (concept, timestamp_s) -- crossed stale_after_s with no new fire

    def observe(self, concept_name, timestamp_s):
        """Call every time real evidence of `concept_name` is seen (a
        matching usage_log row, a matching gateway-log line, etc.).
        Silently ignores concepts not in the tracked set -- this monitor
        answers "is X still active," not "what X exist" (that's Concept
        Shakedown's own extraction step).

        Real bug, caught by testing with a realistic timeline rather than
        assumed correct: the first version counted EVERY concept's
        first-ever observation as a "revival," since _ever_fired started
        False for everything being newly tracked -- a concept nobody has
        ever seen before isn't "reviving," it's activating for the first
        time. A revival specifically means "this had already gone stale,
        and then fired again" -- tracked via _is_stale, only set once
        check_staleness() has actually flagged it, and cleared here the
        moment a genuine revival fire happens."""
        neuron_name = self._name_map.get(concept_name)
        if neuron_name is None:
            return
        neuron = self.runtime.neurons[neuron_name]
        was_stale = self._is_stale[concept_name]
        self.runtime.stimulate(neuron_name, timestamp_s * 1000.0, drive=self.drive)
        if neuron.fire_count > 0:
            if was_stale:
                self._revival_events.append((concept_name, timestamp_s))
                self._is_stale[concept_name] = False
            self._ever_fired[concept_name] = True

    def check_staleness(self, now_s):
        """Call periodically (not per-event) to find concepts that fired
        before but have gone quiet for longer than stale_after_s. Returns
        the list of (concept, seconds_since_last_fire) newly gone stale
        since the last call -- an already-flagged-stale concept won't be
        re-reported unless it fires again (clearing the flag via
        observe()) and then goes stale again."""
        newly_stale = []
        for concept, neuron_name in self._name_map.items():
            if not self._ever_fired[concept] or self._is_stale[concept]:
                continue
            neuron = self.runtime.neurons[neuron_name]
            elapsed = now_s - (neuron.last_spike_time / 1000.0)
            if elapsed > self.stale_after_s:
                newly_stale.append((concept, elapsed))
                self._is_stale[concept] = True
                self._silence_events.append((concept, now_s))
        return newly_stale

    def status(self, concept_name, now_s):
        """One concept's current read: 'never-seen' / 'active' / 'stale'."""
        neuron_name = self._name_map.get(concept_name)
        if neuron_name is None or not self._ever_fired[concept_name]:
            return "never-seen"
        neuron = self.runtime.neurons[neuron_name]
        elapsed = now_s - (neuron.last_spike_time / 1000.0)
        return "stale" if elapsed > self.stale_after_s else "active"

    @property
    def revival_events(self):
        return list(self._revival_events)
