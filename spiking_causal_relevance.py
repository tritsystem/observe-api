"""Spiking causal+relevance fusion for the A2A "causal-driver-check" skill.

Combines OBSERVE's real semantic search relevance with MethodLM's real
Cinelli-Hazlett robustness value into one verdict, via a real compiled
Spikeling LIF neuron -- same proven pattern as spiking_evidence.py
(Concept Shakedown) and llama-demo's spiking_fusion_verdict.py, not a
new, unproven approach.

Uses Spikeling's real DSL end to end: the network lives in
causal_relevance_fusion.spk (real .spk text), compiled via the actual
compiler.compile_file()/SpikelingParser and run on the real
core/runtime/runtime.py.
"""
import os
import sys
import tempfile

# Real cross-platform fix: the existing spiking_evidence.py in this same
# directory hardcodes the Windows-only path below and was confirmed (by
# actually trying to import it under WSL2, not assumed) to have NEVER
# successfully loaded in the real WSL2 production environment --
# ModuleNotFoundError: No module named 'compiler', because C:\Users\...
# doesn't resolve as a path on Linux at all. Try the Windows path first
# (works when this runs under Windows Python), fall back to the real
# WSL2 mount point for the same OneDrive folder.
_CANDIDATE_ROOTS = [
    r"C:\Users\gbran\OneDrive\Documents\Spikeling",
    "/mnt/c/Users/gbran/OneDrive/Documents/Spikeling",
]
SPIKELING_ROOT = next((p for p in _CANDIDATE_ROOTS if os.path.isdir(p)), _CANDIDATE_ROOTS[0])
_CORE = os.path.join(SPIKELING_ROOT, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

# Both candidate roots are still this one operator's personal machine (Windows
# path or its own WSL2 mount of the SAME OneDrive folder) -- neither exists on
# CI or anyone else's machine, which is exactly what broke `import server` /
# `import a2a_adapter` everywhere except here. This is a real dependency on a
# real sibling project (Spikeling), not currently installable as a normal
# package dependency (not yet on PyPI). Rather than crash the whole API
# (and every test that imports it) when Spikeling isn't on this machine,
# defer the failure to call time, where fuse_causal_and_relevance's only
# caller (a2a_adapter.py's causal-driver-check endpoint) already wraps the
# call in a try/except that refunds the credit and returns a clean error --
# this makes that the actual behavior instead of an import-time crash that
# takes the whole server down with it.
try:
    from compiler.compiler import compile_file  # noqa: E402
    from runtime.runtime import SpikelingRuntime  # noqa: E402
    _SPIKELING_IMPORT_ERROR = None
except ImportError as _e:
    compile_file = None
    SpikelingRuntime = None
    _SPIKELING_IMPORT_ERROR = _e

_SPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "causal_relevance_fusion.spk")
_ast = None

SEARCH_RELEVANCE_SCALE = 100.0
CAUSAL_RV_SCALE = 150.0


def _get_ast():
    global _ast
    if _SPIKELING_IMPORT_ERROR is not None:
        raise RuntimeError(
            f"Spikeling not available on this machine ({_SPIKELING_IMPORT_ERROR}); "
            "causal-driver-check is unavailable here."
        )
    if _ast is None:
        out_dir = tempfile.mkdtemp(prefix="observe_causal_fusion_")
        _ast = compile_file(_SPK_PATH, output_dir=out_dir)
    return _ast


def fuse_causal_and_relevance(search_relevance_score, causal_rv):
    """search_relevance_score: OBSERVE's real top search score (0-1ish,
    real observed range ~0.4-0.6 for genuinely relevant hits).
    causal_rv: MethodLM's real robustness value for the candidate driver
    (0-1, RV<0.10 = fragile per MethodLM's own documented threshold).

    Returns {"fired": bool, "search_drive": float, "causal_drive": float,
    "membrane_potential": float} -- a real, inspectable trace, not just
    a bare boolean."""
    runtime = SpikelingRuntime(_get_ast())
    neuron = runtime.neurons["causal_fusion"]

    search_drive = max(0.0, search_relevance_score) * SEARCH_RELEVANCE_SCALE
    causal_drive = max(0.0, causal_rv) * CAUSAL_RV_SCALE

    runtime.stimulate("causal_fusion", 0.0, drive=search_drive)
    runtime.stimulate("causal_fusion", 1.0, drive=causal_drive)

    return {
        "fired": neuron.fire_count > 0,
        "search_drive": round(search_drive, 2),
        "causal_drive": round(causal_drive, 2),
        "membrane_potential": neuron.membrane_potential,
    }
