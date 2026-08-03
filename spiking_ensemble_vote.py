"""Population/ensemble voting network -- a real spiking analogue of the
workflow-orchestration "adversarial verify" pattern (N independent
skeptics, majority survives), built as a converging network of real LIF
neurons instead of N separate LLM calls.

Useful wherever a single point-estimate isn't trustworthy alone: Root
Cause Copilot's confirmed_drivers() currently reads ONE RV number per
candidate off MethodLM's ledger. Feeding it N independent RV estimates
instead (e.g. N bootstrap resamples of the same dataset, each
independently re-run through MethodLM's ADJUST tool) and requiring a
real MAJORITY of them to individually clear the robustness bar is a
stronger claim than one lucky/unlucky single run -- this module is the
counting/voting mechanism, built as a real converging spiking population
(N voter neurons -> one consensus neuron) rather than hand-written
`sum(votes) > len(votes)/2` arithmetic.

Uses a dynamically generated real .spk network (voter count varies per
call, same reason spiking_activity_monitor.py generates its network
per-instance): N voter neurons, each independently driven by one real
vote, converging on a `consensus` neuron via connect weights sized so
that exactly `min_votes` firing voters (and no fewer) cross consensus's
threshold -- see _connection_weight()'s docstring for the exact math.
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

CONSENSUS_THRESHOLD = 100.0
VOTER_THRESHOLD = 50.0
VOTER_LEAK = 5.0
VOTER_DRIVE = 60.0   # clears VOTER_THRESHOLD comfortably


def _connection_weight(min_votes):
    """Each firing voter propagates weight*50.0 into consensus (see
    runtime._fire()'s synapse propagation). weight = 2.0/min_votes would
    put min_votes firing voters at EXACTLY consensus's threshold in
    theory -- but floating-point error (2.0/3*50*3 == 99.999999999999999,
    not 100.0, caught by testing rather than assumed) can land it just
    UNDER threshold, silently failing the exact-majority case. A small
    fixed margin (weight = 2.01/min_votes instead of 2.0/min_votes) fixes
    that: min_votes firing voters now clear threshold by a comfortable
    50*0.01=0.5 units regardless of min_votes, while (min_votes-1) firing
    voters still fall short for any min_votes < ~200 (100*(1-1/min_votes)
    stays under 100 in that range) -- verified for min_votes up to 20."""
    return 2.01 / min_votes


def _generate_spk_text(n_voters, min_votes):
    weight = _connection_weight(min_votes)
    lines = ["# Generated ensemble-vote network -- N independent voter",
             "# neurons converging on one consensus neuron.",
             "# See spiking_ensemble_vote.py.", ""]
    voter_names = [f"voter_{i}" for i in range(n_voters)]
    for name in voter_names:
        lines.append(f"neuron {name} threshold={int(VOTER_THRESHOLD)} leak={int(VOTER_LEAK)} type=LIF")
    lines.append(f"neuron consensus threshold={int(CONSENSUS_THRESHOLD)} leak=5 type=LIF")
    lines.append("")
    for name in voter_names:
        lines.append(f"connect {name} -> consensus weight={weight}")
    lines.append("")
    lines.append("refractory=0ms")
    lines.append("")
    return "\n".join(lines), voter_names


def _compile_vote_network(n_voters, min_votes):
    spk_text, voter_names = _generate_spk_text(n_voters, min_votes)
    work_dir = tempfile.mkdtemp(prefix="observe_ensemble_vote_")
    spk_path = os.path.join(work_dir, "ensemble_vote.spk")
    with open(spk_path, "w", encoding="utf-8") as f:
        f.write(spk_text)
    out_dir = tempfile.mkdtemp(prefix="observe_ensemble_vote_codegen_")
    ast = compile_file(spk_path, output_dir=out_dir)
    return ast, voter_names


def ensemble_vote(votes, min_votes=None):
    """votes: list of bool, one per independent voter (e.g. one bootstrap
    resample's own robustness verdict). min_votes: how many must agree
    for consensus (default majority: len(votes)//2 + 1).

    Returns {"consensus": bool, "n_votes_for": int, "n_total": int,
    "min_votes_required": int}."""
    n_total = len(votes)
    if min_votes is None:
        min_votes = n_total // 2 + 1
    if n_total == 0:
        return {"consensus": False, "n_votes_for": 0, "n_total": 0, "min_votes_required": min_votes}

    ast, voter_names = _compile_vote_network(n_total, min_votes)
    runtime = SpikelingRuntime(ast)

    n_votes_for = 0
    t = 0.0
    for name, vote in zip(voter_names, votes):
        if vote:
            runtime.stimulate(name, t, drive=VOTER_DRIVE)
            n_votes_for += 1
        t += 1.0

    consensus = runtime.neurons["consensus"].fire_count > 0
    return {"consensus": consensus, "n_votes_for": n_votes_for, "n_total": n_total,
            "min_votes_required": min_votes}
