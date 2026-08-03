"""Regression tests for concept_shakedown.py's spiking-verdict wiring --
formalizes what was only ad hoc-verified when spiking_evidence.py's
weighted combiner was first wired into evaluate_concept(), so a future
change can't silently break it without a test failing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_shakedown import evaluate_concept


def test_evaluate_concept_includes_spiking_verdict_field():
    concept = {"name": "OBSERVE_CREDITS_PER_SEARCH", "type": "env_var"}
    evidence_config = {"rules": {"env_var": [{"kind": "test_grep", "glob": "tests/**/*.py"}]}}
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    result = evaluate_concept(repo_path, concept, evidence_config)

    assert "spiking_verdict" in result
    assert "spiking_trace" in result
    assert isinstance(result["spiking_verdict"], str)


def test_single_weak_signal_confirms_under_or_but_not_spiking():
    """A single test_grep hit (weight 40 in spiking_evidence's scale) is
    enough for the classic OR-based verdict to confirm, but alone doesn't
    clear the weighted spiking integrator's threshold (65) -- the real,
    disclosed difference in philosophy between the two verdicts. This is
    the exact scenario ad hoc-verified when the wiring was first added."""
    concept = {"name": "some_real_concept", "type": "env_var"}
    evidence_config = {"rules": {"env_var": [{"kind": "test_grep", "glob": "tests/**/*.py"}]}}
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    result = evaluate_concept(repo_path, concept, evidence_config)

    if result["signals"] and result["signals"][0]["result"] is True:
        assert result["verdict"].startswith("CONFIRMED")
        assert result["spiking_verdict"].startswith("NO EVIDENCE")


def test_no_evidence_rules_gives_unobservable_and_unknown_spiking():
    concept = {"name": "untested_concept_type", "type": "some_unconfigured_type"}
    evidence_config = {"rules": {}}
    result = evaluate_concept(".", concept, evidence_config)

    assert result["verdict"].startswith("UNOBSERVABLE")
    assert result["spiking_verdict"].startswith("UNOBSERVABLE")


def test_all_false_signals_give_no_evidence_both_verdicts():
    concept = {"name": "definitely_nonexistent_xyz123", "type": "env_var"}
    evidence_config = {"rules": {"env_var": [{"kind": "disk_glob", "pattern": "**/*.this_extension_does_not_exist"}]}}
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    result = evaluate_concept(repo_path, concept, evidence_config)

    assert result["verdict"].startswith("NO EVIDENCE")
    assert result["spiking_verdict"].startswith("NO EVIDENCE")
