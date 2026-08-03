"""Regression tests for the standalone Spikeling integrations that, until
now, only had ad hoc verification scripts (run once during development,
never re-run automatically). Each test reproduces the exact real scenario
that was ad hoc-verified when the integration was first built, so a
future change to the .spk files or the runtime wiring can't silently
regress without a test failing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiking_override_precedence import check_override_precedence
from spiking_coincidence_detector import check_coincidence
from spiking_alert_debouncer import AlertDebouncer
from spiking_ensemble_vote import ensemble_vote
from spiking_schedule_resonator import check_schedule


def test_override_precedence_consistent_scenario():
    obs = [(True, True), (True, True), (False, True), (False, False), (True, True)]
    r = check_override_precedence(obs)
    assert r["consistent"] is True
    assert r["base_suppressed"] == 3
    assert r["override_fired"] == 3


def test_override_precedence_detects_violation_under_strong_base_drive():
    r = check_override_precedence([(True, True)], base_drive=150.0)
    assert r["consistent"] is False
    assert r["violations"] == [0]


def test_coincidence_detector_perfectly_paired():
    events_a = [0.0, 10.0, 20.0, 30.0, 40.0]
    events_b = [0.2, 10.1, 20.3, 30.0, 40.4]
    r = check_coincidence(events_a, events_b, window_s=1.0)
    assert r["both"] == 5
    assert r["confirmed_coincidences"] == 5
    assert r["a_only"] == 0 and r["b_only"] == 0


def test_coincidence_detector_flags_real_divergence():
    events_a = [0.0, 10.0, 20.0, 30.0, 40.0]
    events_b = [0.2, 10.1, 40.4]  # missing matches at t=20, t=30
    r = check_coincidence(events_a, events_b, window_s=1.0)
    assert r["both"] == 3
    assert r["a_only"] == 2
    assert r["divergences"] == []  # network itself correctly wired, no false confirms


def test_alert_debouncer_suppresses_flapping_within_cooldown():
    deb = AlertDebouncer(["finding_x"], cooldown_ms=3600.0)
    results = [deb.maybe_alert("finding_x", t) for t in [0, 500, 1000, 2000, 3000]]
    assert results == [True, False, False, False, False]
    assert deb.suppressed_count("finding_x") == 4


def test_alert_debouncer_revives_after_cooldown_elapses():
    deb = AlertDebouncer(["finding_x"], cooldown_ms=3600.0)
    deb.maybe_alert("finding_x", 0)
    assert deb.maybe_alert("finding_x", 5000.0) is True


def test_ensemble_vote_majority_and_short_of_majority():
    assert ensemble_vote([True, True, True, False, False])["consensus"] is True
    assert ensemble_vote([True, True, False, False, False])["consensus"] is False


def test_ensemble_vote_super_majority():
    assert ensemble_vote([True] * 4 + [False] * 3, min_votes=5)["consensus"] is False
    assert ensemble_vote([True] * 5 + [False] * 2, min_votes=5)["consensus"] is True


def test_schedule_resonator_distinguishes_periodic_from_scattered():
    import random
    random.seed(1)
    period_s = 3600.0
    n_cycles = 30
    on_schedule = [i * period_s + random.uniform(-0.08, 0.08) * period_s for i in range(n_cycles)]
    span = n_cycles * period_s
    scattered = sorted(random.uniform(0, span) for _ in range(n_cycles))

    r1 = check_schedule(on_schedule, period_s)
    r2 = check_schedule(scattered, period_s)
    assert r1["verdict"] == "ON_SCHEDULE"
    assert r2["verdict"] == "NOT_ON_SCHEDULE"


def test_schedule_resonator_reports_insufficient_data_honestly():
    r = check_schedule([0, 3600, 7200], period_s=3600.0)
    assert r["verdict"] == "INSUFFICIENT_DATA"
