import numpy as np

from serving.drift import DriftMonitor, psi


def test_identical_distributions_have_near_zero_psi():
    rng = np.random.default_rng(0)
    sample = rng.beta(2, 8, 5000)
    assert psi(sample, sample) < 1e-6


def test_shifted_distribution_exceeds_the_action_threshold():
    rng = np.random.default_rng(0)
    reference = rng.beta(2, 8, 5000)     # mass near zero
    shifted = rng.beta(8, 2, 5000)       # mass near one
    assert psi(reference, shifted) > 0.2


def test_psi_is_symmetric_enough_to_be_usable():
    rng = np.random.default_rng(1)
    a, b = rng.beta(2, 5, 4000), rng.beta(3, 5, 4000)
    assert abs(psi(a, b) - psi(b, a)) < 0.05


def test_monitor_reports_nothing_until_the_window_is_full():
    monitor = DriftMonitor(reference=list(np.random.default_rng(0).beta(2, 8, 500)))
    monitor.observe([0.1] * 10)
    assert monitor.current_psi() is None
    assert monitor.is_drifting() is False


def test_monitor_fires_once_the_window_fills_with_shifted_scores():
    rng = np.random.default_rng(2)
    monitor = DriftMonitor(reference=list(rng.beta(2, 8, 2000)))
    monitor.observe(list(rng.beta(8, 2, monitor.window.maxlen)))
    assert monitor.is_drifting() is True
