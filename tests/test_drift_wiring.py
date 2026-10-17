"""The drift monitor fed from the registry, through the real startup path.

Issue #1: DriftMonitor took a reference, api.py constructed it with none, so
current_psi() returned None forever and prediction_score_psi was never set.
These tests run the lifespan against a real MLflow store and press on the
monitor from both sides: shifted traffic has to fire it, matching traffic has
to leave it quiet.
"""
from __future__ import annotations

import numpy as np
from prometheus_client import REGISTRY

from serving import api as api_module
from serving.api import app, lifespan, metrics
from serving.config import get_settings
from serving.schemas import Features, PredictRequest
from tests.conftest import SERVED_SCORE


def _request(n: int) -> PredictRequest:
    instance = Features(
        amount=10.0, hour_of_day=1, merchant_risk=0.5, account_age_days=30,
        txn_count_24h=2, distinct_countries_7d=1, is_cross_border=False,
    )
    return PredictRequest(instances=[instance] * n)


async def _fill_the_window() -> None:
    monitor = api_module.runtime["drift"]
    while len(monitor.window) < monitor.window_size:
        # Four at a time, under the async threshold, so this stays on the
        # inline path and needs no broker.
        await api_module.predict(_request(4))


async def test_startup_feeds_the_monitor_the_reference_from_the_training_run(register_model):
    register_model(np.random.default_rng(0).beta(2, 8, 2000).astype(np.float32))

    async with lifespan(app):
        monitor = api_module.runtime["drift"]
        assert monitor.reference is not None
        assert len(monitor.reference) == 2000
        assert REGISTRY.get_sample_value("drift_reference_size") == 2000.0


async def test_psi_is_exported_once_the_window_fills_with_shifted_scores(register_model):
    # Reference mass sits near zero. Every served score is 0.982, the far bin.
    register_model(np.random.default_rng(0).beta(2, 8, 2000).astype(np.float32))

    async with lifespan(app):
        await _fill_the_window()
        await metrics()

        monitor = api_module.runtime["drift"]
        assert monitor.is_drifting() is True
        exported = REGISTRY.get_sample_value("prediction_score_psi")
        assert exported is not None
        assert exported > get_settings().drift_psi_threshold


async def test_psi_stays_in_the_stable_band_when_the_scores_match_the_reference(register_model):
    # Same distribution the model is serving. The monitor has to stay quiet,
    # or the test above proves only that any traffic at all trips it.
    register_model(np.full(2000, SERVED_SCORE, dtype=np.float32))

    async with lifespan(app):
        await _fill_the_window()
        await metrics()

        monitor = api_module.runtime["drift"]
        assert monitor.current_psi() < 0.1
        assert monitor.is_drifting() is False
        assert REGISTRY.get_sample_value("prediction_score_psi") < 0.1


async def test_a_run_with_no_reference_still_serves_and_reports_the_gap(register_model):
    register_model(None)

    async with lifespan(app):
        await _fill_the_window()

        monitor = api_module.runtime["drift"]
        assert monitor.reference is None
        assert monitor.current_psi() is None
        assert REGISTRY.get_sample_value("drift_reference_size") == 0.0
