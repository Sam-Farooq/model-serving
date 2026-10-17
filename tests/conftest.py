"""A throwaway MLflow registry for the tests that have to cross that boundary.

The store is real: sqlite backend, real runs, real artifacts, real registered
versions in real stages. Only `mlflow.pytorch.load_model` is stubbed, because
serialising a torch model is the one part of the round trip none of these
tests depend on. Everything else is MLflow doing what it does in production,
so a change in how it stores or resolves a version shows up here.
"""
from __future__ import annotations

import mlflow
import mlflow.pytorch
import numpy as np
import pytest
import torch
from opentelemetry import trace
from torch import nn

from serving import api as api_module
from serving.config import get_settings
from serving.drift import SCORE_REFERENCE_ARTIFACT
from serving.model import registry
from serving.schemas import FEATURE_ORDER

SERVED_LOGIT = 4.0
SERVED_SCORE = float(torch.sigmoid(torch.tensor(SERVED_LOGIT)))


class ConstantLogit(nn.Module):
    """Emits a fixed logit, so the served score distribution is known."""

    def __init__(self, logit: float):
        super().__init__()
        self.logit = logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.logit)


@pytest.fixture
def register_model(tmp_path, monkeypatch):
    """Returns a function that adds a run, its reference and a model version.

    `mlflow.pytorch` is imported at module scope on purpose: it is a lazy
    attribute of `mlflow`, and the first access replaces it, which would throw
    away a patch applied to the loader instead of the module.
    """
    monkeypatch.setenv("MS_MLFLOW_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MS_DRIFT_WINDOW", "40")
    get_settings.cache_clear()
    registry.reset()
    monkeypatch.setattr(mlflow.pytorch, "load_model", lambda uri: ConstantLogit(SERVED_LOGIT))
    monkeypatch.setattr(api_module, "setup_tracing", lambda: trace.get_tracer("test"))

    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    experiment = mlflow.create_experiment(
        "drift-wiring", artifact_location=str(tmp_path / "artifacts")
    )
    client = mlflow.MlflowClient()
    client.create_registered_model(cfg.model_name)

    def setup(reference: np.ndarray | None, stage: str | None = cfg.model_stage) -> str:
        with mlflow.start_run(experiment_id=experiment) as run:
            run_id = run.info.run_id
            mlflow.log_param("feature_order", ",".join(FEATURE_ORDER))
            if reference is not None:
                directory = tmp_path / run_id
                directory.mkdir()
                # The basename is what MLflow stores it under, so it has to be
                # the name the serving side asks for.
                np.save(directory / SCORE_REFERENCE_ARTIFACT, reference)
                mlflow.log_artifact(str(directory / SCORE_REFERENCE_ARTIFACT))
            source = mlflow.get_artifact_uri()

        version = client.create_model_version(cfg.model_name, source=source, run_id=run_id)
        if stage is not None:
            client.transition_model_version_stage(cfg.model_name, version.version, stage)
        return run_id

    yield setup

    registry.reset()
    get_settings.cache_clear()
