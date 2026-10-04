"""MLflow registry access.

The loaded version is pinned at startup and never swapped underneath a running
process. A model that changes mid-request means two instances answering the
same question differently with no way to tell from the response which one did.
Rolling a new version is a deployment, not a hot reload.
"""
from __future__ import annotations

import logging
import threading

import mlflow
import torch

from serving.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()
_state: dict = {}


def load() -> tuple[torch.nn.Module, str, list[str]]:
    cfg = get_settings()
    with _lock:
        if "model" in _state:
            return _state["model"], _state["version"], _state["feature_order"]

        mlflow.set_tracking_uri(cfg.mlflow_uri)
        uri = f"models:/{cfg.model_name}/{cfg.model_stage}"
        model = mlflow.pytorch.load_model(uri)
        model.eval()
        torch.set_num_threads(cfg.torch_threads)

        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(cfg.model_name, stages=[cfg.model_stage])
        version = versions[0].version if versions else "unknown"

        # Feature order travels with the artifact. Reading it back and checking
        # it against the serving schema is what stops a silent reordering from
        # turning "amount" into "hour_of_day" at inference time.
        run = client.get_run(versions[0].run_id) if versions else None
        order = (run.data.params.get("feature_order", "") or "").split(",") if run else []

        _state.update(model=model, version=version, feature_order=[f for f in order if f])
        log.info("loaded %s version=%s", cfg.model_name, version)
        return model, version, _state["feature_order"]


def reset() -> None:
    with _lock:
        _state.clear()
