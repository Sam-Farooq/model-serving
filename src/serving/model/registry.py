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
import numpy as np
import torch

from serving.config import get_settings
from serving.drift import SCORE_REFERENCE_ARTIFACT

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
        # str(): the registry hands the version back as an int, and it is
        # returned to the client in a field typed as a string. Pydantic
        # refuses it at the last line of the handler, after the forward pass.
        version = str(versions[0].version) if versions else "unknown"

        # Feature order travels with the artifact. Reading it back and checking
        # it against the serving schema is what stops a silent reordering from
        # turning "amount" into "hour_of_day" at inference time.
        run = client.get_run(versions[0].run_id) if versions else None
        order = (run.data.params.get("feature_order", "") or "").split(",") if run else []

        _state.update(
            model=model,
            version=version,
            feature_order=[f for f in order if f],
            # Pinned with the model: the drift reference has to come from the
            # run that produced this version.
            run_id=versions[0].run_id if versions else None,
        )
        log.info("loaded %s version=%s", cfg.model_name, version)
        return model, version, _state["feature_order"]


def load_score_reference() -> list[float] | None:
    """The validation score distribution logged by the run behind the model.

    Pinned the same way the model is, and taken from that version's own run
    rather than the newest one. A rollback otherwise compares the old model
    against the new model's reference and reports drift that is nothing but
    the rollback itself.

    A run with no reference artifact returns None. A missing reference is a
    hole in monitoring, not a wrong answer, so the process serves on and says
    so in the log and in `drift_reference_size`.
    """
    load()
    with _lock:
        if "reference" in _state:
            return _state["reference"]

        run_id = _state.get("run_id")
        reference: list[float] | None = None
        if run_id is None:
            log.warning("no registry version behind the model, so no drift reference")
        else:
            try:
                path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=SCORE_REFERENCE_ARTIFACT
                )
                reference = np.load(path).astype(float).ravel().tolist()
                log.info("drift reference: %d scores from run %s", len(reference), run_id)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "run %s has no %s, so PSI cannot be computed: %s",
                    run_id, SCORE_REFERENCE_ARTIFACT, exc,
                )

        _state["reference"] = reference
        return reference


def reset() -> None:
    with _lock:
        _state.clear()
