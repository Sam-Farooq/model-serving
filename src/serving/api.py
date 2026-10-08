"""HTTP surface.

Small batches run inline, large ones go to Celery. The threshold is the only
interesting decision in this file and it is in config, not hard-coded here.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from serving.config import get_settings
from serving.drift import DriftMonitor
from serving.model.predictor import predict_batch, to_labels
from serving.model.registry import load
from serving.schemas import FEATURE_ORDER, Prediction, PredictRequest, PredictResponse
from serving.tasks import predict_task
from serving.telemetry import (
    BATCH_SIZE,
    LATENCY,
    MODEL_INFO,
    PREDICTIONS,
    SCORE_PSI,
    setup_tracing,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
runtime: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    tracer = setup_tracing()
    model, version, feature_order = load()

    # Fail at startup, not on the first request. A reordered feature list is
    # silent at inference: every score is wrong and nothing raises.
    if feature_order and feature_order != FEATURE_ORDER:
        raise RuntimeError(
            f"feature order mismatch: artifact={feature_order} serving={FEATURE_ORDER}"
        )

    runtime.update(model=model, version=version, tracer=tracer, drift=DriftMonitor())
    MODEL_INFO.labels(model_name=cfg.model_name, version=version).set(1)
    log.info("serving %s version=%s", cfg.model_name, version)
    yield


app = FastAPI(title="model-serving", version="2.1.0", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_version": runtime.get("version")}


@app.get("/metrics")
async def metrics() -> Response:
    if (monitor := runtime.get("drift")) and (value := monitor.current_psi()) is not None:
        SCORE_PSI.set(value)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    cfg = get_settings()
    started = time.perf_counter()
    vectors = [f.to_vector() for f in req.instances]
    BATCH_SIZE.observe(len(vectors))

    with runtime["tracer"].start_as_current_span("predict") as span:
        span.set_attribute("batch.size", len(vectors))

        if len(vectors) < cfg.async_batch_threshold:
            path = "inline"
            scores = predict_batch(runtime["model"], vectors)
            version = runtime["version"]
        else:
            path = "celery"
            result = predict_task.delay(vectors)
            # FIXME: blocking get on an async handler. Works because the worker
            # pool is sized well above concurrent batch requests, but it is
            # holding a thread for up to 30s and should be polled instead.
            try:
                payload = result.get(timeout=30)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=504, detail=f"inference timeout: {exc}")
            scores, version = payload["scores"], payload["model_version"]

        span.set_attribute("serving.path", path)

    runtime["drift"].observe(scores)
    PREDICTIONS.labels(model_version=version, path=path).inc(len(scores))
    elapsed = time.perf_counter() - started
    LATENCY.labels(path=path).observe(elapsed)

    return PredictResponse(
        predictions=[Prediction(score=s, label=l)
                     for s, l in zip(scores, to_labels(scores), strict=True)],
        model_version=version,
        served_by=path,
        latency_ms=round(elapsed * 1000, 2),
    )


@app.get("/predict/{task_id}")
async def poll(task_id: str) -> dict:
    result = AsyncResult(task_id, app=predict_task.app)
    return {"state": result.state, "result": result.result if result.ready() else None}
