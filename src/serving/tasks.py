"""Celery worker for batches large enough to be worth the round trip."""
from __future__ import annotations

from celery import Celery

from serving.config import get_settings
from serving.model.predictor import predict_batch
from serving.model.registry import load

cfg = get_settings()
app = Celery("serving", broker=cfg.redis_url, backend=cfg.redis_url)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=600,
    # One task at a time per worker process. Torch already uses threads, and
    # prefetching batches while one is on the GPU just adds queueing latency
    # to every request behind it.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_time_limit=60,
    task_soft_time_limit=50,
)


@app.task(name="serving.predict", bind=True, max_retries=2)
def predict_task(self, vectors: list[list[float]]) -> dict:
    try:
        model, version, _ = load()
        return {"scores": predict_batch(model, vectors), "model_version": version}
    except Exception as exc:  # noqa: BLE001
        raise self.retry(exc=exc, countdown=2)
