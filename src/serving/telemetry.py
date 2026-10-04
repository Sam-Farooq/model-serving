"""OpenTelemetry traces and Prometheus metrics.

Both, because they answer different questions. Prometheus says the p99 moved;
the trace says which of the four spans inside the request moved it.
"""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram

from serving.config import get_settings

PREDICTIONS = Counter(
    "predictions_total", "Predictions served", ["model_version", "path"]
)
LATENCY = Histogram(
    "prediction_latency_seconds", "End to end latency", ["path"],
    # Default buckets top out at 10s and bunch everything useful into one
    # bucket. These are shaped around the 50ms SLO.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
BATCH_SIZE = Histogram(
    "prediction_batch_size", "Instances per request",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
)
SCORE_PSI = Gauge("prediction_score_psi", "PSI of served scores against deploy reference")
MODEL_INFO = Gauge("model_info", "Loaded model version", ["model_name", "version"])


def setup_tracing() -> trace.Tracer:
    cfg = get_settings()
    provider = TracerProvider(resource=Resource.create({"service.name": cfg.service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer(cfg.service_name)
