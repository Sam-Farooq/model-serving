# model-serving

PyTorch inference behind FastAPI, with the model pulled from an MLflow
registry, large batches pushed to Celery, and drift watched on the score
distribution rather than on accuracy.

## Small batches run inline, large ones do not

The Celery round trip costs about 4ms. A single forward pass through this
network costs about 1.5ms. Sending one instance through a broker makes the
p50 three times worse to protect a tail that does not exist at batch size 1.

So `/predict` branches on `MS_ASYNC_BATCH_THRESHOLD`, default 8:

- under 8 instances: inline, on the event loop, p50 around 6ms
- 8 or more: queued to a Celery worker, which keeps a 256-instance nightly
  rescoring batch from blocking the authorisation traffic sharing the process

The response says which path served it, in `served_by`. The load test asserts
it, because a threshold that silently stops applying is worse than one set
wrong.

## Three things that fail silently, and what catches each

**A reordered feature list.** Nothing raises. The model accepts the tensor,
returns plausible probabilities, and every score is wrong.

`train/train.py` logs `feature_order` as an MLflow run parameter. The serving
process reads it back at startup and refuses to boot if it disagrees with
`schemas.FEATURE_ORDER`. Startup is the right place: the first request is
already too late, and by then it is in a load balancer rotation.

**A model swapped underneath a running process.** The registry pins its
version at load and never reloads. Two instances answering the same question
differently, with no way to tell from the response which one did, is not worth
the convenience of a hot reload. Rolling a version is a deployment.

**Input distribution moving.** Labels arrive days later, if at all. Scores
arrive now, and they move first.

`drift.py` keeps a 1,000-score rolling window and computes PSI against a
reference captured at deploy time. Under 0.1 is stable, 0.1 to 0.2 is worth a
look, over 0.2 is a real shift. It is exported as `prediction_score_psi` and
the Grafana panel is banded at those thresholds.

## Why average precision, not ROC-AUC

The positive rate is about 1.5%. ROC-AUC is dominated by true negatives at
that base rate and flatters everything: a model can score 0.95 and be useless
at the operating point. `val_average_precision` is the number logged and the
one worth reading.

Training uses `pos_weight` in the loss rather than resampling, which keeps the
validation distribution honest while still giving the minority class gradient.

## Metrics

Latency buckets are shaped around the 50ms SLO. Prometheus defaults top out at
10s and collapse everything interesting into one bucket, which makes a p99
unreadable exactly where it matters.

```
predictions_total{model_version, path}
prediction_latency_seconds{path}          histogram, 5ms to 2.5s
prediction_batch_size                     histogram, 1 to 512
prediction_score_psi                      gauge
model_info{model_name, version}
```

Traces go out over OTLP alongside. Prometheus says the p99 moved; the trace
says which span moved it.

## Running it

```bash
cp .env.example .env
./run.sh up                               # redis, mlflow, otel, prometheus, grafana
./run.sh train                            # trains and registers fraud-scorer
./run.sh serve                            # in another shell
./run.sh worker                           # in a third
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "instances": [{"amount": 240.5, "hour_of_day": 3, "merchant_risk": 0.8,
                 "account_age_days": 12, "txn_count_24h": 9,
                 "distinct_countries_7d": 3, "is_cross_border": true}]}' | jq
```

Grafana is at `localhost:3000`, Prometheus at `localhost:9090`, MLflow at
`localhost:5000`.

## Load test

```bash
./run.sh load
```

Weighted 20:3:1 across single predictions, batches of 16 to 256, and health
checks, which is roughly the observed shape: most traffic is one transaction
scored at authorisation time, with periodic rescoring batches on top.

## Tests

`pytest -q`. The one worth reading is `test_schema_contract.py`, for the
reason in the first section: it is the failure that produces no error
anywhere.

## Known issues

- Nothing actually loads the drift reference. `DriftMonitor` takes one and
  `api.py` constructs it with none, so `current_psi()` returns None forever
  and `prediction_score_psi` is never set. Training writes a reference array
  to MLflow and the serving side never reads it back. This is the largest gap
  in the repo and it is not subtle once you look for it.
- `/predict/{task_id}` exists for polling but nothing calls it. The sync path
  blocks on `result.get(timeout=30)` instead, which ties up a worker thread
  for the duration. Fine at current volume, wrong at 10x.
- Celery `task_acks_late` plus a 50s soft limit means a worker killed mid-batch
  redelivers the whole batch. Idempotent here because inference has no side
  effects, but it would not be if scores were written anywhere.
