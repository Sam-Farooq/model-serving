.PHONY: up down train serve worker test lint load
up:     ; docker compose up -d
down:   ; docker compose down -v
train:  ; python train/train.py --epochs 20 --register
serve:  ; uvicorn serving.api:app --reload
worker: ; celery -A serving.tasks worker --loglevel=info --concurrency=2
test:   ; pytest -q
lint:   ; ruff check src tests train loadtest
load:   ; locust -f loadtest/locustfile.py --host http://localhost:8000
