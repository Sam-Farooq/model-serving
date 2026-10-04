#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  up)     docker compose up -d ;;
  down)   docker compose down -v ;;
  train)  python train/train.py --epochs 20 --register ;;
  serve)  uvicorn serving.api:app --reload ;;
  worker) celery -A serving.tasks worker --loglevel=info --concurrency=2 ;;
  test)   pytest -q ;;
  load)   locust -f loadtest/locustfile.py --host http://localhost:8000 ;;
  *)      echo "usage: ./run.sh {up|down|train|serve|worker|test|load}" >&2; exit 1 ;;
esac
