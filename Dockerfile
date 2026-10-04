FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# CPU wheels. The GPU build is ~2GB larger and this model is a 3-layer MLP.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install .

COPY src/ ./src/
COPY train/ ./train/
RUN pip install -e . --no-deps

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -fsS localhost:8000/health || exit 1
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
