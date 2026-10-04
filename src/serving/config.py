from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MS_", extra="ignore")

    mlflow_uri: str = "http://localhost:5000"
    model_name: str = "fraud-scorer"
    model_stage: str = "Production"

    redis_url: str = "redis://localhost:6379/0"

    # Anything under this runs inline on the event loop. The Celery round trip
    # is about 4ms; paying it for a single 1.5ms forward pass makes the p50
    # three times worse to protect a tail that does not exist at batch size 1.
    async_batch_threshold: int = 8
    max_batch_size: int = 512

    torch_threads: int = 2
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "model-serving"

    # Scores drift before accuracy does, and drift is visible without labels.
    drift_window: int = 1000
    drift_psi_threshold: float = 0.2


@lru_cache
def get_settings() -> Settings:
    return Settings()
