"""Inference. One batched forward pass under no_grad."""
from __future__ import annotations

import numpy as np
import torch

from serving.config import get_settings

THRESHOLD = 0.5


def predict_batch(model: torch.nn.Module, vectors: list[list[float]]) -> list[float]:
    cfg = get_settings()
    if len(vectors) > cfg.max_batch_size:
        raise ValueError(f"batch of {len(vectors)} exceeds {cfg.max_batch_size}")

    tensor = torch.tensor(np.asarray(vectors, dtype=np.float32))
    with torch.no_grad():
        logits = model(tensor)
    # The head emits logits, not probabilities. Applying sigmoid here rather
    # than in the model keeps the training loss numerically stable
    # (BCEWithLogitsLoss) and the served output interpretable.
    return torch.sigmoid(logits).squeeze(-1).tolist()


def to_labels(scores: list[float], threshold: float = THRESHOLD) -> list[bool]:
    return [s >= threshold for s in scores]
