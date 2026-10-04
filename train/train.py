#!/usr/bin/env python3
"""Train the scorer and register it.

    python train/train.py --epochs 20 --register

Logs the feature order as a run parameter. The serving process reads it back
and refuses to start if it disagrees with its own schema, which is the only
cheap defence against a reordering that would otherwise be silent.
"""
from __future__ import annotations

import argparse

import mlflow
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from serving.config import get_settings
from serving.schemas import FEATURE_ORDER


class Scorer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--register", action="store_true")
    args = ap.parse_args()

    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.model_name)

    # 1.5% positives, which is roughly the real base rate. Training on a
    # balanced set and evaluating on ROC-AUC produces a model that looks
    # excellent and is useless at the operating point.
    X, y = make_classification(
        n_samples=40_000, n_features=len(FEATURE_ORDER), n_informative=5,
        weights=[0.985], flip_y=0.01, random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Scorer(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # pos_weight rather than resampling: it keeps the validation distribution
    # honest while still giving the minority class gradient.
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "epochs": args.epochs,
            "optimizer": "adamw",
            "lr": 1e-3,
            "pos_weight": float(pos_weight),
            "feature_order": ",".join(FEATURE_ORDER),
        })

        for epoch in range(args.epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(xt), yt)
            loss.backward()
            opt.step()
            mlflow.log_metric("train_loss", float(loss), step=epoch)

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(
                model(torch.tensor(X_val, dtype=torch.float32))
            ).squeeze(-1).numpy()

        # Average precision, not ROC-AUC. At a 1.5% base rate ROC-AUC flatters
        # everything, because the true negatives dominate it.
        metrics = {
            "val_average_precision": float(average_precision_score(y_val, scores)),
            "val_roc_auc": float(roc_auc_score(y_val, scores)),
            "val_positive_rate": float(np.mean(y_val)),
        }
        mlflow.log_metrics(metrics)
        print(metrics)

        # The reference distribution the drift monitor compares against.
        np.save("/tmp/score_reference.npy", scores)
        mlflow.log_artifact("/tmp/score_reference.npy")

        if args.register:
            mlflow.pytorch.log_model(model, "model", registered_model_name=cfg.model_name)
            print(f"registered from run {run.info.run_id}")


if __name__ == "__main__":
    main()
