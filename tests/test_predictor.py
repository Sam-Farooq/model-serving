import pytest
import torch
from torch import nn

from serving.config import get_settings
from serving.model.predictor import predict_batch, to_labels


class Identity(nn.Module):
    """Returns the first feature as the logit, so expected output is known."""

    def forward(self, x):
        return x[:, :1]


def test_scores_are_probabilities():
    scores = predict_batch(Identity(), [[0.0], [10.0], [-10.0]])
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] == pytest.approx(0.5, abs=1e-6)


def test_sigmoid_is_applied_at_serve_time_not_in_the_model():
    # A logit of 2.0 must come back as ~0.881, not 2.0. If this fails the head
    # is emitting probabilities and training loss is no longer stable.
    assert predict_batch(Identity(), [[2.0]])[0] == pytest.approx(0.8808, abs=1e-4)


def test_batch_order_is_preserved():
    scores = predict_batch(Identity(), [[-5.0], [5.0], [0.0]])
    assert scores[0] < scores[2] < scores[1]


def test_oversized_batch_is_refused():
    cfg = get_settings()
    with pytest.raises(ValueError, match="exceeds"):
        predict_batch(Identity(), [[0.0]] * (cfg.max_batch_size + 1))


def test_labels_apply_the_threshold_inclusively():
    assert to_labels([0.49, 0.5, 0.51]) == [False, True, True]


def test_no_grad_is_in_effect():
    model = nn.Linear(1, 1)
    predict_batch(model, [[1.0]])
    assert all(p.grad is None for p in model.parameters())
