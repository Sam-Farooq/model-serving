"""Population stability index over served scores.

Labels arrive days later, if at all. The score distribution arrives now, and
it moves before accuracy does, so this is the monitor that fires in time to
matter.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from serving.config import get_settings

BINS = np.linspace(0.0, 1.0, 11)

# The one name for the reference distribution. `train/train.py` logs it under
# this path and `model/registry.py` downloads it back under the same one; the
# whole monitor is dead if those two spellings ever part company.
SCORE_REFERENCE_ARTIFACT = "score_reference.npy"


def psi(reference: np.ndarray, current: np.ndarray, eps: float = 1e-6) -> float:
    ref, _ = np.histogram(reference, bins=BINS)
    cur, _ = np.histogram(current, bins=BINS)
    ref_pct = np.clip(ref / max(ref.sum(), 1), eps, None)
    cur_pct = np.clip(cur / max(cur.sum(), 1), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class DriftMonitor:
    """Rolling window against a reference captured at deploy time.

    Convention: PSI under 0.1 is stable, 0.1 to 0.2 warrants a look, over 0.2
    is a real shift. The threshold is configurable because the right number
    depends on how noisy the upstream traffic is.
    """

    def __init__(self, reference: Sequence[float] | np.ndarray | None = None):
        cfg = get_settings()
        self.threshold = cfg.drift_psi_threshold
        # Kept alongside the deque because deque.maxlen is Optional[int], and
        # the fullness check below has to compare against a plain int.
        self.window_size = cfg.drift_window
        self.window: deque[float] = deque(maxlen=self.window_size)
        # Length, not truthiness: `if reference` raises ValueError on an array
        # of more than one element, and the caller passes through whatever the
        # registry read back.
        self.reference = (
            np.asarray(reference, dtype=float)
            if reference is not None and len(reference) > 0
            else None
        )

    def observe(self, scores: list[float]) -> None:
        self.window.extend(scores)

    def current_psi(self) -> float | None:
        if self.reference is None or len(self.window) < self.window_size:
            return None
        return psi(self.reference, np.asarray(self.window))

    def is_drifting(self) -> bool:
        value = self.current_psi()
        return value is not None and value > self.threshold
