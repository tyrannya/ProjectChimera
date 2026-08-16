"""Baselines the neural model has to beat.

A Transformer that does not outperform "always predict the majority class" on
out-of-sample data has not learned anything about the market — it has learned
the class prior. Both baselines here emit probability matrices in the same
shape and class order as the model, so ``nn/evaluate.py`` scores all three with
identical code and the comparison is apples to apples.
"""

from __future__ import annotations

import numpy as np

from chimera.contracts import CLASS_ORDER, HOLD_IDX, LONG_IDX, SHORT_IDX

N_CLASSES = len(CLASS_ORDER)


class MajorityClassBaseline:
    """Predict the training set's most common class, always.

    The floor. Its accuracy is the class prior, which is exactly why raw
    accuracy is a useless headline metric for this problem: with cost-aware
    labels, HOLD usually dominates, so a model can score 70% accuracy while
    never making a single tradeable prediction.
    """

    name = "majority_class"

    def __init__(self) -> None:
        self.prior = np.full(N_CLASSES, 1.0 / N_CLASSES)

    def fit(self, y: np.ndarray) -> "MajorityClassBaseline":
        counts = np.bincount(y.astype(np.int64), minlength=N_CLASSES).astype(np.float64)
        total = counts.sum()
        self.prior = counts / total if total else np.full(N_CLASSES, 1.0 / N_CLASSES)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.prior, (len(X), 1))


class MomentumBaseline:
    """A rule-based baseline: go with the trend at the end of the window.

    Uses the ``ema_cross`` feature (fast EMA over slow EMA, minus one) of the
    most recent step. Positive means the fast EMA is above the slow one.
    Probabilities are a fixed confidence rather than anything calibrated — this
    is a rule, not a model, and pretending otherwise would flatter it in the
    calibration metrics.
    """

    name = "momentum_rule"

    def __init__(self, feature_index: int, deadband: float = 0.001, confidence: float = 0.6):
        if not 1.0 / N_CLASSES < confidence < 1.0:
            raise ValueError("confidence must be between chance and certainty")
        self.feature_index = feature_index
        self.deadband = deadband
        self.confidence = confidence

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"expected (n, seq_len, features), got {X.shape}")
        signal = X[:, -1, self.feature_index]
        rest = (1.0 - self.confidence) / (N_CLASSES - 1)
        proba = np.full((len(X), N_CLASSES), rest)
        long_mask = signal > self.deadband
        short_mask = signal < -self.deadband
        hold_mask = ~(long_mask | short_mask)
        proba[long_mask, LONG_IDX] = self.confidence
        proba[short_mask, SHORT_IDX] = self.confidence
        proba[hold_mask, HOLD_IDX] = self.confidence
        return proba
