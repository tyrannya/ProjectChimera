"""Baselines the neural model has to beat.

A Transformer that does not outperform "always predict the majority class" on
out-of-sample data has not learned anything about the market — it has learned
the class prior. Both baselines here emit probability matrices in the same
shape and class order as the model, so ``nn/evaluate.py`` scores all three with
identical code and the comparison is apples to apples.

**Those matrices are one-hot, and that is the point.** A baseline is a decision
rule, not a probabilistic model: the vector *is* the action, encoded so it can
flow through the same :func:`nn.evaluate.signals_from_proba` the model uses.
Encoding it any other way makes the baseline's behaviour depend on the decision
threshold, which is fitted for the *model* — so the "floor" would move whenever
the model's threshold moved, and a rerun that differed only by seed would report
a different baseline. That is a real bug this module used to have: the majority
baseline emitted the empirical class prior, so a threshold above the prior's
largest entry silently suppressed it to HOLD.

One consequence worth stating: a one-hot rule claims certainty it does not have,
so its calibration error is poor by construction. That is honest rather than
awkward — these rules have no probability estimate to calibrate — but it means
baseline calibration numbers are not a model-quality claim about them.
"""

from __future__ import annotations

import numpy as np

from chimera.contracts import CLASS_ORDER, HOLD_IDX, LONG_IDX, SHORT_IDX

N_CLASSES = len(CLASS_ORDER)

#: The decision threshold baselines are scored at.
#:
#: Declared here, not fitted anywhere and never taken from the model. Because
#: both baselines emit one-hot actions, *any* value in ``[0, 1]`` produces the
#: same decisions — so this constant cannot influence a baseline's behaviour; it
#: exists so the number recorded in a baseline's report belongs to the baseline
#: rather than being an echo of whatever threshold the model happened to select.
BASELINE_DECISION_THRESHOLD = 0.5


class MajorityClassBaseline:
    """Predict the training set's most common class, always.

    The floor. Its accuracy is the class prior, which is exactly why raw
    accuracy is a useless headline metric for this problem: with cost-aware
    labels, HOLD usually dominates, so a model can score 70% accuracy while
    never making a single tradeable prediction.

    ``prior`` is kept for inspection — it is what makes the majority class the
    majority class — but the emitted action is a hard one-hot on that class, so
    "always" means always, at any decision threshold.
    """

    name = "majority_class"

    def __init__(self) -> None:
        self.prior = np.full(N_CLASSES, 1.0 / N_CLASSES)
        self.majority = HOLD_IDX

    def fit(self, y: np.ndarray) -> "MajorityClassBaseline":
        counts = np.bincount(y.astype(np.int64), minlength=N_CLASSES).astype(np.float64)
        total = counts.sum()
        self.prior = counts / total if total else np.full(N_CLASSES, 1.0 / N_CLASSES)
        # argmax breaks ties towards the lowest class index: arbitrary, but fixed,
        # so two runs on the same training rows cannot disagree.
        self.majority = int(np.argmax(counts)) if total else HOLD_IDX
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(X), N_CLASSES), dtype=np.float64)
        proba[:, self.majority] = 1.0
        return proba


class MomentumBaseline:
    """A rule-based baseline: go with the trend at the end of the window.

    Uses the ``ema_cross`` feature (fast EMA over slow EMA, minus one) of the
    most recent step. Positive means the fast EMA is above the slow one.

    The output is one-hot on the rule's choice, for the reason given in the
    module docstring: a rule has no confidence to express, and expressing a
    fabricated one made the rule's *action* depend on the model's fitted
    threshold. ``deadband`` is the rule's own parameter and is not fitted on
    anything.
    """

    name = "momentum_rule"

    def __init__(self, feature_index: int, deadband: float = 0.001):
        self.feature_index = feature_index
        self.deadband = deadband

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"expected (n, seq_len, features), got {X.shape}")
        signal = X[:, -1, self.feature_index]
        proba = np.zeros((len(X), N_CLASSES), dtype=np.float64)
        long_mask = signal > self.deadband
        short_mask = signal < -self.deadband
        proba[long_mask, LONG_IDX] = 1.0
        proba[short_mask, SHORT_IDX] = 1.0
        proba[~(long_mask | short_mask), HOLD_IDX] = 1.0
        return proba
