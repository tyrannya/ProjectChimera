"""Chronological splitting, windowing and scaling.

This module is where leakage is prevented, so its rules are stated explicitly
and asserted by ``tests/test_dataset.py``.

**Splits are contiguous blocks of rows in time order.** No shuffling, ever.

**A window belongs to exactly one split.** The sample at row ``i`` consists of
feature rows ``[i - seq_len + 1, i]`` and the label derived from prices at row
``i + horizon``. A sample is emitted for split ``[start, end)`` only when::

    start + seq_len - 1  <=  i  <=  end - 1 - horizon

The left bound keeps the input window inside the block; the right bound keeps
the *label* inside it. Together they mean a training label can never be
computed from a price that falls in the validation block, and a validation
window can never contain a training row. That is the embargo — it comes from
the index arithmetic, not from a fudge factor.

When ``segment_ids`` are supplied, windows and their label horizon are also
required to stay inside one contiguous market-data segment. This prevents a
missing exchange candle from being silently treated as if two non-adjacent
hours were adjacent observations.

**The scaler is fitted on training rows only** and applied unchanged to
validation and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class Split:
    """A contiguous, half-open row range ``[start, end)``."""

    name: str
    start: int
    end: int

    def __len__(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class SplitPlan:
    """The three splits, in time order."""

    train: Split
    validation: Split
    test: Split

    def __iter__(self) -> Iterator[Split]:
        return iter((self.train, self.validation, self.test))

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {s.name: {"start": s.start, "end": s.end, "rows": len(s)} for s in self}


def chronological_split(
    n_rows: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> SplitPlan:
    """Split ``n_rows`` into three contiguous blocks, oldest first.

    The test block is whatever remains, so the fractions cannot silently drop
    rows on the floor.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum below 1")

    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))
    return SplitPlan(
        train=Split("train", 0, train_end),
        validation=Split("validation", train_end, val_end),
        test=Split("test", val_end, n_rows),
    )


def sealed_test_start(n_rows: int, train_frac: float = 0.7, val_frac: float = 0.15) -> int:
    """First row of the sealed test block: the boundary research must not cross.

    Delegates to :func:`chronological_split` with the same defaults ``nn.train``
    uses, so the single-split trainer and the walk-forward planner cannot
    disagree by a row about where sealed data begins. Research tooling plans
    over ``[0, sealed_test_start)`` and nothing else.

    Naming a split "validation" does not make its rows unsealed. Walk-forward
    once planned folds over the whole dataset, which put its last two validation
    windows inside the test block — the metrics were labelled validation and
    were, in substance, test. This function exists so that boundary is a number
    both sides compute the same way, and so tests can assert on row indices
    rather than on split names.
    """
    return chronological_split(n_rows, train_frac, val_frac).test.start


def window_indices(split: Split, seq_len: int, horizon: int) -> np.ndarray:
    """Row indices that can produce a valid sample inside ``split``.

    Returns an empty array when the block is too short to hold a single window
    plus its label, rather than silently borrowing rows from a neighbour.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    first = split.start + seq_len - 1
    last = split.end - 1 - horizon
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)


def build_windows(
    features: np.ndarray,
    targets: np.ndarray,
    split: Split,
    seq_len: int,
    horizon: int,
    *,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialise ``(X, y, row_index)`` for one split.

    ``X`` has shape ``(n_samples, seq_len, n_features)``. If ``segment_ids`` is
    provided, a candidate is kept only when both its whole input sequence and
    its embargoed label row belong to the same contiguous data segment.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")
    if segment_ids is not None and len(segment_ids) != len(features):
        raise ValueError("segment_ids must have the same length as features")

    idx = window_indices(split, seq_len, horizon)
    if idx.size == 0:
        return (
            np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            idx,
        )

    # Gather windows without a Python loop over samples.
    offsets = np.arange(-seq_len + 1, 1)
    window_rows = idx[:, None] + offsets[None, :]

    if segment_ids is not None:
        segments = np.asarray(segment_ids)
        current_segment = segments[idx]
        input_is_contiguous = (segments[window_rows] == current_segment[:, None]).all(axis=1)
        label_is_contiguous = segments[idx + horizon] == current_segment
        keep = input_is_contiguous & label_is_contiguous
        idx = idx[keep]
        window_rows = window_rows[keep]

    X = features[window_rows].astype(np.float32)
    y = targets[idx].astype(np.int64)
    return X, y, idx


class StandardScaler:
    """Per-feature standardisation, fitted on one slice and reused verbatim.

    Deliberately not scikit-learn's: the fitted parameters have to round-trip
    through JSON model metadata, and the whole implementation is six lines.
    """

    def __init__(
        self, mean: Sequence[float] | None = None, std: Sequence[float] | None = None
    ):
        self.mean = np.asarray(mean, dtype=np.float64) if mean is not None else None
        self.std = np.asarray(std, dtype=np.float64) if std is not None else None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Fit on 2-D rows. Call this with training rows and nothing else."""
        if X.ndim != 2:
            raise ValueError(f"expected 2-D array, got shape {X.shape}")
        if len(X) == 0:
            raise ValueError("cannot fit a scaler on zero rows")
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        # A constant feature has zero variance; dividing by it yields inf/NaN.
        # Scaling it by 1.0 leaves it as a constant zero after centring, which
        # is the correct, harmless behaviour.
        self.std = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("scaler has not been fitted")
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
