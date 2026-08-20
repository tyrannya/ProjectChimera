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

**The sealed test block begins at an immutable instant, not at a row.** That
instant is not this module's to choose: it comes from a committed
:class:`nn.research_contract.ResearchContract`, one per research generation.
:func:`resolve_sealed_boundary` is the only way to turn a contract's instant
into a row index for a particular dataset, and the row it returns is metadata
about that dataset — never the contract itself.

The boundary used to be ``int(n_rows * (train_frac + val_frac))``. Appending
candles moved it forward, so a timestamp that was sealed yesterday became
research data tomorrow, and the sealed estimate silently shrank every time the
dataset grew. A wall-clock anchor cannot move when rows are appended after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from nn.research_contract import ResearchContract


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


@dataclass(frozen=True)
class SealedBoundary:
    """The research contract, plus where its seal lands in one dataset.

    Both halves travel together on purpose. ``start_row`` is the only form the
    row-based planners can use, and it is exactly the part that is *not* the
    contract: it is a property of the dataset it was resolved against. Carrying
    the contract beside it is what lets an artifact say which seal it was
    produced under, so a row index can never be mistaken for the boundary
    itself — and, because the contract's identity is a hash of its
    research-defining content, two runs cannot be read as one generation merely
    for landing on the same row.
    """

    contract: ResearchContract
    start_row: int

    @property
    def anchor(self) -> pd.Timestamp:
        """The contract's sealed instant. The seal itself, in wall-clock terms."""
        return self.contract.sealed_test_start

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable provenance for a research artifact.

        ``anchor_timestamp`` and ``start_row`` keep the shape the timestamp
        contract introduced; ``research_contract`` adds the exact generation
        that instant came from, which the timestamp alone cannot identify.
        """
        return {
            "anchor_timestamp": self.anchor.isoformat(),
            "start_row": self.start_row,
            "research_contract": self.contract.provenance(),
        }


def resolve_sealed_boundary(dates: Any, *, contract: ResearchContract) -> SealedBoundary:
    """Locate ``contract``'s sealed instant in ``dates``: the first sealed row.

    The partition is defined on timestamps and only then expressed in rows::

        research: timestamp <  contract.sealed_test_start
        sealed:   timestamp >= contract.sealed_test_start

    so the returned ``start_row`` is the first row at or after the anchor.
    Appending any number of rows *after* the anchor cannot change it, which is
    the whole point: the previous row-fraction boundary moved forward every time
    the dataset grew, quietly handing previously-sealed timestamps to research.

    The anchor candle does **not** have to exist. When it is missing because of a
    real market-data gap or a dropped feature row, the first surviving row after
    it is the sealed start; no candle is invented to make the timestamp present.

    Fails closed rather than guessing, because every failure here is a case where
    the temporal partition would otherwise be ambiguous or absent:

    * malformed or unparseable timestamps;
    * timestamps that are not strictly increasing — unsorted *or* duplicated,
      either of which makes "the first row at or after the anchor" mean more
      than one thing;
    * no row before the anchor, so there is no research region;
    * no row at or after the anchor, so there is no sealed block.

    ``contract`` is a whole committed research contract, never a bare date:
    there is no argument here a caller can use to name an instant of its own.
    Production research entrypoints pass the contract their
    ``--research-contract`` selector chose out of the committed registry, and
    record its identity in every artifact they write.
    """
    anchor = contract.sealed_test_start

    # Naive input is read as UTC, which is what every dataset in this repository
    # stores; tz-aware input is converted rather than reinterpreted.
    try:
        stamps = pd.to_datetime(pd.Index(np.asarray(dates)), utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timestamps could not be parsed as UTC datetimes: {exc}") from exc

    n_rows = len(stamps)
    if n_rows == 0:
        raise ValueError("cannot resolve the sealed boundary in an empty date column")
    if stamps.hasnans:
        raise ValueError(
            f"{int(stamps.isna().sum())} timestamp(s) are missing or unparseable, so the "
            "sealed boundary cannot be resolved"
        )
    if not stamps.is_monotonic_increasing:
        raise ValueError(
            "timestamps are not sorted ascending, so 'the first row at or after "
            f"{anchor.isoformat()}' is ambiguous; sort the dataset before splitting it"
        )
    if not stamps.is_unique:
        duplicated = stamps[stamps.duplicated()]
        raise ValueError(
            f"{len(duplicated)} duplicate timestamp(s) (first: {duplicated[0]}) make the "
            "temporal partition ambiguous; de-duplicate the dataset before splitting it"
        )

    start_row = int(stamps.searchsorted(anchor, side="left"))
    if start_row == 0:
        raise ValueError(
            f"every row is at or after the sealed anchor {anchor.isoformat()} (data "
            f"starts at {stamps[0]}), so there is no research region to train on"
        )
    if start_row == n_rows:
        raise ValueError(
            f"every row is before the sealed anchor {anchor.isoformat()} (data ends at "
            f"{stamps[-1]}), so there is no sealed test block to withhold"
        )
    return SealedBoundary(contract=contract, start_row=start_row)


def chronological_split(
    n_rows: int,
    sealed_start: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> SplitPlan:
    """Split ``n_rows`` into three contiguous blocks, oldest first.

    ``sealed_start`` is the first sealed row, resolved from the research
    contract's sealed instant by :func:`resolve_sealed_boundary`. It is a
    required argument rather than something computed here, because computing it
    from ``n_rows`` is exactly the bug this signature exists to make impossible:
    the test block starts where the calendar says it starts, and the dataset's
    length has no vote.

    ``train_frac`` and ``val_frac`` allocate train against validation **inside
    the research region** ``[0, sealed_start)``, by weight — only their ratio is
    used. Under the default 0.70/0.15 the training block is 70/85 of the research
    region, which on the canonical dataset is the same row the previous
    ``int(n_rows * 0.70)`` produced. They cannot move ``test.start``.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum below 1")
    # Independently of the resolver, which refuses the same thing upstream: this
    # is a public partition primitive, and a plan whose sealed block is empty is
    # a plan with no held-out estimate at all.
    if not 0 < sealed_start < n_rows:
        raise ValueError(
            f"sealed_start ({sealed_start}) must leave rows on both sides of the "
            f"boundary in a {n_rows}-row dataset"
        )

    train_end = int(sealed_start * train_frac / (train_frac + val_frac))
    if not 0 < train_end < sealed_start:
        raise ValueError(
            f"a research region of {sealed_start} row(s) split {train_frac}/{val_frac} "
            f"leaves an empty train or validation block (train would end at {train_end})"
        )

    plan = SplitPlan(
        train=Split("train", 0, train_end),
        validation=Split("validation", train_end, sealed_start),
        test=Split("test", sealed_start, n_rows),
    )
    # The arithmetic above already guarantees these; check them anyway, because
    # a plan that quietly hands a sealed row to validation looks exactly like a
    # plan that does not. Raised rather than asserted, as in
    # :func:`nn.walkforward.plan_nested_folds`: a leakage guard that `python -O`
    # removes is not a guard.
    if not (
        plan.train.end <= plan.validation.start
        and plan.validation.end == sealed_start
        and plan.test.start == sealed_start
        and plan.test.end == n_rows
    ):
        raise AssertionError(f"split plan does not partition at {sealed_start}: {plan}")
    return plan


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


def sample_indices(
    split: Split,
    seq_len: int,
    horizon: int,
    *,
    segment_ids: np.ndarray | None = None,
) -> np.ndarray:
    """The rows a split actually yields samples for.

    :func:`window_indices` gives the candidates the block geometry allows; this
    additionally drops any candidate whose input window or embargoed label would
    straddle a market-data gap. The result is exactly the ``row_index`` array
    :func:`build_windows` returns, which is what makes it usable as *the*
    definition of "which rows were scored" — a later analysis can ask that
    question without rebuilding the tensors, and cannot answer it differently.
    """
    idx = window_indices(split, seq_len, horizon)
    if idx.size == 0 or segment_ids is None:
        return idx

    segments = np.asarray(segment_ids)
    offsets = np.arange(-seq_len + 1, 1)
    window_rows = idx[:, None] + offsets[None, :]
    current_segment = segments[idx]
    input_is_contiguous = (segments[window_rows] == current_segment[:, None]).all(axis=1)
    label_is_contiguous = segments[idx + horizon] == current_segment
    return idx[input_is_contiguous & label_is_contiguous]


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

    idx = sample_indices(split, seq_len, horizon, segment_ids=segment_ids)
    if idx.size == 0:
        return (
            np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            idx,
        )

    # Gather windows without a Python loop over samples.
    offsets = np.arange(-seq_len + 1, 1)
    window_rows = idx[:, None] + offsets[None, :]
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
