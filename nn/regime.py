"""Market-regime statistics for a walk-forward outer block.

``nn.wf_diagnostics`` can tell you that fold 2 earned more than fold 1. It
cannot tell you *what was different about the market* in those two stretches,
because it never opens a dataset. This module does, under three rules.

**The sealed boundary is enforced by slicing, not by discipline.**
:func:`load_research_frame` truncates the processed dataset at
``sealed_test_start`` the moment it is read. Every statistic below is computed
from that object, so a sealed row is not merely unused — it is not in the frame
at all, and no off-by-one in a block range can reach one. The row range is
checked against the boundary again on the way in, because a cheap assertion is
worth more than a comment.

**Row indices are only meaningful against the dataset they came from.** The
indices in ``walkforward.json`` address one specific processed dataset. Point
this at a rebuilt one and every index silently means a different candle, so the
loader refuses a dataset whose row count, feature contract or target spec does
not match what the artifact recorded.

**Raw candles are joined on timestamps, never on position.** The processed
dataset has lost warm-up and label rows, so processed row *i* is not raw row
*i*. :func:`align_raw` matches exact timestamps and fails closed on a missing
one, a duplicate, or a timeframe that does not agree — rather than shifting
rows and reporting a confident number about the wrong hours.

Nothing here explains anything. It reports what differed; whether a difference
caused anything is not a question this data can answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chimera.contracts import CLASS_ORDER, TargetSpec
from nn import evaluate as ev
from nn.data_pipeline import OHLCV_COLUMNS, load_dataset, timeframe_to_minutes

logger = logging.getLogger(__name__)

#: Feature columns to summarise, and which statistics each one gets.
#:
#: Keyed by the feature's real name in the dataset, never by column position:
#: the feature contract is recorded in the artifact and can change between
#: dataset builds, and a positional read would silently summarise the wrong
#: column when it does.
FEATURE_STATS: dict[str, tuple[str, ...]] = {
    "ema_cross": ("mean", "median", "std", "fraction_positive", "fraction_negative"),
    "ema_fast_ratio": ("mean", "median"),
    "ema_slow_ratio": ("mean", "median"),
    "atr_norm": ("mean", "median", "p90"),
    "realized_vol": ("mean", "median", "p90"),
    "hl_range": ("mean", "median"),
    "volume_change": ("mean", "median"),
    "volume_z": ("mean", "median", "p90_abs"),
}

_STATISTICS = {
    "mean": lambda x: float(np.mean(x)),
    "median": lambda x: float(np.median(x)),
    "std": lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    "p90": lambda x: float(np.percentile(x, 90)),
    "p90_abs": lambda x: float(np.percentile(np.abs(x), 90)),
    "fraction_positive": lambda x: float(np.mean(x > 0)),
    "fraction_negative": lambda x: float(np.mean(x < 0)),
}

_CLASS_NAMES = [c.value for c in CLASS_ORDER]


class RegimeDataError(Exception):
    """The dataset cannot be trusted to mean what the artifact's rows say."""


@dataclass(frozen=True)
class ResearchFrame:
    """The processed dataset, already truncated at the sealed boundary."""

    #: Rows ``[0, sealed_test_start)`` and nothing else.
    frame: pd.DataFrame
    sealed_test_start: int
    feature_names: list[str]
    target_spec: TargetSpec
    timeframe: str

    def block(self, start: int, end: int) -> pd.DataFrame:
        """Rows ``[start, end)``, refusing anything at or past the boundary."""
        if not 0 <= start < end:
            raise RegimeDataError(f"invalid block range [{start}, {end})")
        if end > self.sealed_test_start:
            raise RegimeDataError(
                f"block [{start}, {end}) reaches row {end - 1}, at or beyond the sealed "
                f"test block starting at {self.sealed_test_start}"
            )
        return self.frame.iloc[start:end]


def load_research_frame(
    path: str | Path,
    *,
    sealed_test_start: int,
    expected_rows: int | None = None,
    expected_features: Sequence[str] | None = None,
    expected_target_spec: Mapping[str, Any] | None = None,
) -> ResearchFrame:
    """Read the processed dataset and cut it at the sealed boundary immediately.

    The ``expected_*`` arguments come from the walk-forward artifact. They are
    checked before anything is computed, because a row index is only a claim
    about a candle if the dataset is the one the index was recorded against.
    """
    frame, meta = load_dataset(path)
    n_rows = len(frame)

    if expected_rows is not None and n_rows != expected_rows:
        raise RegimeDataError(
            f"{path} has {n_rows} rows but the walk-forward artifact was produced "
            f"against a {expected_rows}-row dataset. Row indices would address "
            "different candles; refusing to compute statistics."
        )
    if sealed_test_start > n_rows:
        raise RegimeDataError(
            f"the artifact's sealed boundary is row {sealed_test_start} but {path} has "
            f"only {n_rows} rows"
        )

    feature_names = list(meta.feature_names)
    if expected_features is not None and feature_names != list(expected_features):
        raise RegimeDataError(
            "the dataset's feature contract differs from the one the artifact "
            f"recorded: {feature_names} vs {list(expected_features)}"
        )
    if expected_target_spec is not None and dict(meta.target_spec) != dict(
        expected_target_spec
    ):
        raise RegimeDataError(
            "the dataset's target spec differs from the one the artifact recorded: "
            f"{dict(meta.target_spec)} vs {dict(expected_target_spec)}"
        )

    for column in ("date", "target", "future_return"):
        if column not in frame.columns:
            raise RegimeDataError(f"{path} has no {column!r} column")

    # The slice is the safeguard: sealed rows leave the process here, once, and
    # nothing downstream has to remember not to look at them.
    research = frame.iloc[:sealed_test_start].copy()
    research["date"] = pd.to_datetime(research["date"], utc=True)
    return ResearchFrame(
        frame=research,
        sealed_test_start=sealed_test_start,
        feature_names=feature_names,
        target_spec=TargetSpec.from_dict(meta.target_spec),
        timeframe=meta.timeframe or "1h",
    )


def _summarise(values: np.ndarray, statistics: Sequence[str]) -> dict[str, float]:
    return {name: round(_STATISTICS[name](values), 8) for name in statistics}


def block_statistics(research: ResearchFrame, start: int, end: int) -> dict[str, Any]:
    """Everything the processed dataset knows about one outer block."""
    block = research.block(start, end)
    future_return = block["future_return"].to_numpy(dtype=np.float64)
    targets = block["target"].to_numpy(dtype=np.int64)

    missing = [name for name in FEATURE_STATS if name not in block.columns]
    if missing:
        raise RegimeDataError(
            f"the dataset is missing feature column(s) {missing}; regime statistics "
            "are defined against the feature names the artifact recorded"
        )

    features = {
        name: _summarise(block[name].to_numpy(dtype=np.float64), statistics)
        for name, statistics in FEATURE_STATS.items()
    }

    counts = np.bincount(targets, minlength=len(_CLASS_NAMES))
    target_distribution: dict[str, Any] = {}
    for index, name in enumerate(_CLASS_NAMES):
        selected = future_return[targets == index]
        target_distribution[name] = {
            "count": int(counts[index]),
            "fraction": round(float(counts[index] / len(block)), 8),
            "mean_future_return": round(float(selected.mean()), 8) if selected.size else 0.0,
            "median_future_return": (
                round(float(np.median(selected)), 8) if selected.size else 0.0
            ),
        }

    return {
        "row_range": [start, end],
        "rows": len(block),
        "period": {
            "start": str(block["date"].iloc[0]),
            "end": str(block["date"].iloc[-1]),
        },
        "future_return": {
            "mean": round(float(future_return.mean()), 8),
            "median": round(float(np.median(future_return)), 8),
            "std": round(float(future_return.std(ddof=1)), 8),
            "mean_abs": round(float(np.abs(future_return).mean()), 8),
            "fraction_positive": round(float((future_return > 0).mean()), 8),
            "fraction_negative": round(float((future_return < 0).mean()), 8),
        },
        "features": features,
        "target_distribution": target_distribution,
    }


# --- raw OHLCV ----------------------------------------------------------------
def load_raw_ohlcv(path: str | Path) -> pd.DataFrame:
    """Read raw candles as a timestamp-indexed lookup table.

    Fails closed on anything that would make a timestamp join ambiguous. The
    alternative — dropping duplicates and carrying on — produces a table that
    joins successfully and answers about the wrong candle.
    """
    frame = pd.read_parquet(path)
    missing = [c for c in ("date", *OHLCV_COLUMNS) if c not in frame.columns]
    if missing:
        raise RegimeDataError(f"{path} is missing OHLCV column(s): {missing}")

    dates = pd.to_datetime(frame["date"], utc=True)
    if dates.isna().any():
        raise RegimeDataError(f"{path} has {int(dates.isna().sum())} unparseable timestamps")
    duplicated = dates.duplicated()
    if duplicated.any():
        examples = dates[duplicated].head(3).tolist()
        raise RegimeDataError(
            f"{path} has {int(duplicated.sum())} duplicate timestamps (e.g. {examples}); "
            "a timestamp join would be ambiguous"
        )

    out = frame.assign(date=dates).sort_values("date").set_index("date")
    return out[list(OHLCV_COLUMNS)]


def align_raw(raw: pd.DataFrame, timestamps: pd.Series, timeframe: str) -> pd.DataFrame:
    """Match processed timestamps to raw candles exactly. Never positionally.

    The processed dataset dropped warm-up and label rows, so processed row *i*
    and raw row *i* are different candles. Any missing timestamp is an error:
    silently returning the rows that did match would report a regime computed
    over a different set of hours than the fold was scored on.
    """
    wanted = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True), utc=True)
    if wanted.duplicated().any():
        raise RegimeDataError("the processed block contains duplicate timestamps")

    known = raw.index
    present = wanted.isin(known)
    if not present.all():
        absent = wanted[~present]
        raise RegimeDataError(
            f"{int((~present).sum())} of {len(wanted)} processed timestamps have no raw "
            f"candle (e.g. {absent.head(3).tolist()}); refusing to align by position"
        )

    expected = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
    deltas = known.to_series().diff().dropna()
    if not deltas.empty and deltas.median() != expected:
        raise RegimeDataError(
            f"raw candles are spaced {deltas.median()} apart but the dataset's timeframe "
            f"{timeframe!r} implies {expected}"
        )
    return raw.loc[wanted.to_numpy()]


def raw_block_statistics(
    raw: pd.DataFrame, timestamps: pd.Series, timeframe: str
) -> dict[str, Any]:
    """Market behaviour over one outer block, from the raw candles it covers.

    Candle-to-candle returns are taken only between timestamps exactly one
    timeframe apart. A gap in the exchange's history is not a price move, and
    treating it as one would put a fabricated jump in the volatility estimate;
    the count of skipped pairs is reported instead of hidden.
    """
    candles = align_raw(raw, timestamps, timeframe)
    close = candles["close"].to_numpy(dtype=np.float64)
    open_ = candles["open"].to_numpy(dtype=np.float64)

    step = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
    gaps = candles.index.to_series().diff().to_numpy()[1:]
    adjacent = gaps == step.to_timedelta64()
    returns = (close[1:] / close[:-1] - 1.0)[adjacent]

    peak = np.maximum.accumulate(close)
    candles_per_year = 365 * 24 * 60 / timeframe_to_minutes(timeframe)

    return {
        "candles": len(candles),
        "start_close": round(float(close[0]), 8),
        "end_close": round(float(close[-1]), 8),
        "market_return": round(float(close[-1] / close[0] - 1.0), 8),
        "mean_candle_return": round(float(returns.mean()), 8) if returns.size else 0.0,
        "median_candle_return": (round(float(np.median(returns)), 8) if returns.size else 0.0),
        "candle_return_std": (
            round(float(returns.std(ddof=1)), 8) if returns.size > 1 else 0.0
        ),
        "annualised_volatility": (
            round(float(returns.std(ddof=1) * np.sqrt(candles_per_year)), 8)
            if returns.size > 1
            else 0.0
        ),
        "mean_abs_candle_return": (
            round(float(np.abs(returns).mean()), 8) if returns.size else 0.0
        ),
        # A candle is positive when it closed above its own open.
        "positive_candle_fraction": round(float((close > open_).mean()), 8),
        "negative_candle_fraction": round(float((close < open_).mean()), 8),
        "max_drawdown": round(float((1.0 - close / peak).max()), 8),
        "gap_pairs_skipped": int((~adjacent).sum()),
    }


# --- LONG / SHORT attribution --------------------------------------------------
PREDICTION_COLUMNS = (
    "fold",
    "seed",
    "row_index",
    "timestamp",
    "true_target",
    "future_return",
    "p_short",
    "p_hold",
    "p_long",
    "selected_action",
    "threshold",
)


def load_predictions(path: str | Path, *, sealed_test_start: int) -> pd.DataFrame:
    """Read persisted outer predictions, refusing anything sealed or malformed."""
    frame = pd.read_parquet(path)
    missing = [c for c in PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise RegimeDataError(f"{path} is missing prediction column(s): {missing}")
    if frame.empty:
        raise RegimeDataError(f"{path} contains no predictions")

    highest = int(frame["row_index"].max())
    if highest >= sealed_test_start:
        raise RegimeDataError(
            f"{path} contains row {highest}, at or beyond the sealed test block starting "
            f"at {sealed_test_start}. This artifact is not research output; refusing it."
        )
    return frame


def direction_attribution(
    predictions: pd.DataFrame, target_spec: TargetSpec
) -> dict[str, Any]:
    """Split realised trades into LONG and SHORT, with the same cost model.

    The trades come from :func:`nn.evaluate.realised_trades` — the function
    ``trading_metrics`` itself uses — so the per-direction returns sum to the
    same trades the fold reports, and no second transaction-cost implementation
    exists to drift from the first.

    Trades are generated per ``(fold, seed)`` group: the non-overlap rule is
    about one model walking one block in time order, and pooling the groups
    first would let one run's trade suppress another's.
    """
    short_idx, hold_idx, long_idx = range(len(CLASS_ORDER))
    directions: list[np.ndarray] = []
    returns: list[np.ndarray] = []

    ordered = predictions.sort_values(["fold", "seed", "row_index"])
    for _, group in ordered.groupby(["fold", "seed"], sort=True):
        _, group_directions, group_returns = ev.realised_trades(
            group["selected_action"].to_numpy(dtype=np.int64),
            group["future_return"].to_numpy(dtype=np.float64),
            target_spec,
        )
        directions.append(group_directions)
        returns.append(group_returns)

    all_directions = np.concatenate(directions) if directions else np.empty(0)
    all_returns = np.concatenate(returns) if returns else np.empty(0)

    def side(mask: np.ndarray) -> dict[str, Any]:
        selected = all_returns[mask]
        return {
            "trades": int(mask.sum()),
            "hit_rate": round(float((selected > 0).mean()), 6) if selected.size else 0.0,
            "mean_net_return": round(float(selected.mean()), 8) if selected.size else 0.0,
            "median_net_return": (
                round(float(np.median(selected)), 8) if selected.size else 0.0
            ),
            # Additive contribution: how much of the total came from this side.
            "cumulative_contribution": round(float(selected.sum()), 8),
        }

    actions = predictions["selected_action"].to_numpy(dtype=np.int64)
    n_samples = len(actions)
    return {
        "samples": n_samples,
        "long": side(all_directions > 0),
        "short": side(all_directions < 0),
        "hold_samples": int((actions == hold_idx).sum()),
        # Signal coverage, matching nn.evaluate's definition: how often each
        # side was called at all, before the non-overlap rule drops any.
        "long_coverage": round(float((actions == long_idx).mean()), 6),
        "short_coverage": round(float((actions == short_idx).mean()), 6),
    }
