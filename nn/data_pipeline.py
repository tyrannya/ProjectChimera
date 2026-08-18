"""Market data ingestion, validation and dataset construction.

The contract this module upholds, in order:

1. raw candles are downloaded with UTC timestamps;
2. :func:`validate_ohlcv` proves the frame is well-formed before anything reads
   it — sorted, unique, UTC, internally consistent OHLC, non-negative volume;
3. :func:`build_dataset` joins causal features to a cost-aware label and drops
   every row that cannot be honestly labelled;
4. the result is written to Parquet with a metadata sidecar recording exactly
   which specs produced it.

Downloading needs ``ccxt``; everything else needs only pandas, so validation
and dataset construction are testable without a network stack.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chimera.contracts import HOLD_IDX, LONG_IDX, SHORT_IDX, TargetSpec
from chimera.features import FeatureSpec, compute_features, feature_columns

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
REQUIRED_COLUMNS = ["date", *OHLCV_COLUMNS]

_TIMEFRAME_MINUTES = {"m": 1, "h": 60, "d": 1440, "w": 10080}


class DataValidationError(ValueError):
    """Raised when a candle frame cannot be trusted."""


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert a Freqtrade/ccxt timeframe string such as ``4h`` to minutes."""
    timeframe = timeframe.strip().lower()
    if len(timeframe) < 2 or timeframe[-1] not in _TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    try:
        count = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}") from exc
    if count <= 0:
        raise ValueError(f"Timeframe must be positive: {timeframe!r}")
    return count * _TIMEFRAME_MINUTES[timeframe[-1]]


def _contiguous_segment_ids(dates: pd.Series, timeframe: str) -> pd.Series:
    """Return monotonically increasing ids for uninterrupted candle runs.

    A new segment begins whenever the elapsed time differs from exactly one
    requested candle. Features, targets and sequence windows must never bridge
    such a boundary: doing so would silently turn a multi-hour outage into one
    apparent one-hour transition.
    """
    if len(dates) == 0:
        return pd.Series(dtype="int64", index=dates.index)
    if not timeframe:
        return pd.Series(np.zeros(len(dates), dtype=np.int64), index=dates.index)

    step = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
    parsed = pd.to_datetime(dates, utc=True)
    breaks = parsed.diff().ne(step)
    breaks.iloc[0] = True
    return (breaks.cumsum() - 1).astype("int64")


@dataclass
class ValidationReport:
    """What :func:`validate_ohlcv` found. Logged, and stored with the dataset."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    reordered: bool = False
    missing_candles: int = 0
    gap_count: int = 0
    zero_volume_rows: int = 0
    nan_rows_dropped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe: str | None = None,
    *,
    drop_nan: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Normalise and check a candle frame.

    Returns the cleaned frame and a report. Raises :class:`DataValidationError`
    for problems that cannot be repaired without inventing data — a missing
    column, a non-positive price, or a high/low that contradicts the open/close.

    Repairable problems (unsorted rows, duplicate timestamps, naive datetimes)
    are fixed and counted. Missing candles are *reported, never filled*:
    forward-filling a gap fabricates market data that a backtest would then
    trade on.
    """
    report = ValidationReport(rows_in=len(df))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")

    out = df.loc[:, REQUIRED_COLUMNS].copy()

    # --- timestamps: UTC, unique, monotonic ---------------------------
    out["date"] = pd.to_datetime(out["date"], utc=True)
    if out["date"].isna().any():
        raise DataValidationError("unparseable timestamps in 'date'")

    before = len(out)
    out = out.drop_duplicates(subset="date", keep="last")
    report.duplicates_removed = before - len(out)

    if not out["date"].is_monotonic_increasing:
        report.reordered = True
        out = out.sort_values("date")
    out = out.reset_index(drop=True)

    # --- numeric sanity -----------------------------------------------
    for col in OHLCV_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if drop_nan:
        before = len(out)
        out = out.dropna(subset=OHLCV_COLUMNS).reset_index(drop=True)
        report.nan_rows_dropped = before - len(out)

    if out.empty:
        report.rows_out = 0
        return out, report

    prices = out[["open", "high", "low", "close"]]
    if (prices <= 0).to_numpy().any():
        raise DataValidationError("non-positive price found in OHLC")
    if (out["volume"] < 0).any():
        raise DataValidationError("negative volume found")

    body_high = out[["open", "close"]].max(axis=1)
    body_low = out[["open", "close"]].min(axis=1)
    if (out["high"] < body_high - 1e-9).any():
        raise DataValidationError("high is below max(open, close) on some candles")
    if (out["low"] > body_low + 1e-9).any():
        raise DataValidationError("low is above min(open, close) on some candles")
    if (out["high"] < out["low"]).any():
        raise DataValidationError("high is below low on some candles")

    report.zero_volume_rows = int((out["volume"] == 0).sum())

    # --- gaps ----------------------------------------------------------
    if timeframe is not None and len(out) > 1:
        step = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
        deltas = out["date"].diff().dropna()
        gaps = deltas[deltas > step]
        report.gap_count = int(len(gaps))
        report.missing_candles = int(((gaps / step) - 1).round().sum())
        if report.gap_count:
            logger.warning(
                "%d gaps in %s data, %d candles missing (not filled)",
                report.gap_count,
                timeframe,
                report.missing_candles,
            )

    report.rows_out = len(out)
    if report.zero_volume_rows:
        logger.warning("%d zero-volume candles retained", report.zero_volume_rows)
    return out, report


def compute_future_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return over the next ``horizon`` candles, aligned to the *current* row.

    ``shift(-horizon)`` looks forward on purpose — this is the label, not a
    feature. The last ``horizon`` rows become NaN and are dropped by
    :func:`build_dataset`, which is what keeps the forward look out of the
    training inputs.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return close.shift(-horizon) / close - 1.0


def compute_target(close: pd.Series, spec: TargetSpec) -> pd.Series:
    """Cost-aware SHORT/HOLD/LONG labels. See :class:`TargetSpec`."""
    future_return = compute_future_return(close, spec.horizon)
    threshold = spec.cost_threshold

    labels = pd.Series(np.nan, index=close.index, dtype="float64")
    known = future_return.notna()
    labels[known & (future_return > threshold)] = LONG_IDX
    labels[known & (future_return < -threshold)] = SHORT_IDX
    labels[known & (future_return.abs() <= threshold)] = HOLD_IDX
    return labels


@dataclass
class DatasetMetadata:
    """Provenance for a built dataset, written beside the Parquet file."""

    exchange: str = ""
    pair: str = ""
    timeframe: str = ""
    rows: int = 0
    start: str = ""
    end: str = ""
    feature_names: list[str] = field(default_factory=feature_columns)
    feature_spec: dict[str, int] = field(default_factory=lambda: FeatureSpec().to_dict())
    target_spec: dict[str, Any] = field(default_factory=lambda: TargetSpec().to_dict())
    class_balance: dict[str, int] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetMetadata":
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in fields})


def build_dataset(
    ohlcv: pd.DataFrame,
    feature_spec: FeatureSpec | None = None,
    target_spec: TargetSpec | None = None,
    *,
    exchange: str = "",
    pair: str = "",
    timeframe: str = "",
    validation: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Join causal features to cost-aware labels without crossing data gaps.

    Each uninterrupted candle run is processed independently. This means an
    exchange outage resets rolling/EMA state, and neither a feature nor a target
    can use prices from the other side of a missing-candle gap.

    Within every segment three trims happen, and each removes rows that would
    otherwise be a lie:

    * the first ``feature_spec.warmup`` rows, where indicators have not
      converged;
    * any row with a NaN feature;
    * the last ``target_spec.horizon`` rows, whose future return is unknowable.
    """
    feature_spec = feature_spec or FeatureSpec()
    target_spec = target_spec or TargetSpec()

    if ohlcv.empty:
        raise DataValidationError("cannot build a dataset from an empty frame")

    source = ohlcv.reset_index(drop=True).copy()
    source["date"] = pd.to_datetime(source["date"], utc=True)
    source["segment_id"] = _contiguous_segment_ids(source["date"], timeframe).to_numpy()

    blocks: list[pd.DataFrame] = []
    for segment_id, group in source.groupby("segment_id", sort=True):
        candles = group[["date", *OHLCV_COLUMNS]].reset_index(drop=True)
        features = compute_features(candles, feature_spec)
        target = compute_target(candles["close"], target_spec)
        future_return = compute_future_return(candles["close"], target_spec.horizon)

        block = pd.concat(
            [
                candles[["date", "close"]].reset_index(drop=True),
                pd.Series(segment_id, index=candles.index, name="segment_id", dtype="int64"),
                features.reset_index(drop=True),
                future_return.rename("future_return").reset_index(drop=True),
                target.rename("target").reset_index(drop=True),
            ],
            axis=1,
        )

        warmup = min(feature_spec.warmup, len(block))
        block = block.iloc[warmup:].dropna().reset_index(drop=True)
        if not block.empty:
            blocks.append(block)

    if not blocks:
        raise DataValidationError(
            f"no rows survived warm-up ({feature_spec.warmup}) and horizon "
            f"({target_spec.horizon}); supply a longer history"
        )

    frame = pd.concat(blocks, ignore_index=True)

    # A feature can itself become undefined on an otherwise valid candle (for
    # example volume_change immediately after zero volume). After dropping such
    # a row, split the final dataset again so sequence windows cannot jump over
    # the missing observation either.
    frame["segment_id"] = _contiguous_segment_ids(frame["date"], timeframe).to_numpy()
    frame["target"] = frame["target"].astype("int64")

    counts = frame["target"].value_counts().to_dict()
    metadata = DatasetMetadata(
        exchange=exchange,
        pair=pair,
        timeframe=timeframe,
        rows=len(frame),
        start=frame["date"].iloc[0].isoformat(),
        end=frame["date"].iloc[-1].isoformat(),
        feature_names=feature_columns(),
        feature_spec=feature_spec.to_dict(),
        target_spec=target_spec.to_dict(),
        class_balance={
            "SHORT": int(counts.get(SHORT_IDX, 0)),
            "HOLD": int(counts.get(HOLD_IDX, 0)),
            "LONG": int(counts.get(LONG_IDX, 0)),
        },
        validation=dict(validation or {}),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return frame, metadata


def save_dataset(path: str | Path, frame: pd.DataFrame, metadata: DatasetMetadata) -> Path:
    """Write ``frame`` to Parquet and ``metadata`` to ``<path>.meta.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(metadata.to_dict(), indent=2)
    )
    logger.info("Wrote %d rows to %s", len(frame), path)
    return path


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Read a dataset and its sidecar. Missing sidecar yields empty metadata."""
    path = Path(path)
    frame = pd.read_parquet(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        metadata = DatasetMetadata.from_dict(json.loads(meta_path.read_text()))
    else:
        logger.warning("No metadata sidecar next to %s", path)
        metadata = DatasetMetadata()
    return frame, metadata


def download_ohlcv(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    *,
    end_date: str | None = None,
    sandbox: bool = False,
    limit: int = 1000,
    max_batches: int = 10_000,
) -> pd.DataFrame:
    """Page through an exchange's OHLCV endpoint via ccxt.

    Returns a frame with a UTC ``date`` column plus OHLCV. Not validated here —
    callers pass the result to :func:`validate_ohlcv`.
    """
    import ccxt

    if not hasattr(ccxt, exchange_name):
        raise ValueError(f"Unknown exchange for ccxt: {exchange_name!r}")
    exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    if sandbox:
        if not hasattr(exchange, "set_sandbox_mode"):
            raise ValueError(f"{exchange_name} does not support sandbox mode")
        exchange.set_sandbox_mode(True)

    since = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    until = (
        int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
        if end_date
        else int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    )
    step_ms = timeframe_to_minutes(timeframe) * 60_000

    rows: list[list[Any]] = []
    for _ in range(max_batches):
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        next_since = batch[-1][0] + step_ms
        if next_since <= since or next_since >= until:
            break
        since = next_since
        if len(batch) < limit:
            break
    else:
        logger.warning("Stopped after %d batches; data may be incomplete", max_batches)

    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    frame = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    frame["date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame[REQUIRED_COLUMNS]
