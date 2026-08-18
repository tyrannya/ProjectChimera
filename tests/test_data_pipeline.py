"""OHLCV validation, target construction and dataset assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import HOLD_IDX, LONG_IDX, SHORT_IDX, TargetSpec
from chimera.features import FeatureSpec, feature_columns
from nn.data_pipeline import (
    DataValidationError,
    build_dataset,
    compute_future_return,
    compute_target,
    load_dataset,
    save_dataset,
    timeframe_to_minutes,
    validate_ohlcv,
)


# --- timeframe parsing ---------------------------------------------------
@pytest.mark.parametrize(
    ("timeframe", "minutes"), [("1m", 1), ("5m", 5), ("1h", 60), ("4h", 240), ("1d", 1440)]
)
def test_timeframe_to_minutes(timeframe, minutes):
    assert timeframe_to_minutes(timeframe) == minutes


@pytest.mark.parametrize("bad", ["", "h", "0h", "-1h", "1y", "abc"])
def test_timeframe_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        timeframe_to_minutes(bad)


# --- validation ----------------------------------------------------------
def test_validate_accepts_clean_candles(candles):
    out, report = validate_ohlcv(candles, "1h")
    assert len(out) == len(candles)
    assert report.duplicates_removed == 0
    assert report.gap_count == 0
    assert out["date"].is_monotonic_increasing


def test_validate_sorts_unordered_rows(candles):
    shuffled = candles.iloc[::-1].reset_index(drop=True)
    out, report = validate_ohlcv(shuffled, "1h")
    assert report.reordered is True
    assert out["date"].is_monotonic_increasing
    pd.testing.assert_series_equal(out["close"], candles["close"], check_names=False)


def test_validate_deduplicates_timestamps(candles):
    doubled = pd.concat([candles, candles.iloc[100:110]], ignore_index=True)
    out, report = validate_ohlcv(doubled, "1h")
    assert report.duplicates_removed == 10
    assert out["date"].is_unique


def test_validate_converts_naive_timestamps_to_utc(candles):
    naive = candles.copy()
    naive["date"] = naive["date"].dt.tz_localize(None)
    out, _ = validate_ohlcv(naive, "1h")
    assert str(out["date"].dt.tz) == "UTC"


def test_validate_counts_missing_candles_without_filling(candles):
    with_gap = candles.drop(index=range(200, 205)).reset_index(drop=True)
    out, report = validate_ohlcv(with_gap, "1h")
    assert report.gap_count == 1
    assert report.missing_candles == 5
    # The gap is reported, never fabricated.
    assert len(out) == len(candles) - 5


def test_validate_rejects_missing_columns(candles):
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_ohlcv(candles.drop(columns=["high"]), "1h")


def test_validate_rejects_non_positive_prices(candles):
    broken = candles.copy()
    broken.loc[10, "close"] = 0.0
    with pytest.raises(DataValidationError, match="non-positive price"):
        validate_ohlcv(broken, "1h")


def test_validate_rejects_negative_volume(candles):
    broken = candles.copy()
    broken.loc[10, "volume"] = -1.0
    with pytest.raises(DataValidationError, match="negative volume"):
        validate_ohlcv(broken, "1h")


def test_validate_rejects_inconsistent_ohlc(candles):
    broken = candles.copy()
    broken.loc[10, "high"] = broken.loc[10, "low"] * 0.5
    with pytest.raises(DataValidationError):
        validate_ohlcv(broken, "1h")


# --- targets -------------------------------------------------------------
def test_future_return_is_forward_looking_by_exactly_horizon():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    out = compute_future_return(close, horizon=1)
    assert out.iloc[0] == pytest.approx(0.10)
    assert out.iloc[1] == pytest.approx(0.10)
    assert pd.isna(out.iloc[-1])


def test_target_uses_the_cost_threshold():
    spec = TargetSpec(horizon=1, fee_rate=0.001, slippage_rate=0.001)
    assert spec.cost_threshold == pytest.approx(0.004)

    # +1% clears the 0.4% threshold, +0.2% does not, -1% clears it downward.
    close = pd.Series([100.0, 101.0, 101.2, 100.188, 100.0])
    labels = compute_target(close, spec)
    assert labels.iloc[0] == LONG_IDX
    assert labels.iloc[1] == HOLD_IDX
    assert labels.iloc[2] == SHORT_IDX
    assert pd.isna(labels.iloc[-1])


def test_last_horizon_rows_have_no_target():
    spec = TargetSpec(horizon=5)
    close = pd.Series(np.linspace(100, 200, 50))
    labels = compute_target(close, spec)
    assert labels.iloc[-5:].isna().all()
    assert labels.iloc[:-5].notna().all()


# --- dataset -------------------------------------------------------------
def test_build_dataset_drops_warmup_and_horizon(candles):
    feature_spec = FeatureSpec()
    target_spec = TargetSpec(horizon=6)
    frame, metadata = build_dataset(candles, feature_spec, target_spec, timeframe="1h")

    assert set(feature_columns()) <= set(frame.columns)
    assert {"date", "close", "future_return", "target"} <= set(frame.columns)
    assert frame.notna().to_numpy().all()
    assert len(frame) <= len(candles) - feature_spec.warmup - target_spec.horizon
    assert metadata.rows == len(frame)
    assert sum(metadata.class_balance.values()) == len(frame)


def test_dataset_target_matches_its_own_future_return(candles):
    """The stored label and the stored future return must agree.

    If these ever diverge, the model is trained against one definition and
    evaluated against another.
    """
    target_spec = TargetSpec(horizon=6)
    frame, _ = build_dataset(candles, FeatureSpec(), target_spec, timeframe="1h")
    threshold = target_spec.cost_threshold

    expected = np.where(
        frame["future_return"] > threshold,
        LONG_IDX,
        np.where(frame["future_return"] < -threshold, SHORT_IDX, HOLD_IDX),
    )
    np.testing.assert_array_equal(frame["target"].to_numpy(), expected)


def test_build_dataset_rejects_too_little_history(small_candles):
    with pytest.raises(DataValidationError, match="no rows survived"):
        build_dataset(small_candles.iloc[:60], FeatureSpec(), TargetSpec())


def test_build_dataset_rejects_empty_input():
    with pytest.raises(DataValidationError):
        build_dataset(pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]))


def test_dataset_roundtrips_with_metadata(candles, tmp_path):
    frame, metadata = build_dataset(
        candles, exchange="binance", pair="BTC/USDT", timeframe="1h"
    )
    path = save_dataset(tmp_path / "ds.parquet", frame, metadata)

    loaded, loaded_meta = load_dataset(path)
    pd.testing.assert_frame_equal(loaded, frame)
    assert loaded_meta.pair == "BTC/USDT"
    assert loaded_meta.timeframe == "1h"
    assert loaded_meta.feature_names == feature_columns()
    assert loaded_meta.target_spec["horizon"] == TargetSpec().horizon
