"""Regression tests for missing-candle handling in features, labels and windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec
from nn.data_pipeline import build_dataset
from nn.dataset import Split, build_windows


def _candles_with_one_gap() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
    close = 100.0 * np.exp(np.linspace(0.0, 0.3, len(dates)))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1000.0 + 100.0 * np.sin(np.arange(len(dates)) / 7.0),
        }
    )
    # Remove one real candle in the middle, leaving two long valid segments.
    return frame.drop(index=200).reset_index(drop=True), dates


def test_dataset_resets_features_and_targets_at_gap():
    candles, original_dates = _candles_with_one_gap()
    feature_spec = FeatureSpec()
    target_spec = TargetSpec(horizon=6)

    frame, _ = build_dataset(
        candles,
        feature_spec,
        target_spec,
        exchange="binance",
        pair="BTC/USDT",
        timeframe="1h",
    )

    assert "segment_id" in frame.columns
    assert frame["segment_id"].nunique() == 2

    first = frame[frame["segment_id"] == 0]
    second = frame[frame["segment_id"] == 1]

    # The first segment's final horizon rows have no honest label and must be absent.
    assert first["date"].max() <= original_dates[193]
    # Indicators restart after the outage, so the second segment pays a fresh warm-up.
    assert second["date"].min() >= original_dates[201 + feature_spec.warmup]


def test_windows_never_bridge_segment_boundary():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(30, 3))
    targets = rng.integers(0, 3, size=30)
    segments = np.array([0] * 15 + [1] * 15, dtype=np.int64)
    split = Split("train", 0, 30)

    X_all, _, idx_all = build_windows(features, targets, split, seq_len=5, horizon=2)
    X_safe, _, idx_safe = build_windows(
        features,
        targets,
        split,
        seq_len=5,
        horizon=2,
        segment_ids=segments,
    )

    assert len(X_safe) < len(X_all)
    assert len(idx_safe) > 0
    for i in idx_safe:
        assert segments[i - 4] == segments[i]
        assert segments[i + 2] == segments[i]
