"""Feature engineering invariants.

The causality test is the important one: it is what makes every downstream
claim about "no look-ahead" checkable rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chimera.features import (
    FeatureSpec,
    compute_features,
    feature_columns,
    n_features,
)


def test_column_order_is_fixed(candles):
    out = compute_features(candles)
    assert list(out.columns) == feature_columns()
    assert len(out.columns) == n_features()


def test_output_is_aligned_to_input(candles):
    out = compute_features(candles)
    assert len(out) == len(candles)
    assert out.index.equals(candles.index)


def test_features_are_causal(candles):
    """A feature at row t must not change when future rows are removed.

    This is the whole no-look-ahead claim, tested directly: compute features on
    the full series, then recompute on a prefix, and require the overlapping
    rows to be identical. Any centred window, backfill, or forward shift in
    compute_features would break this.
    """
    full = compute_features(candles)
    for cut in (300, 500, 750):
        prefix = compute_features(candles.iloc[:cut])
        pd.testing.assert_frame_equal(prefix, full.iloc[:cut], check_exact=False, rtol=1e-12)


def test_features_are_deterministic(candles):
    a = compute_features(candles)
    b = compute_features(candles.copy())
    pd.testing.assert_frame_equal(a, b)


def test_features_are_finite_after_warmup(candles):
    spec = FeatureSpec()
    out = compute_features(candles, spec)
    tail = out.iloc[spec.warmup :]
    assert np.isfinite(tail.to_numpy()).all()


def test_features_do_not_depend_on_absolute_price_level(candles):
    """Scaling every price by a constant must barely move scale-free features.

    Guards the property that broke the original model: a network trained on
    2020 price levels cannot generalise to 2024 levels if its inputs are
    absolute prices.
    """
    scaled = candles.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] = scaled[column] * 10.0

    base = compute_features(candles).iloc[200:]
    lifted = compute_features(scaled).iloc[200:]
    # volume_* features are untouched by a price rescale; the price-derived
    # ones are ratios and must be invariant.
    price_features = [c for c in base.columns if not c.startswith("volume")]
    np.testing.assert_allclose(
        base[price_features].to_numpy(),
        lifted[price_features].to_numpy(),
        rtol=1e-9,
        atol=1e-12,
    )


def test_missing_columns_raise(candles):
    with pytest.raises(ValueError, match="missing columns"):
        compute_features(candles.drop(columns=["volume"]))


def test_zero_volume_does_not_produce_infinities():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=80, freq="1h", tz="UTC"),
            "open": np.linspace(100, 110, 80),
            "high": np.linspace(101, 111, 80),
            "low": np.linspace(99, 109, 80),
            "close": np.linspace(100, 110, 80),
            "volume": np.zeros(80),
        }
    )
    out = compute_features(frame)
    assert not np.isinf(out.to_numpy()).any()


def test_warmup_grows_with_windows():
    assert FeatureSpec(ema_slow=100).warmup > FeatureSpec(ema_slow=26).warmup
