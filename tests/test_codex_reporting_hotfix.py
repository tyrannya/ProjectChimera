"""Regressions for the two post-merge Codex findings on PR #28."""

from __future__ import annotations

import numpy as np
import pytest

from chimera.contracts import TargetSpec
from nn import evaluate as ev


def test_wiped_out_buy_and_hold_withholds_sharpe_without_aborting():
    """A hold can lose all equity without making the whole evaluation fail."""
    spec = TargetSpec(horizon=1)
    market = ev.MarketContext(
        close=np.array([100.0, 0.1], dtype=np.float64),
        dates=np.array(
            ["2026-01-01T00:00", "2026-01-01T01:00"], dtype="datetime64[m]"
        ),
        candles_per_year=24 * 365,
    )

    report = ev.buy_and_hold_reference(
        market,
        spec,
        np.array([0], dtype=np.int64),
    )

    assert report["available"] is True
    assert report["gross_return"] == pytest.approx(-0.999)
    assert report["net_return"] == pytest.approx(-1.001)
    assert report["candle_max_drawdown"] == pytest.approx(1.001)
    assert report["annualised_sharpe"] is None
    assert "zero or below" in report["annualised_sharpe_reason"]
    assert report["sharpe_basis"] == ev.SHARPE_BASIS


def test_normal_buy_and_hold_still_reports_the_canonical_sharpe_basis():
    """The failure handling must not alter the ordinary comparable path."""
    spec = TargetSpec(horizon=1)
    market = ev.MarketContext(
        close=np.array([100.0, 101.0, 99.0], dtype=np.float64),
        dates=np.array(
            [
                "2026-01-01T00:00",
                "2026-01-01T01:00",
                "2026-01-01T02:00",
            ],
            dtype="datetime64[m]",
        ),
        candles_per_year=24 * 365,
    )

    report = ev.buy_and_hold_reference(
        market,
        spec,
        np.array([0, 1], dtype=np.int64),
    )

    assert report["available"] is True
    assert report["annualised_sharpe"] is not None
    assert report["annualised_sharpe_reason"] == ""
    assert report["sharpe_basis"] == ev.SHARPE_BASIS
