"""`microstructure_v1`: the 32 columns of docs/microstructure_v1.md, checked.

Three kinds of claim.

**The definitions are what the spec says.** Where a column has an arithmetic
definition it is checked against that arithmetic on hand-built rows, not against
another run of the same code. Where it has a declared range it is checked to
stay inside it on generated data.

**The trailing statistics are causal.** A window that is not full takes the
declared default; a full one equals the mean of exactly the rows before it; a
gap restarts both. The "large trade" threshold is the one quantile in the family
and it is checked to be a function of the previous week alone — the future-
quantile leak is the kind that raises nothing and improves everything.

**Nothing is ever NaN.** Which is not a nicety: a NaN would be dropped by the
dataset builder, every later row would shift, and the fold boundaries would land
on different candles for this arm than for the control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nn.microstructure import (
    MICROSTRUCTURE_FEATURE_COLUMNS,
    MICROSTRUCTURE_FEATURE_FAMILIES,
    MICROSTRUCTURE_SPEC_VERSION,
    MicrostructureError,
    MicrostructureSpec,
    compute_microstructure_features,
    microstructure_feature_columns,
    segment_ids,
)
from nn.research_contract import load_contract
from nn.trade_aggregates import PRICE_BINS, HourlyTradeAggregator
from tools.make_sample_data import generate_trades

STYX_NS = int(load_contract("btc-usdt-1h-gen1").sealed_test_start.value)

#: Short windows so a test can reach "the window is full" in a handful of rows.
#: Never reachable from an entrypoint — see `MicrostructureSpec.with_windows`.
SHORT = MicrostructureSpec().with_windows(w_short=3, w_long=4)

#: Every column whose spec declares a hard range, and that range.
BOUNDED: dict[str, tuple[float, float]] = {
    "ms_qty_imbalance": (-1.0, 1.0),
    "ms_notional_imbalance": (-1.0, 1.0),
    "ms_count_imbalance": (-1.0, 1.0),
    "ms_notional_imbalance_z24": (-5.0, 5.0),
    "ms_trade_count_ratio_24": (0.0, 10.0),
    "ms_trade_count_ratio_168": (0.0, 10.0),
    "ms_trade_count_accel": (-1.0, 5.0),
    "ms_mean_size_ratio_168": (0.0, 10.0),
    "ms_median_size_ratio_168": (0.0, 10.0),
    "ms_large_trade_count_share": (0.0, 1.0),
    "ms_large_trade_qty_share": (0.0, 1.0),
    "ms_active_second_share": (0.0, 1.0),
    "ms_max_second_share": (0.0, 1.0),
    "ms_max_silence_share": (0.0, 1.0),
    "ms_displacement_per_notional": (0.0, 20.0),
    "ms_range_per_notional": (0.0, 20.0),
    "ms_signed_flow_return": (-10.0, 10.0),
    "ms_absorption_like": (0.0, 1.0),
    "ms_volume_without_range": (0.0, 20.0),
    "ms_unconverted_pressure": (-2.0, 2.0),
    "ms_q2_trade_share": (0.0, 1.0),
    "ms_q3_trade_share": (0.0, 1.0),
    "ms_q4_trade_share": (0.0, 1.0),
    "ms_late_early_qty_imbalance": (-1.0, 1.0),
    "ms_q4_flow_imbalance": (-1.0, 1.0),
    "ms_price_hhi": (1.0 / PRICE_BINS, 1.0),
    "ms_price_entropy": (0.0, 1.0),
    "ms_close_from_vwap": (-1.0, 1.0),
    "ms_close_from_node": (-1.0, 1.0),
}


def aggregates(hours: int = 40, **kwargs) -> pd.DataFrame:
    ts, price, qty, maker, ids = generate_trades(hours=hours, **kwargs)
    aggregator = HourlyTradeAggregator(styx_ns=STYX_NS)
    aggregator.add_chunk(ts, price, qty, maker, ids)
    return aggregator.finish()


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return aggregates(hours=60, seed=31, trades_per_hour=20)


@pytest.fixture(scope="module")
def features(table) -> pd.DataFrame:
    return compute_microstructure_features(table, SHORT)


# --------------------------------------------------------------------------- #
# A. the column set
# --------------------------------------------------------------------------- #
def test_there_are_thirty_two_columns_in_the_declared_order():
    columns = microstructure_feature_columns()
    assert len(columns) == 32
    assert columns == list(MICROSTRUCTURE_FEATURE_COLUMNS)
    assert len(set(columns)) == 32
    assert all(name.startswith("ms_") for name in columns)


def test_the_families_partition_the_columns():
    covered = [c for group in MICROSTRUCTURE_FEATURE_FAMILIES.values() for c in group]
    assert sorted(covered) == sorted(MICROSTRUCTURE_FEATURE_COLUMNS)
    assert len(MICROSTRUCTURE_FEATURE_FAMILIES) == 8


def test_the_spec_hash_moves_with_the_constants_and_not_with_the_object():
    default = MicrostructureSpec()
    assert default.spec_hash() == MicrostructureSpec().spec_hash()
    assert default.spec_hash() != default.with_windows(w_short=24, w_long=169).spec_hash()
    assert MICROSTRUCTURE_SPEC_VERSION == "microstructure_v1"


def test_the_engine_names_a_missing_source_column_rather_than_raising_a_key_error(table):
    with pytest.raises(MicrostructureError, match="missing column"):
        compute_microstructure_features(table.drop(columns=["buy_qty"]), SHORT)


# --------------------------------------------------------------------------- #
# B. the definitions
# --------------------------------------------------------------------------- #
def test_every_column_is_present_finite_and_in_range(features):
    assert list(features.columns) == list(MICROSTRUCTURE_FEATURE_COLUMNS)
    values = features.to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()
    for name, (low, high) in BOUNDED.items():
        column = features[name].to_numpy(dtype=np.float64)
        assert column.min() >= low - 1e-12, f"{name} below {low}"
        assert column.max() <= high + 1e-12, f"{name} above {high}"


def test_the_flow_imbalances_are_the_arithmetic_the_spec_states(table, features):
    qty = table["qty"].to_numpy(dtype=np.float64)
    buy_qty = table["buy_qty"].to_numpy(dtype=np.float64)
    notional = table["notional"].to_numpy(dtype=np.float64)
    buy_notional = table["buy_notional"].to_numpy(dtype=np.float64)
    n = table["n_trades"].to_numpy(dtype=np.float64)
    nb = table["buy_trades"].to_numpy(dtype=np.float64)

    assert features["ms_qty_imbalance"].to_numpy() == pytest.approx((2 * buy_qty - qty) / qty)
    assert features["ms_notional_imbalance"].to_numpy() == pytest.approx(
        (2 * buy_notional - notional) / notional
    )
    assert features["ms_count_imbalance"].to_numpy() == pytest.approx((2 * nb - n) / n)


def test_the_intensity_and_burstiness_columns_match_their_definitions(table, features):
    n = table["n_trades"].to_numpy(dtype=np.float64)
    active = table["active_seconds"].to_numpy(dtype=np.float64)
    sq = table["second_count_sq_sum"].to_numpy(dtype=np.float64)

    assert features["ms_log_trade_count"].to_numpy() == pytest.approx(np.log1p(n))
    assert features["ms_trades_per_active_second"].to_numpy() == pytest.approx(
        np.log1p(n / active)
    )
    assert features["ms_second_dispersion"].to_numpy() == pytest.approx(
        np.log1p(np.maximum(sq / n - n / 3600.0, 0.0))
    )
    assert features["ms_active_second_share"].to_numpy() == pytest.approx(active / 3600.0)


def test_a_uniform_hour_has_zero_second_dispersion():
    """The Fano factor is 0 for perfectly regular arrival, by construction."""
    hour = int(pd.Timestamp("2022-01-01T00:00:00+00:00").value)
    seconds = np.arange(3600, dtype=np.int64)
    ts = hour + seconds * 10**9
    aggregator = HourlyTradeAggregator(styx_ns=STYX_NS)
    aggregator.add_chunk(
        ts,
        np.full(3600, 100.0),
        np.full(3600, 0.5),
        np.zeros(3600, dtype=bool),
        np.arange(1, 3601, dtype=np.int64),
    )
    row = compute_microstructure_features(aggregator.finish(), SHORT)
    assert row["ms_second_dispersion"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert row["ms_active_second_share"].iloc[0] == pytest.approx(1.0)
    assert row["ms_max_silence_share"].iloc[0] == 0.0


def test_the_volume_at_price_columns_describe_the_price_histogram(table, features):
    bins = table[[f"price_qty_{j}" for j in range(PRICE_BINS)]].to_numpy(dtype=np.float64)
    shares = bins / bins.sum(axis=1, keepdims=True)
    assert features["ms_price_hhi"].to_numpy() == pytest.approx((shares**2).sum(axis=1))
    entropy = -(shares * np.where(shares > 0, np.log(np.where(shares > 0, shares, 1)), 0)).sum(
        axis=1
    ) / np.log(PRICE_BINS)
    assert features["ms_price_entropy"].to_numpy() == pytest.approx(entropy)


def test_a_single_price_hour_is_maximally_concentrated():
    hour = int(pd.Timestamp("2022-02-02T00:00:00+00:00").value)
    aggregator = HourlyTradeAggregator(styx_ns=STYX_NS)
    aggregator.add_chunk(
        np.array([hour, hour + 10**9], dtype=np.int64),
        np.array([500.0, 500.0]),
        np.array([1.0, 1.0]),
        np.array([False, True]),
        np.array([1, 2], dtype=np.int64),
    )
    row = compute_microstructure_features(aggregator.finish(), SHORT)
    assert row["ms_price_hhi"].iloc[0] == pytest.approx(1.0)
    assert row["ms_price_entropy"].iloc[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# C. the trailing statistics are causal
# --------------------------------------------------------------------------- #
def test_a_window_that_is_not_full_takes_the_declared_default(table):
    features = compute_microstructure_features(table, SHORT)
    # w_short = 3: rows 0, 1 and 2 have no full 3-row history behind them.
    assert features["ms_trade_count_ratio_24"].iloc[:3].tolist() == [1.0, 1.0, 1.0]
    assert features["ms_notional_imbalance_z24"].iloc[:3].tolist() == [0.0, 0.0, 0.0]
    # w_long = 4.
    assert features["ms_trade_count_ratio_168"].iloc[:4].tolist() == [1.0] * 4
    assert features["ms_large_trade_count_share"].iloc[:4].tolist() == [0.0] * 4
    assert features["ms_signed_flow_return"].iloc[:4].tolist() == [0.0] * 4


def test_a_full_window_is_the_mean_of_exactly_the_rows_before_it(table):
    features = compute_microstructure_features(table, SHORT)
    n = table["n_trades"].to_numpy(dtype=np.float64)
    for row in (3, 10, 25):
        expected = min(n[row] / n[row - 3 : row].mean(), 10.0)
        assert features["ms_trade_count_ratio_24"].iloc[row] == pytest.approx(expected)


def test_a_gap_restarts_the_trailing_windows():
    """A missing hour is a new segment, and a segment starts from its defaults."""
    table = aggregates(hours=20, seed=41, trades_per_hour=9, skip_hours=(10,))
    segments = segment_ids(table)
    assert segments.max() == 1
    boundary = int(np.flatnonzero(segments == 1)[0])

    features = compute_microstructure_features(table, SHORT)
    after = features["ms_trade_count_ratio_24"].to_numpy()[boundary : boundary + 3]
    assert after.tolist() == [1.0, 1.0, 1.0]
    # And the row just before the gap did have a real window, so the defaults
    # above are the gap's doing rather than the whole column being constant.
    assert features["ms_trade_count_ratio_24"].iloc[boundary - 1] != 1.0


def test_a_gap_does_not_disturb_the_rows_before_it():
    whole = aggregates(hours=20, seed=41, trades_per_hour=9)
    holed = aggregates(hours=20, seed=41, trades_per_hour=9, skip_hours=(10,))
    before = compute_microstructure_features(whole, SHORT).iloc[:10]
    after = compute_microstructure_features(holed, SHORT).iloc[:10]
    pd.testing.assert_frame_equal(before, after)


def test_the_large_trade_threshold_ignores_the_present_and_the_future():
    """The one quantile in the family, and the leak it would be if it were global.

    One enormous trade is placed in the last hour. If the threshold were a
    quantile of the whole history — or even of the current row — that hour would
    define its own tail and the share would be measured against itself. Because
    the threshold comes from the previous ``w_long`` hours only, the giant trade
    lands above whatever the past called large, and none of the earlier rows can
    see it at all.
    """
    ts, price, qty, maker, ids = generate_trades(hours=12, seed=55, trades_per_hour=8)
    quiet = compute_microstructure_features(_fold(ts, price, qty, maker, ids), SHORT)

    loud_qty = qty.copy()
    loud_qty[-1] = 5_000.0
    loud = compute_microstructure_features(_fold(ts, price, loud_qty, maker, ids), SHORT)

    # Every hour before the last is untouched by a trade in the last hour.
    pd.testing.assert_frame_equal(quiet.iloc[:-1], loud.iloc[:-1])
    # And the last hour's own share reflects the giant trade against the past's
    # threshold rather than against its own.
    assert (
        loud["ms_large_trade_qty_share"].iloc[-1] > quiet["ms_large_trade_qty_share"].iloc[-1]
    )


def _fold(ts, price, qty, maker, ids) -> pd.DataFrame:
    aggregator = HourlyTradeAggregator(styx_ns=STYX_NS)
    aggregator.add_chunk(ts, price, qty, maker, ids)
    return aggregator.finish()


def test_the_large_trade_share_is_zero_when_the_past_saw_nothing_smaller():
    """A uniform history has its 99th percentile in the same bin as everything else."""
    hour = int(pd.Timestamp("2022-03-03T00:00:00+00:00").value)
    rows, per_hour = 10, 8
    ts = np.concatenate(
        [hour + h * 3_600_000_000_000 + np.arange(per_hour) * 10**9 for h in range(rows)]
    ).astype(np.int64)
    table = _fold(
        ts,
        np.full(len(ts), 100.0),
        np.full(len(ts), 0.05),
        np.zeros(len(ts), dtype=bool),
        np.arange(1, len(ts) + 1, dtype=np.int64),
    )
    features = compute_microstructure_features(table, SHORT)
    assert features["ms_large_trade_count_share"].to_numpy().max() == 0.0
