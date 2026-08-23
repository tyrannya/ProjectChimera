"""The P3 trade source: what an hour's row means, and what the verifier refuses.

Three kinds of claim, checked three ways.

**An hour's row is a function of that hour's trades.** The aggregation is tested
against hand-built trades whose answer can be computed by hand, not against
another implementation of itself. Where a property is claimed — order
invariance within an hour, the half-open boundary, the aggressor reading — it is
exercised rather than described.

**The stream rules fail closed.** A duplicated trade id, a regressed timestamp, a
non-positive price and a sealed timestamp are each refused, and the seal
rejection is checked for what it does *not* say: a verifier that printed the
offending row would leak the rows it exists to keep sealed.

**The verifier is pinned by adversarial use.** A checker that only ever sees
good input proves nothing — a function returning no problems would pass — so
every rejection it claims is exercised against a snapshot built to pass and then
broken in exactly one way. Each is paired with the positive control that the
unbroken snapshot verifies, because a guard that always failed would satisfy the
corruption test on its own.

No network, no fit, and no committed evidence is written. One test reads the
committed trade snapshot — the rest exercise the verifier against a synthetic
tree instead.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nn.data_fingerprint import fingerprint_trade_aggregates
from nn.research_contract import load_contract
from nn.trade_aggregates import (
    AGGREGATE_COLUMNS,
    HOUR_NS,
    PRICE_BINS,
    SIZE_BINS,
    HourlyTradeAggregator,
    StyxBreach,
    TradeAggregationError,
    aggregate_hour,
    aggregation_spec,
    aggregation_spec_hash,
    check_table,
    coverage_report,
    resolve_epoch_unit,
    scale_to_nanoseconds,
)
from tools.export_trade_snapshot import (
    MANIFEST_NAME,
    _has_header,
    _sha256,
    plan_archives,
)
from tools.make_sample_data import generate_trades
from tools.verify_trade_snapshot import (
    DEFAULT_MANIFEST,
    PRICE_RTOL,
    QUANTITY_RTOL,
    TradeSnapshotVerificationError,
    verify_trade_snapshot,
)

BTC = load_contract("btc-usdt-1h-gen1")
STYX = BTC.sealed_test_start
STYX_NS = int(STYX.value)
HOUR = int(pd.Timestamp("2021-03-04T05:00:00+00:00").value)


def fold(ts, price, qty, maker, ids, *, styx_ns: int = STYX_NS) -> HourlyTradeAggregator:
    aggregator = HourlyTradeAggregator(styx_ns=styx_ns)
    aggregator.add_chunk(
        np.asarray(ts, dtype=np.int64),
        np.asarray(price, dtype=np.float64),
        np.asarray(qty, dtype=np.float64),
        np.asarray(maker, dtype=bool),
        np.asarray(ids, dtype=np.int64),
    )
    return aggregator


# --------------------------------------------------------------------------- #
# A. an hour's row is a function of that hour's trades
# --------------------------------------------------------------------------- #
def test_an_hours_row_is_the_arithmetic_of_its_trades():
    """Four trades whose aggregate can be worked out on paper."""
    ts = [HOUR, HOUR + 60 * 10**9, HOUR + 1800 * 10**9, HOUR + 3599 * 10**9]
    price = [100.0, 110.0, 90.0, 105.0]
    qty = [1.0, 2.0, 3.0, 4.0]
    # is_buyer_maker: True is an aggressive SELL, False an aggressive BUY.
    maker = [False, True, False, True]
    row = aggregate_hour(HOUR, np.array(ts), np.array(price), np.array(qty), np.array(maker))

    assert row["n_trades"] == 4
    assert row["buy_trades"] == 2
    assert row["qty"] == pytest.approx(10.0)
    assert row["buy_qty"] == pytest.approx(1.0 + 3.0)
    assert row["notional"] == pytest.approx(100 + 220 + 270 + 420)
    assert row["buy_notional"] == pytest.approx(100 + 270)
    assert row["first_price"] == 100.0
    assert row["last_price"] == 105.0
    assert row["high_price"] == 110.0
    assert row["low_price"] == 90.0
    assert row["active_seconds"] == 4
    assert row["max_second_trades"] == 1
    assert row["second_count_sq_sum"] == 4
    # Quarters are 900 seconds each: 0s and 60s in the first, 1800s in the
    # third, 3599s in the fourth.
    assert [row[f"q{k}_trades"] for k in range(4)] == [2, 0, 1, 1]
    assert row["q0_qty"] == pytest.approx(3.0)
    assert row["q0_buy_qty"] == pytest.approx(1.0)


def test_the_hour_boundary_is_half_open():
    """`t + 1h - 1ns` is the last nanosecond of hour `t`; `t + 1h` is the next hour."""
    ts = [HOUR, HOUR + HOUR_NS - 1, HOUR + HOUR_NS]
    aggregator = fold(ts, [10.0, 11.0, 12.0], [1.0, 1.0, 1.0], [False] * 3, [1, 2, 3])
    frame = aggregator.finish()
    assert len(frame) == 2
    assert frame["n_trades"].tolist() == [2, 1]
    assert frame["last_price"].iloc[0] == 11.0
    assert frame["first_price"].iloc[1] == 12.0


def test_only_first_and_last_price_depend_on_order_within_an_hour():
    """Everything else is a sum, a count or a histogram, and cannot see the order."""
    ts, price, qty, maker, _ = generate_trades(hours=1, seed=3, trades_per_hour=40)
    hour = int(ts[0] // HOUR_NS * HOUR_NS)
    straight = aggregate_hour(hour, ts, price, qty, maker)
    order = np.random.default_rng(0).permutation(len(ts))
    shuffled = aggregate_hour(hour, ts[order], price[order], qty[order], maker[order])

    order_dependent = {"first_price", "last_price"}
    for name in AGGREGATE_COLUMNS:
        if name in order_dependent:
            continue
        assert straight[name] == pytest.approx(shuffled[name]), name
    assert shuffled["first_price"] == price[order][0]


def test_the_aggressor_flag_decides_which_side_is_a_buy():
    all_maker = aggregate_hour(
        HOUR, np.array([HOUR]), np.array([10.0]), np.array([2.0]), np.array([True])
    )
    all_taker = aggregate_hour(
        HOUR, np.array([HOUR]), np.array([10.0]), np.array([2.0]), np.array([False])
    )
    assert all_maker["buy_trades"] == 0 and all_maker["buy_qty"] == 0.0
    assert all_taker["buy_trades"] == 1 and all_taker["buy_qty"] == pytest.approx(2.0)
    assert "is_buyer_maker=false is an aggressive BUY" in aggregation_spec()["aggressor_rule"]


def test_the_histograms_account_for_every_trade():
    ts, price, qty, maker, ids = generate_trades(hours=6, seed=8, trades_per_hour=30)
    frame = fold(ts, price, qty, maker, ids).finish()
    sizes = frame[[f"size_count_{b}" for b in range(SIZE_BINS)]].to_numpy().sum(axis=1)
    volumes = frame[[f"size_qty_{b}" for b in range(SIZE_BINS)]].to_numpy().sum(axis=1)
    prices = frame[[f"price_qty_{j}" for j in range(PRICE_BINS)]].to_numpy().sum(axis=1)
    assert sizes.tolist() == frame["n_trades"].tolist()
    assert volumes == pytest.approx(frame["qty"].to_numpy())
    assert prices == pytest.approx(frame["qty"].to_numpy())


def test_a_silent_stretch_is_measured_as_an_empty_run():
    """Two trades an hour apart leave one long silence between them."""
    ts = [HOUR, HOUR + 3000 * 10**9]
    row = aggregate_hour(
        HOUR, np.array(ts), np.array([10.0, 10.0]), np.array([1.0, 1.0]), np.array([False] * 2)
    )
    # Seconds 1..2999 are silent, and so are 3001..3599; the longer run wins.
    assert row["max_empty_run_seconds"] == 2999
    assert row["active_seconds"] == 2


def test_a_degenerate_hour_puts_all_its_volume_in_one_price_bin():
    row = aggregate_hour(
        HOUR,
        np.array([HOUR, HOUR + 10**9]),
        np.array([50.0, 50.0]),
        np.array([1.0, 3.0]),
        np.array([False, True]),
    )
    assert row["price_qty_0"] == pytest.approx(4.0)
    assert sum(row[f"price_qty_{j}"] for j in range(1, PRICE_BINS)) == 0.0


# --------------------------------------------------------------------------- #
# B. the stream rules fail closed
# --------------------------------------------------------------------------- #
def test_a_duplicated_trade_is_refused():
    ts, price, qty, maker, ids = generate_trades(hours=2, seed=4, trades_per_hour=10)
    ids = ids.copy()
    ids[7] = ids[6]
    with pytest.raises(TradeAggregationError, match="strictly increase"):
        fold(ts, price, qty, maker, ids)


def test_an_out_of_order_timestamp_is_refused():
    ts, price, qty, maker, ids = generate_trades(hours=2, seed=4, trades_per_hour=10)
    ts = ts.copy()
    ts[5] = ts[4] - 10**9
    with pytest.raises(TradeAggregationError, match="regression"):
        fold(ts, price, qty, maker, ids)


@pytest.mark.parametrize("column,value", [("price", 0.0), ("price", -1.0), ("qty", 0.0)])
def test_a_nonpositive_price_or_quantity_is_refused(column, value):
    ts, price, qty, maker, ids = generate_trades(hours=1, seed=6, trades_per_hour=8)
    price, qty = price.copy(), qty.copy()
    {"price": price, "qty": qty}[column][3] = value
    with pytest.raises(TradeAggregationError, match="non-positive"):
        fold(ts, price, qty, maker, ids)


def test_a_sealed_trade_is_refused_and_never_described():
    """The one rejection whose *content* must not reach the operator."""
    marker_price = 123456.789
    marker_qty = 98765.4321
    ts = np.array([STYX_NS - HOUR_NS, STYX_NS], dtype=np.int64)
    with pytest.raises(StyxBreach) as excinfo:
        fold(
            ts,
            np.array([10.0, marker_price]),
            np.array([1.0, marker_qty]),
            np.array([False, False]),
            np.array([1, 2]),
        )
    message = str(excinfo.value)
    assert "1 of 2 trades" in message
    assert STYX.isoformat() in message
    assert "123456" not in message and "98765" not in message


def test_a_trade_stream_that_stops_before_the_seal_is_accepted():
    """The positive control: the guard above is not simply always-fail."""
    ts = np.array([STYX_NS - 2 * HOUR_NS, STYX_NS - 1], dtype=np.int64)
    frame = fold(
        ts,
        np.array([10.0, 11.0]),
        np.array([1.0, 1.0]),
        np.array([False, True]),
        np.array([1, 2]),
    ).finish()
    assert len(frame) == 2
    check_table(frame, styx=STYX)


def test_the_stream_counters_describe_what_was_seen():
    ts, price, qty, maker, ids = generate_trades(hours=5, seed=9, trades_per_hour=12)
    aggregator = fold(ts, price, qty, maker, ids)
    aggregator.finish()
    stats = aggregator.stats.to_dict()
    assert stats["trades"] == len(ts)
    assert stats["buy_trades"] == int((~maker).sum())
    assert stats["duplicate_ids"] == 0
    assert stats["timestamp_regressions"] == 0
    assert stats["strictly_sorted"] is True
    assert stats["first_trade_id"] == int(ids[0])
    assert stats["last_trade_id"] == int(ids[-1])


# --------------------------------------------------------------------------- #
# C. epoch units resolve per archive, or the archive is refused
# --------------------------------------------------------------------------- #
def test_milliseconds_and_microseconds_reach_the_same_instants():
    """The unit changed mid-history; the canonical nanoseconds must not."""
    start = pd.Timestamp("2025-03-01T00:00:00+00:00")
    end = pd.Timestamp("2025-04-01T00:00:00+00:00")
    instant = pd.Timestamp("2025-03-15T12:34:56+00:00")
    as_ms = int(instant.value // 1_000_000)
    as_us = int(instant.value // 1_000)

    assert resolve_epoch_unit(as_ms, as_ms, period_start=start, period_end=end) == "ms"
    assert resolve_epoch_unit(as_us, as_us, period_start=start, period_end=end) == "us"
    assert as_ms * 1_000_000 == as_us * 1_000 == instant.value


def test_an_archive_that_fits_no_unit_is_refused():
    start = pd.Timestamp("2020-01-01T00:00:00+00:00")
    end = pd.Timestamp("2020-02-01T00:00:00+00:00")
    stray = int(pd.Timestamp("2023-06-01T00:00:00+00:00").value // 1_000_000)
    with pytest.raises(TradeAggregationError, match="no supported epoch unit"):
        resolve_epoch_unit(stray, stray, period_start=start, period_end=end)


def test_an_ambiguous_period_is_refused_rather_than_guessed():
    """A window wide enough for two readings is a bug in the bounds, not a coin flip."""
    value = int(pd.Timestamp("2021-01-01T00:00:00+00:00").value // 1_000_000)
    with pytest.raises(TradeAggregationError, match="ambiguous"):
        resolve_epoch_unit(
            value,
            value,
            period_start=pd.Timestamp("1970-01-01T00:00:00+00:00"),
            period_end=pd.Timestamp("2100-01-01T00:00:00+00:00"),
        )


def test_a_csv_header_is_recognised_and_a_trade_row_is_not():
    assert _has_header("agg_trade_id,price,quantity,first_trade_id,last,transact,is,best")
    assert not _has_header("1234,50000.10,0.5,11,12,1600000000000,true,true")


# --------------------------------------------------------------------------- #
# D. the acquisition plan is bounded before the seal
# --------------------------------------------------------------------------- #
def test_the_plan_uses_whole_months_then_days_and_stops_where_told():
    archives = plan_archives(
        pd.Timestamp("2020-01-01T00:00:00+00:00"), pd.Timestamp("2020-03-04T00:00:00+00:00")
    )
    kinds = [a.kind for a in archives]
    assert kinds == ["monthly", "monthly", "daily", "daily", "daily"]
    assert archives[0].url.endswith("BTCUSDT-aggTrades-2020-01.zip")
    assert archives[-1].url.endswith("BTCUSDT-aggTrades-2020-03-03.zip")
    assert all(a.period_end <= pd.Timestamp("2020-03-04T00:00:00+00:00") for a in archives)


def test_the_real_acquisition_window_warms_up_and_stops_well_short_of_styx():
    """The window the committed BTC snapshot actually implies. No network needed.

    Three claims at once, all against real committed data: the acquisition ends
    at the hour after the spine's last candle (so the last scored row has its own
    hour of trades), it stops over three months before the seal (so the bound is
    a property of the plan rather than of a filter), and it starts far enough
    back that `microstructure_v1`'s 168-hour window is full at the first row that
    is ever scored.
    """
    from pathlib import Path as _Path

    from tools.export_trade_snapshot import MIN_WARMUP_HOURS, acquisition_window

    root = _Path(__file__).resolve().parents[1]
    ohlcv = root / "data" / "research" / "btc_usdt_1h_gen1_snapshot_manifest.json"
    start, end, source = acquisition_window(ohlcv, BTC)

    spine_start = pd.Timestamp(source["spine_start"])
    spine_end = pd.Timestamp(source["spine_end"])
    assert end == spine_end + pd.Timedelta(hours=1)
    assert end < STYX
    assert (STYX - end) > pd.Timedelta(days=90)
    assert (spine_start - start) / pd.Timedelta(hours=1) >= MIN_WARMUP_HOURS
    assert source["warmup_hours_requested"] >= MIN_WARMUP_HOURS

    archives = plan_archives(start, end)
    assert len(archives) > 60
    # The last daily archive covers its whole UTC day while research needs only
    # part of it, so the plan overshoots `end` by less than a day and the tail is
    # dropped during folding. That is a bounded overshoot three months short of
    # the seal, not a "fetch past it and filter later".
    furthest = max(a.period_end for a in archives)
    assert furthest - end < pd.Timedelta(days=1)
    assert furthest < STYX
    assert (STYX - furthest) > pd.Timedelta(days=90)
    assert all(a.period_start >= start for a in archives)


def test_every_planned_archive_ends_before_the_seal():
    """The bound is on what is *requested*, not on what is filtered afterwards."""
    archives = plan_archives(
        pd.Timestamp("2025-04-01T00:00:00+00:00"), pd.Timestamp("2025-05-19T08:00:00+00:00")
    )
    assert archives
    assert max(a.period_end for a in archives) < STYX


# --------------------------------------------------------------------------- #
# E. the table's own validity rules
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_table():
    ts, price, qty, maker, ids = generate_trades(hours=12, seed=17, trades_per_hour=16)
    return fold(ts, price, qty, maker, ids).finish()


def test_check_table_accepts_a_generated_table(small_table):
    """The positive control for every rejection below."""
    check_table(small_table, styx=STYX)


@pytest.mark.parametrize(
    "break_it,match",
    [
        (lambda f: f.assign(n_trades=f["n_trades"].mask(f.index == 2, 0)), "no trade"),
        (lambda f: f.iloc[::-1], "strictly increasing"),
        (
            lambda f: f.assign(date=f["date"] + pd.Timedelta(minutes=7)),
            "hour boundary",
        ),
        (lambda f: f.assign(buy_trades=f["n_trades"] + 1), "more taker buys than trades"),
        (lambda f: f.assign(buy_qty=f["qty"] * 2), "share of"),
        (lambda f: f.assign(high_price=f["low_price"] - 1.0), "high below its low"),
        (lambda f: f.assign(active_seconds=0), "active_seconds"),
        (lambda f: f.assign(q0_trades=f["q0_trades"] + 1), "quarter trade counts"),
        (lambda f: f.assign(size_count_0=f["size_count_0"] + 1), "size histogram"),
    ],
)
def test_check_table_refuses_a_broken_table(small_table, break_it, match):
    with pytest.raises(TradeAggregationError, match=match):
        check_table(break_it(small_table.copy()), styx=STYX)


def test_check_table_refuses_a_sealed_hour_without_describing_it(small_table):
    moved = small_table.copy()
    moved["date"] = moved["date"] + (STYX - moved["date"].iloc[0])
    with pytest.raises(StyxBreach, match="withheld"):
        check_table(moved, styx=STYX)


def test_a_missing_hour_is_reported_as_a_gap():
    ts, price, qty, maker, ids = generate_trades(
        hours=10, seed=21, trades_per_hour=5, skip_hours=(4, 5)
    )
    frame = fold(ts, price, qty, maker, ids).finish()
    report = coverage_report(frame)
    assert report["hours_present"] == 8
    assert report["hours_missing"] == 2
    assert len(report["missing_intervals"]) == 1
    assert report["missing_intervals"][0]["hours"] == 2


# --------------------------------------------------------------------------- #
# F. semantic identity
# --------------------------------------------------------------------------- #
def fingerprint(frame) -> str:
    return fingerprint_trade_aggregates(
        frame,
        exchange="binance",
        symbol="BTCUSDT",
        market_type="spot",
        source_type="aggTrades",
        aggregation_spec_hash=aggregation_spec_hash(),
    ).aggregate_hash


def test_two_readings_of_the_same_table_agree(small_table):
    assert fingerprint(small_table) == fingerprint(small_table.copy())


def test_editing_one_aggregate_value_changes_the_semantic_hash(small_table):
    """The falsification: an identity that survived an edit would identify nothing."""
    edited = small_table.copy()
    edited.loc[3, "buy_qty"] = float(edited.loc[3, "buy_qty"]) * 0.5
    assert fingerprint(edited) != fingerprint(small_table)


def test_the_semantic_hash_ignores_the_stored_timestamp_unit(small_table):
    """Binance changed epoch unit mid-history; the identity must not notice.

    The columns are canonicalised to int64 UTC nanoseconds before they are
    hashed, so a table whose `date` column is stored in microseconds — which is
    what a round trip through some Parquet writers produces — is the same
    research input as one stored in nanoseconds.
    """
    microseconds = small_table.copy()
    microseconds["date"] = microseconds["date"].astype("datetime64[us, UTC]")
    assert microseconds["date"].dtype != small_table["date"].dtype
    assert fingerprint(microseconds) == fingerprint(small_table)


def test_the_semantic_hash_depends_on_the_aggregation_spec(small_table):
    same_numbers = fingerprint_trade_aggregates(
        small_table,
        exchange="binance",
        symbol="BTCUSDT",
        market_type="spot",
        source_type="aggTrades",
        aggregation_spec_hash="0" * 64,
    ).aggregate_hash
    assert same_numbers != fingerprint(small_table)


def test_the_semantic_hash_depends_on_the_source_identity(small_table):
    other_venue = fingerprint_trade_aggregates(
        small_table,
        exchange="binance",
        symbol="BTCUSDT",
        market_type="futures",
        source_type="aggTrades",
        aggregation_spec_hash=aggregation_spec_hash(),
    ).aggregate_hash
    assert other_venue != fingerprint(small_table)


# --------------------------------------------------------------------------- #
# G. the verifier, pinned by adversarial use
# --------------------------------------------------------------------------- #
@pytest.fixture
def tree(synthetic_trade_snapshot, tmp_path):
    """A writable copy of the fixture snapshot, so a test can break one thing."""
    destination = tmp_path / "tree"
    shutil.copytree(synthetic_trade_snapshot["root"], destination)
    return destination / "data" / "research" / MANIFEST_NAME


def edit(manifest_path, mutate):
    payload = json.loads(manifest_path.read_text())
    mutate(payload)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


def test_the_fixture_snapshot_verifies(tree):
    """The positive control. Every rejection below is a change from this state."""
    results = verify_trade_snapshot(tree)
    assert len(results) >= 20
    names = {result.name for result in results}
    assert {"aggregate_before_styx", "aggregate_semantic_hash", "spine_coverage"} <= names


@pytest.mark.parametrize(
    "mutate,check",
    [
        (lambda p: p.__setitem__("snapshot_schema", "other/1"), "snapshot_schema"),
        (lambda p: p.pop("coverage"), "manifest_keys"),
        (lambda p: p.__setitem__("contains_styx", True), "contains_styx"),
        (
            lambda p: p["safety_checks"].__setitem__("styx_rows_exported", 1),
            "styx_rows_exported",
        ),
        (
            lambda p: p["safety_checks"].__setitem__("spine_fully_covered", False),
            "safety_check_flags",
        ),
        (lambda p: p["contract"].__setitem__("contract_hash", "0" * 64), "contract_hash"),
        (lambda p: p.__setitem__("styx_instant", "2020-01-01T00:00:00+00:00"), "styx_instant"),
        (lambda p: p["source"].__setitem__("venue", "kraken"), "source_identity"),
        (lambda p: p["source"].__setitem__("source_type", "trades"), "source_identity"),
        (
            lambda p: p["source"].__setitem__("identity_column", "trade_id"),
            "source_identity",
        ),
        (
            lambda p: p["aggregation"].__setitem__("aggregation_spec_hash", "0" * 64),
            "aggregation_spec_hash",
        ),
        (lambda p: p["aggregate"].__setitem__("sha256", "0" * 64), "aggregate_sha256"),
        (lambda p: p["aggregate"].__setitem__("rows", 3), "aggregate_rows"),
        (
            lambda p: p["aggregate"].__setitem__("start", "1999-01-01T00:00:00+00:00"),
            "aggregate_span",
        ),
        (
            lambda p: p["aggregate"].__setitem__("semantic_hash", "0" * 64),
            "aggregate_semantic_hash",
        ),
        (lambda p: p["aggregate"].__setitem__("columns", ["date"]), "aggregate_column_order"),
        (lambda p: p["trade_stream"].__setitem__("duplicate_ids", 2), "stream_integrity"),
        (
            lambda p: p["trade_stream"].__setitem__("strictly_sorted", False),
            "stream_integrity",
        ),
        (lambda p: p["trade_stream"].__setitem__("trades", 7), "stream_trade_count"),
        (lambda p: p["acquisition"].__setitem__("archives", []), "archives_declared"),
        (
            lambda p: p["acquisition"]["archives"][0].__setitem__("sha256", "nothex"),
            "archives_declared",
        ),
        (lambda p: p["acquisition"].__setitem__("archive_count", 99), "archives_declared"),
        (lambda p: p["coverage"].__setitem__("hours_missing", 12), "coverage_claims"),
        (
            lambda p: p["alignment"].__setitem__(
                "ohlcv_processed_semantic_prefix_hash", "0" * 64
            ),
            "ohlcv_binding",
        ),
        (lambda p: p["alignment"].__setitem__("spine_rows", 5), "spine_coverage"),
        (lambda p: p["alignment"].__setitem__("spine_hours_missing", 3), "spine_coverage"),
    ],
)
def test_a_broken_manifest_is_refused_by_the_named_check(tree, mutate, check):
    edit(tree, mutate)
    with pytest.raises(TradeSnapshotVerificationError) as excinfo:
        verify_trade_snapshot(tree)
    assert excinfo.value.check == check


def test_a_missing_aggregate_file_is_refused(tree):
    payload = json.loads(tree.read_text())
    (tree.parents[2] / payload["aggregate"]["path"]).unlink()
    with pytest.raises(TradeSnapshotVerificationError) as excinfo:
        verify_trade_snapshot(tree)
    assert excinfo.value.check == "files_present"


def test_an_edited_aggregate_value_is_refused_even_with_a_rewritten_sha(tree):
    """The semantic hash is the check a re-hashed file cannot slip past."""
    payload = json.loads(tree.read_text())
    path = tree.parents[2] / payload["aggregate"]["path"]
    frame = pd.read_parquet(path)
    frame.loc[4, "buy_notional"] = float(frame.loc[4, "buy_notional"]) * 0.5
    frame.to_parquet(path, index=False)

    import hashlib

    payload["aggregate"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(TradeSnapshotVerificationError) as excinfo:
        verify_trade_snapshot(tree)
    assert excinfo.value.check == "aggregate_semantic_hash"


def test_a_sealed_hour_is_refused_before_any_digest_is_recomputed(tree):
    """Order matters: a seal breach is reported as one, not as a hash mismatch."""
    payload = json.loads(tree.read_text())
    path = tree.parents[2] / payload["aggregate"]["path"]
    frame = pd.read_parquet(path)
    frame["date"] = frame["date"] + (STYX - pd.Timestamp(frame["date"].iloc[-1]))
    frame.to_parquet(path, index=False)

    import hashlib

    payload["aggregate"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(TradeSnapshotVerificationError) as excinfo:
        verify_trade_snapshot(tree)
    assert excinfo.value.check == "aggregate_before_styx"
    assert "withheld" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# The two sources, compared by value
#
# Everything above binds the trade table to the candles by *identity* — the same
# contract, the same manifest hash, the same hours — and none of it looks at a
# price. A table aggregated one hour out of step, or read under the wrong epoch
# unit, or built from a different symbol, satisfies every one of those bindings.
# --------------------------------------------------------------------------- #
def _shift_the_aggregate(tree: Path, hours: int) -> None:
    """Move every aggregate row's hour without touching a value."""
    payload = json.loads(tree.read_text())
    root = tree.parents[2]
    path = root / payload["aggregate"]["path"]
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True) + pd.Timedelta(hours=hours)
    frame.to_parquet(path, index=False)


def test_an_aggregate_shifted_by_one_hour_is_refused(tree):
    """Every row internally perfect; every row describing the wrong hour."""
    _shift_the_aggregate(tree, hours=1)
    with pytest.raises(TradeSnapshotVerificationError) as excinfo:
        verify_trade_snapshot(tree)
    # The shift breaks a hash first, which is the correct order; what this pins
    # is that it is refused at all, and `cross_source_values` below is what
    # catches the case where the hashes were rewritten to match.
    assert excinfo.value.check in {
        "aggregate_sha256",
        "aggregate_span",
        "aggregate_semantic_hash",
        "spine_coverage",
        "cross_source_values",
    }


def _rescale_every_price(frame, factor):
    """A table describing a different instrument: internally perfect, wrong market."""
    for column in ("first_price", "last_price", "high_price", "low_price"):
        frame[column] = frame[column] * factor
    return frame


def _rescale_quantity(frame, factor):
    """A table missing part of its trade stream: prices right, volume wrong."""
    frame["qty"] = frame["qty"] * factor
    return frame


def _roll_the_prices(frame, _factor):
    """Every hour's prices taken from the hour before it."""
    for column in ("first_price", "last_price", "high_price", "low_price"):
        frame[column] = frame[column].shift(1).bfill()
    return frame


@pytest.mark.parametrize(
    "mutate,factor",
    [
        (_rescale_every_price, 1.02),
        (_rescale_quantity, 1.5),
        (_roll_the_prices, None),
    ],
)
def test_an_aggregation_of_another_market_is_refused_by_value(tree, mutate, factor):
    """Internally consistent, correctly hashed, and describing something else.

    Each mutation leaves the table passing `check_table` — prices stay ordered,
    counts stay counts — and every hash is rewritten to match, so nothing above
    `cross_source_values` has anything to object to.
    """
    payload = json.loads(tree.read_text())
    root = tree.parents[2]
    path = root / payload["aggregate"]["path"]
    frame = mutate(pd.read_parquet(path), factor)
    frame.to_parquet(path, index=False)

    payload["aggregate"]["sha256"] = _sha256(path)
    payload["aggregate"]["semantic_hash"] = fingerprint_trade_aggregates(
        frame,
        exchange=payload["source"]["venue"],
        market_type=payload["source"]["market_type"],
        symbol=payload["source"]["symbol"],
        source_type=payload["source"]["source_type"],
        aggregation_spec_hash=aggregation_spec_hash(),
    ).aggregate_hash
    tree.write_text(json.dumps(payload, indent=2))

    with pytest.raises(TradeSnapshotVerificationError) as excinfo:
        verify_trade_snapshot(tree)
    assert excinfo.value.check == "cross_source_values"


def test_the_unbroken_fixture_passes_the_value_comparison(tree):
    """Falsifies the four above: the comparison is not simply always-refuse."""
    results = verify_trade_snapshot(tree)
    values = next(r for r in results if r.name == "cross_source_values")
    assert "hours agree with the candle history" in values.detail


def test_the_tolerances_are_measured_rather_than_generous():
    """A tolerance wide enough to admit a corrupt source is not a tolerance.

    One basis point on price and ten on quantity: two orders of magnitude
    looser than the worst disagreement observed on the committed source, and
    four orders tighter than anything a real corruption produces.
    """
    assert PRICE_RTOL == 1e-4
    assert QUANTITY_RTOL == 1e-3


# --------------------------------------------------------------------------- #
# Malformed epoch input is this module's own refusal, not pandas' bounds error
# --------------------------------------------------------------------------- #
def test_an_archive_mixing_two_epoch_units_raises_a_domain_error():
    """The reported span used to be formatted with `pd.Timestamp` unconditionally.

    One candidate unit is always wrong by three or six orders of magnitude —
    that is what the report is *for* — so a nanosecond value rendered as
    milliseconds raised `OutOfBoundsDatetime` from inside the error path, and a
    malformed archive surfaced as a pandas bounds error about a value the caller
    never supplied.
    """
    start = pd.Timestamp("2025-05-01", tz="UTC")
    end = pd.Timestamp("2025-06-01", tz="UTC")
    with pytest.raises(TradeAggregationError, match="no supported epoch unit"):
        resolve_epoch_unit(
            int(start.value) // 1_000_000,  # milliseconds
            int(end.value) - 1,  # nanoseconds
            period_start=start,
            period_end=end,
        )


def test_the_domain_error_says_which_values_are_not_instants():
    start = pd.Timestamp("2025-05-01", tz="UTC")
    end = pd.Timestamp("2025-06-01", tz="UTC")
    with pytest.raises(TradeAggregationError) as excinfo:
        resolve_epoch_unit(
            int(start.value) // 1_000_000,
            int(end.value) - 1,
            period_start=start,
            period_end=end,
        )
    assert "not a representable instant" in str(excinfo.value)


def test_scaling_to_nanoseconds_fails_closed_instead_of_wrapping():
    """`raw * scale` on int64 is silent modular arithmetic.

    Nanosecond values read as milliseconds wrap to arbitrary instants, some of
    which land back inside the archive's declared period and would then pass the
    range check the exporter applies afterwards.
    """
    nanoseconds = np.array([int(pd.Timestamp("2025-05-01", tz="UTC").value)], dtype=np.int64)
    with pytest.raises(TradeAggregationError, match="does not fit in int64"):
        scale_to_nanoseconds(nanoseconds, "ms")


def test_scaling_to_nanoseconds_accepts_a_unit_that_fits():
    milliseconds = np.array(
        [int(pd.Timestamp("2025-05-01", tz="UTC").value) // 1_000_000], dtype=np.int64
    )
    scaled = scale_to_nanoseconds(milliseconds, "ms")
    assert int(scaled[0]) == int(pd.Timestamp("2025-05-01", tz="UTC").value)


def test_an_unsupported_unit_name_is_a_domain_error():
    with pytest.raises(TradeAggregationError, match="not a supported epoch unit"):
        scale_to_nanoseconds(np.array([1], dtype=np.int64), "s")


def test_the_committed_trade_snapshot_is_present_and_verifies():
    """P3's real source, checked the same way a fresh clone would check it.

    Every fail-closed rejection above is exercised against a synthetic tree so
    it never has to touch what is actually committed. This is the one test
    that does: it calls the same `verify_trade_snapshot` against its own
    default manifest path — `data/research/btc_usdt_1h_gen1_trade_snapshot_manifest.json`
    — rather than reimplementing any part of the check here. A manifest that
    went missing, or a committed snapshot that drifted from what it claims,
    fails this the same way it would fail a fresh clone.
    """
    assert (
        DEFAULT_MANIFEST.is_file()
    ), f"no committed trade snapshot manifest at {DEFAULT_MANIFEST}"
    results = verify_trade_snapshot()
    assert len(results) == 28
    values = next(r for r in results if r.name == "cross_source_values")
    assert "47134 hours" in values.detail
