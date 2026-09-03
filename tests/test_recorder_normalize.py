"""The minute normalizer: one row per minute the exchange printed, and no others.

The invariant this file is written around is a single sentence — **missing data
is missing** — and it is asserted from every side that could quietly break it: a
minute with no closed kline has no row and appears in the gap list; a minute
whose mark or book stream was silent has a row whose flag is false and whose
columns are null rather than the previous minute's values; no stream ever
substitutes for another; and the source of the module is scanned for the fill
and interpolate calls that would make any of that untrue.

The second property is determinism. The same raw files produce the same table,
the same value digest and the same metadata whether the exchange delivered the
frames in order or not, whether a late file exists or not, and whether the
Parquet was written once or read back and written again. Where byte identity is
promised — the container written twice by one build — it is asserted; where only
value identity can honestly be promised, that is what is asserted, and the
digest is defined over values for exactly that reason.

Nothing here computes a return, a signal, a funding flow, a basis or a PnL. The
normalizer carries published numbers across and nothing else, and
``test_the_normalizer_computes_nothing`` says so about its source.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.events import (
    MINUTES_PER_DAY,
    NS_PER_MILLISECOND,
    UM_BOOK_TICKER,
    UM_FUNDING,
    UM_KLINE_1M,
    UM_MARK_PRICE,
    EventSource,
    TimeBasis,
)
from chimera.recorder.normalize import (
    CLOCK,
    MARKET_COLUMNS,
    NORMALIZED_META_SCHEMA,
    NORMALIZED_SCHEMA,
    MinuteNormalizer,
    RecorderNormalizeError,
    build_minutes,
    columns_for,
    digest,
    gaps_of,
    meta,
    minute_frame,
)
from chimera.recorder.sink import RawSink

from tests.recorder_synthetic import (
    DAY,
    NEXT_DAY,
    book_event,
    day_ms,
    funding_day,
    funding_event,
    kline_event,
    mark_event,
    minute_ms,
    spot_day,
    um_day,
)

CONTRACT = load_recorder_contract()
NORMALIZE_SOURCE = Path(
    __import__("chimera.recorder.normalize", fromlist=["__file__"]).__file__
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return CONTRACT.storage_root(tmp_path / "data")


def write(root: Path, events_by_stream: dict[str, list]) -> None:
    """Put synthetic observations on disk through the real sink."""
    for stream, events in events_by_stream.items():
        with RawSink(root, stream, contract=CONTRACT) as sink:
            for event in events:
                sink.append(event)
            sink.sync()


def um_minutes(minutes, **kwargs):
    return build_minutes(
        market="um",
        day=DAY,
        kline_stream=UM_KLINE_1M,
        mark_stream=UM_MARK_PRICE,
        book_stream=UM_BOOK_TICKER,
        **{"klines": [], "marks": [], "books": [], **kwargs},
    )


# --- A. the schema -------------------------------------------------------------
def test_the_two_markets_have_fixed_columns_and_the_spot_one_has_no_mark():
    um = [spec.name for spec in columns_for("um")]
    spot = [spec.name for spec in columns_for("spot")]
    assert um[0] == "minute_open_ms"
    assert "mark_present" in um and "index_close" in um
    assert not any(name.startswith("mark_") or name.startswith("index_") for name in spot)
    assert [name for name in um if name in spot] == spot, "spot is a prefix-order subset"
    assert set(MARKET_COLUMNS) == {"um", "spot"}


def test_an_unknown_market_has_no_schema():
    with pytest.raises(RecorderNormalizeError, match="no normalized schema"):
        columns_for("cm")


def test_the_frame_carries_the_declared_dtypes(root):
    write(root, um_day(range(3)))
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    frame = pd.read_parquet(report.parquet_path)
    assert list(frame.columns) == [spec.name for spec in columns_for("um")]
    for spec in columns_for("um"):
        assert str(frame[spec.name].dtype) in (
            spec.dtype,
            f"{spec.dtype}[python]",
        ), f"{spec.name} came back as {frame[spec.name].dtype}"


# --- B. missing data is missing ------------------------------------------------
def test_a_minute_with_no_closed_kline_has_no_row_and_is_listed_as_missing(root):
    present = [0, 1, 3, 4]
    write(root, um_day(present))
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    frame = pd.read_parquet(report.parquet_path)

    assert report.rows == 4
    assert list(frame["minute_open_ms"]) == [minute_ms(i) for i in present]
    assert minute_ms(2) not in set(frame["minute_open_ms"])
    assert minute_ms(2) in report.missing
    assert len(report.missing) == MINUTES_PER_DAY - 4
    assert report.rows + len(report.missing) == MINUTES_PER_DAY


def test_a_missing_minute_is_never_filled_from_its_neighbours(root):
    write(root, um_day([0, 2]))
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    frame = pd.read_parquet(report.parquet_path)

    assert len(frame) == 2
    assert (
        frame["kline_close"].iloc[0] != frame["kline_close"].iloc[1]
    ), "the fixture must give the two minutes different closes for this to mean anything"
    gap = frame[frame["minute_open_ms"] == minute_ms(1)]
    assert gap.empty, "the missing minute was materialised"


def test_a_minute_whose_mark_stream_was_silent_is_null_not_the_previous_minute(root):
    events = um_day([0, 1], with_mark=True)
    # Minute 1 loses every mark update; minute 0 keeps two, at distinct prices.
    events[UM_MARK_PRICE] = [
        event for event in events[UM_MARK_PRICE] if event.minute_open_ms == minute_ms(0)
    ]
    write(root, events)
    frame = pd.read_parquet(MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path)

    first, second = frame.iloc[0], frame.iloc[1]
    assert bool(first["mark_present"]) is True
    assert bool(second["mark_present"]) is False
    for column in ("mark_open", "mark_high", "mark_low", "mark_close", "index_close"):
        assert pd.isna(second[column]), f"{column} was carried forward"
    assert not pd.isna(first["mark_close"])


def test_a_minute_whose_book_was_silent_is_null_and_not_borrowed_from_the_kline(root):
    events = um_day([0], with_book=False)
    write(root, events)
    frame = pd.read_parquet(MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path)
    row = frame.iloc[0]
    assert bool(row["book_present"]) is False
    for column in ("book_bid", "book_ask", "book_bid_qty", "book_ask_qty", "book_update_id"):
        assert pd.isna(row[column])
    assert not pd.isna(row["kline_close"]), "the kline is still there"


def test_the_gap_list_reports_spans_rather_than_a_count(root):
    write(root, um_day([0, 1, 5]))
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    document = json.loads(report.meta_path.read_text(encoding="utf-8"))

    assert document["expected_minutes"] == MINUTES_PER_DAY
    assert document["rows"] == 3
    assert document["missing_minutes"] == MINUTES_PER_DAY - 3
    first_gap = document["gaps"][0]
    assert first_gap["first_missing_ms"] == minute_ms(2)
    assert first_gap["last_missing_ms"] == minute_ms(4)
    assert first_gap["missing_minutes"] == 3
    assert first_gap["first_missing_utc"] == "2026-09-19T00:02:00+00:00"


def test_gaps_of_collapses_runs_and_keeps_isolated_minutes_apart():
    spans = gaps_of([minute_ms(1), minute_ms(2), minute_ms(9)])
    assert [span["missing_minutes"] for span in spans] == [2, 1]
    assert spans[0]["first_missing_ms"] == minute_ms(1)
    assert spans[1]["first_missing_ms"] == minute_ms(9)
    assert gaps_of([]) == []


def test_the_metadata_refuses_to_claim_a_day_whose_minutes_do_not_add_up():
    records, missing, conflicts, tallies = um_minutes([0], klines=[kline_event(minute_ms(0))])
    frame = minute_frame(records, market="um")
    with pytest.raises(RecorderNormalizeError, match="not 1440"):
        meta(
            frame,
            market="um",
            day=DAY,
            contract=CONTRACT,
            missing=missing[:-1],
            conflicts=conflicts,
            streams=tallies,
            parquet_path="normalized/um/1m/x.parquet",
            parquet_sha256="0" * 64,
        )


# --- C. determinism ------------------------------------------------------------
def test_the_same_raw_files_produce_the_same_table_twice(root):
    write(root, um_day(range(4)))
    normalizer = MinuteNormalizer(root, CONTRACT)
    first = normalizer.build_day("um", DAY)
    second = normalizer.build_day("um", DAY)
    assert second.digest == first.digest
    assert second.parquet_sha256 == first.parquet_sha256


def test_out_of_order_delivery_produces_the_same_day_as_ordered_delivery(root, tmp_path):
    ordered = um_day(range(4))
    shuffled = {stream: list(reversed(events)) for stream, events in ordered.items()}
    write(root, ordered)
    other = CONTRACT.storage_root(tmp_path / "shuffled")
    write(other, shuffled)

    left = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    right = MinuteNormalizer(other, CONTRACT).build_day("um", DAY)
    assert right.digest == left.digest
    assert right.rows == left.rows


def test_a_duplicated_observation_produces_one_row(root):
    events = um_day([0, 1])
    events[UM_KLINE_1M] = events[UM_KLINE_1M] + events[UM_KLINE_1M]
    write(root, events)
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    assert report.rows == 2
    assert report.conflicts == ()


def test_the_digest_survives_a_parquet_round_trip(root):
    write(root, um_day(range(3)))
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    read_back = pd.read_parquet(report.parquet_path)
    assert digest(read_back, market="um") == report.digest

    rewritten = report.parquet_path.with_name("rewritten.parquet")
    read_back.to_parquet(rewritten, index=False, compression="gzip")
    assert digest(pd.read_parquet(rewritten), market="um") == report.digest
    assert (
        hashlib.sha256(rewritten.read_bytes()).hexdigest() != report.parquet_sha256
    ), "the container changed, which is exactly why it is not the identity"


def test_the_digest_moves_when_a_published_number_moves(root):
    write(root, um_day(range(2)))
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    frame = pd.read_parquet(report.parquet_path)
    tampered = frame.copy()
    tampered.loc[0, "kline_close"] = float(tampered.loc[0, "kline_close"]) + 0.01
    assert digest(tampered, market="um") != report.digest


def test_the_digest_tells_a_null_apart_from_a_zero(root):
    write(root, um_day([0], with_book=False))
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    frame = pd.read_parquet(report.parquet_path)
    zeroed = frame.copy()
    zeroed.loc[0, "book_bid"] = 0.0
    assert digest(zeroed, market="um") != report.digest


def test_minus_zero_and_zero_are_the_same_published_quantity(root):
    write(root, um_day([0]))
    frame = pd.read_parquet(MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path)
    positive = frame.copy()
    positive.loc[0, "kline_taker_buy_base"] = 0.0
    negative = frame.copy()
    negative.loc[0, "kline_taker_buy_base"] = -0.0
    assert digest(positive, market="um") == digest(negative, market="um")


def test_the_digest_refuses_a_frame_whose_columns_are_not_the_schema(root):
    write(root, um_day([0]))
    frame = pd.read_parquet(MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path)
    with pytest.raises(RecorderNormalizeError, match="has columns"):
        digest(frame.drop(columns=["book_ask"]), market="um")


def test_the_digest_refuses_a_null_in_a_column_that_may_not_hold_one(root):
    write(root, um_day([0]))
    frame = pd.read_parquet(MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path)
    frame.loc[0, "kline_close"] = None
    with pytest.raises(RecorderNormalizeError, match="declared non-null"):
        digest(frame, market="um")


def test_two_markets_with_identical_numbers_do_not_share_a_digest(root):
    """The market is in the header, so a spot day cannot be read as a perpetual one."""
    write(root, spot_day(range(2)))
    spot_report = MinuteNormalizer(root, CONTRACT).build_day("spot", DAY)
    frame = pd.read_parquet(spot_report.parquet_path)
    assert digest(frame, market="spot") == spot_report.digest
    with pytest.raises(RecorderNormalizeError):
        digest(frame, market="um")


# --- D. minute boundaries and no leakage ---------------------------------------
def test_an_event_on_a_minute_boundary_belongs_to_the_minute_that_is_opening(root):
    minute = minute_ms(0)
    events = {
        UM_KLINE_1M: [kline_event(minute), kline_event(minute + 60_000)],
        UM_MARK_PRICE: [
            mark_event(minute + 59_999, mark="60000.00"),
            mark_event(minute + 60_000, mark="61000.00"),
        ],
        UM_BOOK_TICKER: [],
    }
    write(root, events)
    frame = pd.read_parquet(MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path)

    assert float(frame.loc[0, "mark_close"]) == 60000.0
    assert float(frame.loc[1, "mark_close"]) == 61000.0


def test_a_minute_never_sees_the_book_of_the_minute_after_it(root):
    minute = minute_ms(0)
    events = {
        UM_KLINE_1M: [kline_event(minute)],
        UM_MARK_PRICE: [],
        UM_BOOK_TICKER: [
            book_event(1, event_ms=minute + 10_000, bid="1.00"),
            book_event(2, event_ms=minute + 60_000, bid="2.00"),
        ],
    }
    write(root, events)
    frame = pd.read_parquet(MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path)

    assert len(frame) == 1
    assert int(frame.loc[0, "book_update_id"]) == 1
    assert float(frame.loc[0, "book_bid"]) == 1.0
    assert int(frame.loc[0, "book_canonical_ms"]) < minute + 60_000


def test_the_minute_keeps_the_last_book_update_before_it_closed(root):
    minute = minute_ms(0)
    events = {
        UM_KLINE_1M: [kline_event(minute)],
        UM_MARK_PRICE: [],
        UM_BOOK_TICKER: [
            book_event(1, event_ms=minute + 10_000, bid="1.00"),
            book_event(2, event_ms=minute + 59_999, bid="2.00"),
        ],
    }
    write(root, events)
    frame = pd.read_parquet(MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path)
    assert int(frame.loc[0, "book_update_id"]) == 2


def test_the_mark_aggregate_is_the_extremes_actually_observed_in_the_minute(root):
    minute = minute_ms(0)
    prices = ("60000.00", "60500.00", "59500.00", "60100.00")
    events = {
        UM_KLINE_1M: [kline_event(minute)],
        UM_MARK_PRICE: [
            mark_event(minute + offset * 1_000, mark=price, index=price)
            for offset, price in enumerate(prices)
        ],
        UM_BOOK_TICKER: [],
    }
    write(root, events)
    frame = pd.read_parquet(MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path)
    row = frame.iloc[0]
    assert (float(row["mark_open"]), float(row["mark_close"])) == (60000.0, 60100.0)
    assert (float(row["mark_high"]), float(row["mark_low"])) == (60500.0, 59500.0)
    assert (float(row["index_high"]), float(row["index_low"])) == (60500.0, 59500.0)
    assert int(row["mark_events"]) == 4


def test_the_book_time_basis_travels_into_the_normalized_layer(root):
    write(root, um_day([0]))
    um_frame = pd.read_parquet(
        MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path
    )
    assert um_frame.loc[0, "book_time_basis"] == TimeBasis.EXCHANGE.value

    write(root, spot_day([0]))
    spot_frame = pd.read_parquet(
        MinuteNormalizer(root, CONTRACT).build_day("spot", DAY).parquet_path
    )
    assert (
        spot_frame.loc[0, "book_time_basis"] == TimeBasis.RECEIPT.value
    ), "Binance spot bookTicker has no event time; the column must say so"


# --- E. partial frames, late arrivals, conflicts -------------------------------
def test_a_forming_candle_is_kept_in_raw_and_ignored_by_the_normalizer(root):
    minute = minute_ms(0)
    write(
        root,
        {
            UM_KLINE_1M: [
                kline_event(minute, closed=False, close="1.00"),
                kline_event(minute, closed=True, close="2.00"),
            ],
            UM_MARK_PRICE: [],
            UM_BOOK_TICKER: [],
        },
    )
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    frame = pd.read_parquet(report.parquet_path)
    document = json.loads(report.meta_path.read_text(encoding="utf-8"))

    assert len(frame) == 1
    assert float(frame.loc[0, "kline_close"]) == 2.0
    assert document["streams"][UM_KLINE_1M] == {"records": 2, "closed": 1, "partial": 1}


def test_a_late_arrival_is_normalized_and_ordered_after_the_day_s_own_record(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as sink:
        sink.append(kline_event(minute_ms(0), close="1.00"))
        sink.append(kline_event(minute_ms(0, day=NEXT_DAY)))
        sink.append(kline_event(minute_ms(0), close="2.00"))  # late, same minute
        sink.sync()

    frame = pd.read_parquet(MinuteNormalizer(root, CONTRACT).build_day("um", DAY).parquet_path)
    assert len(frame) == 1
    assert (
        float(frame.loc[0, "kline_close"]) == 2.0
    ), "the late record is the last closed frame for the minute and therefore wins"


def test_two_disagreeing_closed_frames_are_recorded_as_a_conflict(root):
    minute = minute_ms(0)
    write(
        root,
        {
            UM_KLINE_1M: [
                kline_event(minute, close="1.00"),
                kline_event(minute, close="2.00", source=EventSource.REST_GAPFILL),
            ],
            UM_MARK_PRICE: [],
            UM_BOOK_TICKER: [],
        },
    )
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    document = json.loads(report.meta_path.read_text(encoding="utf-8"))

    assert report.conflicts == (minute,)
    assert document["conflicting_minutes"] == [minute]
    assert report.rows == 1, "a disagreement is a finding, not a reason to drop the minute"


def test_two_agreeing_closed_frames_are_not_a_conflict(root):
    minute = minute_ms(0)
    write(
        root,
        {
            UM_KLINE_1M: [
                kline_event(minute),
                kline_event(minute, source=EventSource.REST_GAPFILL),
            ],
            UM_MARK_PRICE: [],
            UM_BOOK_TICKER: [],
        },
    )
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY)
    assert report.conflicts == ()
    frame = pd.read_parquet(report.parquet_path)
    assert frame.loc[0, "kline_source"] == EventSource.REST_GAPFILL.value


def test_a_kline_record_stamped_with_the_wrong_canonical_time_is_refused():
    minute = minute_ms(0)
    bad = kline_event(minute)
    object.__setattr__(bad, "canonical_ns", (minute + 60_000) * NS_PER_MILLISECOND)
    with pytest.raises(RecorderNormalizeError, match="stamps canonical_ns"):
        um_minutes([0], klines=[bad])


def test_a_record_from_another_day_is_refused_rather_than_normalized():
    with pytest.raises(RecorderNormalizeError, match="outside the UTC day"):
        um_minutes([0], klines=[kline_event(minute_ms(0, day=NEXT_DAY))])


def test_a_price_that_is_not_a_number_stops_the_day():
    with pytest.raises(RecorderNormalizeError, match="not a number"):
        um_minutes([0], klines=[kline_event(minute_ms(0), close="not-a-price")])


# --- F. metadata and freezing --------------------------------------------------
def test_the_metadata_names_the_contract_the_schema_and_relative_paths(root):
    write(root, um_day(range(2)))
    report = MinuteNormalizer(root, CONTRACT).build_day("um", DAY, provenance={"host": "h"})
    document = json.loads(report.meta_path.read_text(encoding="utf-8"))

    assert document["meta_schema"] == NORMALIZED_META_SCHEMA
    assert document["normalized_schema"] == NORMALIZED_SCHEMA
    assert document["market"] == "um"
    assert document["clock"] == CLOCK
    assert document["day"] == DAY
    assert document["contract"]["contract_hash"] == CONTRACT.contract_hash
    assert document["contract"]["prospective_from"] is None
    assert document["digest"] == report.digest
    assert document["parquet_sha256"] == report.parquet_sha256
    assert document["parquet_path"] == f"normalized/um/{CLOCK}/{DAY}.parquet"
    assert document["provenance"] == {"host": "h"}
    assert document["columns"] == [spec.to_dict() for spec in columns_for("um")]
    assert all(path.startswith("raw/") for path in document["source_paths"])
    assert "\\" not in report.meta_path.read_text(encoding="utf-8")


def test_freezing_a_day_writes_a_checksum_and_makes_it_immutable(root):
    write(root, um_day(range(2)))
    normalizer = MinuteNormalizer(root, CONTRACT)
    report = normalizer.build_day("um", DAY)

    assert normalizer.is_frozen("um", DAY) is False
    checksum = normalizer.freeze_day("um", DAY)
    assert normalizer.is_frozen("um", DAY) is True
    assert checksum.read_text(encoding="utf-8") == (
        f"{report.parquet_sha256}  {DAY}.parquet\n"
    )

    with pytest.raises(RecorderNormalizeError, match="stays frozen"):
        normalizer.freeze_day("um", DAY)
    with pytest.raises(RecorderNormalizeError, match="is frozen"):
        normalizer.build_day("um", DAY)


def test_freezing_a_day_that_was_never_built_is_refused(root):
    with pytest.raises(RecorderNormalizeError, match="never built"):
        MinuteNormalizer(root, CONTRACT).freeze_day("um", DAY)


def test_a_market_the_contract_does_not_declare_cannot_be_normalized(root):
    with pytest.raises(Exception, match="declares markets|no normalized schema"):
        MinuteNormalizer(root, CONTRACT).build_day("cm", DAY)


# --- G. funding settlements ----------------------------------------------------
def test_settlements_are_rebuilt_in_order_and_deduplicated(root):
    events = funding_day(DAY)
    write(root, {UM_FUNDING: events + [events[0]]})
    report = MinuteNormalizer(root, CONTRACT).build_settlements("um")

    records = [
        json.loads(line)
        for line in report.path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert report.rows == 3
    assert [record["funding_time_ms"] for record in records] == sorted(
        record["funding_time_ms"] for record in records
    )
    assert records[0]["funding_time_utc"] == "2026-09-19T00:00:00+00:00"
    assert records[1]["funding_rate"] == "0.00010008"
    assert report.first_funding_time_ms == day_ms()
    assert report.last_funding_time_ms == day_ms() + 16 * 3_600_000


def test_the_settlements_checksum_file_matches_the_bytes_it_names(root):
    write(root, {UM_FUNDING: funding_day(DAY)})
    report = MinuteNormalizer(root, CONTRACT).build_settlements("um")
    line = report.digest_path.read_text(encoding="utf-8")
    assert line == f"{report.sha256}  settlements.ndjson\n"
    assert hashlib.sha256(report.path.read_bytes()).hexdigest() == report.sha256
    assert b"\r\n" not in report.path.read_bytes()


def test_two_settlements_that_disagree_about_one_instant_stop_the_rebuild(root):
    instant = day_ms() + 8 * 3_600_000
    write(
        root,
        {
            UM_FUNDING: [
                funding_event(instant, rate="0.00010000"),
                funding_event(instant, rate="0.00020000"),
            ]
        },
    )
    with pytest.raises(RecorderNormalizeError, match="disagree"):
        MinuteNormalizer(root, CONTRACT).build_settlements("um")


def test_rebuilding_settlements_is_deterministic(root):
    write(root, {UM_FUNDING: funding_day(DAY)})
    normalizer = MinuteNormalizer(root, CONTRACT)
    first = normalizer.build_settlements("um")
    second = normalizer.build_settlements("um")
    assert second.sha256 == first.sha256


def test_a_market_with_no_funding_stream_has_no_settlements(root):
    with pytest.raises(RecorderNormalizeError, match="no spot.funding stream"):
        MinuteNormalizer(root, CONTRACT).build_settlements("spot")


# --- H. the normalizer computes nothing ----------------------------------------
#: Names that would mean a value was invented rather than carried across.
FORBIDDEN_CALLS = frozenset(
    {
        "ffill",
        "bfill",
        "pad",
        "backfill",
        "fillna",
        "interpolate",
        "resample",
        "asfreq",
        "reindex",
        "combine_first",
    }
)

#: Identifiers that would mean the normalizer had started doing research. Matched
#: against the module's *names* rather than its text, so the docstring may say
#: "no PnL" without the scan reading its own promise as a violation.
FORBIDDEN_ECONOMICS = frozenset(
    {
        "future_return",
        "pnl",
        "unrealised_pnl",
        "realised_pnl",
        "alpha",
        "sharpe",
        "information_coefficient",
        "funding_flow",
        "funding_cash_flow",
        "basis",
        "predict",
        "fit",
        "score",
        "equity",
        "liquidation_price",
    }
)


def module_identifiers(tree: ast.AST) -> set[str]:
    """Every name the module defines, reads or calls an attribute of."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def test_the_normalizer_never_calls_a_fill_or_an_interpolation():
    """A positive scan, so "missing data is missing" is a property of the source."""
    tree = ast.parse(NORMALIZE_SOURCE.read_text(encoding="utf-8"))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & FORBIDDEN_CALLS, (
        f"chimera/recorder/normalize.py calls {sorted(called & FORBIDDEN_CALLS)}; a "
        "normalizer that fills a gap has invented a minute the exchange never printed"
    )


def test_the_scan_can_actually_see_a_fill():
    """Catches the scanner going blind, which would pass the module vacuously."""
    tree = ast.parse("frame['x'] = frame['x'].ffill()\nother.interpolate()\n")
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called & FORBIDDEN_CALLS == {"ffill", "interpolate"}


def test_the_offline_core_names_no_economic_quantity():
    """The recorder records. Anything it computed would be a number nobody can check."""
    package = NORMALIZE_SOURCE.parent
    modules = sorted(package.rglob("*.py"))
    assert len(modules) >= 5, "the scan found nothing and would pass vacuously"
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = sorted(module_identifiers(tree) & FORBIDDEN_ECONOMICS)
        assert not offenders, (
            f"{path.name} defines or calls {offenders}; the recorder computes no return, "
            "signal, funding flow, basis, PnL or statistic of any kind"
        )


def test_the_economics_scan_can_actually_see_one():
    tree = ast.parse("def pnl(equity):\n    return equity.basis\n")
    assert module_identifiers(tree) & FORBIDDEN_ECONOMICS == {"pnl", "equity", "basis"}


def test_the_recorder_package_does_not_import_the_research_evaluation_code():
    package = NORMALIZE_SOURCE.parent
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "nn.evaluate" not in text, f"{path} imports the research evaluator"
        assert "chimera.carry" not in text, f"{path} imports the carry accounting"
        assert not re.search(r"^\s*(from|import)\s+nn\b", text, re.MULTILINE), (
            f"{path} imports from nn; the recorder is infrastructure and must not depend "
            "on a research checkpoint's code"
        )
