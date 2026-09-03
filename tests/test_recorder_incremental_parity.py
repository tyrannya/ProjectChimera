"""The incremental normalizer against the full rebuild, on material designed to break it.

This file exists to answer one question: does folding a day from a cursor produce
**exactly** what re-reading it produces? Not approximately, not for the easy
cases — exactly, including the digest, on material chosen because a naive
incremental implementation would get it wrong.

The full rebuild is the oracle. It is not modified, it is not reimplemented here,
and every assertion below compares against it rather than against a value written
into the test. A test that hard-coded the expected rows would pass just as
happily if both paths were wrong together.

**What a naive implementation gets wrong, and what catches it.** The obvious
shortcut is "the last event I saw wins". That is not the rule. The rule is the
last event *in canonical-time order, with file order as the tie-break*, which is
a different event whenever anything arrives late or out of order — and a
recorder's whole reason for existing is that the network does that. So the
adversarial cases below deliberately feed events in an order that differs from
their sort order, and ``test_arrival_order_is_not_the_rule`` proves the material
is capable of telling the two apart before anything else relies on it.
"""

from __future__ import annotations

import gzip
import json
import random

import pandas as pd
import pytest

from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.events import (
    NS_PER_MILLISECOND,
    SPOT_BOOK_TICKER,
    SPOT_KLINE_1M,
    UM_BOOK_TICKER,
    UM_KLINE_1M,
    UM_MARK_PRICE,
    EventSource,
    RawEvent,
    RecorderEventError,
    TimeBasis,
    sort_events,
)
from chimera.recorder.incremental import (
    CACHE_SCHEMA,
    IncrementalNormalizer,
    NormalizeCacheError,
)
from chimera.recorder.normalize import (
    MinuteNormalizer,
    RecorderNormalizeError,
    digest,
    minute_frame,
)
from chimera.recorder.sink import RawSink
from tests.recorder_synthetic import (
    DAY,
    book_event,
    kline_event,
    mark_event,
    minute_ms,
)

CONTRACT = load_recorder_contract()


def write(root, events_by_stream, *, freeze=False):
    """Put events on disk in the order given, which is the order they arrived."""
    for stream, events in events_by_stream.items():
        with RawSink(root, stream, contract=CONTRACT) as sink:
            for event in events:
                sink.append(event)
            sink.sync()
            if freeze:
                sink.freeze_day(DAY)


def full_report(root, market):
    return MinuteNormalizer(root, CONTRACT).build_day(market, DAY)


def compare(tmp_path, events_by_stream, market="um", *, seed_cache=True):
    """Build the same material both ways in two roots and return both reports.

    Two roots rather than one, because the full path and the incremental path
    both write the day's Parquet and metadata and the second would refuse to
    overwrite the first's frozen output — and because comparing two independent
    trees is a stronger statement than comparing one tree with itself.
    """
    oracle_root = tmp_path / "oracle"
    cursor_root = tmp_path / "cursor"
    for root in (oracle_root, cursor_root):
        root.mkdir(parents=True, exist_ok=True)
        write(root, events_by_stream)
    oracle = full_report(oracle_root, market)
    incremental = IncrementalNormalizer(cursor_root, CONTRACT)
    if seed_cache:
        rendered = incremental.build_day(market, DAY)
    else:
        rendered = incremental.build_day(market, DAY)
    return oracle, rendered, oracle_root, cursor_root, incremental


def assert_identical(oracle, rendered, oracle_root, cursor_root, market="um"):
    """Every observable the two paths produce, compared field by field."""
    assert rendered.rows == oracle.rows
    assert rendered.missing == oracle.missing
    assert rendered.conflicts == oracle.conflicts
    assert rendered.digest == oracle.digest, "the value-level digest differs"

    left = pd.read_parquet(MinuteNormalizer(oracle_root, CONTRACT).parquet_path(market, DAY))
    right = pd.read_parquet(MinuteNormalizer(cursor_root, CONTRACT).parquet_path(market, DAY))
    pd.testing.assert_frame_equal(left, right, check_dtype=True, check_exact=True)

    a = json.loads(
        MinuteNormalizer(oracle_root, CONTRACT).meta_path(market, DAY).read_text("utf-8")
    )
    b = json.loads(
        MinuteNormalizer(cursor_root, CONTRACT).meta_path(market, DAY).read_text("utf-8")
    )
    for field in (
        "rows",
        "expected_minutes",
        "missing_minutes",
        "missing",
        "gaps",
        "conflicting_minutes",
        "streams",
        "first_minute_open_ms",
        "last_minute_open_ms",
        "digest",
        "columns",
        "missing_semantics",
    ):
        assert a[field] == b[field], f"metadata field {field!r} differs"


# --- A. the material can tell the two rules apart -------------------------------
def test_arrival_order_is_not_the_rule():
    """Before anything relies on it: canonical order and arrival order differ here.

    A minute whose book updates arrive 3, 1, 2 has a different *last* update
    under each rule. If this ever stopped being true, every parity test below
    would still pass while proving nothing.
    """
    arrival = [
        book_event(3, event_ms=minute_ms(0) + 3_000),
        book_event(1, event_ms=minute_ms(0) + 1_000),
        book_event(2, event_ms=minute_ms(0) + 2_000),
    ]
    assert [e.canonical_ns for e in arrival] != [e.canonical_ns for e in sort_events(arrival)]
    assert (
        arrival[-1].canonical_ns != sort_events(arrival)[-1].canonical_ns
    ), "arrival order and canonical order pick different winners on this material"


# --- B. parity on adversarial material ------------------------------------------
def um_material(events):
    return {stream: list(items) for stream, items in events.items()}


def test_parity_on_ordered_events(tmp_path):
    events = {
        UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(4)],
        UM_MARK_PRICE: [mark_event(minute_ms(i) + 1_000) for i in range(4)],
        UM_BOOK_TICKER: [book_event(i + 1, event_ms=minute_ms(i) + 2_000) for i in range(4)],
    }
    assert_identical(*compare(tmp_path, events)[:4])


def test_parity_on_out_of_order_events(tmp_path):
    """Delivered backwards. The winner is the latest canonical time, not the last line."""
    events = {
        UM_KLINE_1M: [kline_event(minute_ms(i)) for i in (3, 0, 2, 1)],
        UM_MARK_PRICE: [
            mark_event(minute_ms(0) + ms, mark=f"{60000 + ms}.00", index=f"{59999 + ms}.00")
            for ms in (3_000, 1_000, 2_000)
        ],
        UM_BOOK_TICKER: [
            book_event(uid, event_ms=minute_ms(0) + ms)
            for ms, uid in ((3_000, 3), (1_000, 1), (2_000, 2))
        ],
    }
    assert_identical(*compare(tmp_path, events)[:4])


def test_parity_when_two_events_share_a_canonical_time(tmp_path):
    """Equal timestamps: file order breaks the tie, and it must break it the same way."""
    stamp = minute_ms(0) + 5_000
    events = {
        UM_KLINE_1M: [kline_event(minute_ms(0))],
        UM_MARK_PRICE: [
            mark_event(stamp, mark="60001.00", index="60000.00"),
            mark_event(stamp, mark="60002.00", index="60001.00"),
            mark_event(stamp, mark="60003.00", index="60002.00"),
        ],
        UM_BOOK_TICKER: [
            book_event(7, event_ms=stamp, bid="1.00"),
            book_event(8, event_ms=stamp, bid="2.00"),
        ],
    }
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    frame = pd.read_parquet(MinuteNormalizer(cursor_root, CONTRACT).parquet_path("um", DAY))
    assert frame["mark_close"][0] == 60003.0, "the last line at an equal stamp wins"
    assert frame["mark_open"][0] == 60001.0, "and the first line opens"
    assert frame["book_update_id"][0] == 8


def test_parity_on_a_partial_then_closed_kline(tmp_path):
    events = {
        UM_KLINE_1M: [
            kline_event(minute_ms(0), closed=False),
            kline_event(minute_ms(0), closed=True),
        ]
    }
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    assert rendered.rows == 1, "the partial frame is not a minute"
    document = json.loads(
        MinuteNormalizer(cursor_root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
    )
    assert document["streams"][UM_KLINE_1M] == {"records": 2, "closed": 1, "partial": 1}


def test_parity_when_websocket_and_rest_agree_on_a_minute(tmp_path):
    events = {
        UM_KLINE_1M: [
            kline_event(minute_ms(0), source=EventSource.WEBSOCKET),
            kline_event(minute_ms(0), source=EventSource.REST_GAPFILL),
        ]
    }
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    assert rendered.conflicts == (), "identical readings are not a disagreement"


def test_parity_when_websocket_and_rest_disagree_on_a_minute(tmp_path):
    """The conflict must be recorded, by both paths, on the same minute."""
    events = {
        UM_KLINE_1M: [
            kline_event(minute_ms(0), source=EventSource.WEBSOCKET),
            kline_event(minute_ms(0), source=EventSource.REST_GAPFILL, close="60099.99"),
        ]
    }
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    assert rendered.conflicts == (minute_ms(0),)


def test_parity_when_the_disagreeing_frame_arrives_first(tmp_path):
    """The same disagreement, folded the other way round. Still one conflict."""
    events = {
        UM_KLINE_1M: [
            kline_event(minute_ms(0), source=EventSource.REST_GAPFILL, close="60099.99"),
            kline_event(minute_ms(0), source=EventSource.WEBSOCKET),
        ]
    }
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    assert rendered.conflicts == (minute_ms(0),)


def test_parity_on_duplicate_events(tmp_path):
    """The sink refuses the duplicate, so both paths see one record. Still compared."""
    event = kline_event(minute_ms(0))
    events = {UM_KLINE_1M: [event, event, event]}
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)


def test_parity_with_a_late_event(tmp_path):
    """A late arrival lands in the late file, which sorts after the day's own."""
    oracle_root = tmp_path / "oracle"
    cursor_root = tmp_path / "cursor"
    for root in (oracle_root, cursor_root):
        root.mkdir(parents=True)
        sink = RawSink(root, UM_BOOK_TICKER, contract=CONTRACT)
        sink.append(book_event(1, event_ms=minute_ms(0) + 1_000, bid="1.00"))
        sink.sync()
        sink.rotate("2026-09-20")  # the day rolls over, closing it
        sink.append(book_event(99, event_ms=minute_ms(0, day="2026-09-20")))
        sink.append(book_event(2, event_ms=minute_ms(0) + 500, bid="2.00"))  # late
        sink.sync()
        sink.close()
        with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as klines:
            klines.append(kline_event(minute_ms(0)))
            klines.sync()

    oracle = full_report(oracle_root, "um")
    rendered = IncrementalNormalizer(cursor_root, CONTRACT).build_day("um", DAY)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    frame = pd.read_parquet(MinuteNormalizer(cursor_root, CONTRACT).parquet_path("um", DAY))
    assert frame["book_update_id"][0] == 1, (
        "canonical time dominates: the late record carries an earlier one, so it sorts "
        "before the day's own record and does not win the minute"
    )


def test_parity_when_a_late_record_ties_the_canonical_time_of_one_already_written(tmp_path):
    """The case where the late file's rank actually decides.

    Canonical time first; the file only breaks a tie. So a late record at the
    *same* instant as one already in the day's own file sorts after it and wins,
    and a late record at an earlier instant does not — the previous test.
    """
    stamp = minute_ms(0) + 1_000
    oracle_root = tmp_path / "oracle"
    cursor_root = tmp_path / "cursor"
    for root in (oracle_root, cursor_root):
        root.mkdir(parents=True)
        sink = RawSink(root, UM_BOOK_TICKER, contract=CONTRACT)
        sink.append(book_event(1, event_ms=stamp, bid="1.00"))
        sink.sync()
        sink.rotate("2026-09-20")
        sink.append(book_event(2, event_ms=stamp, bid="2.00"))  # late, same instant
        sink.sync()
        sink.close()
        with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as klines:
            klines.append(kline_event(minute_ms(0)))
            klines.sync()

    oracle = full_report(oracle_root, "um")
    rendered = IncrementalNormalizer(cursor_root, CONTRACT).build_day("um", DAY)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    frame = pd.read_parquet(MinuteNormalizer(cursor_root, CONTRACT).parquet_path("um", DAY))
    assert frame["book_update_id"][0] == 2, "the late file sorts after at an equal instant"
    assert frame["book_bid"][0] == 2.0


def test_parity_with_a_missing_kline_minute(tmp_path):
    events = {
        UM_KLINE_1M: [kline_event(minute_ms(i)) for i in (0, 2, 5)],
        UM_MARK_PRICE: [mark_event(minute_ms(i) + 1_000) for i in range(6)],
    }
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    assert rendered.rows == 3 and len(rendered.missing) == 1437
    document = json.loads(
        MinuteNormalizer(cursor_root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
    )
    assert (
        document["gaps"]
        == json.loads(
            MinuteNormalizer(oracle_root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
        )["gaps"]
    )


def test_parity_with_a_missing_mark_and_a_missing_book(tmp_path):
    """Absence travels as a false flag and nulls, identically down both paths."""
    events = {
        UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(3)],
        UM_MARK_PRICE: [mark_event(minute_ms(1) + 1_000)],
        UM_BOOK_TICKER: [book_event(1, event_ms=minute_ms(2) + 1_000)],
    }
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    frame = pd.read_parquet(MinuteNormalizer(cursor_root, CONTRACT).parquet_path("um", DAY))
    assert list(frame["mark_present"]) == [False, True, False]
    assert list(frame["book_present"]) == [False, False, True]
    assert frame["mark_open"].isna()[0] and frame["book_bid"].isna()[0]


def test_parity_with_many_mark_observations_in_one_minute(tmp_path):
    """Open, high, low, close and the count, over a minute that moves."""
    marks = [
        mark_event(minute_ms(0) + ms, mark=f"{60000 + delta}.00", index=f"{59990 + delta}.00")
        for ms, delta in zip(range(0, 60_000, 1_000), [3, 9, 1, 7, 2, 8, 4, 6, 5] * 7)
    ]
    events = {UM_KLINE_1M: [kline_event(minute_ms(0))], UM_MARK_PRICE: marks}
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    frame = pd.read_parquet(MinuteNormalizer(cursor_root, CONTRACT).parquet_path("um", DAY))
    assert frame["mark_events"][0] == len(marks)
    assert frame["mark_high"][0] == 60009.0 and frame["mark_low"][0] == 60001.0


def test_parity_on_the_spot_market_too(tmp_path):
    """Spot has no mark column and its book carries no exchange time."""
    events = {
        SPOT_KLINE_1M: [kline_event(minute_ms(i), stream=SPOT_KLINE_1M) for i in range(3)],
        SPOT_BOOK_TICKER: [
            book_event(
                i + 1,
                stream=SPOT_BOOK_TICKER,
                event_ms=None,
                receipt_wall_ns=(minute_ms(i) + 1_000) * NS_PER_MILLISECOND,
            )
            for i in range(3)
        ],
    }
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events, market="spot")
    assert_identical(oracle, rendered, oracle_root, cursor_root, market="spot")
    frame = pd.read_parquet(MinuteNormalizer(cursor_root, CONTRACT).parquet_path("spot", DAY))
    assert set(frame["book_time_basis"]) == {TimeBasis.RECEIPT.value}


# --- C. a flood, and a randomized sweep -----------------------------------------
def test_parity_on_a_large_book_flood(tmp_path):
    """Sixty thousand book updates across ten minutes, delivered shuffled.

    Small next to a real day and large enough that an implementation which kept
    only the arrival-order winner, or which lost a count, cannot survive it.
    """
    rng = random.Random(20260904)
    books = []
    for minute in range(10):
        for tick in range(6_000):
            books.append(
                book_event(
                    minute * 6_000 + tick + 1,
                    event_ms=minute_ms(minute) + (tick * 10) % 60_000,
                    bid=f"{59000 + tick % 900}.10",
                )
            )
    rng.shuffle(books)
    events = {
        UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(10)],
        UM_BOOK_TICKER: books,
    }
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)
    document = json.loads(
        MinuteNormalizer(cursor_root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
    )
    assert document["streams"][UM_BOOK_TICKER]["records"] == len(books)


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8, 13])
def test_parity_on_randomized_material(tmp_path, seed):
    """Property-style parity, deterministic under a pinned seed.

    Each seed draws a different mixture of ordered, out-of-order, duplicated,
    partial, closed, agreeing and disagreeing material across ten minutes, and
    the two paths must agree on all of it.
    """
    rng = random.Random(seed)
    klines, marks, books = [], [], []
    for minute in range(10):
        for _ in range(rng.randint(0, 3)):
            closed = rng.random() < 0.7
            # A REST row is a finished candle by construction, so only a
            # websocket frame can be a partial one.
            source = (
                rng.choice([EventSource.WEBSOCKET, EventSource.REST_GAPFILL])
                if closed
                else EventSource.WEBSOCKET
            )
            klines.append(
                kline_event(
                    minute_ms(minute),
                    closed=closed,
                    close=f"{60000 + rng.randint(0, 5)}.50",
                    source=source,
                )
            )
        for _ in range(rng.randint(0, 5)):
            offset = rng.randrange(0, 60_000)
            marks.append(
                mark_event(
                    minute_ms(minute) + offset,
                    mark=f"{60000 + rng.randint(0, 40)}.00",
                    index=f"{59990 + rng.randint(0, 40)}.00",
                )
            )
        for _ in range(rng.randint(0, 6)):
            offset = rng.randrange(0, 60_000)
            books.append(
                book_event(
                    rng.randint(1, 10_000_000),
                    event_ms=minute_ms(minute) + offset,
                    bid=f"{59000 + rng.randint(0, 99)}.10",
                )
            )
    rng.shuffle(klines)
    rng.shuffle(marks)
    rng.shuffle(books)
    events = {UM_KLINE_1M: klines, UM_MARK_PRICE: marks, UM_BOOK_TICKER: books}
    oracle, rendered, oracle_root, cursor_root, _ = compare(tmp_path, events)
    assert_identical(oracle, rendered, oracle_root, cursor_root)


# --- D. resuming, crashing, and refusing --------------------------------------
def test_a_second_pass_reads_only_the_tail_and_still_matches(tmp_path):
    """The whole point: fold, append, fold again, and land where a rebuild would."""
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(3)]})
    incremental = IncrementalNormalizer(root, CONTRACT)
    incremental.build_day("um", DAY)
    first = incremental.status[("um", DAY)]
    assert first.resumed is False and first.replayed_records == 3

    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as sink:
        for index in range(3, 6):
            sink.append(kline_event(minute_ms(index)))
        sink.sync()
    rendered = incremental.build_day("um", DAY)
    second = incremental.status[("um", DAY)]
    assert second.resumed is True
    assert second.replayed_records == 3, "only the three new records were read"

    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()
    write(oracle_root, {UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(6)]})
    assert rendered.digest == full_report(oracle_root, "um").digest


def test_a_crash_after_the_append_and_before_the_cache_replays_the_tail_once(tmp_path):
    """Raw is durable, the cache is behind it, and the tail is folded exactly once.

    The counts are what catch a double fold: an event replayed twice would
    inflate ``records`` and a mark's event count, and the digest would still
    match because the extremum is idempotent. So the metadata is compared too.
    """
    root = tmp_path / "cursor"
    root.mkdir()
    write(
        root,
        {UM_KLINE_1M: [kline_event(minute_ms(0))], UM_MARK_PRICE: [mark_event(minute_ms(0))]},
    )
    incremental = IncrementalNormalizer(root, CONTRACT)
    incremental.build_day("um", DAY)

    # More raw lands. The process dies before the cache is written: the cache on
    # disk still describes only the first record.
    with RawSink(root, UM_MARK_PRICE, contract=CONTRACT) as sink:
        for offset in (1_000, 2_000, 3_000):
            sink.append(mark_event(minute_ms(0) + offset))
        sink.sync()

    restarted = IncrementalNormalizer(root, CONTRACT)
    rendered = restarted.build_day("um", DAY)
    assert restarted.status[("um", DAY)].replayed_records == 3

    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()
    write(
        oracle_root,
        {
            UM_KLINE_1M: [kline_event(minute_ms(0))],
            UM_MARK_PRICE: [mark_event(minute_ms(0))]
            + [mark_event(minute_ms(0) + o) for o in (1_000, 2_000, 3_000)],
        },
    )
    oracle = full_report(oracle_root, "um")
    assert rendered.digest == oracle.digest
    left = json.loads(
        MinuteNormalizer(oracle_root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
    )
    right = json.loads(
        MinuteNormalizer(root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
    )
    assert left["streams"] == right["streams"], "a record was folded twice, or not at all"


def test_a_partial_trailing_line_is_not_folded_until_it_is_whole(tmp_path):
    """A writer mid-append has written half a record. Half a record is not one."""
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(0))]})
    path = RawSink(root, UM_KLINE_1M, contract=CONTRACT).events_path(DAY)
    whole = kline_event(minute_ms(1)).canonical_line()
    with path.open("ab") as handle:
        handle.write(whole[: len(whole) // 2])

    incremental = IncrementalNormalizer(root, CONTRACT)
    rendered = incremental.build_day("um", DAY)
    assert rendered.rows == 1, "the torn record was not folded"

    with path.open("ab") as handle:
        handle.write(whole[len(whole) // 2 :])
    rendered = incremental.build_day("um", DAY)
    assert rendered.rows == 2, "and it was folded once the writer finished it"


def test_a_crash_after_the_cache_and_before_the_output_re_renders_from_the_cache(tmp_path):
    """The normalized file is derived. Losing it costs a render, never a record."""
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(4)]})
    incremental = IncrementalNormalizer(root, CONTRACT)
    first = incremental.build_day("um", DAY)
    normalizer = MinuteNormalizer(root, CONTRACT)
    normalizer.parquet_path("um", DAY).unlink()
    normalizer.meta_path("um", DAY).unlink()

    second = incremental.build_day("um", DAY)
    assert second.digest == first.digest
    assert incremental.status[("um", DAY)].replayed_records == 0, "nothing was re-read"
    assert normalizer.parquet_path("um", DAY).exists()


def test_a_corrupt_cache_is_refused_and_the_day_is_rebuilt_whole(tmp_path):
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(4)]})
    incremental = IncrementalNormalizer(root, CONTRACT)
    first = incremental.build_day("um", DAY)
    incremental.cache_path("um", DAY).write_text("{ truncated", encoding="utf-8")

    second = incremental.build_day("um", DAY)
    status = incremental.status[("um", DAY)]
    assert status.rebuilt is True and "unusable" in status.reason
    assert second.digest == first.digest, "the rebuild produced the same day"
    assert second.rows == 4, "and all of it, not a partial one"


def test_a_cache_from_another_build_or_another_contract_is_refused(tmp_path):
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(3)]})
    incremental = IncrementalNormalizer(root, CONTRACT)
    incremental.build_day("um", DAY)
    path = incremental.cache_path("um", DAY)

    for field, value in (
        ("cache_schema", "chimera.recorder-normalize-cache/999"),
        ("contract_hash", "0" * 64),
        ("day", "2026-01-01"),
        ("market", "spot"),
    ):
        document = json.loads(path.read_text(encoding="utf-8"))
        document[field] = value
        path.write_text(json.dumps(document), encoding="utf-8")
        rendered = incremental.build_day("um", DAY)
        assert incremental.status[("um", DAY)].rebuilt is True, field
        assert rendered.rows == 3, f"a refused cache must not shorten the day ({field})"


def test_a_cache_claiming_more_than_the_raw_file_holds_is_refused(tmp_path):
    """The rule that keeps raw authoritative: a cursor may never outrun the file."""
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(4)]})
    incremental = IncrementalNormalizer(root, CONTRACT)
    incremental.build_day("um", DAY)
    path = incremental.cache_path("um", DAY)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["cursors"][f"{UM_KLINE_1M}:main"]["offset"] += 10_000
    path.write_text(json.dumps(document), encoding="utf-8")

    rendered = incremental.build_day("um", DAY)
    assert incremental.status[("um", DAY)].rebuilt is True
    assert rendered.rows == 4


def test_a_day_frozen_since_the_cache_was_written_is_rebuilt_not_guessed(tmp_path):
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(3)]})
    incremental = IncrementalNormalizer(root, CONTRACT)
    incremental.build_day("um", DAY)
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as sink:
        sink.freeze_day(DAY)
    assert RawSink(root, UM_KLINE_1M, contract=CONTRACT).gz_path(DAY).exists()

    rendered = incremental.build_day("um", DAY)
    assert incremental.status[("um", DAY)].rebuilt is True
    assert rendered.rows == 3, "the frozen day was rebuilt in full from the compressed file"


def test_the_cache_says_it_is_not_evidence(tmp_path):
    """A reader who finds this file must not mistake it for a recorded value."""
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(0))]})
    incremental = IncrementalNormalizer(root, CONTRACT)
    incremental.build_day("um", DAY)
    document = json.loads(incremental.cache_path("um", DAY).read_text(encoding="utf-8"))
    assert document["cache_schema"] == CACHE_SCHEMA
    assert CACHE_SCHEMA != "chimera.recorder-normalized-day/1"
    note = document["note"].lower()
    assert "rebuildable" in note and "not evidence" in note and "safe to delete" in note


def test_the_cache_is_outside_everything_that_identifies_a_recording(tmp_path):
    """It must not appear in the day's metadata, its digest, or a day manifest."""
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(0))]})
    incremental = IncrementalNormalizer(root, CONTRACT)
    report = incremental.build_day("um", DAY)
    cache = incremental.cache_path("um", DAY)
    assert cache.exists()

    document = json.loads(
        MinuteNormalizer(root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
    )
    text = json.dumps(document)
    assert "cache" not in text, "the day's metadata mentions the cache"
    assert str(cache) not in text

    # And deleting it changes nothing about the day that comes back.
    cache.unlink()
    MinuteNormalizer(root, CONTRACT).parquet_path("um", DAY).unlink()
    MinuteNormalizer(root, CONTRACT).meta_path("um", DAY).unlink()
    again = IncrementalNormalizer(root, CONTRACT).build_day("um", DAY)
    assert again.digest == report.digest


def test_deleting_the_cache_costs_time_and_nothing_else(tmp_path):
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(5)]})
    incremental = IncrementalNormalizer(root, CONTRACT)
    before = incremental.build_day("um", DAY)
    incremental.drop("um", DAY)
    assert not incremental.cache_path("um", DAY).exists()
    after = incremental.build_day("um", DAY)
    assert after.digest == before.digest
    assert incremental.status[("um", DAY)].replayed_records == 5, "it folded the day again"


def test_the_full_rebuild_remains_available_as_the_oracle(tmp_path):
    """PR-04's path is untouched and still reachable, which is what makes the
    parity above a comparison rather than a tautology."""
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(3)]})
    normalizer = MinuteNormalizer(root, CONTRACT)
    report = normalizer.build_day("um", DAY)
    assert report.rows == 3
    frame = pd.read_parquet(normalizer.parquet_path("um", DAY))
    assert digest(frame, market="um") == report.digest
    assert minute_frame([], market="um").empty


def test_an_unreadable_raw_line_stops_the_fold_rather_than_skipping_it(tmp_path):
    """A record that will not parse is a defect to look at, not one to fold past."""
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(0))]})
    path = RawSink(root, UM_KLINE_1M, contract=CONTRACT).events_path(DAY)
    with path.open("ab") as handle:
        handle.write(b'{"schema": "chimera.recorder-raw-event/1", "stream": "nonsense"}\n')
    incremental = IncrementalNormalizer(root, CONTRACT)
    with pytest.raises(NormalizeCacheError, match="line"):
        incremental.update("um", DAY)
    # The public entry point falls back to the authoritative rebuild, which meets
    # the same unreadable record and refuses it too. Either way the day is not
    # written: a short normalized day would be the silent partial this refuses.
    with pytest.raises((RecorderNormalizeError, RecorderEventError)):
        incremental.build_day("um", DAY)
    assert not MinuteNormalizer(root, CONTRACT).parquet_path("um", DAY).exists()


def test_the_gzip_of_a_frozen_day_is_still_readable_by_the_oracle(tmp_path):
    """A sanity check on the fallback's input, so its parity claim is not empty."""
    root = tmp_path / "cursor"
    root.mkdir()
    write(root, {UM_KLINE_1M: [kline_event(minute_ms(i)) for i in range(3)]}, freeze=True)
    packed = RawSink(root, UM_KLINE_1M, contract=CONTRACT).gz_path(DAY)
    assert packed.exists()
    with gzip.open(packed, "rb") as handle:
        lines = [line for line in handle.read().split(b"\n") if line.strip()]
    assert len(lines) == 3
    assert RawEvent.from_line(lines[0]).canonical_ns == minute_ms(0) * NS_PER_MILLISECOND
