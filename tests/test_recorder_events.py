"""What one recorded observation is, and what it refuses to be.

The property this file exists to protect is the one that is invisible in a
plot and fatal in a research record: **the exchange's clock and this host's
clock are different facts and the record never conflates them.** Binance's
perpetual bookTicker publishes an event time; its spot bookTicker publishes an
update id and nothing else. A record that stamped both with a number called
"canonical" and said no more would have quietly made a local clock reading into
an exchange timestamp, and no later test could tell which was which.

Everything else here is the ordinary discipline of a persisted record: integer
instants, no timezone-naive anything, deterministic bytes, explicit failure on a
payload that cannot be serialised, and a deduplication key that is a function of
the observation rather than of the process that saw it.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from chimera.recorder.events import (
    MAX_CANONICAL_NS,
    MINUTES_PER_DAY,
    MS_PER_MINUTE,
    NS_PER_DAY,
    NS_PER_MILLISECOND,
    RAW_EVENT_SCHEMA,
    REST_KLINE_FIELDS,
    SPOT_BOOK_TICKER,
    SPOT_KLINE_1M,
    UM_BOOK_TICKER,
    UM_FUNDING,
    UM_KLINE_1M,
    UM_MARK_PRICE,
    BookTickerEvent,
    EventSource,
    FundingSettlement,
    KlineEvent,
    MarkPriceEvent,
    RawEvent,
    RecorderEventError,
    TimeBasis,
    canonical_json,
    day_start_ns,
    iso_utc,
    kline_dedup_key,
    minute_open_ms,
    payload_digest,
    require_canonical_ns,
    sort_events,
    utc_day,
)

from tests.recorder_synthetic import (
    DAY,
    NEXT_DAY,
    book_ws_frame,
    day_ms,
    funding_rest_row,
    kline_event,
    kline_rest_row,
    kline_ws_frame,
    mark_ws_frame,
    minute_ms,
)

MINUTE = minute_ms(0)
WALL = (MINUTE + 60_000) * NS_PER_MILLISECOND


def raw(**overrides: object) -> RawEvent:
    fields: dict[str, object] = {
        "stream": UM_KLINE_1M,
        "canonical_ns": MINUTE * NS_PER_MILLISECOND,
        "time_basis": TimeBasis.EXCHANGE,
        "receipt_wall_ns": WALL,
        "receipt_mono_ns": 12_345,
        "source": EventSource.WEBSOCKET,
        "dedup_key": "kline:1:2:C:abc",
        "payload": {"k": {"t": MINUTE}},
    }
    fields.update(overrides)
    return RawEvent(**fields)  # type: ignore[arg-type]


# --- A. time ------------------------------------------------------------------
def test_a_utc_day_is_computed_by_integer_arithmetic():
    start = day_start_ns(DAY)
    assert utc_day(start) == DAY
    assert utc_day(start + NS_PER_DAY - 1) == DAY
    assert utc_day(start + NS_PER_DAY) == NEXT_DAY
    assert day_start_ns(NEXT_DAY) - start == NS_PER_DAY


def test_a_minute_owns_its_open_and_not_its_close():
    """Half-open by construction, which is what makes a boundary event unambiguous."""
    opened = minute_ms(7) * NS_PER_MILLISECOND
    assert minute_open_ms(opened) == minute_ms(7)
    assert minute_open_ms(opened + 59_999 * NS_PER_MILLISECOND) == minute_ms(7)
    assert minute_open_ms(opened + 60_000 * NS_PER_MILLISECOND) == minute_ms(8)
    assert minute_open_ms(opened - 1) == minute_ms(6)


def test_a_day_holds_exactly_fourteen_hundred_and_forty_minutes():
    start = day_start_ns(DAY) // NS_PER_MILLISECOND
    minutes = range(start, start + MINUTES_PER_DAY * MS_PER_MINUTE, MS_PER_MINUTE)
    assert len(list(minutes)) == MINUTES_PER_DAY
    assert utc_day(list(minutes)[-1] * NS_PER_MILLISECOND) == DAY


def test_an_iso_rendering_is_utc_and_matches_the_integer_it_came_from():
    instant = day_start_ns(DAY) + 3_723 * 1_000_000_000
    assert iso_utc(instant) == "2026-09-19T01:02:03+00:00"
    parsed = datetime.fromisoformat(iso_utc(instant))
    assert parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0


@pytest.mark.parametrize(
    "value",
    [1.0, True, False, "1700000000000000000", None, -1, MAX_CANONICAL_NS + 1],
    ids=["float", "true", "false", "string", "none", "negative", "far-future"],
)
def test_a_timestamp_that_is_not_an_integer_utc_instant_is_refused(value):
    with pytest.raises(RecorderEventError):
        require_canonical_ns(value)


def test_a_day_string_that_is_not_a_day_is_refused():
    with pytest.raises(RecorderEventError, match="not a YYYY-MM-DD"):
        day_start_ns("19-09-2026")


# --- B. the generic record -----------------------------------------------------
def test_a_record_round_trips_through_its_canonical_line():
    event = raw()
    restored = RawEvent.from_line(event.canonical_line())
    assert restored == event
    assert restored.canonical_line() == event.canonical_line()


def test_the_canonical_line_is_one_utf8_line_ending_in_a_bare_newline():
    line = raw(payload={"note": "прайс"}).canonical_line()
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    assert b"\r" not in line
    assert "прайс".encode("utf-8") in line, "ensure_ascii=False keeps the text as UTF-8"


def test_the_canonical_line_does_not_depend_on_the_payload_s_key_order():
    first = raw(payload={"a": 1, "b": {"y": 2, "x": 3}})
    second = raw(payload={"b": {"x": 3, "y": 2}, "a": 1})
    assert first.canonical_line() == second.canonical_line()


def test_the_record_names_its_schema_and_its_labels():
    record = json.loads(raw().canonical_line())
    assert record["schema"] == RAW_EVENT_SCHEMA
    assert record["time_basis"] == "EXCHANGE"
    assert record["source"] == "WEBSOCKET"
    assert set(record) == {
        "schema",
        "stream",
        "canonical_ns",
        "time_basis",
        "receipt_wall_ns",
        "receipt_mono_ns",
        "source",
        "dedup_key",
        "payload",
    }


def test_the_day_and_minute_of_a_record_come_from_canonical_time():
    event = raw(canonical_ns=day_start_ns(NEXT_DAY))
    assert event.day == NEXT_DAY
    assert event.minute_open_ms == day_ms(NEXT_DAY)


def test_a_stored_payload_cannot_be_mutated_through_the_event():
    payload = {"k": {"t": MINUTE}}
    event = raw(payload=payload)
    payload["k"] = {"t": 0}
    assert event.payload["k"] == {"t": MINUTE}
    with pytest.raises(TypeError):
        event.payload["k"] = {"t": 0}  # type: ignore[index]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        pytest.param({"stream": "umkline"}, "<market>.<stream> id", id="stream"),
        pytest.param({"time_basis": "EXCHANGE"}, "must be a TimeBasis", id="basis-string"),
        pytest.param({"source": "WEBSOCKET"}, "must be an EventSource", id="source-string"),
        pytest.param(
            {"canonical_ns": 1.5}, "integer nanosecond instant", id="canonical-float"
        ),
        pytest.param({"receipt_wall_ns": -5}, "outside", id="receipt-negative"),
        pytest.param({"receipt_mono_ns": -1}, "is negative", id="mono-negative"),
        pytest.param({"receipt_mono_ns": 1.0}, "must be an integer", id="mono-float"),
        pytest.param({"dedup_key": ""}, "non-empty string", id="key-empty"),
        pytest.param({"dedup_key": "a" * 257}, "256 printable", id="key-long"),
        pytest.param({"dedup_key": "a\nb"}, "256 printable", id="key-newline"),
        pytest.param({"payload": [1, 2]}, "must be a mapping", id="payload-list"),
        pytest.param({"payload": {"p": float("nan")}}, "deterministically", id="payload-nan"),
        pytest.param({"payload": {"p": {1, 2}}}, "deterministically", id="payload-set"),
    ],
)
def test_a_record_that_cannot_be_honest_is_refused_at_construction(overrides, expected):
    with pytest.raises(RecorderEventError, match=expected):
        raw(**overrides)


def test_a_receipt_basis_must_actually_describe_the_value_beside_it():
    with pytest.raises(RecorderEventError, match="must be equal"):
        raw(time_basis=TimeBasis.RECEIPT, canonical_ns=MINUTE * NS_PER_MILLISECOND)
    honest = raw(time_basis=TimeBasis.RECEIPT, canonical_ns=WALL)
    assert honest.canonical_ns == honest.receipt_wall_ns


@pytest.mark.parametrize(
    "mutation, expected",
    [
        pytest.param({"schema": "chimera.recorder-raw-event/2"}, "schema is", id="schema"),
        pytest.param({"time_basis": "GUESSED"}, "unknown label", id="basis"),
        pytest.param({"source": "CARRIER_PIGEON"}, "unknown label", id="source"),
        pytest.param({"extra": 1}, "unexpected", id="unknown-key"),
    ],
)
def test_a_stored_record_this_build_cannot_read_is_refused(mutation, expected):
    record = json.loads(raw().canonical_line())
    record.update(mutation)
    with pytest.raises(RecorderEventError, match=expected):
        RawEvent.from_record(record)


def test_a_record_missing_a_field_is_refused():
    record = json.loads(raw().canonical_line())
    record.pop("dedup_key")
    with pytest.raises(RecorderEventError, match="missing"):
        RawEvent.from_record(record)


def test_a_line_that_is_not_json_is_refused():
    with pytest.raises(RecorderEventError, match="not readable JSON"):
        RawEvent.from_line(b'{"schema": "chimera.recorder-raw-event/1", "stre')


def test_canonical_json_refuses_the_values_that_are_not_json():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    for value in (float("inf"), float("nan"), float("-inf")):
        with pytest.raises(RecorderEventError):
            canonical_json({"x": value})


def test_a_payload_digest_is_a_function_of_the_values_not_of_the_order():
    assert payload_digest({"a": 1, "b": 2}) == payload_digest({"b": 2, "a": 1})
    assert payload_digest({"a": 1}) != payload_digest({"a": 2})


def test_events_sort_by_canonical_time_with_file_order_as_the_tie_break():
    first = raw(canonical_ns=200, dedup_key="first")
    second = raw(canonical_ns=100, dedup_key="second")
    third = raw(canonical_ns=200, dedup_key="third")
    ordered = sort_events([first, second, third])
    assert [event.dedup_key for event in ordered] == ["second", "first", "third"]


# --- C. klines ----------------------------------------------------------------
def test_a_kline_is_stamped_by_its_open_and_carries_the_published_fields():
    frame = kline_ws_frame(MINUTE)
    candle = KlineEvent.from_ws(frame, stream=UM_KLINE_1M)
    assert candle.open_ms == MINUTE
    assert candle.close_ms == MINUTE + 59_999
    assert candle.closed is True
    assert candle.canonical_ns == MINUTE * NS_PER_MILLISECOND
    assert candle.event_ms == MINUTE + 59_000
    assert (candle.open, candle.high, candle.low, candle.close) == (
        "60000.10",
        "60100.00",
        "59900.00",
        "60042.50",
    )
    assert candle.volume == "12.34567890"
    assert candle.trades == 42
    assert candle.taker_buy_base == "6.00000000"
    assert candle.taker_buy_quote == "360000.00"


def test_a_rest_row_adapts_onto_the_same_shape_without_dropping_a_field():
    row = kline_rest_row(MINUTE)
    payload = KlineEvent.rest_payload(row)
    assert set(payload) == {"k"}
    assert set(payload["k"]) == set(REST_KLINE_FIELDS) | {"x"}
    assert payload["k"]["x"] is True
    assert [payload["k"][name] for name in REST_KLINE_FIELDS] == list(row)

    from_rest = KlineEvent.from_rest(row, stream=UM_KLINE_1M)
    from_ws = KlineEvent.from_ws(kline_ws_frame(MINUTE), stream=UM_KLINE_1M)
    assert from_rest.event_ms is None, "a REST row publishes no push time"
    for field in ("open_ms", "close_ms", "closed", "open", "high", "low", "close", "volume"):
        assert getattr(from_rest, field) == getattr(from_ws, field)


@pytest.mark.parametrize(
    "row, expected",
    [
        pytest.param(kline_rest_row(MINUTE)[:-1], "12 fields", id="short"),
        pytest.param(kline_rest_row(MINUTE) + [0], "12 fields", id="long"),
        pytest.param("not a row", "must be a sequence", id="string"),
        pytest.param({"t": MINUTE}, "must be a sequence", id="mapping"),
    ],
)
def test_a_rest_row_of_the_wrong_shape_is_refused(row, expected):
    with pytest.raises(RecorderEventError, match=expected):
        KlineEvent.rest_payload(row)


@pytest.mark.parametrize("field", ["t", "T", "x", "o", "h", "l", "c", "v", "n", "V", "Q"])
def test_a_kline_missing_a_field_the_normalizer_reads_is_refused(field):
    frame = kline_ws_frame(MINUTE)
    frame["k"].pop(field)
    with pytest.raises(RecorderEventError, match=f"missing the .* field '{field}'"):
        KlineEvent.from_ws(frame, stream=UM_KLINE_1M)


@pytest.mark.parametrize(
    "field, value, expected",
    [
        pytest.param("t", "1700000000000", "must be an integer", id="open-as-string"),
        pytest.param("n", 1.5, "must be an integer", id="trades-float"),
        pytest.param("x", "true", "must be a boolean", id="closed-as-string"),
        pytest.param("o", 60000.1, "decimal string", id="price-as-float"),
        pytest.param("c", "", "decimal string", id="price-blank"),
    ],
)
def test_a_kline_field_of_the_wrong_type_is_refused(field, value, expected):
    frame = kline_ws_frame(MINUTE)
    frame["k"][field] = value
    with pytest.raises(RecorderEventError, match=expected):
        KlineEvent.from_ws(frame, stream=UM_KLINE_1M)


def test_a_frame_without_a_candle_object_is_refused():
    with pytest.raises(RecorderEventError, match="payload.k must be an object"):
        KlineEvent.from_ws({"e": "kline", "E": 1}, stream=UM_KLINE_1M)


def test_prices_are_kept_as_the_exchange_s_own_decimal_strings():
    """No float conversion at the raw layer: what was said is what is stored."""
    frame = kline_ws_frame(MINUTE, close="60042.50")
    stored = KlineEvent.from_ws(frame, stream=UM_KLINE_1M).to_raw_event(
        frame, receipt_wall_ns=WALL, receipt_mono_ns=1, source=EventSource.WEBSOCKET
    )
    assert stored.payload["k"]["c"] == "60042.50"
    assert b'"c":"60042.50"' in stored.canonical_line()


# --- D. deduplication ----------------------------------------------------------
def test_the_same_frame_twice_has_the_same_key_and_needs_no_shared_state():
    frame = kline_ws_frame(MINUTE)
    first = KlineEvent.from_ws(frame, stream=UM_KLINE_1M)
    second = KlineEvent.from_payload(json.loads(json.dumps(frame)), stream=UM_KLINE_1M)
    assert kline_dedup_key(first, frame) == kline_dedup_key(second, frame)


def test_a_partial_frame_is_not_a_duplicate_of_the_closed_one():
    """Otherwise the first tick of a minute would eat the candle for that minute."""
    partial = kline_ws_frame(MINUTE, closed=False)
    closed = kline_ws_frame(MINUTE, closed=True)
    keys = {
        kline_dedup_key(KlineEvent.from_ws(partial, stream=UM_KLINE_1M), partial),
        kline_dedup_key(KlineEvent.from_ws(closed, stream=UM_KLINE_1M), closed),
    }
    assert len(keys) == 2


def test_a_websocket_close_and_a_rest_gapfill_of_one_minute_are_both_kept():
    """The reconciliation compares them; a dedup that merged them would hide a conflict."""
    ws_frame = kline_ws_frame(MINUTE)
    rest_payload = KlineEvent.rest_payload(kline_rest_row(MINUTE))
    ws_key = kline_dedup_key(KlineEvent.from_ws(ws_frame, stream=UM_KLINE_1M), ws_frame)
    rest_key = kline_dedup_key(
        KlineEvent.from_payload(rest_payload, stream=UM_KLINE_1M), rest_payload
    )
    assert ws_key != rest_key
    assert ws_key.startswith(f"kline:{MINUTE}:{MINUTE + 59_999}:C:")
    assert rest_key.startswith(f"kline:{MINUTE}:{MINUTE + 59_999}:C:")


def test_two_different_candles_never_share_a_key():
    left = kline_ws_frame(MINUTE, close="60042.50")
    right = kline_ws_frame(MINUTE, close="60042.51")
    assert kline_dedup_key(KlineEvent.from_ws(left, stream=UM_KLINE_1M), left) != (
        kline_dedup_key(KlineEvent.from_ws(right, stream=UM_KLINE_1M), right)
    )


def test_a_book_update_is_identified_by_the_exchange_s_own_sequence():
    event = BookTickerEvent.from_ws(
        book_ws_frame(4009, event_ms=MINUTE), stream=UM_BOOK_TICKER
    )
    stored = event.to_raw_event(
        book_ws_frame(4009, event_ms=MINUTE), receipt_wall_ns=WALL, receipt_mono_ns=1
    )
    assert stored.dedup_key == "book:4009"


def test_a_funding_settlement_is_identified_by_its_instant_and_its_payload():
    """The settlement id is the instant; the observation key is the instant plus the bytes.

    Re-fetching an overlapping window is a duplicate, so a poller can overlap
    freely. Two readings of one settlement that disagree are two observations, so
    the disagreement survives to the reconciliation instead of being resolved in
    favour of whichever arrived first.
    """
    instant = day_ms() + 8 * 3_600_000
    row = funding_rest_row(instant)
    settlement = FundingSettlement.from_rest(row)
    stored = settlement.to_raw_event(row, receipt_wall_ns=WALL, receipt_mono_ns=1)

    assert settlement.settlement_id == instant
    assert stored.dedup_key.startswith(f"funding:{instant}:")
    assert stored.source is EventSource.REST_POLL

    refetched = funding_rest_row(instant)
    again = FundingSettlement.from_rest(refetched).to_raw_event(
        refetched, receipt_wall_ns=WALL + 1, receipt_mono_ns=2
    )
    assert again.dedup_key == stored.dedup_key

    revised = funding_rest_row(instant, rate="0.00099999")
    other = FundingSettlement.from_rest(revised).to_raw_event(
        revised, receipt_wall_ns=WALL + 2, receipt_mono_ns=3
    )
    assert other.dedup_key != stored.dedup_key


# --- E. canonical versus receipt time ------------------------------------------
def test_a_perpetual_book_update_is_stamped_by_the_exchange():
    frame = book_ws_frame(11, event_ms=MINUTE + 30_000)
    stored = BookTickerEvent.from_ws(frame, stream=UM_BOOK_TICKER).to_raw_event(
        frame, receipt_wall_ns=WALL, receipt_mono_ns=7
    )
    assert stored.time_basis is TimeBasis.EXCHANGE
    assert stored.canonical_ns == (MINUTE + 30_000) * NS_PER_MILLISECOND
    assert stored.receipt_wall_ns == WALL
    assert stored.canonical_ns != stored.receipt_wall_ns


def test_a_spot_book_update_says_that_its_stamp_is_this_host_s_clock():
    """Binance spot bookTicker publishes an update id and no timestamp at all."""
    frame = book_ws_frame(12, event_ms=None)
    parsed = BookTickerEvent.from_ws(frame, stream=SPOT_BOOK_TICKER)
    assert parsed.has_exchange_time is False
    assert parsed.event_ms is None

    stored = parsed.to_raw_event(frame, receipt_wall_ns=WALL, receipt_mono_ns=7)
    assert stored.time_basis is TimeBasis.RECEIPT
    assert stored.canonical_ns == WALL
    assert json.loads(stored.canonical_line())["time_basis"] == "RECEIPT"


def test_a_mark_update_is_stamped_by_its_event_time():
    frame = mark_ws_frame(MINUTE + 1_500)
    parsed = MarkPriceEvent.from_ws(frame)
    assert parsed.mark == "60050.00"
    assert parsed.index == "60049.00"
    assert parsed.estimated_settle == "60050.50"
    assert parsed.funding_rate == "0.00010000"
    assert parsed.next_funding_ms == day_ms() + 8 * 3_600_000
    stored = parsed.to_raw_event(frame, receipt_wall_ns=WALL, receipt_mono_ns=1)
    assert stored.time_basis is TimeBasis.EXCHANGE
    assert stored.canonical_ns == (MINUTE + 1_500) * NS_PER_MILLISECOND
    assert stored.stream == UM_MARK_PRICE


def test_a_mark_update_may_omit_the_estimated_settlement_price():
    parsed = MarkPriceEvent.from_ws(mark_ws_frame(MINUTE, settle=None))
    assert parsed.estimated_settle is None


def test_a_funding_row_keeps_the_optional_fields_it_was_given():
    with_type = FundingSettlement.from_rest(funding_rest_row(day_ms(), rate_type="daily"))
    assert with_type.rate_type == "daily"
    assert with_type.mark_price == "60050.00"
    without = FundingSettlement.from_rest(funding_rest_row(day_ms(), mark=None))
    assert without.mark_price is None
    assert without.rate_type is None
    assert without.stream == UM_FUNDING


def test_a_funding_row_without_a_symbol_is_refused():
    row = funding_rest_row(day_ms())
    row.pop("symbol")
    with pytest.raises(RecorderEventError, match="'symbol'"):
        FundingSettlement.from_rest(row)


def test_a_settlement_record_carries_the_publication_and_the_receipt():
    settlement = FundingSettlement.from_rest(funding_rest_row(day_ms() + 8 * 3_600_000))
    record = settlement.to_settlement_record(receipt_wall_ns=WALL)
    assert record["funding_time_ms"] == day_ms() + 8 * 3_600_000
    assert record["funding_time_utc"] == "2026-09-19T08:00:00+00:00"
    assert record["funding_rate"] == "0.00012500"
    assert record["receipt_wall_ns"] == WALL
    assert set(record) == {
        "funding_time_ms",
        "funding_time_utc",
        "funding_rate",
        "mark_price",
        "rate_type",
        "symbol",
        "receipt_wall_ns",
    }


# --- F. day boundaries ---------------------------------------------------------
def test_a_candle_that_opens_in_the_last_minute_of_a_day_belongs_to_that_day():
    last = minute_ms(MINUTES_PER_DAY - 1)
    event = kline_event(last)
    assert event.day == DAY
    assert event.minute_open_ms == last
    assert utc_day(event.canonical_ns + 60_000 * NS_PER_MILLISECOND) == NEXT_DAY


def test_the_first_candle_of_the_next_day_belongs_to_the_next_day():
    event = kline_event(minute_ms(0, day=NEXT_DAY))
    assert event.day == NEXT_DAY


def test_a_spot_stream_and_a_perpetual_stream_are_different_identities():
    perp = kline_event(MINUTE, stream=UM_KLINE_1M)
    spot = kline_event(MINUTE, stream=SPOT_KLINE_1M)
    assert perp.stream != spot.stream
    assert perp.dedup_key == spot.dedup_key, (
        "the key identifies the observation; the sink scopes it by stream, which is why "
        "one sink owns one stream"
    )
    assert perp.canonical_line() != spot.canonical_line()
