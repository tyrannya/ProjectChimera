"""The venue's names, pinned: endpoints, stream names and payload fields.

Every other recorder test asks whether this repository is internally consistent.
This one asks a different question: **does what we believe about Binance still
match what we wrote down?** Nothing here reaches the network — it cannot answer
whether the exchange changed something — but it makes every such belief a single
named constant with a test beside it, so that when a name does change, exactly
one file has to be edited and the diff says what moved.

That is what section 4.1 asks for: "an endpoint discovery test that fails loudly
if a name or payload field changes".

**Three kinds of name are pinned here.**

* *Endpoints* — the three websocket bases and the six REST paths, and which
  endpoint publishes which stream. Their allow-list in
  ``tests/test_recorder_no_network.py`` is the security half of the same fact;
  this file is the correctness half. USD-M is two bases rather than one because
  the venue split its traffic by category on 2026-04-23, and the routing table
  is pinned here because getting it wrong is silent: the retired base still
  accepts a subscription and simply never sends.
* *Stream names* — ``btcusdt@kline_1m`` and friends, built from the contract's
  own symbol rather than hard-coded, so a contract naming a different instrument
  subscribes to that instrument.
* *Payload fields* — the letters. ``p``, ``i``, ``P``, ``r``, ``T`` on the mark
  stream; ``t``, ``T``, ``x`` on a kline; ``u`` on a book update, with ``E``
  present on USD-M and absent on spot; ``fundingTime``, ``fundingRate``,
  ``markPrice`` on a settlement. Each is asserted through the parser that reads
  it, so a test passes only if the field is both named correctly *and* used.

The USD-M/spot asymmetry is the single most load-bearing fact in the file. Spot
``bookTicker`` publishes an update id and no event time, so its records are
stamped with the local receipt clock and say so. A silent change on the venue's
side that started sending ``E`` would not break anything; a change in *this*
repository that assumed one had always been there would silently restamp a
stream. The tests below assert both directions.
"""

from __future__ import annotations

import pytest

from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.events import (
    REST_KLINE_FIELDS,
    SPOT_BOOK_TICKER,
    SPOT_KLINE_1M,
    UM_BOOK_TICKER,
    UM_FUNDING,
    UM_KLINE_1M,
    UM_MARK_PRICE,
    BookTickerEvent,
    FundingSettlement,
    KlineEvent,
    MarkPriceEvent,
    RecorderEventError,
    TimeBasis,
)
from chimera.recorder.rest import (
    INTERVAL_1M,
    KLINE_ENDPOINTS,
    SPOT_KLINES_PATH,
    SPOT_REST_BASE,
    UM_FUNDING_RATE_PATH,
    UM_INDEX_PRICE_KLINES_PATH,
    UM_KLINES_PATH,
    UM_MARK_PRICE_KLINES_PATH,
    UM_PREMIUM_INDEX_PATH,
    UM_REST_BASE,
    PREMIUM_INDEX_FIELDS,
    kline_request,
    premium_index_payload,
)
from chimera.recorder.streams import (
    BOOK_TICKER_SUFFIX,
    ENDPOINT_WS_BASES,
    EVENT_TYPE_BOOK_TICKER,
    EVENT_TYPE_KLINE,
    EVENT_TYPE_MARK_PRICE,
    KLINE_1M_SUFFIX,
    MARK_PRICE_SUFFIX,
    SPOT_WS_BASE,
    UM_MARKET_WS_BASE,
    UM_PUBLIC_WS_BASE,
    Endpoint,
    StreamKind,
    clients_for,
    endpoint_for,
    frame_kind,
    subscriptions_for,
    venue_stream_name,
)
from tests.recorder_synthetic import (
    book_ws_frame,
    funding_rest_row,
    kline_rest_row,
    kline_ws_frame,
    mark_ws_frame,
    minute_ms,
    premium_index_row,
)

CONTRACT = load_recorder_contract()
OPEN_MS = minute_ms(0)


# --- A. endpoints ---------------------------------------------------------------
#: The USD-M bases Binance retired on 2026-04-23. Named so the regression guard
#: below can say what must never come back, and not merely what should be there.
#: The retired base still connects and still answers SUBSCRIBE with success, so a
#: build that returned to it would satisfy every check that only asks whether a
#: socket is healthy — and would record nothing at all on two of three streams.
LEGACY_UM_WS_BASES = (
    "wss://fstream.binance.com/ws",
    "wss://fstream.binance.com/stream",
)


def test_the_three_websocket_bases_are_the_ones_the_venue_publishes_now():
    """USD-M has been two endpoints since 2026-04-23; spot is still one."""
    assert UM_MARKET_WS_BASE == "wss://fstream.binance.com/market/ws"
    assert UM_PUBLIC_WS_BASE == "wss://fstream.binance.com/public/ws"
    assert SPOT_WS_BASE == "wss://stream.binance.com:9443/ws"
    bases = (UM_MARKET_WS_BASE, UM_PUBLIC_WS_BASE, SPOT_WS_BASE)
    assert len(set(bases)) == 3, "each endpoint is its own connection"
    for base in bases:
        assert base.startswith("wss://"), "market data is read over TLS"
        assert base.endswith("/ws"), "the raw-stream endpoint, not the combined one"


def test_every_endpoint_has_a_base_and_the_table_is_exhaustive():
    """An endpoint with no base would raise at startup, not connect somewhere."""
    assert set(ENDPOINT_WS_BASES) == set(Endpoint)
    assert ENDPOINT_WS_BASES[Endpoint.UM_MARKET] == UM_MARKET_WS_BASE
    assert ENDPOINT_WS_BASES[Endpoint.UM_PUBLIC] == UM_PUBLIC_WS_BASE
    assert ENDPOINT_WS_BASES[Endpoint.SPOT] == SPOT_WS_BASE


def test_no_base_is_one_the_venue_retired():
    """The legacy base is not merely different. It is silently wrong."""
    assert set(ENDPOINT_WS_BASES.values()).isdisjoint(LEGACY_UM_WS_BASES)


# --- A2. which endpoint publishes which stream ----------------------------------
def test_the_usd_m_kline_and_mark_price_are_market_traffic():
    """Both stopped arriving on the retired base; both are Market category."""
    assert endpoint_for(UM_KLINE_1M) is Endpoint.UM_MARKET
    assert endpoint_for(UM_MARK_PRICE) is Endpoint.UM_MARKET


def test_the_usd_m_book_is_public_traffic():
    """Which is exactly why it alone kept arriving on the retired base."""
    assert endpoint_for(UM_BOOK_TICKER) is Endpoint.UM_PUBLIC


def test_spot_routing_did_not_change():
    assert endpoint_for(SPOT_KLINE_1M) is Endpoint.SPOT
    assert endpoint_for(SPOT_BOOK_TICKER) is Endpoint.SPOT


def test_funding_is_published_on_no_websocket_endpoint():
    assert endpoint_for(UM_FUNDING) is None


# --- A3. the connections the service opens --------------------------------------
def _clients():
    return clients_for(CONTRACT, lambda event: None)


def test_clients_are_split_by_endpoint_and_not_by_market():
    """Three connections, because USD-M's streams live on two endpoints."""
    by_name = {client.name: client for client in _clients()}
    assert set(by_name) == {"um-market-ws", "um-public-ws", "spot-ws"}
    assert by_name["um-market-ws"].url == UM_MARKET_WS_BASE
    assert by_name["um-public-ws"].url == UM_PUBLIC_WS_BASE
    assert by_name["spot-ws"].url == SPOT_WS_BASE
    assert set(by_name["um-market-ws"].stream_ids) == {UM_KLINE_1M, UM_MARK_PRICE}
    assert set(by_name["um-public-ws"].stream_ids) == {UM_BOOK_TICKER}
    assert set(by_name["spot-ws"].stream_ids) == {SPOT_KLINE_1M, SPOT_BOOK_TICKER}


def test_one_usd_m_connection_would_be_the_defect_coming_back():
    """The regression guard. A single socket carrying all three USD-M streams is
    the shape that connects, reports healthy, and never receives a kline."""
    clients = _clients()
    um = [client for client in clients if "fstream.binance.com" in client.url]
    assert len(um) == 2, "USD-M is two connections; one of them is the old defect"
    for client in clients:
        assert client.url not in LEGACY_UM_WS_BASES
        streams = set(client.stream_ids)
        assert (
            not {UM_KLINE_1M, UM_BOOK_TICKER} <= streams
        ), "Market and Public traffic on one socket is what the venue split apart"


def test_no_recorder_stream_is_subscribed_on_two_connections():
    """A stream carried twice would record every event twice."""
    seen: list[str] = []
    for client in _clients():
        seen.extend(client.stream_ids)
    assert len(seen) == len(set(seen)), "a stream is subscribed on two connections"
    assert set(seen) == {
        UM_KLINE_1M,
        UM_MARK_PRICE,
        UM_BOOK_TICKER,
        SPOT_KLINE_1M,
        SPOT_BOOK_TICKER,
    }
    assert UM_FUNDING not in seen, "funding is polled over REST, not subscribed"


def test_every_connection_carries_only_streams_its_endpoint_publishes():
    """The url a client dials and the streams it asks for cannot disagree."""
    for client in _clients():
        endpoints = {endpoint_for(stream_id) for stream_id in client.stream_ids}
        assert len(endpoints) == 1, f"{client.name} mixes endpoints"
        assert ENDPOINT_WS_BASES[endpoints.pop()] == client.url


def test_the_rest_paths_are_the_public_market_data_ones():
    assert UM_REST_BASE == "https://fapi.binance.com"
    assert SPOT_REST_BASE == "https://api.binance.com"
    assert UM_KLINES_PATH == "/fapi/v1/klines"
    assert UM_MARK_PRICE_KLINES_PATH == "/fapi/v1/markPriceKlines"
    assert UM_INDEX_PRICE_KLINES_PATH == "/fapi/v1/indexPriceKlines"
    assert UM_FUNDING_RATE_PATH == "/fapi/v1/fundingRate"
    assert UM_PREMIUM_INDEX_PATH == "/fapi/v1/premiumIndex"
    assert SPOT_KLINES_PATH == "/api/v3/klines"


def test_each_market_klines_come_from_its_own_host():
    """A spot candle fetched from the perpetual host would be a different market."""
    assert KLINE_ENDPOINTS["um"] == (UM_REST_BASE, UM_KLINES_PATH)
    assert KLINE_ENDPOINTS["spot"] == (SPOT_REST_BASE, SPOT_KLINES_PATH)


def test_the_kline_request_names_the_symbol_interval_and_range():
    request = kline_request("um", "btcusdt", OPEN_MS, OPEN_MS + 60_000, 500)
    assert request.method == "GET"
    assert request.url == "https://fapi.binance.com/fapi/v1/klines"
    assert request.params == {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "startTime": OPEN_MS,
        "endTime": OPEN_MS + 60_000,
        "limit": 500,
    }
    assert INTERVAL_1M == "1m", "the recorder records one clock and it is the minute"
    assert "signature" not in request.params
    assert "apiKey" not in request.params


def test_the_spot_kline_request_goes_to_the_spot_host():
    request = kline_request("spot", "BTCUSDT", OPEN_MS, OPEN_MS + 60_000, 10)
    assert request.url == "https://api.binance.com/api/v3/klines"


# --- B. stream names ------------------------------------------------------------
def test_the_stream_suffixes_are_the_published_ones():
    assert KLINE_1M_SUFFIX == "kline_1m"
    assert MARK_PRICE_SUFFIX == "markPrice@1s"
    assert BOOK_TICKER_SUFFIX == "bookTicker"


def test_a_venue_stream_name_is_the_lowercase_symbol_and_the_suffix():
    assert venue_stream_name("BTCUSDT", KLINE_1M_SUFFIX) == "btcusdt@kline_1m"
    assert venue_stream_name("btcusdt", MARK_PRICE_SUFFIX) == "btcusdt@markPrice@1s"
    assert venue_stream_name(" BTCUSDT ", BOOK_TICKER_SUFFIX) == "btcusdt@bookTicker"


def test_the_subscriptions_come_from_the_contract_and_not_from_a_list():
    """A contract naming another instrument would subscribe to that instrument."""
    um = {s.stream_id: s.venue_name for s in subscriptions_for(CONTRACT, "um")}
    spot = {s.stream_id: s.venue_name for s in subscriptions_for(CONTRACT, "spot")}
    assert um == {
        UM_KLINE_1M: "btcusdt@kline_1m",
        UM_MARK_PRICE: "btcusdt@markPrice@1s",
        UM_BOOK_TICKER: "btcusdt@bookTicker",
    }
    assert spot == {
        SPOT_KLINE_1M: "btcusdt@kline_1m",
        SPOT_BOOK_TICKER: "btcusdt@bookTicker",
    }


def test_funding_is_not_a_websocket_subscription():
    """It is published over REST, and subscribing to a stream that does not exist
    would leave the recorder waiting for frames that never arrive."""
    assert UM_FUNDING in CONTRACT.streams
    assert UM_FUNDING not in {s.stream_id for s in subscriptions_for(CONTRACT, "um")}


def test_no_index_or_mark_minute_stream_is_subscribed():
    """Section 4.1 derives both from ``um.markPrice``; PR-06 reconciles them."""
    subscribed = {s.stream_id for s in subscriptions_for(CONTRACT, "um")}
    assert "um.indexPrice_1m" not in subscribed
    assert "um.markPrice_1m" not in subscribed


# --- C. which frame is which ----------------------------------------------------
def test_the_event_type_field_names_are_the_published_ones():
    assert EVENT_TYPE_KLINE == "kline"
    assert EVENT_TYPE_MARK_PRICE == "markPriceUpdate"
    assert EVENT_TYPE_BOOK_TICKER == "bookTicker"


def test_a_frame_is_classified_by_its_event_type_where_it_has_one():
    assert frame_kind(kline_ws_frame(OPEN_MS)) is StreamKind.KLINE
    assert frame_kind(mark_ws_frame(OPEN_MS)) is StreamKind.MARK_PRICE
    assert frame_kind(book_ws_frame(7, event_ms=OPEN_MS)) is StreamKind.BOOK_TICKER


def test_the_spot_book_frame_is_classified_by_its_shape_because_it_has_no_event_type():
    """Binance's spot ``bookTicker`` publishes no ``e`` at all.

    If it ever starts publishing one, the branch above catches it and this test
    is the record of what was true when the recorder was written.
    """
    frame = book_ws_frame(7, event_ms=None)
    assert "e" not in frame, "the spot shape carries no event type"
    assert frame_kind(frame) is StreamKind.BOOK_TICKER


def test_a_subscribe_acknowledgement_is_not_market_data():
    assert frame_kind({"result": None, "id": 1}) is None
    assert frame_kind({"e": "somethingElse", "E": 1}) is None
    assert frame_kind({}) is None


# --- D. payload fields: kline ---------------------------------------------------
def test_the_kline_frame_fields_are_read_by_their_published_letters():
    frame = kline_ws_frame(OPEN_MS, closed=True)
    parsed = KlineEvent.from_ws(frame, stream=UM_KLINE_1M)
    candle = frame["k"]
    assert parsed.open_ms == candle["t"], "t is the open time and the minute key"
    assert parsed.close_ms == candle["T"], "T is the close time"
    assert parsed.closed is candle["x"], "x is the closed flag"
    assert parsed.open == candle["o"]
    assert parsed.high == candle["h"]
    assert parsed.low == candle["l"]
    assert parsed.close == candle["c"]
    assert parsed.volume == candle["v"]
    assert parsed.trades == candle["n"]
    assert parsed.taker_buy_base == candle["V"]
    assert parsed.taker_buy_quote == candle["Q"]
    assert parsed.event_ms == frame["E"]


def test_the_closed_flag_is_the_whole_difference_between_a_candle_and_a_frame():
    forming = KlineEvent.from_ws(kline_ws_frame(OPEN_MS, closed=False), stream=UM_KLINE_1M)
    finished = KlineEvent.from_ws(kline_ws_frame(OPEN_MS, closed=True), stream=UM_KLINE_1M)
    assert forming.closed is False and finished.closed is True
    assert forming.open_ms == finished.open_ms, "the same minute, said twice"


def test_the_rest_kline_row_columns_are_the_published_twelve_in_order():
    assert REST_KLINE_FIELDS == (
        "t",
        "o",
        "h",
        "l",
        "c",
        "v",
        "T",
        "q",
        "n",
        "V",
        "Q",
        "B",
    )
    row = kline_rest_row(OPEN_MS)
    assert len(row) == len(REST_KLINE_FIELDS)
    parsed = KlineEvent.from_rest(row, stream=UM_KLINE_1M)
    assert parsed.open_ms == row[0] and parsed.close_ms == row[6]
    assert parsed.closed is True, "a REST row for a past minute is a finished candle"


def test_a_rest_row_of_the_wrong_width_is_refused_rather_than_padded():
    with pytest.raises(RecorderEventError, match="fields"):
        KlineEvent.from_rest(kline_rest_row(OPEN_MS)[:-1], stream=UM_KLINE_1M)


# --- E. payload fields: mark price ----------------------------------------------
def test_the_mark_price_frame_carries_p_i_P_r_and_T():
    """The five values section 4.1 lists, each read from its own letter.

    This is the reason there is no separate index subscription: everything the
    derived index and mark minutes need is on this one frame.
    """
    frame = mark_ws_frame(OPEN_MS + 1_000)
    parsed = MarkPriceEvent.from_ws(frame)
    assert parsed.mark == frame["p"], "p is the mark price"
    assert parsed.index == frame["i"], "i is the index price"
    assert parsed.estimated_settle == frame["P"], "P is the estimated settlement price"
    assert parsed.funding_rate == frame["r"], "r is the funding rate in effect"
    assert parsed.next_funding_ms == frame["T"], "T is the next funding time"
    assert parsed.event_ms == frame["E"], "E is the event time and the canonical stamp"


def test_the_estimated_settle_price_is_optional_and_its_absence_is_not_a_zero():
    parsed = MarkPriceEvent.from_ws(mark_ws_frame(OPEN_MS, settle=None))
    assert parsed.estimated_settle is None
    assert parsed.mark and parsed.index, "the rest of the frame still parses"


def test_a_mark_frame_missing_a_required_letter_is_refused():
    frame = mark_ws_frame(OPEN_MS)
    del frame["i"]
    with pytest.raises(RecorderEventError, match="'i'"):
        MarkPriceEvent.from_ws(frame)


# --- F. payload fields: book ticker ---------------------------------------------
def test_the_usd_m_book_frame_carries_an_update_id_and_two_timestamps():
    frame = book_ws_frame(1234, event_ms=OPEN_MS + 500)
    parsed = BookTickerEvent.from_ws(frame, stream=UM_BOOK_TICKER)
    assert parsed.update_id == frame["u"], "u is the order-book update id"
    assert parsed.event_ms == frame["E"], "E is the event time"
    assert parsed.transaction_ms == frame["T"], "T is the transaction time"
    assert parsed.bid == frame["b"] and parsed.bid_qty == frame["B"]
    assert parsed.ask == frame["a"] and parsed.ask_qty == frame["A"]
    assert parsed.has_exchange_time is True


def test_the_spot_book_frame_has_no_event_time_and_says_so_in_the_record():
    """The asymmetry, asserted from both sides.

    Spot publishes ``u`` and the four book fields. The record it produces is
    stamped with the local receipt clock and carries ``TimeBasis.RECEIPT``, so
    nothing downstream has to guess which clock it is reading.
    """
    frame = book_ws_frame(1234, event_ms=None)
    parsed = BookTickerEvent.from_ws(frame, stream=SPOT_BOOK_TICKER)
    assert parsed.event_ms is None and parsed.transaction_ms is None
    assert parsed.has_exchange_time is False
    assert parsed.update_id == frame["u"]

    receipt = OPEN_MS * 1_000_000 + 12_345
    event = parsed.to_raw_event(frame, receipt_wall_ns=receipt, receipt_mono_ns=7)
    assert event.time_basis is TimeBasis.RECEIPT
    assert event.canonical_ns == receipt, "the receipt clock is the stamp, and is said"

    perpetual = BookTickerEvent.from_ws(
        book_ws_frame(1235, event_ms=OPEN_MS + 500), stream=UM_BOOK_TICKER
    )
    perp_event = perpetual.to_raw_event(
        book_ws_frame(1235, event_ms=OPEN_MS + 500), receipt_wall_ns=receipt, receipt_mono_ns=7
    )
    assert perp_event.time_basis is TimeBasis.EXCHANGE
    assert perp_event.canonical_ns == (OPEN_MS + 500) * 1_000_000


def test_the_book_dedup_key_is_the_update_id_alone():
    """Section 4.1's rule for this stream, and the reason ordering works."""
    frame = book_ws_frame(99, event_ms=OPEN_MS)
    parsed = BookTickerEvent.from_ws(frame, stream=UM_BOOK_TICKER)
    event = parsed.to_raw_event(frame, receipt_wall_ns=OPEN_MS * 1_000_000, receipt_mono_ns=1)
    assert event.dedup_key == "book:99"


# --- G. payload fields: funding -------------------------------------------------
def test_the_funding_row_carries_time_rate_and_the_exchanges_own_mark():
    row = funding_rest_row(OPEN_MS, rate="0.00012500", mark="60050.00")
    parsed = FundingSettlement.from_rest(row)
    assert parsed.funding_time_ms == row["fundingTime"]
    assert parsed.funding_rate == row["fundingRate"]
    assert parsed.mark_price == row["markPrice"]
    assert parsed.symbol == row["symbol"]
    assert parsed.settlement_id == row["fundingTime"], "the instant identifies it"
    assert parsed.stream == UM_FUNDING


def test_the_rate_type_field_is_carried_when_the_venue_sends_one():
    parsed = FundingSettlement.from_rest(funding_rest_row(OPEN_MS, rate_type="ADJUSTED"))
    assert parsed.rate_type == "ADJUSTED"
    assert FundingSettlement.from_rest(funding_rest_row(OPEN_MS)).rate_type is None


def test_a_funding_row_without_a_funding_time_is_refused():
    row = funding_rest_row(OPEN_MS)
    del row["fundingTime"]
    with pytest.raises(RecorderEventError, match="fundingTime"):
        FundingSettlement.from_rest(row)


# --- H. payload fields: premium index -------------------------------------------
def test_the_premium_index_response_renames_onto_the_mark_stream_letters():
    """The endpoint publishes the same five values under long names.

    The adapter is a rename and nothing else, which is what lets a polled reading
    and a pushed one be parsed by the same code and stored in the same stream.
    """
    assert PREMIUM_INDEX_FIELDS == {
        "time": "E",
        "markPrice": "p",
        "indexPrice": "i",
        "estimatedSettlePrice": "P",
        "lastFundingRate": "r",
        "nextFundingTime": "T",
    }
    row = premium_index_row(OPEN_MS + 3_000)
    payload = premium_index_payload(row)
    parsed = MarkPriceEvent.from_payload(payload, stream=UM_MARK_PRICE)
    assert parsed.event_ms == row["time"]
    assert parsed.mark == row["markPrice"]
    assert parsed.index == row["indexPrice"]
    assert parsed.estimated_settle == row["estimatedSettlePrice"]
    assert parsed.funding_rate == row["lastFundingRate"]
    assert parsed.next_funding_ms == row["nextFundingTime"]


def test_the_premium_index_rate_is_current_state_and_never_a_settlement():
    """``lastFundingRate`` is the rate in effect, not a payment that happened.

    A realised settlement comes from ``fundingRate`` and is recorded on
    ``um.funding``; this value is recorded on ``um.markPrice``, exactly where the
    websocket puts the same number.
    """
    payload = premium_index_payload(premium_index_row(OPEN_MS))
    assert set(payload) <= {"E", "p", "i", "P", "r", "T", "s"}
    assert "fundingTime" not in payload and "fundingRate" not in payload
    with pytest.raises(RecorderEventError):
        FundingSettlement.from_payload(payload)


def test_a_premium_index_response_missing_a_value_is_refused_not_defaulted():
    row = premium_index_row(OPEN_MS)
    del row["indexPrice"]
    with pytest.raises(Exception, match="indexPrice"):
        premium_index_payload(row)
