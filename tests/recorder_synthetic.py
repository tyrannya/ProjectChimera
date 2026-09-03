"""Synthetic payloads for the prospective recorder's offline tests.

Every byte here is invented. No Binance endpoint is contacted, no archive is
read, and nothing in this module or the tests that use it observes a real
market: the payload *shapes* are the documented ones so that the parsers are
exercised against what they will actually be handed, and the *values* are made
up so that a test can assert an exact number.

The same discipline as :mod:`tests.p13_synthetic`: a fixture that had to be
downloaded could not be broken in exactly one way, and breaking a fixture in
exactly one way is how a refusal gets tested.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from chimera.recorder.events import (
    NS_PER_MILLISECOND,
    BookTickerEvent,
    EventSource,
    FundingSettlement,
    KlineEvent,
    MarkPriceEvent,
    RawEvent,
    RecorderEventError,
    SPOT_BOOK_TICKER,
    SPOT_KLINE_1M,
    UM_BOOK_TICKER,
    UM_FUNDING,
    UM_KLINE_1M,
    UM_MARK_PRICE,
    day_start_ns,
)

#: A UTC day far enough in the future that no real recording can collide with
#: it, and a Thursday, so the funding schedule's 00:00 / 08:00 / 16:00 are
#: unremarkable.
DAY = "2026-09-19"
NEXT_DAY = "2026-09-20"


def day_ms(day: str = DAY) -> int:
    """The first minute of a UTC day, in milliseconds."""
    return day_start_ns(day) // NS_PER_MILLISECOND


def minute_ms(index: int, *, day: str = DAY) -> int:
    """The ``index``-th minute of a UTC day, in milliseconds."""
    return day_ms(day) + index * 60_000


def kline_ws_frame(
    open_ms: int,
    *,
    closed: bool = True,
    open_price: str = "60000.10",
    high: str = "60100.00",
    low: str = "59900.00",
    close: str = "60042.50",
    volume: str = "12.34567890",
    trades: int = 42,
    taker_buy_base: str = "6.00000000",
    taker_buy_quote: str = "360000.00",
    event_ms: int | None = None,
    symbol: str = "BTCUSDT",
) -> dict[str, Any]:
    """One Binance kline websocket frame, in the documented shape."""
    return {
        "e": "kline",
        "E": open_ms + 59_000 if event_ms is None else event_ms,
        "s": symbol,
        "k": {
            "t": open_ms,
            "T": open_ms + 59_999,
            "s": symbol,
            "i": "1m",
            "f": 100,
            "L": 200,
            "o": open_price,
            "c": close,
            "h": high,
            "l": low,
            "v": volume,
            "n": trades,
            "x": closed,
            "q": "740000.00",
            "V": taker_buy_base,
            "Q": taker_buy_quote,
            "B": "0",
        },
    }


def kline_rest_row(
    open_ms: int,
    *,
    open_price: str = "60000.10",
    high: str = "60100.00",
    low: str = "59900.00",
    close: str = "60042.50",
    volume: str = "12.34567890",
    trades: int = 42,
    taker_buy_base: str = "6.00000000",
    taker_buy_quote: str = "360000.00",
) -> list[Any]:
    """One Binance REST kline row: twelve fields, in the published order."""
    return [
        open_ms,
        open_price,
        high,
        low,
        close,
        volume,
        open_ms + 59_999,
        "740000.00",
        trades,
        taker_buy_base,
        taker_buy_quote,
        "0",
    ]


def mark_ws_frame(
    event_ms: int,
    *,
    mark: str = "60050.00",
    index: str = "60049.00",
    settle: str | None = "60050.50",
    rate: str = "0.00010000",
    next_funding_ms: int | None = None,
) -> dict[str, Any]:
    """One Binance USD-M mark-price websocket frame."""
    frame: dict[str, Any] = {
        "e": "markPriceUpdate",
        "E": event_ms,
        "s": "BTCUSDT",
        "p": mark,
        "i": index,
        "r": rate,
        "T": day_ms() + 8 * 3_600_000 if next_funding_ms is None else next_funding_ms,
    }
    if settle is not None:
        frame["P"] = settle
    return frame


def book_ws_frame(
    update_id: int,
    *,
    event_ms: int | None,
    bid: str = "59999.90",
    bid_qty: str = "3.00000000",
    ask: str = "60000.10",
    ask_qty: str = "2.00000000",
) -> dict[str, Any]:
    """One bookTicker frame. ``event_ms=None`` is the spot shape: no event time."""
    frame: dict[str, Any] = {
        "u": update_id,
        "s": "BTCUSDT",
        "b": bid,
        "B": bid_qty,
        "a": ask,
        "A": ask_qty,
    }
    if event_ms is not None:
        frame["e"] = "bookTicker"
        frame["E"] = event_ms
        frame["T"] = event_ms - 2
    return frame


def funding_rest_row(
    funding_time_ms: int,
    *,
    rate: str = "0.00012500",
    mark: str | None = "60050.00",
    rate_type: str | None = None,
) -> dict[str, Any]:
    """One Binance ``fundingRate`` REST row."""
    row: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "fundingTime": funding_time_ms,
        "fundingRate": rate,
    }
    if mark is not None:
        row["markPrice"] = mark
    if rate_type is not None:
        row["rateType"] = rate_type
    return row


def kline_event(
    open_ms: int,
    *,
    stream: str = UM_KLINE_1M,
    closed: bool = True,
    source: EventSource = EventSource.WEBSOCKET,
    receipt_wall_ns: int | None = None,
    receipt_mono_ns: int = 1,
    **overrides: Any,
) -> RawEvent:
    """A raw kline observation, websocket-shaped or REST-adapted."""
    if source is EventSource.WEBSOCKET:
        payload: Mapping[str, Any] = kline_ws_frame(open_ms, closed=closed, **overrides)
    else:
        if not closed:
            raise RecorderEventError("a REST kline row is always a closed minute")
        payload = KlineEvent.rest_payload(kline_rest_row(open_ms, **overrides))
    parsed = KlineEvent.from_payload(payload, stream=stream)
    wall = (
        (open_ms + 60_000) * NS_PER_MILLISECOND if receipt_wall_ns is None else receipt_wall_ns
    )
    return parsed.to_raw_event(
        payload,
        receipt_wall_ns=wall,
        receipt_mono_ns=receipt_mono_ns,
        source=source,
    )


def mark_event(
    event_ms: int,
    *,
    receipt_wall_ns: int | None = None,
    receipt_mono_ns: int = 1,
    **overrides: Any,
) -> RawEvent:
    """A raw mark-price observation."""
    payload = mark_ws_frame(event_ms, **overrides)
    wall = event_ms * NS_PER_MILLISECOND if receipt_wall_ns is None else receipt_wall_ns
    return MarkPriceEvent.from_payload(payload).to_raw_event(
        payload, receipt_wall_ns=wall, receipt_mono_ns=receipt_mono_ns
    )


def book_event(
    update_id: int,
    *,
    stream: str = UM_BOOK_TICKER,
    event_ms: int | None = None,
    receipt_wall_ns: int | None = None,
    receipt_mono_ns: int = 1,
    **overrides: Any,
) -> RawEvent:
    """A raw bookTicker observation.

    ``event_ms`` present is the perpetual shape and produces an exchange-stamped
    record; ``event_ms=None`` is the spot shape and produces a receipt-stamped
    one, which is the distinction :class:`chimera.recorder.events.TimeBasis`
    exists to keep visible.
    """
    payload = book_ws_frame(update_id, event_ms=event_ms, **overrides)
    if receipt_wall_ns is None:
        anchor = event_ms if event_ms is not None else day_ms()
        receipt_wall_ns = anchor * NS_PER_MILLISECOND
    return BookTickerEvent.from_payload(payload, stream=stream).to_raw_event(
        payload, receipt_wall_ns=receipt_wall_ns, receipt_mono_ns=receipt_mono_ns
    )


def funding_event(
    funding_time_ms: int,
    *,
    receipt_wall_ns: int | None = None,
    receipt_mono_ns: int = 1,
    **overrides: Any,
) -> RawEvent:
    """A raw funding settlement observation."""
    payload = funding_rest_row(funding_time_ms, **overrides)
    wall = (
        (funding_time_ms + 60_000) * NS_PER_MILLISECOND
        if receipt_wall_ns is None
        else receipt_wall_ns
    )
    return FundingSettlement.from_payload(payload).to_raw_event(
        payload, receipt_wall_ns=wall, receipt_mono_ns=receipt_mono_ns
    )


def um_day(
    minutes: Sequence[int],
    *,
    day: str = DAY,
    with_mark: bool = True,
    with_book: bool = True,
) -> dict[str, list[RawEvent]]:
    """A synthetic perpetual day: one closed kline per named minute index.

    ``minutes`` is a list of minute indices inside the day, so a caller can omit
    one and get a day with a real gap in it.
    """
    klines: list[RawEvent] = []
    marks: list[RawEvent] = []
    books: list[RawEvent] = []
    for index in minutes:
        opened = minute_ms(index, day=day)
        klines.append(kline_event(opened, close=f"6{index:04d}.50", receipt_mono_ns=index + 1))
        if with_mark:
            for offset, price in ((5_000, "60050.00"), (55_000, "60060.00")):
                marks.append(
                    mark_event(opened + offset, mark=price, receipt_mono_ns=index + 1)
                )
        if with_book:
            books.append(
                book_event(
                    1_000 + index,
                    event_ms=opened + 50_000,
                    receipt_mono_ns=index + 1,
                )
            )
    return {UM_KLINE_1M: klines, UM_MARK_PRICE: marks, UM_BOOK_TICKER: books}


def spot_day(minutes: Sequence[int], *, day: str = DAY) -> dict[str, list[RawEvent]]:
    """A synthetic spot day: closed klines and receipt-stamped book updates."""
    klines: list[RawEvent] = []
    books: list[RawEvent] = []
    for index in minutes:
        opened = minute_ms(index, day=day)
        klines.append(
            kline_event(
                opened,
                stream=SPOT_KLINE_1M,
                close=f"5{index:04d}.50",
                receipt_mono_ns=index + 1,
            )
        )
        books.append(
            book_event(
                7_000 + index,
                stream=SPOT_BOOK_TICKER,
                event_ms=None,
                receipt_wall_ns=(opened + 40_000) * NS_PER_MILLISECOND,
                receipt_mono_ns=index + 1,
            )
        )
    return {SPOT_KLINE_1M: klines, SPOT_BOOK_TICKER: books}


def funding_day(day: str = DAY) -> list[RawEvent]:
    """The three scheduled settlements of a UTC day."""
    base = day_ms(day)
    return [
        funding_event(base + hours * 3_600_000, rate=f"0.0001{hours:04d}")
        for hours in (0, 8, 16)
    ]


__all__ = [
    "DAY",
    "NEXT_DAY",
    "UM_FUNDING",
    "book_event",
    "book_ws_frame",
    "day_ms",
    "funding_day",
    "funding_event",
    "funding_rest_row",
    "kline_event",
    "kline_rest_row",
    "kline_ws_frame",
    "mark_event",
    "mark_ws_frame",
    "minute_ms",
    "spot_day",
    "um_day",
]
