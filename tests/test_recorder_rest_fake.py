"""The REST poller, against a local fake rather than against Binance.

No test in this file touches the network. Every request is served by an
in-process fake — a stand-in for ``requests.Session`` that records what it was
asked for and answers with what the test told it to answer — because the things
that have to be tested here are the ones a real endpoint will not produce on
demand: a timeout, a 429 with a ``Retry-After``, a truncated body, a row of the
wrong width, a page that does not advance.

**What is asserted.** That the poller asks the right host for the right thing;
that it pages a long range instead of silently truncating it; that it retries
what is worth retrying and refuses what is not; that a malformed answer raises
with the reason attached rather than becoming a default value; and that not one
request carries a credential or a signature.

The fake is deliberately dumb. It does not simulate Binance — it replays a
scripted list of responses — because a clever fake would be a second
implementation of the thing under test, and the first bug they shared would be
invisible.
"""

from __future__ import annotations

import json

import pytest
import requests

from chimera.recorder.events import SPOT_KLINE_1M, UM_FUNDING, UM_KLINE_1M
from chimera.recorder.rest import (
    DEFAULT_MIN_INTERVAL_S,
    EXPECTED_FUNDING_HOURS_UTC,
    MAX_RETRY_AFTER_S,
    RecorderRestError,
    RestFailure,
    RestPoller,
    expected_funding_instants_ms,
    premium_index_payload,
)
from tests.recorder_synthetic import (
    funding_rest_row,
    kline_rest_row,
    minute_ms,
    premium_index_row,
)

OPEN_MS = minute_ms(0)
SYMBOL = "BTCUSDT"


class FakeResponse:
    """The subset of ``requests.Response`` the poller reads."""

    def __init__(self, status_code=200, payload=None, *, text=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is _INVALID:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


_INVALID = object()


class FakeSession:
    """Replays scripted answers and records every call. Opens no socket."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if not self.answers:
            raise AssertionError(f"the fake was asked for {url} with nothing left to answer")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]

    @property
    def params(self) -> list[dict]:
        return [params for _, params in self.calls]


def poller(*answers, **options) -> RestPoller:
    """A poller wired to a fake session, with the sleep replaced by a recorder."""
    options.setdefault("min_interval_s", 0.0)
    slept: list[float] = []
    instance = RestPoller(session=FakeSession(answers), sleep=slept.append, **options)
    instance.slept = slept  # type: ignore[attr-defined]
    return instance


# --- A. the requests that go out ------------------------------------------------
def test_um_klines_are_asked_of_the_perpetual_host():
    api = poller(FakeResponse(payload=[kline_rest_row(OPEN_MS)]))
    api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert api.session.urls == ["https://fapi.binance.com/fapi/v1/klines"]
    assert api.session.params[0] == {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "startTime": OPEN_MS,
        "endTime": OPEN_MS,
        "limit": 1500,
    }


def test_spot_klines_are_asked_of_the_spot_host():
    api = poller(FakeResponse(payload=[kline_rest_row(OPEN_MS)]))
    api.klines("spot", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert api.session.urls == ["https://api.binance.com/api/v3/klines"]


def test_an_unknown_market_is_refused_rather_than_guessed():
    api = poller()
    with pytest.raises(RecorderRestError, match="no public kline endpoint") as excinfo:
        api.klines("perp", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert excinfo.value.failure is RestFailure.SHAPE


def test_funding_and_the_premium_index_are_perpetual_endpoints():
    api = poller(
        FakeResponse(payload=[funding_rest_row(OPEN_MS)]),
        FakeResponse(payload=premium_index_row(OPEN_MS)),
    )
    api.funding_rate(OPEN_MS, OPEN_MS + 1, symbol=SYMBOL)
    api.premium_index(symbol=SYMBOL)
    assert api.session.urls == [
        "https://fapi.binance.com/fapi/v1/fundingRate",
        "https://fapi.binance.com/fapi/v1/premiumIndex",
    ]
    assert api.session.params[1] == {"symbol": "BTCUSDT"}


def test_the_price_kline_endpoints_are_reachable_and_keyed_as_the_venue_keys_them():
    """Mark price is keyed by ``symbol``; index price by ``pair``.

    Both are part of the adopted REST interface and neither is used by PR-05:
    the recorder derives its per-minute mark and index from the ``markPrice``
    stream, and comparing those against these archives is PR-06's reconciliation.
    """
    api = poller(FakeResponse(payload=[]), FakeResponse(payload=[]))
    api.mark_price_klines(OPEN_MS, OPEN_MS + 60_000, symbol=SYMBOL)
    api.index_price_klines(OPEN_MS, OPEN_MS + 60_000, pair=SYMBOL)
    assert api.session.urls == [
        "https://fapi.binance.com/fapi/v1/markPriceKlines",
        "https://fapi.binance.com/fapi/v1/indexPriceKlines",
    ]
    assert "symbol" in api.session.params[0] and "pair" not in api.session.params[0]
    assert "pair" in api.session.params[1] and "symbol" not in api.session.params[1]


def test_no_request_carries_a_credential_or_a_signature():
    """The security property, asserted about what actually went out."""
    api = poller(
        FakeResponse(payload=[kline_rest_row(OPEN_MS)]),
        FakeResponse(payload=[funding_rest_row(OPEN_MS)]),
        FakeResponse(payload=premium_index_row(OPEN_MS)),
    )
    api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    api.funding_rate(OPEN_MS, OPEN_MS + 1, symbol=SYMBOL)
    api.premium_index(symbol=SYMBOL)
    for params in api.session.params:
        keys = {key.lower() for key in params}
        assert not keys & {"signature", "apikey", "api_key", "timestamp", "recvwindow"}
    assert not getattr(api.session, "headers", {}), "the fake session was given no headers"


def test_a_range_that_ends_before_it_starts_is_refused():
    api = poller()
    with pytest.raises(RecorderRestError, match="before start_ms"):
        api.klines("um", OPEN_MS + 60_000, OPEN_MS, symbol=SYMBOL)
    with pytest.raises(RecorderRestError, match="integer millisecond"):
        api.klines("um", "yesterday", OPEN_MS, symbol=SYMBOL)  # type: ignore[arg-type]


# --- B. paging ------------------------------------------------------------------
def test_a_long_range_is_paged_from_the_last_row_received():
    """Three minutes at a page size of two: two calls, three rows, no duplicate."""
    first = [kline_rest_row(OPEN_MS), kline_rest_row(OPEN_MS + 60_000)]
    second = [kline_rest_row(OPEN_MS + 120_000)]
    api = poller(FakeResponse(payload=first), FakeResponse(payload=second))
    rows = api.klines("um", OPEN_MS, OPEN_MS + 120_000, symbol=SYMBOL, limit=2)
    assert [row[0] for row in rows] == [OPEN_MS, OPEN_MS + 60_000, OPEN_MS + 120_000]
    assert len(api.session.calls) == 2
    assert api.session.params[1]["startTime"] == OPEN_MS + 120_000


def test_a_short_page_ends_the_walk():
    api = poller(FakeResponse(payload=[kline_rest_row(OPEN_MS)]))
    rows = api.klines("um", OPEN_MS, OPEN_MS + 600_000, symbol=SYMBOL, limit=500)
    assert len(rows) == 1 and len(api.session.calls) == 1


def test_an_empty_page_ends_the_walk_rather_than_looping():
    api = poller(FakeResponse(payload=[]))
    assert api.klines("um", OPEN_MS, OPEN_MS + 600_000, symbol=SYMBOL, limit=2) == []


def test_a_page_that_does_not_advance_is_refused_rather_than_fetched_for_ever():
    """The failure mode a paging loop has: the same page, returned again."""
    page = [kline_rest_row(OPEN_MS), kline_rest_row(OPEN_MS)]
    api = poller(FakeResponse(payload=page), FakeResponse(payload=page))
    with pytest.raises(RecorderRestError, match="went backwards"):
        api.klines("um", OPEN_MS + 120_000, OPEN_MS + 600_000, symbol=SYMBOL, limit=2)


# --- C. parsing -----------------------------------------------------------------
def test_kline_events_are_parsed_under_the_right_stream_id():
    api = poller(
        FakeResponse(payload=[kline_rest_row(OPEN_MS)]),
        FakeResponse(payload=[kline_rest_row(OPEN_MS)]),
    )
    um = api.kline_events("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    spot = api.kline_events("spot", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert um[0].stream == UM_KLINE_1M and spot[0].stream == SPOT_KLINE_1M
    assert um[0].closed is True


def test_funding_settlements_are_parsed_under_the_funding_stream():
    api = poller(FakeResponse(payload=[funding_rest_row(OPEN_MS)]))
    settlements = api.funding_settlements(OPEN_MS, OPEN_MS + 1, symbol=SYMBOL)
    assert [s.stream for s in settlements] == [UM_FUNDING]
    assert settlements[0].settlement_id == OPEN_MS


def test_a_malformed_row_raises_with_the_reason_rather_than_being_dropped():
    api = poller(FakeResponse(payload=[kline_rest_row(OPEN_MS)[:-2]]))
    with pytest.raises(RecorderRestError, match="not one") as excinfo:
        api.kline_events("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert excinfo.value.failure is RestFailure.SHAPE


def test_an_error_object_where_a_list_was_expected_is_named_as_one():
    """Binance answers errors with ``{"code": ..., "msg": ...}``."""
    api = poller(FakeResponse(payload={"code": -1121, "msg": "Invalid symbol."}))
    with pytest.raises(RecorderRestError, match="object rather than a list") as excinfo:
        api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert excinfo.value.failure is RestFailure.SHAPE
    assert "Invalid symbol" in str(excinfo.value)


def test_a_premium_index_list_where_an_object_was_expected_is_refused():
    api = poller(FakeResponse(payload=[premium_index_row(OPEN_MS)]))
    with pytest.raises(RecorderRestError, match="object for one symbol"):
        api.premium_index(symbol=SYMBOL)


def test_a_body_that_is_not_json_is_a_decode_failure_and_is_not_retried():
    api = poller(FakeResponse(payload=_INVALID, text="<html>502</html>"))
    with pytest.raises(RecorderRestError) as excinfo:
        api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert excinfo.value.failure is RestFailure.DECODE
    assert len(api.session.calls) == 1


def test_nothing_is_silently_defaulted_when_a_published_field_is_missing():
    row = premium_index_row(OPEN_MS)
    del row["markPrice"]
    with pytest.raises(RecorderRestError, match="markPrice") as excinfo:
        premium_index_payload(row)
    assert excinfo.value.failure is RestFailure.SHAPE


# --- D. failure and retry -------------------------------------------------------
def test_a_timeout_is_retried_and_then_raised_as_a_timeout():
    api = poller(
        requests.Timeout("read timed out"),
        requests.Timeout("read timed out"),
        max_attempts=2,
    )
    with pytest.raises(RecorderRestError) as excinfo:
        api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert excinfo.value.failure is RestFailure.TIMEOUT
    assert len(api.session.calls) == 2 and api.retries == 1


def test_a_transport_failure_is_retried_and_then_recovers():
    api = poller(
        requests.ConnectionError("connection reset"),
        FakeResponse(payload=[kline_rest_row(OPEN_MS)]),
        max_attempts=3,
    )
    rows = api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert len(rows) == 1 and api.retries == 1
    assert api.slept, "a retry waits before asking again"


def test_a_server_error_is_retried_and_a_client_error_is_not():
    """A 500 is the endpoint's problem; a 400 is the request's, and asking twice
    changes nothing except how fast it is refused."""
    server = poller(
        FakeResponse(500, text="internal error"),
        FakeResponse(payload=[kline_rest_row(OPEN_MS)]),
        max_attempts=3,
    )
    assert len(server.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)) == 1
    assert len(server.session.calls) == 2

    client = poller(FakeResponse(400, text="bad symbol"), max_attempts=3)
    with pytest.raises(RecorderRestError) as excinfo:
        client.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert excinfo.value.failure is RestFailure.STATUS
    assert len(client.session.calls) == 1, "a client error is not retried"


def test_a_rate_limit_waits_for_retry_after_and_then_succeeds():
    api = poller(
        FakeResponse(429, text="too many requests", headers={"Retry-After": "7"}),
        FakeResponse(payload=[kline_rest_row(OPEN_MS)]),
        max_attempts=3,
    )
    assert len(api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)) == 1
    assert api.rate_limited == 1
    assert api.slept == [7.0], "the endpoint said how long to wait, so it waited that long"


def test_a_ban_status_is_treated_as_a_rate_limit_too():
    api = poller(
        FakeResponse(418, text="banned", headers={"Retry-After": "1"}),
        FakeResponse(payload=[]),
        max_attempts=2,
    )
    api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert api.rate_limited == 1


def test_an_absurd_retry_after_is_capped():
    """A header asking for a two-hour wait must not stall the recorder for two hours."""
    api = poller(
        FakeResponse(429, text="", headers={"Retry-After": "100000"}),
        FakeResponse(payload=[]),
        max_attempts=2,
    )
    api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert api.slept == [MAX_RETRY_AFTER_S]


def test_a_nonsense_retry_after_falls_back_to_the_default_delay():
    api = poller(
        FakeResponse(429, text="", headers={"Retry-After": "soon"}),
        FakeResponse(payload=[]),
        max_attempts=2,
    )
    api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert api.slept == [1.0]


def test_a_rate_limit_on_the_last_attempt_raises_rather_than_returning_nothing():
    api = poller(FakeResponse(429, text="", headers={"Retry-After": "1"}), max_attempts=1)
    with pytest.raises(RecorderRestError) as excinfo:
        api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert excinfo.value.failure is RestFailure.RATE_LIMITED


def test_the_minimum_interval_keeps_a_retry_loop_from_becoming_a_tight_loop():
    clock = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    api = RestPoller(
        session=FakeSession([FakeResponse(payload=[]), FakeResponse(payload=[])]),
        sleep=lambda seconds: None,
        mono=lambda: next(clock),
        min_interval_s=DEFAULT_MIN_INTERVAL_S,
    )
    waits: list[float] = []
    api._sleep = waits.append
    api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    assert waits and waits[-1] == pytest.approx(DEFAULT_MIN_INTERVAL_S)


def test_max_attempts_below_one_is_refused_at_construction():
    with pytest.raises(RecorderRestError, match="max_attempts"):
        RestPoller(max_attempts=0)


def test_the_poller_counts_what_it_did():
    api = poller(
        FakeResponse(payload=[kline_rest_row(OPEN_MS)]),
        FakeResponse(payload=[funding_rest_row(OPEN_MS)]),
    )
    api.klines("um", OPEN_MS, OPEN_MS, symbol=SYMBOL)
    api.funding_rate(OPEN_MS, OPEN_MS + 1, symbol=SYMBOL)
    assert api.requests_made == 2 and api.retries == 0 and api.rate_limited == 0


# --- E. the funding schedule is a schedule, not a claim -------------------------
def test_the_expected_funding_instants_are_the_cadence_in_force():
    day = minute_ms(0)
    assert expected_funding_instants_ms(day) == tuple(
        day + hour * 3_600_000 for hour in EXPECTED_FUNDING_HOURS_UTC
    )
    assert EXPECTED_FUNDING_HOURS_UTC == (0, 8, 16)


def test_the_poller_records_whatever_the_endpoint_returns_and_checks_no_count():
    """A day that settled four times, or once, is recorded as it happened.

    The schedule above says when to *ask*. What was actually scheduled for a day
    is established from the archive by PR-06, and nothing here compares the
    number of rows against a constant.
    """
    rows = [funding_rest_row(minute_ms(0) + hour * 3_600_000) for hour in (0, 4, 8, 12)]
    api = poller(FakeResponse(payload=rows))
    settlements = api.funding_settlements(
        minute_ms(0), minute_ms(0) + 86_400_000, symbol=SYMBOL
    )
    assert len(settlements) == 4, "four rows in, four settlements out, no opinion"


def test_a_day_the_endpoint_reports_nothing_for_returns_nothing_and_raises_nothing():
    api = poller(FakeResponse(payload=[]))
    assert api.funding_settlements(OPEN_MS, OPEN_MS + 86_400_000, symbol=SYMBOL) == []
