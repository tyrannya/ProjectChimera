"""Binance public REST endpoints: kline gap-fill, funding settlements, premium index.

The websocket streams are the recorder's primary source. This module is the
secondary one, and it exists for the two things a push stream cannot do:

* **fill a gap.** A disconnect loses whatever the exchange published while the
  socket was down. After every reconnect the service asks this poller for the
  closed klines it may have missed, and those rows enter the *same* raw sink,
  through the same parsers, labelled :attr:`EventSource.REST_GAPFILL`. Nothing
  is repaired in the normalized layer, nothing is interpolated, and a minute
  neither source produced stays missing.
* **read a value that is published rather than pushed.** Realised funding
  settlements arrive on ``GET /fapi/v1/fundingRate`` and nowhere else, so they
  are polled on the cadence section 4.1 fixes.

**Public, unauthenticated, unsigned.** Every path below is a public market-data
endpoint. This module sends no API key, computes no signature and reads no
credential from anywhere — there is no header, no query parameter and no
environment variable here that could carry one, and
``tests/test_recorder_no_network.py`` asserts it about the source rather than
trusting this paragraph.

**Synchronous on purpose.** Section 12.3 specifies ``RestPoller(base_url,
session)``, and the session is a :class:`requests.Session`, which the repository
already depends on. The recorder's service is asyncio, and it calls these
methods through :func:`asyncio.to_thread`: a blocking call moved off the loop is
easier to reason about — and much easier to test — than a second HTTP stack
pulled in for the four requests a day this makes.

**Nothing is silently defaulted.** A malformed row, a payload of the wrong
shape, a status the endpoint should not return: each raises
:class:`RecorderRestError` carrying what was asked for and what came back. The
caller decides whether that is fatal; this module never invents a value to keep
going.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import requests

from chimera.recorder.events import (
    SPOT_KLINE_1M,
    UM_FUNDING,
    UM_KLINE_1M,
    FundingSettlement,
    KlineEvent,
    RecorderEventError,
)

logger = logging.getLogger(__name__)

#: The public REST bases, as section 4.1 names them. Both serve market data
#: without authentication.
UM_REST_BASE = "https://fapi.binance.com"
SPOT_REST_BASE = "https://api.binance.com"

#: Paths. USD-M lives under ``/fapi/v1``; spot under ``/api/v3``.
UM_KLINES_PATH = "/fapi/v1/klines"
UM_MARK_PRICE_KLINES_PATH = "/fapi/v1/markPriceKlines"
UM_INDEX_PRICE_KLINES_PATH = "/fapi/v1/indexPriceKlines"
UM_FUNDING_RATE_PATH = "/fapi/v1/fundingRate"
UM_PREMIUM_INDEX_PATH = "/fapi/v1/premiumIndex"
SPOT_KLINES_PATH = "/api/v3/klines"

#: Which base and path serve one market's 1m klines.
KLINE_ENDPOINTS: Mapping[str, tuple[str, str]] = {
    "um": (UM_REST_BASE, UM_KLINES_PATH),
    "spot": (SPOT_REST_BASE, SPOT_KLINES_PATH),
}

#: The recorder stream id each market's klines are recorded under.
KLINE_STREAM_IDS: Mapping[str, str] = {"um": UM_KLINE_1M, "spot": SPOT_KLINE_1M}

#: The interval every kline endpoint here is asked for. The recorder records one
#: clock; a second interval would be a second dataset nothing reads.
INTERVAL_1M = "1m"

#: Binance's documented maximum rows per kline page, and per funding page.
#: USD-M and spot do not agree, and the disagreement is not loud: asking spot's
#: ``/api/v3/klines`` for 1500 rows does not fail, it silently answers with at
#: most 1000. A paginator that treated "fewer rows than I asked for" as "the
#: range is exhausted" would therefore stop after one page on spot and report a
#: truncated fetch as a complete one — which is exactly what a cold start did,
#: gap-filling 1000 of 1440 spot minutes and calling it done. The per-market
#: ceiling below is what pagination is measured against, never the caller's ask.
MAX_KLINE_LIMIT = 1500
MAX_SPOT_KLINE_LIMIT = 1000
MAX_FUNDING_LIMIT = 1000

#: Market -> the most rows that market's kline endpoint will return in one page.
#: ``MAX_KLINE_LIMIT`` also governs the mark- and index-price kline paths, which
#: are USD-M and unpaged.
MAX_KLINE_PAGE: Mapping[str, int] = {"um": MAX_KLINE_LIMIT, "spot": MAX_SPOT_KLINE_LIMIT}

#: Statuses that mean "you are asking too often". 418 is Binance's own: an IP
#: that ignored a 429 for long enough. Both carry ``Retry-After``.
RATE_LIMITED_STATUSES = frozenset({429, 418})

#: Ceilings on how long a single call may spend being polite. A poller that
#: retried forever would look healthy while recording nothing.
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_S = 1.0
DEFAULT_MAX_BACKOFF_S = 30.0
MAX_RETRY_AFTER_S = 120.0
DEFAULT_TIMEOUT_S = 15.0

#: A floor on the interval between two requests from one poller. Not a rate
#: limiter — the exchange's own limits are much higher than anything here asks
#: for — just a guard against a retry loop becoming a tight loop.
DEFAULT_MIN_INTERVAL_S = 0.05

#: The eight-hour funding cadence in force at the time of writing, and the delay
#: after each expected settlement before the poll. The *count* of settlements a
#: day actually had is never assumed from this: it is read from what the
#: endpoint returns, and establishing what was scheduled is PR-06's job.
EXPECTED_FUNDING_HOURS_UTC: tuple[int, ...] = (0, 8, 16)
FUNDING_POLL_DELAY_S = 60.0
FUNDING_CATCHUP_INTERVAL_S = 3600.0


class RestFailure(str, Enum):
    """Why a request did not produce a usable answer. A bounded, logged label."""

    TRANSPORT = "TRANSPORT"
    TIMEOUT = "TIMEOUT"
    STATUS = "STATUS"
    RATE_LIMITED = "RATE_LIMITED"
    DECODE = "DECODE"
    SHAPE = "SHAPE"


class RecorderRestError(RuntimeError):
    """A REST call failed, or answered with something that is not what it claims."""

    def __init__(self, failure: RestFailure, message: str) -> None:
        super().__init__(message)
        self.failure = failure


@dataclass(frozen=True)
class RestRequest:
    """One prepared call. Built and asserted by tests before anything is sent."""

    method: str
    url: str
    params: Mapping[str, Any]


def kline_request(
    market: str, symbol: str, start_ms: int, end_ms: int, limit: int
) -> RestRequest:
    """The exact GET one kline page is fetched with."""
    try:
        base, path = KLINE_ENDPOINTS[market]
    except KeyError:
        raise RecorderRestError(
            RestFailure.SHAPE,
            f"no public kline endpoint is known for market {market!r}; "
            f"this build knows {sorted(KLINE_ENDPOINTS)}",
        ) from None
    return RestRequest(
        method="GET",
        url=base + path,
        params={
            "symbol": symbol.upper(),
            "interval": INTERVAL_1M,
            "startTime": int(start_ms),
            "endTime": int(end_ms),
            "limit": int(limit),
        },
    )


class RestPoller:
    """Public REST access for one exchange, over one :class:`requests.Session`.

    ``base_url`` is the USD-M host, because most of these endpoints are USD-M
    only; the spot host is a separate argument because spot klines live on it
    and nothing else here does.
    """

    def __init__(
        self,
        base_url: str = UM_REST_BASE,
        session: requests.Session | None = None,
        *,
        spot_base_url: str = SPOT_REST_BASE,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_s: float = DEFAULT_BACKOFF_S,
        max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        sleep: Callable[[float], None] = time.sleep,
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts < 1:
            raise RecorderRestError(
                RestFailure.SHAPE, f"max_attempts must be >= 1, got {max_attempts}"
            )
        self.base_url = base_url.rstrip("/")
        self.spot_base_url = spot_base_url.rstrip("/")
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout
        self.max_attempts = int(max_attempts)
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self.min_interval_s = min_interval_s
        self._sleep = sleep
        self._mono = mono
        self._last_request_at: float | None = None
        self.requests_made = 0
        self.retries = 0
        self.rate_limited = 0

    # --- the endpoints ----------------------------------------------------
    def klines(
        self,
        market: str,
        start_ms: int,
        end_ms: int,
        *,
        symbol: str,
        limit: int | None = None,
    ) -> list[list[Any]]:
        """Closed 1m klines in ``[start_ms, end_ms]``, paged until exhausted.

        Binance's kline range is inclusive at both ends and the endpoint returns
        at most one page, so a longer range is walked one page at a time from the
        open time of the last row received. A page that does not advance ends the
        walk rather than looping forever.

        ``limit`` is the caller's *request*, clamped to the market's own ceiling
        (:data:`MAX_KLINE_PAGE`) and defaulting to it. Pagination is decided
        against that clamped value, never against what the caller asked for: the
        venue applies its own cap without saying so, and a full page that is
        smaller than the ask is the middle of a range, not the end of one.
        """
        start, end = _require_range(start_ms, end_ms)
        base, path = self._kline_endpoint(market)
        page_limit = self._kline_page_limit(market, limit)
        rows: list[list[Any]] = []
        cursor = start
        while cursor <= end:
            page = self._get(
                base + path,
                {
                    "symbol": symbol.upper(),
                    "interval": INTERVAL_1M,
                    "startTime": cursor,
                    "endTime": end,
                    "limit": page_limit,
                },
            )
            batch = _require_rows(page, f"{market} klines")
            if not batch:
                break
            rows.extend(batch)
            last_open = _row_open_ms(batch[-1], f"{market} klines")
            if last_open < cursor:
                raise RecorderRestError(
                    RestFailure.SHAPE,
                    f"{market} klines went backwards: asked from {cursor}, last row opens at "
                    f"{last_open}. A page that does not advance is not a page",
                )
            cursor = last_open + 60_000
            if len(batch) < page_limit:
                break
        return rows

    def kline_events(
        self, market: str, start_ms: int, end_ms: int, *, symbol: str
    ) -> list[KlineEvent]:
        """:meth:`klines`, parsed. Every row must parse or the call fails."""
        stream = KLINE_STREAM_IDS.get(market)
        if stream is None:
            raise RecorderRestError(
                RestFailure.SHAPE, f"no recorder stream id is known for market {market!r}"
            )
        events: list[KlineEvent] = []
        for row in self.klines(market, start_ms, end_ms, symbol=symbol):
            try:
                events.append(KlineEvent.from_rest(row, stream=stream))
            except RecorderEventError as exc:
                raise RecorderRestError(
                    RestFailure.SHAPE, f"{market} kline row is not one: {exc}"
                ) from exc
        return events

    def funding_rate(
        self, start_ms: int, end_ms: int, *, symbol: str, limit: int = MAX_FUNDING_LIMIT
    ) -> list[dict[str, Any]]:
        """Realised funding settlements in ``[start_ms, end_ms]``.

        Authoritative for a *final* settlement: a row here is one the exchange
        has settled. What the day was *scheduled* to settle is a different
        question, established from the archive by PR-06's reconciliation, and
        this method neither answers it nor assumes an answer.
        """
        start, end = _require_range(start_ms, end_ms)
        payload = self._get(
            self.base_url + UM_FUNDING_RATE_PATH,
            {
                "symbol": symbol.upper(),
                "startTime": start,
                "endTime": end,
                "limit": int(limit),
            },
        )
        rows = _require_rows(payload, "fundingRate")
        settlements: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise RecorderRestError(
                    RestFailure.SHAPE,
                    f"a fundingRate row must be an object, got {type(row).__name__}",
                )
            settlements.append(dict(row))
        return settlements

    def funding_settlements(
        self, start_ms: int, end_ms: int, *, symbol: str
    ) -> list[FundingSettlement]:
        """:meth:`funding_rate`, parsed into settlements."""
        parsed: list[FundingSettlement] = []
        for row in self.funding_rate(start_ms, end_ms, symbol=symbol):
            try:
                parsed.append(FundingSettlement.from_rest(row, stream=UM_FUNDING))
            except RecorderEventError as exc:
                raise RecorderRestError(
                    RestFailure.SHAPE, f"fundingRate row is not a settlement: {exc}"
                ) from exc
        return parsed

    def premium_index(self, *, symbol: str) -> dict[str, Any]:
        """Current mark, index, funding rate and next funding time.

        The live counterpart of the mark-price stream, used as a catch-up when
        the socket has been down. It is *current state*, never a settlement:
        the ``lastFundingRate`` it carries is the rate in effect, not a realised
        payment, and nothing in this recorder treats it as one.
        """
        payload = self._get(self.base_url + UM_PREMIUM_INDEX_PATH, {"symbol": symbol.upper()})
        if not isinstance(payload, Mapping):
            raise RecorderRestError(
                RestFailure.SHAPE,
                f"premiumIndex must answer with an object for one symbol, got "
                f"{type(payload).__name__}",
            )
        return dict(payload)

    def mark_price_klines(
        self, start_ms: int, end_ms: int, *, symbol: str, limit: int = MAX_KLINE_LIMIT
    ) -> list[list[Any]]:
        """Per-minute mark-price klines.

        Part of the adopted REST interface and reachable from here, but **not**
        used by PR-05: the recorder derives its per-minute mark from the
        ``markPrice`` stream it already records, and comparing that against this
        archive is the reconciliation PR-06 owns. Present so that the interface
        is complete and pinned by a test, not so that PR-05 can quietly become
        PR-06.
        """
        return self._price_klines(
            UM_MARK_PRICE_KLINES_PATH, "symbol", symbol, start_ms, end_ms, limit
        )

    def index_price_klines(
        self, start_ms: int, end_ms: int, *, pair: str, limit: int = MAX_KLINE_LIMIT
    ) -> list[list[Any]]:
        """Per-minute index-price klines, keyed by ``pair`` rather than ``symbol``.

        The same standing as :meth:`mark_price_klines`: available, unused by
        PR-05, reconciled by PR-06.
        """
        return self._price_klines(
            UM_INDEX_PRICE_KLINES_PATH, "pair", pair, start_ms, end_ms, limit
        )

    def _price_klines(
        self,
        path: str,
        key: str,
        value: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[list[Any]]:
        start, end = _require_range(start_ms, end_ms)
        payload = self._get(
            self.base_url + path,
            {
                key: value.upper(),
                "interval": INTERVAL_1M,
                "startTime": start,
                "endTime": end,
                "limit": int(limit),
            },
        )
        return _require_rows(payload, path)

    # --- transport --------------------------------------------------------
    def _kline_page_limit(self, market: str, requested: int | None) -> int:
        """The page size to send: the market's ceiling, or less if asked for less.

        Clamped rather than refused. A caller asking for more than the endpoint
        will give is not making an error worth failing a recovery over — but it
        must not then be told the range ended early, so the clamped value is what
        both the request and the exhaustion test use.
        """
        try:
            ceiling = MAX_KLINE_PAGE[market]
        except KeyError:
            raise RecorderRestError(
                RestFailure.SHAPE,
                f"no kline page limit is known for market {market!r}; "
                f"this build knows {sorted(MAX_KLINE_PAGE)}",
            ) from None
        if requested is None:
            return ceiling
        requested = int(requested)
        if requested < 1:
            raise RecorderRestError(
                RestFailure.SHAPE,
                f"a kline page limit must be at least 1, got {requested}",
            )
        return min(requested, ceiling)

    def _kline_endpoint(self, market: str) -> tuple[str, str]:
        try:
            base, path = KLINE_ENDPOINTS[market]
        except KeyError:
            raise RecorderRestError(
                RestFailure.SHAPE,
                f"no public kline endpoint is known for market {market!r}; "
                f"this build knows {sorted(KLINE_ENDPOINTS)}",
            ) from None
        return (self.base_url if market == "um" else self.spot_base_url), path

    def _get(self, url: str, params: Mapping[str, Any]) -> Any:
        """One GET, with bounded retries, and no credential anywhere on it."""
        last: RecorderRestError | None = None
        for attempt in range(self.max_attempts):
            if attempt:
                self.retries += 1
            self._respect_min_interval()
            try:
                response = self.session.get(url, params=dict(params), timeout=self.timeout)
            except requests.Timeout as exc:
                last = RecorderRestError(RestFailure.TIMEOUT, f"GET {url} timed out: {exc}")
            except requests.RequestException as exc:
                last = RecorderRestError(RestFailure.TRANSPORT, f"GET {url} failed: {exc}")
            else:
                self.requests_made += 1
                status = int(getattr(response, "status_code", 0))
                if status in RATE_LIMITED_STATUSES:
                    self.rate_limited += 1
                    wait = _retry_after_seconds(response)
                    last = RecorderRestError(
                        RestFailure.RATE_LIMITED,
                        f"GET {url} was rate limited with {status}",
                    )
                    if attempt + 1 < self.max_attempts:
                        logger.warning("rate limited by %s; waiting %.1fs", url, wait)
                        self._sleep(wait)
                        continue
                    raise last
                if status >= 400:
                    last = RecorderRestError(
                        RestFailure.STATUS,
                        f"GET {url} answered {status}: {_body_excerpt(response)}",
                    )
                    if 400 <= status < 500 and status != 408:
                        # A client error is a defect in the request, and asking
                        # again changes nothing except how fast it is refused.
                        raise last
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise RecorderRestError(
                            RestFailure.DECODE,
                            f"GET {url} answered {status} with something that is not JSON: "
                            f"{exc}",
                        ) from exc
            if attempt + 1 < self.max_attempts:
                delay = min(self.max_backoff_s, self.backoff_s * (2**attempt))
                delay *= 1.0 + 0.1 * random.random()
                logger.warning("%s; retrying in %.1fs", last, delay)
                self._sleep(delay)
        assert last is not None  # every path above sets it before looping
        raise last

    def _respect_min_interval(self) -> None:
        if self.min_interval_s <= 0:
            return
        now = self._mono()
        if self._last_request_at is not None:
            wait = self.min_interval_s - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
                now = self._mono()
        self._last_request_at = now


# --- helpers ----------------------------------------------------------------
def _require_range(start_ms: Any, end_ms: Any) -> tuple[int, int]:
    for name, value in (("start_ms", start_ms), ("end_ms", end_ms)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise RecorderRestError(
                RestFailure.SHAPE, f"{name} must be an integer millisecond, got {value!r}"
            )
        if value < 0:
            raise RecorderRestError(RestFailure.SHAPE, f"{name} {value} is negative")
    if end_ms < start_ms:
        raise RecorderRestError(
            RestFailure.SHAPE, f"end_ms {end_ms} is before start_ms {start_ms}"
        )
    return int(start_ms), int(end_ms)


def _require_rows(payload: Any, where: str) -> list[Any]:
    if isinstance(payload, Mapping):
        raise RecorderRestError(
            RestFailure.SHAPE,
            f"{where} answered with an object rather than a list; Binance uses that shape "
            f"for errors, and it reads {dict(payload)!r}",
        )
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise RecorderRestError(
            RestFailure.SHAPE, f"{where} must answer with a list, got {type(payload).__name__}"
        )
    return list(payload)


def _row_open_ms(row: Any, where: str) -> int:
    if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or not row:
        raise RecorderRestError(RestFailure.SHAPE, f"{where} row is not a kline row: {row!r}")
    value = row[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecorderRestError(
            RestFailure.SHAPE, f"{where} row open time must be an integer, got {value!r}"
        )
    return int(value)


def _retry_after_seconds(response: Any) -> float:
    """``Retry-After`` where the endpoint sent one, bounded, else the default."""
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if raw is None:
        return DEFAULT_BACKOFF_S
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_BACKOFF_S
    if seconds < 0:
        return DEFAULT_BACKOFF_S
    return min(seconds, MAX_RETRY_AFTER_S)


def _body_excerpt(response: Any, limit: int = 200) -> str:
    text = getattr(response, "text", "")
    if not isinstance(text, str):
        return ""
    return text[:limit]


#: ``GET /fapi/v1/premiumIndex`` publishes the same five values the mark-price
#: stream does, under its own long field names. This is the rename, and nothing
#: else: no value is computed, defaulted or dropped.
PREMIUM_INDEX_FIELDS: Mapping[str, str] = {
    "time": "E",
    "markPrice": "p",
    "indexPrice": "i",
    "estimatedSettlePrice": "P",
    "lastFundingRate": "r",
    "nextFundingTime": "T",
}


def premium_index_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one ``premiumIndex`` response into the mark-price payload shape.

    The counterpart of :meth:`KlineEvent.rest_payload`: the endpoint's own field
    names are renamed onto the ones the websocket publishes, so a polled reading
    and a pushed one are parsed by the same code and stored in the same stream.

    ``lastFundingRate`` is the rate currently in effect. It is carried here as
    the mark stream's ``r``, exactly as the websocket carries it, and it is never
    turned into a settlement: a realised settlement comes from ``fundingRate``
    and from nowhere else.
    """
    if not isinstance(row, Mapping):
        raise RecorderRestError(
            RestFailure.SHAPE,
            f"premiumIndex must answer with an object, got {type(row).__name__}",
        )
    payload: dict[str, Any] = {}
    for published, stream_field in PREMIUM_INDEX_FIELDS.items():
        if published not in row:
            if published == "estimatedSettlePrice":
                continue
            raise RecorderRestError(
                RestFailure.SHAPE,
                f"premiumIndex answered without {published!r}; the mark-price record needs "
                f"it and this module invents nothing. Got {sorted(row)}",
            )
        payload[stream_field] = row[published]
    symbol = row.get("symbol")
    if symbol is not None:
        payload["s"] = symbol
    return payload


def expected_funding_instants_ms(day_start_ms: int) -> tuple[int, ...]:
    """When settlements are *expected* on a UTC day, at the cadence in force.

    A polling schedule and nothing else. It says when to ask, never what the
    answer must be: a day that settles a different number of times is recorded
    exactly as it happened, and what the venue actually scheduled is established
    from the archive by PR-06.
    """
    return tuple(day_start_ms + hour * 3_600_000 for hour in EXPECTED_FUNDING_HOURS_UTC)
