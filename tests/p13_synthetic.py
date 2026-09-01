"""Synthetic worlds for the P13 offline runtime. No network, no market data.

Every fixture here is arithmetic the test author chose, so a witness can assert an
exact number rather than a plausible one. Nothing in this file reads a Binance
archive, a committed parquet, or anything else that has ever been a market
observation — the P13 sources have never been obtained, and these tests must not
be the thing that changes that.

The default world is FLAT: open, high, low and close all equal, on both legs and
on the mark. A flat world makes the accounting identities exact — basis PnL is
zero, the excursion is exactly the entry friction — so a witness that fails is
failing about the rule under test rather than about arithmetic noise.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

from nn.p13_alignment import AlignedSources
from nn.p13_blocks import CalendarBlock
from nn.p13_carry import NOMINAL_BAR_NS
from nn.p13_sources import FundingRow, KlineRow

HOUR = NOMINAL_BAR_NS

#: A quiet, unremarkable window well inside the frozen span and far from the
#: research boundary, so no fixture can accidentally test the boundary rule by
#: being near it.
DEFAULT_START = "2021-03-01T00:00:00+00:00"

SPOT_PRICE = Decimal("30000")
PERP_PRICE = Decimal("30030")
MARK_PRICE = Decimal("30010")


def ns(iso: str) -> int:
    """An ISO instant as integer UTC nanoseconds."""
    return int(datetime.fromisoformat(iso).timestamp() * 1_000_000_000)


def instants(start_ns: int, hours: int) -> tuple[int, ...]:
    return tuple(start_ns + index * HOUR for index in range(hours))


def flat_rows(
    start_ns: int,
    hours: int,
    price: Decimal,
    *,
    skip: Iterable[int] = (),
    high: Mapping[int, Decimal] | None = None,
    close: Mapping[int, Decimal] | None = None,
) -> tuple[KlineRow, ...]:
    """A contiguous hourly series, minus whatever ``skip`` names.

    ``skip`` holds INDEX offsets, not instants, so a test can say "hour 5 is
    missing" without arithmetic. ``high`` and ``close`` override individual bars,
    which is how a liquidation witness makes one hour spike.
    """
    skipped = set(skip)
    highs = dict(high or {})
    closes = dict(close or {})
    rows = []
    for index in range(hours):
        if index in skipped:
            continue
        instant = start_ns + index * HOUR
        bar_close = closes.get(index, price)
        rows.append(
            KlineRow(
                instant_ns=instant,
                open=price,
                high=highs.get(index, max(price, bar_close)),
                low=min(price, bar_close),
                close=bar_close,
            )
        )
    return tuple(rows)


def world(
    *,
    start: str = DEFAULT_START,
    hours: int = 12,
    spot_price: Decimal = SPOT_PRICE,
    perp_price: Decimal = PERP_PRICE,
    mark_price: Decimal = MARK_PRICE,
    missing_spot: Iterable[int] = (),
    missing_perp: Iterable[int] = (),
    missing_mark: Iterable[int] = (),
    mark_high: Mapping[int, Decimal] | None = None,
    mark_close: Mapping[int, Decimal] | None = None,
    funding: Sequence[FundingRow] = (),
    published_mark_periods: Iterable[str] | None = None,
) -> AlignedSources:
    """One aligned synthetic world, with holes exactly where a test asks for them."""
    start_ns = ns(start)
    if published_mark_periods is None:
        moment = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=timezone.utc)
        published_mark_periods = {f"{moment.year:04d}-{moment.month:02d}"}
    return AlignedSources.build(
        spot=flat_rows(start_ns, hours, spot_price, skip=missing_spot),
        perpetual=flat_rows(start_ns, hours, perp_price, skip=missing_perp),
        mark=flat_rows(
            start_ns,
            hours,
            mark_price,
            skip=missing_mark,
            high=mark_high,
            close=mark_close,
        ),
        funding=funding,
        published_mark_periods=published_mark_periods,
    )


def block(
    start: str = DEFAULT_START, hours: int = 12, label: str = "synthetic"
) -> CalendarBlock:
    """A calendar block covering exactly ``hours`` grid instants from ``start``.

    Its last instant is the INTENDED CLOSE, so a block of ``hours`` bars holds
    through ``hours - 1`` of them.
    """
    start_ns = ns(start)
    return CalendarBlock(
        label=label, start_ns=start_ns, end_exclusive_ns=start_ns + hours * HOUR
    )


def funding_row(offset_hours: int, rate: str, *, start: str = DEFAULT_START) -> FundingRow:
    """One settlement at an hour offset from the world's start."""
    return FundingRow(instant_ns=ns(start) + offset_hours * HOUR, rate=Decimal(rate))


# ---------------------------------------------------------------------------
# Archive bytes
# ---------------------------------------------------------------------------


def kline_csv(
    rows: Sequence[Sequence[object]],
    *,
    header: bool = False,
    columns: Sequence[str] | None = None,
) -> bytes:
    """A kline CSV member, header optional, exactly as Binance publishes both."""
    from nn.p13_sources import KLINE_COLUMNS

    lines = []
    if header:
        lines.append(",".join(columns if columns is not None else KLINE_COLUMNS))
    for row in rows:
        lines.append(",".join(str(cell) for cell in row))
    return ("\n".join(lines) + "\n").encode("utf-8")


def kline_row_fields(instant_ms: int, price: str, high: str | None = None) -> list[object]:
    """One twelve-column kline record in Binance's published order."""
    top = high if high is not None else price
    return [
        instant_ms,
        price,
        top,
        price,
        price,
        "1",
        instant_ms + 3_599_999,
        "1",
        1,
        "1",
        "1",
        "0",
    ]


def zip_bytes(name: str, payload: bytes) -> bytes:
    """One published object: a zip holding exactly one CSV member."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    return buffer.getvalue()


def ms(iso: str) -> int:
    """An ISO instant as epoch MILLISECONDS, the futures archives' unit."""
    return ns(iso) // 1_000_000
