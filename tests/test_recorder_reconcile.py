"""The archive reconciliation, against archives this file builds byte by byte.

Every published object below is manufactured here: a real ZIP holding a real CSV
member, with a real ``.CHECKSUM`` companion whose digest is the SHA-256 of the
bytes the fake venue is about to serve. Nothing is downloaded, nothing is
recorded from a fixture somebody once fetched, and no socket is opened — the
fetcher is a dictionary. That is deliberate and it is the same discipline as
:mod:`tests.p13_synthetic`: an object that had to be downloaded could not be
broken in exactly one way, and breaking an object in exactly one way is how a
refusal gets tested.

**Two-sided throughout.** Every refusal below is paired with the object it is a
mutation of, and that object is shown to be accepted. A test that only asserts
"this is refused" cannot tell a working check from a reader that refuses
everything.

**The recorder's side is the recorder's own.** The normalized days come from
``RawSink`` and ``MinuteNormalizer`` driven by :mod:`tests.recorder_synthetic`,
so the comparison is between two independently constructed things rather than
between the reconciliation and itself. The published values are written down as
constants: they were read once from the synthetic day and pinned, which is what
makes a change in either direction a failure rather than an agreement that moved.

**And the parsing here is the recorder's own** (amendment A8). One test asserts
it about the source: no module of the recorder imports ``nn``, so the historical
P13 readers and their historical data boundary are untouched by everything below.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import hashlib
import io
import json
import math
import os
import urllib.error
import zipfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.coverage import RECONCILIATION_SCHEMA
from chimera.recorder.events import SPOT_KLINE_1M, UM_FUNDING, UM_KLINE_1M, UM_MARK_PRICE
from chimera.recorder.normalize import MinuteNormalizer
from chimera.recorder.reconcile import (
    ARCHIVE_BASE,
    ARCHIVE_FAMILIES,
    ARCHIVE_HOST,
    INDEX_DIAGNOSTIC_FAMILY,
    RELATIVE_TOLERANCE,
    ArchiveCache,
    ArchiveFetchError,
    ArchiveOutcome,
    RecorderReconcileError,
    archive_object,
    archive_url,
    parse_checksum_companion,
    read_recorder_settlements,
    reconcile_day,
    values_agree,
    write_reconciliation,
)
from chimera.recorder.sink import RawSink
from tests.recorder_synthetic import (
    DAY,
    NEXT_DAY,
    day_ms,
    funding_day,
    minute_ms,
    spot_day,
    um_day,
)

CONTRACT = load_recorder_contract()
MINUTES = 3

#: The published values of the synthetic perpetual day, read once from the day
#: :mod:`tests.recorder_synthetic` builds and pinned here. A change to either
#: side is now a failure instead of a coincidence.
UM_OPEN, UM_HIGH, UM_LOW = "60000.10", "60100.00", "59900.00"
UM_VOLUME, UM_TAKER = "12.34567890", "6.00000000"
MARK_OPEN, MARK_HIGH, MARK_LOW, MARK_CLOSE = "60050.00", "60060.00", "60050.00", "60060.00"
INDEX_VALUE = "60049.00"

#: The three settlements of the synthetic funding day, at the eight-hour cadence.
FUNDING_HOURS = (0, 8, 16)


def um_close(index: int) -> str:
    return f"6{index:04d}.50"


def spot_close(index: int) -> str:
    return f"5{index:04d}.50"


def funding_rate(hours: int) -> str:
    return f"0.0001{hours:04d}"


def month_calendar(day: str = DAY) -> list[str]:
    """Every UTC day of the month ``day`` belongs to, in order."""
    first = date.fromisoformat(day).replace(day=1)
    walked, current = [], first
    while current.month == first.month:
        walked.append(current.isoformat())
        current += timedelta(days=1)
    return walked


def funding_settlements(day: str) -> list[list[str]]:
    """One day's three published settlements, at the eight-hour cadence."""
    return [
        [str(day_ms(day) + hours * 3_600_000), "8", funding_rate(hours)]
        for hours in FUNDING_HOURS
    ]


def funding_month(day_rows: list[list[str]], *, day: str = DAY) -> list[list[str]]:
    """A whole month of settlements, with ``day``'s own rows put in their place.

    A "monthly" object holding a single day is a month published before it
    closed, and the reconciliation refuses to establish a schedule from one: an
    object whose settlements stop before the day says nothing about the day, and
    treating its silence as an empty schedule is exactly how missing evidence
    becomes a pass. Every funding case below is therefore built inside a month
    that really does cover the month, so that what is under test is the case and
    not the truncation — and the truncation gets its own two-sided test instead.
    """
    rows: list[list[str]] = []
    for other in month_calendar(day):
        rows.extend(day_rows if other == day else funding_settlements(other))
    return rows


# --- building published objects --------------------------------------------------
def kline_row(
    open_ms: int,
    *,
    unit: str,
    open_price: str = UM_OPEN,
    high: str = UM_HIGH,
    low: str = UM_LOW,
    close: str,
    volume: str = UM_VOLUME,
    taker: str = UM_TAKER,
) -> list[str]:
    """One published kline row: twelve columns, in the venue's order.

    ``unit`` decides how the two instants are written and nothing else, which is
    exactly the property the reader must not infer from magnitude.
    """
    scale = {"ms": 1, "us": 1_000}[unit]
    return [
        str(open_ms * scale),
        open_price,
        high,
        low,
        close,
        volume,
        str((open_ms + 59_999) * scale),
        "740000.00",
        "42",
        taker,
        "360000.00",
        "0",
    ]


def csv_text(rows: list[list[str]], *, header: tuple[str, ...] | None) -> str:
    lines = [",".join(header)] if header is not None else []
    lines += [",".join(row) for row in rows]
    return "\n".join(lines) + "\n"


KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
).split(",")

FUNDING_HEADER = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def zip_bytes(member: str, text: str) -> bytes:
    """A real ZIP holding exactly one member. Deterministic: no timestamp of today."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
        archive.writestr(info, text)
    return buffer.getvalue()


def multi_member_zip(names: list[str], text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), text)
    return buffer.getvalue()


def companion_bytes(body: bytes, name: str) -> bytes:
    """The venue's ``.CHECKSUM`` companion: the digest and the file it vouches for."""
    return f"{hashlib.sha256(body).hexdigest()}  {name}\n".encode("utf-8")


class FakeVenue:
    """A dictionary of published objects, and a record of what was asked for.

    Not a server. There is no socket here and no test in this file opens one:
    the reconciliation takes its fetcher as an argument precisely so that the
    parsing, the verification and the comparison can be exercised with the
    network absent rather than mocked.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.requests: list[str] = []

    def publish(self, path: str, body: bytes, *, companion: bytes | None = None) -> None:
        self.objects[path] = body
        name = path.rsplit("/", 1)[-1]
        self.objects[path + ".CHECKSUM"] = (
            companion_bytes(body, name) if companion is None else companion
        )

    def withdraw(self, path: str) -> None:
        self.objects.pop(path, None)
        self.objects.pop(path + ".CHECKSUM", None)

    def __call__(self, path: str) -> bytes | None:
        self.requests.append(path)
        return self.objects.get(path)


def object_path(stream: str, day: str = DAY) -> str:
    return archive_object(ARCHIVE_FAMILIES[stream], CONTRACT, day).path


def index_path(day: str = DAY) -> str:
    return archive_object(INDEX_DIAGNOSTIC_FAMILY, CONTRACT, day).path


def publish_day(
    venue: FakeVenue,
    *,
    day: str = DAY,
    minutes: int = MINUTES,
    um_rows: list[list[str]] | None = None,
    spot_rows: list[list[str]] | None = None,
    mark_rows: list[list[str]] | None = None,
    index_rows: list[list[str]] | None = None,
    funding_rows: list[list[str]] | None = None,
) -> FakeVenue:
    """Publish one agreeing day of every archive the gate and the diagnostic read."""
    indices = range(minutes)
    um = (
        um_rows
        if um_rows is not None
        else [kline_row(minute_ms(i, day=day), unit="ms", close=um_close(i)) for i in indices]
    )
    spot = (
        spot_rows
        if spot_rows is not None
        else [
            kline_row(minute_ms(i, day=day), unit="us", close=spot_close(i)) for i in indices
        ]
    )
    mark = (
        mark_rows
        if mark_rows is not None
        else [
            kline_row(
                minute_ms(i, day=day),
                unit="ms",
                open_price=MARK_OPEN,
                high=MARK_HIGH,
                low=MARK_LOW,
                close=MARK_CLOSE,
                volume="0",
                taker="0",
            )
            for i in indices
        ]
    )
    index = (
        index_rows
        if index_rows is not None
        else [
            kline_row(
                minute_ms(i, day=day),
                unit="ms",
                open_price=INDEX_VALUE,
                high=INDEX_VALUE,
                low=INDEX_VALUE,
                close=INDEX_VALUE,
                volume="0",
                taker="0",
            )
            for i in indices
        ]
    )
    funding = (
        funding_rows
        if funding_rows is not None
        else funding_month(funding_settlements(day), day=day)
    )
    for stream, rows, header in (
        (UM_KLINE_1M, um, KLINE_HEADER),
        (UM_MARK_PRICE, mark, KLINE_HEADER),
        (SPOT_KLINE_1M, spot, None),
    ):
        path = object_path(stream, day)
        member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
        venue.publish(path, zip_bytes(member, csv_text(rows, header=header)))
    path = index_path(day)
    venue.publish(
        path,
        zip_bytes(
            path.rsplit("/", 1)[-1].replace(".zip", ".csv"),
            csv_text(index, header=KLINE_HEADER),
        ),
    )
    path = object_path(UM_FUNDING, day)
    venue.publish(
        path,
        zip_bytes(
            path.rsplit("/", 1)[-1].replace(".zip", ".csv"),
            csv_text(funding, header=FUNDING_HEADER),
        ),
    )
    return venue


# --- the recorder's side ----------------------------------------------------------
@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A storage root holding three recorded minutes of every stream, normalized."""
    storage = CONTRACT.storage_root(tmp_path / "data")
    material = {
        **um_day(range(MINUTES)),
        **spot_day(range(MINUTES)),
        UM_FUNDING: funding_day(DAY),
    }
    for stream, events in material.items():
        with RawSink(storage, stream, contract=CONTRACT) as sink:
            for event in events:
                sink.append(event)
            sink.sync()
    normalizer = MinuteNormalizer(storage, CONTRACT)
    for market in CONTRACT.market_keys():
        normalizer.build_day(market, DAY)
    normalizer.build_settlements("um")
    return storage


@pytest.fixture
def venue() -> FakeVenue:
    return publish_day(FakeVenue())


def reconciliation_of(root: Path, venue: FakeVenue, **kwargs):
    """The report object, for the few assertions that are about the run and not the record.

    ``ArchiveRead.cached`` is the example: whether a body came from the local
    cache is a fact about this host rather than about the evidence, so it is
    deliberately absent from the persisted document and has to be read here.
    """
    return reconcile_day(root, DAY, venue, contract=CONTRACT, **kwargs)


def report_of(root: Path, venue: FakeVenue, **kwargs) -> dict:
    return reconciliation_of(root, venue, **kwargs).to_dict()


def published_funding(venue: FakeVenue, rows: list[list[str]], *, day: str = DAY) -> None:
    """Republish the monthly funding object holding exactly ``rows``."""
    path = object_path(UM_FUNDING, day)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=FUNDING_HEADER)))


def published_kline(
    venue: FakeVenue, rows: list[list[str]], *, stream: str = UM_KLINE_1M
) -> None:
    """Republish one daily kline object holding exactly ``rows``."""
    path = object_path(stream)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=KLINE_HEADER)))


# --- A. the source integrity rules --------------------------------------------------
def test_the_archive_paths_are_the_ones_section_4_7_names():
    """Pinned, because the denominator of the whole gate is which object was read."""
    assert object_path(UM_KLINE_1M) == (
        "data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-09-19.zip"
    )
    assert object_path(UM_MARK_PRICE) == (
        "data/futures/um/daily/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2026-09-19.zip"
    )
    assert object_path(SPOT_KLINE_1M) == (
        "data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-09-19.zip"
    )
    assert object_path(UM_FUNDING) == (
        "data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-09.zip"
    )
    assert index_path() == (
        "data/futures/um/daily/indexPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2026-09-19.zip"
    )


def test_only_the_allow_listed_host_can_be_addressed():
    """The two-sided form: the archive path is fetched, anything else is refused."""
    assert archive_url("data/spot/daily/klines/BTCUSDT/1m/x.zip") == (
        f"{ARCHIVE_BASE}/data/spot/daily/klines/BTCUSDT/1m/x.zip"
    )
    assert ARCHIVE_HOST in ARCHIVE_BASE and ARCHIVE_BASE.startswith("https://")
    for hostile in (
        "https://evil.example.com/x.zip",
        "//evil.example.com/x.zip",
        "/data/spot/x.zip",
        "data/../../etc/passwd",
        "data/spot/x.zip?key=1",
        "data\\spot\\x.zip",
        # urlsplit strips these three before it parses, so a path carrying one
        # would be host-checked as a URL that is not the URL the request is then
        # made with. Refused in the path rather than relied on downstream.
        "data/spot/x\ny.zip",
        "data/spot/x\ty.zip",
        "data/spot/x\ry.zip",
        "",
    ):
        with pytest.raises(RecorderReconcileError):
            archive_url(hostile)


def test_a_redirect_is_refused_rather_than_followed(monkeypatch):
    """The allow-list is a property of every request, not only of the first one.

    ``urllib``'s default opener follows a ``3xx`` to whatever host the
    ``Location`` header names — including a plain-HTTP one — and nothing
    re-checks it, so the one-host rule would apply to the request that left and
    to none of the ones that returned the bytes. The digest does not save it
    either: the ``.CHECKSUM`` companion travels the same transport, so a
    redirecting origin would supply both the object and the digest that vouches
    for it. No socket is opened here; the opener is replaced.
    """
    from chimera.recorder import reconcile as module

    assert module._RefuseRedirect().redirect_request(None, None, 302, "Found", {}, "x") is None

    path = "data/spot/daily/klines/BTCUSDT/1m/x.zip"

    class Redirecting:
        def open(self, request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 302, "Found", {"Location": "http://elsewhere/x.zip"}, None
            )

    monkeypatch.setattr(module, "_ARCHIVE_OPENER", Redirecting())
    with pytest.raises(ArchiveFetchError, match="redirect is refused"):
        module.HttpsArchiveFetcher()(path)


def test_bytes_served_from_another_origin_are_refused(monkeypatch):
    """Belt and braces: the URL the response says it came from is checked too."""
    from chimera.recorder import reconcile as module

    class Response:
        def __init__(self, url: str) -> None:
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def geturl(self) -> str:
            return self.url

        def read(self, size: int) -> bytes:
            return b"body"

    path = "data/spot/daily/klines/BTCUSDT/1m/x.zip"

    class Serving:
        def __init__(self, url: str) -> None:
            self.url = url

        def open(self, request, timeout=None):
            return Response(self.url)

    monkeypatch.setattr(module, "_ARCHIVE_OPENER", Serving(archive_url(path)))
    assert module.HttpsArchiveFetcher()(path) == b"body", "the control: the right origin"

    for elsewhere in (
        "https://evil.example.com/x.zip",
        f"http://{ARCHIVE_HOST}/{path}",
    ):
        monkeypatch.setattr(module, "_ARCHIVE_OPENER", Serving(elsewhere))
        with pytest.raises(ArchiveFetchError, match="was served from"):
            module.HttpsArchiveFetcher()(path)


def test_the_checksum_companion_must_name_the_object_it_vouches_for():
    body = b"published bytes"
    digest = hashlib.sha256(body).hexdigest()
    assert (
        parse_checksum_companion(
            companion_bytes(body, "wanted.zip"), expected_name="wanted.zip"
        )
        == digest
    )
    for broken in (
        f"{digest}  other.zip\n".encode("utf-8"),
        f"{digest}\n".encode("utf-8"),
        f"{digest}  wanted.zip\n{digest}  wanted.zip\n".encode("utf-8"),
        b"not-a-digest  wanted.zip\n",
        b"",
    ):
        with pytest.raises(RecorderReconcileError):
            parse_checksum_companion(broken, expected_name="wanted.zip")


def test_a_checksum_mismatch_is_refused_and_is_not_an_empty_published_set(root, venue):
    """The refusal, and the object it is one byte away from being."""
    path = object_path(UM_KLINE_1M)
    good = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert good["judged"] and good["published_minutes"] == MINUTES

    venue.objects[path + ".CHECKSUM"] = f"{'0' * 64}  {path.rsplit('/', 1)[-1]}\n".encode(
        "utf-8"
    )
    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.CHECKSUM_MISMATCH.value
    assert entry["judged"] is False
    assert entry["published_minutes"] == 0
    assert "refusal" in entry["archive"]["detail"]


def test_an_absent_object_is_absence_and_never_an_empty_published_set(root, venue):
    venue.withdraw(object_path(SPOT_KLINE_1M))
    entry = report_of(root, venue)["streams"][SPOT_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.ABSENT.value
    assert entry["judged"] is False, "a 404 says nothing about how many minutes exist"
    assert entry["published_minutes"] == 0
    assert entry["recorder_minutes"] == MINUTES, "the recorder still holds what it holds"


def test_a_transport_failure_is_not_an_absent_object(root, venue):
    """A network that could not answer has said nothing about the venue.

    Reported as its own outcome, because recording it as a 404 would turn a
    local fault — a refused proxy, a name that would not resolve, a dropped
    connection — into a claim that the exchange published nothing that day.
    """

    def refuses(path: str) -> bytes | None:
        raise ArchiveFetchError(f"{path} could not be reached: connection refused")

    document = reconcile_day(root, DAY, refuses, contract=CONTRACT).to_dict()
    for stream, entry in document["streams"].items():
        assert entry["archive"]["outcome"] == ArchiveOutcome.FETCH_FAILED.value, stream
        assert entry["judged"] is False, stream
        assert entry["published_minutes"] == 0, stream
    assert document["funding"]["archive"]["outcome"] == ArchiveOutcome.FETCH_FAILED.value
    assert document["funding"]["schedule_established"] is False
    assert document["funding"]["outcome"] == "FUNDING_SCHEDULE_UNAVAILABLE"

    control = report_of(root, venue)
    assert all(
        entry["judged"] for entry in control["streams"].values()
    ), "the control: the same day is fully judged when the objects can be fetched"


def test_an_object_published_without_its_companion_is_its_own_finding(root, venue):
    path = object_path(SPOT_KLINE_1M)
    del venue.objects[path + ".CHECKSUM"]
    entry = report_of(root, venue)["streams"][SPOT_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.CHECKSUM_ABSENT.value
    assert entry["judged"] is False


def test_a_multi_member_archive_is_refused_rather_than_resolved(root, venue):
    path = object_path(UM_KLINE_1M)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    text = csv_text(
        [kline_row(minute_ms(i), unit="ms", close=um_close(i)) for i in range(MINUTES)],
        header=KLINE_HEADER,
    )
    venue.publish(path, multi_member_zip([member, "extra.csv"], text))
    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.UNEXPECTED_MEMBER.value
    assert entry["judged"] is False


def test_a_single_member_under_the_wrong_name_is_refused(root, venue):
    path = object_path(UM_KLINE_1M)
    text = csv_text(
        [kline_row(minute_ms(i), unit="ms", close=um_close(i)) for i in range(MINUTES)],
        header=KLINE_HEADER,
    )
    venue.publish(path, zip_bytes("something-else.csv", text))
    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.UNEXPECTED_MEMBER.value


def test_bytes_that_are_not_an_archive_are_refused(root, venue):
    venue.publish(object_path(UM_KLINE_1M), b"PK\x03\x04 truncated and then nothing")
    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.CORRUPT_ARCHIVE.value
    assert entry["judged"] is False


@pytest.mark.parametrize(
    "member,why",
    [
        (b"\xff\xfe\x00 not text at all", "not UTF-8"),
        (("x" * 200_000 + ",1,2\n").encode("utf-8"), "a field csv will not parse"),
    ],
    ids=["not-utf-8", "oversized-field"],
)
def test_member_bytes_that_cannot_be_read_are_unparseable_and_not_an_exception(
    root, venue, member, why
):
    """Both of these used to escape ``acquire`` and abort the whole day's record.

    ``bytes.decode`` raises ``UnicodeDecodeError`` and ``csv.reader`` raises
    ``csv.Error`` on a field over its limit, and neither is one of the typed
    refusals ``acquire`` maps onto an outcome — so a verified object holding
    hostile bytes destroyed the record for every other stream as well, collapsing
    two of the eight distinct outcomes into a traceback. They are ``UNPARSEABLE``,
    which is a finding about one object and nothing else.
    """
    path = object_path(UM_KLINE_1M)
    name = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), member)
    venue.publish(path, buffer.getvalue())

    document = report_of(root, venue)
    entry = document["streams"][UM_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.UNPARSEABLE.value, why
    assert entry["judged"] is False
    assert entry["published_minutes"] == 0
    assert (
        document["streams"][SPOT_KLINE_1M]["judged"] is True
    ), "the other streams still got their verdicts; one unreadable object is one finding"


def test_an_unknown_layout_is_refused_in_both_directions(root, venue):
    """A header where the layout says there is none, and none where it says there is.

    The epoch unit hangs off the layout, so accepting the other shape would
    reinterpret every instant in the file rather than merely read a stray row.
    """
    rows = [kline_row(minute_ms(i), unit="us", close=spot_close(i)) for i in range(MINUTES)]
    path = object_path(SPOT_KLINE_1M)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=KLINE_HEADER)))
    entry = report_of(root, venue)["streams"][SPOT_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.UNKNOWN_LAYOUT.value

    venue = publish_day(FakeVenue())
    rows = [kline_row(minute_ms(i), unit="ms", close=um_close(i)) for i in range(MINUTES)]
    path = object_path(UM_KLINE_1M)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=None)))
    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.UNKNOWN_LAYOUT.value


def test_the_epoch_unit_is_the_layout_s_and_is_not_inferred_from_magnitude(root, venue):
    """Spot is microseconds and USD-M is milliseconds, per file, by declaration.

    The control is the pair: the same minutes agree when each object is written
    in its own layout's unit, and the spot object written in milliseconds — whose
    numbers are perfectly plausible instants — is refused instead of being read
    as a day 56 years earlier.
    """
    agreeing = report_of(root, venue)["streams"]
    assert agreeing[SPOT_KLINE_1M]["agreeing_minutes"] == MINUTES
    assert agreeing[UM_KLINE_1M]["agreeing_minutes"] == MINUTES

    rows = [kline_row(minute_ms(i), unit="ms", close=spot_close(i)) for i in range(MINUTES)]
    path = object_path(SPOT_KLINE_1M)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=None)))
    entry = report_of(root, venue)["streams"][SPOT_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.UNPARSEABLE.value
    assert entry["judged"] is False


def test_the_verified_read_records_the_layout_and_the_unit_it_used(root, venue):
    streams = report_of(root, venue)["streams"]
    assert streams[UM_KLINE_1M]["archive"]["epoch_unit"] == "ms"
    assert streams[SPOT_KLINE_1M]["archive"]["epoch_unit"] == "us"
    assert streams[SPOT_KLINE_1M]["archive"]["layout"] == "binance-spot-kline-headerless-us"


def test_a_row_from_another_day_is_refused_rather_than_filtered(root, venue):
    rows = [kline_row(minute_ms(i), unit="ms", close=um_close(i)) for i in range(MINUTES)]
    rows.append(kline_row(minute_ms(0) - 60_000, unit="ms", close="1.0"))
    path = object_path(UM_KLINE_1M)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=KLINE_HEADER)))
    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["archive"]["outcome"] == ArchiveOutcome.UNPARSEABLE.value


# --- B. the cache -------------------------------------------------------------------
def test_the_cache_is_keyed_by_the_full_archive_path_and_not_by_the_basename(
    root, venue, tmp_path
):
    """Three daily objects share the basename ``BTCUSDT-1m-<day>.zip``.

    A basename-keyed cache would serve the spot object for the perpetual one and
    report a clean agreement between two different markets, so the pin is that
    each stream still gets its own published values back through the cache.
    """
    basenames = {
        object_path(stream).rsplit("/", 1)[-1]
        for stream in (UM_KLINE_1M, UM_MARK_PRICE, SPOT_KLINE_1M)
    }
    assert len(basenames) == 1, "the collision this cache has to survive"

    cache = ArchiveCache(tmp_path / "cache")
    first = report_of(root, venue, cache=cache)
    warm = FakeVenue()
    for path, body in venue.objects.items():
        if path.endswith(".CHECKSUM"):
            warm.objects[path] = body
    run = reconcile_day(root, DAY, warm, contract=CONTRACT, cache=cache)

    assert run.to_dict()["streams"] == first["streams"], (
        "the cache served each object to itself rather than to a sibling with the same "
        "basename, and the record it produced is byte-for-byte the record the cold run "
        "produced: whether a body came from this host's disk is not evidence and does not "
        "appear in the document"
    )
    assert all(
        entry.read.cached for entry in run.streams
    ), "the second pass read the cache rather than the venue"
    assert all(path.endswith(".CHECKSUM") for path in warm.requests), (
        "only the companions were fetched on the second pass, and every body came from "
        "the cache"
    )


def test_a_cached_body_is_re_verified_and_a_changed_one_is_not_read(root, venue, tmp_path):
    cache = ArchiveCache(tmp_path / "cache")
    report_of(root, venue, cache=cache)
    path = object_path(UM_KLINE_1M)
    entry = cache.entry_path(path)
    assert entry.exists(), "the first pass filled the cache"
    entry.write_bytes(entry.read_bytes() + b"tampered")

    assert (
        cache.read(
            path,
            published_digest=parse_checksum_companion(
                venue.objects[path + ".CHECKSUM"], expected_name=path.rsplit("/", 1)[-1]
            ),
        )
        is None
    ), "a cache that cannot vouch for its value is not read"

    again = reconcile_day(root, DAY, venue, contract=CONTRACT, cache=cache)
    fresh = {entry.stream: entry for entry in again.streams}[UM_KLINE_1M]
    assert fresh.read.cached is False, "the tampered entry was refetched"
    assert fresh.agreeing == MINUTES


def test_a_cache_entry_without_its_sidecar_is_not_read(root, venue, tmp_path):
    cache = ArchiveCache(tmp_path / "cache")
    report_of(root, venue, cache=cache)
    path = object_path(UM_KLINE_1M)
    digest = hashlib.sha256(cache.entry_path(path).read_bytes()).hexdigest()
    assert (
        cache.read(path, published_digest=digest) is not None
    ), "the control: it was readable"
    cache.sidecar_path(path).unlink()
    assert cache.read(path, published_digest=digest) is None


def test_a_cache_sidecar_naming_another_object_is_not_read(root, venue, tmp_path):
    cache = ArchiveCache(tmp_path / "cache")
    report_of(root, venue, cache=cache)
    path = object_path(UM_KLINE_1M)
    sidecar = cache.sidecar_path(path)
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    digest = document["sha256"]
    assert cache.read(path, published_digest=digest) is not None
    document["archive_path"] = object_path(SPOT_KLINE_1M)
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    assert cache.read(path, published_digest=digest) is None


def test_a_cached_body_the_venue_no_longer_vouches_for_is_not_read(root, venue, tmp_path):
    """The digest is the venue's current one, not the one cached beside the body."""
    cache = ArchiveCache(tmp_path / "cache")
    report_of(root, venue, cache=cache)
    path = object_path(UM_KLINE_1M)
    assert cache.read(path, published_digest="0" * 64) is None


# --- C. comparing values --------------------------------------------------------------
def test_the_tolerance_is_relative_and_pinned_on_both_sides_of_1e_9():
    """``1e-9 * 1e9`` is exactly ``1.0``, so the boundary itself can be asserted."""
    reference = Decimal("1000000000")
    allowed = RELATIVE_TOLERANCE * 1e9
    assert allowed == 1.0, "the boundary is exact at this scale, which is why it is used"
    assert values_agree(reference, 1e9), "identical values agree"
    assert values_agree(reference, 1e9 + 1.0), "exactly at the tolerance agrees"
    assert not values_agree(
        reference, math.nextafter(1e9 + 1.0, math.inf)
    ), "one float above the tolerance disagrees"
    assert values_agree(reference, math.nextafter(1e9 + 1.0, -math.inf))


def test_zero_is_compared_exactly_rather_than_divided_by():
    """A published zero volume is a normal minute, not an error and not a wildcard."""
    assert values_agree(Decimal("0"), 0.0)
    assert values_agree(Decimal("0.00000000"), 0.0)
    assert not values_agree(Decimal("0"), 1e-12)
    assert not values_agree(Decimal("0"), -1e-30)
    assert not values_agree(Decimal("1"), float("nan"))
    assert not values_agree(Decimal("1"), float("inf"))


def test_a_published_value_that_overflows_float64_agrees_with_nothing():
    """The fail-open this closes sat in the numerator of published_coverage.

    ``Decimal("1E400")`` is a finite decimal, so the parser accepts it, and
    ``float()`` of it is ``inf``. The relative comparison then reads
    ``|x - inf| <= tol * inf``, which is ``inf <= inf``, which is true for every
    recorded value — so a corrupt published number whose ``.CHECKSUM`` matched
    would have counted as agreement on every minute it appeared in. The control
    is the same magnitude one exponent lower, which float64 holds and which
    behaves normally.
    """
    assert values_agree(Decimal("1E307"), 1e307), "the control: this one still fits"
    assert not values_agree(Decimal("1E307"), 1e306)
    for recorded in (123.45, -9e99, 0.0, float("inf"), float("-inf")):
        assert not values_agree(Decimal("1E400"), recorded)
        assert not values_agree(Decimal("-1E400"), recorded)


@pytest.mark.parametrize(
    "column,replacement",
    [
        ("open", "60000.11"),
        ("high", "60100.01"),
        ("low", "59900.01"),
        ("close", "69999.99"),
        # Both perturbations are outside the relative tolerance and both are
        # small: the pair pins that the fields are read at all, not that a wild
        # value is noticed. 12.34568000 differs from the recorded volume by
        # 8.9e-8 relative, and 6.00000001 from the recorded taker-buy base by
        # 1.7e-9 relative, each comfortably above 1e-9 and far below anything a
        # careless comparison would catch by accident.
        ("volume", "12.34568000"),
        ("taker_buy_volume", "6.00000001"),
    ],
)
def test_every_compared_kline_field_is_actually_compared(root, venue, column, replacement):
    """Six fields, six mutations, six disagreements — including taker-buy base.

    The control is the unmutated day in :func:`test_a_clean_day_agrees_on_every_stream`:
    without it, a comparison that disagreed with everything would pass here.
    """
    index = KLINE_HEADER.index(column)
    rows = [kline_row(minute_ms(i), unit="ms", close=um_close(i)) for i in range(MINUTES)]
    rows[1][index] = replacement
    path = object_path(UM_KLINE_1M)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=KLINE_HEADER)))

    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["agreeing_minutes"] == MINUTES - 1
    assert entry["disagreeing_minutes"] == 1
    disagreement = entry["disagreements"][0]
    assert disagreement["minute_open_ms"] == minute_ms(1)
    assert [field["field"] for field in disagreement["fields"]] == [column]
    assert disagreement["fields"][0]["archive"] == replacement


def test_a_clean_day_agrees_on_every_stream(root, venue):
    document = report_of(root, venue)
    assert set(document["streams"]) == set(CONTRACT.minute_indexed_required())
    for stream, entry in document["streams"].items():
        assert entry["judged"] is True, stream
        assert entry["published_minutes"] == MINUTES, stream
        assert entry["agreeing_minutes"] == MINUTES, stream
        assert entry["disagreeing_minutes"] == 0, stream
        assert entry["archive_only_minutes"] == [], stream
        assert entry["recorder_only_minutes"] == [], stream
    assert document["funding"]["funding_complete"] is True


def test_minutes_only_the_archive_has_and_minutes_only_the_recorder_has_are_kept_apart(
    root, venue
):
    rows = [kline_row(minute_ms(i), unit="ms", close=um_close(i)) for i in range(MINUTES + 2)]
    del rows[0]
    path = object_path(UM_KLINE_1M)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=KLINE_HEADER)))
    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["published_minutes"] == MINUTES + 1
    assert entry["recorder_minutes"] == MINUTES
    assert entry["compared_minutes"] == MINUTES - 1
    assert entry["archive_only_minutes"] == [minute_ms(3), minute_ms(4)]
    assert entry["recorder_only_minutes"] == [minute_ms(0)]


def test_the_mark_stream_is_compared_against_the_mark_price_archive(root, venue):
    rows = [
        kline_row(
            minute_ms(i),
            unit="ms",
            open_price=MARK_OPEN,
            high=MARK_HIGH,
            low=MARK_LOW,
            close="60060.01",
            volume="0",
            taker="0",
        )
        for i in range(MINUTES)
    ]
    path = object_path(UM_MARK_PRICE)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=KLINE_HEADER)))
    entry = report_of(root, venue)["streams"][UM_MARK_PRICE]
    assert entry["agreeing_minutes"] == 0
    assert entry["disagreeing_minutes"] == MINUTES
    assert [field["field"] for field in entry["disagreements"][0]["fields"]] == ["close"]


# --- D. the index diagnostic (amendment A6) --------------------------------------------
def test_an_index_disagreement_is_a_diagnostic_and_does_not_fail_the_mark(root, venue):
    """One required stream, one verdict, one source family.

    The index archive is made to disagree on every minute while the mark archive
    is left agreeing. ``um.markPrice`` must still be fully agreeing, and the
    index result must live outside the section the gate reads.
    """
    rows = [
        kline_row(
            minute_ms(i),
            unit="ms",
            open_price="1.00",
            high="1.00",
            low="1.00",
            close="1.00",
            volume="0",
            taker="0",
        )
        for i in range(MINUTES)
    ]
    path = index_path()
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=KLINE_HEADER)))

    document = report_of(root, venue)
    assert document["streams"][UM_MARK_PRICE]["agreeing_minutes"] == MINUTES
    assert document["streams"][UM_MARK_PRICE]["disagreeing_minutes"] == 0
    diagnostic = document["diagnostics"]["um.indexPrice"]
    assert diagnostic["disagreeing_minutes"] == MINUTES
    assert diagnostic["gates_nothing"] is True
    assert "um.indexPrice" not in document["streams"], (
        "the index is not a stream the gate walks; a diagnostic in the gated section "
        "would be a second pass condition however it was labelled"
    )


def test_an_absent_index_archive_does_not_disturb_anything_the_gate_reads(root, venue):
    venue.withdraw(index_path())
    document = report_of(root, venue)
    assert document["streams"][UM_MARK_PRICE]["judged"] is True
    assert document["streams"][UM_MARK_PRICE]["agreeing_minutes"] == MINUTES
    assert document["diagnostics"]["um.indexPrice"]["judged"] is False


def test_an_index_side_fault_cannot_withhold_the_mark_s_verdict(root, venue):
    """Amendment A6 the whole way: the diagnostic cannot deny a gated stream a record.

    The diagnostic runs through the same function the gated streams use, and that
    function raises rather than records when the normalized table has no column
    it needs. A raise would have escaped ``reconcile_day`` and left the day with
    no record at all — no verdict for ``um.markPrice``, a day the gate reads as
    missing, and a streak broken by an index-side fault. Here the index columns
    are dropped from the perpetual's normalized day and the mark still gets its
    verdict; the diagnostic says it could not be computed, which is what a
    diagnostic that gates nothing is for.
    """
    normalizer = MinuteNormalizer(root, CONTRACT)
    parquet = normalizer.parquet_path("um", DAY)
    frame = pd.read_parquet(parquet)
    frame.drop(
        columns=[name for name in frame.columns if name.startswith("index_")]
    ).to_parquet(parquet, index=False)

    document = report_of(root, venue)
    assert document["streams"][UM_MARK_PRICE]["judged"] is True
    assert document["streams"][UM_MARK_PRICE]["agreeing_minutes"] == MINUTES
    diagnostic = document["diagnostics"]["um.indexPrice"]
    assert diagnostic["judged"] is False
    assert diagnostic["gates_nothing"] is True
    assert "no column" in diagnostic["reason"]


def test_the_index_diagnostic_can_be_skipped_without_changing_a_gated_number(root, venue):
    with_index = report_of(root, venue)
    without = report_of(root, venue, index_diagnostic=False)
    assert without["diagnostics"] == {}
    assert without["streams"] == with_index["streams"]
    assert without["funding"] == with_index["funding"]


# --- E. funding (amendments A2, A4, A9) --------------------------------------------------
def test_funding_agrees_on_settlement_time_and_realised_rate(root, venue):
    funding = report_of(root, venue)["funding"]
    assert funding["schedule_established"] is True
    assert funding["outcome"] == "OK"
    assert funding["scheduled"] == len(FUNDING_HOURS)
    assert funding["captured"] == len(FUNDING_HOURS)
    assert funding["funding_complete"] is True
    assert funding["missing_settlements"] == []
    assert funding["disagreeing_settlements"] == []
    assert [entry["funding_interval_hours"] for entry in funding["scheduled_settlements"]] == [
        8,
        8,
        8,
    ]


def test_funding_never_compares_the_settlement_mark_price(root, venue):
    """The archive publishes none, so agreement cannot depend on one (amendment A4).

    The recorder's settlements file carries ``mark_price`` and the reconciliation
    is shown not to read it: the file's mark prices are rewritten to a value
    nothing else in the fixture holds, and the day stays funding-complete.
    """
    normalizer = MinuteNormalizer(root, CONTRACT)
    path = normalizer.settlements_path("um")
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        record["mark_price"] = "1.00"
        lines.append(json.dumps(record, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert read_recorder_settlements(root, "um") == {
        minute_ms(0) + hours * 3_600_000: Decimal(funding_rate(hours))
        for hours in FUNDING_HOURS
    }, "the reader hands out the instant and the rate and nothing else"
    funding = report_of(root, venue)["funding"]
    assert funding["funding_complete"] is True
    assert "mark" not in json.dumps(funding["scheduled_settlements"])


def test_a_disagreeing_realised_rate_fails_the_day_outright(root, venue):
    rows = [
        [
            str(minute_ms(0) + hours * 3_600_000),
            "8",
            "0.99999999" if hours == 8 else funding_rate(hours),
        ]
        for hours in FUNDING_HOURS
    ]
    published_funding(venue, funding_month(rows))
    funding = report_of(root, venue)["funding"]
    assert funding["schedule_established"] is True
    assert funding["funding_complete"] is False
    assert len(funding["disagreeing_settlements"]) == 1
    assert funding["disagreeing_settlements"][0]["archive"] == "0.99999999"


def test_a_missing_settlement_fails_the_day_outright(root, venue):
    rows = [
        [str(minute_ms(0) + hours * 3_600_000), "8", funding_rate(hours)]
        for hours in (0, 8, 16, 20)
    ]
    published_funding(venue, funding_month(rows))
    funding = report_of(root, venue)["funding"]
    assert funding["schedule_established"] is True
    assert funding["scheduled"] == 4
    assert funding["captured"] == 3
    assert funding["missing_settlements"] == [minute_ms(0) + 20 * 3_600_000]
    assert funding["funding_complete"] is False


def test_an_unpublished_month_is_schedule_unavailable_and_not_an_empty_schedule(root, venue):
    """Amendment A2's whole point, and amendment A9's expected evidence latency."""
    venue.withdraw(object_path(UM_FUNDING))
    funding = report_of(root, venue)["funding"]
    assert funding["schedule_established"] is False
    assert funding["outcome"] == "FUNDING_SCHEDULE_UNAVAILABLE"
    assert funding["funding_complete"] is False
    assert funding["scheduled"] == 0 and funding["scheduled_settlements"] == []
    assert funding["archive"]["outcome"] == ArchiveOutcome.ABSENT.value


def test_an_established_empty_schedule_is_funding_complete(root, venue):
    """The universal over the empty set holds, and no quotient is evaluated.

    The month is published, verified, and brackets this day — it lists
    settlements before it and after it — and simply lists none inside it. That is
    the venue saying there was nothing to miss. It is a different fact from the
    month being unreadable, and a different fact again from the month stopping
    before the day: the first is the test above and the second is the test below,
    and all three are kept apart because collapsing any two of them is how a day
    passes that should not.
    """
    published_funding(venue, funding_month([]))
    funding = report_of(root, venue)["funding"]
    assert funding["schedule_established"] is True
    assert funding["covers_day"] is True
    assert funding["outcome"] == "OK"
    assert funding["scheduled"] == 0
    assert funding["captured"] == 0
    assert funding["funding_complete"] is True, "the universal holds over the empty set"


def test_a_month_that_stops_before_the_day_establishes_nothing(root, venue):
    """The fourth clause of schedule_established: "covers only part of D".

    The two-sided partner of the test above, and the one that matters most: the
    object verifies, parses, and lists no settlement inside this day — exactly
    like the established-empty case — but its coverage stops ten days earlier. An
    implementation that read the empty intersection as an empty schedule would
    make every day after a partially published month funding-complete, so a
    month published before it closed would hand a pass to every day it never
    covered. That is precisely the reading amendment A2 exists to remove,
    arriving through partial publication instead of through an unreadable object.
    """
    stops_at = month_calendar()[: month_calendar().index(DAY) - 9]
    assert stops_at and stops_at[-1] < DAY, "the month really does stop before the day"
    published_funding(venue, [row for other in stops_at for row in funding_settlements(other)])
    funding = report_of(root, venue)["funding"]
    assert funding["archive"]["outcome"] == ArchiveOutcome.VERIFIED.value
    assert funding["schedule_established"] is False
    assert funding["covers_day"] is False
    assert funding["outcome"] == "FUNDING_SCHEDULE_UNAVAILABLE"
    assert funding["funding_complete"] is False
    assert funding["scheduled"] == 0 and funding["scheduled_settlements"] == []
    assert "covers only part of" in funding["reason"]


def test_the_last_day_of_a_month_is_established_by_a_month_that_reaches_into_it(root, venue):
    """The carve-out, pinned on both sides, because there is no later day to reach.

    A complete month's settlements stop inside its own last day, so requiring the
    object to reach *past* that day would make the last day of every month
    permanently unjudgeable and break every streak that spans a month boundary.
    The bar there is that the object reaches into the day; a month that stops
    before it still establishes nothing.
    """
    last = month_calendar()[-1]
    whole = [row for other in month_calendar() for row in funding_settlements(other)]
    published_funding(venue, whole, day=last)
    report = reconcile_day(root, last, venue, contract=CONTRACT).to_dict()
    assert report["funding"]["schedule_established"] is True
    assert report["funding"]["covers_day"] is True

    without_the_last_day = [
        row for other in month_calendar()[:-1] for row in funding_settlements(other)
    ]
    published_funding(venue, without_the_last_day, day=last)
    report = reconcile_day(root, last, venue, contract=CONTRACT).to_dict()
    assert report["funding"]["schedule_established"] is False
    assert report["funding"]["outcome"] == "FUNDING_SCHEDULE_UNAVAILABLE"


def test_a_corrupt_funding_archive_is_unavailable_rather_than_empty(root, venue):
    venue.publish(object_path(UM_FUNDING), b"not a zip at all")
    funding = report_of(root, venue)["funding"]
    assert funding["archive"]["outcome"] == ArchiveOutcome.CORRUPT_ARCHIVE.value
    assert funding["schedule_established"] is False
    assert funding["outcome"] == "FUNDING_SCHEDULE_UNAVAILABLE"
    assert funding["funding_complete"] is False


def test_an_unreadable_settlements_file_is_a_finding_and_not_an_absence(root, venue):
    normalizer = MinuteNormalizer(root, CONTRACT)
    normalizer.settlements_path("um").write_text("{ not json\n", encoding="utf-8")
    with pytest.raises(RecorderReconcileError, match="settlement record"):
        report_of(root, venue)


def test_funding_has_no_wallclock_coverage(root, venue):
    funding = report_of(root, venue)["funding"]
    assert funding["wallclock_coverage"] is None
    assert "1440" not in json.dumps(funding)


# --- F. what is not gated ----------------------------------------------------------------
def test_book_ticker_is_absent_from_the_archive_gate(root, venue):
    """Amendment A5: it is recorded, and no archive publishes a denominator for it."""
    document = report_of(root, venue)
    assert "um.bookTicker" in CONTRACT.streams
    assert "um.bookTicker" not in CONTRACT.required_for_coverage
    assert "um.bookTicker" not in document["streams"]
    assert "um.bookTicker" not in ARCHIVE_FAMILIES
    assert "bookTicker" not in json.dumps(document)
    assert not any(
        "bookTicker" in path for path in venue.requests
    ), "no substitute denominator was fetched for it"


def test_a_required_stream_with_no_named_archive_is_refused(root, venue):
    """A skipped required stream would leave the gate dividing by nothing."""
    invented = dataclasses.replace(
        CONTRACT, required_for_coverage=CONTRACT.required_for_coverage + ("um.bookTicker",)
    )
    with pytest.raises(RecorderReconcileError, match="minute denominator"):
        reconcile_day(root, DAY, venue, contract=invented)


# --- G. the record itself ------------------------------------------------------------------
def test_the_record_is_a_function_of_the_bytes_and_carries_no_clock(root, venue, tmp_path):
    """The same recorder files plus the same archive bytes, twice, byte for byte.

    The second pass runs with a warmed cache and the third without one, because
    that is the difference two hosts holding identical evidence would actually
    have: one that had fetched the objects before and one that had not. A record
    that differed between them could not be diffed, and diffing is the only way a
    reviewer checks a verdict without re-running the fetch.
    """
    first = write_reconciliation(root, reconcile_day(root, DAY, venue, contract=CONTRACT))
    body = first.read_bytes()
    first.unlink()

    cache = ArchiveCache(tmp_path / "cache")
    reconcile_day(root, DAY, venue, contract=CONTRACT, cache=cache)
    warmed = write_reconciliation(
        root, reconcile_day(root, DAY, venue, contract=CONTRACT, cache=cache)
    )
    assert warmed.read_bytes() == body, "a local cache is not evidence and must not show"
    warmed.unlink()

    second = write_reconciliation(root, reconcile_day(root, DAY, venue, contract=CONTRACT))
    assert second.read_bytes() == body, "the report moved without the evidence moving"

    document = json.loads(body)
    assert document["reconciliation_schema"] == RECONCILIATION_SCHEMA
    assert document["day"] == DAY
    assert document["contract_hash"] == CONTRACT.contract_hash
    assert document["prospective_from"] is None
    assert document["evidence_class"] == "engineering"
    text = body.decode("utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert str(tmp_path) not in text, "no machine-absolute path reaches the record"

    def keys_of(node) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {name for value in node.values() for name in keys_of(value)}
        if isinstance(node, list):
            return {name for value in node for name in keys_of(value)}
        return set()

    named = {key.lower() for key in keys_of(document)}
    for economic in ("return", "pnl", "profit", "basis", "carry", "alpha", "flow"):
        assert not any(economic in key for key in named), (
            f"the record has a field naming {economic}; the recorder reports what was "
            "published and computes no economic quantity"
        )


def test_a_day_before_the_boundary_is_engineering_data_in_an_activated_root(root, venue):
    """The class is a fact about the day, not about the contract having a boundary.

    An activated root may hold engineering observations recorded before the
    boundary, and only observations at or after it count toward the streak. A
    record that stamped ``prospective`` on a pre-boundary day would relabel
    engineering data as scientific evidence inside a committed evidence file —
    the one document a later reviewer reads to check the claim. Pinned on both
    sides of the boundary, one day apart.
    """
    activated = CONTRACT.with_prospective_from(datetime.fromisoformat(f"{DAY}T00:00:00+00:00"))
    on_the_day = reconcile_day(root, DAY, venue, contract=activated).to_dict()
    assert on_the_day["prospective_from"] == activated.prospective_from.isoformat()
    assert on_the_day["evidence_class"] == "prospective"

    later = CONTRACT.with_prospective_from(
        datetime.fromisoformat(f"{NEXT_DAY}T00:00:00+00:00")
    )
    before = reconcile_day(root, DAY, venue, contract=later).to_dict()
    assert (
        before["evidence_class"] == "engineering"
    ), "the day is a day before the boundary, whatever the contract now says"
    assert CONTRACT.prospective_from is None, "and the committed contract is untouched"


def test_a_rerun_may_establish_more_and_may_never_establish_less(root, venue):
    """Re-running is amendment A9's normal pattern, and it runs in one direction.

    A day whose monthly funding archive had not been published is
    ``FUNDING_SCHEDULE_UNAVAILABLE`` and takes its real verdict when the archive
    appears — that direction is expected and is exercised first. The other
    direction is not evidence about anything: during a transport outage every
    object is ``FETCH_FAILED``, and a cron re-running yesterday's days would
    otherwise overwrite good records with empty ones and break a streak for a
    reason that has nothing to do with the recorder.
    """
    unpublished = FakeVenue()
    for path, body in venue.objects.items():
        if not path.startswith(object_path(UM_FUNDING).rsplit("/", 1)[0]):
            unpublished.objects[path] = body
    path = write_reconciliation(root, reconcile_day(root, DAY, unpublished, contract=CONTRACT))
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["funding"]["schedule_established"] is False

    write_reconciliation(root, reconcile_day(root, DAY, venue, contract=CONTRACT))
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["funding"]["schedule_established"] is True, "unavailable to established"

    def refuse_everything(_: str) -> bytes | None:
        raise ArchiveFetchError("the transport is down")

    with pytest.raises(RecorderReconcileError, match="would destroy evidence"):
        write_reconciliation(
            root, reconcile_day(root, DAY, refuse_everything, contract=CONTRACT)
        )
    assert (
        json.loads(path.read_text(encoding="utf-8")) == stored
    ), "the stored record is unchanged, byte for byte"


def test_the_record_lands_where_the_gate_and_gitignore_expect_it(root, venue):
    path = write_reconciliation(root, reconcile_day(root, DAY, venue, contract=CONTRACT))
    assert path.parent.name == "reconciliation"
    assert path.name == f"{DAY}.json"
    assert path.parent.parent == root


def test_the_reconciliation_never_modifies_the_recorder_s_files(root, venue):
    """A disagreement is a finding, not a repair."""
    before = {path: path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}
    rows = [kline_row(minute_ms(i), unit="ms", close="1.00") for i in range(MINUTES)]
    path = object_path(UM_KLINE_1M)
    member = path.rsplit("/", 1)[-1].replace(".zip", ".csv")
    venue.publish(path, zip_bytes(member, csv_text(rows, header=KLINE_HEADER)))
    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["disagreeing_minutes"] == MINUTES, "the control: it really disagreed"

    after = {path: path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}
    assert after == before, "the recorder's own bytes were touched"


def test_the_reconciliation_opens_no_recorder_file_for_writing(root, venue, monkeypatch):
    """The structural half of "a disagreement is a finding, not a repair".

    Comparing the bytes before and after cannot see a write that happens to
    produce the same bytes, and it cannot see one at all if the repair is
    idempotent — so the control that matters is not "the bytes are unchanged" but
    "nothing under the storage root was opened for writing, renamed or removed".
    The reconciliation reads the recorder's normalized days and settlements file
    and persists its record through a separate call, so during
    :func:`reconcile_day` the recorder's tree is strictly read-only, and this
    asserts exactly that at the point where a repair would have to happen.
    """
    guarded_root = root.resolve()
    writes: list[str] = []

    def note(target) -> None:
        try:
            resolved = Path(target).resolve()
        except (TypeError, ValueError):
            return
        if guarded_root == resolved or guarded_root in resolved.parents:
            writes.append(str(resolved))

    real_open = io.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            note(file)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_open)
    for name in ("replace", "rename", "remove", "unlink", "truncate"):
        real = getattr(os, name)

        def guarded(first, *args, _real=real, **kwargs):
            note(first)
            return _real(first, *args, **kwargs)

        monkeypatch.setattr(os, name, guarded)

    # Both shapes a repair could take are put in front of it at once: minutes
    # the archive publishes and the recorder does not hold, which is what a
    # back-fill would fill, and minutes both hold with different values, which is
    # what a correction would overwrite.
    rows = [kline_row(minute_ms(i), unit="ms", close="1.00") for i in range(MINUTES + 2)]
    published_kline(venue, rows)
    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["disagreeing_minutes"] == MINUTES, "the control: it really disagreed"
    assert len(entry["archive_only_minutes"]) == 2, "and the archive really had more"
    assert writes == [], f"the reconciliation opened {writes} for writing under the root"

    # And the guard is not blind: one deliberate write under the root is seen.
    (root / "written-by-the-test").write_bytes(b"x")
    assert writes, "the write guard saw nothing at all and would pass vacuously"


def test_a_missing_normalized_day_is_a_capture_of_nothing_and_not_a_refusal(root, venue):
    MinuteNormalizer(root, CONTRACT).parquet_path("um", DAY).unlink()
    entry = report_of(root, venue)["streams"][UM_KLINE_1M]
    assert entry["judged"] is True, "the archive still supplies the denominator"
    assert entry["published_minutes"] == MINUTES
    assert entry["recorder_minutes"] == 0
    assert entry["agreeing_minutes"] == 0


def test_a_normalized_day_missing_a_compared_column_is_refused(root, venue):
    normalizer = MinuteNormalizer(root, CONTRACT)
    parquet = normalizer.parquet_path("um", DAY)
    frame = pd.read_parquet(parquet).drop(columns=["kline_taker_buy_base"])
    frame.to_parquet(parquet, index=False)
    with pytest.raises(RecorderReconcileError, match="kline_taker_buy_base"):
        report_of(root, venue)


# --- H. the boundary this package must not cross ---------------------------------------------
def test_neither_new_module_imports_anything_from_the_research_package():
    """Amendment A8, asserted about the source rather than promised in prose."""
    import chimera.recorder as package

    directory = Path(package.__file__).resolve().parent
    for name in ("reconcile.py", "coverage.py"):
        source = (directory / name).read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "nn" not in imported, f"{name} imports nn; the P13 readers stay P13's"


def test_the_settlement_reader_holds_the_recorder_s_published_rate_exactly(root):
    """Decimal in, Decimal out: a realised rate never travels through a float."""
    settlements = read_recorder_settlements(root, "um")
    assert settlements[minute_ms(0)] == Decimal("0.00010000")
    assert isinstance(settlements[minute_ms(0)], Decimal)
    assert read_recorder_settlements(root / "nowhere", "um") == {}
