"""Archive reconciliation: what the venue published, against what the recorder holds.

The normalizer can say what this recorder captured. It cannot say what the
exchange published, and no amount of looking at the recorder's own files will
ever tell it — a recorder outage and a venue publication gap are
indistinguishable from inside. This module is where the second fact comes from:
the venue's own daily and monthly archives, fetched, verified against the digest
the venue publishes beside them, parsed under the recorder's own semantics, and
compared minute by minute with what was recorded. The coverage gate's
``published_minutes`` denominator exists because of this module and nowhere else.

**It acquires, so it is not part of the offline core.** It reaches exactly one
allow-listed first-party host over HTTPS, with no credential, no request
signature and no private path, and ``tests/test_recorder_no_network.py`` holds it
to every rule the live collection layer is held to. It is not part of the live
layer either: it opens no websocket, runs no event loop and collects nothing. It
is its own group in that barrier, named there, and adding a host to it is a
reviewed edit to that file rather than something that happens by writing a URL.

**The parsing is the recorder's own** (amendment A8). Nothing here imports from
``nn``. The historical P13 readers enforce P13's historical data boundary and
refuse an object outside it, so they could not read a prospective object even if
the recorder were allowed to import them, and it is not. What is reproduced here
is not their code but the source-integrity *rules* this repository already
applies to historical acquisition:

* one allow-listed first-party host and no other, refused rather than redirected;
* HTTPS, with no credential and no request signature;
* the ``.CHECKSUM`` companion fetched, parsed, and the filename it names checked
  against the object actually requested — a digest that vouches for a different
  file vouches for nothing;
* SHA-256 recomputed over the bytes received and required to match exactly;
* a mismatch **refused** — never repaired, never retried into acceptance;
* a cache keyed by the object's full archive path, never by its basename, whose
  value is re-verified against the published digest before it is used and which
  is not read at all when it cannot vouch for what it holds;
* exactly one member in the archive, and the member the object's name implies,
  verified before extraction;
* the epoch unit resolved explicitly per file from the layout the object is
  published in, never inferred from the magnitude of what arrived;
* only recognised layouts accepted, an unknown layout being a refusal.

**Nothing becomes an empty published set.** A 404, an unpublished month, an
unverified object, a checksum mismatch, a corrupt or truncated archive, a
multi-member archive, an unparseable member and an unknown layout are eight
different findings, each recorded under its own :class:`ArchiveOutcome`, and not
one of them is the same as "the venue published nothing that day". A day whose
denominator could not be established is unjudgeable, and the gate treats an
unjudgeable day as a day that did not pass.

**Nothing is repaired.** The recorder's files are never modified here — not
opened for writing, not corrected, not re-derived, not back-filled from the
archive. A disagreement is a finding and it is written down; the archive is the
witness the recorder is measured against, not a source to patch the recorder
from.

**Funding agrees on time and rate, and on nothing else** (amendment A4). The
authorised monthly ``fundingRate`` archive publishes ``calc_time``,
``funding_interval_hours`` and ``last_funding_rate``, and no settlement mark
price. So archive equality is the settlement instant and the realised rate; the
interval metadata is recorded where it is published; and the mark price the
recorder captured from its live public source keeps its provenance and is
verified by nothing here. It is never reconstructed from mark-price candles,
never matched to a nearest minute, never filled, never interpolated and never
replaced by a REST value. There is no ``/1440`` funding denominator and no
settlement ratio anywhere in this module.

**And a schedule is established only by an object that shows it covers the day**
(amendment A2). The monthly object carries no declaration of its own extent, so
the extent is witnessed from the settlements it publishes: a month that stops
before the day does not establish that day, and its silence is missing evidence
rather than a venue that scheduled nothing. Without that check a month published
before it closed would hand every later day an empty schedule, which the
universal over the empty set reads as completeness — missing evidence becoming a
pass, which is the exact reading amendment A2 exists to remove.

**The mark is gated; the index is a diagnostic** (amendment A6). ``um.markPrice``
is one required stream with one verdict from one source family, and that family
is ``markPriceKlines``. The index values recorded alongside the mark may be
reconciled against ``indexPriceKlines``, and they are — but the result is written
into a separate ``diagnostics`` section that the coverage gate does not read, no
threshold is attached to it, and an index disagreement cannot fail an otherwise
agreeing ``um.markPrice`` day.

**The report is a function of the bytes.** The same recorder files plus the same
archive bytes produce the same document: no wall-clock instant appears in its
semantic content, no machine-absolute path, no NaN and no Infinity. Two hosts
that fetched the same objects can diff their reports and find nothing, which is
the only way a reviewer can check a verdict without re-running the fetch. The
allow-listed host is named — it is the provenance of what was read, and a record
that did not say where its denominator came from would be worth less — and a
transport failure records the message the transport gave, because a run that
established nothing has to say why it established nothing.

**And nothing economic is computed.** Comparing two published prices for equality
is not a return, a basis, a carry or a profit, and no such quantity is derived,
stored or reported by anything below.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from chimera.recorder.contract import RecorderContract
from chimera.recorder.coverage import (
    FUNDING_SCHEDULE_UNAVAILABLE,
    RECONCILIATION_DIRECTORY,
    RECONCILIATION_SCHEMA,
    RecorderCoverageError,
    read_reconciliation,
    reconciliation_path,
)
from chimera.recorder.events import (
    MS_PER_MINUTE,
    NS_PER_DAY,
    NS_PER_MILLISECOND,
    UM_FUNDING,
    day_start_ns,
    iso_utc,
)
from chimera.recorder.normalize import (
    FUNDING_DIRECTORY,
    SETTLEMENTS_FILE,
    MinuteNormalizer,
)
from chimera.recorder.sink import (
    RecorderSinkError,
    require_day,
    write_bytes_atomic,
    write_json_atomic,
)

#: The one host this module may reach. Named once, checked on every request, and
#: allow-listed in ``tests/test_recorder_no_network.py`` as a deliberate edit.
ARCHIVE_HOST = "data.binance.vision"
ARCHIVE_BASE = "https://data.binance.vision"

#: What the venue names the digest companion of an object.
CHECKSUM_SUFFIX = ".CHECKSUM"

#: The parity tolerance ``nn.multiclock`` already uses, and the one section 4.7
#: names. Relative, and see :func:`values_agree` for what it means at zero.
RELATIVE_TOLERANCE = 1e-9

#: How many bytes a published object may have before this module refuses to hold
#: it in memory. A day of 1m klines is a few hundred kilobytes; a month of
#: funding rates is smaller still. The bound exists so that a wrong path cannot
#: turn into an unbounded download.
MAX_OBJECT_BYTES = 64 << 20

#: How long one request may take. A reconciliation that hangs is a
#: reconciliation that never writes a record, which is worse than one that fails.
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Names the shape of one cache entry's sidecar.
CACHE_ENTRY_SCHEMA = "chimera.recorder-archive-cache/1"


class RecorderReconcileError(RuntimeError):
    """A day cannot be reconciled into an honest record."""


class ArchiveFetchError(RecorderReconcileError):
    """The transport failed, so whether the object exists is unknown.

    Deliberately not the same as "absent": a network that could not answer has
    said nothing about the venue, and recording it as a 404 would turn a local
    failure into a claim that the exchange published nothing.
    """


class ArchiveChecksumError(RecorderReconcileError):
    """The published digest cannot be read, or does not vouch for this object."""


class ArchiveCorruptError(RecorderReconcileError):
    """The bytes received are not a readable archive."""


class ArchiveMemberError(RecorderReconcileError):
    """The archive does not hold exactly the one member its name implies."""


class ArchiveLayoutError(RecorderReconcileError):
    """The member is not published in the layout this build recognises."""


class ArchiveContentError(RecorderReconcileError):
    """The member is in a recognised layout and still cannot be read as rows.

    Kept apart from :class:`ArchiveLayoutError` because "this is a shape nobody
    told me about" and "this is the right shape with a broken value in it" are
    different findings, and only the first is an argument for adding a layout.
    """


class ArchiveOutcome(str, Enum):
    """What became of one archive object. A bounded, recorded label.

    Every value other than :attr:`VERIFIED` means the object's contents are
    **unknown**, which is not the same as empty. The gate reads ``judged`` and
    refuses to form a coverage quotient without a denominator it can name.
    """

    #: Fetched, digest matched, single expected member, recognised layout, parsed.
    VERIFIED = "VERIFIED"
    #: The venue answered 404: this object is not published (yet, or at all).
    ABSENT = "ABSENT"
    #: The object exists and its ``.CHECKSUM`` companion does not.
    CHECKSUM_ABSENT = "CHECKSUM_ABSENT"
    #: The companion is unreadable, or names a file other than the one requested.
    CHECKSUM_MALFORMED = "CHECKSUM_MALFORMED"
    #: The digest published and the digest of the bytes received disagree.
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    #: The bytes are not a readable archive: corrupt, truncated or not a zip.
    CORRUPT_ARCHIVE = "CORRUPT_ARCHIVE"
    #: The archive holds more than one member, or not the member it should.
    UNEXPECTED_MEMBER = "UNEXPECTED_MEMBER"
    #: The member is not in a layout this build recognises.
    UNKNOWN_LAYOUT = "UNKNOWN_LAYOUT"
    #: The member is in a recognised layout and still cannot be read.
    UNPARSEABLE = "UNPARSEABLE"
    #: The request itself failed; the venue said nothing either way.
    FETCH_FAILED = "FETCH_FAILED"


@dataclass(frozen=True)
class ArchiveLayout:
    """One recognised published layout: its columns, its header, its epoch unit.

    The epoch unit is a property of the layout and therefore of the object, not
    of the numbers inside it. Resolving it by magnitude — "these look like
    microseconds" — is how a reader silently reinterprets a day when a venue
    changes its unit, and section 4.7 forbids it.
    """

    layout_id: str
    #: Whether the published member's first row names its columns.
    has_header: bool
    #: The columns, in the order a headerless member carries them. With a header
    #: the names are looked up instead, so an added trailing column is not a
    #: reinterpretation of the ones before it.
    columns: tuple[str, ...]
    #: ``ms`` or ``us``. Declared, never inferred.
    epoch_unit: str


#: The twelve columns Binance publishes for a 1m kline object, in order.
KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)

#: The three columns the monthly funding object publishes (amendment A4). There
#: is deliberately no settlement mark price here, because the source has none.
FUNDING_COLUMNS: tuple[str, ...] = (
    "calc_time",
    "funding_interval_hours",
    "last_funding_rate",
)

#: The recognised layouts, exhaustively. A member that does not match the layout
#: its family declares is refused, and adding a layout is a reviewed edit here
#: rather than a heuristic that widens itself when an object surprises it.
UM_KLINE_LAYOUT = ArchiveLayout("binance-um-kline-header-ms", True, KLINE_COLUMNS, "ms")
SPOT_KLINE_LAYOUT = ArchiveLayout(
    "binance-spot-kline-headerless-us", False, KLINE_COLUMNS, "us"
)
UM_FUNDING_LAYOUT = ArchiveLayout("binance-um-funding-header-ms", True, FUNDING_COLUMNS, "ms")

#: The published fields a kline stream is reconciled on: section 4.7's
#: ``o/h/l/c/v/V``. ``V`` is the taker-buy **base** volume, which the archive
#: calls ``taker_buy_volume`` and the recorder stores as ``kline_taker_buy_base``.
KLINE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "taker_buy_volume")

#: The published fields a mark or index candle is reconciled on. Those archives
#: publish per-minute o/h/l/c and no volume.
CANDLE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")

#: Archive field to normalized column, per kind. Written out rather than derived
#: from a prefix, because ``taker_buy_volume`` and ``kline_taker_buy_base`` are
#: the same published number under two names and a rule that guessed would get
#: exactly that one wrong.
KLINE_COLUMN_OF: Mapping[str, str] = {
    "open": "kline_open",
    "high": "kline_high",
    "low": "kline_low",
    "close": "kline_close",
    "volume": "kline_volume",
    "taker_buy_volume": "kline_taker_buy_base",
}
MARK_COLUMN_OF: Mapping[str, str] = {
    "open": "mark_open",
    "high": "mark_high",
    "low": "mark_low",
    "close": "mark_close",
}
INDEX_COLUMN_OF: Mapping[str, str] = {
    "open": "index_open",
    "high": "index_high",
    "low": "index_low",
    "close": "index_close",
}


@dataclass(frozen=True)
class ArchiveFamily:
    """One first-party archive: where it lives, what it publishes, how it is read.

    The mapping from a required stream to its family is section 4.7's table and
    lives in :data:`ARCHIVE_FAMILIES`. Which streams are *required* is never
    written down here — it is read from the contract — so a stream added to or
    removed from ``required_for_coverage`` changes what is reconciled without
    this module being edited, and a required stream with no family named here is
    a refusal rather than a silently skipped one.
    """

    family_id: str
    #: The archive directory, with ``{symbol}`` filled from the contract.
    prefix: str
    #: The object's basename, with ``{symbol}`` and ``{period}``.
    basename: str
    #: ``daily`` or ``monthly``: which calendar period one object covers.
    period_kind: str
    layout: ArchiveLayout
    #: The published fields compared, and the normalized columns they are
    #: compared against.
    fields: tuple[str, ...]
    column_of: Mapping[str, str]
    #: The market whose normalized day holds the recorder's side.
    market: str
    #: The column that says the recorder captured this stream in that minute, or
    #: ``None`` when the row's existence already says it.
    presence_column: str | None


ARCHIVE_FAMILIES: Mapping[str, ArchiveFamily] = {
    "um.kline_1m": ArchiveFamily(
        family_id="futures-um-daily-klines-1m",
        prefix="data/futures/um/daily/klines/{symbol}/1m",
        basename="{symbol}-1m-{period}.zip",
        period_kind="daily",
        layout=UM_KLINE_LAYOUT,
        fields=KLINE_FIELDS,
        column_of=KLINE_COLUMN_OF,
        market="um",
        presence_column=None,
    ),
    "um.markPrice": ArchiveFamily(
        family_id="futures-um-daily-markPriceKlines-1m",
        prefix="data/futures/um/daily/markPriceKlines/{symbol}/1m",
        basename="{symbol}-1m-{period}.zip",
        period_kind="daily",
        layout=UM_KLINE_LAYOUT,
        fields=CANDLE_FIELDS,
        column_of=MARK_COLUMN_OF,
        market="um",
        presence_column="mark_present",
    ),
    "spot.kline_1m": ArchiveFamily(
        family_id="spot-daily-klines-1m",
        prefix="data/spot/daily/klines/{symbol}/1m",
        basename="{symbol}-1m-{period}.zip",
        period_kind="daily",
        layout=SPOT_KLINE_LAYOUT,
        fields=KLINE_FIELDS,
        column_of=KLINE_COLUMN_OF,
        market="spot",
        presence_column=None,
    ),
    UM_FUNDING: ArchiveFamily(
        family_id="futures-um-monthly-fundingRate",
        prefix="data/futures/um/monthly/fundingRate/{symbol}",
        basename="{symbol}-fundingRate-{period}.zip",
        period_kind="monthly",
        layout=UM_FUNDING_LAYOUT,
        fields=(),
        column_of={},
        market="um",
        presence_column=None,
    ),
}

#: The index-price archive. Deliberately not in :data:`ARCHIVE_FAMILIES` and
#: deliberately not keyed by a stream id: it gates nothing, and keeping it out of
#: the mapping the gate walks is what makes "diagnostic" structural rather than a
#: promise in a docstring (amendment A6).
INDEX_DIAGNOSTIC_FAMILY = ArchiveFamily(
    family_id="futures-um-daily-indexPriceKlines-1m",
    prefix="data/futures/um/daily/indexPriceKlines/{symbol}/1m",
    basename="{symbol}-1m-{period}.zip",
    period_kind="daily",
    layout=UM_KLINE_LAYOUT,
    fields=CANDLE_FIELDS,
    column_of=INDEX_COLUMN_OF,
    market="um",
    presence_column="mark_present",
)


@dataclass(frozen=True)
class ArchiveObject:
    """One published object: the family it belongs to and the period it covers."""

    family: ArchiveFamily
    symbol: str
    period: str

    @property
    def basename(self) -> str:
        return self.family.basename.format(symbol=self.symbol, period=self.period)

    @property
    def path(self) -> str:
        prefix = self.family.prefix.format(symbol=self.symbol)
        return f"{prefix}/{self.basename}"

    @property
    def checksum_path(self) -> str:
        return self.path + CHECKSUM_SUFFIX

    @property
    def member_name(self) -> str:
        """The one member the object is expected to hold: the zip's name as CSV."""
        return self.basename[: -len(".zip")] + ".csv"


def month_of(day: str) -> str:
    """The ``YYYY-MM`` a UTC day belongs to."""
    return require_day(day)[:7]


def schedule_coverage_witness_ms(day: str) -> int:
    """The instant a monthly funding object must reach before it establishes ``day``.

    Section 4.9 makes ``schedule_established(D)`` false when the source object
    "covers only part of D", and a monthly object carries no field that declares
    its own extent — so the extent has to be witnessed from the settlements it
    actually publishes. The witness is the latest settlement instant in the
    object, and the bar it must clear is this:

    * for a day that is not the month's last, the **end of that day**. A complete
      month reaches past every day inside it; an object truncated before ``day``
      does not, and its silence about ``day`` is then missing evidence rather
      than a venue that scheduled nothing.
    * for the month's last day, the **start of that day**, because there is no
      later day inside the month for the object to reach into and requiring one
      would make the last day of every month permanently unjudgeable.

    This is deliberately a statement about the evidence and never about the
    venue: an object that does not demonstrate it covers ``day`` is
    ``FUNDING_SCHEDULE_UNAVAILABLE``, which does not pass, is not a recorder
    outage, and takes its real verdict when the complete month is published. The
    cadence is not assumed anywhere — no count of settlements is expected and no
    interval is imputed; only the object's own reach is read.
    """
    first_ms = day_start_ns(day) // NS_PER_MILLISECOND
    end_of_day_ms = first_ms + 24 * 60 * MS_PER_MINUTE
    month = month_of(day)
    year, number = int(month[:4]), int(month[5:7])
    first_of_next_month = f"{year + number // 12}-{number % 12 + 1:02d}-01"
    last_day_start_ms = (day_start_ns(first_of_next_month) - NS_PER_DAY) // NS_PER_MILLISECOND
    return min(end_of_day_ms, last_day_start_ms)


def archive_object(
    family: ArchiveFamily, contract: RecorderContract, day: str
) -> ArchiveObject:
    """The object of ``family`` covering ``day``, for the contract's symbol."""
    symbol = contract.market(family.market).symbol
    period = month_of(day) if family.period_kind == "monthly" else require_day(day)
    return ArchiveObject(family=family, symbol=symbol, period=period)


def require_archive_path(path: str) -> str:
    """A relative archive path that cannot become a request to somewhere else.

    Everything that reaches the transport goes through here. A path with a
    scheme, a leading slash, a backslash, a query, a fragment or a ``..``
    segment could be joined onto the allow-listed base and still address another
    host or another part of the site, so each is refused by name rather than
    trusted to be harmless.

    A control character is refused for a subtler reason. ``urllib.parse.urlsplit``
    silently strips tab, carriage return and newline before it parses, so a path
    carrying one would be host-checked as a URL that is not the URL the request
    is then made with — a validate-one-string, send-another mismatch in the one
    function whose whole job is to guarantee the host. No archive path has ever
    contained one; that is the argument for refusing them, not for allowing them.
    """
    if not isinstance(path, str) or not path.strip():
        raise RecorderReconcileError(f"archive path must be a non-empty string, got {path!r}")
    if path != path.strip():
        raise RecorderReconcileError(f"archive path {path!r} is padded with whitespace")
    for token in ("://", "\\", "?", "#", " "):
        if token in path:
            raise RecorderReconcileError(f"archive path {path!r} carries {token!r}")
    control = [
        character for character in path if ord(character) < 0x20 or ord(character) == 0x7F
    ]
    if control:
        raise RecorderReconcileError(
            f"archive path {path!r} carries the control character(s) "
            f"{[hex(ord(character)) for character in control]}. urlsplit strips some of them "
            "before parsing, so the host that was checked would not be the host that was "
            "addressed"
        )
    if path.startswith("/"):
        raise RecorderReconcileError(
            f"archive path {path!r} is not relative to the archive root"
        )
    segments = path.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise RecorderReconcileError(f"archive path {path!r} has an empty or relative segment")
    return path


def archive_url(path: str) -> str:
    """The one URL an archive path may become, checked after it is built.

    Built and then re-parsed rather than merely concatenated: the check that
    matters is the host the resulting request would actually go to, and that is a
    property of the finished URL rather than of the string that went into it.
    """
    url = f"{ARCHIVE_BASE}/{require_archive_path(path)}"
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise RecorderReconcileError(
            f"{url} is not HTTPS; the archive is fetched over TLS only"
        )
    if parts.netloc != ARCHIVE_HOST:
        raise RecorderReconcileError(
            f"{url} addresses {parts.netloc!r}, and this module reaches {ARCHIVE_HOST!r} and "
            "no other host. Adding one is a reviewed edit to this constant and to the "
            "endpoint allow-list the recorder's barrier test holds"
        )
    return url


def parse_checksum_companion(body: bytes, *, expected_name: str) -> str:
    """The digest a ``.CHECKSUM`` companion publishes, with the filename checked.

    The companion is ``<sha256>  <filename>``. The filename is not decoration: a
    companion that names another object is a digest for another object, and
    accepting it would let a stale or misfiled companion vouch for bytes nobody
    published a digest for. Refused rather than ignored.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArchiveChecksumError(f"the checksum companion is not text: {exc}") from exc
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ArchiveChecksumError(
            f"the checksum companion holds {len(lines)} lines; a published companion names "
            "exactly one object"
        )
    fields = lines[0].split()
    if len(fields) != 2:
        raise ArchiveChecksumError(f"the checksum companion line {lines[0]!r} is not a pair")
    digest, name = fields[0].strip().lower(), fields[1].strip().lstrip("*")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArchiveChecksumError(f"{digest!r} is not a SHA-256 digest")
    if name != expected_name:
        raise ArchiveChecksumError(
            f"the checksum companion vouches for {name!r} and the object requested is "
            f"{expected_name!r}. A digest that names another file is a digest for another "
            "file"
        )
    return digest


def extract_expected_member(raw: bytes, *, expected_name: str) -> bytes:
    """The one member the object should hold, verified before it is extracted.

    Two refusals rather than one resolution. More than one member is refused
    because choosing among several would be an unrecorded decision about which
    rows the result came from; a single member under an unexpected name is
    refused because an object whose content is not what its name says is an
    object nobody can cite.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise ArchiveMemberError(
                    f"the archive holds {len(names)} members {sorted(names)}; a published "
                    "object holds exactly one, and choosing among several would be an "
                    "unrecorded decision about which rows the result came from"
                )
            if names[0] != expected_name:
                raise ArchiveMemberError(
                    f"the archive holds {names[0]!r} and the object's name implies "
                    f"{expected_name!r}"
                )
            return archive.read(names[0])
    except zipfile.BadZipFile as exc:
        raise ArchiveCorruptError(f"not a readable archive: {exc}") from exc


# --- the transport -------------------------------------------------------------
#: What a fetcher is: given an archive path, the bytes, or ``None`` when the
#: venue answered that the object is not published. A transport failure raises
#: :class:`ArchiveFetchError`, which is a third outcome and not either of those.
Fetcher = Callable[[str], "bytes | None"]


class ArchiveCache:
    """Objects already fetched, keyed by their full archive path.

    **Keyed by the path, never by the basename.** Every daily object of every
    family is called ``BTCUSDT-1m-<day>.zip``; a cache that keyed on that would
    serve the spot day for the perpetual one, and the reconciliation would report
    a clean agreement between two different markets. The entry's location mirrors
    the archive's own directory structure and the sidecar records the full path
    again, so a collision cannot happen and a mislaid entry cannot be read as if
    it were another.

    **Re-verified before it is used, and unread when it cannot vouch.** A read
    recomputes the SHA-256 of the stored bytes, compares it with the sidecar and
    with the digest the venue published for this fetch, and treats any
    disagreement — a missing sidecar, an unreadable one, a wrong path, a changed
    byte — as a miss. A miss costs a download. Reading a value the cache cannot
    vouch for costs a verdict about bytes nobody checked, which is not a trade
    this module makes.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def entry_path(self, archive_path: str) -> Path:
        return self.directory / require_archive_path(archive_path)

    def sidecar_path(self, archive_path: str) -> Path:
        entry = self.entry_path(archive_path)
        return entry.with_name(entry.name + ".cache.json")

    def read(self, archive_path: str, *, published_digest: str | None = None) -> bytes | None:
        """The cached bytes when everything about them checks out, else ``None``."""
        entry, sidecar = self.entry_path(archive_path), self.sidecar_path(archive_path)
        try:
            body = entry.read_bytes()
            document = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(document, dict):
            return None
        if document.get("cache_entry_schema") != CACHE_ENTRY_SCHEMA:
            return None
        if document.get("archive_path") != archive_path:
            return None
        if document.get("bytes") != len(body):
            return None
        digest = hashlib.sha256(body).hexdigest()
        if document.get("sha256") != digest:
            return None
        if published_digest is not None and published_digest.lower() != digest:
            return None
        return body

    def write(self, archive_path: str, body: bytes) -> None:
        """Store bytes with the seal that lets a later read check them."""
        entry = self.entry_path(archive_path)
        entry.parent.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(entry, body)
        write_json_atomic(
            self.sidecar_path(archive_path),
            {
                "cache_entry_schema": CACHE_ENTRY_SCHEMA,
                "archive_path": archive_path,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "note": (
                    "Engineering cache of a published object. Re-verified on every read "
                    "against the digest the venue publishes; safe to delete, and deleting "
                    "it costs a download and nothing else."
                ),
            },
        )


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect handler that refuses every redirect instead of following it.

    :func:`archive_url` checks the host of the URL that *leaves* this process.
    The default opener installs ``HTTPRedirectHandler``, which would then follow
    a ``3xx`` to whatever host the ``Location`` header names — including a plain
    ``http`` one — and nothing would re-check it, so the allow-list would apply
    to the first request and to none of the ones that actually returned the
    bytes. The digest does not close that hole either: the ``.CHECKSUM``
    companion travels the same transport, so whoever controls the redirect
    supplies both the object and the digest that vouches for it.

    Returning ``None`` refuses the redirect the way ``urllib`` expects a handler
    to refuse it: the original ``3xx`` surfaces as an :class:`urllib.error.HTTPError`
    and :meth:`HttpsArchiveFetcher.__call__` turns it into an
    :class:`ArchiveFetchError`, which is the honest outcome — the venue did not
    hand this process the object, and a fetch that did not happen is not an
    empty published set.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


#: The one opener this module uses. Built once, with redirects refused rather
#: than followed, because the allow-listed host is a property of every request
#: that is made and not only of the first one.
_ARCHIVE_OPENER = urllib.request.build_opener(_RefuseRedirect)


class HttpsArchiveFetcher:
    """Fetches published objects over HTTPS from the one allow-listed host.

    Public, unauthenticated and unsigned. There is no header, no query parameter
    and no environment variable here that could carry a credential, because the
    archive is a public site and no version of reading it needs a key.

    **Refused rather than redirected.** The request goes through
    :data:`_ARCHIVE_OPENER`, which will not follow a ``3xx``, and the URL the
    response says it was served from is checked against the allow-list again
    before a byte of it is returned. Either check firing is an
    :class:`ArchiveFetchError`: the object this process holds must have come from
    the host that was asked, over TLS, or it is not evidence about that host.

    A 404 is returned as ``None`` — the venue answering that an object is not
    published is information, and it is exactly the answer amendment A9 expects
    for a month that has not closed. Every other failure raises: a proxy that
    refused, a name that would not resolve and a connection that dropped have
    said nothing about the venue at all, and turning one of them into "not
    published" would turn a local fault into a claim about the exchange.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_bytes: int = MAX_OBJECT_BYTES,
    ) -> None:
        self.timeout = float(timeout)
        self.max_bytes = int(max_bytes)

    def __call__(self, path: str) -> bytes | None:
        url = archive_url(path)
        request = urllib.request.Request(url, method="GET")
        try:
            with _ARCHIVE_OPENER.open(request, timeout=self.timeout) as response:
                served = urllib.parse.urlsplit(response.geturl())
                if (served.scheme, served.netloc) != ("https", ARCHIVE_HOST):
                    raise ArchiveFetchError(
                        f"{url} was served from {response.geturl()!r}. This module reads "
                        f"{ARCHIVE_HOST!r} over HTTPS and no other origin, and bytes that "
                        "arrived from somewhere else are not evidence about that host"
                    )
                body = response.read(self.max_bytes + 1)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if 300 <= exc.code < 400:
                location = None if exc.headers is None else exc.headers.get("Location")
                raise ArchiveFetchError(
                    f"{url} answered HTTP {exc.code} redirecting to "
                    f"{location!r}. A redirect is refused rather than "
                    f"followed: {ARCHIVE_HOST!r} is the one host this module reads, and a "
                    "redirected object would arrive with a digest from the same redirected "
                    "origin"
                ) from exc
            raise ArchiveFetchError(f"{url} answered HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ArchiveFetchError(f"{url} could not be reached: {exc.reason}") from exc
        except OSError as exc:
            raise ArchiveFetchError(f"{url} could not be reached: {exc}") from exc
        if len(body) > self.max_bytes:
            raise ArchiveFetchError(f"{url} is larger than {self.max_bytes} bytes")
        return body


# --- reading a published object ------------------------------------------------
@dataclass
class ArchiveRead:
    """One object's acquisition: the outcome, and the provenance behind it."""

    obj: ArchiveObject
    outcome: ArchiveOutcome
    detail: str | None = None
    sha256: str | None = None
    published_digest: str | None = None
    member: str | None = None
    rows: int | None = None
    #: Whether the bytes came from the path-keyed cache. Deliberately **not**
    #: persisted: it is a fact about this host's disk rather than about the
    #: evidence, a cached body reached this point only by hashing to the digest
    #: the venue published on this run, and a record that carried it would differ
    #: between two hosts holding identical bytes — which is precisely the
    #: determinism the report claims and a reviewer relies on when diffing two
    #: reports instead of re-running the fetch.
    cached: bool = False
    #: Minute open (ms) to published field values, or settlement rows for the
    #: funding family. Empty and meaningless unless the outcome is ``VERIFIED``.
    values: dict[int, dict[str, Decimal]] = field(default_factory=dict)
    intervals: dict[int, int] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.outcome is ArchiveOutcome.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.obj.family.family_id,
            "path": self.obj.path,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "layout": self.obj.family.layout.layout_id if self.verified else None,
            "epoch_unit": self.obj.family.layout.epoch_unit if self.verified else None,
            "member": self.member,
            "sha256": self.sha256,
            "published_digest": self.published_digest,
            "rows": self.rows,
        }


def _rows_of(payload: bytes) -> list[list[str]]:
    """The member's records, or a refusal that says the bytes could not be read.

    Both failures below are :class:`ArchiveContentError` rather than the
    exceptions the standard library raises, because :func:`acquire` maps the
    typed refusals onto :class:`ArchiveOutcome` values and an exception it does
    not catch is not one of the eight outcomes — it would escape the whole
    reconciliation and leave the day with no record at all, including for the
    streams that were fine. A member that is not UTF-8 and a member with a field
    larger than :mod:`csv` will parse are both "the venue published bytes this
    build cannot read", which is exactly ``UNPARSEABLE``.
    """
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ArchiveContentError(f"the member is not UTF-8 text: {exc}") from exc
    try:
        records = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        raise ArchiveContentError(f"the member is not readable as CSV: {exc}") from exc
    return [row for row in records if row and any(cell.strip() for cell in row)]


def _looks_like_header(row: Sequence[str], columns: Sequence[str]) -> bool:
    """Whether the first record names columns rather than carrying values.

    Decided by names. A "the first field did not parse as a number" heuristic
    silently discards a corrupt first data row as though it were a header, which
    loses a row and hides the corruption in the same move.
    """
    cells = {cell.strip().lower() for cell in row}
    return bool(cells) and set(name.lower() for name in columns) <= cells


def _column_index(
    layout: ArchiveLayout, records: list[list[str]], needed: Sequence[str]
) -> tuple[dict[str, int], int]:
    """Where each needed column is, and where the data starts. Layout-driven.

    A header where the layout says there is none, or none where it says there is
    one, is an unknown layout and a refusal — not something to auto-detect. The
    epoch unit hangs off the layout, so quietly accepting the other shape would
    quietly reinterpret every instant in the file.
    """
    first = records[0]
    header_present = _looks_like_header(first, layout.columns)
    if layout.has_header and not header_present:
        raise ArchiveLayoutError(
            f"layout {layout.layout_id} publishes a header row naming {list(layout.columns)} "
            f"and the member starts with {first[:4]}"
        )
    if not layout.has_header and header_present:
        raise ArchiveLayoutError(
            f"layout {layout.layout_id} publishes no header row and the member starts with "
            "one. The epoch unit is a property of the layout, so reading this member under "
            "it would reinterpret every instant in the file"
        )
    if layout.has_header:
        names = [cell.strip().lower() for cell in first]
        missing = [name for name in needed if name not in names]
        if missing:
            raise ArchiveLayoutError(f"the member's header does not name {missing}")
        return {name: names.index(name) for name in needed}, 1
    if len(first) != len(layout.columns):
        raise ArchiveLayoutError(
            f"layout {layout.layout_id} publishes {len(layout.columns)} columns and the "
            f"member's first row has {len(first)}"
        )
    positions = {name: index for index, name in enumerate(layout.columns)}
    return {name: positions[name] for name in needed}, 0


def _decimal(text: str, *, column: str, line: int) -> Decimal:
    stripped = text.strip()
    if not stripped:
        raise ArchiveContentError(f"line {line}: column {column!r} is empty")
    try:
        value = Decimal(stripped)
    except InvalidOperation as exc:
        raise ArchiveContentError(
            f"line {line}: column {column!r} is not a number: {text!r}"
        ) from exc
    if not value.is_finite():
        raise ArchiveContentError(f"line {line}: column {column!r} is not finite: {text!r}")
    return value


def _instant_ms(text: str, *, layout: ArchiveLayout, column: str, line: int) -> int:
    """One published instant, in milliseconds, under the layout's declared unit.

    The unit comes from the layout and never from the magnitude. The exact
    division is a consequence, not a resolution: a microsecond instant that is
    not a whole number of milliseconds is refused rather than rounded, because a
    rounded instant is a minute key nobody published.
    """
    value = _decimal(text, column=column, line=line)
    if value != value.to_integral_value():
        raise ArchiveContentError(f"line {line}: column {column!r} is not integral: {text!r}")
    instant = int(value)
    if layout.epoch_unit == "ms":
        return instant
    if layout.epoch_unit == "us":
        if instant % 1_000:
            raise ArchiveContentError(
                f"line {line}: column {column!r} is {instant} microseconds, which is not a "
                f"whole millisecond under layout {layout.layout_id}"
            )
        return instant // 1_000
    raise ArchiveLayoutError(f"layout {layout.layout_id} declares no known epoch unit")


def read_minute_object(
    payload: bytes, *, family: ArchiveFamily, day: str
) -> dict[int, dict[str, Decimal]]:
    """Every minute the object publishes for ``day``, as published decimals.

    Every row is checked to belong to the day the object claims and to open on a
    minute boundary. That is not a filter: a row outside is a refusal, because an
    object holding another day's minutes is not the object that was asked for,
    and it is also the cross-check that catches a mis-declared epoch unit before
    a single value is compared.
    """
    layout = family.layout
    records = _rows_of(payload)
    if not records:
        raise ArchiveContentError("the member holds no rows")
    needed = ("open_time",) + tuple(family.fields)
    index, start = _column_index(layout, records, needed)
    if len(records) <= start:
        raise ArchiveContentError("the member holds a header and no data rows")
    first_ms = day_start_ns(day) // NS_PER_MILLISECOND
    last_ms = first_ms + 24 * 60 * MS_PER_MINUTE
    minutes: dict[int, dict[str, Decimal]] = {}
    for line, row in enumerate(records[start:], start=start + 1):
        if len(row) <= max(index.values()):
            raise ArchiveContentError(f"line {line}: {len(row)} columns, fewer than needed")
        opened = _instant_ms(
            row[index["open_time"]], layout=layout, column="open_time", line=line
        )
        if opened % MS_PER_MINUTE:
            raise ArchiveContentError(f"line {line}: open_time {opened} is not a minute open")
        if not first_ms <= opened < last_ms:
            raise ArchiveContentError(
                f"line {line}: open_time {opened} ({iso_utc(opened * NS_PER_MILLISECOND)}) is "
                f"outside {day}, so this object is not the day it is named for"
            )
        if opened in minutes:
            raise ArchiveContentError(f"line {line}: minute {opened} is published twice")
        minutes[opened] = {
            name: _decimal(row[index[name]], column=name, line=line) for name in family.fields
        }
    return minutes


def read_funding_object(
    payload: bytes, *, family: ArchiveFamily, month: str
) -> tuple[dict[int, Decimal], dict[int, int]]:
    """Every settlement the monthly object publishes, and its interval metadata.

    Returns the realised rate per settlement instant and, separately, the funding
    interval where the source publishes one. The interval is metadata that is
    recorded because the source carries it; the count and the cadence of
    settlements are read from the source and never assumed, so a venue that
    changes its interval does not silently fail or silently pass a day.
    """
    layout = family.layout
    records = _rows_of(payload)
    if not records:
        raise ArchiveContentError("the member holds no rows")
    needed = ("calc_time", "last_funding_rate")
    index, start = _column_index(layout, records, needed)
    interval_at: int | None = None
    if layout.has_header:
        names = [cell.strip().lower() for cell in records[0]]
        if "funding_interval_hours" in names:
            interval_at = names.index("funding_interval_hours")
    elif "funding_interval_hours" in layout.columns:
        interval_at = layout.columns.index("funding_interval_hours")
    if len(records) <= start:
        raise ArchiveContentError("the member holds a header and no data rows")
    rates: dict[int, Decimal] = {}
    intervals: dict[int, int] = {}
    for line, row in enumerate(records[start:], start=start + 1):
        if len(row) <= max(index.values()):
            raise ArchiveContentError(f"line {line}: {len(row)} columns, fewer than needed")
        settled = _instant_ms(
            row[index["calc_time"]], layout=layout, column="calc_time", line=line
        )
        if iso_utc(settled * NS_PER_MILLISECOND)[:7] != month:
            raise ArchiveContentError(
                f"line {line}: calc_time {settled} is outside {month}, so this object is not "
                "the month it is named for"
            )
        if settled in rates:
            raise ArchiveContentError(f"line {line}: settlement {settled} is published twice")
        rates[settled] = _decimal(
            row[index["last_funding_rate"]], column="last_funding_rate", line=line
        )
        if interval_at is not None and interval_at < len(row) and row[interval_at].strip():
            hours = _decimal(row[interval_at], column="funding_interval_hours", line=line)
            if hours == hours.to_integral_value():
                intervals[settled] = int(hours)
    return rates, intervals


def acquire(
    obj: ArchiveObject,
    fetch: Fetcher,
    *,
    day: str,
    cache: ArchiveCache | None = None,
) -> ArchiveRead:
    """Fetch, verify and parse one published object, or say exactly why not.

    Every failure below is a distinct recorded outcome and not one of them is
    "the venue published nothing".

    **The companion is fetched first, and it is fetched every time.** It is a
    hundred bytes, and it is the digest every other step is decided against — so
    asking for it first is what lets a cached body be re-verified against what
    the venue publishes *now* rather than against a digest that was cached beside
    it. A companion that is absent is not treated as an absent object: the object
    itself is asked for, and the two cases are reported apart, because "the venue
    has not published this day" and "the venue published it without a digest" are
    different findings and only the first is expected.
    """
    try:
        companion = fetch(obj.checksum_path)
    except ArchiveFetchError as exc:
        return ArchiveRead(obj, ArchiveOutcome.FETCH_FAILED, detail=str(exc))
    if companion is None:
        try:
            body = fetch(obj.path)
        except ArchiveFetchError as exc:
            return ArchiveRead(obj, ArchiveOutcome.FETCH_FAILED, detail=str(exc))
        if body is None:
            return ArchiveRead(
                obj,
                ArchiveOutcome.ABSENT,
                detail="the venue answers that this object is not published",
            )
        return ArchiveRead(
            obj,
            ArchiveOutcome.CHECKSUM_ABSENT,
            detail="the object is published and its checksum companion is not",
            sha256=hashlib.sha256(body).hexdigest(),
        )
    try:
        published = parse_checksum_companion(companion, expected_name=obj.basename)
    except ArchiveChecksumError as exc:
        return ArchiveRead(obj, ArchiveOutcome.CHECKSUM_MALFORMED, detail=str(exc))

    body = None if cache is None else cache.read(obj.path, published_digest=published)
    cached = body is not None
    if body is None:
        try:
            body = fetch(obj.path)
        except ArchiveFetchError as exc:
            return ArchiveRead(
                obj, ArchiveOutcome.FETCH_FAILED, detail=str(exc), published_digest=published
            )
        if body is None:
            return ArchiveRead(
                obj,
                ArchiveOutcome.ABSENT,
                detail=(
                    "the venue publishes a checksum companion for this object and answers "
                    "that the object itself is not published"
                ),
                published_digest=published,
            )
    digest = hashlib.sha256(body).hexdigest()
    if published != digest:
        return ArchiveRead(
            obj,
            ArchiveOutcome.CHECKSUM_MISMATCH,
            detail=(
                f"the venue publishes {published} and the bytes received hash to {digest}. A "
                "mismatch is a refusal: the object is not read, not recorded with a flag, "
                "not repaired and not retried into acceptance"
            ),
            sha256=digest,
            published_digest=published,
        )
    if cache is not None and not cached:
        cache.write(obj.path, body)
    try:
        member = extract_expected_member(body, expected_name=obj.member_name)
    except ArchiveCorruptError as exc:
        return ArchiveRead(
            obj,
            ArchiveOutcome.CORRUPT_ARCHIVE,
            detail=str(exc),
            sha256=digest,
            published_digest=published,
        )
    except ArchiveMemberError as exc:
        return ArchiveRead(
            obj,
            ArchiveOutcome.UNEXPECTED_MEMBER,
            detail=str(exc),
            sha256=digest,
            published_digest=published,
        )
    read = ArchiveRead(
        obj,
        ArchiveOutcome.VERIFIED,
        sha256=digest,
        published_digest=published,
        member=obj.member_name,
        cached=cached,
    )
    try:
        if obj.family.period_kind == "monthly":
            rates, intervals = read_funding_object(
                member, family=obj.family, month=month_of(day)
            )
            read.values = {key: {"last_funding_rate": value} for key, value in rates.items()}
            read.intervals = intervals
            read.rows = len(rates)
        else:
            minutes = read_minute_object(member, family=obj.family, day=day)
            read.values = minutes
            read.rows = len(minutes)
    except (ArchiveLayoutError, ArchiveContentError) as exc:
        return ArchiveRead(
            obj,
            (
                ArchiveOutcome.UNKNOWN_LAYOUT
                if isinstance(exc, ArchiveLayoutError)
                else ArchiveOutcome.UNPARSEABLE
            ),
            detail=str(exc),
            sha256=digest,
            published_digest=published,
            member=obj.member_name,
            cached=cached,
        )
    return read


# --- comparing ------------------------------------------------------------------
def values_agree(
    published: Decimal, recorded: float, tolerance: float = RELATIVE_TOLERANCE
) -> bool:
    """Whether one published value and one recorded value are the same number.

    Exact equality first, which is the case that actually happens: the recorder
    parsed the same decimal string the archive publishes, so the two float64
    values are usually bit-identical.

    **At zero the comparison is exact rather than relative.** A relative
    tolerance around zero admits only zero itself — ``|x - 0| <= tol * 0`` is
    ``x == 0`` — so a published zero agrees with a recorded zero and with nothing
    else. That is stated rather than implemented as a division, because dividing
    by the published value would raise on precisely the minute a venue published
    no volume, which is a normal minute rather than an error.

    A non-finite recorded value never agrees. The normalized schema cannot
    produce one, and a comparison that quietly returned ``False`` for a NaN
    without saying so would hide a corrupt table as a disagreement.

    **And neither does a published value that does not survive float64.**
    ``Decimal("1E400")`` is a perfectly finite decimal, so the parser accepts it,
    and ``float()`` of it is ``inf``. Without the guard below the last line would
    then read ``|x - inf| <= tol * inf``, which is ``inf <= inf``, which is
    ``True`` for *every* recorded value — a corrupt published number would count
    as agreement on every minute it appeared in, straight into the numerator of
    ``published_coverage``. A published value that cannot be represented cannot
    be shown to agree with anything, so it does not.
    """
    reference = float(published)
    if not math.isfinite(reference):
        return False
    if reference == recorded:
        return True
    if recorded != recorded or recorded in (float("inf"), float("-inf")):
        return False
    if reference == 0.0:
        return False
    return abs(recorded - reference) <= tolerance * abs(reference)


@dataclass
class StreamReconciliation:
    """One minute-indexed stream's comparison against its archive."""

    stream: str
    read: ArchiveRead
    judged: bool
    reason: str | None
    published: int
    recorder_present: int
    compared: int
    agreeing: int
    archive_only: tuple[int, ...]
    recorder_only: tuple[int, ...]
    disagreements: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_kind": "minute",
            "judged": self.judged,
            "reason": self.reason,
            "archive": self.read.to_dict(),
            "published_minutes": self.published,
            "recorder_minutes": self.recorder_present,
            "compared_minutes": self.compared,
            "agreeing_minutes": self.agreeing,
            "disagreeing_minutes": len(self.disagreements),
            "archive_only_minutes": list(self.archive_only),
            "recorder_only_minutes": list(self.recorder_only),
            "disagreements": [dict(entry) for entry in self.disagreements],
        }


def _minute_values(frame: pd.DataFrame, family: ArchiveFamily) -> dict[int, dict[str, float]]:
    """The recorder's side of the comparison, straight out of the normalized day.

    Read, never written. A minute whose presence flag is false, or whose value is
    null, is a minute this recorder does not hold, and it is left out rather than
    defaulted to zero — the whole point of the normalized schema's nullability is
    that "no value" and "the value zero" stay different facts.
    """
    columns = ["minute_open_ms"] + [family.column_of[name] for name in family.fields]
    if family.presence_column is not None:
        columns.append(family.presence_column)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise RecorderReconcileError(
            f"the normalized day has no column(s) {missing}; a table this build does not "
            "recognise is refused rather than compared field by field against guesses"
        )
    captured: dict[int, dict[str, float]] = {}
    for row in frame[columns].itertuples(index=False):
        values = dict(zip(columns, row))
        if family.presence_column is not None:
            flag = values[family.presence_column]
            if pd.isna(flag) or not bool(flag):
                continue
        minute = values["minute_open_ms"]
        if pd.isna(minute):
            continue
        fields: dict[str, float] = {}
        incomplete = False
        for name in family.fields:
            value = values[family.column_of[name]]
            if pd.isna(value):
                incomplete = True
                break
            fields[name] = float(value)
        if incomplete:
            continue
        captured[int(minute)] = fields
    return captured


def _compare_minutes(
    published: Mapping[int, Mapping[str, Decimal]],
    captured: Mapping[int, Mapping[str, float]],
    *,
    fields: Sequence[str],
    tolerance: float,
) -> tuple[int, tuple[dict[str, Any], ...], tuple[int, ...], tuple[int, ...]]:
    """Agreeing count, the disagreements listed, and each side's exclusive minutes."""
    agreeing = 0
    disagreements: list[dict[str, Any]] = []
    for minute in sorted(set(published) & set(captured)):
        differing = [
            {
                "field": name,
                "archive": str(published[minute][name]),
                "recorder": repr(captured[minute][name]),
            }
            for name in fields
            if not values_agree(published[minute][name], captured[minute][name], tolerance)
        ]
        if differing:
            disagreements.append(
                {
                    "minute_open_ms": minute,
                    "minute_utc": iso_utc(minute * NS_PER_MILLISECOND),
                    "fields": differing,
                }
            )
        else:
            agreeing += 1
    archive_only = tuple(sorted(set(published) - set(captured)))
    recorder_only = tuple(sorted(set(captured) - set(published)))
    return agreeing, tuple(disagreements), archive_only, recorder_only


def _reconcile_stream(
    stream: str,
    family: ArchiveFamily,
    *,
    root: Path,
    day: str,
    contract: RecorderContract,
    fetch: Fetcher,
    tolerance: float,
    cache: ArchiveCache | None,
) -> StreamReconciliation:
    read = acquire(archive_object(family, contract, day), fetch, day=day, cache=cache)
    normalizer = MinuteNormalizer(root, contract)
    parquet = normalizer.parquet_path(family.market, day)
    if parquet.exists():
        try:
            frame = pd.read_parquet(parquet)
        except Exception as exc:  # pragma: no cover - pyarrow raises many shapes
            raise RecorderReconcileError(
                f"{parquet.relative_to(root).as_posix()} cannot be read: {exc}. A normalized "
                "day that will not open is a finding, never an empty capture"
            ) from exc
        captured = _minute_values(frame, family)
    else:
        # No normalized day is a capture of nothing, which is a real and
        # reportable state: the archive still supplies the denominator, and the
        # coverage that comes out is zero rather than undefined.
        captured = {}
    if not read.verified:
        return StreamReconciliation(
            stream=stream,
            read=read,
            judged=False,
            reason=f"{read.outcome.value}: {read.detail}",
            published=0,
            recorder_present=len(captured),
            compared=0,
            agreeing=0,
            archive_only=(),
            recorder_only=(),
            disagreements=(),
        )
    published = read.values
    agreeing, disagreements, archive_only, recorder_only = _compare_minutes(
        published, captured, fields=family.fields, tolerance=tolerance
    )
    return StreamReconciliation(
        stream=stream,
        read=read,
        judged=True,
        reason=None,
        published=len(published),
        recorder_present=len(captured),
        compared=len(set(published) & set(captured)),
        agreeing=agreeing,
        archive_only=archive_only,
        recorder_only=recorder_only,
        disagreements=disagreements,
    )


# --- funding ---------------------------------------------------------------------
def read_recorder_settlements(root: str | Path, market: str) -> dict[int, Decimal]:
    """The realised rate per settlement instant the recorder holds for a market.

    Only the settlement instant and the realised rate are read. The mark price
    the recorder captured is deliberately not returned: it is not published by
    the funding archive, it is not part of ``funding_complete``, and a value this
    function did not hand out cannot accidentally become a comparison
    (amendment A4).
    """
    path = Path(root) / FUNDING_DIRECTORY / market / SETTLEMENTS_FILE
    if not path.exists():
        return {}
    settlements: dict[int, Decimal] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecorderReconcileError(f"{path} cannot be read: {exc}") from exc
    for line, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
            instant = int(record["funding_time_ms"])
            rate = Decimal(str(record["funding_rate"]))
        except (ValueError, KeyError, TypeError, InvalidOperation) as exc:
            raise RecorderReconcileError(
                f"{path} line {line} is not a settlement record: {exc}. A settlements file "
                "that cannot be read is a finding, never an absence of settlements"
            ) from exc
        settlements[instant] = rate
    return settlements


@dataclass
class FundingReconciliation:
    """The day's funding verdict, and the evidence it rests on."""

    read: ArchiveRead
    schedule_established: bool
    outcome: str
    scheduled: tuple[dict[str, Any], ...]
    captured: tuple[int, ...]
    missing: tuple[int, ...]
    disagreeing: tuple[dict[str, Any], ...]
    recorder_only: tuple[int, ...]
    #: Whether the verified monthly object demonstrated that it covers the whole
    #: of the day — see :func:`schedule_coverage_witness_ms`. False whenever the
    #: object was not verified at all, so that the record never has to be read
    #: as claiming coverage it did not establish.
    covers_day: bool = False
    #: The latest settlement the month object published, as UTC text, or ``None``
    #: when there is no verified object to have published one. Recorded so that
    #: what establishment rested on is visible rather than implied.
    month_last_settlement_utc: str | None = None
    #: Why the schedule could not be established, when it could not.
    reason: str | None = None

    @property
    def funding_complete(self) -> bool:
        """Established, and every scheduled settlement captured in agreement.

        A quantifier over an established set, never a quotient: a day whose
        established schedule is empty is complete because the universal holds
        over the empty set, and ``0 / 0`` is evaluated nowhere.
        """
        return self.schedule_established and not self.missing and not self.disagreeing

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_kind": "settlement",
            "archive": self.read.to_dict(),
            "schedule_established": self.schedule_established,
            "covers_day": self.covers_day,
            "month_last_settlement_utc": self.month_last_settlement_utc,
            "reason": self.reason,
            "outcome": self.outcome,
            "funding_complete": self.funding_complete,
            "scheduled": len(self.scheduled),
            "captured": len(self.captured),
            "scheduled_settlements": [dict(entry) for entry in self.scheduled],
            "captured_settlements": list(self.captured),
            "missing_settlements": list(self.missing),
            "disagreeing_settlements": [dict(entry) for entry in self.disagreeing],
            "recorder_only_settlements": list(self.recorder_only),
            "wallclock_coverage": None,
            "settlement_mark_price": (
                "The recorder may capture the settlement mark price from its live public "
                "source and keeps its provenance. The funding archive publishes none, so "
                "that value is verified by nothing here, is not part of funding_complete, "
                "and is never reconstructed, nearest-minute matched, filled or REST-replaced."
            ),
        }


def _reconcile_funding(
    *,
    root: Path,
    day: str,
    contract: RecorderContract,
    fetch: Fetcher,
    cache: ArchiveCache | None,
) -> FundingReconciliation:
    """The day's funding verdict, from the monthly object and the recorder's file.

    Three findings are kept apart, because collapsing any two of them is how a
    day passes that should not (amendment A2). The object may fail to establish
    the schedule at all — absent, unverified, unparseable, or verified and
    covering only part of the day; the object may establish that nothing was
    scheduled, which is completeness over the empty set; or a scheduled
    settlement may be missing or disagree, which fails the day outright.
    Agreement is on the settlement instant and the realised rate and on nothing
    else, because the monthly archive publishes no settlement mark price
    (amendment A4).
    """
    family = ARCHIVE_FAMILIES[UM_FUNDING]
    obj = archive_object(family, contract, day)
    read = acquire(obj, fetch, day=day, cache=cache)
    if not read.verified:
        return FundingReconciliation(
            read=read,
            schedule_established=False,
            outcome=FUNDING_SCHEDULE_UNAVAILABLE,
            scheduled=(),
            captured=(),
            missing=(),
            disagreeing=(),
            recorder_only=(),
            reason=f"{read.outcome.value}: {read.detail}",
        )
    published_settlements = sorted(read.values)
    last_published_ms = published_settlements[-1] if published_settlements else None
    last_published_utc = (
        None if last_published_ms is None else iso_utc(last_published_ms * NS_PER_MILLISECOND)
    )
    witness_ms = schedule_coverage_witness_ms(day)
    if last_published_ms is None or last_published_ms < witness_ms:
        # The fourth clause of schedule_established(D): the object verifies and
        # still does not show that it covers the whole day. Refusing here is what
        # stops a month published before it closed from handing every day after
        # its last settlement an empty schedule — which the universal over the
        # empty set would then read as completeness, turning missing evidence
        # into a pass on exactly the days amendment A2 was written about.
        return FundingReconciliation(
            read=read,
            schedule_established=False,
            outcome=FUNDING_SCHEDULE_UNAVAILABLE,
            scheduled=(),
            captured=(),
            missing=(),
            disagreeing=(),
            recorder_only=(),
            month_last_settlement_utc=last_published_utc,
            reason=(
                f"the monthly object {obj.path} verifies and its last settlement is "
                f"{last_published_utc}, which does not reach "
                f"{iso_utc(witness_ms * NS_PER_MILLISECOND)}. It therefore covers only part "
                f"of {day}, and a schedule that covers only part of a day is not an "
                "established empty schedule and is never recorded as one"
            ),
        )
    first_ms = day_start_ns(day) // NS_PER_MILLISECOND
    last_ms = first_ms + 24 * 60 * MS_PER_MINUTE
    scheduled_rates = {
        instant: values["last_funding_rate"]
        for instant, values in read.values.items()
        if first_ms <= instant < last_ms
    }
    recorded = read_recorder_settlements(root, family.market)
    captured: list[int] = []
    missing: list[int] = []
    disagreeing: list[dict[str, Any]] = []
    scheduled: list[dict[str, Any]] = []
    for instant in sorted(scheduled_rates):
        rate = scheduled_rates[instant]
        entry: dict[str, Any] = {
            "funding_time_ms": instant,
            "funding_time_utc": iso_utc(instant * NS_PER_MILLISECOND),
            "funding_rate": str(rate),
        }
        if instant in read.intervals:
            entry["funding_interval_hours"] = read.intervals[instant]
        scheduled.append(entry)
        if instant not in recorded:
            missing.append(instant)
        elif recorded[instant] != rate:
            disagreeing.append(
                {
                    "funding_time_ms": instant,
                    "funding_time_utc": iso_utc(instant * NS_PER_MILLISECOND),
                    "archive": str(rate),
                    "recorder": str(recorded[instant]),
                }
            )
        else:
            captured.append(instant)
    recorder_only = tuple(
        sorted(
            instant
            for instant in recorded
            if first_ms <= instant < last_ms and instant not in scheduled_rates
        )
    )
    return FundingReconciliation(
        read=read,
        schedule_established=True,
        outcome="OK",
        scheduled=tuple(scheduled),
        captured=tuple(captured),
        missing=tuple(missing),
        disagreeing=tuple(disagreeing),
        recorder_only=recorder_only,
        covers_day=True,
        month_last_settlement_utc=last_published_utc,
    )


# --- the report -------------------------------------------------------------------
@dataclass
class ReconciliationReport:
    """One UTC day's reconciliation, exactly as it is persisted."""

    day: str
    contract_id: str
    contract_hash: str
    prospective_from: str | None
    streams: tuple[StreamReconciliation, ...]
    funding: FundingReconciliation
    diagnostics: tuple[tuple[str, dict[str, Any]], ...]
    tolerance: float

    @property
    def evidence_class(self) -> str:
        """What this **day** is, which is not the same as what the contract is.

        An activated root may hold engineering observations recorded before the
        boundary, and only observations at or after ``prospective_from`` count
        toward the coverage streak — so the class is a fact about the day and is
        tested against the boundary rather than inferred from the contract having
        one. Writing ``prospective`` onto a pre-boundary day would relabel
        engineering data as scientific evidence in a committed evidence file,
        which is the one thing the boundary rule forbids in terms, and it would
        do it in the very document a later reviewer reads to check the claim.
        """
        if self.prospective_from is None:
            return "engineering"
        return "prospective" if self.day >= self.prospective_from[:10] else "engineering"

    def to_dict(self) -> dict[str, Any]:
        return {
            "reconciliation_schema": RECONCILIATION_SCHEMA,
            "day": self.day,
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "prospective_from": self.prospective_from,
            "evidence_class": self.evidence_class,
            "archive_host": ARCHIVE_HOST,
            "tolerance_relative": self.tolerance,
            "streams": {entry.stream: entry.to_dict() for entry in self.streams},
            "funding": self.funding.to_dict(),
            "diagnostics": {name: dict(body) for name, body in self.diagnostics},
            "note": (
                "Coverage evidence. Every number here was published by the venue or "
                "recorded from it and carried across unchanged; no return, funding flow, "
                "basis, carry or profit is computed anywhere. The diagnostics section gates "
                "nothing."
            ),
        }


def _forbid_non_finite(document: Any, where: str = "report") -> None:
    """Refuse a NaN or an Infinity before it reaches a JSON file.

    ``json.dumps`` writes both as bare ``NaN`` and ``Infinity`` tokens, which are
    not JSON, which every strict reader refuses, and which no evidence file may
    contain. Checked here rather than hoped for.
    """
    if isinstance(document, Mapping):
        for key, value in document.items():
            _forbid_non_finite(value, f"{where}.{key}")
    elif isinstance(document, (list, tuple)):
        for index, value in enumerate(document):
            _forbid_non_finite(value, f"{where}[{index}]")
    elif isinstance(document, float):
        if document != document or document in (float("inf"), float("-inf")):
            raise RecorderReconcileError(
                f"{where} is {document!r}; an evidence file carries no NaN and no Infinity"
            )


def reconcile_day(
    root: str | Path,
    day: str,
    fetch: Fetcher,
    *,
    contract: RecorderContract,
    cache: ArchiveCache | None = None,
    tolerance: float = RELATIVE_TOLERANCE,
    index_diagnostic: bool = True,
) -> ReconciliationReport:
    """Reconcile one UTC day against the published archives, and report it.

    The minute-indexed streams come from
    :meth:`chimera.recorder.contract.RecorderContract.minute_indexed_required`,
    so this function reconciles exactly what the contract requires and never a
    list of its own. A required stream this module names no archive family for
    is refused outright rather than skipped, because a skipped required stream
    would leave the gate to divide by a denominator that was never established.

    Nothing under ``root`` is written by this call. The report is returned; it is
    persisted, when the caller wants it persisted, by
    :func:`write_reconciliation`.
    """
    resolved_root = Path(root)
    require_day(day)
    streams: list[StreamReconciliation] = []
    for stream in contract.minute_indexed_required():
        family = ARCHIVE_FAMILIES.get(stream)
        if family is None:
            raise RecorderReconcileError(
                f"recorder contract {contract.label} requires {stream!r} for coverage and "
                "this build names no first-party archive that publishes a minute "
                "denominator for it. A required stream with no denominator is a refusal, "
                "never a stream quietly left out of the report"
            )
        streams.append(
            _reconcile_stream(
                stream,
                family,
                root=resolved_root,
                day=day,
                contract=contract,
                fetch=fetch,
                tolerance=tolerance,
                cache=cache,
            )
        )
    funding = _reconcile_funding(
        root=resolved_root, day=day, contract=contract, fetch=fetch, cache=cache
    )

    diagnostics: list[tuple[str, dict[str, Any]]] = []
    if index_diagnostic:
        # The diagnostic is computed through the same function the gated streams
        # use, and that function *raises* on a normalized table it cannot read.
        # Letting such a raise escape would abort the whole day: no record for
        # any stream, and um.markPrice denied a verdict by an index-side fault.
        # Amendment A6 says index agreement is not a pass condition for
        # um.markPrice, so it must not be able to withhold that stream's record
        # either. A diagnostic that could not be computed says so and gates
        # nothing, which is what a diagnostic is for.
        try:
            body = _reconcile_stream(
                "um.indexPrice",
                INDEX_DIAGNOSTIC_FAMILY,
                root=resolved_root,
                day=day,
                contract=contract,
                fetch=fetch,
                tolerance=tolerance,
                cache=cache,
            ).to_dict()
        except RecorderReconcileError as exc:
            body = {"index_kind": "minute", "judged": False, "reason": str(exc)}
        body["gates_nothing"] = True
        body["note"] = (
            "Diagnostic only (amendment A6). um.markPrice's archive criterion is the "
            "mark-price agreement criterion supported by markPriceKlines and nothing else; "
            "index agreement is not a second pass condition and no threshold is attached to "
            "it. The coverage gate never reads this section."
        )
        diagnostics.append(("um.indexPrice", body))

    report = ReconciliationReport(
        day=day,
        contract_id=contract.contract_id,
        contract_hash=contract.contract_hash,
        prospective_from=(
            None
            if contract.prospective_from is None
            else contract.prospective_from.isoformat()
        ),
        streams=tuple(streams),
        funding=funding,
        diagnostics=tuple(diagnostics),
        tolerance=tolerance,
    )
    _forbid_non_finite(report.to_dict())
    return report


def _establishments(document: Mapping[str, Any]) -> set[str]:
    """What one record establishes: the streams it judged, and the funding schedule.

    Named rather than counted, so that the refusal below can say which
    establishment a rewrite would have destroyed instead of reporting that a
    number went down.
    """
    established: set[str] = set()
    streams = document.get("streams")
    if isinstance(streams, Mapping):
        for name, entry in streams.items():
            if isinstance(entry, Mapping) and entry.get("judged") is True:
                established.add(f"stream {name}")
    funding = document.get("funding")
    if isinstance(funding, Mapping) and funding.get("schedule_established") is True:
        established.add("the funding schedule")
    return established


def write_reconciliation(root: str | Path, report: ReconciliationReport) -> Path:
    """Persist one day's record atomically under ``reconciliation/``.

    Atomic because a reader that found half a document would find a day it could
    not judge, and a day it cannot judge is a day that breaks a streak — a
    torn write would be a recorder outage invented by the writer.

    **A rewrite may add an establishment and may never remove one.** Amendment A9
    makes re-running the reconciliation the normal operating pattern: a day whose
    monthly funding archive had not been published yet is
    ``FUNDING_SCHEDULE_UNAVAILABLE`` and takes its real verdict when the archive
    appears. That direction is expected. The other direction is not evidence
    about anything: a transport outage on this host makes every object
    ``FETCH_FAILED``, and a cron re-running yesterday's days during one would
    otherwise overwrite good records with empty ones, silently destroying days
    that had been established and breaking a streak for a reason that has
    nothing to do with the recorder. So a write that would drop an establishment
    the stored record already holds is refused, the stored record is left exactly
    as it was, and the operator sees a failure instead of a quietly shorter
    streak.
    """
    document = report.to_dict()
    _forbid_non_finite(document)
    path = reconciliation_path(root, report.day)
    if path.exists():
        try:
            stored = read_reconciliation(root, report.day)
        except RecorderCoverageError:
            # A stored record this build cannot even read establishes nothing,
            # so replacing it can lose nothing. It is also exactly the case an
            # operator needs to be able to repair by re-running.
            stored = {}
        if stored.get("contract_hash") == report.contract_hash:
            lost = sorted(_establishments(stored) - _establishments(document))
            if lost:
                raise RecorderReconcileError(
                    f"the stored record for {report.day} establishes {lost} and this run "
                    f"does not, so writing it would destroy evidence. Re-running a "
                    "reconciliation is expected to move a day from unavailable to "
                    "established and never the other way; a run that establishes less than "
                    "the record it would replace is a fault on this host, not a finding "
                    "about the venue. The stored record is unchanged"
                )
    try:
        write_json_atomic(path, document)
    except RecorderSinkError as exc:
        raise RecorderReconcileError(str(exc)) from exc
    return path


__all__ = [
    "ARCHIVE_BASE",
    "ARCHIVE_FAMILIES",
    "ARCHIVE_HOST",
    "ArchiveCache",
    "ArchiveFamily",
    "ArchiveFetchError",
    "ArchiveLayout",
    "ArchiveObject",
    "ArchiveOutcome",
    "ArchiveRead",
    "CHECKSUM_SUFFIX",
    "FundingReconciliation",
    "HttpsArchiveFetcher",
    "INDEX_DIAGNOSTIC_FAMILY",
    "RECONCILIATION_DIRECTORY",
    "RECONCILIATION_SCHEMA",
    "RELATIVE_TOLERANCE",
    "ReconciliationReport",
    "RecorderReconcileError",
    "StreamReconciliation",
    "acquire",
    "archive_object",
    "archive_url",
    "extract_expected_member",
    "month_of",
    "parse_checksum_companion",
    "read_funding_object",
    "read_minute_object",
    "read_recorder_settlements",
    "reconcile_day",
    "require_archive_path",
    "schedule_coverage_witness_ms",
    "values_agree",
    "write_reconciliation",
]
