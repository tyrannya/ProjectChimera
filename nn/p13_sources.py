"""Reading P13's preregistered archives from LOCAL BYTES, and refusing everything else.

This module is the loader half of ``SOURCE_FREEZE_FIELDS``. It turns the bytes of
one published Binance archive object into typed rows, and it records what it did
so an auditor can check the reading rather than trust it.

**It contains no network code at all**, and that is structural rather than
incidental: every entry point takes ``bytes`` or a local ``Path``. There is no
URL parameter to accidentally populate, no session, no retry, and nothing that
could reach ``data.binance.vision``. Acquisition — fetching objects and verifying
them against Binance's published ``.CHECKSUM`` — is a separate chronology step
that has not happened. ``tools/acquire_p13_sources.py`` already plans the object
list without a network; this reads objects that a later stage will supply.

**Decimal, not float.** Prices arrive as text in the archive and stay text until
they become :class:`~decimal.Decimal`. Going through ``float`` would round every
price to 53 bits before the accounting engine — whose whole design premise is
that a carry result is what survives the cancellation of two large, nearly equal
legs — ever saw it.

**The unit is resolved, never assumed.** ``TIMESTAMP_UNIT_POLICY`` requires it:
this checkpoint straddles Binance's 2025-01-01 spot microsecond change, and spot
and futures may carry different units for the same calendar month. Every object's
unit is resolved against that object's OWN calendar period by
:func:`nn.trade_aggregates.resolve_epoch_unit`, recorded in the provenance, and
refused when ambiguous.

**The research boundary is asserted here, not filtered.** ``DATA_BOUNDARY``
says "a row at or after the boundary is a refusal, not a filter", with exactly one
carve-out that ``the_boundary_straddling_month`` names: the single archive month
that contains the boundary is fetched whole — a partial object cannot be
checksum-verified — and is TRUNCATED AT LOAD, with the dropped count and the
maximum surviving instant recorded. Every other month is refused rather than
truncated. This module implements that asymmetry literally: truncation requires
the caller to have identified the straddling month, and any other object carrying
a boundary-crossing row raises.

**Layouts are first-party facts, not choices.** ``FIRST_PARTY_SOURCE_EVIDENCE``
records that the archive column layouts were read from Binance's own published
code, and the funding layout is frozen outright in
``nn.p4_preregistration.FUNDING_CSV_COLUMN_POLICY``, which
``FUNDING_COLUMN_POLICY_SOURCE`` points this checkpoint at rather than restating.
An unrecognised layout is a refusal here for the same reason it is there: a reader
that infers a mapping is a reader that can silently read the wrong column.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from nn.p13_preregistration import DATA_BOUNDARY
from nn.p4_preregistration import FUNDING_CSV_COLUMN_POLICY
from nn.trade_aggregates import TradeAggregationError, resolve_epoch_unit

__all__ = [
    "SourceError",
    "CHECKSUM_NOT_SUPPLIED",
    "CHECKSUM_UNVERIFIED",
    "CHECKSUM_VERIFIED",
    "CHECKSUM_MISMATCH",
    "CHECKSUM_STATES",
    "verify_published_checksum",
    "KLINE_COLUMNS",
    "KlineRow",
    "FundingRow",
    "ObjectProvenance",
    "KlineTable",
    "FundingTable",
    "RESEARCH_BOUNDARY_NS",
    "period_bounds",
    "straddles_boundary",
    "extract_single_member",
    "read_kline_object",
    "read_funding_object",
    "read_object_bytes",
]


class SourceError(RuntimeError):
    """An archive object cannot be read into honest rows under the frozen rules."""


# ---------------------------------------------------------------------------
# Checksum verification state
# ---------------------------------------------------------------------------
#
# FOUR distinct facts, because they were previously collapsed into one boolean
# computed as ``published_checksum is not None`` — which reports "verified" for an
# object whose publisher digest was recorded and never compared with anything.
# That is the failure mode a checksum exists to prevent, reproduced by the field
# that claims to prevent it.

#: No publisher checksum was supplied for this object.
CHECKSUM_NOT_SUPPLIED = "no_publisher_checksum_supplied"

#: A publisher checksum was supplied, but the bytes it should be compared against
#: were not, so NO comparison was performed. This is not a failure and it is not a
#: pass; it is the honest answer when a member is parsed without its archive.
CHECKSUM_UNVERIFIED = "supplied_not_verified"

#: A publisher checksum was supplied AND an equality check against the
#: independently recomputed digest of the received bytes succeeded.
CHECKSUM_VERIFIED = "verified_match"

#: A publisher checksum was supplied and DISAGREED with the received bytes. It is
#: never stored on a provenance record, because a mismatch is a REFUSAL — the
#: object does not become evidence with a flag set. The name exists so the refusal
#: has one.
CHECKSUM_MISMATCH = "mismatch_refused"

CHECKSUM_STATES: tuple[str, ...] = (
    CHECKSUM_NOT_SUPPLIED,
    CHECKSUM_UNVERIFIED,
    CHECKSUM_VERIFIED,
    CHECKSUM_MISMATCH,
)


def verify_published_checksum(raw_object: bytes | None, published: str | None) -> str:
    """Compare a publisher digest against the bytes actually received, or refuse.

    Returns the state that ACTUALLY holds. It never returns
    :data:`CHECKSUM_VERIFIED` without having performed the equality check, and it
    never returns at all on a mismatch — an object whose bytes disagree with its
    publisher's digest is not read, because everything downstream would then be
    describing bytes nobody vouched for.

    ``published`` is accepted in either the bare-hex form or Binance's published
    ``<digest>  <filename>`` companion form, and the comparison is
    case-insensitive because hex digests are.
    """
    if published is None:
        return CHECKSUM_NOT_SUPPLIED
    expected = published.strip().split()[0].lower() if published.strip() else ""
    if not expected:
        raise SourceError(
            "a published checksum was supplied but is empty. An empty digest cannot be "
            "compared, and treating it as absent would silently downgrade a verification "
            "the caller asked for."
        )
    if raw_object is None:
        return CHECKSUM_UNVERIFIED
    actual = hashlib.sha256(raw_object).hexdigest()
    if actual != expected:
        raise SourceError(
            f"published checksum {expected} does not match the sha256 of the received "
            f"bytes {actual}. A mismatch is a REFUSAL: the object is not read, not "
            "recorded with a flag, and not repaired. Either the archive was revised — see "
            "ARCHIVE_REVISION_POLICY, which requires a revision event to be reported "
            "rather than silently accepted or silently rejected — or the bytes are "
            "corrupt, and neither is something a loader may decide on its own."
        )
    return CHECKSUM_VERIFIED


#: The research boundary in integer UTC nanoseconds, resolved from the frozen
#: design rather than restated, exactly as ``nn.p13_carry`` resolves it.
RESEARCH_BOUNDARY_NS = int(
    datetime.fromisoformat(DATA_BOUNDARY["span_end_exclusive"]).timestamp() * 1_000_000_000
)

#: Binance's published kline column layout, in order, as recorded by
#: ``FIRST_PARTY_SOURCE_EVIDENCE`` from the venue's own repository. The same
#: twelve columns carry ``klines`` and ``markPriceKlines``; the mark family simply
#: leaves the volume-shaped columns at zero, which this module never reads.
#:
#: Only the first five are used. The rest are named so an unrecognised HEADER can
#: be refused by column-name set — the discipline
#: ``FUNDING_CSV_COLUMN_POLICY.on_unrecognised_layout`` imposes on the funding
#: archive — rather than by counting fields and hoping.
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

#: The columns this module actually consumes from a kline row. ``high`` is here
#: because the mark family's high is the preferred liquidation touch; the spot and
#: perpetual families' highs are read and simply never used for anything, because
#: no frozen rule authorises a spot or perpetual high as a liquidation touch.
_KLINE_REQUIRED = ("open_time", "open", "high", "low", "close")


@dataclass(frozen=True)
class KlineRow:
    """One hourly candle, stamped by its OPEN instant.

    ``instant_ns`` is the candle's OPEN — ``DATA_SOURCES`` states the semantics as
    "the candle OPEN; the candle is complete at open + 1h" — so the only price in
    this row already knowable at ``instant_ns`` is :attr:`open`. Everything else
    is a fact about the hour that follows it, which is why the accounting engine
    fills at opens and marks at closes.
    """

    instant_ns: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True)
class FundingRow:
    """One realised funding settlement, exactly as published.

    ``interval_hours`` is carried when the layout provides it, per
    ``DATA_SOURCES``' funding ``interval_policy``: the settlement CADENCE is not
    assumed, every row is one settlement event, and nothing multiplies a rate by
    an assumed settlements-per-day count.
    """

    instant_ns: int
    rate: Decimal
    interval_hours: Decimal | None = None


@dataclass(frozen=True)
class ObjectProvenance:
    """What was read, from which bytes, under which unit, and what was dropped.

    Every field here is one of ``SOURCE_FREEZE_FIELDS``. It exists so the evidence
    can state the reading rather than assert it: an auditor who disagrees with the
    resolved unit or the truncation count can recompute both from the recorded
    digest and the object name.
    """

    field: str
    object_name: str
    period: str
    #: The size in bytes of the WHOLE PUBLISHED OBJECT — the zip — and ``None``
    #: when the caller supplied only the extracted member. It is not the member's
    #: size; that is :attr:`member_byte_size`.
    byte_size: int | None
    #: sha256 of the WHOLE PUBLISHED OBJECT, which is the digest Binance's
    #: ``.CHECKSUM`` companion carries and therefore the only one a publisher
    #: comparison can use. ``None`` when the archive bytes were not supplied.
    #:
    #: It used to fall back to the MEMBER's digest when the archive was absent,
    #: which silently produced a manifest whose "sha256" could not be checked
    #: against the publisher and gave no hint that it was a different quantity.
    #: The two digests are now separate fields and neither stands in for the other.
    sha256: str | None
    resolved_epoch_unit: str
    rows_read: int
    rows_dropped_at_boundary: int
    first_instant_ns: int | None
    last_instant_ns: int | None
    #: Instants withheld because two rows there disagreed — the ambiguity
    #: ``POSITION_LIFECYCLE.validity_definition`` calls invalid. Counted rather
    #: than merely dropped, because a hole this loader created must be as visible
    #: as one the archive shipped with.
    ambiguous_instants: int = 0
    #: Instants withheld because a price was not strictly positive, the other
    #: condition the same frozen sentence names. Kept as a SEPARATE count: a
    #: corrupt price and a contradictory duplicate are different facts about an
    #: archive, and one total would hide which had happened.
    non_positive_instants: int = 0
    #: Instants carrying more than one row that were KEPT rather than withheld —
    #: the funding path only. The frozen design resolves a repeated funding row in
    #: the accounting engine, so this loader counts them and passes them through.
    repeated_instants: int = 0
    #: The name of the single CSV member inside the published object, exactly as
    #: the archive carries it. ``SOURCE_FREEZE_FIELDS`` asks for the member to be
    #: identified and not merely counted.
    member_name: str | None = None
    #: sha256 of the EXTRACTED MEMBER — the CSV bytes the rows were actually
    #: parsed from. ``SOURCE_FREEZE_FIELDS`` requires BOTH this and the
    #: whole-object digest; only this one attests to what was READ, and only the
    #: whole-object one can be checked against the publisher.
    member_sha256: str = ""
    #: The extracted member's size in bytes, kept beside its digest for the same
    #: reason the archive's is kept beside the archive's.
    member_byte_size: int = 0
    #: Binance's own published digest for this object, when the acquisition
    #: supplied one.
    published_checksum: str | None = None
    #: WHICH of the four checksum facts actually holds — see
    #: :data:`CHECKSUM_STATES`. This replaces a boolean computed as
    #: ``published_checksum is not None``, which reported "verified" for any
    #: object that merely carried a digest string, whether or not anything had
    #: ever been compared with it.
    #:
    #: :meth:`__post_init__` refuses to construct a record claiming
    #: :data:`CHECKSUM_VERIFIED` unless the published digest and the recomputed
    #: archive digest are actually EQUAL, so the state cannot be set to "verified"
    #: by an edit that does not also make it true.
    checksum_state: str = CHECKSUM_NOT_SUPPLIED

    def __post_init__(self) -> None:
        if self.checksum_state not in CHECKSUM_STATES:
            raise SourceError(f"{self.checksum_state!r} is not one of {CHECKSUM_STATES}")
        if self.checksum_state == CHECKSUM_MISMATCH:
            raise SourceError(
                "a checksum mismatch is a REFUSAL, not a provenance record. An object "
                "whose bytes disagree with its publisher's digest is not read at all, so "
                "no provenance describing it may exist."
            )
        if self.checksum_state == CHECKSUM_NOT_SUPPLIED and self.published_checksum:
            raise SourceError(
                "a published checksum was supplied, so the state cannot be "
                f"{CHECKSUM_NOT_SUPPLIED!r}"
            )
        if self.checksum_state in (CHECKSUM_UNVERIFIED, CHECKSUM_VERIFIED) and not (
            self.published_checksum
        ):
            raise SourceError(
                f"state {self.checksum_state!r} claims a published checksum, and none is "
                "recorded"
            )
        if self.checksum_state == CHECKSUM_VERIFIED:
            # The equality check, performed HERE as well as at verification time.
            # A state that can be typed into a constructor is a state that can be
            # wrong; one the constructor re-derives cannot be.
            if self.sha256 is None:
                raise SourceError(
                    "a verified checksum requires the whole-object digest it was verified "
                    "against, and none is recorded"
                )
            expected = self.published_checksum.strip().split()[0].lower()
            if expected != self.sha256.lower():
                raise SourceError(
                    f"provenance claims {CHECKSUM_VERIFIED!r} but the published checksum "
                    f"{expected} and the recomputed archive digest {self.sha256} differ"
                )

    @property
    def checksum_verified(self) -> bool:
        """True ONLY for a digest that was actually compared and actually matched."""
        return self.checksum_state == CHECKSUM_VERIFIED

    def as_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "object": self.object_name,
            "period": self.period,
            "archive_byte_size": self.byte_size,
            "archive_sha256": self.sha256,
            "member_name": self.member_name,
            "member_sha256": self.member_sha256,
            "member_byte_size": self.member_byte_size,
            "resolved_epoch_unit": self.resolved_epoch_unit,
            "rows_read": self.rows_read,
            "rows_dropped_at_boundary": self.rows_dropped_at_boundary,
            "first_instant_ns": self.first_instant_ns,
            "last_instant_ns": self.last_instant_ns,
            "ambiguous_instants": self.ambiguous_instants,
            "non_positive_instants": self.non_positive_instants,
            "repeated_instants": self.repeated_instants,
            "published_checksum": self.published_checksum,
            "checksum_state": self.checksum_state,
            # Derived from the state rather than from the mere presence of a
            # digest string, which is what it used to be.
            "checksum_verified": self.checksum_verified,
        }


@dataclass(frozen=True)
class KlineTable:
    """The rows of one kline object, with the provenance of the reading."""

    provenance: ObjectProvenance
    rows: tuple[KlineRow, ...]


@dataclass(frozen=True)
class FundingTable:
    """The rows of one fundingRate object, with the provenance of the reading."""

    provenance: ObjectProvenance
    rows: tuple[FundingRow, ...]


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


def period_bounds(period: str) -> tuple[datetime, datetime]:
    """The UTC half-open span of a ``YYYY-MM`` monthly archive period.

    The archive's own calendar period is what decides its timestamp unit, so it
    is derived from the object's declared period rather than from the rows —
    deriving it from the rows would let a misread unit justify itself.
    """
    try:
        year, month = (int(part) for part in period.split("-"))
    except ValueError as exc:  # pragma: no cover - defensive
        raise SourceError(f"period {period!r} is not YYYY-MM") from exc
    if not 1 <= month <= 12:
        raise SourceError(f"period {period!r} names month {month}")
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(
        year + (month == 12), 1 if month == 12 else month + 1, 1, tzinfo=timezone.utc
    )
    return start, end


def straddles_boundary(period: str) -> bool:
    """Whether this monthly period is the ONE the boundary falls inside.

    ``DATA_BOUNDARY.the_boundary_straddling_month`` grants truncation to exactly
    this month and to nothing else, so the question is answered by the calendar
    rather than by whether a row happened to cross.
    """
    start, end = period_bounds(period)
    boundary = datetime.fromisoformat(DATA_BOUNDARY["span_end_exclusive"])
    return start < boundary < end


# ---------------------------------------------------------------------------
# Bytes in
# ---------------------------------------------------------------------------


def read_object_bytes(path: str | Path) -> bytes:
    """The bytes of a LOCAL archive object. No URL, no session, no fallback."""
    return Path(path).read_bytes()


def extract_single_member(raw: bytes) -> tuple[str, bytes]:
    """The one CSV inside a published archive zip.

    Binance publishes exactly one member per object. More than one is refused
    rather than resolved by picking the first: which member a reader chose would
    be an unrecorded decision about which rows the result came from.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise SourceError(
                    f"archive holds {len(names)} members {names}; a published Binance object "
                    "holds exactly one, and choosing among several would be an unrecorded "
                    "decision about which rows the result came from"
                )
            return names[0], archive.read(names[0])
    except zipfile.BadZipFile as exc:
        raise SourceError(f"not a readable zip archive: {exc}") from exc


def _decimal(text: str, *, column: str, line: int) -> Decimal:
    stripped = text.strip()
    if not stripped:
        raise SourceError(f"line {line}: column {column!r} is empty")
    try:
        value = Decimal(stripped)
    except InvalidOperation as exc:
        raise SourceError(f"line {line}: column {column!r} is not a number: {text!r}") from exc
    if not value.is_finite():
        raise SourceError(f"line {line}: column {column!r} is not finite: {text!r}")
    return value


def _integer(text: str, *, column: str, line: int) -> int:
    stripped = text.strip()
    try:
        # Binance has published integer instants in exponent form in some
        # objects. Decimal parses those exactly; int() does not.
        value = Decimal(stripped)
    except InvalidOperation as exc:
        raise SourceError(
            f"line {line}: column {column!r} is not an integer instant: {text!r}"
        ) from exc
    if value != value.to_integral_value():
        raise SourceError(f"line {line}: column {column!r} is not integral: {text!r}")
    return int(value)


def _rows_of(payload: bytes) -> list[list[str]]:
    text = payload.decode("utf-8-sig")
    return [
        row for row in csv.reader(io.StringIO(text)) if row and any(c.strip() for c in row)
    ]


def _looks_like_header(row: Sequence[str], expected: Iterable[str]) -> bool:
    """Whether the first record names columns rather than carrying values.

    Decided by NAMES, not by "the first field failed to parse as a number". A
    parse-failure heuristic silently discards a corrupt first data row as though
    it were a header, which loses a row and hides the corruption at once.
    """
    return {cell.strip().lower() for cell in row} == {name.lower() for name in expected}


def _resolve_unit(
    instants: Sequence[int], *, period: str, object_name: str
) -> tuple[str, int]:
    start, end = period_bounds(period)
    try:
        unit = resolve_epoch_unit(
            min(instants),
            max(instants),
            period_start=pd.Timestamp(start),
            period_end=pd.Timestamp(end),
        )
    except TradeAggregationError as exc:
        raise SourceError(
            f"{object_name}: {exc}. TIMESTAMP_UNIT_POLICY.fail_closed makes an archive whose "
            "unit cannot be resolved unambiguously a refusal, not a guess."
        ) from exc
    scale = {"ms": 1_000_000, "us": 1_000, "ns": 1}[unit]
    return unit, scale


def _enforce_boundary(instants_ns: Sequence[int], *, period: str, object_name: str) -> int:
    """Return how many trailing rows the boundary carve-out permits dropping.

    Refuses outright unless this is the one straddling month, which is what makes
    the carve-out a carve-out rather than a filter applied wherever it is
    convenient.
    """
    crossing = sum(1 for instant in instants_ns if instant >= RESEARCH_BOUNDARY_NS)
    if not crossing:
        return 0
    if not straddles_boundary(period):
        raise SourceError(
            f"{object_name}: {crossing} row(s) at or after the research boundary "
            f"{DATA_BOUNDARY['span_end_exclusive']}, and {period} is not the "
            "boundary-straddling month. DATA_BOUNDARY.enforcement makes such a row a "
            "REFUSAL, not a filter — truncation is granted to exactly one month and to "
            "the pre-existing committed spot snapshot, and to nothing else."
        )
    return crossing


# ---------------------------------------------------------------------------
# Klines
# ---------------------------------------------------------------------------


def read_kline_object(
    payload: bytes,
    *,
    field: str,
    object_name: str,
    period: str,
    published_checksum: str | None = None,
    raw_object: bytes | None = None,
    member_name: str | None = None,
) -> KlineTable:
    """Parse one ``klines`` or ``markPriceKlines`` CSV member into typed rows.

    ``payload`` is the extracted CSV; ``raw_object`` is the whole published object.
    BOTH digests are recorded, because ``SOURCE_FREEZE_FIELDS`` requires both and
    they attest to different things: the ZIP's digest is the one checkable against
    Binance's ``.CHECKSUM``, and the member's is the one that attests to the bytes
    the rows were actually parsed from. Neither stands in for the other — the
    archive digest is ``None`` when no archive was supplied rather than quietly
    becoming the member's.

    ``published_checksum`` is COMPARED, not merely recorded. Supplied together with
    ``raw_object`` it is checked for equality against the recomputed digest and a
    disagreement raises; supplied without it, the provenance records
    ``supplied_not_verified`` rather than claiming a check that did not happen.

    Rows are returned in INSTANT order, and two kinds of unusable row are
    WITHHELD rather than repaired — each becoming an instant this object does not
    supply, counted in the provenance so the hole stays visible:

    * a **non-positive price**, which ``POSITION_LIFECYCLE.validity_definition``
      makes the instant invalid for. It is neither clipped nor allowed through;
    * a **contradictory duplicate**, two rows at one instant that disagree, which
      is the ambiguity the same sentence calls invalid. An IDENTICAL duplicate is
      collapsed instead, because nothing is ambiguous about a row delivered twice
      and nothing is decided by collapsing it.

    Withholding rather than raising is the literal reading of the frozen sentence:
    it makes the INSTANT invalid, and what an invalid instant then does depends on
    whether the position is open — which is A2R1's business, not this module's. An
    isolated bad hour therefore cannot silently destroy a whole object, and cannot
    silently survive either.
    """
    records = _rows_of(payload)
    if not records:
        raise SourceError(f"{object_name}: no rows")
    if _looks_like_header(records[0], KLINE_COLUMNS):
        header = [cell.strip().lower() for cell in records[0]]
        records = records[1:]
        index = {name: header.index(name) for name in _KLINE_REQUIRED}
    else:
        if len(records[0]) < len(_KLINE_REQUIRED):
            raise SourceError(
                f"{object_name}: headerless row has {len(records[0])} columns, fewer than the "
                f"{len(_KLINE_REQUIRED)} the published kline layout requires. An unrecognised "
                "layout is a refusal; a mapping is never inferred."
            )
        index = {name: position for position, name in enumerate(KLINE_COLUMNS[:5])}
    if not records:
        raise SourceError(f"{object_name}: header present but no data rows")

    raw_instants: list[int] = []
    parsed: list[tuple[int, Decimal, Decimal, Decimal, Decimal]] = []
    withheld_non_positive = 0
    for line, record in enumerate(records, start=1):
        needed = max(index.values()) + 1
        if len(record) < needed:
            raise SourceError(
                f"{object_name} line {line}: {len(record)} columns, needs at least {needed}"
            )
        instant = _integer(record[index["open_time"]], column="open_time", line=line)
        values = tuple(
            _decimal(record[index[name]], column=name, line=line)
            for name in ("open", "high", "low", "close")
        )
        # The unit is resolved from EVERY instant the object carries, including
        # the ones whose prices are unusable: a withheld row is still evidence of
        # what the object's timestamps mean.
        raw_instants.append(instant)
        if any(value <= 0 for value in values):
            withheld_non_positive += 1
            continue
        parsed.append((instant, *values))

    unit, scale = _resolve_unit(raw_instants, period=period, object_name=object_name)
    scaled = sorted(
        ((instant * scale, *rest) for instant, *rest in parsed), key=lambda row: row[0]
    )

    deduped, withheld_ambiguous = _collapse_duplicate_instants(scaled)
    dropped = _enforce_boundary(
        [row[0] for row in deduped], period=period, object_name=object_name
    )
    kept = [row for row in deduped if row[0] < RESEARCH_BOUNDARY_NS]
    rows = tuple(
        KlineRow(instant_ns=instant, open=o, high=h, low=low, close=c)
        for instant, o, h, low, c in kept
    )
    # The verification happens BEFORE the provenance is built, so a mismatch
    # raises instead of being recorded. Both digests are computed independently
    # and neither substitutes for the other.
    state = verify_published_checksum(raw_object, published_checksum)
    provenance = ObjectProvenance(
        field=field,
        object_name=object_name,
        period=period,
        byte_size=len(raw_object) if raw_object is not None else None,
        sha256=(hashlib.sha256(raw_object).hexdigest() if raw_object is not None else None),
        member_name=member_name,
        member_sha256=hashlib.sha256(payload).hexdigest(),
        member_byte_size=len(payload),
        resolved_epoch_unit=unit,
        rows_read=len(rows),
        rows_dropped_at_boundary=dropped,
        first_instant_ns=rows[0].instant_ns if rows else None,
        last_instant_ns=rows[-1].instant_ns if rows else None,
        ambiguous_instants=withheld_ambiguous,
        non_positive_instants=withheld_non_positive,
        published_checksum=published_checksum,
        checksum_state=state,
    )
    return KlineTable(provenance=provenance, rows=rows)


def _collapse_duplicate_instants(
    scaled: Sequence[tuple],
) -> tuple[list[tuple], int]:
    """Collapse identical repeats; WITHHOLD instants whose rows disagree.

    Two rows at one instant are only ambiguous when they differ. An identical
    redelivery decides nothing and is collapsed; a contradictory pair is exactly
    what ``POSITION_LIFECYCLE.validity_definition`` calls invalid, so the instant
    is withheld entirely — BOTH copies, not one of them — because keeping either
    would let the archive's row order pick which candle the result came from.
    """
    by_instant: dict[int, set[tuple]] = {}
    for row in scaled:
        by_instant.setdefault(row[0], set()).add(tuple(row[1:]))
    kept: list[tuple] = []
    ambiguous = 0
    for instant in sorted(by_instant):
        variants = by_instant[instant]
        if len(variants) > 1:
            ambiguous += 1
            continue
        kept.append((instant, *next(iter(variants))))
    return kept, ambiguous


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------


def read_funding_object(
    payload: bytes,
    *,
    object_name: str,
    period: str,
    published_checksum: str | None = None,
    raw_object: bytes | None = None,
    member_name: str | None = None,
) -> FundingTable:
    """Parse one ``fundingRate`` CSV member under the FROZEN column policy.

    The mapping is not inferred: ``FUNDING_CSV_COLUMN_POLICY`` lists the allowed
    header maps and the single headerless positional layout, and
    ``on_unrecognised_layout`` makes anything else a refusal. P13 reuses that
    policy verbatim through ``FUNDING_COLUMN_POLICY_SOURCE`` rather than restating
    it, so a layout rule cannot drift between two checkpoints reading one archive.

    Two rows at one settlement instant are NOT resolved here. An identical
    redelivery is harmless and a contradictory pair is invalid, and the accounting
    engine already distinguishes them — ``evaluate_block`` deduplicates the first
    and raises on the second. Deciding it twice, in two places, is how the two
    decisions come to disagree.
    """
    records = _rows_of(payload)
    if not records:
        raise SourceError(f"{object_name}: no rows")

    instant_at, rate_at, interval_at, layout = _funding_layout(records[0], object_name)
    if layout != "headerless positional":
        records = records[1:]
    if not records:
        raise SourceError(f"{object_name}: header present but no data rows")

    raw_instants: list[int] = []
    parsed: list[tuple[int, Decimal, Decimal | None]] = []
    for line, record in enumerate(records, start=1):
        needed = (
            max(
                position
                for position in (instant_at, rate_at, interval_at)
                if position is not None
            )
            + 1
        )
        if len(record) < needed:
            raise SourceError(
                f"{object_name} line {line}: {len(record)} columns, needs at least {needed}"
            )
        instant = _integer(record[instant_at], column="settlement_instant", line=line)
        rate = _decimal(record[rate_at], column="realised_funding_rate", line=line)
        interval = (
            _decimal(record[interval_at], column="funding_interval_hours", line=line)
            if interval_at is not None
            else None
        )
        raw_instants.append(instant)
        parsed.append((instant, rate, interval))

    unit, scale = _resolve_unit(raw_instants, period=period, object_name=object_name)
    scaled = sorted(
        ((instant * scale, rate, interval) for instant, rate, interval in parsed),
        key=lambda row: row[0],
    )
    dropped = _enforce_boundary(
        [row[0] for row in scaled], period=period, object_name=object_name
    )
    kept = [row for row in scaled if row[0] < RESEARCH_BOUNDARY_NS]
    rows = tuple(
        FundingRow(instant_ns=instant, rate=rate, interval_hours=interval)
        for instant, rate, interval in kept
    )
    # COUNTED, NOT WITHHELD, and the asymmetry with the kline path is the frozen
    # design's rather than this module's. ``FUNDING_SEMANTICS.application`` says a
    # redelivered or duplicated funding row "changes nothing", and
    # ``evaluate_block`` already implements exactly that: it deduplicates an
    # identical repeat and RAISES on a contradictory pair. Withholding either here
    # would decide, in a second place, something the accounting engine already
    # decides — and two places that decide one thing are two places that can come
    # to disagree.
    repeated = len(rows) - len({row.instant_ns for row in rows})
    state = verify_published_checksum(raw_object, published_checksum)
    provenance = ObjectProvenance(
        field="funding_settlement",
        object_name=object_name,
        period=period,
        byte_size=len(raw_object) if raw_object is not None else None,
        sha256=(hashlib.sha256(raw_object).hexdigest() if raw_object is not None else None),
        member_name=member_name,
        member_sha256=hashlib.sha256(payload).hexdigest(),
        member_byte_size=len(payload),
        resolved_epoch_unit=unit,
        rows_read=len(rows),
        rows_dropped_at_boundary=dropped,
        first_instant_ns=rows[0].instant_ns if rows else None,
        last_instant_ns=rows[-1].instant_ns if rows else None,
        repeated_instants=repeated,
        published_checksum=published_checksum,
        checksum_state=state,
    )
    return FundingTable(provenance=provenance, rows=rows)


def _funding_layout(
    first: Sequence[str], object_name: str
) -> tuple[int, int, int | None, str]:
    """Match the first record against the FROZEN allowed layouts, or refuse."""
    names = [cell.strip() for cell in first]
    lowered = {name.lower() for name in names}
    for allowed in FUNDING_CSV_COLUMN_POLICY["allowed_header_maps"]:
        if lowered == {column.lower() for column in allowed["columns"]}:
            position = {name.lower(): index for index, name in enumerate(names)}
            interval = position.get("funding_interval_hours")
            return (
                position[allowed["settlement_instant"].lower()],
                position[allowed["realised_funding_rate"].lower()],
                interval,
                allowed["layout"],
            )
    positional = FUNDING_CSV_COLUMN_POLICY["headerless_positional_layout"]
    if len(names) == positional["columns"]:
        # The policy admits this layout "only when the first column parses as an
        # epoch instant inside the archive's own calendar period under exactly one
        # supported unit". That check is _resolve_unit's, applied to every row of
        # the object rather than to this one, which is strictly stronger.
        return (
            positional["settlement_instant"],
            positional["realised_funding_rate"],
            None,
            ("headerless positional"),
        )
    raise SourceError(
        f"{object_name}: unrecognised fundingRate layout {names}. "
        f"{FUNDING_CSV_COLUMN_POLICY['on_unrecognised_layout']}"
    )
