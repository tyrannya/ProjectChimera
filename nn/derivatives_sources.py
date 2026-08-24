"""The shape of P4's derivatives source, and the rules for reading it.

``docs/p4_preregistration.md`` §3 fixes *which* archives P4 reads and *how* their
columns may be interpreted; :mod:`nn.p4_preregistration` is the machine-readable
copy of that. This module is the schema half of the acquisition: the hourly table
the exporter writes, the archive-path arithmetic, the funding column allow-list
resolved against a real header, and the bounded parse of an open-interest
snapshot instant.

**Nothing here touches the network.** ``--plan`` is required to be networkless,
so everything a plan needs — which archives, which periods, which paths — is
computed here from the preregistration and a calendar, and
:mod:`tools.export_derivatives_snapshot` adds HTTP on top of it. That split is
also what makes the refusals testable: a funding header this repository must
refuse can be handed to :func:`resolve_funding_columns` directly, which is not
true of anything that has to download a ZIP first.

**Why the exported table looks like this.** The eight ``derivatives_v1`` columns
are not stored here. Storing them would put the feature windows — the numbers
§5 fixes and forbids re-choosing — inside the *source*, where a re-export could
move them without moving the feature-spec hash. What is stored is the hourly
point-in-time observation each feature is a function of: the funding settlement
visible at the hour, the last open-interest snapshot at or before it, the
perpetual close, and, for each of the three, how stale the observation is. The
engine in :mod:`nn.derivatives` turns those into features, and the staleness
bounds of §3.4 are applied twice — once here, so the source never carries a
carry-forward it is not allowed to make, and once in the engine, which is where
the sample universe is decided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from nn import p4_preregistration
from nn.p4_preregistration import DATA_SOURCES, FUNDING_CSV_COLUMN_POLICY, MAX_STALENESS_HOURS
from nn.trade_aggregates import resolve_epoch_unit, scale_to_nanoseconds

#: Names the meaning of the exported hourly table. A file under another schema
#: is a different document and is refused rather than read with defaults.
DERIVATIVES_SOURCE_SCHEMA = "chimera.derivatives-hourly/1"

HOUR_NS = 3_600_000_000_000

BASE_URL = "https://data.binance.vision"
PROVIDER = "binance-public-data"
VENUE = "binance"
MARKET_TYPE = "um"
SYMBOL = "BTCUSDT"
PAIR = "BTC/USDT"
CHECKSUM_SUFFIX = ".CHECKSUM"

#: The three fields acquired from the public archive. Spot is the fourth source
#: of §3 and is not acquired at all: it is the committed candle history.
FUNDING = "funding_rate"
OPEN_INTEREST = "open_interest"
PERPETUAL = "perpetual_price"
ACQUIRED_FIELDS: tuple[str, ...] = (FUNDING, OPEN_INTEREST, PERPETUAL)

FUNDING_TEMPLATE = (
    "{base}/data/futures/um/monthly/fundingRate/{symbol}/"
    "{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
)
#: **Daily**, and deliberately so. ``docs/p4_preregistration.md`` §3.0a records
#: that the first version of the preregistration named a monthly metrics path
#: Binance does not publish, and corrected it before any probe. There is no
#: monthly metrics archive to fall back to and none is constructed here.
METRICS_TEMPLATE = (
    "{base}/data/futures/um/daily/metrics/{symbol}/"
    "{symbol}-metrics-{year:04d}-{month:02d}-{day:02d}.zip"
)
KLINE_TEMPLATE = (
    "{base}/data/futures/um/monthly/klines/{symbol}/1h/"
    "{symbol}-1h-{year:04d}-{month:02d}.zip"
)

#: The earliest UTC day §3.0a intends to request from the metrics archive. An
#: intent, not a measurement: the probe establishes the real first available day
#: and a later one narrows the universe rather than being worked around.
EARLIEST_METRICS_DAY = "2020-09-01"

# There is deliberately no `EARLIEST_FUNDING_MONTH` or `EARLIEST_KLINE_MONTH`
# beside the line above. Each archive's first protocol month lives in its own
# hashed preregistration object — `FUNDING_ARCHIVE_INCEPTION_POLICY` (amendment
# A2) and `PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY` (amendment A3) — and a
# constant here would be a second copy that a later edit could leave disagreeing
# with the design while the suite stayed green. `funding_inception()` and
# `perpetual_inception()` read their policy at call time instead, so moving a rule
# moves its plan. They are two rules and not one: the months agree today because
# two archives were measured to begin in the same month, which is an observation
# about two sources rather than a rule about the venue.

#: USD-M kline archives carry twelve columns. Checked against the width of the
#: first row: the spot archive's layout differs, and reading one as the other
#: shifts every field.
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

#: The metrics columns §3.0a names, and nothing else is read from that archive.
#: A file without all three has no open interest in it under this schema.
METRICS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "create_time",
    "sum_open_interest",
    "sum_open_interest_value",
)

#: The hourly point-in-time table the exporter writes, with the canonical kind
#: :mod:`nn.data_fingerprint` hashes each column under.
#:
#: Availability is a flag and staleness is an integer age rather than a nullable
#: instant, because a semantic fingerprint has no representation for "missing":
#: ``t8`` refuses a NaT and ``f8`` folds every NaN together. A row that has no
#: observation says so in ``*_available`` and carries a declared zero, and the
#: verifier checks that pairing rather than trusting it.
DERIVATIVES_COLUMN_KINDS: tuple[tuple[str, str], ...] = (
    ("date", "t8"),
    ("funding_settled", "i8"),
    ("funding_settled_rate", "f8"),
    ("funding_visible_count", "i8"),
    ("funding_available", "i8"),
    ("funding_last_rate", "f8"),
    ("funding_age_ns", "i8"),
    ("oi_available", "i8"),
    ("oi_contracts", "f8"),
    ("oi_notional", "f8"),
    ("oi_age_ns", "i8"),
    ("perp_available", "i8"),
    ("perp_close", "f8"),
    ("perp_age_ns", "i8"),
)

DERIVATIVES_COLUMNS: tuple[str, ...] = tuple(name for name, _ in DERIVATIVES_COLUMN_KINDS)

#: What an unavailable observation carries, so that "no observation" is one
#: value rather than whatever the writer happened to leave behind.
UNAVAILABLE_AGE_NS = -1

#: Rows parsed from one archive member at a time. A daily metrics archive holds
#: 288 rows and a monthly funding archive about 93, so this only ever bounds the
#: kline archives — but the exporter must not depend on which of its three
#: sources happens to be small.
CSV_CHUNK_ROWS = 500_000


class DerivativesSourceError(ValueError):
    """A derivatives archive cannot be read as the source P4 preregistered."""


def staleness_bound_ns(field: str) -> int:
    """The §3.4 carry-forward bound for one field, in nanoseconds."""
    try:
        hours = MAX_STALENESS_HOURS[field]
    except KeyError as exc:
        raise DerivativesSourceError(
            f"{field!r} has no preregistered staleness bound; the fields with one are "
            f"{sorted(MAX_STALENESS_HOURS)}"
        ) from exc
    return int(hours) * HOUR_NS


def preregistered_source(field: str) -> Mapping[str, Any]:
    """The ``DATA_SOURCES`` entry for one field, or a named refusal."""
    for source in DATA_SOURCES:
        if source["field"] == field:
            return source
    raise DerivativesSourceError(
        f"{field!r} is not a preregistered P4 data source; the sources are "
        f"{[s['field'] for s in DATA_SOURCES]}"
    )


# --------------------------------------------------------------------------- #
# archives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Archive:
    """One period of one public archive, and the calendar window it is named for.

    ``period_start``/``period_end`` are not decoration: every timestamp the
    archive yields must fall inside them, which is what lets an epoch unit be
    *resolved* rather than guessed (:func:`nn.trade_aggregates.resolve_epoch_unit`)
    and what turns a mislabelled archive into a refusal.
    """

    field: str
    name: str
    url: str
    checksum_url: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "name": self.name,
            "url": self.url,
            "kind": self.kind,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


def _archive(
    field: str, url: str, start: pd.Timestamp, end: pd.Timestamp, kind: str
) -> Archive:
    return Archive(
        field=field,
        name=url.rsplit("/", 1)[-1],
        url=url,
        checksum_url=url + CHECKSUM_SUFFIX,
        period_start=start,
        period_end=end,
        kind=kind,
    )


def funding_archive(year: int, month: int) -> Archive:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    return _archive(
        FUNDING,
        FUNDING_TEMPLATE.format(base=BASE_URL, symbol=SYMBOL, year=year, month=month),
        start,
        start + pd.offsets.MonthBegin(1),
        "monthly",
    )


def kline_archive(year: int, month: int) -> Archive:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    return _archive(
        PERPETUAL,
        KLINE_TEMPLATE.format(base=BASE_URL, symbol=SYMBOL, year=year, month=month),
        start,
        start + pd.offsets.MonthBegin(1),
        "monthly",
    )


def metrics_archive(day: pd.Timestamp) -> Archive:
    day = pd.Timestamp(day).tz_convert("UTC").normalize()
    return _archive(
        OPEN_INTEREST,
        METRICS_TEMPLATE.format(
            base=BASE_URL, symbol=SYMBOL, year=day.year, month=day.month, day=day.day
        ),
        day,
        day + pd.Timedelta(days=1),
        "daily",
    )


def _utc_day(value: Any) -> pd.Timestamp:
    """One UTC midnight, however the caller spelled it.

    ``pd.Timestamp(x, tz="UTC")`` refuses an already-aware input, and both forms
    reach here: a probe hands back an aware instant and the preregistration
    hands back a date string.
    """
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.normalize()


def _months(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[pd.Timestamp]:
    cursor = pd.Timestamp(start).tz_convert("UTC").normalize().replace(day=1)
    end = pd.Timestamp(end).tz_convert("UTC")
    while cursor < end:
        yield cursor
        cursor = cursor + pd.offsets.MonthBegin(1)


def _inception_month(policy: Mapping[str, Any]) -> pd.Timestamp:
    """The month an inception policy names, as an instant. Shared arithmetic only.

    Two amendments state the same *kind* of fact about two different archives —
    A2 about the monthly ``fundingRate`` archive, A3 about the monthly 1h kline
    archive — so the YYYY-MM parse and the month-begin conversion live once. The
    *rules* are not shared: each caller passes its own hashed policy object, and
    nothing here reads a default.
    """
    month = str(policy["first_protocol_month"])
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise DerivativesSourceError(
            f"the {policy['scope']['field']} archive inception policy (amendment "
            f"{policy['amendment']}) names {month!r}, which is not a YYYY-MM month. "
            "The plan cannot be built from it."
        )
    return pd.Timestamp(f"{month}-01", tz="UTC")


def _source_boundary(policy: Mapping[str, Any], start: pd.Timestamp) -> dict[str, Any]:
    """What an inception policy did to a requested start, as provenance.

    The clamp must never make the requested month *disappear*: a plan that simply
    began later would read as a design that always started there. Both months are
    recorded, together with how many the policy removed, which amendment removed
    them and why, so a reader can see that 2019-12 was asked for and that the
    source is what refused it.
    """
    requested = pd.Timestamp(start).tz_convert("UTC").normalize().replace(day=1)
    inception = _inception_month(policy)
    effective = max(requested, inception)
    months = 0
    cursor = requested
    while cursor < effective:
        months += 1
        cursor = cursor + pd.offsets.MonthBegin(1)
    return {
        "amendment": policy["amendment"],
        "field": policy["scope"]["field"],
        "generic_requested_from": requested.strftime("%Y-%m"),
        "source_inception_month": policy["first_protocol_month"],
        "effective_from": effective.strftime("%Y-%m"),
        "months_clamped": months,
        "rule": policy["acquisition_start_rule"],
        "reason": (
            policy["pre_inception_behaviour"]
            if months
            else "the requested start is at or after the source inception; nothing is clamped"
        ),
        "not_an_internal_gap": bool(months),
        "no_substitution": policy["no_substitution"],
    }


def _clamped_monthly_start(
    policy: Mapping[str, Any], start: pd.Timestamp, end: pd.Timestamp
) -> pd.Timestamp:
    """``max(requested start, inception)``, or a refusal if that names no month.

    A month before the archive's first protocol month is outside the source
    rather than a hole in it, so it is not requested — and never becomes a
    missing month, because a missing month stops the acquisition and an
    unpublished pre-inception month is not one. A window lying wholly before the
    inception names nothing the source publishes, and inventing history for it is
    exactly what the policy forbids, so it is refused rather than silently empty.
    """
    inception = _inception_month(policy)
    first = max(start, inception)
    if first >= end:
        raise DerivativesSourceError(
            f"the {policy['scope']['field']} window [{start.isoformat()}, "
            f"{end.isoformat()}) ends at or before the archive's first protocol month "
            f"{inception.date().isoformat()}, so it names no published month. "
            f"Amendment {policy['amendment']} clamps the start to the source "
            "inception; it does not invent history before it."
        )
    return first


def funding_inception() -> pd.Timestamp:
    """The first month of the funding archive this protocol reads, as an instant.

    Amendment A2, read from
    :data:`nn.p4_preregistration.FUNDING_ARCHIVE_INCEPTION_POLICY` rather than
    written here.
    """
    return _inception_month(p4_preregistration.FUNDING_ARCHIVE_INCEPTION_POLICY)


def funding_source_boundary(start: pd.Timestamp) -> dict[str, Any]:
    """What A2's inception did to a requested funding start, as provenance."""
    return _source_boundary(p4_preregistration.FUNDING_ARCHIVE_INCEPTION_POLICY, start)


def perpetual_inception() -> pd.Timestamp:
    """The first month of the perpetual kline archive this protocol reads.

    Amendment A3, read from
    :data:`nn.p4_preregistration.PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY` rather
    than written here. A separate measurement of a separate archive from A2's: it
    names the same month today, and an edit to either moves only its own plan.
    """
    return _inception_month(p4_preregistration.PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY)


def perpetual_source_boundary(start: pd.Timestamp) -> dict[str, Any]:
    """What A3's inception did to a requested perpetual-price start, as provenance."""
    return _source_boundary(p4_preregistration.PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY, start)


def plan_archives(
    field: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    oi_first_day: pd.Timestamp | None = None,
) -> list[Archive]:
    """Every archive of ``field`` covering ``[start, end)``. No network.

    Monthly for funding and klines, because Binance publishes them that way and
    the whole history is a few dozen requests. **Daily for open interest**, one
    UTC day per archive, because §3.0a establishes that no monthly metrics
    archive exists: a plan that named one would be a plan to fetch nothing.

    ``oi_first_day`` clips the open-interest plan at the earliest day the source
    is intended — or, after a probe, known — to publish. A day before it is not
    requested, and the rows it would have covered leave the sample universe like
    any other absent day rather than being sought from somewhere else.
    """
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    if end <= start:
        raise DerivativesSourceError(
            f"the acquisition window [{start.isoformat()}, {end.isoformat()}) is empty"
        )
    if field == FUNDING:
        # Amendment A2, which governs the monthly fundingRate archive and only it.
        first = _clamped_monthly_start(
            p4_preregistration.FUNDING_ARCHIVE_INCEPTION_POLICY, start, end
        )
        return [funding_archive(m.year, m.month) for m in _months(first, end)]
    if field == PERPETUAL:
        # Amendment A3, which governs the monthly 1h kline archive and only it.
        # Its own object, measured on its own source: the two amendments share the
        # clamp arithmetic above and nothing about what either archive publishes.
        first = _clamped_monthly_start(
            p4_preregistration.PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY, start, end
        )
        return [kline_archive(m.year, m.month) for m in _months(first, end)]
    if field == OPEN_INTEREST:
        floor = _utc_day(oi_first_day if oi_first_day is not None else EARLIEST_METRICS_DAY)
        first = max(start.normalize(), floor)
        days = pd.date_range(first, end - pd.Timedelta(nanoseconds=1), freq="D", tz="UTC")
        return [metrics_archive(day) for day in days]
    raise DerivativesSourceError(
        f"{field!r} is not an acquired P4 field; the acquired fields are "
        f"{list(ACQUIRED_FIELDS)}"
    )


def canonical_member(names: Sequence[str], archive_name: str) -> str:
    """The one CSV inside an archive, or a refusal that will not guess.

    Two rules, and no third: exactly one non-directory member is the member, and
    otherwise the member must be the archive's own name with ``.csv`` for
    ``.zip`` **at the root**. Anything else — two candidates, a nested copy, a
    member whose name only ends the right way — is ambiguous, and an acquisition
    that picked one would be reporting a source nobody can reproduce.
    """
    files = [name for name in names if not name.endswith("/")]
    if not files:
        raise DerivativesSourceError(f"{archive_name} holds no files")
    if len(files) == 1:
        only = files[0]
        if "/" in only:
            raise DerivativesSourceError(
                f"{archive_name} holds one member {only!r}, which is nested rather than "
                "at the archive root. A nested member is not the canonical layout and is "
                "refused rather than read as though it were."
            )
        return only
    expected = archive_name.removesuffix(".zip") + ".csv"
    if expected in files:
        return expected
    raise DerivativesSourceError(
        f"{archive_name} holds {len(files)} members ({sorted(files)[:4]}); expected the "
        f"canonical root member {expected!r} and will not guess which one is the data"
    )


# --------------------------------------------------------------------------- #
# funding: the column allow-list of §3.0b
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FundingLayout:
    """How one funding archive's columns map onto the two canonical fields."""

    layout: str
    settlement_instant: str | int
    realised_funding_rate: str | int
    headerless: bool
    columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "settlement_instant": self.settlement_instant,
            "realised_funding_rate": self.realised_funding_rate,
            "headerless": self.headerless,
            "observed_columns": list(self.columns),
        }


def _looks_like_header(fields: Sequence[str]) -> bool:
    """True when the first row names columns rather than carrying a settlement.

    Decided on the first field alone, as :mod:`tools.export_trade_snapshot` does:
    a header read as data makes the instant unparseable and the failure would
    name the wrong cause.
    """
    head = fields[0].strip().strip('"')
    if not head:
        return True
    return not re.fullmatch(r"-?\d+", head)


def resolve_funding_columns(first_line: str, archive: Archive) -> FundingLayout:
    """Match a funding archive's first line against §3.0b's allow-list, or refuse.

    The allow-list is :data:`nn.p4_preregistration.FUNDING_CSV_COLUMN_POLICY`,
    read rather than restated. Three things it permits and one it does not:

    * a recognised layout may carry columns the policy does not name — they are
      ignored, so a source that adds a field does not break the acquisition;
    * two entries may both match (``calc_time``/``last_funding_rate`` is a subset
      of the three-column layout) and that is fine **only while they agree** on
      which column is the instant and which is the rate. A disagreement is a
      refusal, because then the mapping would be this function's choice;
    * a headerless file is accepted in the single unambiguous two-column shape,
      and only when its first column parses as an epoch instant inside the
      archive's own calendar period under exactly one supported unit.

    What it does not permit is inference. An unrecognised column-name set is a
    refusal — not a fallback to positional order, not a guess from the widths,
    and not a switch to the REST endpoint. Extending the allow-list moves
    :func:`nn.p4_preregistration.preregistration_hash`, which is the point.
    """
    fields = [field.strip().strip('"') for field in first_line.rstrip("\r\n").split(",")]
    if not first_line.strip():
        raise DerivativesSourceError(f"{archive.name}: the funding member is empty")

    if not _looks_like_header(fields):
        shape = FUNDING_CSV_COLUMN_POLICY["headerless_positional_layout"]
        if len(fields) != int(shape["columns"]):
            raise DerivativesSourceError(
                f"{archive.name}: a headerless funding archive is accepted only in the "
                f"single unambiguous {shape['columns']}-column shape, and this one has "
                f"{len(fields)}. {FUNDING_CSV_COLUMN_POLICY['on_unrecognised_layout']}"
            )
        # The archive's own calendar period decides the unit, exactly as it does
        # for a trade archive. A first column that fits no unit — or two — is a
        # refusal, so "headerless" never becomes "positional whatever the file".
        raw = int(fields[int(shape["settlement_instant"])])
        resolve_epoch_unit(
            raw, raw, period_start=archive.period_start, period_end=archive.period_end
        )
        return FundingLayout(
            layout="headerless-positional",
            settlement_instant=int(shape["settlement_instant"]),
            realised_funding_rate=int(shape["realised_funding_rate"]),
            headerless=True,
            columns=tuple(fields),
        )

    observed = set(fields)
    matches = [
        allowed
        for allowed in FUNDING_CSV_COLUMN_POLICY["allowed_header_maps"]
        if set(allowed["columns"]) <= observed
    ]
    if not matches:
        allowed = [
            entry["columns"] for entry in FUNDING_CSV_COLUMN_POLICY["allowed_header_maps"]
        ]
        raise DerivativesSourceError(
            f"{archive.name}: funding header {sorted(observed)} matches none of the "
            f"preregistered layouts {allowed}. "
            f"{FUNDING_CSV_COLUMN_POLICY['on_unrecognised_layout']}"
        )
    instants = {entry["settlement_instant"] for entry in matches}
    rates = {entry["realised_funding_rate"] for entry in matches}
    if len(instants) != 1 or len(rates) != 1:
        raise DerivativesSourceError(
            f"{archive.name}: funding header {sorted(observed)} matches "
            f"{[entry['layout'] for entry in matches]}, which disagree about which "
            f"column is the settlement instant ({sorted(instants)}) or the rate "
            f"({sorted(rates)}). Choosing between them would be this reader deciding "
            "the mapping after seeing the file."
        )
    # The most specific matching entry names the layout, so a three-column file
    # is recorded as the three-column layout rather than as its own subset.
    best = max(matches, key=lambda entry: len(entry["columns"]))
    return FundingLayout(
        layout=best["layout"],
        settlement_instant=best["settlement_instant"],
        realised_funding_rate=best["realised_funding_rate"],
        headerless=False,
        columns=tuple(fields),
    )


# --------------------------------------------------------------------------- #
# instants
# --------------------------------------------------------------------------- #
_DATETIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$"
)


def parse_instants(values: pd.Series, archive: Archive, *, what: str) -> np.ndarray:
    """Read one archive column as int64 UTC nanoseconds, or refuse.

    Two representations are accepted and each is decided by the archive's own
    calendar period rather than by magnitude: an integer epoch, whose unit is
    resolved by :func:`nn.trade_aggregates.resolve_epoch_unit`, and a UTC
    wall-clock string of the ``YYYY-MM-DD HH:MM:SS`` family that Binance's
    metrics archives publish. A column that is neither is a refusal.

    The period is the arbiter for both. A parsed instant outside the archive's
    own window means the reading is wrong — a mislabelled unit, an implied local
    timezone — and no reading of it is trustworthy, so it stops rather than
    contributing rows a later reader could not reproduce.
    """
    text = values.astype(str).str.strip()
    if text.empty:
        raise DerivativesSourceError(f"{archive.name}: {what} has no rows")

    sample = text.iloc[0]
    if re.fullmatch(r"-?\d+", sample):
        raw = text.to_numpy(dtype=np.int64)
        unit = resolve_epoch_unit(
            int(raw.min()),
            int(raw.max()),
            period_start=archive.period_start,
            period_end=archive.period_end,
        )
        stamps = scale_to_nanoseconds(raw, unit)
    elif _DATETIME_PATTERN.match(sample):
        parsed = pd.to_datetime(text, utc=True, errors="coerce", format="mixed")
        if parsed.isna().any():
            bad = sorted(set(text[parsed.isna()]))[:4]
            raise DerivativesSourceError(
                f"{archive.name}: {what} holds unparseable instant(s) {bad}"
            )
        stamps = parsed.dt.tz_convert("UTC").to_numpy(dtype="datetime64[ns]").astype(np.int64)
    else:
        raise DerivativesSourceError(
            f"{archive.name}: {what} begins {sample!r}, which is neither an integer epoch "
            "nor a UTC wall-clock instant. Refusing to read it under a guessed "
            "representation."
        )

    lo = int(archive.period_start.value)
    hi = int(archive.period_end.value)
    outside = int(((stamps < lo) | (stamps >= hi)).sum())
    if outside:
        raise DerivativesSourceError(
            f"{archive.name}: {outside} {what} value(s) fall outside the archive's own "
            f"period [{archive.period_start.isoformat()}, {archive.period_end.isoformat()}). "
            "The representation is not what it was read as, and no reading of the archive "
            "is trustworthy."
        )
    return stamps


def check_strictly_increasing(stamps: np.ndarray, archive: Archive, *, what: str) -> None:
    """§3.4's duplicate rule: two rows claiming one instant is a rejection.

    Not a de-duplication. Two rows claiming one instant means the reader cannot
    tell which one is the observation, and picking either is a decision about
    the data made by the code that reads it.
    """
    if len(stamps) == 0:
        return
    step = np.diff(stamps)
    if (step == 0).any():
        first = int(np.flatnonzero(step == 0)[0])
        instant = pd.Timestamp(int(stamps[first]), unit="ns", tz="UTC")
        raise DerivativesSourceError(
            f"{archive.name}: {what} carries two rows at {instant.isoformat()}. A "
            "duplicate instant is a rejection of the acquisition, not something to "
            "de-duplicate: the reader cannot tell which row is the observation."
        )
    if (step < 0).any():
        first = int(np.flatnonzero(step < 0)[0])
        raise DerivativesSourceError(
            f"{archive.name}: {what} goes backwards at row {first + 1}. The archive is "
            "not in instant order and reading it as though it were would put an "
            "observation on the wrong hour."
        )


# --------------------------------------------------------------------------- #
# the metrics archive's exact duplicate rows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricsDuplicateNormalisation:
    """What collapsing exact duplicate rows did to one daily metrics archive.

    Four counts, kept together because any one of them alone is a claim nobody
    can check: rows retained without rows read hides what was removed, and rows
    removed without the instants they came from hides whether one instant was
    duplicated fifty times or fifty instants were duplicated once.
    """

    rows_read: int
    observations_retained: int
    exact_duplicate_rows_collapsed: int
    duplicate_instants: int

    def to_dict(self) -> dict[str, int]:
        return {
            "rows_read": self.rows_read,
            "observations_retained": self.observations_retained,
            "exact_duplicate_rows_collapsed": self.exact_duplicate_rows_collapsed,
            "duplicate_instants": self.duplicate_instants,
        }


def collapse_exact_duplicate_metrics_rows(
    rows: Sequence[Sequence[str]],
    instants: np.ndarray,
    archive: Archive,
    *,
    columns: Sequence[str],
) -> tuple[np.ndarray, MetricsDuplicateNormalisation]:
    """Collapse rows that repeat one instant *identically*, and refuse the rest.

    **The rule is not written here.** It is
    :data:`nn.p4_preregistration.OPEN_INTEREST_DUPLICATE_POLICY` — amendment A1,
    inside the hashed preregistration — and this function is its implementation,
    not a second statement of it. The refusal below quotes the policy's own
    ``on_conflict`` wording rather than paraphrasing it, so an edit to the rule
    changes what the code says as well as what the design says.

    **The rule, and its two halves.** Rows sharing an instant are collapsed to
    one logical observation *only* when every source field in them is identical.
    When any field differs, this raises: the reader still cannot tell which row
    is the observation, so it does not take the first, does not take the last,
    does not average, and does not infer. That half is
    :func:`check_strictly_increasing`'s policy, unchanged and merely reached by a
    different door.

    **"Every source field" means the whole row.** ``rows`` carries every column
    the CSV published, verbatim, including the columns the P4 feature engine
    never reads. Two rows agreeing on ``create_time``, ``sum_open_interest`` and
    ``sum_open_interest_value`` but differing anywhere else are *not* exact
    duplicates and are refused, because a source that disagrees with itself about
    a field this design ignores is a source disagreeing with itself.

    ``instants`` must already be ascending, and ``rows`` in the same order.
    Returns the keep-mask over them and the counts to record in provenance. An
    instant repeated more than twice with every row equivalent collapses to one,
    and every row removed is counted.
    """
    count = len(instants)
    keep = np.ones(count, dtype=bool)
    if count == 0:
        return keep, MetricsDuplicateNormalisation(0, 0, 0, 0)

    # Group boundaries rather than a key-based de-duplication: the keys of a
    # generic drop_duplicates would have to be spelled out anyway, and it would
    # silently accept the conflicting case this must refuse.
    boundary = np.ones(count, dtype=bool)
    boundary[1:] = instants[1:] != instants[:-1]
    starts = np.flatnonzero(boundary).tolist()
    ends = starts[1:] + [count]

    collapsed = 0
    duplicate_instants = 0
    for start, end in zip(starts, ends):
        if end - start == 1:
            continue
        first = rows[start]
        for index in range(start + 1, end):
            other = rows[index]
            if list(other) == list(first):
                continue
            instant = pd.Timestamp(int(instants[start]), unit="ns", tz="UTC")
            differing = [
                f"{name}={left!r} vs {right!r}"
                for name, left, right in zip(columns, first, other)
                if left != right
            ]
            policy = p4_preregistration.OPEN_INTEREST_DUPLICATE_POLICY
            raise DerivativesSourceError(
                f"{archive.name}: two rows at {instant.isoformat()} disagree "
                f"({'; '.join(differing[:4])}). {policy['acceptance']}. "
                f"These are conflicting observations, so: {policy['on_conflict']}."
            )
        keep[start + 1 : end] = False
        collapsed += end - start - 1
        duplicate_instants += 1

    return keep, MetricsDuplicateNormalisation(
        rows_read=count,
        observations_retained=count - collapsed,
        exact_duplicate_rows_collapsed=collapsed,
        duplicate_instants=duplicate_instants,
    )


# --------------------------------------------------------------------------- #
# the metrics archive's zero-valued open-interest rows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricsObservationValidity:
    """Which of one archive's logical rows are open-interest observations.

    Eight counts, and an identity between them. ``logical_observations`` is what
    A1 left behind — the rows the archive published, after identical repeats of
    one instant were collapsed — and every one of them is either a valid positive
    observation or an invalid zero one, with the invalid ones partitioned by
    which consumed field was zero. ``negative_observations`` and
    ``nonfinite_observations`` are 0 in anything that was successfully read,
    because either one stops the acquisition; they are recorded anyway so that a
    reader sees the number rather than inferring it from a missing field.
    """

    logical_observations: int
    valid_positive_observations: int
    invalid_zero_observations: int
    invalid_both_zero_observations: int
    invalid_zero_contracts_only: int
    invalid_zero_notional_only: int
    negative_observations: int = 0
    nonfinite_observations: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "logical_observations": self.logical_observations,
            "valid_positive_observations": self.valid_positive_observations,
            "invalid_zero_observations": self.invalid_zero_observations,
            "invalid_both_zero_observations": self.invalid_both_zero_observations,
            "invalid_zero_contracts_only": self.invalid_zero_contracts_only,
            "invalid_zero_notional_only": self.invalid_zero_notional_only,
            "negative_observations": self.negative_observations,
            "nonfinite_observations": self.nonfinite_observations,
        }


def classify_open_interest_observations(
    contracts: np.ndarray,
    notional: np.ndarray,
    instants: np.ndarray,
    archive: Archive,
) -> tuple[np.ndarray, MetricsObservationValidity]:
    """Which logical rows are valid open-interest observations, and which are not.

    **The rule is not written here.** It is
    :data:`nn.p4_preregistration.OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY` —
    amendment A4, inside the hashed preregistration — and this function is its
    implementation rather than a second statement of it. The refusal below quotes
    the policy's own ``on_negative_or_nonfinite`` wording, so an edit to the rule
    changes what the code says as well as what the design says.

    **The rule.** A row is a valid observation only when ``sum_open_interest``
    and ``sum_open_interest_value`` are *both* strictly positive. A row with
    either consumed field exactly zero is a published source row and an invalid
    observation: it is counted, and it is kept out of the causal sequence. It is
    not given a substitute value of any kind, and it does not make its archive a
    missing day.

    **Order.** ``contracts``, ``notional`` and ``instants`` are the *logical*
    rows — schema already validated, A1's identical repeats already collapsed —
    so this classification is strictly downstream of both. That ordering is the
    policy's ``applies_after`` and is what keeps A1's duplicate accounting a
    description of what the archive published.

    **Negative and non-finite stop everything.** Neither was observed by the scan
    behind A4, so neither has a preregistered meaning, and a run that meets one
    has met something the inspection did not measure. An unparseable field
    arrives here as NaN and is refused on the same terms.

    Returns the keep-mask over the logical rows and the counts to record in
    provenance.
    """
    policy = p4_preregistration.OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
    contracts = np.asarray(contracts, dtype=np.float64)
    notional = np.asarray(notional, dtype=np.float64)

    for name, values in (
        ("sum_open_interest", contracts),
        ("sum_open_interest_value", notional),
    ):
        nonfinite = ~np.isfinite(values)
        if nonfinite.any():
            index = int(np.flatnonzero(nonfinite)[0])
            raise DerivativesSourceError(
                f"{archive.name}: {name} at {_instant_text(instants, index)} is not a "
                f"finite number. Amendment A4: {policy['on_negative_or_nonfinite']}."
            )
        negative = values < 0
        if negative.any():
            index = int(np.flatnonzero(negative)[0])
            raise DerivativesSourceError(
                f"{archive.name}: {name} at {_instant_text(instants, index)} is "
                f"{values[index]!r}, which is negative. Amendment A4: "
                f"{policy['on_negative_or_nonfinite']}."
            )

    # The rule as the policy states it — strictly positive in both consumed
    # metrics — with the zero masks kept only to partition the rejections. After
    # the two refusals above every value is finite and non-negative, so "not
    # positive" and "exactly zero" are the same set; saying it the first way is
    # saying what validity_rule says.
    valid = (contracts > 0.0) & (notional > 0.0)
    zero_contracts = contracts == 0.0
    zero_notional = notional == 0.0
    both = zero_contracts & zero_notional
    return valid, MetricsObservationValidity(
        logical_observations=int(len(contracts)),
        valid_positive_observations=int(valid.sum()),
        invalid_zero_observations=int((~valid).sum()),
        invalid_both_zero_observations=int(both.sum()),
        invalid_zero_contracts_only=int((zero_contracts & ~zero_notional).sum()),
        invalid_zero_notional_only=int((zero_notional & ~zero_contracts).sum()),
    )


def _instant_text(instants: np.ndarray, index: int) -> str:
    """The instant one logical row carries, for a refusal that names the row."""
    try:
        return pd.Timestamp(int(np.asarray(instants)[index]), unit="ns", tz="UTC").isoformat()
    except (IndexError, ValueError):  # pragma: no cover - defensive
        return f"row {index}"


def hour_floor(stamps: np.ndarray) -> np.ndarray:
    """The hour-open instant each timestamp falls in, in int64 nanoseconds."""
    return (np.asarray(stamps, dtype=np.int64) // HOUR_NS) * HOUR_NS


def source_spec() -> dict[str, Any]:
    """What the exported table is, as data, for the manifest and every artifact.

    Recorded rather than described because it is part of what a P4 number
    *means*: two runs that resolved staleness differently, carried an
    observation forward further, or wrote another column layout are not two
    readings of one source.
    """
    return {
        "source_schema": DERIVATIVES_SOURCE_SCHEMA,
        "venue": VENUE,
        "market_type": MARKET_TYPE,
        "symbol": SYMBOL,
        "pair": PAIR,
        "bucket_ns": HOUR_NS,
        "bucket_convention": (
            "row 'date' is the hour's OPEN instant, and every observation on the row "
            "was published at or before that instant. The conservative rule of §4: a "
            "row decides at date + 1h, so an observation admitted at date has a full "
            "hour between publication and use"
        ),
        "columns": [list(entry) for entry in DERIVATIVES_COLUMN_KINDS],
        "funding_rule": (
            "funding_last_rate is the realised rate of the most recent settlement with "
            "instant <= date; funding_settled_rate is the rate of the settlement that "
            "became visible in (date - 1h, date], and is 0.0 with funding_settled = 0 "
            "when none did. The predicted funding rate is never read"
        ),
        "open_interest_rule": (
            "oi_contracts and oi_notional are sum_open_interest and "
            "sum_open_interest_value at the last snapshot with create_time <= date. "
            "Not a mean over the hour: a mean is not a state anything was in"
        ),
        "perpetual_rule": (
            "perp_close is the close of the 1h perpetual candle opening at date, or of "
            "the candle one hour earlier when that candle is absent"
        ),
        "staleness_rule": (
            "an observation older than its MAX_STALENESS_HOURS bound is not carried "
            "forward: the field is marked unavailable at that hour, and every arm "
            "including the control loses the row"
        ),
        "max_staleness_hours": dict(MAX_STALENESS_HOURS),
        # Not a restatement — the preregistered object itself. There is exactly
        # one authoritative copy of this rule, in nn.p4_preregistration, and it
        # is read here rather than paraphrased. Editing it therefore moves
        # preregistration_hash AND source_spec_hash together, and a checkout
        # where the two disagree cannot be constructed.
        "duplicate_rule": dict(p4_preregistration.OPEN_INTEREST_DUPLICATE_POLICY),
        # The same identity binding, for the same reason: amendment A2's inception
        # is the preregistered object, not a date this module also happens to
        # know. plan_archives reads it too, so "preregistration says 2020-01 and
        # the planner asks for 2019-12" is not a constructible checkout.
        "funding_inception_rule": dict(p4_preregistration.FUNDING_ARCHIVE_INCEPTION_POLICY),
        # And amendment A3's, on the same terms and as a separate object: the
        # perpetual kline archive was measured separately and carries its own rule.
        "perpetual_inception_rule": dict(
            p4_preregistration.PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY
        ),
        # And amendment A4's, on the same terms: which metrics rows become
        # open-interest observations at all is part of what an exported table
        # means, so the preregistered object is carried here rather than
        # paraphrased, and editing it moves both hashes together.
        "open_interest_validity_rule": dict(
            p4_preregistration.OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY
        ),
        "gap_rule": "a gap is never filled or interpolated; it becomes staleness",
        "missing_day_rule": (
            "a metrics day that 404s, fails its published checksum, or arrives short is "
            "a MISSING DAY: it is recorded with its reason, its rows are never "
            "interpolated, and the hours it leaves without a snapshot within the "
            "staleness bound leave the sample universe for every arm. A day that cannot "
            "be classified — a transport failure that outlived its retries, an "
            "unreadable archive, an unrecognised schema — stops the acquisition instead"
        ),
        "rest_substitution": (
            "never. fapi.binance.com/futures/data/openInterestHist retains 30 days and "
            "may not stand in for a missing archive day; no REST row enters this table"
        ),
        "unavailable_age_ns": UNAVAILABLE_AGE_NS,
        "spot_source": (
            "not acquired: the spot denominator of the basis is the committed candle "
            "history named in DATA_SOURCES"
        ),
    }
