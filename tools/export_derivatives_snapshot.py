"""Acquire Binance USD-M BTCUSDT derivatives and export P4's hourly source.

    python -m tools.export_derivatives_snapshot --plan
    python -m tools.export_derivatives_snapshot --probe
    python -m tools.export_derivatives_snapshot --workdir /tmp/chimera-derivatives

**What this fetches.** The three sources ``docs/p4_preregistration.md`` §3 fixes,
and nothing else: the monthly ``fundingRate`` archive, the **daily** ``metrics``
archive (§3.0a — Binance publishes no monthly one, and the preregistration was
corrected before any probe rather than after), and the monthly 1h perpetual
``klines`` archive. The fourth source of §3 is the spot denominator of the basis,
and it is not fetched at all: it is the committed candle history this repository
already holds.

**Three layers, in the order P3 established.**

``--plan``
    Networkless. Enumerates every archive that would be requested, the window
    they cover, and the coverage that implies — so the acquisition can be
    reviewed before a byte moves. The seal and the P4-HOLD boundary are checked
    *here*, as properties of the plan, rather than as a filter applied to data
    that has already been read.

``--probe``
    Bounded. Establishes the open-interest availability window from metadata
    (HEAD requests, binary-searched, ``O(log n)``), reads one archive of each
    field to match its schema against the preregistered allow-list, and writes
    nothing. This is where §3.6's "we do not know when the metrics archive
    begins" stops being an unknown, and it is deliberately the only step allowed
    to discover it.

acquisition (no flag)
    Streaming and resumable. One archive at a time, reduced to the hourly
    point-in-time observation immediately and deleted, so peak disk is one
    archive and peak memory is one archive's rows — not five years of
    five-minute snapshots.

**Fail-closed, and the one place it is not.** An archive that cannot be
classified — a transport failure that outlived its retries, an unreadable ZIP, a
funding header that matches no preregistered layout, a timestamp that will not
parse inside the archive's own period — stops the acquisition. A **metrics day
that 404s, fails its published checksum or arrives short** is the exception, and
only because §3.0a says exactly what it is: a missing day. It is recorded with
its reason, its rows are never interpolated, and the hours it leaves without a
snapshot inside the one-hour staleness bound leave the sample universe for every
arm — control included. The availability gate then decides whether what survived
can be evaluated at all, which is §9.0's ``not_evaluable`` rather than a negative
result.

**No REST row ever enters this table.** ``fapi.binance.com`` retains 30 days and
cannot build this history; §3.0a forbids it standing in for a missing archive
day, and there is no code path here that reads it.

**One source defect is normalised, and only one.** The official USD-M daily
metrics archives for BTCUSDT dated **2020-09-01** and **2020-09-02** were
observed to hold 576 data rows for 288 five-minute ``create_time`` instants —
every observation published twice, the paired rows identical across the full CSV
row. :func:`read_metrics` collapses rows that repeat an instant to one logical
observation **if and only if every source field in them is identical**, counts
what it removed in ``acquisition.open_interest_source_integrity``, and still
stops the acquisition when any field disagrees. Funding and kline archives are
untouched by this: a duplicate instant in either is a rejection, as §3.4 says.
Nothing here claims other metrics days are duplicated — two archives were
measured, and the rule is written for whatever the source actually serves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from nn import p4_preregistration
from nn.derivatives_sources import (
    ACQUIRED_FIELDS,
    BASE_URL,
    CSV_CHUNK_ROWS,
    DERIVATIVES_COLUMNS,
    EARLIEST_METRICS_DAY,
    FUNDING,
    HOUR_NS,
    KLINE_COLUMNS,
    METRICS_REQUIRED_COLUMNS,
    OPEN_INTEREST,
    PERPETUAL,
    PROVIDER,
    UNAVAILABLE_AGE_NS,
    Archive,
    DerivativesSourceError,
    MetricsDuplicateNormalisation,
    MetricsObservationValidity,
    canonical_member,
    check_strictly_increasing,
    classify_open_interest_observations,
    collapse_exact_duplicate_metrics_rows,
    funding_source_boundary,
    perpetual_source_boundary,
    parse_instants,
    _utc_day,
    plan_archives,
    resolve_funding_columns,
    source_spec,
    staleness_bound_ns,
)
from nn.p4_holdout import (
    assert_stage_one_bound,
    assert_stage_one_instants,
    check_holdout_boundary,
    holdout_first_instant,
)
from nn.p4_preregistration import WARMUP_HOURS, preregistration_hash
from nn.research_contract import ResearchContract, load_contract

logger = logging.getLogger(__name__)

#: Names the meaning of a derivatives-snapshot manifest. A manifest under another
#: schema is a different document and is refused rather than read with defaults.
DERIVATIVES_SNAPSHOT_SCHEMA = "chimera.derivatives-snapshot/1"

DEFAULT_OUT_DIR = Path("data/research")
SOURCE_NAME = "btc_usdt_1h_gen1_derivatives_hourly_pre_p4hold.parquet"
MANIFEST_NAME = "btc_usdt_1h_gen1_derivatives_snapshot_manifest.json"
DEFAULT_OHLCV_MANIFEST = Path("data/research/btc_usdt_1h_gen1_snapshot_manifest.json")

#: Whole months of warm-up before the spine's first hour. ``derivatives_v1``'s
#: binding window is 30 funding settlements — 240 hours — and the acquisition
#: plans whole monthly archives anyway, so a month is both the natural unit and
#: comfortably more than the minimum, which is asserted rather than assumed.
WARMUP_MONTHS = 1

MAX_ATTEMPTS = 5
BACKOFF_SECONDS = (2, 4, 8, 16)

#: How many metrics days ``--probe`` samples across the window to report a
#: missing-day rate. Bounded on purpose: the probe measures, it does not acquire.
PROBE_METRICS_SAMPLES = 12


class DerivativesExportError(SystemExit):
    """The derivatives source cannot be acquired or exported honestly."""


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
def _request(url: str, timeout: int, *, method: str = "GET") -> tuple[int, bytes]:
    """One HTTP request with bounded retries. Returns ``(status, body)``.

    A 404 comes back as ``(404, b"")`` rather than as an exception, because for
    a metrics day that is a *measurement* — §3.0a's missing day — and the caller
    is the only thing that knows whether an absent archive is data or a stop.
    Every other failure is retried and then raised: an archive this tool could
    not read is never the same thing as one the source does not publish.
    """
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            request = urllib.request.Request(url, method=method)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 404, b""
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < len(BACKOFF_SECONDS):
            time.sleep(BACKOFF_SECONDS[attempt])
    raise DerivativesExportError(
        f"{url} failed after {MAX_ATTEMPTS} attempts: {last}. An archive this tool "
        "could not fetch is not an archive the source does not publish, so the "
        "acquisition stops here rather than recording a missing day it did not measure."
    )


def archive_exists(archive: Archive, *, timeout: int) -> bool:
    """Whether the source publishes this archive, from metadata alone.

    A HEAD request. This is what ``--probe`` uses to establish the open-interest
    availability window without downloading five years of daily archives, and it
    is the only thing in this module that answers a question about the archive
    without reading it.
    """
    status, _ = _request(archive.url, timeout, method="HEAD")
    return status != 404


@dataclass
class Fetched:
    """One archive's bytes on disk, and what the source said about them."""

    archive: Archive
    path: Path
    record: dict[str, Any]


def download_archive(archive: Archive, dest: Path, *, timeout: int) -> Fetched | None:
    """Fetch one archive and its published checksum, or report it absent.

    Returns ``None`` when the source answers 404 — the archive is not published
    — and raises when the bytes disagree with the published digest. Both are
    handled by the caller: for a metrics day both are §3.0a missing days, and for
    funding or klines both stop the acquisition, because a gap in those is a gap
    in a source the design assumes is continuous.
    """
    status, _ = _request(archive.url, timeout, method="HEAD")
    if status == 404:
        return None

    digest_obj = hashlib.sha256()
    byte_count = 0
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            digest_obj = hashlib.sha256()
            byte_count = 0
            with (
                urllib.request.urlopen(archive.url, timeout=timeout) as response,
                dest.open("wb") as handle,
            ):
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest_obj.update(chunk)
                    byte_count += len(chunk)
            last = None
            break
        except urllib.error.HTTPError as exc:
            dest.unlink(missing_ok=True)
            if exc.code == 404:
                return None
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            dest.unlink(missing_ok=True)
            last = exc
        if attempt < len(BACKOFF_SECONDS):
            time.sleep(BACKOFF_SECONDS[attempt])
    if last is not None:
        raise DerivativesExportError(f"{archive.url} failed after {MAX_ATTEMPTS}: {last}")

    digest = digest_obj.hexdigest()
    published: str | None = None
    status, body = _request(archive.checksum_url, timeout)
    if status != 404 and body:
        parts = body.decode("utf-8", "replace").split()
        published = parts[0].strip().lower() if parts else None
    checksum_mismatch = bool(published and published != digest)
    return Fetched(
        archive=archive,
        path=dest,
        record={
            **archive.to_dict(),
            "bytes": byte_count,
            "sha256": digest,
            "published_sha256": published,
            "checksum_verified": bool(published) and not checksum_mismatch,
            "checksum_mismatch": checksum_mismatch,
        },
    )


# --------------------------------------------------------------------------- #
# archive readers
# --------------------------------------------------------------------------- #
def _open_member(path: Path, archive: Archive) -> tuple[zipfile.ZipFile, str, list[str], bool]:
    """The one CSV inside an archive, its first line's fields, and whether it is a header."""
    zf = zipfile.ZipFile(path)
    member = canonical_member(zf.namelist(), archive.name)
    with zf.open(member) as handle:
        first = handle.readline().decode("utf-8", "replace")
    if not first.strip():
        zf.close()
        raise DerivativesSourceError(f"{archive.name}: {member} is empty")
    fields = [value.strip().strip('"') for value in first.rstrip("\r\n").split(",")]
    return zf, member, fields, first


def read_funding(
    path: Path, archive: Archive
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One funding archive's settlements, as ascending instants and realised rates.

    The column mapping is :func:`nn.derivatives_sources.resolve_funding_columns`
    — §3.0b's allow-list — and an unrecognised layout stops here. Nothing about
    the mapping is inferred from the values, which is the whole point of fixing
    the allow-list before any archive was opened.
    """
    zf, member, fields, first = _open_member(path, archive)
    try:
        layout = resolve_funding_columns(first, archive)
        with zf.open(member) as handle:
            if layout.headerless:
                frame = pd.read_csv(handle, header=None, dtype=str)
                instant_col, rate_col = 0, 1
            else:
                frame = pd.read_csv(handle, dtype=str)
                instant_col = layout.settlement_instant
                rate_col = layout.realised_funding_rate
        if frame.empty:
            raise DerivativesSourceError(f"{archive.name}: {member} holds no settlements")
        instants = parse_instants(
            frame.iloc[:, instant_col] if layout.headerless else frame[instant_col],
            archive,
            what="the funding settlement instant",
        )
        order = np.argsort(instants, kind="stable")
        instants = instants[order]
        raw_rates = frame.iloc[:, rate_col] if layout.headerless else frame[rate_col]
        rates = pd.to_numeric(raw_rates, errors="coerce").to_numpy(dtype=np.float64)[order]
    finally:
        zf.close()
    if not np.isfinite(rates).all():
        bad = int(np.flatnonzero(~np.isfinite(rates))[0])
        raise DerivativesSourceError(
            f"{archive.name}: the realised funding rate at row {bad} is not a number. "
            "The rate is the observation; a row that does not carry one is refused "
            "rather than dropped."
        )
    check_strictly_increasing(instants, archive, what="funding settlements")
    return instants, rates, {**layout.to_dict(), "settlements": int(len(instants))}


def read_metrics(path: Path, archive: Archive) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    MetricsDuplicateNormalisation,
    MetricsObservationValidity,
]:
    """One daily metrics archive's open-interest observations, and what they cost.

    Only the three columns §3.0a names are *used*. A file without all three has no
    open interest in it under this schema and is refused rather than read for
    whatever it does have — and that refusal happens on the header, before a row
    is parsed, so normalisation can never reach a file whose schema is not the
    preregistered one.

    **Every column is read even though three are used.** The exact-duplicate test
    of :func:`nn.derivatives_sources.collapse_exact_duplicate_metrics_rows`
    compares the complete source observation, so a column this design ignores
    still decides whether two rows repeating an instant are the same row. Reading
    only the three consumed columns would make rows that differ elsewhere look
    identical, which is the one thing that rule must not do. A daily metrics
    archive is a few hundred rows, so the whole file costs nothing to hold.

    ``keep_default_na=False`` keeps every field as the literal text the CSV
    carries, so the comparison is over what the source published rather than over
    what a NA-sentinel table folded together.

    The column rows are grouped on is
    :data:`nn.p4_preregistration.OPEN_INTEREST_DUPLICATE_POLICY`'s
    ``grouping_key``, read from the hashed preregistration rather than spelled
    again here.

    **What comes back is the VALID observations, not every logical row.**
    Amendment A4 — :data:`nn.p4_preregistration.OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY`
    — is applied here, in the one function that turns an archive into
    observations, so no caller can reach the reducer around it. The order is the
    policy's own and is visible in the body: schema, then A1's exact-duplicate
    collapse over the complete published rows, then the numeric parse of the two
    consumed fields, then classification. A row with either consumed field
    exactly zero is counted in the returned
    :class:`~nn.derivatives_sources.MetricsObservationValidity` and dropped from
    the observation sequence; a negative or non-finite one stops the acquisition.
    """
    key = p4_preregistration.OPEN_INTEREST_DUPLICATE_POLICY["grouping_key"]
    zf, member, fields, _ = _open_member(path, archive)
    try:
        missing = [name for name in METRICS_REQUIRED_COLUMNS if name not in fields]
        if missing:
            raise DerivativesSourceError(
                f"{archive.name}: {member} has no column(s) {missing}; its header is "
                f"{fields}. docs/p4_preregistration.md §3.0a reads sum_open_interest and "
                "sum_open_interest_value at create_time and nothing else, so an archive "
                "without them is not the preregistered source."
            )
        with zf.open(member) as handle:
            frame = pd.read_csv(handle, dtype=str, keep_default_na=False)
    finally:
        zf.close()
    if frame.empty:
        raise DerivativesSourceError(f"{archive.name}: {member} holds no snapshots")
    instants = parse_instants(frame[key], archive, what=key)
    order = np.argsort(instants, kind="stable")
    instants = instants[order]
    rows = [tuple(row) for row in frame.to_numpy()[order]]
    keep, normalisation = collapse_exact_duplicate_metrics_rows(
        rows, instants, archive, columns=tuple(frame.columns)
    )
    instants = instants[keep]
    contracts = pd.to_numeric(frame["sum_open_interest"], errors="coerce").to_numpy(
        dtype=np.float64
    )[order][keep]
    notional = pd.to_numeric(frame["sum_open_interest_value"], errors="coerce").to_numpy(
        dtype=np.float64
    )[order][keep]
    # Still the fail-closed check it always was. A duplicate instant reaching it
    # now is one the exact-duplicate rule declined to collapse, which cannot
    # happen — and a non-monotonic archive is refused here exactly as before. It
    # runs on the logical rows, before A4 removes any of them, so an archive that
    # disagrees with itself about its own ordering is still refused for that.
    check_strictly_increasing(instants, archive, what="open-interest snapshots")
    valid, validity = classify_open_interest_observations(
        contracts, notional, instants, archive
    )
    return instants[valid], contracts[valid], notional[valid], normalisation, validity


def iter_klines(path: Path, archive: Archive) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """One monthly kline archive's ``(open_time, close)`` in ascending chunks."""
    zf, member, fields, _ = _open_member(path, archive)
    try:
        width = len(fields)
        if width != len(KLINE_COLUMNS):
            raise DerivativesSourceError(
                f"{archive.name}: {member} has {width} columns, expected "
                f"{len(KLINE_COLUMNS)} ({list(KLINE_COLUMNS)}). A different archive "
                "layout shifts every field; refusing to read it as USD-M 1h klines."
            )
        header = not all(
            value.replace(".", "", 1).replace("-", "", 1).isdigit() for value in fields[:1]
        )
        with zf.open(member) as handle:
            reader = pd.read_csv(
                handle,
                header=0 if header else None,
                names=list(KLINE_COLUMNS),
                usecols=[0, 4],
                dtype=str,
                chunksize=CSV_CHUNK_ROWS,
            )
            for chunk in reader:
                if chunk.empty:
                    continue
                instants = parse_instants(
                    chunk["open_time"], archive, what="the kline open instant"
                )
                closes = pd.to_numeric(chunk["close"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                order = np.argsort(instants, kind="stable")
                yield instants[order], closes[order]
    finally:
        zf.close()


# --------------------------------------------------------------------------- #
# hourly reduction
# --------------------------------------------------------------------------- #
@dataclass
class HourlyLastObservation:
    """Last observation at or before each hour of a fixed grid, streamed.

    The point-in-time rule of §4, as a data structure. Observations arrive in
    ascending chunks, one archive at a time, and are folded into the grid
    immediately; the only state carried between archives is the single most
    recent observation, which is what an hour at a day boundary needs and is
    also all that a five-year acquisition may hold in memory.

    ``at`` records *which instant* each hour's observation came from, so the age
    — and therefore §3.4's staleness bound — is a measurement rather than an
    assumption about how the source is spaced.
    """

    hours: np.ndarray
    width: int
    values: np.ndarray = dataclass_field(init=False)
    at: np.ndarray = dataclass_field(init=False)
    observations: int = 0
    _cursor: int = 0
    _last_ts: int | None = None
    _last_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.values = np.zeros((len(self.hours), self.width), dtype=np.float64)
        self.at = np.full(len(self.hours), UNAVAILABLE_AGE_NS, dtype=np.int64)

    def add(self, instants: np.ndarray, values: np.ndarray) -> None:
        if len(instants) == 0:
            return
        self.observations += int(len(instants))
        if self._last_ts is not None:
            instants = np.concatenate(([self._last_ts], instants))
            values = np.concatenate((self._last_values[None, :], values), axis=0)
        self._fill(instants, values, int(instants[-1]))
        self._last_ts = int(instants[-1])
        self._last_values = np.asarray(values[-1], dtype=np.float64)

    def _fill(self, instants: np.ndarray, values: np.ndarray, upto: int) -> None:
        end = int(np.searchsorted(self.hours, upto, side="right"))
        if end <= self._cursor:
            return
        target = self.hours[self._cursor : end]
        position = np.searchsorted(instants, target, side="right") - 1
        have = position >= 0
        if have.any():
            self.values[self._cursor : end][have] = values[position[have]]
            self.at[self._cursor : end][have] = instants[position[have]]
        self._cursor = end

    def finish(self) -> None:
        """Carry the last observation across the grid's tail.

        Not a repair: the carry is bounded by §3.4's staleness rule, which
        :func:`resolve_availability` applies to the ages this leaves behind. An
        hour past the bound is marked unavailable and leaves the sample universe
        for every arm; it is never filled with a number nobody observed.
        """
        if self._last_ts is None or self._cursor >= len(self.hours):
            return
        self._fill(
            np.asarray([self._last_ts], dtype=np.int64),
            self._last_values[None, :],
            int(self.hours[-1]),
        )


def resolve_availability(
    reducer: HourlyLastObservation, field: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(available, age_ns)`` for one field, under its §3.4 staleness bound.

    An hour with no observation, or one older than the bound, is **unavailable**.
    The bound is applied here rather than only in the engine so the committed
    source never carries a carry-forward the preregistration does not permit —
    and the engine applies it again, because the universe is decided there.
    """
    bound = staleness_bound_ns(field)
    seen = reducer.at >= 0
    age = np.where(seen, reducer.hours - reducer.at, UNAVAILABLE_AGE_NS)
    available = seen & (age >= 0) & (age <= bound)
    return available, np.where(available, age, UNAVAILABLE_AGE_NS)


def funding_columns(reducer: HourlyLastObservation) -> dict[str, np.ndarray]:
    """The five funding columns of the hourly schema, from the reduced grid.

    ``funding_settled`` marks the hours at which a *new* settlement became
    visible, and ``funding_settled_rate`` carries its rate. Together they are the
    visible-settlement series ``drv_funding_sum_9`` and ``drv_funding_z`` are
    windowed over, in visibility order — which is what makes those windows count
    settlements rather than hours without the engine having to hold a second
    table.

    Two settlements inside one hour would collapse to one row here and silently
    shorten that series, so the count of marked hours is checked against the
    count of distinct settlement instants the grid actually saw.
    """
    available, age = resolve_availability(reducer, FUNDING)
    at = reducer.at
    settled = np.zeros(len(at), dtype=np.int64)
    seen = at >= 0
    settled[seen] = 1
    settled[1:][seen[1:] & (at[1:] == at[:-1])] = 0
    distinct = len(np.unique(at[seen]))
    if int(settled.sum()) != distinct:
        raise DerivativesExportError(
            f"the hourly grid marks {int(settled.sum())} funding settlements but saw "
            f"{distinct} distinct settlement instants. Two settlements fell inside one "
            "hour, so the visible-settlement series drv_funding_sum_9 and drv_funding_z "
            "are windowed over would be short by one. Refusing to export it."
        )
    rate = reducer.values[:, 0]
    return {
        "funding_settled": settled,
        "funding_settled_rate": np.where(settled == 1, rate, 0.0),
        "funding_visible_count": np.cumsum(settled),
        "funding_available": available.astype(np.int64),
        "funding_last_rate": np.where(available, rate, 0.0),
        "funding_age_ns": age,
    }


def build_hourly_table(
    hours: np.ndarray,
    funding: HourlyLastObservation,
    open_interest: HourlyLastObservation,
    perpetual: HourlyLastObservation,
) -> pd.DataFrame:
    """The committed hourly source table, in the schema of :mod:`nn.derivatives_sources`."""
    oi_available, oi_age = resolve_availability(open_interest, OPEN_INTEREST)
    perp_available, perp_age = resolve_availability(perpetual, PERPETUAL)
    columns: dict[str, Any] = {
        "date": pd.to_datetime(hours, unit="ns", utc=True),
        **funding_columns(funding),
        "oi_available": oi_available.astype(np.int64),
        "oi_contracts": np.where(oi_available, open_interest.values[:, 0], 0.0),
        "oi_notional": np.where(oi_available, open_interest.values[:, 1], 0.0),
        "oi_age_ns": oi_age,
        "perp_available": perp_available.astype(np.int64),
        "perp_close": np.where(perp_available, perpetual.values[:, 0], 0.0),
        "perp_age_ns": perp_age,
    }
    frame = pd.DataFrame({name: columns[name] for name in DERIVATIVES_COLUMNS})
    check_hourly_table(frame)
    return frame


def check_hourly_table(frame: pd.DataFrame) -> None:
    """Everything the exported table must be true of, checked before it is written."""
    missing = [name for name in DERIVATIVES_COLUMNS if name not in frame.columns]
    if missing:
        raise DerivativesExportError(f"the hourly table is missing column(s) {missing}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    if len(dates) == 0:
        raise DerivativesExportError("the hourly table is empty")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise DerivativesExportError("hourly rows must be strictly increasing in time")
    step = np.diff(dates.asi8)
    if len(step) and (step != HOUR_NS).any():
        raise DerivativesExportError(
            "the hourly grid has a hole. It is built as a contiguous range so that "
            "'the last 168h' means 168 rows; a missing hour would make a trailing "
            "window silently longer than the specification allows."
        )
    for prefix, field in (
        ("funding", FUNDING),
        ("oi", OPEN_INTEREST),
        ("perp", PERPETUAL),
    ):
        available = frame[f"{prefix}_available"].to_numpy(dtype=np.int64)
        age = frame[f"{prefix}_age_ns"].to_numpy(dtype=np.int64)
        if not np.isin(available, (0, 1)).all():
            raise DerivativesExportError(f"{prefix}_available is not a 0/1 flag")
        if (age[available == 0] != UNAVAILABLE_AGE_NS).any():
            raise DerivativesExportError(
                f"{prefix}_age_ns carries an age on an hour {prefix}_available says has "
                "no observation. An unavailable field has one representation."
            )
        bound = staleness_bound_ns(field)
        if (age[available == 1] > bound).any():
            raise DerivativesExportError(
                f"{prefix}_age_ns exceeds the preregistered staleness bound of "
                f"{bound // HOUR_NS}h on an hour marked available. §3.4 does not permit "
                "carrying that observation forward."
            )
        if (age[available == 1] < 0).any():
            raise DerivativesExportError(f"{prefix}_age_ns is negative on an available hour")
    for name in ("funding_last_rate", "oi_contracts", "oi_notional", "perp_close"):
        values = frame[name].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise DerivativesExportError(f"{name} carries a non-finite value")
    # Amendment A4 is what makes these two checks provable rather than hopeful:
    # only strictly positive observations enter the reducer, so an hour marked
    # available carries a positive state in BOTH consumed metrics. A published
    # zero never reaches here — it was refused as an observation, and the hours it
    # left without one are unavailable.
    oi_hours = frame["oi_available"].to_numpy() == 1
    if (frame["oi_contracts"].to_numpy()[oi_hours] <= 0).any():
        raise DerivativesExportError("open interest is non-positive on an available hour")
    if (frame["oi_notional"].to_numpy()[oi_hours] <= 0).any():
        raise DerivativesExportError(
            "open-interest notional is non-positive on an available hour"
        )
    if (frame["perp_close"].to_numpy()[frame["perp_available"].to_numpy() == 1] <= 0).any():
        raise DerivativesExportError(
            "the perpetual close is non-positive on an available hour"
        )


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def acquisition_window(
    ohlcv_manifest: Path, contract: ResearchContract
) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, Any]]:
    """The half-open window to acquire, taken from the committed OHLCV snapshot.

    The end is the hour *after* the spine's last hour, so the last scored candle
    has its own perpetual close and open-interest snapshot and nothing later
    exists in the file at all. The start is :data:`WARMUP_MONTHS` before the
    spine's first hour, floored to a month so the plan is whole monthly archives.

    Read from the OHLCV manifest rather than written here, so the derivatives
    source cannot silently cover a different region than the candles it will be
    joined to — and refused outright if the window reaches the seal **or**
    P4-HOLD. The second bound is the one that matters for this checkpoint: stage
    1 screens on the burned blocks in order to decide whether to open the
    holdout, so a source file that contained the holdout would let the screen
    read the region it is screening for.
    """
    payload = json.loads(Path(ohlcv_manifest).read_text())
    processed = payload["processed_outer_coverage"]
    spine_start = pd.Timestamp(processed["start"]).tz_convert("UTC")
    spine_end = pd.Timestamp(processed["end"]).tz_convert("UTC")
    row_range = list(processed.get("row_range") or [0, int(processed["rows"])])
    assert_stage_one_bound(row_range[1], what=f"the OHLCV snapshot at {ohlcv_manifest}")

    start = spine_start.normalize().replace(day=1) - pd.offsets.MonthBegin(WARMUP_MONTHS)
    end = spine_end + pd.Timedelta(hours=1)
    warmup_hours = (spine_start - start) / pd.Timedelta(hours=1)
    if warmup_hours < WARMUP_HOURS:
        raise DerivativesExportError(
            f"the window would give {warmup_hours:.0f} hours of derivatives history "
            f"before the first spine hour, fewer than the {WARMUP_HOURS} "
            "docs/p4_preregistration.md §6.2 requires. Scored rows would sit inside "
            "the warm-up and leave the sample universe."
        )
    if end > contract.sealed_test_start:
        raise DerivativesExportError(
            f"the acquisition window would end at {end.isoformat()}, at or after the "
            f"sealed instant {contract.sealed_test_start.isoformat()}"
        )
    boundary = check_holdout_boundary(Path(ohlcv_manifest))
    holdout_start = pd.Timestamp(holdout_first_instant()).tz_convert("UTC")
    if end > holdout_start:
        raise DerivativesExportError(
            f"the acquisition window would end at {end.isoformat()}, past P4-HOLD's "
            f"first instant {holdout_start.isoformat()}. Stage 1's source file must "
            "structurally not contain the region stage 1 decides whether to open."
        )
    return (
        start,
        end,
        {
            "ohlcv_manifest": str(ohlcv_manifest),
            "warmup_hours_requested": int(warmup_hours),
            "ohlcv_raw_semantic_hash": payload["raw_pre_styx"]["semantic_hash"],
            "ohlcv_processed_semantic_prefix_hash": processed["semantic_prefix_hash"],
            "ohlcv_contract_hash": payload["contract"]["contract_hash"],
            "spine_rows": int(processed["rows"]),
            "spine_row_range": row_range,
            "spine_start": spine_start.isoformat(),
            "spine_end": spine_end.isoformat(),
            "p4_hold_boundary": boundary,
        },
    )


def plan(
    start: pd.Timestamp, end: pd.Timestamp, *, oi_first_day: pd.Timestamp | None = None
) -> dict[str, list[Archive]]:
    """Every archive the acquisition would request, per field. No network."""
    return {
        field: plan_archives(field, start, end, oi_first_day=oi_first_day)
        for field in ACQUIRED_FIELDS
    }


def describe_plan(
    plan_by_field: dict[str, list[Archive]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    contract: ResearchContract,
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """The plan as a reviewable object: paths, periods and the coverage implied."""
    hours = int((end - start) / pd.Timedelta(hours=1))
    return {
        "preregistration_hash": preregistration_hash(),
        "window": {"from": start.isoformat(), "through": end.isoformat(), "hours": hours},
        "styx": contract.sealed_test_start.isoformat(),
        "p4_hold_first_instant": holdout_first_instant().isoformat(),
        "bounded_before_styx": bool(end <= contract.sealed_test_start),
        "bounded_before_p4_hold": bool(end <= pd.Timestamp(holdout_first_instant())),
        "spine": {
            "rows": alignment["spine_rows"],
            "row_range": alignment["spine_row_range"],
            "start": alignment["spine_start"],
            "end": alignment["spine_end"],
            "warmup_hours_requested": alignment["warmup_hours_requested"],
        },
        "fields": {
            field: {
                "archives": len(archives),
                "granularity": archives[0].kind if archives else None,
                "first": archives[0].url if archives else None,
                "last": archives[-1].url if archives else None,
                "period_from": archives[0].period_start.isoformat() if archives else None,
                "period_through": archives[-1].period_end.isoformat() if archives else None,
            }
            for field, archives in plan_by_field.items()
        },
        "spot_denominator": (
            "not requested: the committed candle history "
            "data/research/btc_usdt_1h_gen1_raw_pre_styx.parquet"
        ),
        "earliest_intended_metrics_day": EARLIEST_METRICS_DAY,
        # Amendment A2, in the plan rather than only in the design: the month the
        # generic warm-up asked for and the month the source begins at are both
        # here, so a clamped month is visibly clamped instead of absent.
        "funding_source_boundary": funding_source_boundary(start),
        # Amendment A3, on the same terms and for the same reason, for the
        # perpetual kline archive's own boundary.
        "perpetual_source_boundary": perpetual_source_boundary(start),
        "network_accessed": False,
    }


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def first_available_metrics_day(
    lo: pd.Timestamp, hi: pd.Timestamp, *, timeout: int
) -> pd.Timestamp | None:
    """The earliest published metrics day in ``[lo, hi]``, by binary search on HEAD.

    §3.6 names this as the thing nobody knows yet and forbids working around it:
    the availability window is established from metadata, and a first day later
    than the preregistration intended *narrows the sample universe* rather than
    licensing a different source. ``O(log n)`` requests, so five years costs
    about eleven.

    The search assumes the archive is published from some first day onwards
    without a permanent hole at its start — which the sampled days in
    :func:`probe` are there to falsify rather than assert.
    """
    lo = _utc_day(lo)
    hi = _utc_day(hi)
    if not archive_exists(metrics_day(hi), timeout=timeout):
        return None
    if archive_exists(metrics_day(lo), timeout=timeout):
        return lo
    while (hi - lo) > pd.Timedelta(days=1):
        mid = lo + (hi - lo) / 2
        mid = _utc_day(mid)
        if archive_exists(metrics_day(mid), timeout=timeout):
            hi = mid
        else:
            lo = mid
    return hi


def metrics_day(day: pd.Timestamp) -> Archive:
    from nn.derivatives_sources import metrics_archive

    return metrics_archive(day)


def probe(
    plan_by_field: dict[str, list[Archive]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: int,
) -> dict[str, Any]:
    """Measure the source, bounded, and write nothing.

    Three things, and only these three: where the open-interest archive actually
    begins, what each field's schema actually is, and how often a metrics day is
    missing across a small sample. Everything it learns narrows the plan; nothing
    it learns may widen what the acquisition is allowed to read.
    """
    result: dict[str, Any] = {"preregistration_hash": preregistration_hash(), "fields": {}}

    with tempfile.TemporaryDirectory(prefix="chimera-p4-probe-") as tmp:
        root = Path(tmp)
        for field in (FUNDING, PERPETUAL):
            archives = plan_by_field[field]
            sample = archives[len(archives) // 2]
            fetched = download_archive(sample, root / sample.name, timeout=timeout)
            if fetched is None:
                raise DerivativesExportError(
                    f"{sample.url} is not published. {field} is a continuous source in "
                    "this design; a missing period in it is not a missing day and the "
                    "probe stops rather than planning around it."
                )
            if field == FUNDING:
                instants, rates, layout = read_funding(fetched.path, sample)
                result["fields"][field] = {
                    **fetched.record,
                    "layout": layout,
                    "settlements": int(len(instants)),
                    "first_settlement": pd.Timestamp(
                        int(instants[0]), unit="ns", tz="UTC"
                    ).isoformat(),
                    "rate_min": float(rates.min()),
                    "rate_max": float(rates.max()),
                    "cadence_hours_observed": sorted(
                        {int(v) for v in (np.diff(instants) // HOUR_NS)}
                    )[:6],
                }
            else:
                rows = 0
                first: int | None = None
                for instants, closes in iter_klines(fetched.path, sample):
                    rows += len(instants)
                    first = int(instants[0]) if first is None else first
                result["fields"][field] = {
                    **fetched.record,
                    "candles": rows,
                    "first_open": pd.Timestamp(first, unit="ns", tz="UTC").isoformat(),
                    "columns_expected": len(KLINE_COLUMNS),
                }
            fetched.path.unlink(missing_ok=True)

        first_day = first_available_metrics_day(
            pd.Timestamp(EARLIEST_METRICS_DAY, tz="UTC"),
            (end - pd.Timedelta(days=1)).normalize(),
            timeout=timeout,
        )
        window_start = max(
            _utc_day(first_day if first_day is not None else EARLIEST_METRICS_DAY), start
        )
        days = pd.date_range(
            window_start, end - pd.Timedelta(nanoseconds=1), freq="D", tz="UTC"
        )
        step = max(len(days) // PROBE_METRICS_SAMPLES, 1)
        sampled = list(days[::step][:PROBE_METRICS_SAMPLES])
        present, absent = [], []
        schema: dict[str, Any] | None = None
        for day in sampled:
            archive = metrics_day(day)
            fetched = download_archive(archive, root / archive.name, timeout=timeout)
            if fetched is None:
                absent.append(day.date().isoformat())
                continue
            present.append(day.date().isoformat())
            if schema is None:
                instants, contracts, notional, normalisation, validity = read_metrics(
                    fetched.path, archive
                )
                # Every figure below is over the VALID observations, because that
                # is what the acquisition would use. A day whose rows were all
                # rejected by amendment A4 has no minimum and no maximum, and the
                # probe says so rather than raising on an empty reduction.
                schema = {
                    "columns_read": list(METRICS_REQUIRED_COLUMNS),
                    "snapshots": int(len(instants)),
                    "normalisation": normalisation.to_dict(),
                    "validity": validity.to_dict(),
                    "cadence_minutes_observed": sorted(
                        {int(v) for v in (np.diff(instants) // (60 * 1_000_000_000))}
                    )[:6],
                    "contracts_min": float(contracts.min()) if len(contracts) else None,
                    "contracts_max": float(contracts.max()) if len(contracts) else None,
                    "notional_min": float(notional.min()) if len(notional) else None,
                    "notional_max": float(notional.max()) if len(notional) else None,
                    "measured_on": day.date().isoformat(),
                }
            fetched.path.unlink(missing_ok=True)

        result["fields"][OPEN_INTEREST] = {
            "granularity": "daily",
            "earliest_intended_day": EARLIEST_METRICS_DAY,
            "first_available_day": None if first_day is None else first_day.date().isoformat(),
            "search": "binary search on HEAD; metadata only",
            "sampled_days": len(sampled),
            "sampled_present": present,
            "sampled_absent": absent,
            "sampled_missing_fraction": (
                round(len(absent) / len(sampled), 4) if sampled else None
            ),
            "schema": schema,
        }

    result["planned_days_open_interest"] = len(
        plan_archives(OPEN_INTEREST, start, end, oi_first_day=first_day)
    )
    result["note"] = (
        "a sampled missing-day rate is a planning figure, not the availability gate. "
        "The gate of docs/p4_preregistration.md §8.0 is computed per block from the "
        "rows that actually survive the sample-universe conditions, after acquisition."
    )
    return result


# --------------------------------------------------------------------------- #
# acquisition
# --------------------------------------------------------------------------- #
def acquire(
    plan_by_field: dict[str, list[Archive]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: int,
    workdir: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Stream every archive into the hourly grid, one at a time.

    Returns the hourly table, the per-archive records, and the missing metrics
    days with the reason each was missing. Nothing is held between archives
    except the single most recent observation per field, so a five-year
    acquisition costs one archive of disk and one archive of memory.
    """
    hours = np.arange(int(start.value), int(end.value), HOUR_NS, dtype=np.int64)
    funding = HourlyLastObservation(hours=hours, width=1)
    open_interest = HourlyLastObservation(hours=hours, width=2)
    perpetual = HourlyLastObservation(hours=hours, width=1)

    records: list[dict[str, Any]] = []
    missing_days: list[dict[str, Any]] = []
    context = tempfile.TemporaryDirectory(prefix="chimera-p4-") if workdir is None else None
    root = Path(context.name) if context is not None else Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        for archive in plan_by_field[FUNDING]:
            fetched = _require(archive, root, timeout=timeout)
            instants, rates, layout = read_funding(fetched.path, archive)
            funding.add(instants, rates[:, None])
            records.append({**fetched.record, "rows": int(len(instants)), "layout": layout})
            fetched.path.unlink(missing_ok=True)

        for archive in plan_by_field[PERPETUAL]:
            fetched = _require(archive, root, timeout=timeout)
            rows = 0
            for instants, closes in iter_klines(fetched.path, archive):
                perpetual.add(instants, closes[:, None])
                rows += len(instants)
            records.append({**fetched.record, "rows": rows})
            fetched.path.unlink(missing_ok=True)

        for index, archive in enumerate(plan_by_field[OPEN_INTEREST], start=1):
            if index % 100 == 0:
                logger.info("open interest: %d/%d", index, len(plan_by_field[OPEN_INTEREST]))
            record, reason = _fetch_metrics_day(archive, root, timeout=timeout)
            if reason is not None:
                missing_days.append(
                    {
                        "day": archive.period_start.date().isoformat(),
                        "reason": reason,
                        **record,
                    }
                )
                continue
            path = root / archive.name
            instants, contracts, notional, normalisation, validity = read_metrics(
                path, archive
            )
            open_interest.add(instants, np.column_stack((contracts, notional)))
            # "rows" is the VALID observations this archive contributed to the
            # causal sequence. The raw row count it was read from is in the
            # normalisation block beside it and the logical count A1 left behind
            # is in the validity block, so neither a collapse nor a rejected
            # observation is ever invisible in the per-archive record.
            records.append(
                {
                    **record,
                    "rows": int(len(instants)),
                    "normalisation": normalisation.to_dict(),
                    "validity": validity.to_dict(),
                }
            )
            path.unlink(missing_ok=True)
    finally:
        if context is not None:
            context.cleanup()
        elif workdir is not None:
            shutil.rmtree(root, ignore_errors=True)

    for reducer in (funding, open_interest, perpetual):
        reducer.finish()
    return build_hourly_table(hours, funding, open_interest, perpetual), records, missing_days


def _require(archive: Archive, root: Path, *, timeout: int) -> Fetched:
    """Fetch an archive the design assumes is continuous, or stop."""
    fetched = download_archive(archive, root / archive.name, timeout=timeout)
    if fetched is None:
        raise DerivativesExportError(
            f"{archive.url} is not published. {archive.field} is a continuous source in "
            "this design, so a missing period in it is not a §3.0a missing day and the "
            "acquisition stops rather than exporting a table with a hole it did not "
            "measure."
        )
    if fetched.record["checksum_mismatch"]:
        raise DerivativesExportError(
            f"{archive.name} hashes to {fetched.record['sha256']} but the published "
            f"checksum says {fetched.record['published_sha256']}. Refusing to read an "
            "archive the source does not recognise as what it served."
        )
    return fetched


def _fetch_metrics_day(
    archive: Archive, root: Path, *, timeout: int
) -> tuple[dict[str, Any], str | None]:
    """Fetch one metrics day, or classify it as §3.0a's missing day.

    The three reasons §3.0a names — a 404, a failed checksum, a short archive —
    and nothing else. A transport failure has already raised inside
    :func:`download_archive`, and a schema this reader does not recognise raises
    in :func:`read_metrics`: neither is a measurement of what the archive
    contains, so neither may be recorded as a missing day.
    """
    fetched = download_archive(archive, root / archive.name, timeout=timeout)
    if fetched is None:
        return {**archive.to_dict()}, "not_published"
    if fetched.record["checksum_mismatch"]:
        fetched.path.unlink(missing_ok=True)
        return fetched.record, "checksum_mismatch"
    try:
        with zipfile.ZipFile(fetched.path) as zf:
            member = canonical_member(zf.namelist(), archive.name)
            if zf.getinfo(member).file_size == 0:
                fetched.path.unlink(missing_ok=True)
                return fetched.record, "empty_member"
    except zipfile.BadZipFile as exc:
        raise DerivativesExportError(
            f"{archive.name} is not a readable ZIP ({exc}). An archive this tool cannot "
            "open is not an archive the source does not publish."
        ) from exc
    return fetched.record, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spine_alignment(frame: pd.DataFrame, ohlcv_manifest: Path) -> dict[str, Any]:
    """Whether every hour the OHLCV spine holds has a derivatives row.

    A *row*, not an observation. Every spine hour must be present in the grid;
    whether its funding, open interest and perpetual close were available is the
    sample universe's question and is answered in :mod:`nn.p4_universe`, on the
    numbers this file records rather than on a claim made here.
    """
    payload = json.loads(Path(ohlcv_manifest).read_text())
    root = Path(ohlcv_manifest).resolve().parents[2]
    spine = pd.read_parquet(
        root / payload["processed_outer_coverage"]["path"], columns=["date"]
    )
    spine_hours = pd.DatetimeIndex(pd.to_datetime(spine["date"], utc=True))
    have = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    missing = spine_hours.difference(have)
    return {
        "spine_rows": int(len(spine_hours)),
        "spine_start": spine_hours[0].isoformat(),
        "spine_end": spine_hours[-1].isoformat(),
        "spine_hours_covered": int(len(spine_hours) - len(missing)),
        "spine_hours_missing": int(len(missing)),
        "missing_spine_hours": [stamp.isoformat() for stamp in missing[:32]],
        "warmup_hours_before_spine": int((have < spine_hours[0]).sum()),
        "trailing_hours_after_spine": int((have > spine_hours[-1]).sum()),
    }


def coverage_report(
    frame: pd.DataFrame, missing_days: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Per-field availability over the exported grid, and the missing days behind it."""
    hours = len(frame)
    out: dict[str, Any] = {"hours": hours, "missing_metrics_days": list(missing_days)}
    for prefix, field in (("funding", FUNDING), ("oi", OPEN_INTEREST), ("perp", PERPETUAL)):
        available = frame[f"{prefix}_available"].to_numpy(dtype=np.int64) == 1
        age = frame[f"{prefix}_age_ns"].to_numpy(dtype=np.int64)
        gaps = _longest_false_run(available)
        out[field] = {
            "hours_available": int(available.sum()),
            "hours_unavailable": int((~available).sum()),
            "available_fraction": round(float(available.mean()), 6) if hours else None,
            "max_contiguous_unavailable_hours": gaps,
            "max_age_hours": (
                int(age[available].max() // HOUR_NS) if available.any() else None
            ),
            "staleness_bound_hours": staleness_bound_ns(field) // HOUR_NS,
        }
    return out


def metrics_source_integrity(archives: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The source accounting for the metrics archives — A1's and A4's — summed.

    Derived from the per-archive ``normalisation`` blocks rather than counted
    separately, so the aggregate and the records it came from cannot drift; the
    verifier re-adds the same sums from the manifest's own records.

    ``conflicting_duplicate_instants`` is 0 in every manifest that exists, and
    that is a property rather than a measurement: a conflicting duplicate stops
    the acquisition inside
    :func:`nn.derivatives_sources.collapse_exact_duplicate_metrics_rows`, so no
    snapshot can be written after one was seen. It is recorded anyway, because a
    reader should be able to see the number rather than infer it from the absence
    of a field.

    ``negative_observations`` and ``nonfinite_observations`` are 0 in every
    manifest for the same reason, and by amendment A4 rather than A1: either one
    stops the acquisition inside
    :func:`nn.derivatives_sources.classify_open_interest_observations`, so no
    snapshot can be written after one was seen.

    **The A4 half is what keeps a rejected observation from looking like a
    missing one.** ``logical_observations`` is what the archive published after
    A1 collapsed identical repeats; ``valid_positive_observations`` is what
    entered the causal sequence; the difference is counted, partitioned by which
    consumed field was zero, and adds up exactly. A reader can therefore see that
    the source served those rows and that this protocol declined them, which is
    not the same statement as "the archive did not publish them".

    The rules these counts were produced under are not restated here: they are
    ``source.duplicate_rule`` and ``source.open_interest_validity_rule`` in the
    same manifest, bound by ``source_spec_hash``, and a second copy could only
    ever disagree with the first.
    """
    entries = [entry["normalisation"] for entry in archives if "normalisation" in entry]
    valid = [entry["validity"] for entry in archives if "validity" in entry]
    return {
        "archives_read": len(entries),
        "raw_rows_read": sum(int(e["rows_read"]) for e in entries),
        "logical_observations_retained": sum(int(e["observations_retained"]) for e in entries),
        "exact_duplicate_rows_collapsed": sum(
            int(e["exact_duplicate_rows_collapsed"]) for e in entries
        ),
        "duplicate_instants_collapsed": sum(int(e["duplicate_instants"]) for e in entries),
        "archives_with_exact_duplicates": sum(
            1 for e in entries if int(e["exact_duplicate_rows_collapsed"])
        ),
        "conflicting_duplicate_instants": 0,
        "archives_classified": len(valid),
        "logical_observations": sum(int(e["logical_observations"]) for e in valid),
        "valid_positive_observations": sum(
            int(e["valid_positive_observations"]) for e in valid
        ),
        "invalid_zero_observations": sum(int(e["invalid_zero_observations"]) for e in valid),
        "invalid_both_zero_observations": sum(
            int(e["invalid_both_zero_observations"]) for e in valid
        ),
        "invalid_zero_contracts_only": sum(
            int(e["invalid_zero_contracts_only"]) for e in valid
        ),
        "invalid_zero_notional_only": sum(int(e["invalid_zero_notional_only"]) for e in valid),
        "archives_with_invalid_zero_observations": sum(
            1 for e in valid if int(e["invalid_zero_observations"])
        ),
        "negative_observations": 0,
        "nonfinite_observations": 0,
    }


def _longest_false_run(flags: np.ndarray) -> int:
    """Longest run of ``False``. The §8.0 outage statistic, computed once."""
    longest = current = 0
    for value in np.asarray(flags, dtype=bool):
        current = 0 if value else current + 1
        longest = max(longest, current)
    return int(longest)


def write_snapshot(
    frame: pd.DataFrame,
    archives: Sequence[dict[str, Any]],
    missing_days: Sequence[dict[str, Any]],
    *,
    contract: ResearchContract,
    out_dir: Path,
    ohlcv_manifest: Path,
    alignment_source: dict[str, Any],
    requested_from: pd.Timestamp,
    requested_through: pd.Timestamp,
    acquired_at: str,
    oi_first_available_day: str | None = None,
) -> dict[str, Any]:
    """Validate the hourly table, write it, and describe it exactly once.

    Split from :func:`acquire` for the reason P3's exporter splits the same two
    steps: the manifest's shape and the verifier's rejections can then be
    exercised against a synthetic table, which is the only way to test them at
    all — the public archive cannot be corrupted to order.
    """
    from nn.data_fingerprint import fingerprint_derivatives_hourly

    check_hourly_table(frame)
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    assert_stage_one_instants(dates, what="the exported derivatives source")
    if dates[-1] >= contract.sealed_test_start:
        raise DerivativesExportError("the exported table reaches the sealed instant")

    alignment = {**alignment_source, **spine_alignment(frame, ohlcv_manifest)}
    if alignment["spine_hours_missing"]:
        raise DerivativesExportError(
            f"{alignment['spine_hours_missing']} spine hour(s) have no derivatives row, "
            "so the P4 row for those candles does not exist. Refusing to export a source "
            "that cannot be aligned to the research spine."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_out = out_dir / SOURCE_NAME
    manifest_out = out_dir / MANIFEST_NAME
    frame.to_parquet(source_out, index=False)

    tree = out_dir.resolve().parents[1]

    def relative(path: Path | str) -> str:
        candidate = Path(path)
        try:
            return str(candidate.resolve().relative_to(tree))
        except ValueError:
            return str(candidate)

    alignment["ohlcv_manifest"] = relative(alignment["ohlcv_manifest"])
    fingerprint = fingerprint_derivatives_hourly(frame, source_spec_hash=source_spec_hash())

    manifest = {
        "snapshot_schema": DERIVATIVES_SNAPSHOT_SCHEMA,
        "preregistration_hash": preregistration_hash(),
        "contract": contract.provenance(),
        "contains_styx": False,
        "contains_p4_hold": False,
        "styx_instant": contract.sealed_test_start.isoformat(),
        "p4_hold_first_instant": holdout_first_instant().isoformat(),
        "source": {
            **source_spec(),
            "provider": PROVIDER,
            "base_url": BASE_URL,
            "known_limitations": [
                "the funding rate is the realised settlement only; the continuously "
                "published predicted rate is never read",
                "open interest is a 5-minute snapshot reduced to the last observation "
                "at or before each hour, so an hour's row is a state and not a flow",
                "the perpetual close is the USD-M contract's own candle, not a mark or "
                "index price, and not a quarterly future",
                "a metrics day the archive does not publish is a missing day; its hours "
                "leave the sample universe for every arm and are never interpolated",
                "the public archive is republished from time to time; the per-archive "
                "sha256 recorded here is what this acquisition read",
                "the official USD-M daily metrics archives for BTCUSDT dated 2020-09-01 "
                "and 2020-09-02 were observed to hold 576 data rows for 288 five-minute "
                "instants, every observation appearing twice with the paired rows "
                "identical across the full CSV row. Rows like those are collapsed to one "
                "logical observation and counted under "
                "acquisition.open_interest_source_integrity; rows sharing an instant "
                "that disagree in any field still stop the acquisition. No claim is made "
                "about other days: this is what two archives were measured to contain",
                "the daily metrics archive publishes rows whose consumed open-interest "
                "metrics are exactly zero — both fields zero, and, on 2023-04-10, twelve "
                "rows with positive contracts and zero notional. Under amendment A4 such "
                "a row is a published source row and an invalid observation: it is "
                "counted under acquisition.open_interest_source_integrity, it does not "
                "enter the causal observation sequence, and it is never interpolated, "
                "averaged or substituted. No claim is made about what the true open "
                "interest was during those rows, and none that the venue documents zero "
                "as a sentinel. The hours a run of them leaves without a valid "
                "observation inside the 1-hour staleness bound are unavailable for every "
                "arm — the longest such run observed was 102 consecutive five-minute "
                "rows, 8.5 hours",
            ],
        },
        "acquisition": {
            "acquired_at": acquired_at,
            "requested_from": pd.Timestamp(requested_from).isoformat(),
            "requested_through": pd.Timestamp(requested_through).isoformat(),
            "archive_count": len(archives),
            "raw_bytes_total": int(sum(a.get("bytes", 0) for a in archives)),
            "checksums_verified": int(sum(1 for a in archives if a.get("checksum_verified"))),
            "funding_layouts_seen": sorted(
                {a["layout"]["layout"] for a in archives if a.get("layout")}
            ),
            "funding_source_boundary": funding_source_boundary(pd.Timestamp(requested_from)),
            "perpetual_source_boundary": perpetual_source_boundary(
                pd.Timestamp(requested_from)
            ),
            "open_interest_first_available_day": oi_first_available_day,
            "missing_metrics_days": len(missing_days),
            "open_interest_source_integrity": metrics_source_integrity(archives),
            "archives": list(archives),
        },
        "hourly": {
            "path": relative(source_out),
            "rows": int(len(frame)),
            "start": dates[0].isoformat(),
            "end": dates[-1].isoformat(),
            "semantic_hash": fingerprint.source_hash,
            "sha256": _sha256(source_out),
            "columns": list(fingerprint.columns),
            "source_spec_hash": source_spec_hash(),
        },
        "coverage": coverage_report(frame, missing_days),
        "alignment": alignment,
        "normalisation": {
            "canonical_timezone": "UTC",
            "epoch_representation": "int64 nanoseconds since 1970-01-01T00:00:00Z",
            "epoch_units_resolved_per_archive": True,
            "notes": [
                "an archive's own calendar period resolves its epoch unit, and an "
                "archive that fits no unit or two units is refused",
                "availability is a 0/1 flag and staleness an int64 age, because a "
                "semantic fingerprint has no representation for a missing instant",
            ],
        },
        "safety_checks": {
            "acquisition_bounded_before_styx": bool(
                pd.Timestamp(requested_through) <= contract.sealed_test_start
            ),
            "acquisition_bounded_before_p4_hold": bool(
                pd.Timestamp(requested_through) <= pd.Timestamp(holdout_first_instant())
            ),
            "table_strictly_before_styx": bool(dates[-1] < contract.sealed_test_start),
            "table_strictly_before_p4_hold": bool(
                dates[-1] < pd.Timestamp(holdout_first_instant())
            ),
            "spine_fully_covered": alignment["spine_hours_missing"] == 0,
            "styx_rows_exported": 0,
            "p4_hold_rows_exported": 0,
            "rest_rows_exported": 0,
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "derivatives snapshot exported:",
        f"hours={len(frame)}",
        f"span={dates[0].isoformat()}..{dates[-1].isoformat()}",
        "styx_rows=0",
        "p4_hold_rows=0",
    )
    print(f"manifest={manifest_out}")
    print(f"source={source_out} ({source_out.stat().st_size / 1024:.0f} KiB)")
    return manifest


def source_spec_hash() -> str:
    """Identity of the exported table's definition, for the manifest and cells."""
    return hashlib.sha256(
        json.dumps(source_spec(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ohlcv-manifest", type=Path, default=DEFAULT_OHLCV_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--research-contract", default="btc-usdt-1h-gen1")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="list every archive that would be fetched and stop; no network access",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="establish the open-interest availability window from metadata, read one "
        "archive of each field to match its schema, and write nothing",
    )
    parser.add_argument(
        "--oi-first-day",
        default=None,
        help="the earliest metrics day to request, when a probe has established one "
        "later than the preregistration's intent. Never earlier: an earlier day is a "
        "day the source does not publish.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="where archives are staged; defaults to a temporary directory removed on "
        "exit. Point it at a large disk for a long acquisition.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)
    contract = load_contract(args.research_contract)
    start, end, alignment_source = acquisition_window(args.ohlcv_manifest, contract)

    oi_first_day = None
    if args.oi_first_day:
        oi_first_day = _utc_day(args.oi_first_day)
        if oi_first_day < _utc_day(EARLIEST_METRICS_DAY):
            raise DerivativesExportError(
                f"--oi-first-day {args.oi_first_day} is before the preregistered "
                f"earliest intended day {EARLIEST_METRICS_DAY}. A probe may narrow the "
                "window; it may not widen it."
            )
    plan_by_field = plan(start, end, oi_first_day=oi_first_day)

    if args.plan:
        print(
            json.dumps(
                describe_plan(plan_by_field, start, end, contract, alignment_source), indent=2
            )
        )
        return 0

    if args.probe:
        print(
            json.dumps(
                probe(plan_by_field, start=start, end=end, timeout=args.timeout), indent=2
            )
        )
        return 0

    frame, records, missing_days = acquire(
        plan_by_field, start=start, end=end, timeout=args.timeout, workdir=args.workdir
    )
    write_snapshot(
        frame,
        records,
        missing_days,
        contract=contract,
        out_dir=args.out_dir,
        ohlcv_manifest=args.ohlcv_manifest,
        alignment_source=alignment_source,
        requested_from=start,
        requested_through=end,
        acquired_at=pd.Timestamp.utcnow().tz_localize(None).isoformat() + "Z",
        oi_first_available_day=(
            None if oi_first_day is None else oi_first_day.date().isoformat()
        ),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
