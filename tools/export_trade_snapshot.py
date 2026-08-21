"""Acquire Binance BTCUSDT trades and export a strictly-pre-Styx research source.

    python -m tools.export_trade_snapshot --probe --months 2
    python -m tools.export_trade_snapshot --plan
    python -m tools.export_trade_snapshot

**What this fetches.** Binance publishes historical market data as ZIP archives
under ``data.binance.vision``, one file per symbol per period, with a published
``.CHECKSUM`` sidecar. P3 reads the **spot** market's ``aggTrades`` for
``BTCUSDT``: one row per aggressed order, carrying the aggregate trade id, the
price, the quantity, the first and last underlying trade ids, the transaction
time and ``is_buyer_maker``.

``aggTrades`` rather than ``trades`` is a deliberate choice of *one* canonical
representation, and the two are never mixed. They describe the same executions —
``aggTrades`` folds consecutive fills of one taker order at one price into a
single row — so a table containing both would double-count every aggregated
execution, and no deduplication rule can undo that from the archives alone.
aggTrades wins on three counts: it is the aggressed-order unit the microstructure
question is actually about, it carries a stable primary key (``agg_trade_id``)
and an aggressor flag, and it is small enough that five and a half years of it
can be streamed in one pass.

**This tool has never been run against the live archive from this repository's
CI or from the session that wrote it.** Outbound access to ``data.binance.vision``
is denied by the egress policy in use, so the URL layout, the header presence and
the epoch unit below are what Binance documents and publishes, not what this code
has observed. Everything that could differ therefore fails closed rather than
being assumed: the archive member count, the CSV column count, the header row,
the published checksum and the epoch unit are all checked at run time, and
``--probe`` exists so an operator can read back exactly what the source returned
before committing to a bulk download.

**Bounded before acquisition, not filtered after it.** The seal is
``2025-08-27T23:00:00+00:00``. The last hour research needs is the final hour of
the committed OHLCV spine — ``2025-05-19T07:00`` — which is over three months
earlier, so the acquisition window is closed at that hour and no archive that
could contain a sealed trade is ever requested. The aggregator is *also*
constructed with the seal and refuses a sealed trade unconditionally, so the
bound is a property of two independent mechanisms rather than of the operator's
arithmetic.

**Streaming, one archive at a time.** Each period is downloaded to a temporary
file, folded into hourly rows, and deleted before the next is requested. Peak
disk is one archive plus the growing aggregate table, which is what makes a
five-year acquisition possible on a machine that cannot hold the raw trades.

**What is committed is the hourly aggregate, and it says so.** The raw trades are
on the order of a billion rows; the committed research source is the hourly
sufficient statistics defined in :mod:`nn.trade_aggregates`, with every archive
identified by name, byte count and SHA-256 so the derivation is auditable by
anyone who can fetch them.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from nn.data_fingerprint import fingerprint_trade_aggregates
from nn.research_contract import ResearchContract, load_contract
from nn.trade_aggregates import (
    HOUR_NS,
    HourlyTradeAggregator,
    StreamStats,
    aggregation_spec,
    aggregation_spec_hash,
    check_table,
    coverage_report,
    resolve_epoch_unit,
)

logger = logging.getLogger(__name__)

#: Names the meaning of a trade-snapshot manifest, hashed into nothing but
#: checked by the verifier: a manifest under another schema is a different
#: document and is refused rather than read with defaults.
TRADE_SNAPSHOT_SCHEMA = "chimera.trade-snapshot/1"

BASE_URL = "https://data.binance.vision"
PROVIDER = "binance-public-data"
VENUE = "binance"
MARKET_TYPE = "spot"
SOURCE_TYPE = "aggTrades"
SYMBOL = "BTCUSDT"
PAIR = "BTC/USDT"

MONTHLY_TEMPLATE = (
    "{base}/data/{market}/monthly/{source}/{symbol}/"
    "{symbol}-{source}-{year:04d}-{month:02d}.zip"
)
DAILY_TEMPLATE = (
    "{base}/data/{market}/daily/{source}/{symbol}/"
    "{symbol}-{source}-{year:04d}-{month:02d}-{day:02d}.zip"
)
CHECKSUM_SUFFIX = ".CHECKSUM"

#: The spot ``aggTrades`` CSV columns, in file order. Checked against the width
#: of the first data row: the USD-M futures archive has seven columns rather
#: than eight, and reading one as the other silently shifts every field.
RAW_COLUMNS = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
)

#: Rows parsed from an archive at a time. Large enough that the per-chunk
#: overhead is irrelevant, small enough that a busy month never materialises.
CSV_CHUNK_ROWS = 2_000_000

#: How ``is_buyer_maker`` may be spelled. Anything else is refused rather than
#: coerced: a value that falls through to ``False`` would silently relabel every
#: aggressive sell in the archive as a buy.
BOOL_WORDS = {
    "true": True,
    "True": True,
    "TRUE": True,
    "false": False,
    "False": False,
    "FALSE": False,
}

#: Where the exported source lands, beside the OHLCV research snapshot.
DEFAULT_OUT_DIR = Path("data/research")
AGGREGATE_NAME = "btc_usdt_1h_gen1_trades_hourly_pre_styx.parquet"
MANIFEST_NAME = "btc_usdt_1h_gen1_trade_snapshot_manifest.json"
DEFAULT_OHLCV_MANIFEST = Path("data/research/btc_usdt_1h_gen1_snapshot_manifest.json")

#: The longest trailing window in docs/microstructure_v1.md §1.4, in hours. The
#: acquisition must reach back at least this far before the first scored row, or
#: that row takes a declared default instead of a measurement.
MIN_WARMUP_HOURS = 168

#: One calendar month of trades before the OHLCV spine starts.
#:
#: ``microstructure_v1``'s longest trailing window is 168 hours, and a feature
#: whose window is not full takes a declared default. Starting the trade source
#: at the OHLCV *raw* start would leave the first ~90 spine rows on defaults for
#: every trailing feature; one extra archive removes that entirely and costs a
#: single month of download. The warm-up hours are never scored — they precede
#: the spine — so this widens what the features may look back at without
#: widening what research is measured on.
WARMUP_MONTHS = 1

#: Network retries, and the backoff between them. Archives are large and the
#: acquisition is long; a transient failure two hours in must not lose the run.
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = (2, 4, 8, 16)


class TradeExportError(SystemExit):
    """The trade source cannot be acquired or exported honestly."""


@dataclass(frozen=True)
class Archive:
    """One period of the public archive, and where to get it."""

    name: str
    url: str
    checksum_url: str
    #: The UTC half-open window this archive is named for. The epoch unit is
    #: resolved by requiring every timestamp inside it, so a wrong window is a
    #: refusal rather than a misread archive.
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    kind: str


def monthly_archive(year: int, month: int) -> Archive:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    url = MONTHLY_TEMPLATE.format(
        base=BASE_URL,
        market=MARKET_TYPE,
        source=SOURCE_TYPE,
        symbol=SYMBOL,
        year=year,
        month=month,
    )
    return Archive(
        name=url.rsplit("/", 1)[-1],
        url=url,
        checksum_url=url + CHECKSUM_SUFFIX,
        period_start=start,
        period_end=start + pd.offsets.MonthBegin(1),
        kind="monthly",
    )


def daily_archive(day: pd.Timestamp) -> Archive:
    day = pd.Timestamp(day).tz_convert("UTC").normalize()
    url = DAILY_TEMPLATE.format(
        base=BASE_URL,
        market=MARKET_TYPE,
        source=SOURCE_TYPE,
        symbol=SYMBOL,
        year=day.year,
        month=day.month,
        day=day.day,
    )
    return Archive(
        name=url.rsplit("/", 1)[-1],
        url=url,
        checksum_url=url + CHECKSUM_SUFFIX,
        period_start=day,
        period_end=day + pd.Timedelta(days=1),
        kind="daily",
    )


def plan_archives(start: pd.Timestamp, end: pd.Timestamp) -> list[Archive]:
    """The archives covering ``[start, end)``: whole months, then odd days.

    Monthly archives wherever a whole month is inside the window, because 65
    requests beat 1,966. The tail of the window falls mid-month, so those days
    come from daily archives — which is also why the acquisition can stop at an
    hour rather than at a month boundary, and therefore stop three months short
    of the seal instead of at the month containing it.
    """
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    if end <= start:
        raise TradeExportError(f"acquisition window [{start}, {end}) is empty")
    archives: list[Archive] = []
    cursor = start
    if cursor != cursor.normalize().replace(day=1):
        raise TradeExportError(
            f"acquisition must start on the first of a month, got {start.isoformat()}"
        )
    while True:
        month = monthly_archive(cursor.year, cursor.month)
        if month.period_end > end:
            break
        archives.append(month)
        cursor = month.period_end
    day = cursor
    while day < end:
        archives.append(daily_archive(day))
        day = day + pd.Timedelta(days=1)
    if not archives:
        raise TradeExportError(
            f"no archive covers [{start.isoformat()}, {end.isoformat()}); the window is "
            "shorter than a day"
        )
    return archives


# --------------------------------------------------------------------------- #
# acquisition
# --------------------------------------------------------------------------- #
def _get(url: str, timeout: int) -> bytes:
    """One HTTP GET with bounded retries on transient failures.

    A 404 is not retried: the archive does not exist, and hammering the public
    archive to find that out again is neither faster nor polite. A timeout or a
    5xx is retried with the backoff this repository uses everywhere else.
    """
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise TradeExportError(
                    f"{url} returned 404. Either the period is not published or the "
                    "archive layout differs from the one this tool targets; check with "
                    "--probe before assuming the former."
                ) from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < len(BACKOFF_SECONDS):
            time.sleep(BACKOFF_SECONDS[attempt])
    raise TradeExportError(f"{url} failed after {MAX_ATTEMPTS} attempts: {last}")


def download_archive(archive: Archive, dest: Path, *, timeout: int = 300) -> dict[str, Any]:
    """Fetch one archive and its published checksum, verifying the bytes.

    The published ``.CHECKSUM`` is the source's own statement about what it
    served, so it is checked when present and its absence is recorded rather
    than treated as a pass. Either way the bytes are hashed here: the manifest's
    ``sha256`` is what this run actually read, which is the number a later
    reader can reproduce.
    """
    payload = _get(archive.url, timeout)
    dest.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    published: str | None = None
    try:
        text = _get(archive.checksum_url, timeout).decode("utf-8", "replace")
        published = text.split()[0].strip().lower() if text.split() else None
    except TradeExportError:
        published = None
    if published and published != digest:
        raise TradeExportError(
            f"{archive.name} hashes to {digest} but the published checksum says "
            f"{published}. Refusing to aggregate an archive the source does not "
            "recognise as what it served."
        )
    return {
        "name": archive.name,
        "url": archive.url,
        "kind": archive.kind,
        "period_start": archive.period_start.isoformat(),
        "period_end": archive.period_end.isoformat(),
        "bytes": len(payload),
        "sha256": digest,
        "published_sha256": published,
        "checksum_verified": bool(published),
    }


def _member(zf: zipfile.ZipFile, archive_name: str) -> str:
    names = [name for name in zf.namelist() if not name.endswith("/")]
    if len(names) != 1:
        raise TradeExportError(
            f"{archive_name} holds {len(names)} files ({names[:4]}); this tool expects "
            "exactly one CSV per archive and will not guess which one is the trades"
        )
    return names[0]


def _has_header(first_line: str) -> bool:
    """True when the archive begins with a column header rather than a trade.

    Binance added headers to the spot archives partway through their history, so
    both shapes are live. Sniffed from the first field rather than from the
    period: a header row read as a trade would make ``agg_trade_id`` unparseable
    and the failure would name the wrong cause.
    """
    head = first_line.split(",", 1)[0].strip().strip('"')
    return not (head and (head.isdigit() or (head[0] == "-" and head[1:].isdigit())))


def _to_bool(values: pd.Series, archive_name: str) -> np.ndarray:
    mapped = values.astype(str).str.strip().map(BOOL_WORDS)
    if mapped.isna().any():
        bad = sorted(set(values.astype(str).str.strip()[mapped.isna()]))[:4]
        raise TradeExportError(
            f"{archive_name}: is_buyer_maker holds unrecognised value(s) {bad}. The "
            "aggressor side decides the sign of every flow feature, so an unknown "
            "spelling is refused rather than coerced to false."
        )
    return mapped.to_numpy(dtype=bool)


def iter_archive_chunks(
    path: Path, archive: Archive
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Parse one archive into ascending chunks of canonical-nanosecond trades.

    The epoch unit is resolved once, from the first chunk, against the calendar
    period the archive is named for — and every later chunk is held to the same
    window, so an archive that changed unit halfway through (which no source
    should do, and which would silently move half a month by 55 years) is a
    refusal rather than a table.
    """
    with zipfile.ZipFile(path) as zf:
        member = _member(zf, archive.name)
        with zf.open(member) as handle:
            first = handle.readline().decode("utf-8", "replace")
        if not first.strip():
            raise TradeExportError(f"{archive.name}: {member} is empty")
        header = _has_header(first)
        width = len(first.split(","))
        if width != len(RAW_COLUMNS):
            raise TradeExportError(
                f"{archive.name}: {member} has {width} columns, expected "
                f"{len(RAW_COLUMNS)} ({list(RAW_COLUMNS)}). A different archive layout "
                "shifts every field; refusing to read it as spot aggTrades."
            )
        unit: str | None = None
        with zf.open(member) as handle:
            reader = pd.read_csv(
                handle,
                header=0 if header else None,
                names=list(RAW_COLUMNS),
                usecols=[0, 1, 2, 5, 6],
                dtype={
                    "agg_trade_id": "int64",
                    "price": "float64",
                    "quantity": "float64",
                    "transact_time": "int64",
                    "is_buyer_maker": "object",
                },
                chunksize=CSV_CHUNK_ROWS,
            )
            for chunk in reader:
                raw = chunk["transact_time"].to_numpy(dtype=np.int64)
                if len(raw) == 0:
                    continue
                if unit is None:
                    unit = resolve_epoch_unit(
                        int(raw.min()),
                        int(raw.max()),
                        period_start=archive.period_start,
                        period_end=archive.period_end,
                    )
                scale = {"ms": 1_000_000, "us": 1_000, "ns": 1}[unit]
                ts = raw * scale
                lo = int(archive.period_start.value)
                hi = int(archive.period_end.value)
                outside = int(((ts < lo) | (ts >= hi)).sum())
                if outside:
                    raise TradeExportError(
                        f"{archive.name}: {outside} row(s) fall outside the archive's own "
                        f"period under the resolved epoch unit {unit!r}. The unit is not "
                        "constant across the archive and no reading of it is trustworthy."
                    )
                yield (
                    ts,
                    chunk["price"].to_numpy(dtype=np.float64),
                    chunk["quantity"].to_numpy(dtype=np.float64),
                    _to_bool(chunk["is_buyer_maker"], archive.name),
                    chunk["agg_trade_id"].to_numpy(dtype=np.int64),
                    unit,
                )


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def probe(archives: Sequence[Archive], *, timeout: int) -> dict[str, Any]:
    """Measure a couple of real archives and project the full acquisition.

    The point is to replace an estimate with a measurement *before* committing
    to a multi-hour download: compressed bytes, decompressed bytes, rows, the
    epoch unit actually in use, and the canonical bytes each period contributes
    to the committed aggregate. Nothing is written to ``data/research`` and no
    snapshot is produced.

    The projection is a per-day rate scaled to the whole window, and it says so.
    BTCUSDT trade volume is not stationary across five years, so the number is a
    planning figure with a stated basis and not a claim about the archive.
    """
    measured: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="chimera-probe-") as tmp:
        for archive in archives:
            path = Path(tmp) / archive.name
            started = time.monotonic()
            record = download_archive(archive, path, timeout=timeout)
            elapsed = time.monotonic() - started
            with zipfile.ZipFile(path) as zf:
                member = _member(zf, archive.name)
                decompressed = zf.getinfo(member).file_size
            rows = 0
            units: set[str] = set()
            hours: set[int] = set()
            for ts, _price, _qty, _maker, ids, unit in iter_archive_chunks(path, archive):
                rows += len(ids)
                units.add(unit)
                hours.update(np.unique(ts // HOUR_NS).tolist())
            days = max((archive.period_end - archive.period_start).days, 1)
            measured.append(
                {
                    **record,
                    "decompressed_bytes": int(decompressed),
                    "rows": rows,
                    "epoch_units": sorted(units),
                    "hours_touched": len(hours),
                    "days_in_period": days,
                    "rows_per_day": round(rows / days, 1),
                    "compressed_bytes_per_day": round(record["bytes"] / days, 1),
                    "download_seconds": round(elapsed, 2),
                }
            )
            path.unlink(missing_ok=True)

    total_days = sum(m["days_in_period"] for m in measured)
    rows_per_day = sum(m["rows"] for m in measured) / total_days
    bytes_per_day = sum(m["bytes"] for m in measured) / total_days
    decompressed_per_day = sum(m["decompressed_bytes"] for m in measured) / total_days
    return {
        "probed": measured,
        "basis": (
            f"{len(measured)} archive(s) covering {total_days} day(s), scaled linearly. "
            "BTCUSDT activity is not stationary over five years, so this is a planning "
            "figure with a stated basis, not a measurement of the full history."
        ),
        "rows_per_day": round(rows_per_day, 1),
        "compressed_bytes_per_day": round(bytes_per_day, 1),
        "decompressed_bytes_per_day": round(decompressed_per_day, 1),
        "canonical_bytes_per_hour": canonical_bytes_per_hour(),
    }


def canonical_bytes_per_hour() -> int:
    """Bytes one hour contributes to the committed table, uncompressed.

    Every aggregate column is eight bytes wide once canonicalised — that is what
    ``t8``/``i8``/``f8`` mean — so the committed source's size is a function of
    the number of hours and nothing else. This is the number that makes the
    aggregate approach worth it: it does not grow with trade count.
    """
    from nn.trade_aggregates import AGGREGATE_COLUMN_KINDS

    return 8 * len(AGGREGATE_COLUMN_KINDS)


def project(probe_result: dict[str, Any], archives: Sequence[Archive]) -> dict[str, Any]:
    """Scale a probe to the full planned acquisition."""
    days = sum((a.period_end - a.period_start).days for a in archives)
    hours = days * 24
    return {
        "planned_archives": len(archives),
        "planned_days": days,
        "projected_rows": int(probe_result["rows_per_day"] * days),
        "projected_compressed_bytes": int(probe_result["compressed_bytes_per_day"] * days),
        "projected_decompressed_bytes": int(probe_result["decompressed_bytes_per_day"] * days),
        "projected_aggregate_hours": hours,
        "projected_aggregate_bytes_uncompressed": hours
        * int(probe_result["canonical_bytes_per_hour"]),
        "peak_disk_note": (
            "one archive at a time plus the growing aggregate table; the raw trades are "
            "never all on disk at once"
        ),
    }


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def acquisition_window(
    ohlcv_manifest: Path, contract: ResearchContract
) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, Any]]:
    """The half-open window to acquire, taken from the committed OHLCV snapshot.

    The end is the hour *after* the spine's last hour, because the microstructure
    row for the last spine candle aggregates that candle's own trades. The start
    is :data:`WARMUP_MONTHS` before the spine's first hour, rounded down to a
    month so the plan is whole monthly archives.

    Read from the OHLCV manifest rather than written here so the trade source
    cannot silently cover a different region than the candles it will be aligned
    to, and refused outright if the resulting window reaches the seal.
    """
    payload = json.loads(Path(ohlcv_manifest).read_text())
    processed = payload["processed_outer_coverage"]
    spine_start = pd.Timestamp(processed["start"]).tz_convert("UTC")
    spine_end = pd.Timestamp(processed["end"]).tz_convert("UTC")
    # Floor to the spine's own month *first*, then step back whole months.
    # Subtracting `MonthBegin` from a mid-month timestamp rolls back only to the
    # start of that same month, which left three days of warm-up where a month
    # was intended — and `microstructure_v1`'s longest trailing window is 168
    # hours, so three days would have put the first ninety scored rows on
    # declared defaults.
    start = spine_start.normalize().replace(day=1) - pd.offsets.MonthBegin(WARMUP_MONTHS)
    end = spine_end + pd.Timedelta(hours=1)
    warmup_hours = (spine_start - start) / pd.Timedelta(hours=1)
    if warmup_hours < MIN_WARMUP_HOURS:
        raise TradeExportError(
            f"the window would give {warmup_hours:.0f} hours of trades before the first "
            f"spine hour, fewer than the {MIN_WARMUP_HOURS} microstructure_v1's longest "
            "trailing window needs. Scored rows would fall back to declared defaults."
        )
    if end > contract.sealed_test_start:
        raise TradeExportError(
            f"the acquisition window would end at {end.isoformat()}, at or after the "
            f"sealed instant {contract.sealed_test_start.isoformat()}. The window is "
            "derived from the committed spine and must not reach the seal."
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
            "spine_start": spine_start.isoformat(),
            "spine_end": spine_end.isoformat(),
        },
    )


def acquire(
    archives: Sequence[Archive],
    *,
    contract: ResearchContract,
    end: pd.Timestamp,
    timeout: int,
    workdir: Path | None = None,
) -> tuple[pd.DataFrame, StreamStats, list[dict[str, Any]]]:
    """Stream every archive through the aggregator, one at a time.

    The aggregator is constructed with the seal, so a sealed trade cannot reach
    a bucket even if the plan were wrong. Trades past ``end`` — the tail of the
    final daily archive, which covers a whole day while research needs part of
    one — are dropped here, far from the seal, and the drop is counted.
    """
    aggregator = HourlyTradeAggregator(styx_ns=int(contract.sealed_test_start.value))
    bound = int(pd.Timestamp(end).value)
    records: list[dict[str, Any]] = []
    dropped = 0
    context = (
        tempfile.TemporaryDirectory(prefix="chimera-trades-") if workdir is None else None
    )
    root = Path(context.name) if context is not None else Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        for index, archive in enumerate(archives, start=1):
            path = root / archive.name
            logger.info("[%d/%d] %s", index, len(archives), archive.url)
            record = download_archive(archive, path, timeout=timeout)
            rows = 0
            units: set[str] = set()
            for ts, price, qty, maker, ids, unit in iter_archive_chunks(path, archive):
                units.add(unit)
                keep = ts < bound
                if not keep.all():
                    dropped += int((~keep).sum())
                    ts, price, qty, maker, ids = (
                        ts[keep],
                        price[keep],
                        qty[keep],
                        maker[keep],
                        ids[keep],
                    )
                if len(ts):
                    aggregator.add_chunk(ts, price, qty, maker, ids)
                rows += int(keep.sum())
            record.update({"rows_admitted": rows, "epoch_units": sorted(units)})
            records.append(record)
            path.unlink(missing_ok=True)
            logger.info(
                "  %s: %d rows admitted, epoch unit %s", archive.name, rows, sorted(units)
            )
    finally:
        if context is not None:
            context.cleanup()
        elif workdir is not None:
            shutil.rmtree(root, ignore_errors=True)

    frame = aggregator.finish()
    stats = aggregator.stats
    for record in records:
        record["rows_dropped_past_window"] = None
    if records:
        records[-1]["rows_dropped_past_window"] = dropped
    return frame, stats, records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spine_alignment(frame: pd.DataFrame, ohlcv_manifest: Path) -> dict[str, Any]:
    """Whether every hour the OHLCV spine holds has a trade-aggregate row.

    Coverage is checked against the spine's *own* timestamps rather than against
    a date range, because a range that contains the endpoints proves nothing
    about the hours between them. A spine hour without trades is a hard failure:
    the microstructure row for that candle would have to be invented.
    """
    payload = json.loads(Path(ohlcv_manifest).read_text())
    root = Path(ohlcv_manifest).resolve().parents[2]
    spine = pd.read_parquet(
        root / payload["processed_outer_coverage"]["path"], columns=["date"]
    )
    spine_hours = pd.DatetimeIndex(pd.to_datetime(spine["date"], utc=True))
    have = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    missing = spine_hours.difference(have)
    warmup = int((have < spine_hours[0]).sum())
    return {
        "spine_rows": int(len(spine_hours)),
        "spine_start": spine_hours[0].isoformat(),
        "spine_end": spine_hours[-1].isoformat(),
        "spine_hours_covered": int(len(spine_hours) - len(missing)),
        "spine_hours_missing": int(len(missing)),
        "missing_spine_hours": [stamp.isoformat() for stamp in missing[:32]],
        "warmup_hours_before_spine": warmup,
        "trailing_hours_after_spine": int((have > spine_hours[-1]).sum()),
    }


def write_snapshot(
    frame: pd.DataFrame,
    stats: StreamStats,
    archives: Sequence[dict[str, Any]],
    *,
    contract: ResearchContract,
    out_dir: Path,
    ohlcv_manifest: Path,
    alignment_source: dict[str, Any],
    requested_from: pd.Timestamp,
    requested_through: pd.Timestamp,
    acquired_at: str,
) -> dict[str, Any]:
    """Validate the aggregate table, write it, and describe it exactly once.

    Split from :func:`acquire` so the manifest's shape can be exercised against
    a synthetic table without a network — which is the only way the verifier's
    rejections can be tested at all, since the archives cannot be corrupted on
    the source's side to order.
    """
    check_table(frame, styx=contract.sealed_test_start)
    alignment = {**alignment_source, **spine_alignment(frame, ohlcv_manifest)}
    if alignment["spine_hours_missing"]:
        raise TradeExportError(
            f"{alignment['spine_hours_missing']} spine hour(s) have no trade-aggregate "
            "row, so the microstructure row for those candles does not exist. Refusing "
            "to export a source that cannot be aligned to the research spine."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_out = out_dir / AGGREGATE_NAME
    manifest_out = out_dir / MANIFEST_NAME
    frame.to_parquet(aggregate_out, index=False)

    # Paths inside the manifest are repository-relative, because that is how the
    # verifier resolves them: two directories above `data/research/<manifest>`.
    # Recording an absolute path would make the snapshot verify only from the
    # machine that wrote it, and a copy of the tree would silently verify
    # against the original's files rather than against itself.
    tree = out_dir.resolve().parents[1]

    def relative(path: Path | str) -> str:
        candidate = Path(path)
        try:
            return str(candidate.resolve().relative_to(tree))
        except ValueError:
            return str(candidate)

    alignment["ohlcv_manifest"] = relative(alignment["ohlcv_manifest"])
    fingerprint = fingerprint_trade_aggregates(
        frame,
        exchange=VENUE,
        symbol=SYMBOL,
        market_type=MARKET_TYPE,
        source_type=SOURCE_TYPE,
        aggregation_spec_hash=aggregation_spec_hash(),
    )
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
    stream = stats.to_dict()

    manifest = {
        "snapshot_schema": TRADE_SNAPSHOT_SCHEMA,
        "contract": contract.provenance(),
        "contains_styx": False,
        "styx_instant": contract.sealed_test_start.isoformat(),
        "source": {
            "venue": VENUE,
            "market_type": MARKET_TYPE,
            "symbol": SYMBOL,
            "pair": PAIR,
            "source_type": SOURCE_TYPE,
            "provider": PROVIDER,
            "base_url": BASE_URL,
            "url_templates": {"monthly": MONTHLY_TEMPLATE, "daily": DAILY_TEMPLATE},
            "checksum_suffix": CHECKSUM_SUFFIX,
            "raw_schema": list(RAW_COLUMNS),
            "identity_column": "agg_trade_id",
            "ordering_rule": aggregation_spec()["ordering_rule"],
            "duplicate_rule": aggregation_spec()["duplicate_rule"],
            "aggressor_semantics": aggregation_spec()["aggressor_rule"],
            "uses_buyer_maker": True,
            "representation_choice": (
                "aggTrades only. aggTrades and trades describe the same executions at "
                "two granularities, so mixing them double-counts every aggregated "
                "order and no deduplication rule can undo it from the archives alone."
            ),
            "known_limitations": [
                "aggTrades folds consecutive same-price fills of one taker order into "
                "one row, so trade-size statistics are statistics of aggressed orders "
                "rather than of individual fills",
                "the aggressor side is the exchange's own is_buyer_maker flag; there is "
                "no independent way to verify it from the archive",
                "an hour in which BTCUSDT recorded no trade has no row at all, and is "
                "reported in coverage.missing_intervals rather than filled with zeros",
                "the public archive is republished from time to time; the per-archive "
                "sha256 recorded here is what this acquisition read",
                "spot only; no futures, funding, open interest, basis, liquidation or "
                "order-book data is present, by design for this checkpoint",
            ],
        },
        "acquisition": {
            "acquired_at": acquired_at,
            "requested_from": pd.Timestamp(requested_from).isoformat(),
            "requested_through": pd.Timestamp(requested_through).isoformat(),
            "archive_count": len(archives),
            "raw_bytes_total": int(sum(a["bytes"] for a in archives)),
            "raw_rows_total": int(stream["trades"]),
            "checksums_verified": int(sum(1 for a in archives if a["checksum_verified"])),
            "epoch_units_seen": sorted(
                {unit for a in archives for unit in a.get("epoch_units", [])}
            ),
            "archives": list(archives),
        },
        "aggregation": {
            **aggregation_spec(),
            "aggregation_spec_hash": aggregation_spec_hash(),
        },
        "trade_stream": stream,
        "aggregate": {
            "path": relative(aggregate_out),
            "rows": int(len(frame)),
            "start": dates[0].isoformat(),
            "end": dates[-1].isoformat(),
            "semantic_hash": fingerprint.aggregate_hash,
            "sha256": _sha256(aggregate_out),
            "columns": list(fingerprint.columns),
        },
        "coverage": coverage_report(frame),
        "alignment": alignment,
        "normalisation": {
            "canonical_timezone": "UTC",
            "epoch_representation": "int64 nanoseconds since 1970-01-01T00:00:00Z",
            "epoch_units_resolved_per_archive": True,
            "price_dtype": "float64",
            "qty_dtype": "float64",
            "notes": [
                "source epochs are milliseconds in Binance spot archives up to 2024 and "
                "microseconds from 2025-01-01; the unit is resolved per archive by "
                "requiring every timestamp inside that archive's own calendar period, "
                "and an archive that fits no unit or two units is refused",
                "every timestamp is int64 UTC nanoseconds before it is bucketed, stored "
                "or fingerprinted, so the semantic identity cannot depend on the unit "
                "the source happened to publish or on the reader's storage resolution",
                "counts are int64 and money/size are float64; a count has an exact "
                "answer and must not be recorded as a float",
            ],
        },
        "safety_checks": {
            "acquisition_bounded_before_styx": bool(
                pd.Timestamp(requested_through) <= contract.sealed_test_start
            ),
            "aggregate_strictly_before_styx": bool(dates[-1] < contract.sealed_test_start),
            "spine_fully_covered": alignment["spine_hours_missing"] == 0,
            "no_duplicate_trade_ids": stream["duplicate_ids"] == 0,
            "no_timestamp_regressions": stream["timestamp_regressions"] == 0,
            "no_invalid_rows": stream["invalid_rows"] == 0,
            "styx_rows_exported": 0,
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "trade snapshot exported:",
        f"hours={len(frame)}",
        f"trades={stream['trades']}",
        f"span={dates[0].isoformat()}..{dates[-1].isoformat()}",
        "styx_rows=0",
    )
    print(f"manifest={manifest_out}")
    size_kb = aggregate_out.stat().st_size / 1024
    print(f"aggregate={aggregate_out} ({size_kb:.0f} KiB)")
    if size_kb > 512:
        print(
            "note: this exceeds the pre-commit check-added-large-files limit of 512 KiB. "
            "Committing it is a deliberate decision about how this repository stores a "
            "research source, not something this tool should make for you."
        )
    return manifest


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
        help="list the archives that would be fetched and stop; no network access",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="download a few archives, report their real size and row count, project "
        "the full acquisition, and write nothing",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=2,
        help="how many monthly archives --probe measures (default 2)",
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
    archives = plan_archives(start, end)

    if args.plan:
        print(
            json.dumps(
                {
                    "window": {"from": start.isoformat(), "through": end.isoformat()},
                    "styx": contract.sealed_test_start.isoformat(),
                    "archives": len(archives),
                    "first": archives[0].url,
                    "last": archives[-1].url,
                    "monthly": sum(1 for a in archives if a.kind == "monthly"),
                    "daily": sum(1 for a in archives if a.kind == "daily"),
                    "canonical_bytes_per_hour": canonical_bytes_per_hour(),
                },
                indent=2,
            )
        )
        return 0

    if args.probe:
        sample = [a for a in archives if a.kind == "monthly"][: max(args.months, 1)]
        if not sample:
            sample = list(archives[: max(args.months, 1)])
        result = probe(sample, timeout=args.timeout)
        print(json.dumps({**result, "projection": project(result, archives)}, indent=2))
        return 0

    frame, stats, records = acquire(
        archives,
        contract=contract,
        end=end,
        timeout=args.timeout,
        workdir=args.workdir,
    )
    write_snapshot(
        frame,
        stats,
        records,
        contract=contract,
        out_dir=args.out_dir,
        ohlcv_manifest=args.ohlcv_manifest,
        alignment_source=alignment_source,
        requested_from=start,
        requested_through=end,
        acquired_at=pd.Timestamp.utcnow().tz_localize(None).isoformat() + "Z",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
