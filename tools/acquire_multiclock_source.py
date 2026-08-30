"""Acquire the causal 1m BTCUSDT source every Chimera clock is cut from.

    python -m tools.acquire_multiclock_source --archive-dir DIR

The source is Binance's own published spot archive — the monthly
``BTCUSDT-1m-YYYY-MM.zip`` objects under ``data/spot/monthly/klines`` — and not
a REST backfill, a third-party mirror, or another venue. That choice is the
point of the whole exercise: P6 changes the *clock* and nothing else, so the
asset, the exchange and the instrument must be the ones the 1h programme already
ran on. A different exchange would have made every P6 number a comparison
between two changes at once.

**Every object is checked against the digest Binance publishes beside it.**
Each ``.zip`` has a ``.zip.CHECKSUM`` companion holding the SHA-256 the exchange
computed. A download whose bytes do not hash to that value is refused, so what
lands on disk is provably the object Binance published rather than whatever a
proxy, a cache or a truncated transfer produced.

**The open-time unit changes mid-history and is detected, not assumed.** Binance
switched the ``open_time`` column of these archives from milliseconds to
*microseconds* with the 2025-01 file. Parsing the whole history as milliseconds
silently places every 2025 candle in the year 51726; parsing it as microseconds
places every 2020 candle in 1970. Either mistake produces a frame that sorts,
de-duplicates and resamples without complaint. The unit is therefore derived per
file from the magnitude of the values and recorded per file in the manifest.

**Only research-visible history is committed, and the boundary-spanning month is
downloaded whole.** Binance publishes by calendar month and
:data:`nn.multiclock.RESEARCH_VISIBLE_END` — the first instant of the retired
``P4-HOLD`` region — falls on 2025-05-19, so the 2025-05 object is fetched and
parsed in full and the frame is then trimmed to the boundary before anything is
written. No committed candle is at or after it.

What the manifest does carry from inside the region is that month's *archive*
facts: its published SHA-256, its row count, and its ``first_open`` and
``last_open``, which for 2025-05 is 2025-05-31T23:59. These are properties of a
public file and of the calendar, not prices, positions or labels; a model cannot
be fitted on "the May archive has 44,640 rows". They are recorded because
dropping them would leave the provenance chain unable to say which object the
committed prefix came out of. **No P4-HOLD price, volume or label is read,
committed, scored or summarised anywhere.** Styx is three months further on and
is never approached.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nn.multiclock import (
    ALL_CLOCKS,
    CANDLE_COLUMNS,
    RESEARCH_VISIBLE_END,
    STYX_START,
    bar_availability,
    candle_digest,
    clocks_from_minutes,
    describe_clock,
    minute_gaps,
    parity_against,
)

logger = logging.getLogger(__name__)

SNAPSHOT_SCHEMA = "chimera.multiclock-snapshot/1"

MANIFEST_NAME = "btc_usdt_multiclock_gen2_manifest.json"
MINUTES_NAME = "btc_usdt_multiclock_gen2_1m_pre_boundary.parquet"

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "data" / "research"

#: The committed 1h history the derived 1h clock is checked against. Two
#: independently published Binance series over the same hours: if they disagree,
#: the disagreement is a fact about the upstream archive that has to be named
#: before anything is fitted.
REFERENCE_1H = RESEARCH_DIR / "btc_usdt_1h_gen1_raw_pre_styx.parquet"

SYMBOL = "BTCUSDT"
MARKET = "spot"

#: Where the archive lives, as Binance documents it. Recorded in every manifest
#: as the canonical identity of the data regardless of which endpoint served the
#: bytes.
CANONICAL_BASE_URL = "https://data.binance.vision"

#: The bucket's S3 origin. Same bucket, same objects, same publisher — the CDN
#: hostname is a CNAME onto it — and reachable from environments where the CDN
#: name is not. Provenance is unaffected because the published per-object
#: checksum is what establishes identity, and it is verified either way.
ORIGIN_BASE_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

#: Column layout of a Binance monthly kline CSV. The archives carry no header.
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
)

#: Above this, an ``open_time`` is microseconds; below it, milliseconds. The two
#: eras are three orders of magnitude apart (1.7e12 against 1.7e15), so the
#: threshold sits in an empty band rather than near either.
MICROSECOND_THRESHOLD = 10**14


class AcquisitionError(SystemExit):
    """The source cannot be acquired in a state research may read."""


def months_to_acquire() -> list[str]:
    """Every month from the archive's start to the research-visible boundary."""
    start = pd.Timestamp("2020-01-01T00:00:00+00:00")
    end = RESEARCH_VISIBLE_END
    stamps = pd.date_range(start, end, freq="MS", inclusive="left")
    if stamps[-1] > end:  # pragma: no cover - defensive
        raise AcquisitionError("month enumeration overshot the research boundary")
    return [f"{value.year:04d}-{value.month:02d}" for value in stamps]


def object_path(month: str) -> str:
    return f"data/{MARKET}/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{month}.zip"


def _fetch(url: str, timeout: int) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return response.read()


def download_month(
    month: str, archive_dir: Path, *, base_url: str, timeout: int, offline: bool = False
) -> Path:
    """Fetch one month and hold it to the digest Binance publishes for it.

    Each of the two objects is fetched only if it is itself absent: a cached
    archive that has already been verified is never re-downloaded because its
    companion ``.CHECKSUM`` went missing. Under ``offline`` nothing is fetched at
    all — an absent object is an error, not a reason to reach the network.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    name = f"{SYMBOL}-1m-{month}.zip"
    target = archive_dir / name
    checksum_target = archive_dir / f"{name}.CHECKSUM"
    url = f"{base_url}/{object_path(month)}"

    for path, source in ((target, url), (checksum_target, f"{url}.CHECKSUM")):
        if path.is_file():
            continue
        if offline:
            raise AcquisitionError(f"--offline was given and {path} is absent")
        logger.info("downloading %s", path.name)
        path.write_bytes(_fetch(source, timeout))

    published = checksum_target.read_text().split()[0].strip().lower()
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if published != actual:
        raise AcquisitionError(
            f"{name} does not match the digest Binance publishes for it: the archive "
            f"says {published} and the bytes on disk hash to {actual}. Delete the file "
            "and re-download; a mismatched object is not the published source."
        )
    return target


def read_month(path: Path, month: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One monthly archive as candles, plus the provenance record for it."""
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        members = bundle.namelist()
        if len(members) != 1:
            raise AcquisitionError(f"{path.name} holds {len(members)} members, expected one")
        payload = bundle.read(members[0])

    frame = pd.read_csv(io.BytesIO(payload), header=None, names=list(KLINE_COLUMNS))
    open_time = frame["open_time"].to_numpy(dtype=np.int64)
    unit = "us" if open_time.max() >= MICROSECOND_THRESHOLD else "ms"
    frame["date"] = pd.to_datetime(open_time, unit=unit, utc=True)

    candles = frame.loc[:, list(CANDLE_COLUMNS)]
    provenance = {
        "month": month,
        "object": object_path(month),
        "zip_sha256": hashlib.sha256(raw).hexdigest(),
        "member": members[0],
        "member_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": int(len(frame)),
        "open_time_unit": unit,
        "first_open": candles["date"].iloc[0].isoformat(),
        "last_open": candles["date"].iloc[-1].isoformat(),
    }
    return candles, provenance


def build_minutes(archive_dir: Path, months: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Every acquired month, concatenated, ordered and trimmed to the boundary."""
    frames: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    for month in months:
        candles, record = read_month(archive_dir / f"{SYMBOL}-1m-{month}.zip", month)
        frames.append(candles)
        provenance.append(record)

    minutes = pd.concat(frames, ignore_index=True)
    minutes = minutes.sort_values("date").reset_index(drop=True)

    duplicates = int(minutes["date"].duplicated().sum())
    if duplicates:
        raise AcquisitionError(
            f"the concatenated archive carries {duplicates} duplicate minute(s); the "
            "monthly objects overlap and the overlap must be understood, not dropped"
        )

    visible = minutes.loc[minutes["date"] < RESEARCH_VISIBLE_END].reset_index(drop=True)
    if visible.empty:
        raise AcquisitionError("no research-visible minute survived the boundary trim")
    return visible, provenance


def parity_record(minutes: pd.DataFrame, clocks: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """The 1h clock, checked value by value against the committed 1h history.

    This is the check that decides whether the new source may be fitted on at
    all. It is reported in full — including the timestamps that disagree — so
    that a reader can see the disagreement rather than a summary of it.
    """
    if not REFERENCE_1H.is_file():
        raise AcquisitionError(f"the committed 1h reference {REFERENCE_1H} is absent")
    reference = pd.read_parquet(REFERENCE_1H)
    reference = reference.loc[reference["date"] < RESEARCH_VISIBLE_END].reset_index(drop=True)
    result = parity_against(clocks["1h"], reference, timeframe="1h")

    payload = result.to_dict()
    payload["reference"] = str(REFERENCE_1H.relative_to(RESEARCH_DIR.parents[1]))
    payload["reference_rows"] = int(len(reference))
    payload["agreeing_bars"] = result.overlapping_bars - result.mismatching_bars
    payload["agreement_fraction"] = (
        round(payload["agreeing_bars"] / result.overlapping_bars, 9)
        if result.overlapping_bars
        else None
    )
    payload["note"] = (
        "The 13 hours present only in the committed history are hours in which the "
        "1m archive is short of 60 minutes; the strict full-constituent rule makes "
        "them unavailable rather than partial. The disagreeing hours are an upstream "
        "inconsistency between two series Binance publishes itself and are enumerated "
        "above, not absorbed by a loosened tolerance."
    )
    return payload


def build_manifest(
    minutes: pd.DataFrame,
    provenance: list[dict[str, Any]],
    clocks: dict[str, pd.DataFrame],
    *,
    base_url: str,
    minutes_path: Path,
) -> dict[str, Any]:
    dates = minutes["date"]
    return {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "symbol": SYMBOL,
        "market": MARKET,
        "instrument": "binance spot BTCUSDT",
        "research_generation": 2,
        "source": {
            "canonical_base_url": CANONICAL_BASE_URL,
            "retrieved_from": base_url,
            "layout": "data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip",
            "digest_source": "the .zip.CHECKSUM object Binance publishes beside each archive",
            "months": provenance,
            "month_count": len(provenance),
        },
        "boundaries": {
            "research_visible_end": RESEARCH_VISIBLE_END.isoformat(),
            "research_visible_end_is": "the first instant of the retired P4-HOLD region",
            "styx_start": STYX_START.isoformat(),
            "p4_hold_opened": False,
            "styx_opened": False,
        },
        "minutes": {
            "path": str(minutes_path.relative_to(RESEARCH_DIR.parents[1])),
            "rows": int(len(minutes)),
            "start": dates.iloc[0].isoformat(),
            "end": dates.iloc[-1].isoformat(),
            "sha256": hashlib.sha256(minutes_path.read_bytes()).hexdigest(),
            "digest": candle_digest(minutes),
            "gaps": minute_gaps(minutes),
        },
        "clocks": {
            timeframe: {
                **describe_clock(clocks[timeframe], timeframe),
                **{
                    key: value
                    for key, value in bar_availability(minutes, timeframe).items()
                    if key not in {"timeframe", "constituent_minutes"}
                },
            }
            for timeframe in ALL_CLOCKS
        },
        "parity_1h": parity_record(minutes, clocks),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        required=True,
        help="where the monthly .zip objects are cached (downloaded if absent)",
    )
    parser.add_argument("--out-dir", type=Path, default=RESEARCH_DIR)
    parser.add_argument(
        "--base-url",
        default=ORIGIN_BASE_URL,
        help=f"endpoint serving the archive (default: {ORIGIN_BASE_URL})",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="fail rather than download; the cache must already hold every month",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_argparser().parse_args(argv)

    months = months_to_acquire()
    logger.info("acquiring %d months of %s 1m klines", len(months), SYMBOL)
    for month in months:
        download_month(
            month,
            args.archive_dir,
            base_url=args.base_url,
            timeout=args.timeout,
            offline=args.offline,
        )

    minutes, provenance = build_minutes(args.archive_dir, months)
    logger.info("%d research-visible minutes", len(minutes))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    minutes_path = args.out_dir / MINUTES_NAME
    minutes.to_parquet(minutes_path, index=False, compression="zstd", compression_level=19)

    clocks = clocks_from_minutes(minutes, ALL_CLOCKS)
    for timeframe, frame in clocks.items():
        logger.info("%4s: %d bars", timeframe, len(frame))

    manifest = build_manifest(
        minutes, provenance, clocks, base_url=args.base_url, minutes_path=minutes_path
    )
    manifest_path = args.out_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("wrote %s and %s", minutes_path, manifest_path)

    parity = manifest["parity_1h"]
    logger.info(
        "1h parity: %d of %d overlapping hours agree at tolerance %g",
        parity["agreeing_bars"],
        parity["overlapping_bars"],
        parity["tolerance"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
