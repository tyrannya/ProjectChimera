"""Plan, and attempt, the P13 source acquisition — and record what happened.

Three modes, and the split is the point.

``--plan`` is **networkless**. Every object P13 needs is computed from the frozen
preregistration and a calendar, so the plan can be reviewed, committed and
verified without a single request leaving the machine. That is the same division
``tools.export_derivatives_snapshot`` uses, and it is what makes a refusal
testable: a plan that names the wrong archive is a bug you can see before you
have downloaded anything.

``--acquire`` adds HTTP on top and **fails closed**. It does not guess, it does
not fall back to a REST endpoint, and it does not substitute a source it can
reach for one it cannot. When acquisition is impossible it writes a machine-
readable refusal record naming exactly which host refused and why — because "the
data could not be obtained" is a claim that needs evidence like any other, and
NOT EVALUABLE is a real research outcome rather than an error to be worked
around.

``--download`` actually fetches the planned objects and their published
``.CHECKSUM`` companions, verifies every one against the digest Binance publishes
for it, extracts the single CSV member, and writes an acquisition manifest. It
**stops at the first failure**: a month that cannot be obtained is not skipped,
not marked optional and not substituted from anywhere else, because a screen
assembled from whatever happened to be reachable is a screen over a universe the
frozen design did not specify. The verification and manifest logic lives in
:mod:`nn.p13_acquisition` so it can be tested without a network.

Raw archives are written to a CACHE OUTSIDE the repository. They are market data
and are not committed; what is committed is the manifest that pins their exact
identities and digests.

Nothing here reads P4-HOLD or Styx: the plan stops at the preregistered research
boundary, and :func:`plan_objects` refuses to emit an object whose period begins
at or after it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from nn import p13_preregistration as prereg
from nn.p13_acquisition import (
    AcquisitionError,
    acquire_all,
    acquisition_manifest,
    assert_allowed_url,
)

#: Binance's own path grammar, read from `binance/binance-public-data`
#: `python/utility.py::get_path` rather than guessed. Spot and futures differ
#: only in the prefix; the fundingRate family carries no interval segment.
BASE_URL = "https://data.binance.vision"
CHECKSUM_SUFFIX = ".CHECKSUM"

REFUSAL_SCHEMA = "chimera.p13-acquisition-refusal/1"
PLAN_SCHEMA = "chimera.p13-acquisition-plan/1"


class AcquisitionRefused(RuntimeError):
    """The preregistered sources cannot be obtained under the frozen rules."""


@dataclass(frozen=True)
class PlannedObject:
    """One archive object the frozen design requires, and where it lives."""

    field: str
    market: str
    data_type: str
    symbol: str
    interval: str | None
    period: str
    path: str
    url: str
    checksum_url: str

    @property
    def object_name(self) -> str:
        """The published object's filename, derived rather than stored.

        A PROPERTY and not a field, deliberately: ``plan_payload`` hashes
        ``asdict`` of these records, and the committed acquisition plan quotes
        that digest. Adding a field would move a digest that describes an object
        list which has not changed.
        """
        return self.path.rsplit("/", 1)[-1]


def _months(start: str, end_exclusive: str) -> Iterator[tuple[int, int]]:
    """Every ``(year, month)`` whose month begins strictly before the boundary."""
    first = datetime.fromisoformat(start)
    last = datetime.fromisoformat(end_exclusive)
    year, month = first.year, first.month
    while datetime(year, month, 1, tzinfo=timezone.utc) < last:
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def _path(market: str, data_type: str, symbol: str, interval: str | None) -> str:
    prefix = "data/spot" if market == "spot" else f"data/futures/{market}"
    if interval is None:
        return f"{prefix}/monthly/{data_type}/{symbol}/"
    return f"{prefix}/monthly/{data_type}/{symbol}/{interval}/"


#: The four sources, in the shape the preregistration froze them.
SOURCE_LAYOUT: tuple[dict[str, Any], ...] = (
    {"field": "spot_price", "market": "spot", "data_type": "klines", "interval": "1h"},
    {"field": "perpetual_price", "market": "um", "data_type": "klines", "interval": "1h"},
    {"field": "mark_price", "market": "um", "data_type": "markPriceKlines", "interval": "1h"},
    {
        "field": "funding_settlement",
        "market": "um",
        "data_type": "fundingRate",
        "interval": None,
    },
)


def plan_objects(symbol: str = "BTCUSDT") -> list[PlannedObject]:
    """Every object the frozen design needs. No network, no guessing.

    The boundary is enforced here rather than downstream: an object whose month
    begins at or after the research boundary is never planned, so it cannot be
    fetched by accident and then filtered by something that forgets to.
    """
    boundary = prereg.DATA_BOUNDARY["span_end_exclusive"]
    start = prereg.DATA_BOUNDARY["span_start_inclusive"]
    planned: list[PlannedObject] = []
    for source in SOURCE_LAYOUT:
        interval = source["interval"]
        path = _path(source["market"], source["data_type"], symbol, interval)
        for year, month in _months(start, boundary):
            period = f"{year:04d}-{month:02d}"
            if interval is None:
                name = f"{symbol}-{source['data_type']}-{period}.zip"
            else:
                name = f"{symbol}-{interval}-{period}.zip"
            planned.append(
                PlannedObject(
                    field=source["field"],
                    market=source["market"],
                    data_type=source["data_type"],
                    symbol=symbol,
                    interval=interval,
                    period=period,
                    path=path + name,
                    url=f"{BASE_URL}/{path}{name}",
                    checksum_url=f"{BASE_URL}/{path}{name}{CHECKSUM_SUFFIX}",
                )
            )
    return planned


def plan_payload(symbol: str = "BTCUSDT") -> dict[str, Any]:
    objects = plan_objects(symbol)
    by_field: dict[str, int] = {}
    for obj in objects:
        by_field[obj.field] = by_field.get(obj.field, 0) + 1
    blob = json.dumps([asdict(o) for o in objects], sort_keys=True, separators=(",", ":"))
    return {
        "plan_schema": PLAN_SCHEMA,
        "preregistration_hash": prereg.preregistration_hash(),
        "symbol": symbol,
        "span_start_inclusive": prereg.DATA_BOUNDARY["span_start_inclusive"],
        "span_end_exclusive": prereg.DATA_BOUNDARY["span_end_exclusive"],
        "archive_host": BASE_URL,
        "object_count": len(objects),
        "objects_by_field": by_field,
        "plan_digest": "sha256:" + hashlib.sha256(blob.encode()).hexdigest(),
        "objects": [asdict(o) for o in objects],
    }


def probe(url: str, timeout: int = 30) -> dict[str, Any]:
    """One HEAD-shaped reachability probe. Never downloads, never retries blindly."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"url": url, "reachable": True, "status": response.status}
    except urllib.error.HTTPError as exc:
        return {"url": url, "reachable": True, "status": exc.code}
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return {"url": url, "reachable": False, "status": None, "error": str(reason)}


def source_provenance() -> dict[str, Any]:
    """The revision this record was generated at, and whether the tree was dirty.

    Recorded rather than assumed: P6's primary cells carry ``dirty: true`` and
    their fits are consequently not reconstructible from a clean checkout, which
    is the defect this checkpoint refuses to repeat.
    """

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=30, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    status = _git("status", "--porcelain")
    return {
        "revision": _git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": sorted(
            line.split(maxsplit=1)[-1] for line in status.splitlines() if line.strip()
        ),
    }


def refusal_record(probes: list[dict[str, Any]], symbol: str, note: str) -> dict[str, Any]:
    """The evidence behind a NOT EVALUABLE determination.

    Written rather than raised, because "the sources could not be obtained" is a
    claim a reviewer has to be able to check without re-running the network.
    """
    plan = plan_payload(symbol)
    unreachable = [p for p in probes if not p.get("reachable")]
    return {
        "refusal_schema": REFUSAL_SCHEMA,
        "checkpoint": prereg.CHECKPOINT,
        "preregistration_hash": prereg.preregistration_hash(),
        "provenance": source_provenance(),
        "outcome": "NOT EVALUABLE",
        "reason": note,
        "planned_object_count": plan["object_count"],
        "plan_digest": plan["plan_digest"],
        "probes": probes,
        "unreachable_count": len(unreachable),
        "hosts_refused": sorted({p["url"].split("/")[2] for p in unreachable}),
        "what_this_does_not_mean": (
            "this is not a negative economic result. No P13 return, funding total, basis "
            "figure or gate decision was computed, and none is estimated from anything that "
            "was reachable. The frozen design stays executable exactly as written."
        ),
        "what_was_not_done": [
            "no substitution of a different venue",
            "no substitution of a REST endpoint for the historical archive",
            "no synthetic or reconstructed perpetual or funding series",
            "no relaxation of the frozen source set to fit reachable data",
            "no P4-HOLD read, no Styx read",
        ],
    }


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    """Fetch ONE public archive object, from the one permitted host.

    The host check runs on every call rather than once at the top of a loop, so a
    URL assembled anywhere — a redirect followed by hand, a hard-coded mirror, a
    future edit — cannot reach the network without passing it. No credential, no
    header and no query string is added: these are public objects, and an
    authenticated request would be a different source.
    """
    assert_allowed_url(url)
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def download(symbol: str, cache_dir: Path, out: Path | None, timeout: int, quiet: bool) -> int:
    """Acquire every planned object, verify it, and write the manifest.

    Returns 0 on a COMPLETE acquisition and 2 on a refusal. There is no partial
    success: an incomplete acquisition writes no manifest, because a manifest is a
    statement that the frozen source set was obtained.
    """
    planned = plan_objects(symbol)
    plan = plan_payload(symbol)

    def report(index: int, total: int, obj) -> None:
        if quiet or index % 20 and index != total:
            return
        print(f"  {index}/{total}  {obj.field} {obj.period}", file=sys.stderr)

    try:
        acquired = acquire_all(
            planned,
            cache_dir,
            fetch=lambda url: fetch_bytes(url, timeout),
            progress=report,
        )
    except AcquisitionError as refusal:
        print(f"P13 ACQUISITION REFUSED: {refusal}", file=sys.stderr)
        return 2

    manifest = acquisition_manifest(
        acquired,
        symbol=symbol,
        plan_digest=plan["plan_digest"],
        planned_count=plan["object_count"],
        cache_dir=str(cache_dir),
        active_design=prereg.ACTIVE_DESIGN,
        preregistration_hash=prereg.preregistration_hash(),
        span_start_inclusive=plan["span_start_inclusive"],
        span_end_exclusive=plan["span_end_exclusive"],
        provenance=source_provenance(),
    )
    if not manifest["complete"]:  # pragma: no cover - acquire_all raises first
        print("P13 ACQUISITION INCOMPLETE; no manifest written", file=sys.stderr)
        return 2

    text = json.dumps(manifest, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"manifest written to {out}", file=sys.stderr)
    else:
        print(text)
    print(
        f"acquired and checksum-verified {manifest['acquired_object_count']} of "
        f"{manifest['planned_object_count']} objects, "
        f"{manifest['total_archive_bytes']} archive bytes",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true", help="networkless: print the plan")
    parser.add_argument(
        "--acquire", action="store_true", help="probe the archive host and fail closed"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch and checksum-verify every planned object; stops at the first failure",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="where raw archives are cached; required with --download and never committed",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--out", type=Path, default=None, help="where to write the record")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not (args.plan or args.acquire or args.download):
        parser.error("choose --plan (networkless), --acquire or --download")
    if args.download and args.cache_dir is None:
        parser.error("--download requires --cache-dir")

    if args.plan:
        payload = plan_payload(args.symbol)
        text = json.dumps(payload, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        else:
            print(text)
        return 0

    if args.download:
        return download(args.symbol, args.cache_dir, args.out, args.timeout, args.quiet)

    # --acquire: one probe per distinct source family, not per object. A host that
    # refuses the family refuses every month of it, and hammering it to prove that
    # would be noise rather than evidence.
    objects = plan_objects(args.symbol)
    seen: set[str] = set()
    probes: list[dict[str, Any]] = []
    for obj in objects:
        if obj.field in seen:
            continue
        seen.add(obj.field)
        probes.append({"field": obj.field, **probe(obj.url, args.timeout)})

    unreachable = [p for p in probes if not p.get("reachable")]
    if unreachable:
        record = refusal_record(
            probes,
            args.symbol,
            "the preregistered Binance archive host could not be reached from this "
            "environment; every required source family is unreachable",
        )
        text = json.dumps(record, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        else:
            print(text)
        print(
            f"P13 acquisition REFUSED: {len(unreachable)} of {len(probes)} source families "
            "unreachable. Recorded as NOT EVALUABLE; no economics computed.",
            file=sys.stderr,
        )
        return 2

    print(
        "all source families reachable; full acquisition is a separate step and is not "
        "performed by this probe",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
