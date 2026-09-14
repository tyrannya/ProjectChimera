"""Command line for the prospective recorder, from collection to the coverage gate.

Six commands, and each of them says which side of the network it is on.

``run`` starts the collector against the public Binance market-data endpoints
and writes under a storage root you name. ``reconcile`` fetches the venue's
published daily and monthly archives for one UTC day, verifies them against the
digests the venue publishes beside them, compares them with what was recorded,
and writes that day's reconciliation record. Those two reach the network and
nothing else here does.

``status``, ``coverage`` and ``verify-day`` read what is already on disk.
``coverage`` recomputes the 30-day gate from the reconciliation records every
time it is asked — there is no streak file anywhere — and writes
``coverage/GATE.json``. ``verify-day`` re-checks the checksums a day's own
manifests and metadata claim, and repairs nothing. ``freeze-day`` closes a day:
it compresses and checksums the raw files and writes the normalized day's
digest, after which that day is immutable.

**Nothing here decides anything is a repair.** ``reconcile`` never writes into
the recorder's raw, normalized or settlement files: a disagreement with the
archive is a finding, it goes into the record, and the recorder's own bytes are
left exactly as they were recorded.

**Nothing here starts a prospective clock.** The committed gen3 contract carries
``prospective_from: null`` and this tool never writes it. Every minute recorded
by ``run`` is engineering data until that boundary is committed in a reviewed
pull request, every command says so in its output rather than leaving it to be
inferred, and ``coverage`` reports ``BOUNDARY_UNSET`` rather than a pass however
long the streak of engineering days becomes.

**No credentials.** The recorder reads public market data. This module reads no
environment variable holding a secret, takes no key argument, and passes none to
anything it constructs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from chimera.recorder.contract import (
    GEN3_CONTRACT_ID,
    RecorderContractError,
    available_recorder_contract_ids,
    load_recorder_contract,
)
from chimera.recorder.coverage import (
    DEFAULT_WINDOW,
    RecorderCoverageError,
    coverage_for_day,
    gate,
    reconciliation_path,
    summarise,
    write_gate,
)
from chimera.recorder.health import RecorderHealthError, read_status
from chimera.recorder.normalize import MinuteNormalizer, RecorderNormalizeError
from chimera.recorder.service import RecorderService, RecorderServiceError, build_service
from chimera.recorder.sink import RawSink, RecorderSinkError

logger = logging.getLogger("chimera.recorder")

#: Where a recording goes unless ``--root`` says otherwise. Under
#: ``data/prospective/`` — which ``.gitignore`` excludes — so that engineering
#: data cannot be committed by accident.
DEFAULT_BASE_DIR = Path("data")

#: Exit codes. Distinct so a supervisor can tell "the operator asked for
#: something impossible" from "the recorder failed while running".
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.recorder",
        description=(
            "Record public Binance market data under a committed recorder contract, "
            "reconcile a recorded day against the venue's published archives, and "
            "recompute the coverage gate from those records."
        ),
    )
    parser.add_argument(
        "--contract",
        default=GEN3_CONTRACT_ID,
        help=f"recorder contract id (default: {GEN3_CONTRACT_ID})",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=(
            "directory the contract's storage root is resolved under "
            f"(default: {DEFAULT_BASE_DIR})"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="logging verbosity (default: INFO)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="record until interrupted, or for a bounded number of seconds",
        description=(
            "Connect to the public market-data streams, record, normalize the open day "
            "and write a heartbeat. Stops cleanly on SIGINT and SIGTERM."
        ),
    )
    run.add_argument(
        "--seconds",
        type=float,
        default=None,
        help=(
            "stop after this many seconds instead of running until interrupted; "
            "used for the reviewer's bounded acceptance run"
        ),
    )
    run.add_argument(
        "--no-gapfill",
        action="store_true",
        help="do not REST gap-fill on start (the websocket streams still run)",
    )
    run.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="serve Prometheus metrics on this port (default: do not serve)",
    )
    run.add_argument(
        "--json",
        action="store_true",
        help="print the run's result as JSON on exit",
    )

    status = commands.add_parser(
        "status",
        help="print what is on disk; reads only, writes nothing",
        description=(
            "Report the contract in force, the prospective boundary, the last heartbeat "
            "and the days recorded. Opens no socket and modifies no file."
        ),
    )
    status.add_argument("--json", action="store_true", help="print the report as JSON")

    reconcile = commands.add_parser(
        "reconcile",
        help="compare one recorded UTC day against the venue's published archives",
        description=(
            "Fetch the first-party daily and monthly archives covering one UTC day, verify "
            "each against its published checksum companion, compare them minute by minute "
            "with the normalized day, and write reconciliation/<day>.json. The record is "
            "written whatever the archives turned out to be, because an object that could "
            "not be established is itself the finding; whether the day passes is what "
            "coverage then reads out of it. The recorder's own files are never modified: a "
            "disagreement is a finding, not a repair."
        ),
    )
    reconcile.add_argument(
        "--day", required=True, help="the UTC day to reconcile (YYYY-MM-DD)"
    )
    reconcile.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "keep fetched archive objects here, keyed by their full archive path and "
            "re-verified against the published digest on every read (default: fetch each "
            "time)"
        ),
    )
    reconcile.add_argument(
        "--no-index-diagnostic",
        action="store_true",
        help=(
            "skip the index-price comparison, which is a diagnostic and gates nothing "
            "either way"
        ),
    )
    reconcile.add_argument("--json", action="store_true", help="print the record as JSON")

    coverage = commands.add_parser(
        "coverage",
        help="recompute the coverage gate from the reconciliation records",
        description=(
            "Recompute every day's verdict from its reconciliation record, find the current "
            "streak of consecutive passing UTC days, and write coverage/GATE.json. Opens no "
            "socket. While the contract's prospective_from is null the verdict is "
            "BOUNDARY_UNSET and is never a pass."
        ),
    )
    coverage.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"consecutive passing days the gate requires (default: {DEFAULT_WINDOW})",
    )
    coverage.add_argument("--json", action="store_true", help="print the verdict as JSON")

    verify_day = commands.add_parser(
        "verify-day",
        help="re-check one day's checksums against the files on disk",
        description=(
            "Recompute the digests the day's raw manifests, normalized metadata and "
            "settlements file claim, and report every disagreement. Reads only, writes "
            "nothing and repairs nothing."
        ),
    )
    verify_day.add_argument("--day", required=True, help="the UTC day to verify (YYYY-MM-DD)")
    verify_day.add_argument("--json", action="store_true", help="print the report as JSON")

    freeze_day = commands.add_parser(
        "freeze-day",
        help="close one day for ever: compress, checksum, write the manifests",
        description=(
            "Freeze every stream's raw file and every market's normalized day for one UTC "
            "day. A frozen day is immutable: a correction afterwards is a new file with a "
            "note, never an overwrite."
        ),
    )
    freeze_day.add_argument("--day", required=True, help="the UTC day to freeze (YYYY-MM-DD)")
    freeze_day.add_argument("--json", action="store_true", help="print the result as JSON")
    return parser


def _load(contract_id: str) -> Any:
    try:
        return load_recorder_contract(contract_id)
    except RecorderContractError as exc:
        known = ", ".join(available_recorder_contract_ids()) or "none"
        raise SystemExit(f"{exc}\ncommitted recorder contracts: {known}") from exc


def _source_revision() -> str | None:
    """The recorder's source identity, where the repository can supply it.

    Imported here rather than in :mod:`chimera.recorder` on purpose: the
    recorder package must not depend on research code, and a test asserts it
    does not. The CLI is allowed to, and stamps the revision onto the heartbeat
    so a running process can be traced back to a commit.
    """
    try:
        from nn.source_identity import source_identity
    except ImportError:  # pragma: no cover - the recorder runs without it
        return None
    try:
        identity = source_identity(Path(__file__).resolve().parents[1])
    except Exception:  # pragma: no cover - a checkout without git metadata
        return None
    revision = identity.get("revision")
    if not revision:
        return None
    return f"{revision}{'+dirty' if identity.get('dirty') else ''}"


def command_status(args: argparse.Namespace) -> int:
    contract = _load(args.contract)
    root = contract.storage_root(args.base_dir)
    try:
        report = read_status(contract, root, now_ns=time.time_ns())
    except RecorderHealthError as exc:
        print(f"status unavailable: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return EXIT_OK
    _print_status(report)
    return EXIT_OK


def _print_status(report: dict[str, Any]) -> None:
    print(f"contract      {report['contract_id']}  {report['contract_hash']}")
    boundary = report["prospective_from"] or "not set"
    print(f"boundary      prospective_from={boundary}  ({report['evidence_class']} data)")
    print(f"root          {report['root']}  {'present' if report['exists'] else 'absent'}")
    heartbeat = report["heartbeat"]
    if heartbeat is None:
        print("heartbeat     none written")
    else:
        age = report["heartbeat_age_seconds"]
        age_text = "unknown" if age is None else f"{age:.0f}s ago"
        print(
            f"heartbeat     {heartbeat.get('heartbeat_utc')}  ({age_text})  "
            f"write_errors={heartbeat.get('write_errors')}"
        )
    halted = (heartbeat or {}).get("halted_streams") or []
    if halted:
        print(f"HALTED        {', '.join(halted)} — storage failed; recording nothing")
    for stream in report["streams"]:
        days = stream["days"]
        last = stream["last_day"] or "-"
        print(
            f"  {stream['stream']:<18} days={len(days):<4} last={last:<12} "
            f"frozen={len(stream['frozen_days'])}"
        )
    for market in report["markets"]:
        print(
            f"  {market['market']:<18} normalized={len(market['normalized_days']):<4} "
            f"last={market['last_day'] or '-':<12} rows={market['rows']} "
            f"missing={market['missing_minutes']}"
        )
    print(f"settlements   {report['settlements_rows']} row(s)")
    free = report["disk_free_bytes"]
    if free is not None:
        print(f"disk free     {free / (1 << 30):.1f} GiB")


def command_run(args: argparse.Namespace) -> int:
    contract = _load(args.contract)
    if args.metrics_port is not None:
        from chimera.metrics import serve_metrics

        serve_metrics(args.metrics_port)
    service = build_service(
        contract,
        args.base_dir,
        gapfill=not args.no_gapfill,
        source_revision=_source_revision(),
    )
    print(
        f"recording {contract.contract_id} into {service.root}\n"
        f"  contract hash     {contract.contract_hash}\n"
        f"  prospective_from  {contract.prospective_from or 'null'}  "
        f"({'prospective' if contract.activated else 'engineering'} data)\n"
        f"  streams           {', '.join(contract.streams)}",
        flush=True,
    )
    try:
        result = asyncio.run(_run(service, args.seconds))
    except RecorderServiceError as exc:
        print(f"recorder failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"stopped after {result.seconds:.0f}s: events={result.events} "
            f"duplicates={result.duplicates} late={result.late} "
            f"gapfill={result.gapfill_rows} reconnects={result.reconnects} "
            f"heartbeats={result.heartbeats} write_errors={result.write_errors}"
        )
        for note in result.errors:
            print(f"  ! {note}")
    return EXIT_FAILED if result.write_errors else EXIT_OK


def command_reconcile(args: argparse.Namespace) -> int:
    """Fetch the day's archives, compare, and persist the record.

    The acquisition module is imported here rather than at the top of the file
    on purpose: it is the one thing in this tool that speaks to a network, and a
    reader running ``status``, ``coverage`` or ``verify-day`` should not have an
    HTTP stack imported into their process to do it.
    """
    from chimera.recorder.reconcile import (
        ArchiveCache,
        HttpsArchiveFetcher,
        RecorderReconcileError,
        reconcile_day,
        write_reconciliation,
    )

    contract = _load(args.contract)
    root = contract.storage_root(args.base_dir)
    cache = None if args.cache_dir is None else ArchiveCache(args.cache_dir)
    try:
        report = reconcile_day(
            root,
            args.day,
            HttpsArchiveFetcher(),
            contract=contract,
            cache=cache,
            index_diagnostic=not args.no_index_diagnostic,
        )
        path = write_reconciliation(root, report)
    except RecorderReconcileError as exc:
        print(f"reconciliation failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    document = report.to_dict()
    if args.json:
        print(json.dumps(document, indent=2, sort_keys=True))
        return EXIT_OK
    print(f"contract      {contract.contract_id}  {contract.contract_hash}")
    print(f"day           {report.day}  ({document['evidence_class']} data)")
    print(f"record        {path}")
    for stream, entry in sorted(document["streams"].items()):
        if entry["judged"]:
            print(
                f"  {stream:<18} published={entry['published_minutes']:<5} "
                f"agreeing={entry['agreeing_minutes']:<5} "
                f"disagreeing={entry['disagreeing_minutes']:<4} "
                f"archive_only={len(entry['archive_only_minutes'])}"
            )
        else:
            print(f"  {stream:<18} NOT JUDGED  {entry['reason']}")
    funding = document["funding"]
    if funding["schedule_established"]:
        print(
            f"  {'um.funding':<18} scheduled={funding['scheduled']} "
            f"captured={funding['captured']} complete={funding['funding_complete']}"
        )
    else:
        why = funding["reason"] or funding["archive"]["detail"]
        print(f"  {'um.funding':<18} {funding['outcome']}  {why}")
    for name in sorted(document["diagnostics"]):
        print(f"  {name:<18} diagnostic only; gates nothing")
    return EXIT_OK


def command_coverage(args: argparse.Namespace) -> int:
    contract = _load(args.contract)
    root = contract.storage_root(args.base_dir)
    try:
        verdict = gate(root, args.window, contract=contract)
        path = write_gate(root, verdict)
    except RecorderCoverageError as exc:
        print(f"coverage unavailable: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if args.json:
        print(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
        return EXIT_OK
    for line in summarise(verdict):
        print(line)
    print(f"verdict file  {path}")
    return EXIT_OK


def _verify_digest(path: Path, expected: str | None, label: str, findings: list[str]) -> None:
    """One recorded digest, recomputed. A file that is absent is itself a finding."""
    if expected is None:
        return
    if not path.exists():
        findings.append(f"{label}: {path} is named by a manifest and is not on disk")
        return
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        findings.append(
            f"{label}: {path} hashes to {actual}, and its manifest says {expected}"
        )


def command_verify_day(args: argparse.Namespace) -> int:
    """Recompute what one day's own manifests claim about it. Read-only.

    Deliberately not a reconciliation: this asks whether the files on disk are
    the files the day's manifests describe, which is a question about this host.
    Whether the recorder captured what the venue published is a different
    question with a different answer, and ``reconcile`` is where it is asked.
    """
    contract = _load(args.contract)
    root = contract.storage_root(args.base_dir)
    findings: list[str] = []
    checked = 0
    report: dict[str, Any] = {"day": args.day, "root": str(root), "streams": {}, "markets": {}}
    for stream in contract.streams:
        sink = RawSink(root, stream, contract=contract)
        manifest_path = sink.manifest_path(args.day)
        if not manifest_path.exists():
            report["streams"][stream] = "not frozen"
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            findings.append(f"{stream}: {manifest_path} is unreadable: {exc}")
            continue
        raw = manifest.get("raw") or {}
        _verify_digest(root / raw.get("path", ""), raw.get("sha256_gz"), stream, findings)
        late = manifest.get("late")
        if late:
            _verify_digest(
                root / late.get("path", ""), late.get("sha256"), f"{stream} late", findings
            )
        checked += 1
        report["streams"][stream] = {
            "rows": raw.get("rows"),
            "sha256_gz": raw.get("sha256_gz"),
        }

    normalizer = MinuteNormalizer(root, contract)
    for market in contract.market_keys():
        meta_path = normalizer.meta_path(market, args.day)
        if not meta_path.exists():
            report["markets"][market] = "not normalized"
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            findings.append(f"{market}: {meta_path} is unreadable: {exc}")
            continue
        _verify_digest(
            normalizer.parquet_path(market, args.day),
            meta.get("parquet_sha256"),
            f"{market} parquet",
            findings,
        )
        checked += 1
        report["markets"][market] = {"rows": meta.get("rows"), "digest": meta.get("digest")}
        settlements = normalizer.settlements_path(market)
        digest_file = normalizer.settlements_digest_path(market)
        if settlements.exists() and digest_file.exists():
            recorded = digest_file.read_text(encoding="utf-8").split()
            _verify_digest(
                settlements,
                recorded[0] if recorded else None,
                f"{market} settlements",
                findings,
            )

    record = reconciliation_path(root, args.day)
    if record.exists():
        try:
            coverage = coverage_for_day(root, args.day, contract=contract)
            report["coverage"] = coverage.to_dict()
        except RecorderCoverageError as exc:
            findings.append(f"reconciliation: {exc}")
    else:
        report["coverage"] = None
    report["findings"] = findings
    report["checked"] = checked
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return EXIT_OK if not findings else EXIT_FAILED
    print(f"contract      {contract.contract_id}  {contract.contract_hash}")
    print(f"day           {args.day}")
    print(f"checked       {checked} manifest(s) and metadata document(s)")
    coverage_block = report.get("coverage")
    if coverage_block is None:
        print("coverage      no reconciliation record for this day")
    else:
        print(f"coverage      {coverage_block['verdict']}")
    for finding in findings:
        print(f"  ! {finding}")
    if not findings:
        print("every recorded digest matches the bytes on disk")
    return EXIT_OK if not findings else EXIT_FAILED


def command_freeze_day(args: argparse.Namespace) -> int:
    """Close one day. Streams first, then the normalized markets built from them."""
    contract = _load(args.contract)
    root = contract.storage_root(args.base_dir)
    frozen: list[str] = []
    skipped: list[str] = []
    try:
        for stream in contract.streams:
            sink = RawSink(root, stream, contract=contract)
            if sink.manifest_path(args.day).exists():
                skipped.append(f"{stream}: already frozen")
                continue
            if not sink.day_dir(args.day).is_dir():
                skipped.append(f"{stream}: nothing recorded on this day")
                continue
            frozen.append(str(sink.freeze_day(args.day, provenance=_provenance())))
        normalizer = MinuteNormalizer(root, contract)
        for market in contract.market_keys():
            if normalizer.is_frozen(market, args.day):
                skipped.append(f"{market}: already frozen")
                continue
            if not normalizer.parquet_path(market, args.day).exists():
                skipped.append(f"{market}: no normalized day to freeze")
                continue
            frozen.append(str(normalizer.freeze_day(market, args.day)))
    except (RecorderSinkError, RecorderNormalizeError) as exc:
        print(f"freeze failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if args.json:
        print(json.dumps({"day": args.day, "frozen": frozen, "skipped": skipped}, indent=2))
        return EXIT_OK
    print(f"day           {args.day}")
    for path in frozen:
        print(f"  froze  {path}")
    for note in skipped:
        print(f"  -      {note}")
    return EXIT_OK


def _provenance() -> dict[str, Any]:
    """What is stamped onto a manifest this tool writes: the source revision only."""
    revision = _source_revision()
    return {} if revision is None else {"source_revision": revision}


async def _run(service: RecorderService, seconds: float | None) -> Any:
    """Run the service until a signal, or until ``seconds`` have passed."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            loop.add_signal_handler(signum, stop.set)
        except (NotImplementedError, RuntimeError):
            # Windows has no add_signal_handler for SIGTERM; KeyboardInterrupt
            # still unwinds through asyncio.run and the finally-block shutdown.
            pass
    timer = None
    if seconds is not None:
        if seconds <= 0:
            raise SystemExit("--seconds must be positive")
        timer = loop.call_later(seconds, stop.set)
    try:
        return await service.run(stop)
    finally:
        if timer is not None:
            timer.cancel()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    handlers = {
        "status": command_status,
        "run": command_run,
        "reconcile": command_reconcile,
        "coverage": command_coverage,
        "verify-day": command_verify_day,
        "freeze-day": command_freeze_day,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"unknown command {args.command!r}")
        return EXIT_USAGE
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
