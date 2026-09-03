"""Command line for the prospective recorder: ``run`` and ``status``.

Two commands, and deliberately only two.

``run`` starts the collector against the public Binance market-data endpoints
and writes under a storage root you name. ``status`` reads what is already on
disk and prints it; it opens no socket, starts no task and writes no file.

**What is not here, and why.** Section 12.3 sketches a longer-term CLI with
``reconcile``, ``coverage``, ``verify-day`` and ``freeze-day``. Those belong to
PR-06 — the archive reconciliation and the 30-day coverage gate — and they are
absent rather than stubbed. A subcommand that parsed its arguments and then said
"not implemented" would appear in ``--help`` as a capability this build does not
have, and the first thing a reader would conclude from seeing ``coverage`` in
the help text is that coverage can be computed. It cannot, yet.

**Nothing here starts a prospective clock.** The committed gen3 contract carries
``prospective_from: null`` and this tool never writes it. Every minute recorded
by ``run`` is engineering data until that boundary is committed in a reviewed
pull request, and both commands say so in their output rather than leaving it to
be inferred.

**No credentials.** The recorder reads public market data. This module reads no
environment variable holding a secret, takes no key argument, and passes none to
anything it constructs.
"""

from __future__ import annotations

import argparse
import asyncio
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
from chimera.recorder.health import RecorderHealthError, read_status
from chimera.recorder.service import RecorderService, RecorderServiceError, build_service

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
            "Record public Binance market data under a committed recorder contract. "
            "Reconciliation and the coverage gate are not part of this build."
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
    except Exception:  # pragma: no cover - the recorder runs without it
        return None
    try:
        identity = source_identity()
    except Exception:  # pragma: no cover - a checkout without git metadata
        return None
    revision = getattr(identity, "revision", None)
    dirty = getattr(identity, "dirty", None)
    if revision is None:
        return None
    return f"{revision}{'+dirty' if dirty else ''}"


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
    if args.command == "status":
        return command_status(args)
    if args.command == "run":
        return command_run(args)
    parser.error(f"unknown command {args.command!r}")
    return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
