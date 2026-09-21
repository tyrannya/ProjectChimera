"""Section 8.3's operator CLI for the demo runner.

Exactly six subcommands, as the plan lists them:

    run --config PATH [--replay FROM TO] [--allow-dirty] [--metrics-port PORT]
    status
    flatten --note TEXT
    resume --note TEXT
    resolve (--symbol S | --equity) --note TEXT
    report --day D

"Every operator command writes a decision-log record with `kind = OPERATOR`."

`resolve` clears one dispute, and which one is named rather than guessed:
``--symbol`` is a leg's reconciliation dispute, ``--equity`` the one a restart
raises when the persisted risk state and the carry ledger state different
equities (R1-b). One or the other, never both and never neither, so the command
cannot be run in the hope that it clears whatever it finds. Six subcommands
still, as section 8.3 lists them.

``--note`` is REQUIRED on `flatten`, `resume` and `resolve`, and a note that is
empty or only whitespace is refused. That matches
`FuturesExecutor.resolve_reconciliation`, which already refuses one: an operator
action with no stated reason is an unexplained change to a position, and the
decision log is the only place that reason will ever exist.

This tool builds a dry-run venue through `chimera.carry.factory` -- the one
place section 7.6 permits -- and reads the recorder's files read-only. It reads
no credential, and the only socket it can open is the Prometheus scrape endpoint
`run --metrics-port` asks for: a listening port that serves counters, reachable
from nothing the runner reads back.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

from chimera.carry.factory import build_hedged_position
from chimera.demo.config import ConfigProfile, parse_demo_config
from chimera.demo.inspection import inspect_demo_state
from chimera.demo.risk_wiring import build_risk_engine
from chimera.demo.rules import RuleRegistry
from chimera.demo.rules_carry import CarryParams, CarryRule
from chimera.demo.rules_shadow import DailyMomentumRule, FrozenLogisticRule, ShadowParams
from chimera.demo.runner import DemoRunner, RunnerError
from chimera.demo.telemetry import NullTelemetry
from chimera.futures.fills import RecordedQuoteFillModel
from chimera.recorder.contract import load_recorder_contract

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_HALTED = 3


def _require_note(value: str, command: str) -> str:
    note = (value or "").strip()
    if not note:
        raise SystemExit(
            f"{command} requires a non-empty --note. An operator action with no stated "
            "reason is an unexplained change, and this note is the only record of why"
        )
    return note


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo_run",
        description="Operator commands for the demo runner (section 8.3).",
    )
    parser.add_argument("--config", type=Path, help="campaign configuration JSON")
    parser.add_argument("--root", type=Path, help="the recorder's storage root")
    parser.add_argument(
        "--profile",
        default=ConfigProfile.CAMPAIGN.value,
        choices=[p.value for p in ConfigProfile],
        help="which profile the configuration must declare",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the tick loop")
    run.add_argument("--replay", nargs=2, metavar=("FROM", "TO"), type=int)
    run.add_argument(
        "--allow-dirty",
        action="store_true",
        help="soak runs only: skip the source-identity check",
    )
    run.add_argument("--max-minutes", type=int, default=None)
    run.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="serve Prometheus metrics on this port (default: do not serve)",
    )

    sub.add_parser("status", help="print the runner's state as JSON")

    flatten = sub.add_parser("flatten", help="reduce the position to flat")
    flatten.add_argument("--note", required=True)

    resume = sub.add_parser("resume", help="leave HALT")
    resume.add_argument("--note", required=True)

    resolve = sub.add_parser("resolve", help="clear one dispute")
    which = resolve.add_mutually_exclusive_group(required=True)
    which.add_argument("--symbol", help="a leg whose reconciliation is disputed")
    which.add_argument(
        "--equity",
        action="store_true",
        help="the restart equity dispute between risk.json and carry_ledger.json",
    )
    resolve.add_argument("--note", required=True)

    report = sub.add_parser("report", help="summarise one day's decision log")
    report.add_argument("--day", required=True)
    return parser


def _load(args: argparse.Namespace) -> DemoRunner:
    if args.config is None or args.root is None:
        raise SystemExit("--config and --root are required")
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = parse_demo_config(payload, expected_profile=ConfigProfile(args.profile))

    contract = load_recorder_contract("btcusdt-prospective-gen3")
    state_dir = Path(config.runner_setting("state_dir"))
    capital = Decimal("1000000")
    rules = RuleRegistry([CarryRule(CarryParams.from_config(config.rule_params("R1_carry")))])
    for rule_id, cls in (
        ("R2_frozen_logistic", FrozenLogisticRule),
        ("R3_daily_momentum", DailyMomentumRule),
    ):
        params = config.rule_params(rule_id)
        if params:
            rules.register(cls(ShadowParams.from_config(rule_id, params)))

    # Section 7.4's limits reach Aegis through `chimera.demo.risk_wiring` and
    # nowhere else. Building `RiskLimits` here is what let a campaign be hashed
    # under one risk regime and enforced under another; the mapping is one
    # audited function now, and `tests/test_demo_risk_wiring.py` holds this file
    # to using it.
    def _risk_factory(clock: Any, *, persist: bool) -> Any:
        return build_risk_engine(
            config, capital=capital, state_dir=state_dir, clock=clock, persist=persist
        )

    def _position_factory(risk: Any, clock: Any) -> Any:
        return build_hedged_position(
            risk=risk,
            capital=capital,
            state_dir=state_dir,
            fill_model=RecordedQuoteFillModel(),
            clock=clock,
        )

    def _inspection_factory():
        return inspect_demo_state(config, capital=capital, state_dir=state_dir)

    return DemoRunner(
        config,
        args.root,
        contract=contract,
        inspection_factory=_inspection_factory,
        risk_factory=_risk_factory,
        position_factory=_position_factory,
        rules=rules,
        capital=capital,
        software=_software(),
        # `status` is an inspection, not a process-liveness event. In
        # particular it must not move Prometheus state merely by being read.
        telemetry=NullTelemetry() if getattr(args, "command", None) == "status" else None,
    )


#: The checkout whose source `_software` identifies. `tools/` sits directly
#: under the repository root, so the root is this file's grandparent -- the same
#: derivation `nn/p2b.py` uses for the same call.
CHECKOUT_ROOT = Path(__file__).resolve().parent.parent


def _software(root: Path | None = None) -> dict[str, Any]:
    """The revision block section 9.1 stamps into every record.

    ``root`` is the checkout to identify and defaults to this one. It is a
    parameter because the only honest test of the CAMPAIGN self-check is
    against a *real* checkout that is genuinely clean and then genuinely dirty,
    and a test cannot dirty the checkout it is itself running from.

    The block is built from fail-closed values that a COMPLETE identification
    replaces, and never the other way round. `source_identity` REQUIRES its
    ``root`` and returns a MAPPING: calling it bare raised `TypeError` on every
    run, and reading the mapping with `getattr` then yielded ""/""/False -- so
    once the first half was fixed alone, a genuinely dirty tree would have
    reported itself CLEAN and a campaign would have run on source nobody could
    reconstruct. Both halves are load-bearing, which is why the keys below are
    read by key and why an incomplete identity does not reach the caller.
    """
    revision = ""
    digest = ""
    dirty = True
    try:
        from nn.source_identity import source_identity

        identity = source_identity(CHECKOUT_ROOT if root is None else Path(root))
        found_revision = identity.get("revision")
        found_digest = identity.get("source_digest")
        found_dirty = identity.get("dirty")
        # `revision` and `dirty` are `None` when git could not be asked, and
        # `bool(None)` is False: falling through on a partial answer would claim
        # a clean tree on the strength of a question nobody answered.
        if found_revision and found_digest and found_dirty is not None:
            revision = str(found_revision)
            digest = str(found_digest)
            dirty = bool(found_dirty)
    except Exception:
        # A runner that cannot establish its own revision says so rather than
        # claiming one; SELF_CHECK then refuses a campaign on the dirty flag.
        pass
    return {
        "revision": revision,
        "source_digest": digest,
        "dirty": dirty,
        "python": sys.version.split()[0],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "report":
        return _report(args)

    runner = _load(args)

    if args.command == "run":
        if args.metrics_port is not None:
            # Imported and called here rather than in `chimera/demo`, because
            # `serve_metrics` opens a listening socket and the whole demo
            # package is held to opening none (tests/test_demo_no_live_path.py,
            # and PR-10's "the runner never opens a socket"). The endpoint is
            # therefore the CLI's, exactly as it is in tools/recorder.py, and
            # nothing listens unless an operator asks for it.
            from chimera.metrics import serve_metrics

            serve_metrics(args.metrics_port)

        state = runner.start(allow_dirty=args.allow_dirty)
        if state.value == "HALT":
            print(json.dumps({"state": "HALT", "reason": runner.halt_reason}))
            return EXIT_HALTED
        if args.replay:
            runner.replay(args.replay[0], args.replay[1])
        else:
            runner.catch_up()
        outcome = runner.shutdown("cli run complete")
        print(json.dumps({"state": runner.state.value, "last_record": outcome.record_hash}))
        return EXIT_OK

    if args.command == "status":
        print(json.dumps(_status(runner), indent=2, sort_keys=True))
        return EXIT_OK

    if args.command == "flatten":
        note = _require_note(args.note, "flatten")
        runner.start(allow_dirty=True)
        outcome = runner.flatten(note)
        print(json.dumps({"flattened": True, "record": outcome.record_hash}))
        return EXIT_OK

    if args.command == "resume":
        note = _require_note(args.note, "resume")
        try:
            outcome = runner.resume(note)
        except RunnerError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_REFUSED
        print(json.dumps({"resumed": True, "record": outcome.record_hash}))
        return EXIT_OK

    if args.command == "resolve":
        note = _require_note(args.note, "resolve")
        if args.equity:
            # Started first, unlike the `--symbol` arm below. `resolve_equity`
            # reads Aegis, and Aegis does not exist until `start()` builds it --
            # which is also where the restart reconciliation runs and raises the
            # halt this settles, so there is nothing to settle before it. The
            # `--symbol` arm is left exactly as it was: making THAT one reachable
            # from a fresh process is canonical R1-i's item ("`resume` and
            # `resolve` succeed from the CLI via `start()` with log-tail clock
            # seeding"), and changing it here would be taking that decision.
            #
            # Deliberately without `allow_dirty`: a dirty tree makes `start()`
            # refuse and halt, but `RiskEngine.halt` keeps the FIRST reason, so
            # the dispute survives and can still be settled. An operator is never
            # locked out of a recovery path by the state of the working tree.
            try:
                runner.start()
                outcome = runner.resolve_equity(note)
            except RunnerError as exc:
                print(str(exc), file=sys.stderr)
                return EXIT_REFUSED
            print(
                json.dumps(
                    {
                        "resolved": "equity",
                        "state": runner.state.value,
                        "record": outcome.record_hash,
                    }
                )
            )
            return EXIT_OK
        try:
            outcome = runner.resolve(args.symbol, note)
        except RunnerError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_REFUSED
        print(json.dumps({"resolved": args.symbol, "record": outcome.record_hash}))
        return EXIT_OK

    raise SystemExit(f"unknown command {args.command!r}")  # pragma: no cover


def _status(runner: DemoRunner) -> dict[str, Any]:
    inspection = runner.inspection
    ledger = inspection.ledger.state
    return {
        "campaign_id": runner.config.campaign_id,
        "profile": runner.config.profile.value,
        "protocol_frozen": runner.config.protocol_frozen,
        "runner_state": runner.state.value,
        "halt_reason": runner.halt_reason,
        "hedge_state": inspection.hedge_state.value,
        "imbalance": str(inspection.imbalance),
        "last_minute_processed": runner.cursor.last_minute_processed,
        "last_record_hash": runner.last_record_hash,
        "risk_halted": inspection.risk_state.halted,
        # R1-c. Decided at construction from the read-only snapshot, so it is
        # available here for the same reason `ledger_may_speak` is: `status`
        # never calls `start()`, and a campaign that cannot start is exactly the
        # one an operator runs this command on. `outcome` is the verdict,
        # `reason` is the halt text a start would raise, and both come from the
        # files as found rather than from anything this process has written.
        "risk_continuity": {
            "outcome": runner.risk_continuity.outcome.value,
            "risk_state_load": runner.risk_continuity.load.value,
            "disputed": runner.risk_continuity.disputed,
            "reason": runner.risk_continuity.reason or None,
        },
        # The one read-only inspection command must not report a ledger the guard
        # has refused as if it were the campaign's. `_ledger_regression` is
        # decided in `__init__`, so it is available here without `start()` --
        # which matters, because this command never calls `start()`.
        "ledger_may_speak": runner._ledger_may_speak(),
        "ledger_complaint": (
            None if runner._ledger_may_speak() else runner._ledger_state_complaint()
        ),
        "ledger": {
            "fees": str(ledger.fees),
            "funding_received": str(ledger.funding_received),
            "funding_paid": str(ledger.funding_paid),
            "equity": str(ledger.last_equity) if ledger.last_equity is not None else None,
        },
        "disputed": inspection.ledger.disputed,
    }


def _report(args: argparse.Namespace) -> int:
    """Summarise one day's decision log. Counting only; PR-12 owns the report."""
    if args.config is None:
        raise SystemExit("--config is required to locate the state directory")
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = parse_demo_config(payload, expected_profile=ConfigProfile(args.profile))
    path = Path(config.runner_setting("state_dir")) / "decision_log" / f"{args.day}.ndjson"
    if not path.is_file():
        print(json.dumps({"day": args.day, "records": 0, "note": "no log for that day"}))
        return EXIT_OK
    kinds: dict[str, int] = {}
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        kind = json.loads(line).get("kind", "?")
        kinds[kind] = kinds.get(kind, 0) + 1
    print(json.dumps({"day": args.day, "records": total, "kinds": kinds}, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
