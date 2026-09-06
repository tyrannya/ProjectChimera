"""Section 8.3's operator CLI for the demo runner.

Exactly six subcommands, as the plan lists them:

    run --config PATH [--replay FROM TO] [--allow-dirty] [--metrics-port PORT]
    status
    flatten --note TEXT
    resume --note TEXT
    resolve --symbol S --note TEXT
    report --day D

"Every operator command writes a decision-log record with `kind = OPERATOR`."

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
from chimera.demo.rules import RuleRegistry
from chimera.demo.rules_carry import CarryParams, CarryRule
from chimera.demo.rules_shadow import DailyMomentumRule, FrozenLogisticRule, ShadowParams
from chimera.demo.runner import DemoRunner, RunnerError
from chimera.futures.fills import RecordedQuoteFillModel
from chimera.recorder.contract import load_recorder_contract
from chimera.risk import RiskEngine, RiskLimits

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

    resolve = sub.add_parser("resolve", help="clear a reconciliation dispute")
    resolve.add_argument("--symbol", required=True)
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
    limits = config.limits
    risk = RiskEngine(
        RiskLimits(
            max_position_pct=float(limits.max_exposure_per_asset_pct),
            risk_per_trade_pct=0.5,
        ),
        state_path=state_dir / "risk.json",
        kill_switch_path=state_dir / "KILL_SWITCH",
    )
    capital = Decimal("1000000")
    risk.update_equity(float(capital))
    position = build_hedged_position(
        risk=risk,
        capital=capital,
        state_dir=state_dir,
        fill_model=RecordedQuoteFillModel(),
    )
    rules = RuleRegistry([CarryRule(CarryParams.from_config(config.rule_params("R1_carry")))])
    for rule_id, cls in (
        ("R2_frozen_logistic", FrozenLogisticRule),
        ("R3_daily_momentum", DailyMomentumRule),
    ):
        params = config.rule_params(rule_id)
        if params:
            rules.register(cls(ShadowParams.from_config(rule_id, params)))
    return DemoRunner(
        config,
        args.root,
        contract=contract,
        risk=risk,
        position=position,
        rules=rules,
        capital=capital,
        software=_software(),
    )


def _software() -> dict[str, Any]:
    """The revision block section 9.1 stamps into every record."""
    try:
        from nn.source_identity import source_identity

        identity = source_identity()
        return {
            "revision": getattr(identity, "revision", ""),
            "source_digest": getattr(identity, "source_digest", ""),
            "dirty": bool(getattr(identity, "dirty", False)),
            "python": sys.version.split()[0],
        }
    except Exception:
        # A runner that cannot establish its own revision says so rather than
        # claiming one; SELF_CHECK then refuses a campaign on the dirty flag.
        return {
            "revision": "",
            "source_digest": "",
            "dirty": True,
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
        try:
            outcome = runner.resolve(args.symbol, note)
        except RunnerError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_REFUSED
        print(json.dumps({"resolved": args.symbol, "record": outcome.record_hash}))
        return EXIT_OK

    raise SystemExit(f"unknown command {args.command!r}")  # pragma: no cover


def _status(runner: DemoRunner) -> dict[str, Any]:
    ledger = runner.position.ledger.state
    return {
        "campaign_id": runner.config.campaign_id,
        "profile": runner.config.profile.value,
        "protocol_frozen": runner.config.protocol_frozen,
        "runner_state": runner.state.value,
        "halt_reason": runner.halt_reason,
        "hedge_state": runner.position.state.value,
        "imbalance": str(runner.position.imbalance()),
        "last_minute_processed": runner.cursor.last_minute_processed,
        "last_record_hash": runner.last_record_hash,
        "risk_halted": runner.risk.state.halted,
        "ledger": {
            "fees": str(ledger.fees),
            "funding_received": str(ledger.funding_received),
            "funding_paid": str(ledger.funding_paid),
            "equity": str(ledger.last_equity) if ledger.last_equity is not None else None,
        },
        "disputed": runner.position.ledger.disputed,
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
