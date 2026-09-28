"""R1-i: dispute lifecycle, operator commands through start(), correction,
per-leg liquidation, persistence failure and the emergency record.

The adopted roadmap's item, every clause of which is load-bearing:

    every dispute kind has exactly one clearing path (`resolve` reachable from
    the CLI, requires a note, re-books what it must, refuses what it cannot);
    `resume` and `resolve` succeed from the CLI via `start()` with log-tail
    clock seeding; `flatten` on CAMPAIGN without `allow_dirty` and without
    persisting a spurious halt reason; `HedgedPosition.correct()` wired with a
    bounded timeout or deleted; one-legged positions liquidation-checked per
    leg; `resume` ordering fixed; the state-machine crash harness ... added to
    CI, including a disk-full fault; on any persistence failure (ENOSPC
    included) the runner exits non-zero after attempting to append the halt
    reason to a pre-allocated emergency record, so a restart never lacks a
    trace.

The crash harness is ``tests/test_r1i_crash_harness.py``. This module holds the
semantic tests, each from both sides.

The CLI tests stub `demo_run._software` to a CLEAN identity: the source-identity
check has its own real-checkout tests (``tests/test_demo_cli.py``), and these
must not depend on whether the checkout running them has uncommitted edits.
"""

from __future__ import annotations

import errno
import json
import re
from argparse import Namespace
from decimal import Decimal
from pathlib import Path

import pytest

import tools.demo_run as demo_run
from chimera.carry.hedge import HedgeState
from chimera.carry.ledger import LEDGER_SCHEMA, LEDGER_SCHEMAS_READ, CarryLedger
from chimera.demo import emergency
from chimera.demo.decision_log import RecordKind, verify_log
from chimera.demo.runner import (
    LEDGER_DISPUTES,
    REBOOK,
    RecoveryCause,
    RunnerError,
    RunnerState,
)
from chimera.futures.executor import FlattenCause
from chimera.persistence import PersistenceFailure

from demo_harness import build
from test_demo_cli import written_config

D = Decimal
NOTE = "operator: the files the runbook names were checked"
REPO = Path(__file__).resolve().parents[1]
CLEAN = {"revision": "a" * 40, "source_digest": "b" * 64, "dirty": False, "python": "3.11"}


@pytest.fixture
def clean_identity(monkeypatch):
    monkeypatch.setattr(demo_run, "_software", lambda root=None: dict(CLEAN))


def records(state_dir: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted((state_dir / "decision_log").glob("*.ndjson")):
        out.extend(json.loads(line) for line in path.read_text("utf-8").splitlines() if line)
    return out


def argv_for(tmp_path: Path, harness) -> list[str]:
    config = written_config(tmp_path, harness)
    return ["--config", str(config), "--root", str(harness.root), "--profile", "TEST"]


def restarted(tmp_path: Path, harness, **kwargs):
    return build(tmp_path, config=harness.runner.config, start=False, **kwargs)


def halted_by_the_switch(tmp_path: Path):
    """Ten minutes, then an ordinary kill-switch halt; the switch is then removed."""
    harness = build(tmp_path)
    harness.run(10)
    (harness.state_dir / "KILL_SWITCH").write_text("drill\n", encoding="utf-8")
    harness.tick(harness.first_minute_ms() + 10 * 60_000)
    assert harness.runner.state is RunnerState.HALT
    harness.runner.shutdown("halted")
    (harness.state_dir / "KILL_SWITCH").unlink()
    return harness


def monotone(state_dir: Path) -> bool:
    stamps = [record["runner_now_ns"] for record in records(state_dir)]
    return stamps == sorted(stamps)


# ---------------------------------------------------------------------------
# `resume` and `resolve` from the CLI, via start(), with log-tail clock seeding
# ---------------------------------------------------------------------------
def test_resume_from_a_fresh_cli_process_goes_through_start(tmp_path, clean_identity, capsys):
    harness = halted_by_the_switch(tmp_path)
    argv = argv_for(tmp_path, harness)
    before = len(records(harness.state_dir))

    assert demo_run.main(argv + ["resume", "--note", NOTE]) == demo_run.EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["resumed"] is True and out["state"] == RunnerState.READY.value

    added = records(harness.state_dir)[before:]
    kinds = [r["kind"] for r in added]
    # STARTUP (start()), HALT (start() found the persisted halt), the request,
    # then the RESUME that completes it.
    assert kinds[0] == RecordKind.STARTUP.value
    assert kinds[-2:] == [RecordKind.OPERATOR.value, RecordKind.RESUME.value]
    assert added[-2]["operator"]["phase"] == "requested"
    assert added[-1]["operator"]["request_seq"] == added[-2]["seq"]
    assert monotone(harness.state_dir)
    assert verify_log(harness.state_dir / "decision_log").ok
    assert json.loads((harness.state_dir / "risk.json").read_text("utf-8"))["halted"] is False

    # And the campaign runs on from it.
    assert demo_run.main(argv + ["run", "--once"]) == demo_run.EXIT_OK


def test_resume_without_start_is_refused_by_the_runner_it_would_have_built(tmp_path):
    """The control: the pre-R1-i CLI built a runner and called `resume` on it
    without `start()`, and the runner -- still in STARTUP, Aegis not built --
    refused. The same runner, started, resumes."""
    harness = halted_by_the_switch(tmp_path)
    cold = restarted(tmp_path, harness).runner
    with pytest.raises(RunnerError):
        cold.resume(NOTE)
    warm = restarted(tmp_path, harness).runner
    assert warm.start() is RunnerState.HALT
    warm.resume(NOTE)
    assert warm.state is RunnerState.READY


def _reconciliation_dispute(tmp_path: Path):
    """A REAL reconciliation mismatch: the dry-run venue reports a different
    perpetual quantity for one hourly reconciliation."""
    harness = build(tmp_path)
    harness.run(59)
    perp = harness.runner.position.perp
    symbol = harness.runner.position.config.perp_symbol
    real = perp.venue.reported_position

    def once(sym):
        held = real(sym)
        if sym != symbol:
            return held
        perp.venue.reported_position = real
        return type(held)(
            symbol=held.symbol,
            side=held.side,
            quantity=held.quantity + D("0.001"),
            entry_price=held.entry_price,
            leverage=held.leverage,
            margin_mode=held.margin_mode,
        )

    perp.venue.reported_position = once
    outcome = harness.tick(harness.first_minute_ms() + 60 * 60_000)
    assert outcome.kind is RecordKind.HALT, outcome
    harness.runner.shutdown("halted on the dispute")
    return harness, symbol


def test_resolve_symbol_from_a_fresh_cli_process_goes_through_start(
    tmp_path, clean_identity, capsys
):
    harness, symbol = _reconciliation_dispute(tmp_path)
    argv = argv_for(tmp_path, harness)

    # `resume` refuses: the dispute has its own clearing path, and names it.
    assert demo_run.main(argv + ["resume", "--note", NOTE]) == demo_run.EXIT_REFUSED
    assert f"resolve --symbol {symbol}" in capsys.readouterr().err

    assert demo_run.main(argv + ["resolve", "--symbol", symbol, "--note", NOTE]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["resolved"] == symbol and resolved["still_disputed"] == []
    assert demo_run.main(argv + ["resume", "--note", NOTE]) == demo_run.EXIT_OK
    capsys.readouterr()
    assert demo_run.main(argv + ["run", "--once"]) == demo_run.EXIT_OK
    assert monotone(harness.state_dir)
    store = json.loads((harness.state_dir / "perp_store.json").read_text("utf-8"))
    assert store["disputed"] == {}
    assert (
        json.loads((harness.state_dir / "risk.json").read_text("utf-8"))[
            "reconciliation_disputed"
        ]
        == {}
    )


def test_the_clock_is_seeded_from_the_verified_log_tail(tmp_path):
    """Two-sided. A runner state restored from five minutes earlier puts the
    cursor's seed behind the log's tail. Seeded from the tail, start() writes
    records that never move the clock backwards; seeded from the cursor alone,
    its very first record would be refused as moving the clock backwards."""
    harness = build(tmp_path)
    harness.run(10)
    old = (harness.state_dir / "runner_state.json").read_bytes()
    harness.run(5, start=harness.first_minute_ms() + 10 * 60_000)
    harness.runner.shutdown("stop")
    (harness.state_dir / "runner_state.json").write_bytes(old)
    tail = records(harness.state_dir)[-1]["runner_now_ns"]

    again = restarted(tmp_path, harness).runner
    assert again.start() is RunnerState.READY
    assert again.clock.now_ns >= tail
    assert monotone(harness.state_dir)
    assert verify_log(harness.state_dir / "decision_log").ok


def test_without_the_tail_seed_the_same_start_would_move_the_clock_backwards(
    tmp_path, monkeypatch
):
    """The negative control for the test above: the seed removed, the same
    restored state makes start() fail on its first record."""
    from chimera.demo import runner as runner_module
    from chimera.demo.decision_log import DecisionLogError

    harness = build(tmp_path)
    harness.run(10)
    old = (harness.state_dir / "runner_state.json").read_bytes()
    harness.run(5, start=harness.first_minute_ms() + 10 * 60_000)
    harness.runner.shutdown("stop")
    (harness.state_dir / "runner_state.json").write_bytes(old)

    real = runner_module.verify_log

    def no_tail(root):
        verification = real(root)
        object.__setattr__(verification, "last_runner_now_ns", None)
        return verification

    monkeypatch.setattr(runner_module, "verify_log", no_tail)
    with pytest.raises(DecisionLogError, match="precedes"):
        restarted(tmp_path, harness).runner.start()


def test_a_forged_tail_is_refused_not_bypassed(tmp_path, clean_identity, capsys):
    harness = halted_by_the_switch(tmp_path)
    day = sorted((harness.state_dir / "decision_log").glob("*.ndjson"))[-1]
    lines = day.read_text("utf-8").splitlines()
    forged = json.loads(lines[-3])
    forged["minute"] = forged["minute"].replace(":0", ":1", 1)
    lines[-3] = json.dumps(forged, sort_keys=True, separators=(",", ":"))
    day.write_text("\n".join(lines) + "\n", encoding="utf-8")
    before_day = day.read_bytes()
    argv = argv_for(tmp_path, harness)

    assert demo_run.main(argv + ["resume", "--note", NOTE]) == demo_run.EXIT_REFUSED
    assert "log_forged" in capsys.readouterr().err
    assert day.read_bytes() == before_day, "the forged log is left as found"
    assert json.loads((harness.state_dir / "risk.json").read_text("utf-8"))["halted"]


# ---------------------------------------------------------------------------
# flatten on CAMPAIGN
# ---------------------------------------------------------------------------
def _campaign(tmp_path: Path, harness, *, dirty: bool = False):
    again = restarted(tmp_path, harness)
    config = again.runner.config
    object.__setattr__(config, "profile", type(config.profile).CAMPAIGN)
    again.runner.software = {**CLEAN, "dirty": dirty}
    return again


def _flatten_through_the_cli(runner) -> int:
    args = Namespace(command="flatten", note=NOTE)
    return demo_run._dispatch(
        args, runner, operational_clock=lambda: 0.0, sleep=lambda _s: None
    )


def test_campaign_flatten_needs_no_allow_dirty_and_persists_no_spurious_halt(tmp_path, capsys):
    harness = build(tmp_path)
    harness.run(10)
    assert harness.runner.position.state is HedgeState.HEDGED
    harness.runner.shutdown("stop")
    campaign = _campaign(tmp_path, harness)

    assert _flatten_through_the_cli(campaign.runner) == demo_run.EXIT_OK
    risk = json.loads((harness.state_dir / "risk.json").read_text("utf-8"))
    assert risk["halted"] is False, risk["halt_reason"]
    assert not any(
        "allow_dirty" in json.dumps(r)
        for r in records(harness.state_dir)
        if r["kind"] == "HALT"
    )
    assert campaign.runner.position.leg("spot").quantity == 0
    assert campaign.runner.position.leg("perp").quantity == 0
    operators = [r for r in records(harness.state_dir) if r["kind"] == "OPERATOR"]
    assert [o["operator"]["phase"] for o in operators] == ["requested", "completed"]
    assert operators[-1]["position_after"]["perp_qty"] == "0"

    # The restart reconstructs exactly what the flatten left.
    again = _campaign(tmp_path, harness)
    assert again.runner.start() is RunnerState.READY
    assert again.runner.position.state is HedgeState.FLAT


def test_the_old_allow_dirty_flatten_would_persist_the_spurious_halt(tmp_path):
    """The witness: the pre-R1-i CLI's `start(allow_dirty=True)` on CAMPAIGN."""
    harness = build(tmp_path)
    harness.run(10)
    harness.runner.shutdown("stop")
    campaign = _campaign(tmp_path, harness)
    campaign.runner.start(allow_dirty=True)
    risk = json.loads((harness.state_dir / "risk.json").read_text("utf-8"))
    assert risk["halted"] and "allow_dirty" in risk["halt_reason"]


def test_a_dirty_campaign_flatten_still_reduces_and_halts_for_the_real_reason(tmp_path):
    harness = build(tmp_path)
    harness.run(10)
    harness.runner.shutdown("stop")
    campaign = _campaign(tmp_path, harness, dirty=True)
    assert _flatten_through_the_cli(campaign.runner) == demo_run.EXIT_OK
    assert campaign.runner.position.leg("perp").quantity == 0
    risk = json.loads((harness.state_dir / "risk.json").read_text("utf-8"))
    assert risk["halt_reason"].startswith("source_identity")


# ---------------------------------------------------------------------------
# resume ordering
# ---------------------------------------------------------------------------
class _Killed(BaseException):
    pass


def test_resume_appends_the_request_before_aegis_moves_and_resume_after(tmp_path, monkeypatch):
    harness = halted_by_the_switch(tmp_path)
    runner = restarted(tmp_path, harness).runner
    runner.start()
    order: list[str] = []
    real_append, real_resume = runner.log.append, runner.risk.resume

    def append(record):
        order.append(f"{record['kind']}:{record.get('operator', {}).get('phase', '')}")
        return real_append(record)

    def resume():
        order.append("aegis")
        return real_resume()

    monkeypatch.setattr(runner.log, "append", append)
    monkeypatch.setattr(runner.risk, "resume", resume)
    runner.resume(NOTE)
    assert order == ["OPERATOR:requested", "aegis", "RESUME:completed"]


def test_the_old_resume_order_left_a_sealed_continuity_dispute(tmp_path):
    """The witness for the fix: Aegis persisted running, killed before the
    RESUME record. The log's last statement is the HALT, the file is running,
    and R1-c seals -- an unclearable dispute out of an ordinary resume."""
    harness = halted_by_the_switch(tmp_path)
    runner = restarted(tmp_path, harness).runner
    runner.start()
    runner.risk.resume()  # the old order's first step, then the kill
    again = restarted(tmp_path, harness).runner
    assert again.start() is RunnerState.HALT
    assert again.risk.continuity_disputed


@pytest.mark.parametrize("where", ["before_aegis", "after_aegis"])
def test_a_kill_inside_resume_is_recoverable_and_reported(tmp_path, monkeypatch, where):
    harness = halted_by_the_switch(tmp_path)
    runner = restarted(tmp_path, harness).runner
    runner.start()
    real = runner.risk.resume

    def killed():
        if where == "after_aegis":
            real()
        raise _Killed()

    monkeypatch.setattr(runner.risk, "resume", killed)
    with pytest.raises(_Killed):
        runner.resume(NOTE)

    again = restarted(tmp_path, harness).runner
    state = again.start()
    assert not again.risk.continuity_disputed
    causes = [
        r["recovery"]["cause"] for r in records(harness.state_dir) if r["kind"] == "RECOVERY"
    ]
    assert RecoveryCause.OPERATOR_INCOMPLETE.value in causes
    if where == "before_aegis":
        assert state is RunnerState.HALT, "the halt stands until the request completes"
        again.resume(NOTE)
    assert again.state is RunnerState.READY


def test_resume_refuses_a_dispute_and_names_its_resolve(tmp_path):
    harness, symbol = _reconciliation_dispute(tmp_path)
    runner = restarted(tmp_path, harness).runner
    runner.start()
    with pytest.raises(RunnerError, match=re.escape(f"resolve --symbol {symbol}")):
        runner.resume(NOTE)
    assert runner.risk.state.halted


def test_resume_refuses_an_engaged_switch(tmp_path):
    harness = halted_by_the_switch(tmp_path)
    (harness.state_dir / "KILL_SWITCH").write_text("again\n", encoding="utf-8")
    runner = restarted(tmp_path, harness).runner
    runner.start()
    with pytest.raises(RunnerError, match="kill switch is engaged"):
        runner.resume(NOTE)


# ---------------------------------------------------------------------------
# disputes: exactly one clearing path each
# ---------------------------------------------------------------------------
_RAISERS = (
    REPO / "chimera" / "carry" / "hedge.py",
    REPO / "chimera" / "carry" / "ledger.py",
    REPO / "chimera" / "demo" / "runner.py",
)
_DISPUTE_TEXT = re.compile(
    r"""(?:dispute\(\s*|disputed\s*=\s*\(?\s*)f?["']([a-z{}_]+?)[:"' ]"""
)


def _raised_kinds() -> set[str]:
    kinds: set[str] = set()
    for path in _RAISERS:
        for kind in _DISPUTE_TEXT.findall(path.read_text("utf-8")):
            if "{name}" in kind:
                kinds.update(kind.replace("{name}", leg) for leg in ("spot", "perp"))
            else:
                kinds.add(kind)
    return kinds


def test_every_carry_dispute_the_code_can_raise_has_its_clearing_path():
    """The inventory, read off the code: every kind a `dispute(...)` or
    `disputed = ...` can write is either a `resolve --ledger` kind or a leg's
    reconciliation dispute (`resolve --symbol`)."""
    raised = _raised_kinds()
    assert raised, "the scan found nothing: it is not reading the raisers"
    symbol_kinds = {"spot_reconciliation_mismatch", "perp_reconciliation_mismatch"}
    catalogued = set(LEDGER_DISPUTES) | symbol_kinds
    assert raised - {"ledger_behind_log"} <= catalogued, raised - catalogued
    # And nothing is catalogued that nothing raises (ledger_behind_log is the
    # runner's load-time guard, raised in words rather than by `dispute`).
    assert set(LEDGER_DISPUTES) - {"ledger_behind_log"} <= raised


def test_the_cli_offers_exactly_the_catalogued_ledger_kinds():
    parser = demo_run.build_parser()
    for kind in LEDGER_DISPUTES:
        assert parser.parse_args(["resolve", "--ledger", kind, "--note", "n"]).ledger == kind
    with pytest.raises(SystemExit):
        parser.parse_args(["resolve", "--ledger", "whatever_it_finds", "--note", "n"])
    for pair in (["--equity", "--emergency"], ["--risk-state", "--symbol", "X"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["resolve", *pair, "--note", "n"])


@pytest.mark.parametrize(
    "selector",
    [
        ["--symbol", "BTC/USDT"],
        ["--equity"],
        ["--ledger", "stale_leg"],
        ["--risk-state"],
        ["--emergency"],
    ],
)
@pytest.mark.parametrize("note", ["", "   ", "\t\n"])
def test_every_resolve_requires_a_real_note(selector, note):
    with pytest.raises(SystemExit):
        demo_run._require_note(note, "resolve")
    args = demo_run.build_parser().parse_args(["resolve", *selector, "--note", note])
    assert args.note == note


def _damage(state_dir: Path, name: str, payload: bytes) -> None:
    (state_dir / name).write_bytes(payload)


@pytest.mark.parametrize(
    ("kind", "damage", "fact"),
    [
        (
            "ledger_unreadable",
            lambda d: _damage(d, "carry_ledger.json", b"{not json"),
            "exists in no other file",
        ),
        (
            "ledger_capital_mismatch",
            lambda d: _set(d, "carry_ledger.json", "capital", "999"),
            "which capital",
        ),
        (
            "identity_violation",
            lambda d: _set(d, "carry_ledger.json", "entry_basis", "12345"),
            "which of the ledger's own entry fields",
        ),
    ],
)
def test_a_dispute_nothing_can_reconstruct_is_refused_and_left_intact(
    tmp_path, kind, damage, fact
):
    harness = build(tmp_path)
    harness.run(5)
    harness.runner.shutdown("stop")
    damage(harness.state_dir)
    before = {p.name: p.read_bytes() for p in harness.state_dir.iterdir() if p.is_file()}
    runner = restarted(tmp_path, harness).runner
    runner.start()
    held = runner.position.ledger.disputed or ""
    if kind == "identity_violation":
        # Checked at the first mark; `start()` does not mark.
        runner.position.mark_to_market(
            runner.cursor.state_for(runner.cursor.last_minute_processed)
        )
        held = runner.position.ledger.disputed or ""
    assert held.startswith(kind), held
    ledger_before = (harness.state_dir / "carry_ledger.json").read_bytes()
    with pytest.raises(RunnerError, match=re.escape(fact)):
        runner.resolve_ledger(kind, NOTE)
    assert runner.position.ledger.disputed == held
    assert (harness.state_dir / "carry_ledger.json").read_bytes() == ledger_before
    assert before["carry_ledger.json"] == ledger_before


def _set(state_dir: Path, name: str, key: str, value: str) -> None:
    path = state_dir / name
    payload = json.loads(path.read_text("utf-8"))
    payload[key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_a_resolve_that_names_another_kind_is_refused(tmp_path):
    harness = build(tmp_path)
    harness.run(5)
    harness.runner.shutdown("stop")
    _damage(harness.state_dir, "carry_ledger.json", b"{not json")
    runner = restarted(tmp_path, harness).runner
    runner.start()
    with pytest.raises(RunnerError, match="not 'stale_leg'"):
        runner.resolve_ledger("stale_leg", NOTE)


def test_a_symbol_with_no_dispute_is_refused(tmp_path):
    harness = build(tmp_path)
    harness.run(5)
    runner = harness.runner
    with pytest.raises(RunnerError, match="no reconciliation dispute"):
        runner.resolve(runner.position.config.perp_symbol, NOTE)


def test_the_risk_state_path_refuses_and_names_the_unknowable(tmp_path):
    harness = build(tmp_path)
    harness.run(5)
    harness.runner.shutdown("stop")
    (harness.state_dir / "risk.json").unlink()
    runner = restarted(tmp_path, harness).runner
    assert runner.start() is RunnerState.HALT
    assert runner.risk.continuity_disputed
    with pytest.raises(RunnerError, match="RISK_STATE_ABSENT.*recorded there only as a hash"):
        runner.resolve_risk_state(NOTE)
    assert not (harness.state_dir / "risk.json").exists(), "the absence is left as found"
    # Without a seal the path has nothing to do and says so.
    clean = build(tmp_path / "clean").runner
    with pytest.raises(RunnerError, match="nothing for"):
        clean.resolve_risk_state(NOTE)


def _stale_leg(tmp_path: Path):
    """A REAL stale leg: one leg observed more than a minute after the other."""
    harness = build(tmp_path)
    harness.run(10)
    ledger = harness.runner.position.ledger
    marks = dict(ledger.state.marked_at_ns)
    ledger.state.marked_at_ns["perp"] = marks["spot"] - 5 * 60_000_000_000
    ledger.save()
    harness.runner.shutdown("stop")
    return harness


def test_stale_leg_is_re_marked_and_cleared_by_its_resolve(tmp_path):
    harness = _stale_leg(tmp_path)
    runner = restarted(tmp_path, harness).runner
    assert runner.start() is RunnerState.HALT
    assert (runner.position.ledger.disputed or "").startswith("stale_leg")
    runner.resolve_ledger("stale_leg", NOTE)
    assert runner.standing_disputes() == []
    completed = [r for r in records(harness.state_dir) if r["kind"] == "OPERATOR"][-1]
    assert completed["operator"]["phase"] == "completed"
    assert completed["operator"]["kind"] == "stale_leg"
    again = restarted(tmp_path, harness).runner
    assert again.start() is RunnerState.HALT, "the carry dispute's halt remains"
    again.resume(NOTE)
    assert again.state is RunnerState.READY


def test_the_ledger_catalogue_is_complete_and_classified():
    rebook = {kind for kind, action in LEDGER_DISPUTES.items() if action == REBOOK}
    assert rebook == {
        "ledger_store_mismatch",
        "funding_booking_torn",
        "stale_leg",
        "asymmetric_close",
    }
    for kind, action in LEDGER_DISPUTES.items():
        if action != REBOOK:
            assert len(action) > 40, f"{kind}: a refusal must name the unknowable fact"


# ---------------------------------------------------------------------------
# HedgedPosition.correct(): wired, bounded, and restart-proof
# ---------------------------------------------------------------------------
def _refusing(harness, bps: str = "0") -> None:
    harness.runner.position.fill_models["spot"].max_reference_deviation_bps = D(bps)


def test_a_partial_is_corrected_and_then_flattened_within_the_window(tmp_path):
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    _refusing(harness)
    harness.tick(first)
    position = harness.runner.position
    assert position.state is HedgeState.PARTIAL
    correction = position.ledger.state.correction
    assert correction["started_ns"] == first * 1_000_000 + 60_000_000_000

    window = position.config.max_correction_minutes
    for index in range(1, window + 1):
        harness.tick(first + index * 60_000)
        assert position.state is HedgeState.PARTIAL, index
    harness.tick(first + (window + 1) * 60_000)
    assert position.state is HedgeState.FLAT
    assert position.ledger.state.correction is None
    reasons = json.loads((harness.state_dir / "perp_store.json").read_text("utf-8"))[
        "flatten_reasons"
    ]
    assert reasons[-1]["reason"] == "HEDGE_CORRECTION"


def test_a_correction_that_succeeds_ends_hedged(tmp_path):
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    _refusing(harness)
    harness.tick(first)
    _refusing(harness, "50")
    harness.tick(first + 60_000)
    assert harness.runner.position.state is HedgeState.HEDGED
    assert harness.runner.position.ledger.state.correction is None


def test_a_restart_does_not_reset_the_correction_window(tmp_path):
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    _refusing(harness)
    harness.run(3, start=first)
    harness.runner.shutdown("mid-correction")

    again = restarted(tmp_path, harness)
    assert again.runner.start() is RunnerState.READY, "no stale_leg dispute mid-correction"
    _refusing(again)
    window = again.runner.position.config.max_correction_minutes
    again.tick(first + 3 * 60_000)
    assert again.runner.position.state is HedgeState.PARTIAL
    again.tick(first + (window + 1) * 60_000 - 0)
    # The window counts from the persisted start, not from this process: the
    # flatten comes at the same minute it would have without the restart.
    assert again.runner.position.state is HedgeState.FLAT


def test_a_version_1_ledger_mid_correction_is_expired_not_renewed(tmp_path):
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    _refusing(harness)
    harness.tick(first)
    harness.runner.shutdown("stop")
    path = harness.state_dir / "carry_ledger.json"
    payload = json.loads(path.read_text("utf-8"))
    payload["schema"] = "chimera.carry-ledger/1"
    payload.pop("correction")
    path.write_text(json.dumps(payload), encoding="utf-8")

    again = restarted(tmp_path, harness)
    again.runner.start()
    assert again.runner.position.ledger.state.correction["started_ns"] == 0
    _refusing(again)
    again.tick(first + 60_000)
    assert again.runner.position.state is HedgeState.FLAT, "an unknown start has expired"


def test_the_ledger_schema_is_versioned_and_reads_both(tmp_path):
    assert LEDGER_SCHEMA == "chimera.carry-ledger/2"
    assert LEDGER_SCHEMAS_READ == ("chimera.carry-ledger/2", "chimera.carry-ledger/1")
    path = tmp_path / "carry_ledger.json"
    path.write_text(
        json.dumps({"schema": LEDGER_SCHEMA, "capital": "1000000", "correction": "soon"}),
        encoding="utf-8",
    )
    assert CarryLedger.open(path, capital=D("1000000")).state.disputed.startswith(
        "ledger_unreadable"
    ), "a malformed correction is refused, never defaulted"


# ---------------------------------------------------------------------------
# one-legged positions: liquidation-checked per leg
# ---------------------------------------------------------------------------
def _touch_state(position, state, *, equity: D):
    return position.liquidation_touched(state, equity=equity)


def test_liquidation_is_checked_on_the_perpetual_leg_alone(tmp_path):
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    _refusing(harness)
    harness.tick(first)
    position = harness.runner.position
    assert position.leg("spot").is_flat and not position.leg("perp").is_flat
    state = harness.runner.cursor.state_for(first)
    maintenance = (
        position.leg("perp").quantity
        * (state.mark_high or state.mark)
        * position.config.maintenance_margin_rate
    )
    # Perp-only: the leveraged leg is checked on its own quantity.
    assert _touch_state(position, state, equity=maintenance - 1) is True
    assert _touch_state(position, state, equity=maintenance + 10_000) is False


def test_the_liquidation_checks_by_leg(tmp_path):
    """Spot-only is never touched (it has no liquidation); unequal legs are
    checked on the perpetual's quantity; hedged and flat as before."""
    harness = build(tmp_path)
    harness.run(3)
    position = harness.runner.position
    state = harness.runner.cursor.state_for(harness.first_minute_ms() + 2 * 60_000)
    q = position.leg("perp").quantity
    maintenance = q * (state.mark_high or state.mark) * position.config.maintenance_margin_rate
    assert _touch_state(position, state, equity=maintenance - 1) is True
    assert _touch_state(position, state, equity=maintenance + 10_000) is False
    # Spot-only: flatten the perpetual leg alone.
    position.perp.emergency_flatten(
        position.config.perp_symbol, FlattenCause.RISK_HALT, state.perp_close
    )
    assert position.leg("perp").is_flat and not position.leg("spot").is_flat
    assert _touch_state(position, state, equity=D("1")) is False


def test_a_perp_only_position_is_liquidation_checked_by_the_tick(tmp_path):
    """Through the runner: a one-legged short whose equity erodes is touched,
    flattened and halted -- it used to read min(spot, perp) = 0 and pass."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    _refusing(harness)
    harness.tick(first)
    position = harness.runner.position
    state = harness.runner.cursor.state_for(first + 60_000)
    adverse = state.mark_high or state.mark
    maintenance = (
        position.leg("perp").quantity * adverse * position.config.maintenance_margin_rate
    )
    position.ledger.state.free_cash -= position.equity_at(state) - maintenance + D("1")
    outcome = harness.tick(first + 60_000)
    assert outcome.kind is RecordKind.LIQUIDATION_TOUCH
    assert harness.runner.state is RunnerState.HALT
    assert position.leg("perp").is_flat


# ---------------------------------------------------------------------------
# persistence failure: non-zero exit, the emergency record, the restart
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["risk", "ledger", "store", "log", "runner_state"])
def test_any_persistence_failure_exits_4_with_a_trace(
    tmp_path, clean_identity, monkeypatch, capsys, target
):
    harness = build(tmp_path)
    harness.run(10)
    harness.runner.shutdown("stop")
    argv = argv_for(tmp_path, harness)
    import os

    real_fsync = os.fsync
    failing = {"left": 1}
    names = {
        "risk": "risk.json",
        "ledger": "carry_ledger.json",
        "store": "perp_store.json",
        "log": ".ndjson",
        "runner_state": "runner_state.json",
    }
    real_open = open

    opened: dict[int, str] = {}

    def tracking_open(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        opened[handle.fileno()] = str(file)
        return handle

    def fsync(fd):
        name = opened.get(fd, "")
        if failing["left"] and names[target] in name:
            failing["left"] -= 1
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_fsync(fd)

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(os, "fsync", fsync)
    code = demo_run.main(argv + ["run", "--once"])
    monkeypatch.undo()
    capsys.readouterr()
    if failing["left"]:
        pytest.skip(f"no {target} write in this pass")
    assert code == demo_run.EXIT_PERSISTENCE_FAILURE
    assert code not in (demo_run.EXIT_OK, demo_run.EXIT_HALTED)
    trace = emergency.read_trace(harness.state_dir)
    assert trace.state is emergency.EmergencyState.TRIPPED
    assert trace.slots[0]["failure"]["errno"] == "ENOSPC"

    # The restart finds the trace, records it once, and halts on it.
    again = restarted(tmp_path, harness).runner
    assert again.start() is RunnerState.HALT
    assert again.halt_reason.startswith("persistence_failure")
    causes = [
        r["recovery"]["cause"] for r in records(harness.state_dir) if r["kind"] == "RECOVERY"
    ]
    assert causes.count(RecoveryCause.PERSISTENCE_FAILURE.value) == 1
    again.shutdown("seen")
    third = restarted(tmp_path, harness).runner
    third.start()
    causes = [
        r["recovery"]["cause"] for r in records(harness.state_dir) if r["kind"] == "RECOVERY"
    ]
    assert causes.count(RecoveryCause.PERSISTENCE_FAILURE.value) == 1, "recorded once"
    with pytest.raises(RunnerError, match="resolve --emergency"):
        third.resume(NOTE)
    third.resolve_emergency(NOTE)
    assert emergency.read_trace(harness.state_dir).state is emergency.EmergencyState.ARMED
    third.shutdown("acknowledged")


def test_the_reserve_exists_at_full_size_before_anything_is_processed(tmp_path):
    harness = build(tmp_path, start=False)
    assert not (harness.state_dir / emergency.EMERGENCY_NAME).exists()
    harness.runner.start()
    path = harness.state_dir / emergency.EMERGENCY_NAME
    assert path.stat().st_size == emergency.FILE_BYTES
    assert emergency.read_trace(harness.state_dir).state is emergency.EmergencyState.ARMED
    # A trip rewrites in place: the size, and so the allocation, does not change.
    assert emergency.trip(harness.state_dir, {"failure": {"errno": "ENOSPC"}})
    assert path.stat().st_size == emergency.FILE_BYTES


def test_a_second_failure_never_overwrites_the_first(tmp_path):
    harness = build(tmp_path)
    assert emergency.trip(harness.state_dir, {"failure": {"errno": "ENOSPC", "role": "a"}})
    assert emergency.trip(harness.state_dir, {"failure": {"errno": "EIO", "role": "b"}})
    assert emergency.trip(harness.state_dir, {"failure": {"errno": "EROFS", "role": "c"}})
    trace = emergency.read_trace(harness.state_dir)
    assert trace.slots[0]["failure"]["role"] == "a", "the first failure is kept"
    assert trace.slots[1]["failure"]["role"] == "c", "the latest is kept beside it"
    # And a start never re-arms (overwrites) a tripped record.
    again = restarted(tmp_path, harness).runner
    again.start()
    assert emergency.read_trace(harness.state_dir).slots[0]["failure"]["role"] == "a"


def test_an_unreadable_reserve_is_treated_as_a_trace(tmp_path):
    harness = build(tmp_path)
    harness.runner.shutdown("stop")
    (harness.state_dir / emergency.EMERGENCY_NAME).write_bytes(b"half a record")
    again = restarted(tmp_path, harness).runner
    assert again.start() is RunnerState.HALT
    assert "unreadable" in again.halt_reason


def test_a_failed_emergency_write_still_exits_non_zero(
    tmp_path, clean_identity, monkeypatch, capsys
):
    harness = build(tmp_path)
    harness.run(10)
    harness.runner.shutdown("stop")
    argv = argv_for(tmp_path, harness)
    from chimera.risk import RiskEngine

    real = RiskEngine._persist

    def full(self):
        raise PersistenceFailure("risk.json", "write", "ENOSPC")

    monkeypatch.setattr(RiskEngine, "_persist", full)
    monkeypatch.setattr(emergency, "trip", lambda *a, **k: False)
    code = demo_run.main(argv + ["run", "--once"])
    monkeypatch.setattr(RiskEngine, "_persist", real)
    captured = capsys.readouterr()
    assert code == demo_run.EXIT_PERSISTENCE_FAILURE
    assert "could NOT be written" in captured.err
    assert json.loads(captured.out.splitlines()[-1])["emergency_record_written"] is False


def test_a_persistence_failure_is_not_an_exception_any_stage_can_absorb():
    assert not issubclass(PersistenceFailure, Exception)
    assert issubclass(PersistenceFailure, BaseException)


def test_status_reports_the_emergency_record_without_changing_it(
    tmp_path, clean_identity, capsys
):
    harness = build(tmp_path)
    harness.runner.shutdown("stop")
    assert emergency.trip(harness.state_dir, {"failure": {"errno": "ENOSPC"}})
    before = (harness.state_dir / emergency.EMERGENCY_NAME).read_bytes()
    argv = argv_for(tmp_path, harness)
    assert demo_run.main(argv + ["status"]) == demo_run.EXIT_OK
    status = json.loads(capsys.readouterr().out)
    assert status["emergency"]["state"] == "TRIPPED"
    assert (harness.state_dir / emergency.EMERGENCY_NAME).read_bytes() == before


def test_the_emergency_record_is_not_evidence_and_moves_no_contract():
    """It is read by no decision, carries no economics, and the gen3 contract
    hash is unchanged by R1-i."""
    from chimera.recorder.contract import load_recorder_contract

    contract = load_recorder_contract("btcusdt-prospective-gen3")
    assert contract.contract_hash == (
        "55268a0fe805f937824e1ae45ee3a32e6db00f0714910b31c1cc3266d63cde7c"
    )
    for name in ("rules.py", "rules_carry.py", "rules_shadow.py", "feed.py", "reports.py"):
        assert "emergency" not in (REPO / "chimera" / "demo" / name).read_text("utf-8")


# ---------------------------------------------------------------------------
# the R1-g crash window, revisited under R1-i
# ---------------------------------------------------------------------------
def test_the_owed_funding_bound_is_on_disk_before_aegis_halts_on_a_touch(
    tmp_path, monkeypatch
):
    """R1-g's accepted residual: a kill between Aegis's touch halt and the
    owed-funding bound. The bound is now saved first (and the touch record
    before the halt), so no kill leaves a halted touch without its bound."""
    harness = build(tmp_path)
    harness.run(3)
    position = harness.runner.position
    order: list[str] = []
    real_save, real_halt = position.ledger.save, harness.runner.risk.halt

    def save(*args, **kwargs):
        order.append("ledger")
        return real_save(*args, **kwargs)

    def halt(reason):
        order.append(f"halt:{reason}")
        return real_halt(reason)

    real_append = harness.runner.log.append

    def append(record):
        order.append(record["kind"])
        return real_append(record)

    position.ledger.owe_funding(
        position.ledger.state.open_instant_ns + 1, open_instant_ns=0, perp={"q": "1"}
    )
    monkeypatch.setattr(position.ledger, "save", save)
    monkeypatch.setattr(harness.runner.risk, "halt", halt)
    monkeypatch.setattr(harness.runner.log, "append", append)
    state = harness.runner.cursor.state_for(harness.first_minute_ms() + 3 * 60_000)
    adverse = state.mark_high or state.mark
    maintenance = (
        position.leg("perp").quantity * adverse * position.config.maintenance_margin_rate
    )
    position.ledger.state.free_cash -= position.equity_at(state) - maintenance + D("1")
    harness.tick(harness.first_minute_ms() + 3 * 60_000)
    first_ledger = order.index("ledger")
    assert (
        first_ledger < order.index("LIQUIDATION_TOUCH") < order.index("halt:liquidation_touch")
    ), order


# ---------------------------------------------------------------------------
# the funding-streak crash window (found by the harness; R1-c's `funding`)
# ---------------------------------------------------------------------------
def _risk_snapshot(tmp_path: Path) -> dict:
    harness = build(tmp_path)
    harness.run(3)
    return dict(harness.runner.risk.snapshot())


@pytest.mark.parametrize(
    ("prior", "found", "sign", "cost"),
    [
        ((0, False), (1, False), 1, "paid"),
        ((2, False), (3, True), 1, "paid"),
        ((3, True), (0, False), -1, "received"),
        ((5, False), (0, False), -1, "received"),
    ],
)
def test_the_funding_window_is_proved_in_each_direction(tmp_path, prior, found, sign, cost):
    from chimera.demo.risk_continuity import funding_note_prior, risk_state_hash

    base = _risk_snapshot(tmp_path)
    before = {**base, "funding_adverse_streak": prior[0], "funding_halt": prior[1]}
    after = {**base, "funding_adverse_streak": found[0], "funding_halt": found[1]}
    proved = funding_note_prior(after, risk_state_hash(before))
    assert proved is not None and proved["cost"] == cost
    assert funding_note_prior(after, risk_state_hash(before), cost_sign=sign) is not None
    # The other direction cannot explain it.
    assert funding_note_prior(after, risk_state_hash(before), cost_sign=-sign) is None


@pytest.mark.parametrize(
    "moved",
    [
        {"funding_adverse_streak": 2},  # two settlements, not one
        {"funding_adverse_streak": 1, "equity": 1.0},  # and something else moved
        {"funding_adverse_streak": 0, "funding_halt": False},  # nothing moved
    ],
)
def test_the_funding_window_proves_nothing_else(tmp_path, moved):
    from chimera.demo.risk_continuity import funding_note_prior, risk_state_hash

    base = {**_risk_snapshot(tmp_path), "funding_adverse_streak": 0, "funding_halt": False}
    assert funding_note_prior({**base, **moved}, risk_state_hash(base)) is None
