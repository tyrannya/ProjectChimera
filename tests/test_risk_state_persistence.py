"""What the risk engine remembers across a restart, and what it refuses to guess.

Every guard in :mod:`chimera.risk` reads state. Before this file's subject
existed, only the halt flag survived a process restart, so a bot that stopped
between its equity peak and the fall that breached the drawdown limit came back
measuring that fall from the wrong peak — and approved the trade the limit
exists to stop. The tests here are therefore paired: a restart that carries the
state trips the guard, and the same numbers with no carried state do not, which
is what makes them evidence about persistence rather than about the guard.

The other half is the reading side. A file that cannot be parsed, that carries a
schema this build does not know, or that is missing a field is never turned into
a default: it starts the engine halted. And a write that is interrupted leaves
the previous file exactly as it was, because a half-written state file is a
worse answer than a stale one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chimera.risk import RISK_STATE_SCHEMA, RiskEngine, RiskLimits, RiskViolation

REPO_ROOT = Path(__file__).resolve().parents[1]

DAY_ONE = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


class FakeClock:
    """A hand-advanced ``time.time``. Nothing here may depend on real time."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def build(tmp_path, limits=None, clock=None, state="risk.json"):
    """An engine whose state file and kill-switch path are both under ``tmp_path``.

    The kill-switch path is named explicitly so these tests say nothing about
    whatever happens to exist under the working directory.
    """
    return RiskEngine(
        limits or RiskLimits(),
        state_path=tmp_path / state,
        clock=clock or FakeClock(),
        kill_switch_path=tmp_path / "no-kill-switch",
    )


def entry(engine, **overrides):
    kwargs = {
        "pair": "BTC/USDT",
        "equity": 10_000.0,
        "entry_price": 100.0,
        "stop_price": 95.0,
        **overrides,
    }
    return engine.evaluate_entry(**kwargs)


# --- the state that guards read ------------------------------------------
def test_the_peak_equity_survives_a_restart_between_the_peak_and_the_breach(tmp_path):
    """The failure this whole file exists for.

    A restart used to reset ``peak_equity`` to zero, after which the next equity
    reading became the new peak and a 12.5% fall from the real one measured as no
    drawdown at all.
    """
    limits = RiskLimits(max_drawdown_pct=0.10, max_daily_loss_pct=0.90)
    first = build(tmp_path, limits)
    first.update_equity(10_000.0, now=DAY_ONE)
    first.update_equity(12_000.0, now=DAY_ONE)
    assert not first.halted

    revived = build(tmp_path, limits)
    assert revived.state.peak_equity == pytest.approx(12_000.0)
    revived.update_equity(10_500.0, now=DAY_ONE)

    assert revived.halted, "12.5% below the carried peak is past a 10% limit"
    assert "drawdown" in revived.state.halt_reason
    assert revived.current_drawdown() == pytest.approx(0.125)

    # The control: the same equity with no carried peak is not a drawdown.
    fresh = build(tmp_path, limits, state="other.json")
    fresh.update_equity(10_500.0, now=DAY_ONE)
    assert not fresh.halted


def test_the_days_starting_equity_survives_a_restart(tmp_path):
    limits = RiskLimits(max_daily_loss_pct=0.02, max_drawdown_pct=0.90)
    first = build(tmp_path, limits)
    first.update_equity(10_000.0, now=DAY_ONE)

    revived = build(tmp_path, limits)
    assert revived.state.day == DAY_ONE.date().isoformat()
    assert revived.state.day_start_equity == pytest.approx(10_000.0)
    revived.update_equity(9_700.0, now=DAY_ONE)

    assert revived.halted, "3% below the carried day start is past a 2% limit"
    assert "daily loss" in revived.state.halt_reason

    # The control: without the carried day start, 9,700 *is* the day start.
    fresh = build(tmp_path, limits, state="other.json")
    fresh.update_equity(9_700.0, now=DAY_ONE)
    assert not fresh.halted


def test_the_days_pnl_is_persisted_for_the_report(tmp_path):
    engine = build(tmp_path, RiskLimits(max_daily_loss_pct=0.90, max_drawdown_pct=0.90))
    engine.update_equity(10_000.0, now=DAY_ONE)
    engine.update_equity(9_800.0, now=DAY_ONE)

    written = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    assert written["daily_pnl"] == pytest.approx(-200.0)
    assert written["day_start_equity"] == pytest.approx(10_000.0)
    # And it is a report field, not a second authority on the drawdown, which
    # stays derived from the two equities.
    assert "drawdown" not in written


def test_the_order_rate_window_survives_a_restart(tmp_path):
    limits = RiskLimits(max_orders_per_minute=2)
    clock = FakeClock(1_000.0)
    first = build(tmp_path, limits, clock)
    first.record_order()
    first.record_order()
    assert not first.halted

    revived = build(tmp_path, limits, FakeClock(1_010.0))
    assert revived.state.order_times == [1_000.0, 1_000.0]
    revived.record_order()
    assert revived.halted, "the third order in the carried window is over the limit"
    assert "order rate" in revived.state.halt_reason


def test_the_restored_order_window_still_expires(tmp_path):
    """The mirror: a window that survived must also be allowed to run out."""
    limits = RiskLimits(max_orders_per_minute=2)
    first = build(tmp_path, limits, FakeClock(1_000.0))
    first.record_order()
    first.record_order()

    revived = build(tmp_path, limits, FakeClock(1_060.5))
    revived.record_order()

    assert not revived.halted
    assert revived.state.order_times == [1_060.5]


def test_only_the_live_window_is_written_to_disk(tmp_path):
    """A state file is not an order log; it holds the window a guard reads."""
    clock = FakeClock(1_000.0)
    engine = build(tmp_path, RiskLimits(max_orders_per_minute=10), clock)
    engine.record_order()
    engine.record_order()
    assert json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))["order_times"] == [
        1_000.0,
        1_000.0,
    ]

    clock.now = 1_070.0
    engine.record_trade_result(1.0)  # any mutation; the point is the write

    assert (
        json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))["order_times"] == []
    )


def test_the_cooldown_survives_a_restart(tmp_path):
    limits = RiskLimits(loss_streak_limit=2, cooldown_seconds=3_600.0)
    first = build(tmp_path, limits, FakeClock(1_000.0))
    first.record_trade_result(-5.0)
    first.record_trade_result(-5.0)
    assert first.state.cooldown_until == pytest.approx(4_600.0)

    inside = build(tmp_path, limits, FakeClock(1_010.0))
    decision = entry(inside)
    assert not decision.allowed
    assert "cooldown" in decision.reason

    # The mirror: the carried deadline is a deadline, not a permanent block.
    after = build(tmp_path, limits, FakeClock(4_600.5))
    assert entry(after).allowed


def test_the_loss_streak_survives_a_restart(tmp_path):
    limits = RiskLimits(loss_streak_limit=2, cooldown_seconds=3_600.0)
    first = build(tmp_path, limits, FakeClock(1_000.0))
    first.record_trade_result(-5.0)
    assert first.state.cooldown_until == 0.0

    revived = build(tmp_path, limits, FakeClock(1_000.0))
    assert revived.state.consecutive_losses == 1
    revived.record_trade_result(-5.0)
    assert revived.state.cooldown_until == pytest.approx(4_600.0)

    # The control: one loss on a fresh engine starts no cooldown.
    fresh = build(tmp_path, limits, FakeClock(1_000.0), state="other.json")
    fresh.record_trade_result(-5.0)
    assert fresh.state.cooldown_until == 0.0


def test_open_positions_survive_a_restart(tmp_path):
    first = build(tmp_path, RiskLimits(max_open_positions=1))
    first.set_position_exposure("BTC/USDT", 500.0)

    revived = build(tmp_path, RiskLimits(max_open_positions=1))
    assert revived.state.open_positions == {"BTC/USDT": 500.0}
    assert revived.total_exposure == pytest.approx(500.0)

    decision = entry(revived, pair="ETH/USDT")
    assert not decision.allowed
    assert "max open positions" in decision.reason


def test_the_whole_state_round_trips_through_the_file(tmp_path):
    """Every persisted field, not only the ones a particular guard reads."""
    clock = FakeClock(1_000.0)
    engine = build(tmp_path, RiskLimits(max_drawdown_pct=0.9, max_daily_loss_pct=0.9), clock)
    engine.update_equity(10_000.0, now=DAY_ONE)
    engine.update_equity(9_500.0, now=DAY_ONE)
    engine.set_position_exposure("BTC/USDT", 250.0)
    engine.record_order()
    engine.record_trade_result(-1.0)
    engine.note_reconciliation("BTC/USDT", "venue says 2, we say 1")
    engine.note_feed(last_minute_close_ns=0, now_ns=10**12)
    engine.note_funding_settlement("BTC/USDT", "LONG", 0.001)
    engine.halt("operator")

    revived = build(tmp_path, RiskLimits(max_drawdown_pct=0.9, max_daily_loss_pct=0.9), clock)

    assert revived.snapshot() == engine.snapshot()
    assert revived.state == engine.state


def test_the_written_document_names_its_schema(tmp_path):
    engine = build(tmp_path)
    engine.halt("operator")

    written = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    assert written["schema"] == RISK_STATE_SCHEMA
    assert written["updated_at"]


# --- reading a file that cannot be believed -------------------------------
def test_a_legacy_halt_only_file_still_halts(tmp_path):
    """The pre-schema format. Losing the halt while upgrading would be the bug."""
    (tmp_path / "risk.json").write_text(
        json.dumps(
            {
                "halted": True,
                "halt_reason": "drawdown breach",
                "updated_at": "2026-02-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    engine = build(tmp_path)

    assert engine.halted
    assert engine.state.halt_reason == "drawdown breach"
    assert not entry(engine).allowed


def test_a_legacy_file_that_records_no_halt_does_not_invent_one(tmp_path):
    (tmp_path / "risk.json").write_text(
        json.dumps({"halted": False, "halt_reason": "", "updated_at": "2026-02-01T00:00:00Z"}),
        encoding="utf-8",
    )

    assert not build(tmp_path).halted


def test_a_schemaless_file_that_is_not_the_legacy_record_fails_closed(tmp_path):
    """Recognising the legacy shape must not become "read whatever is there".

    A document with no schema and fields beyond the three the old format wrote
    is not a state file this build understands, and reading the halt out of it
    while silently dropping the rest would present unknown equity, exposure and
    cooldown as zeroes.
    """
    (tmp_path / "risk.json").write_text(
        json.dumps({"halted": False, "equity": 10_000.0}), encoding="utf-8"
    )

    engine = build(tmp_path)

    assert engine.halted
    assert "equity" in engine.state.halt_reason


def test_an_unreadable_state_file_fails_closed_and_names_the_problem(tmp_path):
    (tmp_path / "risk.json").write_text("{not json", encoding="utf-8")

    engine = build(tmp_path)

    assert engine.halted
    assert engine.state.halt_reason.startswith("unreadable persisted risk state")
    assert not entry(engine).allowed


def test_a_state_file_that_is_not_an_object_fails_closed(tmp_path):
    (tmp_path / "risk.json").write_text("[1, 2, 3]", encoding="utf-8")

    assert build(tmp_path).halted


def test_an_unknown_schema_fails_closed(tmp_path):
    """A future build's file is not this build's to interpret."""
    (tmp_path / "risk.json").write_text(
        json.dumps({"schema": "chimera.risk-state/2", "halted": False}), encoding="utf-8"
    )

    engine = build(tmp_path)

    assert engine.halted
    assert "chimera.risk-state/2" in engine.state.halt_reason


def test_a_state_file_missing_a_field_fails_closed_rather_than_defaulting(tmp_path):
    engine = build(tmp_path)
    engine.update_equity(10_000.0, now=DAY_ONE)
    document = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    del document["peak_equity"]
    (tmp_path / "risk.json").write_text(json.dumps(document), encoding="utf-8")

    revived = build(tmp_path)

    assert revived.halted
    assert "peak_equity" in revived.state.halt_reason
    assert revived.state.peak_equity == 0.0, "the halt is what stands in for the number"


def test_a_state_file_with_a_mistyped_field_fails_closed(tmp_path):
    engine = build(tmp_path)
    engine.update_equity(10_000.0, now=DAY_ONE)
    document = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    document["open_positions"] = ["BTC/USDT"]
    (tmp_path / "risk.json").write_text(json.dumps(document), encoding="utf-8")

    revived = build(tmp_path)

    assert revived.halted
    assert "open_positions" in revived.state.halt_reason


def test_a_good_file_written_by_this_build_loads_without_halting(tmp_path):
    """The control for every fail-closed case above."""
    engine = build(tmp_path)
    engine.update_equity(10_000.0, now=DAY_ONE)

    revived = build(tmp_path)

    assert not revived.halted
    assert revived.state.equity == pytest.approx(10_000.0)


# --- durability -----------------------------------------------------------
def test_the_swap_is_an_os_replace_from_a_temporary_sibling(tmp_path, monkeypatch):
    """Pins the mechanism, because the guarantee is the mechanism.

    A plain ``write_text`` onto the live path can be interrupted with the file
    truncated; only a fully written temporary that is renamed over the original
    makes "the previous file is intact" true.
    """
    engine = build(tmp_path)
    engine.update_equity(10_000.0, now=DAY_ONE)

    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("chimera.risk.os.replace", spy)
    engine.update_equity(11_000.0, now=DAY_ONE)

    assert calls == [(str(tmp_path / "risk.json.tmp"), str(tmp_path / "risk.json"))]
    assert json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))[
        "equity"
    ] == pytest.approx(11_000.0)


def test_an_interrupted_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    engine = build(tmp_path)
    engine.update_equity(10_000.0, now=DAY_ONE)
    before = (tmp_path / "risk.json").read_text(encoding="utf-8")

    def refuse(src, dst):
        raise OSError("simulated crash between the temporary file and the rename")

    monkeypatch.setattr("chimera.risk.os.replace", refuse)
    engine.update_equity(11_000.0, now=DAY_ONE)

    assert (tmp_path / "risk.json").read_text(encoding="utf-8") == before
    revived = build(tmp_path)
    assert revived.state.equity == pytest.approx(10_000.0)
    assert not revived.halted, "the previous file is valid, so it must load normally"


def test_a_write_that_cannot_land_does_not_unwind_the_halt_in_memory(tmp_path, monkeypatch):
    """A disk problem must not become a skipped guard."""

    def refuse(src, dst):
        raise OSError("read-only file system")

    engine = build(tmp_path)
    monkeypatch.setattr("chimera.risk.os.replace", refuse)

    engine.halt("drawdown breach")

    assert engine.halted
    assert not entry(engine).allowed


def test_an_engine_with_no_state_path_still_works(tmp_path):
    """Persistence is optional; the in-memory guards are not."""
    engine = RiskEngine(RiskLimits(), kill_switch_path=tmp_path / "no-kill-switch")
    engine.update_equity(10_000.0, now=DAY_ONE)
    engine.halt("operator")

    assert engine.halted
    assert not list(tmp_path.iterdir())


# --- the semantic snapshot ------------------------------------------------
def test_the_snapshot_carries_no_write_time_and_nothing_about_this_process(tmp_path):
    engine = build(tmp_path)
    engine.update_equity(10_000.0, now=DAY_ONE)

    snapshot = engine.snapshot()

    assert "updated_at" not in snapshot
    assert snapshot["schema"] == RISK_STATE_SCHEMA
    rendered = json.dumps(snapshot)
    assert str(tmp_path) not in rendered
    assert "risk.json" not in rendered


def test_the_snapshot_does_not_move_when_nothing_semantic_moves(tmp_path):
    """The property a decision log's ``risk_state_hash`` needs.

    Time passing, and the state being written to disk again, both change the
    file. Neither changes what a guard would decide, so neither may change the
    snapshot.
    """
    clock = FakeClock(1_000.0)
    engine = build(tmp_path, clock=clock)
    engine.update_equity(10_000.0, now=DAY_ONE)
    before = engine.snapshot()

    clock.now = 9_999.0
    engine.note_reconciliation("BTC/USDT", None)  # clears nothing; writes the file

    assert engine.snapshot() == before
    assert build(tmp_path, clock=clock).snapshot() == before


def test_the_snapshot_moves_when_a_decision_input_moves(tmp_path):
    """The mirror, so the test above is not satisfied by a constant."""
    engine = build(tmp_path)
    engine.update_equity(10_000.0, now=DAY_ONE)
    before = engine.snapshot()

    engine.note_reconciliation("BTC/USDT", "venue says 2, we say 1")

    assert engine.snapshot() != before


# --- import graph ---------------------------------------------------------
def test_the_risk_engine_imports_on_its_own_and_the_futures_package_still_imports():
    """``PositionSide`` is type-only here, and this is why.

    ``chimera.futures.__init__`` imports the executor, which imports this
    module. A real import of ``chimera.futures.domain`` from the risk engine
    would therefore be a cycle that only shows up as an ``ImportError`` in
    whichever of the two a process happens to import first — so both orders are
    checked, in a subprocess, from a clean interpreter.
    """
    program = (
        "import sys\n"
        "import chimera.risk\n"
        "assert 'chimera.futures' not in sys.modules, sorted(sys.modules)\n"
        "import chimera.futures\n"
        "assert chimera.futures.PositionSide.LONG.value == 'LONG'\n"
    )
    reversed_program = "import chimera.futures\nimport chimera.risk\n"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    for source in (program, reversed_program):
        result = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        assert result.returncode == 0, result.stderr


def test_a_state_file_missing_the_stale_feed_mark_fails_closed(tmp_path):
    """``None`` is written as ``null``, so an absent key is not "the feed was fresh"."""
    engine = build(tmp_path)
    engine.update_equity(10_000.0, now=DAY_ONE)
    document = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    assert document["stale_feed_since"] is None
    del document["stale_feed_since"]
    (tmp_path / "risk.json").write_text(json.dumps(document), encoding="utf-8")

    revived = build(tmp_path)

    assert revived.halted
    assert "stale_feed_since" in revived.state.halt_reason


# --- a file that could not be read is evidence, not scratch space ---------
def test_an_unreadable_file_is_not_overwritten_by_the_next_mutation(tmp_path):
    """The halt is only half the guarantee; the bytes are the other half.

    Every mutation persists, so without this the first tick after a failed load
    replaces the one record of what the account was doing with an all-defaults
    document — and two steps turn "I cannot tell" into a confident "flat, no
    drawdown, no cooldown".
    """
    engine = build(tmp_path)
    engine.update_equity(120_000.0, now=DAY_ONE)
    engine.set_position_exposure("BTC/USDT:USDT", 50_000.0)
    good = (tmp_path / "risk.json").read_text(encoding="utf-8")
    truncated = good[: len(good) // 2]
    (tmp_path / "risk.json").write_text(truncated, encoding="utf-8")

    revived = build(tmp_path)
    assert revived.halted
    revived.note_feed(0, 0)  # the runner's very next tick
    revived.update_equity(99_000.0, now=DAY_ONE)
    revived.record_order()

    assert (tmp_path / "risk.json").read_text(encoding="utf-8") == truncated


def test_a_resume_after_an_unreadable_file_does_not_survive_a_restart(tmp_path):
    """Resuming is not resolving. The next boot re-reads the same bad bytes."""
    (tmp_path / "risk.json").write_text("{not json", encoding="utf-8")
    engine = build(tmp_path)
    engine.resume()
    assert not engine.halted

    revived = build(tmp_path)

    assert revived.halted, "the unreadable file is still there, so it still halts"
    assert revived.state.halt_reason.startswith("unreadable persisted risk state")


def test_adopting_after_an_unreadable_file_preserves_it_and_resumes_writing(tmp_path):
    (tmp_path / "risk.json").write_text("{not json", encoding="utf-8")
    engine = build(tmp_path)

    preserved = engine.adopt_after_unreadable("disk filled mid-write during the incident")

    assert preserved == tmp_path / "risk.json.corrupt"
    assert preserved.read_text(encoding="utf-8") == "{not json"
    assert engine.halted, "adopting the empty state is not the operator resuming"
    document = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    assert document["schema"] == RISK_STATE_SCHEMA
    engine.update_equity(10_000.0, now=DAY_ONE)
    assert json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))[
        "equity"
    ] == pytest.approx(10_000.0)


def test_adopting_requires_a_written_reason(tmp_path):
    (tmp_path / "risk.json").write_text("{not json", encoding="utf-8")
    engine = build(tmp_path)

    with pytest.raises(RiskViolation):
        engine.adopt_after_unreadable("")

    assert (tmp_path / "risk.json").read_text(encoding="utf-8") == "{not json"


def test_adopting_is_refused_when_the_file_read_cleanly(tmp_path):
    """The mirror: it is a way past a specific failure, not a way to wipe state."""
    engine = build(tmp_path)
    engine.update_equity(10_000.0, now=DAY_ONE)

    with pytest.raises(RiskViolation):
        engine.adopt_after_unreadable("no reason to")


def test_a_state_path_that_cannot_be_examined_fails_closed(tmp_path):
    """A regular file where the containing directory should be.

    ``Path.exists()`` reports that as a confident ``False``, so gating the load
    on it read a state file on a degraded mount as "there is no state file" and
    started the engine on unhalted defaults — losing the halt, the peak equity,
    the cooldown, the order window and any open dispute in one step.
    """
    (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")

    engine = RiskEngine(
        RiskLimits(),
        state_path=tmp_path / "blocked" / "risk.json",
        clock=FakeClock(),
        kill_switch_path=tmp_path / "no-kill-switch",
    )

    assert engine.halted
    assert engine.state.halt_reason.startswith("unreadable persisted risk state")
    assert not entry(engine).allowed


def test_a_fail_closed_halt_reason_names_no_path(tmp_path):
    """``snapshot()`` hashes the reason, so it may not carry this host's paths."""
    (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")

    engine = RiskEngine(
        RiskLimits(),
        state_path=tmp_path / "blocked" / "risk.json",
        clock=FakeClock(),
        kill_switch_path=tmp_path / "no-kill-switch",
    )

    rendered = json.dumps(engine.snapshot())
    assert str(tmp_path) not in rendered
    assert "blocked" not in rendered


# --- boundaries the persisted window and snapshot depend on ---------------
@pytest.mark.parametrize(
    ("gap", "kept"),
    [(59.9, True), (60.0, False), (60.1, False)],
)
def test_the_persisted_order_window_is_exactly_sixty_seconds(tmp_path, gap, kept):
    """Exactly 60 s is out: the window is ``now - t < 60``.

    It decides what a restarted engine believes about the order rate, so the
    edge is load-bearing rather than incidental.
    """
    clock = FakeClock(1_000.0)
    engine = build(tmp_path, RiskLimits(max_orders_per_minute=10), clock)
    engine.record_order()

    clock.now = 1_000.0 + gap
    engine.record_trade_result(1.0)  # any mutation; the point is the write

    document = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    assert document["order_times"] == ([1_000.0] if kept else [])
    assert build(tmp_path, clock=clock).state.order_times == ([1_000.0] if kept else [])


def test_the_snapshot_does_not_depend_on_the_order_things_were_recorded_in(tmp_path):
    """Two routes to one semantic state must serialise to the same bytes.

    Dict iteration is insertion order, so without sorting an engine that set BTC
    then ETH and one that set ETH then BTC would hash differently while holding
    exactly the same exposure.
    """
    first = build(tmp_path, state="a.json")
    first.set_position_exposure("BTC/USDT", 1_000.0)
    first.set_position_exposure("ETH/USDT", 2_000.0)
    first.note_reconciliation("BTC/USDT", "venue says 2")
    first.note_reconciliation("ETH/USDT", "venue says 3")

    second = build(tmp_path, state="b.json")
    second.set_position_exposure("ETH/USDT", 2_000.0)
    second.set_position_exposure("BTC/USDT", 1_000.0)
    second.note_reconciliation("ETH/USDT", "venue says 3")
    second.note_reconciliation("BTC/USDT", "venue says 2")

    assert first.snapshot() == second.snapshot()
    assert json.dumps(first.snapshot()) == json.dumps(second.snapshot())


# --- the engine's clock governs every timestamp it writes -----------------
def test_the_write_time_comes_from_the_engine_clock_not_the_wall_clock(tmp_path):
    engine = build(tmp_path, clock=FakeClock(1_700_000_000.0))
    engine.update_equity(10_000.0, now=DAY_ONE)

    written = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))["updated_at"]

    assert written == datetime.fromtimestamp(1_700_000_000.0, timezone.utc).isoformat()


def test_the_utc_day_comes_from_the_engine_clock_when_no_time_is_given(tmp_path):
    """``day`` and ``day_start_equity`` are persisted and hashed into the snapshot.

    A replayed run must roll its day on the clock that drives its cooldown and
    its order window, not on whatever the host's wall clock happens to say.
    """
    engine = build(tmp_path, clock=FakeClock(1_700_000_000.0))
    engine.update_equity(10_000.0)

    assert engine.state.day == "2023-11-14"
    assert engine.state.day_start_equity == pytest.approx(10_000.0)


def test_a_wipeout_is_recorded_before_it_is_halted_on(tmp_path):
    """The daily report is written from the file; it may not miss the day it mattered."""
    engine = build(tmp_path)
    engine.update_equity(10_000.0, now=DAY_ONE)

    engine.update_equity(0.0, now=DAY_ONE)

    document = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    assert document["halted"] is True
    assert document["equity"] == pytest.approx(0.0)
    assert document["daily_pnl"] == pytest.approx(-10_000.0)
