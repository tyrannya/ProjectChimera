"""R1-b (AEG-1): what a restarting Aegis believes its equity to be.

`build_risk_engine` used to end with an unconditional
``engine.update_equity(float(capital))``, which ran *after* the `RiskEngine`
constructor had already restored the persisted state. Capital is what a FIRST
start is worth; feeding it to `update_equity` on every start made a restart lie
to the drawdown guard, to the daily-loss guard, and to `risk.state_hash`.

Every test here is two-sided: the healthy restart that must NOT halt is paired
with the genuine breach that still must, and the witnesses that the repair
satisfies are paired with the mutants that break them again. A one-sided
"it does not halt" would pass just as well against an engine with no guards at
all.

The limits are the committed campaign's, through `tests.demo_harness`, because
a witness stated against limits a test invented would be a witness about the
test. Section 7.4's values that decide these cases are ``max_drawdown_pct
0.05`` and ``max_daily_loss_pct 0.02``.

Nothing here is scientific evidence. The prices come from
`chimera.demo.fixtures` and the equities are the test author's own arithmetic.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

import chimera.demo.risk_wiring as risk_wiring
from chimera.carry.ledger import CarryLedger
from chimera.demo.risk_wiring import (
    build_risk_engine,
    ledger_equity,
    seed_or_reconcile_equity,
)
from chimera.demo.runner import RunnerState, _risk_hash
from chimera.risk import RiskEngine, RiskState, RiskStateLoad
from tests.demo_harness import CAPITAL, DAY, NEXT_DAY, build, campaign_config

#: The campaign's own drawdown ceiling, restated only so the arithmetic below
#: can be read without opening the config. `committed_limits` asserts it.
MAX_DRAWDOWN_PCT = 0.05
MAX_DAILY_LOSS_PCT = 0.02

#: The roadmap's witness-B construction: a peak at or above this makes
#: ``capital`` itself a drawdown breach, so re-seeding from capital halts.
PEAK_THAT_MAKES_CAPITAL_A_BREACH = float(CAPITAL) / (1.0 - MAX_DRAWDOWN_PCT)

LAST_MINUTE_OF_DAY = f"{DAY}T23:59:30+00:00"
FIRST_MINUTE_OF_NEXT_DAY = f"{NEXT_DAY}T00:00:30+00:00"


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def committed_limits(state_dir: Path):
    """The committed campaign on a TEST profile, and a check that it still is."""
    config = campaign_config(state_dir)
    assert config.limits.max_drawdown_pct == pytest.approx(MAX_DRAWDOWN_PCT)
    assert config.limits.max_daily_loss_pct == pytest.approx(MAX_DAILY_LOSS_PCT)
    return config


def clock_at(stamp: str):
    """A clock frozen at one UTC instant, in the shape `RiskEngine` wants."""
    when = datetime.fromisoformat(stamp)
    assert when.tzinfo is not None, "a demo clock instant is always explicit UTC"
    return lambda: when.timestamp()


def persist_risk_state(state_dir: Path, state: RiskState) -> Path:
    """Write ``state`` through `RiskState.to_dict`, the engine's own writer.

    Hand-rolling the document here would test this file's idea of the schema
    rather than the engine's, and `RiskState.from_dict` refuses a document that
    is missing any field -- which is exactly the refusal a hand-rolled fixture
    would keep tripping over as the state grows.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "risk.json"
    path.write_text(
        json.dumps(state.to_dict(updated_at="2026-09-19T23:59:30+00:00"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def persist_ledger(state_dir: Path, *, equity: float | Decimal) -> Path:
    """Record ``equity`` as the accounting truth, through the ledger's own API.

    `CarryLedger.mark` is what the runner's `mark_to_market` calls, so the value
    reaching ``last_equity`` here arrives the same way it arrives in production.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    ledger = CarryLedger.open(state_dir / "carry_ledger.json", capital=CAPITAL)
    ledger.mark(
        spot_close=Decimal("100"), perp_close=Decimal("100"), equity=Decimal(str(equity))
    )
    ledger.save()
    return ledger.path


def healthy_end_of_day(
    *, equity: float, peak: float, day_start: float, day: str = DAY
) -> RiskState:
    """A persisted state a campaign really could have reached, and a check of it.

    A witness about "a healthy state must not halt on restart" is worth nothing
    if the state it restarts from would already have halted the engine that
    wrote it. Both account guards are therefore re-derived here, against the
    campaign's own limits, before the state is used.
    """
    drawdown = (peak - equity) / peak
    assert drawdown < MAX_DRAWDOWN_PCT, (
        f"this fixture is already a drawdown breach ({drawdown:.4%}); the engine that "
        "wrote it would have halted, so it cannot witness a healthy restart"
    )
    daily_loss = (day_start - equity) / day_start
    assert daily_loss < MAX_DAILY_LOSS_PCT, (
        f"this fixture is already a daily-loss breach ({daily_loss:.4%}); so is this one, "
        "and the engine that wrote it would have halted too"
    )
    return RiskState(
        equity=equity,
        peak_equity=peak,
        day_start_equity=day_start,
        day=day,
        daily_pnl=equity - day_start,
    )


def engine_at(state_dir: Path, *, stamp: str) -> RiskEngine:
    """`build_risk_engine` for the committed campaign, on a frozen clock."""
    return build_risk_engine(
        committed_limits(state_dir),
        capital=CAPITAL,
        state_dir=state_dir,
        clock=clock_at(stamp),
    )


def unconditional_reseed(engine, *, capital, state_dir) -> None:
    """The defect, restored verbatim: `build_risk_engine`'s old last line.

    Every mutation test below installs exactly this over
    `seed_or_reconcile_equity`. A repair whose witnesses still pass against it
    is not witnessing the repair.
    """
    engine.update_equity(float(capital))


def no_reconciliation(engine, *, capital, state_dir) -> None:
    """The half-repair: stop re-seeding, but never check the ledger either."""
    if engine.load_outcome in (RiskStateLoad.MISSING, RiskStateLoad.LEGACY):
        engine.update_equity(float(capital))


# ---------------------------------------------------------------------------
# 1. first start vs restart
# ---------------------------------------------------------------------------
def test_a_genuine_first_start_still_seeds_the_configured_capital(tmp_path):
    """The behaviour that was always right, and that the repair must keep."""
    engine = engine_at(tmp_path / "state", stamp=LAST_MINUTE_OF_DAY)

    assert engine.load_outcome is RiskStateLoad.MISSING
    assert engine.state.equity == pytest.approx(float(CAPITAL))
    assert engine.state.peak_equity == pytest.approx(float(CAPITAL))
    assert engine.state.day_start_equity == pytest.approx(float(CAPITAL))
    assert engine.state.day == DAY
    assert not engine.state.halted
    assert (tmp_path / "state" / "risk.json").is_file()


def test_a_restart_does_not_re_seed_equity_from_capital(tmp_path):
    """AEG-1 itself: the persisted equity survives construction untouched."""
    state_dir = tmp_path / "state"
    persisted = healthy_end_of_day(equity=970_000.0, peak=1_000_000.0, day_start=985_000.0)
    persist_risk_state(state_dir, persisted)
    persist_ledger(state_dir, equity=970_000.0)

    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert engine.load_outcome is RiskStateLoad.LOADED
    assert engine.state.equity == pytest.approx(970_000.0)
    assert engine.state.equity != pytest.approx(float(CAPITAL))


def test_a_restart_erases_no_part_of_the_persisted_state(tmp_path):
    """Not only equity: every Aegis field this task does not own is untouched.

    Compared as a whole dataclass rather than field by field, so a field ADDED
    to `RiskState` later is covered by this test on the day it is added rather
    than on the day somebody remembers to extend a list here.
    """
    state_dir = tmp_path / "state"
    persisted = RiskState(
        equity=1_020_000.0,
        peak_equity=1_060_000.0,
        day_start_equity=1_010_000.0,
        day=DAY,
        daily_pnl=10_000.0,
        open_positions={"BTC/USDT:USDT": 12_345.0},
        order_times=[1.0, 2.0, 3.0],
        consecutive_losses=2,
        cooldown_until=99_999.0,
        halted=False,
        halt_reason="",
        kill_switch=False,
        stale_feed_since=17.5,
        reconciliation_disputed={"BTC/USDT:USDT": "venue says 3, we say 2"},
        funding_adverse_streak=2,
        funding_halt=False,
    )
    persist_risk_state(state_dir, persisted)
    persist_ledger(state_dir, equity=1_020_000.0)

    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert engine.state == persisted


#: A persisted campaign whose equity happens to BE the configured capital, and
#: which is nonetheless healthy: 3.85% off a peak it really earned and 0.99%
#: below the day's start. Both are inside section 7.4's ceilings, so the engine
#: that wrote this state had no reason to halt.
EQUITY_EQUALS_CAPITAL = dict(equity=float(CAPITAL), peak=1_040_000.0, day_start=1_010_000.0)


def _equity_equals_capital_state_dir(tmp_path: Path) -> Path:
    state_dir = tmp_path / "state"
    persist_risk_state(state_dir, healthy_end_of_day(**EQUITY_EQUALS_CAPITAL))
    persist_ledger(state_dir, equity=float(CAPITAL))
    return state_dir


def test_the_first_start_is_not_inferred_from_an_equity_equal_to_capital(tmp_path):
    """An equity equal to capital is also what a campaign that gave back its gains is.

    This is the case that makes `RiskEngine.load_outcome` load-bearing rather
    than decorative. The persisted equity here IS the configured capital, so any
    implementation that recognises a first start by comparing the two treats a
    restart as one -- and rolls the day and re-measures the drawdown from a peak
    the campaign really earned.
    """
    state_dir = _equity_equals_capital_state_dir(tmp_path)

    engine = engine_at(state_dir, stamp=FIRST_MINUTE_OF_NEXT_DAY)

    assert engine.load_outcome is RiskStateLoad.LOADED
    assert not engine.state.halted, engine.state.halt_reason
    assert engine.state.peak_equity == pytest.approx(EQUITY_EQUALS_CAPITAL["peak"])
    assert engine.state.day == DAY, (
        "the day rolls on the first real equity update, not on construction: "
        "rolling it here would open the new day's budget at a number no mark produced"
    )
    assert engine.state.day_start_equity == pytest.approx(EQUITY_EQUALS_CAPITAL["day_start"])


def test_the_pre_fix_re_seed_cannot_tell_that_restart_from_a_first_start(
    monkeypatch, tmp_path
):
    """The other side of it: the defect treats the restart above as a first start."""
    state_dir = _equity_equals_capital_state_dir(tmp_path)
    monkeypatch.setattr(risk_wiring, "seed_or_reconcile_equity", unconditional_reseed)

    engine = engine_at(state_dir, stamp=FIRST_MINUTE_OF_NEXT_DAY)

    assert engine.state.day == NEXT_DAY
    assert engine.state.day_start_equity == pytest.approx(float(CAPITAL))


def test_an_unreadable_risk_state_is_not_given_a_fabricated_equity(tmp_path):
    """The fail-closed state means "no claim". Seeding it would make one."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "risk.json").write_text("{not json", encoding="utf-8")

    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert engine.load_outcome is RiskStateLoad.UNREADABLE
    assert engine.state.halted
    assert engine.state.equity == 0.0, "an absence of a claim, not a claim of capital"
    assert engine.state.peak_equity == 0.0
    preserved = (state_dir / "risk.json").read_text(encoding="utf-8")
    assert preserved == "{not json", "the unreadable file is preserved for inspection"


def test_a_legacy_halt_only_file_is_still_seeded_from_capital(tmp_path):
    """The pre-schema record makes no claim about equity, so there is none to keep."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "risk.json").write_text(
        json.dumps({"halted": True, "halt_reason": "operator stop"}), encoding="utf-8"
    )

    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert engine.load_outcome is RiskStateLoad.LEGACY
    assert engine.state.equity == pytest.approx(float(CAPITAL))
    # The seed records an equity; it does not clear a persisted halt.
    assert engine.state.halted
    assert engine.state.halt_reason == "operator stop"


# ---------------------------------------------------------------------------
# 2. WITNESS A -- a restart across a UTC day boundary
# ---------------------------------------------------------------------------
#: Witness A's persisted campaign: 3.0% below capital in total, and 1.52% below
#: the day's start -- inside the 2% daily budget it was measured against, so the
#: engine that wrote it was healthy.
#:
#: The peak sits close to the equity ON PURPOSE. Witness A is about the DAILY
#: budget, and a peak at capital would leave only 2% of drawdown headroom -- so
#: the breach half below would trip `max_drawdown_pct` first and the test would
#: be asserting the wrong guard. At this peak the whole of witness A stays a
#: statement about `max_daily_loss_pct`; witness B is where the peak does the work.
WITNESS_A = dict(equity=970_000.0, peak=990_000.0, day_start=985_000.0)

#: The next ordinary minute of the new day. A further 0.21% below, which is
#: 0.21% of the new day's real budget and 3.2% of a budget re-opened at capital.
WITNESS_A_NEXT_MINUTE = 968_000.0


def _witness_a_state_dir(tmp_path: Path) -> Path:
    state_dir = tmp_path / "state"
    persist_risk_state(state_dir, healthy_end_of_day(**WITNESS_A))
    persist_ledger(state_dir, equity=WITNESS_A["equity"])
    assert (
        abs(WITNESS_A["equity"] - float(CAPITAL)) / float(CAPITAL) >= 0.02
    ), "witness A is stated for a mark/equity move of at least 2%"
    return state_dir


def test_witness_a_a_day_boundary_restart_does_not_halt_a_healthy_campaign(tmp_path):
    """Roadmap R1-b (a): restart across UTC midnight after a >= 2% move -> no halt."""
    state_dir = _witness_a_state_dir(tmp_path)

    engine = engine_at(state_dir, stamp=FIRST_MINUTE_OF_NEXT_DAY)

    assert not engine.state.halted, engine.state.halt_reason
    assert engine.state.day == DAY
    assert engine.state.day_start_equity == pytest.approx(WITNESS_A["day_start"])

    # And the first ordinary minute of the new day is decided against a budget
    # that opens at what the account is actually worth.
    engine.update_equity(WITNESS_A_NEXT_MINUTE)

    assert not engine.state.halted, engine.state.halt_reason
    assert engine.state.day == NEXT_DAY
    assert engine.state.day_start_equity == pytest.approx(WITNESS_A_NEXT_MINUTE)


def test_witness_a_the_pre_fix_re_seed_halts_that_healthy_campaign(monkeypatch, tmp_path):
    """The other side: the defect, reproduced, halting on a breach that never happened."""
    state_dir = _witness_a_state_dir(tmp_path)
    monkeypatch.setattr(risk_wiring, "seed_or_reconcile_equity", unconditional_reseed)

    engine = engine_at(state_dir, stamp=FIRST_MINUTE_OF_NEXT_DAY)

    assert engine.state.day == NEXT_DAY
    assert engine.state.day_start_equity == pytest.approx(float(CAPITAL)), (
        "the re-seed opens the new day's loss budget at capital rather than at "
        "what the account was worth when the day rolled"
    )

    engine.update_equity(WITNESS_A_NEXT_MINUTE)

    assert engine.state.halted, "the defect must be reproducible, or nothing is repaired"
    assert "max daily loss breached" in engine.state.halt_reason


def test_witness_a_a_genuine_daily_loss_breach_still_halts(tmp_path):
    """Two-sided. The repair must not be "the daily-loss guard stopped firing"."""
    state_dir = _witness_a_state_dir(tmp_path)
    engine = engine_at(state_dir, stamp=FIRST_MINUTE_OF_NEXT_DAY)

    engine.update_equity(WITNESS_A_NEXT_MINUTE)
    assert not engine.state.halted
    # 2.1% below the day's real start: a genuine breach of the campaign's 2%.
    engine.update_equity(WITNESS_A_NEXT_MINUTE * (1.0 - 0.021))

    assert engine.state.halted
    assert "max daily loss breached" in engine.state.halt_reason


# ---------------------------------------------------------------------------
# 3. WITNESS B -- a persisted peak above the drawdown ceiling
# ---------------------------------------------------------------------------
#: Witness B's persisted campaign. The peak satisfies the roadmap's
#: ``peak >= capital / (1 - max_drawdown_pct)``, which is precisely the
#: condition that makes re-seeding equity to capital a drawdown breach.
WITNESS_B = dict(equity=1_020_000.0, peak=1_060_000.0, day_start=1_010_000.0)


def _witness_b_state_dir(tmp_path: Path) -> Path:
    assert WITNESS_B["peak"] >= PEAK_THAT_MAKES_CAPITAL_A_BREACH, (
        "witness B is stated for a peak at or above capital / (1 - max_drawdown_pct); "
        "below it the re-seed produces no breach and the witness is vacuous"
    )
    state_dir = tmp_path / "state"
    persist_risk_state(state_dir, healthy_end_of_day(**WITNESS_B))
    persist_ledger(state_dir, equity=WITNESS_B["equity"])
    return state_dir


def test_witness_b_a_persisted_peak_does_not_create_a_false_drawdown_halt(tmp_path):
    """Roadmap R1-b (b), healthy side: restart with that peak -> no halt."""
    state_dir = _witness_b_state_dir(tmp_path)

    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert not engine.state.halted, engine.state.halt_reason
    assert engine.state.peak_equity == pytest.approx(WITNESS_B["peak"])
    assert engine.state.equity == pytest.approx(WITNESS_B["equity"])
    assert engine.current_drawdown() < MAX_DRAWDOWN_PCT


def test_witness_b_a_genuine_drawdown_breach_still_halts_after_that_restart(tmp_path):
    """Roadmap R1-b (b), breach side. The peak is preserved to be MEASURED FROM."""
    state_dir = _witness_b_state_dir(tmp_path)
    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    # A real mark 5.66% below the real peak: past the campaign's 5%.
    breach = WITNESS_B["peak"] * (1.0 - MAX_DRAWDOWN_PCT) - 1.0
    engine.update_equity(breach)

    assert engine.state.halted
    assert "max drawdown breached" in engine.state.halt_reason


def test_witness_b_the_pre_fix_re_seed_halts_on_its_own_startup(monkeypatch, tmp_path):
    """The defect reproduced: the campaign halts before it decides anything."""
    state_dir = _witness_b_state_dir(tmp_path)
    monkeypatch.setattr(risk_wiring, "seed_or_reconcile_equity", unconditional_reseed)

    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert engine.state.halted, "the defect must be reproducible"
    assert "max drawdown breached" in engine.state.halt_reason


# ---------------------------------------------------------------------------
# 4. WITNESS C -- replay parity across a day boundary with a restart
# ---------------------------------------------------------------------------
#: The synthetic day is 1440 minutes. Deciding from 1435 puts the UTC midnight
#: inside a ten-minute run, which is what the witness needs; deciding the whole
#: day to reach it would say nothing extra.
FIRST_DECIDED_MINUTE = 1435
MINUTES_DECIDED = 10
#: The restart lands so that the last minute DECIDED before it still belongs to
#: `DAY` while the first minute decided after it belongs to `NEXT_DAY`. The
#: runner stamps a decision with the minute's CLOSE, so minute 1439 closes at
#: 00:00 of the next day and the boundary is the split below, not 1440.
MINUTES_BEFORE_RESTART = 4


def _campaign(where: Path, *, restart_at: int | None):
    """One campaign over the boundary, optionally stopped and restarted at it."""
    harness = build(where, days=(DAY, NEXT_DAY))
    first = harness.first_minute_ms() + FIRST_DECIDED_MINUTE * 60_000
    if restart_at is None:
        harness.run(MINUTES_DECIDED, start=first)
        return harness

    harness.run(restart_at, start=first)
    config = harness.runner.config
    harness.runner.shutdown("planned restart at the UTC day boundary")

    resumed = build(where, days=(DAY, NEXT_DAY), config=config)
    assert resumed.runner.risk.load_outcome is RiskStateLoad.LOADED, (
        "the restarted campaign must really be restoring a persisted state, or "
        "this witness is a comparison of two first starts"
    )
    resumed.run(MINUTES_DECIDED - restart_at, start=first + restart_at * 60_000)
    return resumed


def test_witness_c_a_restart_at_the_day_boundary_reaches_the_same_state_hash(tmp_path):
    """Roadmap R1-b (c): uninterrupted and restarted runs agree on `risk.state_hash`."""
    uninterrupted = _campaign(tmp_path / "uninterrupted", restart_at=None)
    restarted = _campaign(tmp_path / "restarted", restart_at=MINUTES_BEFORE_RESTART)

    assert _risk_hash(uninterrupted.runner.risk) == _risk_hash(restarted.runner.risk)
    assert uninterrupted.runner.risk.state == restarted.runner.risk.state

    # Not vacuous: the fixture's equity has to differ from capital, or a run
    # that re-seeded capital would reach the same hash by coincidence.
    assert uninterrupted.runner.risk.state.day_start_equity != pytest.approx(
        float(CAPITAL)
    ), "the boundary must move the day's starting equity away from capital"


def test_witness_c_crosses_the_day_boundary_it_claims_to(tmp_path):
    """Otherwise the parity above is a parity across no boundary at all."""
    before = build(tmp_path / "probe", days=(DAY, NEXT_DAY))
    first = before.first_minute_ms() + FIRST_DECIDED_MINUTE * 60_000
    before.run(MINUTES_BEFORE_RESTART, start=first)
    assert before.runner.risk.state.day == DAY

    after = _campaign(tmp_path / "after", restart_at=None)
    assert after.runner.risk.state.day == NEXT_DAY


def test_witness_c_fails_under_the_pre_fix_re_seed(monkeypatch, tmp_path):
    """The mutation: restore the old last line and the hashes must diverge."""
    uninterrupted = _campaign(tmp_path / "uninterrupted", restart_at=None)
    healthy = _risk_hash(uninterrupted.runner.risk)

    monkeypatch.setattr(risk_wiring, "seed_or_reconcile_equity", unconditional_reseed)
    restarted = _campaign(tmp_path / "restarted", restart_at=MINUTES_BEFORE_RESTART)

    assert _risk_hash(restarted.runner.risk) != healthy
    assert restarted.runner.risk.state.day_start_equity == pytest.approx(float(CAPITAL))


# ---------------------------------------------------------------------------
# 5. reconciliation against the ledger
# ---------------------------------------------------------------------------
def test_a_healthy_restart_reconciles_because_the_two_files_really_do_agree(tmp_path):
    """The control the dispute tests need: a real campaign leaves them equal.

    If an ordinary clean stop left the risk state and the ledger disagreeing,
    every restart below would dispute and the guard would be noise rather than a
    guard. This asserts the invariant the reconciliation is built on, from a
    campaign that actually ran rather than from fixtures written by hand.
    """
    harness = build(tmp_path, days=(DAY,))
    harness.run(20)
    harness.runner.shutdown("clean stop")
    state_dir = harness.state_dir

    persisted = json.loads((state_dir / "risk.json").read_text(encoding="utf-8"))
    accounted = ledger_equity(state_dir, capital=CAPITAL)

    assert accounted is not None
    assert float(accounted) == persisted["equity"], (
        "exact, not approximate: the runner hands `update_equity` the very "
        "`float()` of the Decimal the ledger recorded one statement earlier"
    )
    seeded = float(CAPITAL)
    assert persisted["equity"] != seeded, "a campaign that moved nothing proves nothing"


def test_a_disagreement_with_the_ledger_is_a_dispute(tmp_path):
    """Neither side is overwritten to make them agree; the engine halts saying so."""
    state_dir = tmp_path / "state"
    persist_risk_state(state_dir, healthy_end_of_day(**WITNESS_B))
    persist_ledger(state_dir, equity=999_000.0)  # the accounting says something else
    ledger_before = (state_dir / "carry_ledger.json").read_text(encoding="utf-8")

    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert engine.state.halted
    assert "equity_dispute" in engine.state.halt_reason
    assert "1020000" in engine.state.halt_reason.replace(",", "")
    assert "999000" in engine.state.halt_reason.replace(",", "")
    # Neither side overwritten: the ledger file is untouched and the risk state
    # keeps every number it loaded, with the halt written beside them.
    assert (state_dir / "carry_ledger.json").read_text(encoding="utf-8") == ledger_before
    assert engine.state.equity == pytest.approx(WITNESS_B["equity"])
    assert engine.state.peak_equity == pytest.approx(WITNESS_B["peak"])
    assert engine.state.day_start_equity == pytest.approx(WITNESS_B["day_start"])


def test_removing_the_reconciliation_leaves_that_disagreement_unnoticed(monkeypatch, tmp_path):
    """The mutation for the dispute half: stop checking, and nothing is said."""
    state_dir = tmp_path / "state"
    persist_risk_state(state_dir, healthy_end_of_day(**WITNESS_B))
    persist_ledger(state_dir, equity=999_000.0)
    monkeypatch.setattr(risk_wiring, "seed_or_reconcile_equity", no_reconciliation)

    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert not engine.state.halted, (
        "the half-repair must really be silent here, or the dispute witness above "
        "is passing for some other reason"
    )


def test_an_unreadable_ledger_is_a_dispute_rather_than_a_skipped_check(tmp_path):
    """A guard the failure it guards against can switch off is not a guard."""
    state_dir = tmp_path / "state"
    persist_risk_state(state_dir, healthy_end_of_day(**WITNESS_B))
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "carry_ledger.json").write_text("{ truncated", encoding="utf-8")

    assert ledger_equity(state_dir, capital=CAPITAL) is None

    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert engine.state.halted
    assert "equity_reconciliation" in engine.state.halt_reason
    assert (state_dir / "carry_ledger.json").read_text(encoding="utf-8") == "{ truncated"


def test_a_ledger_that_has_never_been_marked_accounts_for_the_configured_capital(
    tmp_path,
):
    """The runner's own reading of an unmarked ledger, not a second one.

    `DemoRunner._portfolio` is ``ledger.last_equity if ... is not None else
    self.capital``; a first start that persisted its seed and died before the
    first mark must therefore restart without a dispute.
    """
    state_dir = tmp_path / "state"
    first = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)
    assert first.state.equity == pytest.approx(float(CAPITAL))
    assert not (state_dir / "carry_ledger.json").exists()

    resumed = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert resumed.load_outcome is RiskStateLoad.LOADED
    assert not resumed.state.halted, resumed.state.halt_reason
    assert ledger_equity(state_dir, capital=CAPITAL) == Decimal(CAPITAL)


def test_repeated_restarts_are_idempotent_with_respect_to_risk_state(tmp_path):
    """Restarting N times is restarting once. Both for a healthy state..."""
    state_dir = tmp_path / "state"
    persist_risk_state(state_dir, healthy_end_of_day(**WITNESS_B))
    persist_ledger(state_dir, equity=WITNESS_B["equity"])

    states = [engine_at(state_dir, stamp=FIRST_MINUTE_OF_NEXT_DAY).state for _ in range(4)]

    assert states[0] == states[1] == states[2] == states[3]
    assert not states[-1].halted


def test_repeated_restarts_are_idempotent_for_a_disputed_state_too(tmp_path):
    """...and for a disputed one, where the halt is itself written to the file."""
    state_dir = tmp_path / "state"
    persist_risk_state(state_dir, healthy_end_of_day(**WITNESS_B))
    persist_ledger(state_dir, equity=999_000.0)

    first = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)
    after_first = (state_dir / "risk.json").read_text(encoding="utf-8")
    second = engine_at(state_dir, stamp=FIRST_MINUTE_OF_NEXT_DAY)

    assert first.state.halted and second.state.halted
    assert first.state == second.state
    assert (state_dir / "risk.json").read_text(encoding="utf-8") == after_first, (
        "an already-halted engine re-halting writes nothing, so a second restart "
        "cannot move `risk.state_hash`"
    )


# ---------------------------------------------------------------------------
# 6. the seam itself
# ---------------------------------------------------------------------------
def test_build_risk_engine_no_longer_re_seeds_unconditionally():
    """The literal R1-b clause, read off the source it is about.

    A behavioural test can be satisfied by an `update_equity(capital)` that
    happens to be harmless in the cases the tests cover. The clause is about the
    call not being there.
    """
    import inspect

    source = inspect.getsource(risk_wiring.build_risk_engine)
    assert "update_equity" not in source, (
        "`build_risk_engine` must not call `update_equity` itself; whether capital "
        "seeds equity is `seed_or_reconcile_equity`'s decision"
    )
    assert "seed_or_reconcile_equity" in source


def test_seed_or_reconcile_equity_is_reached_through_the_module_attribute(tmp_path):
    """The mutants above are only meaningful if `build_risk_engine` really uses it."""
    calls: list[tuple] = []

    def spy(engine, *, capital, state_dir):
        calls.append((engine, capital, Path(state_dir)))

    original = risk_wiring.seed_or_reconcile_equity
    try:
        risk_wiring.seed_or_reconcile_equity = spy
        engine = engine_at(tmp_path / "state", stamp=LAST_MINUTE_OF_DAY)
    finally:
        risk_wiring.seed_or_reconcile_equity = original

    assert len(calls) == 1
    assert calls[0][0] is engine
    assert calls[0][1] == CAPITAL
    assert calls[0][2] == tmp_path / "state"
    assert engine.state.equity == 0.0, "the spy seeded nothing, so nothing was seeded"


def test_seed_or_reconcile_equity_takes_no_decision_for_an_unreadable_state(tmp_path):
    """Called directly, so the UNREADABLE branch is exercised without a ledger."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "risk.json").write_text("[]", encoding="utf-8")
    engine = RiskEngine(state_path=state_dir / "risk.json", clock=clock_at(LAST_MINUTE_OF_DAY))
    assert engine.load_outcome is RiskStateLoad.UNREADABLE
    before = engine.state

    seed_or_reconcile_equity(engine, capital=CAPITAL, state_dir=state_dir)

    assert engine.state == before
    assert engine.state.equity == 0.0


# ---------------------------------------------------------------------------
# 7. the window this item deliberately does not close
# ---------------------------------------------------------------------------
#: `DemoRunner.tick` calls `_save_ledger()` and then `update_equity()`. A process
#: killed between those two statements leaves the ledger one mark ahead of the
#: risk state. In this synthetic fixture the hedge is delta-neutral, so the mark
#: only moves the equity at a funding settlement -- the minute whose CLOSE is the
#: 08:00 boundary, which is this one.
MINUTE_BEFORE_A_SETTLEMENT = 479


class _SimulatedKill(Exception):
    """Stands in for a SIGKILL arriving at one exact statement."""


def test_a_kill_between_the_ledger_write_and_update_equity_is_reported_as_a_dispute(
    tmp_path,
):
    """A KNOWN window, pinned here rather than left to be discovered in a soak.

    This is not a defect introduced by R1-b and it is not R1-b's to close. The
    ordering is the runner's, the crash semantics are canonical **R1-i**'s ("kill
    at every persistence step, recover, assert consistency"), and R1-b does not
    reorder the runner's writes. What R1-b changes is what the restart then does
    about it: on the previous build the same crash restarted with equity silently
    re-seeded to `capital`, so the disagreement was not merely unreported, it was
    overwritten by a third number that neither file held. Here it is named.

    The test exists so that the behaviour is a decision with a witness rather
    than a surprise, and so that R1-i changing it is a loud failure here.
    """
    harness = build(tmp_path, days=(DAY,))
    first = harness.first_minute_ms()
    harness.run(MINUTE_BEFORE_A_SETTLEMENT, start=first)

    def killed(*args, **kwargs):
        raise _SimulatedKill()

    harness.runner.risk.update_equity = killed
    with pytest.raises(_SimulatedKill):
        harness.tick(first + MINUTE_BEFORE_A_SETTLEMENT * 60_000)

    config = harness.runner.config
    persisted = json.loads((harness.state_dir / "risk.json").read_text(encoding="utf-8"))
    accounted = ledger_equity(harness.state_dir, capital=CAPITAL)
    assert accounted is not None
    assert float(accounted) != persisted["equity"], (
        "this fixture's equity must actually move at this minute, or the window "
        "is not being exercised and the assertions below prove nothing. The "
        "carry hedge is delta-neutral, so the mark moves the equity only at a "
        "funding settlement; re-pick MINUTE_BEFORE_A_SETTLEMENT if that changes"
    )

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    state = resumed.runner.start()

    assert state is RunnerState.HALT
    assert resumed.runner.risk.state.halted
    assert "equity_dispute" in resumed.runner.risk.state.halt_reason
    # Both numbers are on record, and neither file was rewritten to agree.
    assert str(persisted["equity"]) in resumed.runner.risk.state.halt_reason
    assert str(accounted) in resumed.runner.risk.state.halt_reason
    assert resumed.runner.risk.state.equity == pytest.approx(persisted["equity"])
    assert ledger_equity(resumed.state_dir, capital=CAPITAL) == accounted
