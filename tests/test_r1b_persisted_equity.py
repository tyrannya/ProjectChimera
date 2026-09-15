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
from chimera.carry.ledger import CarryLedger, LoadOutcome
from chimera.demo.risk_wiring import (
    EQUITY_DISPUTE_PREFIX,
    EQUITY_RECONCILIATION_PREFIX,
    build_risk_engine,
    is_equity_reconciliation_halt,
    ledger_equity,
    seed_or_reconcile_equity,
)
from chimera.demo.runner import RunnerError, RunnerState, _risk_hash
from chimera.risk import RiskEngine, RiskState, RiskStateLoad, RiskViolation
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
    assert abs(WITNESS_A["equity"] - float(CAPITAL)) / float(CAPITAL) >= 0.02, (
        "witness A is stated for a persisted equity at least 2% away from the "
        "configured capital -- the gap the re-seed turns into one day's loss. The "
        "witness's own docstring says why that is not an intraday move."
    )
    return state_dir


def test_witness_a_a_day_boundary_restart_does_not_halt_a_healthy_campaign(tmp_path):
    """Roadmap R1-b (a): a restart across UTC midnight after a >= 2% move -> no halt.

    WHICH >= 2% MOVE. The roadmap writes the clause as "a >= 2% intraday mark
    move", and the quantity this witness is stated for is the gap between what
    the account is WORTH and the configured capital -- 3.0% here. That is the
    quantity the defect is a function of: the re-seed opened the new day's loss
    budget at `capital`, so the first honest mark of the new day read as a loss
    of very nearly that whole gap, and 2% is where the campaign's
    `max_daily_loss_pct` turns it into a halt.

    It is deliberately not a >= 2% move WITHIN the persisted day, and no witness
    of a healthy restart could be: the campaign's daily-loss limit is 2%, so a
    persisted state holding a >= 2% intraday loss is one the engine that wrote it
    would already have halted on. `healthy_end_of_day` re-derives both guards and
    refuses exactly that fixture, so the restriction is enforced rather than
    merely described. The persisted intraday move here is 1.52%.
    """
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


# ---------------------------------------------------------------------------
# 8. the crash-free divergence: an ordinary operator flatten
# ---------------------------------------------------------------------------
# `DemoRunner.flatten` marks the ledger and persists it. It used to tell Aegis
# nothing, so every ordinary flatten left `risk.json` and `carry_ledger.json`
# stating different equities -- with no crash anywhere -- and R1-b's restart
# reconciliation then halted the campaign on its next start. The divergence
# pre-dates R1-b; what R1-b changed was the consequence, from a silent re-seed to
# a halt. It is fixed in the runner: `flatten` hands the flattened equity to the
# same single writer `tick` uses.
MINUTES_BEFORE_FLATTEN = 12


def _flattened_campaign(tmp_path: Path, *, tell_aegis: bool = True):
    """Run a few real minutes, then flatten, through the production paths."""
    harness = build(tmp_path, days=(DAY,))
    harness.run(MINUTES_BEFORE_FLATTEN)
    assert (
        harness.runner.state is not RunnerState.HALT
    ), "the fixture halted before it flattened"
    if not tell_aegis:
        # The mutant: `flatten` marks and persists the ledger and leaves Aegis's
        # equity where it was, which is exactly what it used to do.
        harness.runner.risk.update_equity = lambda *a, **k: None
    harness.runner.flatten("operator: reduce to flat for maintenance")
    return harness


def _two_equities(state_dir: Path) -> tuple[float, Decimal]:
    persisted = json.loads((state_dir / "risk.json").read_text(encoding="utf-8"))
    accounted = ledger_equity(state_dir, capital=CAPITAL)
    assert accounted is not None
    return persisted["equity"], accounted


def test_an_ordinary_flatten_leaves_the_two_persisted_equities_agreeing(tmp_path):
    """The B1 defect, closed at its source rather than absorbed by the dispute."""
    harness = _flattened_campaign(tmp_path)

    persisted, accounted = _two_equities(harness.state_dir)
    assert float(accounted) == persisted
    assert not harness.runner.risk.state.halted, harness.runner.risk.state.halt_reason


def test_a_flatten_really_moves_the_equity_it_reports(tmp_path):
    """Otherwise the agreement above is an agreement about a number that never moved."""
    harness = build(tmp_path, days=(DAY,))
    harness.run(MINUTES_BEFORE_FLATTEN)
    before, _ = _two_equities(harness.state_dir)

    harness.runner.flatten("operator: reduce to flat for maintenance")

    after, _ = _two_equities(harness.state_dir)
    assert after != before, (
        "the reduce orders' fees and slippage must move the equity, or this whole "
        "section is witnessing nothing"
    )


def test_a_flatten_then_a_restart_runs_rather_than_disputing(tmp_path):
    """The regression witness: flatten, lose the process, come back, keep going.

    Nothing here edits a persistence file: the campaign is flattened through
    `DemoRunner.flatten`, the runner is thrown away, and a second one is built
    from what is on disk by the same factories production uses.
    """
    harness = _flattened_campaign(tmp_path)
    config = harness.runner.config
    state_dir = harness.state_dir
    persisted, accounted = _two_equities(state_dir)

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    state = resumed.runner.start()

    assert state is not RunnerState.HALT, resumed.runner.halt_reason
    assert not resumed.runner.risk.state.halted, resumed.runner.risk.state.halt_reason
    assert (
        resumed.runner.risk.load_outcome is RiskStateLoad.LOADED
    ), "a restart that did not restore a persisted state is not the case under test"
    assert resumed.runner.risk.state.equity == pytest.approx(persisted)
    assert float(accounted) == persisted

    # And it really can decide again, which is the property the operator cares
    # about: a campaign that starts and cannot tick is not recovered.
    minute = resumed.runner.cursor.next_minute_ms()
    assert minute is not None
    resumed.tick(minute)
    assert resumed.runner.state is not RunnerState.HALT, resumed.runner.halt_reason


def test_a_flatten_that_tells_aegis_nothing_wedges_the_next_start(tmp_path):
    """The other side: the defect, reproduced, so the repair is witnessed.

    This is what the independent review demonstrated, and it is why the fix is in
    `flatten` rather than only in the runbook.
    """
    harness = _flattened_campaign(tmp_path, tell_aegis=False)
    config = harness.runner.config
    persisted, accounted = _two_equities(harness.state_dir)
    assert float(accounted) != persisted, "the mutant must actually diverge the two files"

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)

    assert resumed.runner.start() is RunnerState.HALT
    assert resumed.runner.risk.state.halt_reason.startswith(EQUITY_DISPUTE_PREFIX)


def test_a_flatten_tells_aegis_nothing_when_the_ledger_may_not_speak(tmp_path):
    """The equity would come from a placeholder, and Aegis is not told fictions.

    The same condition the OPERATOR record uses for its `ledger_effect` block:
    where `_save_ledger` skips and the record omits the economics, Aegis is not
    handed them either.
    """
    harness = build(tmp_path, days=(DAY,))
    harness.run(MINUTES_BEFORE_FLATTEN)
    before = harness.runner.risk.state.equity
    (harness.state_dir / "carry_ledger.json").write_text("{ truncated", encoding="utf-8")
    config = harness.runner.config

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    resumed.runner.start()
    assert not resumed.runner._ledger_may_speak()
    resumed.runner.flatten("operator: removing exposure on a damaged ledger")

    assert resumed.runner.risk.state.equity == pytest.approx(before), (
        "a placeholder ledger's equity is not an accounting claim and may not "
        "become Aegis's"
    )


# ---------------------------------------------------------------------------
# 9. the clearing path: `resolve --equity`
# ---------------------------------------------------------------------------
# The dispute a restart raises is unlike every other halt the campaign can carry:
# its cause is re-read at EVERY construction, so `resume` -- which clears the flag
# and touches neither equity -- is undone by the next process before a tick can
# re-synchronise them. Without a clearing path the only way out is to edit a
# state file, which the runbook forbids and R1's acceptance ("zero manual state
# edits") rules out. Canonical R1-i asks that every dispute kind have exactly
# one clearing path; this is R1-b's.


def _crashed_campaign(tmp_path: Path):
    """A real divergence, made the way the runner really makes one.

    The kill lands between `_save_ledger()` and `update_equity()` in `tick`, which
    is the window this item pins rather than closes. Nothing is hand-edited.
    """
    harness = build(tmp_path, days=(DAY,))
    first = harness.first_minute_ms()
    harness.run(MINUTE_BEFORE_A_SETTLEMENT, start=first)

    def killed(*args, **kwargs):
        raise _SimulatedKill()

    harness.runner.risk.update_equity = killed
    with pytest.raises(_SimulatedKill):
        harness.tick(first + MINUTE_BEFORE_A_SETTLEMENT * 60_000)
    persisted, accounted = _two_equities(harness.state_dir)
    assert float(accounted) != persisted, "the crash must actually diverge the two files"
    return harness.runner.config, persisted, accounted


def test_a_disputed_restart_is_settled_by_resolve_equity_and_then_runs(tmp_path):
    """The whole recovery, end to end, through production paths only.

    crash -> restart HALTS -> `resolve --equity --note` -> restart RUNS. The
    recovery never touches `risk.json` or `carry_ledger.json` directly; it is the
    operator command doing it, and the proof it worked is that the campaign
    decides another minute afterwards.
    """
    config, persisted, accounted = _crashed_campaign(tmp_path)

    halted = build(tmp_path, days=(DAY,), config=config, start=False)
    assert halted.runner.start() is RunnerState.HALT
    assert halted.runner.risk.state.halt_reason.startswith(EQUITY_DISPUTE_PREFIX)

    # A separate process, as the CLI really is: the settlement has to survive
    # being written down and read back, not merely hold in the one that made it.
    settling = build(tmp_path, days=(DAY,), config=config, start=False)
    settling.runner.start()
    outcome = settling.runner.resolve_equity(
        "operator: crash at the settlement; ledger is right"
    )

    assert outcome.record_hash
    assert not settling.runner.risk.state.halted, settling.runner.risk.state.halt_reason
    assert settling.runner.risk.state.equity == pytest.approx(float(accounted)), (
        "the accounting is what is adopted, and it is the number the next start "
        "will compare against"
    )

    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    state = resumed.runner.start()

    assert state is not RunnerState.HALT, resumed.runner.halt_reason
    assert not resumed.runner.risk.state.halted, resumed.runner.risk.state.halt_reason
    minute = resumed.runner.cursor.next_minute_ms()
    assert minute is not None
    resumed.tick(minute)
    assert resumed.runner.state is not RunnerState.HALT, resumed.runner.halt_reason


def test_resume_alone_does_not_settle_the_dispute(tmp_path):
    """Why the command has to exist. The review's finding, pinned.

    `resume` clears the flag and changes neither equity, so the next construction
    reads the same disagreement and halts again. A campaign with only `resume`
    cannot leave this state by any permitted action.
    """
    config, _, _ = _crashed_campaign(tmp_path)

    resuming = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resuming.runner.start() is RunnerState.HALT
    resuming.runner.resume("operator: I looked at it")
    assert not resuming.runner.risk.state.halted

    again = build(tmp_path, days=(DAY,), config=config, start=False)

    assert again.runner.start() is RunnerState.HALT
    assert again.runner.risk.state.halt_reason.startswith(EQUITY_DISPUTE_PREFIX)


def test_resolve_equity_is_idempotent(tmp_path):
    """Retrying must not walk the equity, the peak or the day baselines anywhere."""
    config, _, accounted = _crashed_campaign(tmp_path)

    first = build(tmp_path, days=(DAY,), config=config, start=False)
    first.runner.start()
    first.runner.resolve_equity("operator: settled once")
    settled = first.runner.risk.state
    on_disk = (first.state_dir / "risk.json").read_text(encoding="utf-8")

    second = build(tmp_path, days=(DAY,), config=config, start=False)
    second.runner.start()
    with pytest.raises(RunnerError, match="no equity dispute to settle"):
        second.runner.resolve_equity("operator: settled twice")

    assert second.runner.risk.state == settled
    assert (second.state_dir / "risk.json").read_text(encoding="utf-8") == on_disk


def test_resolve_equity_refuses_a_ledger_that_may_not_speak(tmp_path):
    """The accounting is what it adopts, so it cannot run without one.

    This is also the `equity_reconciliation:` case: the ledger could not be read,
    which is why the restart halted, and settling from the placeholder would write
    a number no file holds into the central risk authority.
    """
    harness = build(tmp_path, days=(DAY,))
    harness.run(MINUTES_BEFORE_FLATTEN)
    config = harness.runner.config
    before = json.loads((harness.state_dir / "risk.json").read_text(encoding="utf-8"))
    (harness.state_dir / "carry_ledger.json").write_text("{ truncated", encoding="utf-8")

    halted = build(tmp_path, days=(DAY,), config=config, start=False)
    assert halted.runner.start() is RunnerState.HALT
    assert halted.runner.risk.state.halt_reason.startswith(EQUITY_RECONCILIATION_PREFIX)

    with pytest.raises(RunnerError, match="cannot settle the equity dispute") as refusal:
        halted.runner.resolve_equity("operator: clear it anyway")

    assert "UNREADABLE" in str(refusal.value), (
        "the two ways a ledger may not speak take different repairs, and this "
        "refusal shares its opening words with the log-regression one below"
    )
    assert halted.runner.risk.state.halted
    assert halted.runner.risk.state.halt_reason.startswith(
        EQUITY_RECONCILIATION_PREFIX
    ), "the refusal must leave the reconciliation halt exactly as it found it"
    after = json.loads((halted.state_dir / "risk.json").read_text(encoding="utf-8"))
    assert after["equity"] == before["equity"]
    assert after["peak_equity"] == before["peak_equity"]
    assert after["day"] == before["day"]
    assert after["day_start_equity"] == before["day_start_equity"]


def test_resolve_equity_refuses_when_there_is_no_dispute(tmp_path):
    """It is not a resume with a different name."""
    harness = build(tmp_path, days=(DAY,))
    harness.run(MINUTES_BEFORE_FLATTEN)

    with pytest.raises(RunnerError, match="no equity dispute to settle"):
        harness.runner.resolve_equity("operator: nothing is wrong")


def test_resolve_equity_refuses_to_clear_a_genuine_breach(tmp_path):
    """The one thing a clearing path must never become: a way out of a real halt."""
    harness = build(tmp_path, days=(DAY,))
    harness.run(MINUTES_BEFORE_FLATTEN)
    peak = harness.runner.risk.state.peak_equity
    harness.runner.risk.update_equity(peak * (1.0 - MAX_DRAWDOWN_PCT) - 1.0)
    assert harness.runner.risk.state.halted
    assert "max drawdown breached" in harness.runner.risk.state.halt_reason

    with pytest.raises(RunnerError, match="no equity dispute to settle"):
        harness.runner.resolve_equity("operator: let me out")

    assert harness.runner.risk.state.halted
    assert "max drawdown breached" in harness.runner.risk.state.halt_reason


def test_resolve_equity_requires_a_note(tmp_path):
    config, _, _ = _crashed_campaign(tmp_path)
    halted = build(tmp_path, days=(DAY,), config=config, start=False)
    halted.runner.start()

    with pytest.raises(RunnerError, match="requires an operator note"):
        halted.runner.resolve_equity("   ")

    assert halted.runner.risk.state.halted


def test_resolve_equity_writes_an_operator_record_naming_both_numbers(tmp_path):
    """Section 8.3: every operator command writes a record, and this one is audited."""
    config, persisted, accounted = _crashed_campaign(tmp_path)
    settling = build(tmp_path, days=(DAY,), config=config, start=False)
    settling.runner.start()

    settling.runner.resolve_equity("operator: the ledger is the campaign's accounting")

    operator = [
        r
        for r in settling.records()
        if r.get("kind") == "OPERATOR"
        and r.get("operator", {}).get("command") == "resolve-equity"
    ]
    assert len(operator) == 1
    block = operator[0]["operator"]
    assert block["note"] == "operator: the ledger is the campaign's accounting"
    assert block["settled"].startswith(EQUITY_DISPUTE_PREFIX)
    assert block["risk_equity_before"] == repr(persisted)
    assert block["adopted_equity"] == str(accounted)


# --- the other half of "may not speak": valid, and behind the log ----------
# `_ledger_may_speak` is false for EITHER a ledger that could not be read OR one
# holding less than the decision log has already committed. The test above is
# the first case. This is the second, and they are not interchangeable.
#
# An unreadable ledger reaches `resolve_equity` as `ledger_equity() is None`, so
# the `accounted is None` refusal two statements further down would have caught
# it even with the guard deleted. A ledger restored from an older copy loads
# `LOADED` and answers with a real number -- a STALE one -- and nothing but the
# guard stands between that number and the central risk authority. This is the
# case where the guard is the only thing doing any work, and the suite had it
# nowhere: `tests/test_demo_runner.py` proves such a ledger never speaks again,
# but nothing asked what it does to `risk.json`.


def _regressed_campaign(tmp_path: Path):
    """A VALID ledger holding less than the log, made the way an operator makes one.

    The restore a runbook reader really performs: a copy taken before the
    position was closed, written back over the current file. It parses, it loads
    `LOADED`, and its cumulative `slippage` is behind the log's last
    `ledger_effect` block -- which is `ledger_behind_log`, not corruption.

    `risk.json` is not touched by hand here either. `flatten` hands the flattened
    equity to Aegis -- this branch's own fix for the crash-free divergence -- so
    the persisted risk equity is the post-flatten one while the restored ledger
    still states the pre-flatten one, and the restart's reconciliation has a
    genuine disagreement to find rather than a manufactured one.
    """
    harness = build(tmp_path, days=(DAY,))
    harness.run(MINUTES_BEFORE_FLATTEN)
    ledger_path = harness.runner.position.ledger.path
    stale = ledger_path.read_bytes()
    stale_slippage = harness.runner.position.ledger.state.slippage

    # Trade on, so the log commits more than the copy holds.
    harness.runner.flatten("operator closed the position")
    harness.runner.shutdown("stop")
    committed = [r for r in harness.records() if "ledger_effect" in r][-1]["ledger_effect"]
    assert Decimal(committed["slippage"]) > stale_slippage, (
        "the fixture must really leave the log ahead of the copy, or there is no "
        "regression here for the guard to refuse"
    )

    ledger_path.write_bytes(stale)
    return harness.runner.config, ledger_path


def _halted_on_a_regressed_ledger(tmp_path: Path, config):
    """The restart, with every precondition the refusal below depends on asserted.

    Four separate facts, and a test that assumed any of them would pass for the
    wrong reason: the ledger is VALID, it is behind the log, Aegis is holding the
    dispute `resolve_equity` answers, and the ledger may not speak. In particular
    the runner's own halt is `ledger_behind_log` while AEGIS's is
    `equity_dispute:` -- `_halt` keeps the first reason and Aegis was halted
    first, inside `build_risk_engine` -- and it is Aegis's that the command reads.
    """
    resumed = build(tmp_path, days=(DAY,), config=config, start=False)
    assert resumed.runner.start() is RunnerState.HALT

    assert (
        resumed.runner.position.ledger.outcome is LoadOutcome.LOADED
    ), "the point of this case is a ledger that is perfectly readable"
    assert "ledger_behind_log" in (resumed.runner.halt_reason or "")
    assert resumed.runner.risk.state.halt_reason.startswith(EQUITY_DISPUTE_PREFIX), (
        "the stale ledger states a real equity, so the reconciliation finds a "
        "disagreement rather than an unreadable file"
    )
    assert resumed.runner._ledger_may_speak() is False
    assert resumed.runner.position.ledger.state.last_equity is not None, (
        "a never-marked ledger would dispute too -- `ledger_equity` answers with "
        "the configured capital -- and this case has to be the stale MARK"
    )
    return resumed


def test_resolve_equity_refuses_a_valid_ledger_that_regressed_against_the_log(tmp_path):
    """It adopts the accounting, so it may not run on accounting that went backwards.

    Settling here would write the pre-flatten equity into Aegis and clear the
    halt, leaving the campaign READY against a ledger the log has already
    overtaken -- and the next `ledger_effect` would quote it, which is how the
    log's cumulative series goes backwards for good. The refusal must therefore
    change nothing at all, and `risk.json` is compared BYTE for byte: `_persist`
    re-stamps `updated_at` on every write, so an unchanged file is proof that the
    engine was not touched rather than proof that it was put back.
    """
    config, ledger_path = _regressed_campaign(tmp_path)
    resumed = _halted_on_a_regressed_ledger(tmp_path, config)
    risk_path = resumed.state_dir / "risk.json"
    before = risk_path.read_bytes()
    ledger_before = ledger_path.read_bytes()

    with pytest.raises(RunnerError, match="cannot settle the equity dispute") as refusal:
        resumed.runner.resolve_equity("operator: settle it from the copy I restored")

    assert "ledger_behind_log" in str(refusal.value), (
        "the operator has to be told WHICH way the ledger is wrong: restoring a "
        "newer copy is the fix, and it is a different fix from a corrupt file"
    )
    assert risk_path.read_bytes() == before, "the refusal wrote to the risk state"
    assert ledger_path.read_bytes() == ledger_before, "the refusal wrote to the ledger"
    assert resumed.runner.risk.state.halted
    assert resumed.runner.risk.state.halt_reason.startswith(EQUITY_DISPUTE_PREFIX)


def test_without_that_guard_the_settlement_adopts_the_stale_accounting(monkeypatch, tmp_path):
    """The mutation, so the refusal above is not passing for some other reason.

    Deleting the `_ledger_may_speak` check in `resolve_equity` is survivable
    against every other test of that guard, because all of them reach it through
    an UNREADABLE ledger and are caught by the `accounted is None` refusal below
    it. Here `ledger_equity` answers with a number, so the guard is load-bearing
    and nothing else is: with it gone the command succeeds, writes the stale
    equity into Aegis, and releases the campaign.

    The mutant is installed on the INSTANCE. `_ledger_may_speak` also mutes
    `_save_ledger`, `_ledger_effect` and `resume`, so patching the class would
    disable four guards to witness one. `resolve_equity` calls none of those --
    it appends a record and saves the runner state -- so the only harm this
    mutant can do is the harm being witnessed: `risk.json` and one OPERATOR
    record. The ledger file is not written on this path at all.
    """
    config, _ = _regressed_campaign(tmp_path)
    resumed = _halted_on_a_regressed_ledger(tmp_path, config)
    stale_equity = resumed.runner.position.ledger.state.last_equity
    assert stale_equity is not None
    assert float(stale_equity) != resumed.runner.risk.state.equity

    monkeypatch.setattr(resumed.runner, "_ledger_may_speak", lambda: True)
    resumed.runner.resolve_equity("operator: the guard is gone")

    assert resumed.runner.risk.state.equity == pytest.approx(float(stale_equity)), (
        "with the guard removed the campaign's central risk authority is holding "
        "a number the decision log has already overtaken"
    )
    assert not resumed.runner.risk.state.halted, (
        "and the halt that was protecting it is cleared, so the mutant really is "
        "the harm and not merely a different refusal"
    )


# ---------------------------------------------------------------------------
# 10. what `adopt_reconciled_equity` may and may not move
# ---------------------------------------------------------------------------
DISPUTE = f"{EQUITY_DISPUTE_PREFIX} the two files disagree"


def disputed_engine(tmp_path: Path, state: RiskState) -> RiskEngine:
    """An engine restored from ``state`` and halted on an equity dispute."""
    state_dir = tmp_path / "state"
    persist_risk_state(state_dir, state)
    engine = RiskEngine(
        risk_wiring.risk_limits(committed_limits(state_dir).limits),
        state_path=state_dir / "risk.json",
        clock=clock_at(FIRST_MINUTE_OF_NEXT_DAY),
    )
    assert engine.load_outcome is RiskStateLoad.LOADED
    engine.halt(DISPUTE)
    return engine


def test_adopting_a_reconciled_equity_settles_the_named_halt(tmp_path):
    engine = disputed_engine(tmp_path, healthy_end_of_day(**WITNESS_A))

    engine.adopt_reconciled_equity(968_000.0, clearing=DISPUTE, note="operator: checked")

    assert not engine.state.halted
    assert engine.state.halt_reason == ""
    assert engine.state.equity == pytest.approx(968_000.0)


def test_adopting_does_not_roll_the_day(tmp_path):
    """The AEG-1 discipline, applied to the repair itself.

    The engine's clock is the first minute of the NEXT day, so `update_equity`
    would roll. Adoption is a correction of a reading from the old day and may not
    manufacture a new day's baseline out of an operator action -- which is the
    same class of mistake as the defect this item exists to fix. The first real
    mark of the new day still rolls it, and that is witness A's second half.
    """
    engine = disputed_engine(tmp_path, healthy_end_of_day(**WITNESS_A))

    engine.adopt_reconciled_equity(968_000.0, clearing=DISPUTE, note="operator: checked")

    assert engine.state.day == DAY
    assert engine.state.day_start_equity == pytest.approx(WITNESS_A["day_start"])
    assert engine.state.daily_pnl == pytest.approx(968_000.0 - WITNESS_A["day_start"])

    engine.update_equity(967_000.0)
    assert engine.state.day == NEXT_DAY
    assert engine.state.day_start_equity == pytest.approx(967_000.0)


def test_adopting_never_lowers_the_peak_and_may_raise_it(tmp_path):
    engine = disputed_engine(tmp_path, healthy_end_of_day(**WITNESS_B))
    # Below the peak but inside both limits, so this witnesses the peak rule and
    # not a breach: at 1,020,000 against a peak of 1,060,000 the drawdown is
    # 3.77%, under the campaign's 5%.
    engine.adopt_reconciled_equity(1_020_000.0, clearing=DISPUTE, note="operator: lower")
    assert not engine.state.halted, engine.state.halt_reason
    assert engine.state.peak_equity == pytest.approx(WITNESS_B["peak"])

    engine.halt(DISPUTE)
    engine.adopt_reconciled_equity(1_100_000.0, clearing=DISPUTE, note="operator: higher")
    assert engine.state.peak_equity == pytest.approx(1_100_000.0)


def test_adopting_moves_no_unrelated_risk_control(tmp_path):
    """Compared as a whole `RiskState`, so a field added later is covered too."""
    state = healthy_end_of_day(**WITNESS_A)
    state.consecutive_losses = 3
    state.cooldown_until = 1234.5
    state.funding_adverse_streak = 2
    state.open_positions = {"BTC/USDT": 5.5}
    state.reconciliation_disputed = {"BTC/USDT:USDT": "an earlier dispute"}
    state.stale_feed_since = 99.0
    engine = disputed_engine(tmp_path, state)
    before = RiskState(**{**engine.state.__dict__})

    engine.adopt_reconciled_equity(968_000.0, clearing=DISPUTE, note="operator: checked")

    after = engine.state
    moved = {
        name
        for name in RiskState.__dataclass_fields__
        if getattr(before, name) != getattr(after, name)
    }
    assert moved == {"equity", "daily_pnl", "halted", "halt_reason"}, (
        "adoption records an equity and settles one halt; anything else it moved "
        "is a risk control it had no authority over"
    )


def test_adopting_an_equity_that_is_itself_a_breach_halts_on_the_breach(tmp_path):
    """The settlement still happens; what remains is a genuine halt, named.

    `halt` keeps the FIRST reason, so clearing the dispute before the guards run
    is what stops a real breach being swallowed by the dispute it replaced.
    """
    engine = disputed_engine(tmp_path, healthy_end_of_day(**WITNESS_B))
    breach = WITNESS_B["peak"] * (1.0 - MAX_DRAWDOWN_PCT) - 1.0

    engine.adopt_reconciled_equity(breach, clearing=DISPUTE, note="operator: checked")

    assert engine.state.halted
    assert "max drawdown breached" in engine.state.halt_reason
    assert not is_equity_reconciliation_halt(engine.state.halt_reason)
    assert engine.state.equity == pytest.approx(breach)


def test_adopting_refuses_a_halt_it_was_not_asked_to_settle(tmp_path):
    """Checked in Aegis, not only in the runner that calls it."""
    engine = disputed_engine(tmp_path, healthy_end_of_day(**WITNESS_A))
    engine.resume()
    engine.halt("liquidation_touch: a real one")
    before = RiskState(**{**engine.state.__dict__})

    with pytest.raises(RiskViolation, match="not halted on"):
        engine.adopt_reconciled_equity(968_000.0, clearing=DISPUTE, note="operator: sneaky")

    assert engine.state == before


def test_adopting_refuses_an_unhalted_engine(tmp_path):
    engine = disputed_engine(tmp_path, healthy_end_of_day(**WITNESS_A))
    engine.resume()
    before = RiskState(**{**engine.state.__dict__})

    with pytest.raises(RiskViolation, match="not halted at all"):
        engine.adopt_reconciled_equity(968_000.0, clearing=DISPUTE, note="operator: sneaky")

    assert engine.state == before


def test_adopting_requires_a_note(tmp_path):
    engine = disputed_engine(tmp_path, healthy_end_of_day(**WITNESS_A))

    with pytest.raises(RiskViolation, match="requires a stated reason"):
        engine.adopt_reconciled_equity(968_000.0, clearing=DISPUTE, note="")

    assert engine.state.halted


def test_adopting_refuses_an_engine_that_failed_closed_on_its_own_file(tmp_path):
    """There is no persisted equity to reconcile, and `_persist` would write nothing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "risk.json").write_text("[]", encoding="utf-8")
    engine = RiskEngine(state_path=state_dir / "risk.json", clock=clock_at(LAST_MINUTE_OF_DAY))
    assert engine.load_outcome is RiskStateLoad.UNREADABLE

    with pytest.raises(RiskViolation, match="failed closed"):
        engine.adopt_reconciled_equity(
            968_000.0, clearing=engine.state.halt_reason, note="operator: sneaky"
        )


def test_the_settled_state_survives_the_round_trip_to_disk(tmp_path):
    """A settlement that lived only in memory would be undone by the next start."""
    engine = disputed_engine(tmp_path, healthy_end_of_day(**WITNESS_A))
    engine.adopt_reconciled_equity(968_000.0, clearing=DISPUTE, note="operator: checked")

    reloaded = RiskEngine(
        risk_wiring.risk_limits(committed_limits(tmp_path / "state").limits),
        state_path=tmp_path / "state" / "risk.json",
        clock=clock_at(FIRST_MINUTE_OF_NEXT_DAY),
    )

    assert reloaded.state == engine.state


# ---------------------------------------------------------------------------
# 11. M1 -- a ledger that cannot be EXAMINED is not a ledger that is absent
# ---------------------------------------------------------------------------
# `CarryLedger.open` decided "absent" with `Path.exists()`, which answers False
# for a path it merely could not examine. R1-b reads that outcome to decide
# whether a restart's persisted equity can be checked at all, and MISSING is the
# one answer that lets the check pass with no accounting consulted -- so an
# unexaminable ledger read as "worth the configured capital", a claim no file had
# made. The read decides now, as `RiskEngine._load_state` already did.


def blocked_ledger_path(tmp_path: Path) -> Path:
    """A ledger path whose PARENT is a regular file.

    POSIX raises ``NotADirectoryError`` for a path blocked by a regular file;
    Windows collapses that onto ``FileNotFoundError``, which is what the ancestor
    walk is for. Both must reach UNREADABLE, so this fixture exercises the same
    boundary on either platform.
    """
    blocked = tmp_path / "state" / "blocked"
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("not a directory", encoding="utf-8")
    return blocked


def test_a_ledger_path_that_cannot_be_examined_is_unreadable_not_missing(tmp_path):
    ledger = CarryLedger.open(
        blocked_ledger_path(tmp_path) / "carry_ledger.json", capital=CAPITAL
    )

    assert ledger.outcome is LoadOutcome.UNREADABLE
    assert ledger.disputed is not None


def test_a_genuinely_absent_ledger_is_still_missing(tmp_path):
    """The two-sided control: absence must keep meaning absence.

    Including a directory nobody has created yet, which is what a first start
    really looks like.
    """
    present = CarryLedger.open(tmp_path / "state" / "carry_ledger.json", capital=CAPITAL)
    assert present.outcome is LoadOutcome.MISSING
    assert present.state.last_equity is None

    deeper = CarryLedger.open(
        tmp_path / "never" / "made" / "carry_ledger.json", capital=CAPITAL
    )
    assert deeper.outcome is LoadOutcome.MISSING


def test_an_unexaminable_ledger_reaches_reconciliation_as_no_answer(tmp_path):
    """`ledger_equity` returns None rather than the configured capital."""
    assert ledger_equity(blocked_ledger_path(tmp_path), capital=CAPITAL) is None


def test_an_unexaminable_ledger_disputes_instead_of_reading_as_capital(tmp_path):
    """The case that made it a safety boundary: equity that happens to equal capital.

    A symlink loop, because it is the shape that isolates the defect: `Path.exists()`
    follows symlinks and answers ``False`` for a loop, so the OLD loader read this
    as "there is no ledger" and `ledger_equity` answered "worth the configured
    capital" -- which this persisted state agrees with, so the restart passed
    reconciliation against an accounting nothing had consulted, and the campaign
    started. A file that merely fails to PARSE would not witness this: the old
    loader caught that one and called it UNREADABLE too.

    Skipped where the platform will not make a symlink; the blocked-parent tests
    above cover the same boundary on both platforms.
    """
    state_dir = tmp_path / "state"
    persist_risk_state(
        state_dir,
        healthy_end_of_day(
            equity=float(CAPITAL), peak=float(CAPITAL), day_start=float(CAPITAL)
        ),
    )
    loop = state_dir / "carry_ledger.json"
    try:
        loop.symlink_to(loop)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"this platform will not create a symlink loop: {exc}")
    assert not loop.exists(), (
        "this fixture is only the `Path.exists()` trap while exists() answers False "
        "for a path that is really there"
    )

    engine = engine_at(state_dir, stamp=LAST_MINUTE_OF_DAY)

    assert engine.state.halted
    assert engine.state.halt_reason.startswith(EQUITY_RECONCILIATION_PREFIX)
    assert engine.state.equity == pytest.approx(
        float(CAPITAL)
    ), "the persisted state is preserved; the halt is what stops it being traded on"
