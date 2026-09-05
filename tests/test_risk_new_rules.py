"""The demo-grade Aegis rules, each with a case that passes and one that fails.

A guard is only evidence if it can also *not* fire, so every rule here is
exercised from both sides and every threshold is pinned just below, exactly at,
and just above. The funding rules get both position sides, because a rule that
looks at the magnitude of a funding rate and a rule that looks at what a
particular side would pay disagree on exactly the cases that matter.

The last section is the reduction exception. Aegis gates exposure *increases*;
a halted, kill-switched, disputed or funding-halted account must still be able
to get out. Those tests drive the real futures executor rather than reading the
purpose enum, because "the exit is not gated" is a claim about the path, not
about a boolean.

Everything runs in-process: an in-memory store, a dry-run venue, temporary
paths, no network.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from chimera.futures import (
    DeterministicFillModel,
    DryRunFuturesVenue,
    FlattenCause,
    FuturesExecutionConfig,
    FuturesExecutor,
    FuturesStore,
    OrderIntent,
    OrderPurpose,
    OrderSide,
    OrderState,
    Position,
    PositionError,
    PositionSide,
    TargetPosition,
    load_constraint_source,
)
from chimera.futures.executor import _veto_label
from chimera.risk import DEFAULT_KILL_SWITCH_PATH, RiskEngine, RiskLimits, RiskViolation

SYMBOL = "BTC/USDT:USDT"
PRICE = Decimal("60000")
EQUITY = 100_000.0

#: 180 s, the demo's ``max_data_delay_s``, expressed where ``note_feed`` reads it.
LIMIT_S = 180.0
NS = 10**9
CLOSE_NS = 1_700_000_000 * NS


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def build(tmp_path, limits=None, clock=None, state="risk.json", kill_switch="absent"):
    return RiskEngine(
        limits or RiskLimits(),
        state_path=tmp_path / state,
        clock=clock or FakeClock(),
        kill_switch_path=tmp_path / kill_switch,
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


# --- kill switch ----------------------------------------------------------
def test_a_present_kill_switch_file_halts_and_an_absent_one_does_not(tmp_path):
    absent = build(tmp_path, state="a.json")
    assert absent.check_kill_switch() is False
    assert not absent.halted
    assert entry(absent).allowed

    (tmp_path / "KILL").write_text("", encoding="utf-8")
    engaged = build(tmp_path, state="b.json", kill_switch="KILL")

    assert engaged.check_kill_switch() is True
    assert engaged.halted
    assert engaged.state.halt_reason == "kill_switch"
    assert not entry(engaged).allowed


def test_the_kill_switch_is_read_when_the_engine_starts(tmp_path):
    """Not only when a caller remembers to ask."""
    (tmp_path / "KILL").write_text("", encoding="utf-8")

    engine = build(tmp_path, kill_switch="KILL")

    assert engine.halted, "constructing an engine over a live kill switch must halt it"


def test_a_kill_switch_path_that_cannot_be_examined_fails_closed(tmp_path):
    """Absent and unexaminable are different answers; only one is safe to act on.

    A regular file where the containing directory should be makes ``stat`` raise
    ``NotADirectoryError``. ``Path.exists()`` would report that as a confident
    ``False`` — the guard would go quiet exactly when the filesystem stopped
    making sense.
    """
    (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")

    engine = build(tmp_path, kill_switch="blocked/KILL_SWITCH")

    assert engine.check_kill_switch() is True
    assert engine.halted
    assert engine.state.halt_reason.startswith("kill_switch:")
    assert "could not be examined" in engine.state.halt_reason


def test_the_kill_switch_flag_survives_a_restart(tmp_path):
    (tmp_path / "KILL").write_text("", encoding="utf-8")
    build(tmp_path, kill_switch="KILL")

    revived = build(tmp_path, kill_switch="KILL")

    assert revived.state.kill_switch is True
    assert revived.halted


def test_removing_the_file_clears_the_flag_but_not_the_halt(tmp_path):
    """The file arms the switch; only an operator disarms the halt it caused."""
    (tmp_path / "KILL").write_text("", encoding="utf-8")
    engine = build(tmp_path, kill_switch="KILL")
    assert engine.halted

    (tmp_path / "KILL").unlink()

    assert engine.check_kill_switch() is False
    assert engine.state.kill_switch is False
    assert engine.halted, "removing the file is not the operator saying it is safe"

    engine.resume()
    assert not engine.halted
    assert entry(engine).allowed


def test_a_resume_cannot_outlive_the_file_it_did_not_remove(tmp_path):
    (tmp_path / "KILL").write_text("", encoding="utf-8")
    engine = build(tmp_path, kill_switch="KILL")

    engine.resume()
    assert not engine.halted

    assert engine.check_kill_switch() is True
    assert engine.halted, "the switch is still on disk, so the next check re-halts"


# --- stale feed -----------------------------------------------------------
@pytest.mark.parametrize(
    "delay_s, stale",
    [(179.9, False), (180.0, False), (180.1, True)],
    ids=["just-below", "exactly-at", "just-above"],
)
def test_the_stale_feed_boundary_is_above_the_limit_not_at_it(tmp_path, delay_s, stale):
    """Matches ``evaluate_entry``'s own ``data_delay_s > max_data_delay_s``.

    A feed that is exactly at the limit is late by exactly what the limit
    permits; making that stale would move the limit by one tick of the clock.
    """
    engine = build(tmp_path, RiskLimits(max_data_delay_s=LIMIT_S))

    engine.note_feed(CLOSE_NS, CLOSE_NS + round(delay_s * NS))

    assert (engine.state.stale_feed_since is not None) is stale
    assert entry(engine).allowed is not stale


def test_a_stale_feed_veto_names_the_feed_and_collapses_to_stale_data(tmp_path):
    engine = build(tmp_path, RiskLimits(max_data_delay_s=LIMIT_S))
    engine.note_feed(CLOSE_NS, CLOSE_NS + 300 * NS)

    decision = entry(engine)

    assert not decision.allowed
    assert decision.reason.startswith("market data late")
    assert _veto_label(decision.reason) == "stale_data"


def test_the_stale_mark_is_when_it_went_stale_not_the_latest_look(tmp_path):
    engine = build(tmp_path, RiskLimits(max_data_delay_s=LIMIT_S))
    engine.note_feed(CLOSE_NS, CLOSE_NS + 300 * NS)
    first = engine.state.stale_feed_since

    engine.note_feed(CLOSE_NS, CLOSE_NS + 900 * NS)

    assert engine.state.stale_feed_since == first


def test_a_fresh_minute_clears_the_stale_mark(tmp_path):
    engine = build(tmp_path, RiskLimits(max_data_delay_s=LIMIT_S))
    engine.note_feed(CLOSE_NS, CLOSE_NS + 300 * NS)
    assert not entry(engine).allowed

    engine.note_feed(CLOSE_NS + 300 * NS, CLOSE_NS + 310 * NS)

    assert engine.state.stale_feed_since is None
    assert entry(engine).allowed


def test_the_stale_feed_mark_survives_a_restart(tmp_path):
    limits = RiskLimits(max_data_delay_s=LIMIT_S)
    first = build(tmp_path, limits)
    first.note_feed(CLOSE_NS, CLOSE_NS + 300 * NS)

    revived = build(tmp_path, limits)

    assert revived.state.stale_feed_since == first.state.stale_feed_since
    assert not entry(revived).allowed, "restarting into a dead feed is not a fresh feed"


# --- reconciliation dispute ----------------------------------------------
def test_a_disputed_symbol_refuses_increases_and_its_neighbour_does_not(tmp_path):
    engine = build(tmp_path)
    assert entry(engine, pair="BTC/USDT").allowed

    engine.note_reconciliation("BTC/USDT", "venue reports 2.0, we hold 1.0")

    decision = entry(engine, pair="BTC/USDT")
    assert not decision.allowed
    assert "venue reports 2.0" in decision.reason
    assert entry(engine, pair="ETH/USDT").allowed, "the dispute is about one symbol"


def test_a_dispute_survives_a_restart(tmp_path):
    build(tmp_path).note_reconciliation("BTC/USDT", "venue reports 2.0, we hold 1.0")

    revived = build(tmp_path)

    assert revived.state.reconciliation_disputed == {
        "BTC/USDT": "venue reports 2.0, we hold 1.0"
    }
    assert not entry(revived, pair="BTC/USDT").allowed


def test_only_an_explicit_note_clears_a_dispute(tmp_path):
    engine = build(tmp_path)
    engine.note_reconciliation("BTC/USDT", "quantities disagree")
    assert not entry(engine, pair="BTC/USDT").allowed

    # Restarting, updating equity and writing the file again all leave it.
    engine.update_equity(10_000.0)
    assert (
        not build(tmp_path)
        .evaluate_entry(pair="BTC/USDT", equity=10_000.0, entry_price=100.0, stop_price=95.0)
        .allowed
    )

    engine.note_reconciliation("BTC/USDT", None)

    assert engine.state.reconciliation_disputed == {}
    assert entry(engine, pair="BTC/USDT").allowed


# --- funding: the entry cost ---------------------------------------------
@pytest.mark.parametrize(
    "side, rate, allowed",
    [
        (PositionSide.LONG, 0.001, False),
        (PositionSide.LONG, -0.001, True),
        (PositionSide.SHORT, -0.001, False),
        (PositionSide.SHORT, 0.001, True),
    ],
    ids=["long-pays", "long-receives", "short-pays", "short-receives"],
)
def test_the_funding_check_is_about_what_this_side_would_pay(tmp_path, side, rate, allowed):
    """A long pays a positive rate and a short pays a negative one.

    The sign-blind check refused both signs at the same magnitude, which refuses
    the leg that is being *paid* to hold the position — the whole point of a
    carry trade — while a hedge's two legs see the same rate with opposite signs.
    """
    engine = build(tmp_path, RiskLimits(max_funding_cost_rate=0.0005))

    assert entry(engine, funding_rate=rate, position_side=side).allowed is allowed


@pytest.mark.parametrize(
    "rate, allowed",
    [(0.00049, True), (0.0005, True), (0.00051, False)],
    ids=["just-below", "exactly-at", "just-above"],
)
def test_the_funding_cost_boundary_is_above_the_limit_not_at_it(tmp_path, rate, allowed):
    engine = build(tmp_path, RiskLimits(max_funding_cost_rate=0.0005))

    decision = entry(engine, funding_rate=rate, position_side=PositionSide.LONG)

    assert decision.allowed is allowed
    if not allowed:
        assert _veto_label(decision.reason) == "funding"


def test_a_caller_with_no_side_keeps_the_sign_blind_check(tmp_path):
    """The old behaviour, unchanged, for the callers that still have no side."""
    engine = build(tmp_path, RiskLimits(max_funding_rate=0.0005, max_funding_cost_rate=0.0005))

    assert not entry(engine, funding_rate=0.002).allowed
    assert not entry(engine, funding_rate=-0.002).allowed
    assert entry(engine, funding_rate=0.0001).allowed
    # The case the two rules disagree about: a rebate a long would be paid.
    assert not entry(engine, funding_rate=-0.002, position_side=None).allowed
    assert entry(engine, funding_rate=-0.002, position_side=PositionSide.LONG).allowed


def test_a_side_given_as_a_bare_string_is_understood(tmp_path):
    """The enum is imported here only for type checking, so the string path matters."""
    engine = build(tmp_path, RiskLimits(max_funding_cost_rate=0.0005))

    assert not entry(engine, funding_rate=0.001, position_side="LONG").allowed
    assert entry(engine, funding_rate=0.001, position_side="SHORT").allowed


def test_an_unusable_side_falls_back_to_the_stricter_sign_blind_check(tmp_path):
    """FLAT is not a side a position pays funding on, so it may not soften a limit."""
    engine = build(tmp_path, RiskLimits(max_funding_rate=0.0005, max_funding_cost_rate=0.0005))

    assert not entry(engine, funding_rate=-0.002, position_side=PositionSide.FLAT).allowed
    assert not entry(engine, funding_rate=-0.002, position_side="nonsense").allowed


# --- funding: the adverse streak -----------------------------------------
def test_the_streak_halts_at_the_limit_and_not_before(tmp_path):
    engine = build(tmp_path, RiskLimits(funding_adverse_streak_limit=3))

    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
    assert engine.state.funding_adverse_streak == 2
    assert not engine.state.funding_halt
    assert entry(engine).allowed

    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)

    assert engine.state.funding_halt
    decision = entry(engine)
    assert not decision.allowed
    assert decision.reason.startswith("funding halt")


def test_a_short_that_pays_a_negative_rate_counts_the_same_way(tmp_path):
    engine = build(tmp_path, RiskLimits(funding_adverse_streak_limit=2))

    engine.note_funding_settlement(SYMBOL, PositionSide.SHORT, -0.001)
    engine.note_funding_settlement(SYMBOL, PositionSide.SHORT, -0.001)

    assert engine.state.funding_halt


def test_a_settlement_the_position_receives_resets_the_streak(tmp_path):
    engine = build(tmp_path, RiskLimits(funding_adverse_streak_limit=3))
    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)

    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, -0.001)

    assert engine.state.funding_adverse_streak == 0
    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
    assert not engine.state.funding_halt, "the run was broken, so it is not three in a row"


def test_a_received_settlement_lifts_a_funding_halt(tmp_path):
    engine = build(tmp_path, RiskLimits(funding_adverse_streak_limit=2))
    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
    assert not entry(engine).allowed

    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, -0.001)

    assert not engine.state.funding_halt
    assert entry(engine).allowed


def test_a_zero_settlement_neither_extends_nor_forgives_the_streak(tmp_path):
    engine = build(tmp_path, RiskLimits(funding_adverse_streak_limit=3))
    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)

    engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.0)

    assert engine.state.funding_adverse_streak == 1


def test_the_streak_and_the_funding_halt_survive_a_restart(tmp_path):
    limits = RiskLimits(funding_adverse_streak_limit=2)
    first = build(tmp_path, limits)
    first.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
    first.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)

    revived = build(tmp_path, limits)

    assert revived.state.funding_adverse_streak == 2
    assert revived.state.funding_halt
    assert not entry(revived).allowed


def test_a_settlement_with_no_side_is_refused_rather_than_guessed(tmp_path):
    engine = build(tmp_path)

    with pytest.raises(RiskViolation, match="no position side"):
        engine.note_funding_settlement(SYMBOL, PositionSide.FLAT, 0.001)
    assert engine.state.funding_adverse_streak == 0


# --- the executor's liquidation veto --------------------------------------
def make_executor(risk=None, fill_model=None):
    """A bootstrapped executor over a flat dry-run venue and an in-memory store."""
    engine = risk or RiskEngine(
        RiskLimits(
            max_position_pct=1.0,
            risk_per_trade_pct=0.5,
            max_total_exposure_pct=10.0,
            max_exposure_per_asset_pct=10.0,
        )
    )
    engine.update_equity(EQUITY)
    executor = FuturesExecutor(
        venue=DryRunFuturesVenue(
            source=load_constraint_source(), fill_model=fill_model or DeterministicFillModel()
        ),
        risk=engine,
        store=FuturesStore.open(None),
        config=FuturesExecutionConfig(),
    )
    executor.recover({})
    return executor


def move_to(executor, side, quantity, **kwargs):
    target = TargetPosition(SYMBOL, side, Decimal(quantity))
    return executor.execute_target(target, PRICE, equity=EQUITY, **kwargs)


def test_an_increase_whose_liquidation_price_cannot_be_computed_is_vetoed(monkeypatch):
    """``None`` from a non-flat prospective is a missing input, not a free pass.

    Aegis reads ``liquidation_price=None`` as "there is nothing here to judge",
    which is true of a flat position. Handing it that when the figure merely
    could not be computed let an exposure increase through the distance guard at
    the one moment the guard had no input.
    """
    executor = make_executor()
    monkeypatch.setattr("chimera.futures.executor.margin_state", lambda *args, **kwargs: None)

    (record,) = move_to(executor, PositionSide.LONG, "0.5")

    assert record.state is OrderState.REJECTED
    assert record.reason.startswith("liquidation unknown")
    assert _veto_label(record.reason) == "liquidation_unknown"
    assert executor.position(SYMBOL).side is PositionSide.FLAT


def test_the_same_increase_is_allowed_when_the_liquidation_price_is_available():
    """The control. Without it the test above would pass on a broken executor."""
    executor = make_executor()

    (record,) = move_to(executor, PositionSide.LONG, "0.5")

    assert record.state is OrderState.FILLED
    assert executor.position(SYMBOL).quantity == Decimal("0.5")


def test_a_prospective_position_this_order_cannot_produce_is_vetoed(monkeypatch):
    """The ``PositionError`` fallback: an unknown prospective is refused too."""
    executor = make_executor()

    def refuse(*args, **kwargs):
        raise PositionError("a quantized close cannot cancel an off-grid position exactly")

    monkeypatch.setattr(Position, "apply_fill", refuse)
    (record,) = move_to(executor, PositionSide.LONG, "0.5")

    assert record.state is OrderState.REJECTED
    assert record.reason.startswith("liquidation unknown")


def test_a_genuinely_flat_prospective_position_still_reaches_aegis():
    """The case ``None`` really does mean "nothing to judge".

    A flat position has no liquidation price to be far from, so the executor
    must keep passing ``None`` and let Aegis decide on everything else. Built by
    hand because the planner never emits an opening order that lands flat, and
    the branch would otherwise be untested in exactly the direction that would
    turn the new veto into a block on ordinary trading.
    """
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    intent = OrderIntent(
        symbol=SYMBOL,
        side=OrderSide.SELL,
        quantity=Decimal("0.5"),
        purpose=OrderPurpose.OPEN,
        reduce_only=False,
        position_side=PositionSide.SHORT,
    )

    decision = executor._ask_aegis(
        intent,
        executor.constraints(SYMBOL),
        PRICE,
        equity=EQUITY,
        stop_price=None,
        data_delay_s=None,
        inference_age_s=None,
        funding_rate=None,
        exchange_healthy=True,
    )

    assert decision.allowed, decision.reason


def test_the_executor_tells_aegis_which_side_the_order_would_hold():
    """A short leg must not be refused for a rate it is being paid."""
    limits = RiskLimits(
        max_position_pct=1.0,
        risk_per_trade_pct=0.5,
        max_total_exposure_pct=10.0,
        max_exposure_per_asset_pct=10.0,
        max_funding_cost_rate=0.0005,
    )
    paying = make_executor(RiskEngine(limits))
    (rejected,) = move_to(paying, PositionSide.LONG, "0.5", funding_rate=0.002)
    assert rejected.state is OrderState.REJECTED
    assert _veto_label(rejected.reason) == "funding"

    receiving = make_executor(RiskEngine(limits))
    (filled,) = move_to(receiving, PositionSide.SHORT, "0.5", funding_rate=0.002)
    assert filled.state is OrderState.FILLED


# --- the reduction exception ----------------------------------------------
def test_only_opening_and_increasing_orders_are_gated():
    assert OrderPurpose.OPEN.increases_exposure
    assert OrderPurpose.INCREASE.increases_exposure
    assert not OrderPurpose.REDUCE.increases_exposure
    assert not OrderPurpose.CLOSE.increases_exposure
    assert not OrderPurpose.FLATTEN.increases_exposure


def refusing_engine(tmp_path, how):
    """An engine that refuses every increase, for the reason ``how`` names."""
    limits = RiskLimits(
        max_position_pct=1.0,
        risk_per_trade_pct=0.5,
        max_total_exposure_pct=10.0,
        max_exposure_per_asset_pct=10.0,
        max_data_delay_s=LIMIT_S,
        funding_adverse_streak_limit=2,
        loss_streak_limit=1,
        cooldown_seconds=3_600.0,
    )
    if how == "kill_switch":
        (tmp_path / "KILL").write_text("", encoding="utf-8")
        engine = RiskEngine(limits, kill_switch_path=tmp_path / "KILL")
    else:
        engine = RiskEngine(limits, kill_switch_path=tmp_path / "absent")
    if how == "halt":
        engine.halt("drawdown breach")
    elif how == "dispute":
        engine.note_reconciliation(SYMBOL, "venue reports 2.0, we hold 0.5")
    elif how == "funding_halt":
        engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
        engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
    elif how == "stale_feed":
        engine.note_feed(CLOSE_NS, CLOSE_NS + 600 * NS)
    elif how == "cooldown":
        engine.record_trade_result(-1.0)
    return engine


REFUSALS = ["halt", "kill_switch", "dispute", "funding_halt", "stale_feed", "cooldown"]


@pytest.mark.parametrize("how", REFUSALS)
def test_a_refusing_aegis_still_lets_a_position_be_closed(tmp_path, how):
    """The kill switch must be a brake, never a trap."""
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    engine = refusing_engine(tmp_path, how)
    executor.risk = engine
    engine.update_equity(EQUITY)

    # The increase is genuinely refused...
    (rejected,) = move_to(executor, PositionSide.LONG, "1.0")
    assert rejected.state is OrderState.REJECTED

    # ...and the exit is not.
    (closed,) = move_to(executor, PositionSide.FLAT, "0")
    assert closed.intent.purpose is OrderPurpose.CLOSE
    assert closed.state is OrderState.FILLED
    assert executor.position(SYMBOL).side is PositionSide.FLAT


@pytest.mark.parametrize("how", REFUSALS)
def test_a_refusing_aegis_still_lets_a_position_be_reduced(tmp_path, how):
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    engine = refusing_engine(tmp_path, how)
    executor.risk = engine
    engine.update_equity(EQUITY)

    (reduced,) = move_to(executor, PositionSide.LONG, "0.2")

    assert reduced.intent.purpose is OrderPurpose.REDUCE
    assert reduced.state is OrderState.FILLED
    assert executor.position(SYMBOL).quantity == Decimal("0.2")


@pytest.mark.parametrize("how", REFUSALS)
def test_a_refusing_aegis_still_lets_an_emergency_flatten_through(tmp_path, how):
    executor = make_executor()
    move_to(executor, PositionSide.LONG, "0.5")
    engine = refusing_engine(tmp_path, how)
    executor.risk = engine
    engine.update_equity(EQUITY)

    record = executor.emergency_flatten(SYMBOL, FlattenCause.RISK_HALT, PRICE)

    assert record.intent.purpose is OrderPurpose.FLATTEN
    assert record.state is OrderState.FILLED
    assert executor.position(SYMBOL).side is PositionSide.FLAT


# --- rules the demo re-pins two-sided -------------------------------------
def test_an_unhealthy_exchange_blocks_an_entry_and_a_healthy_one_does_not(tmp_path):
    engine = build(tmp_path)

    assert not entry(engine, exchange_healthy=False).allowed
    assert entry(engine, exchange_healthy=True).allowed


def test_non_positive_equity_halts_and_positive_equity_does_not(tmp_path):
    engine = build(tmp_path)
    engine.update_equity(10_000.0)
    assert not engine.halted

    engine.update_equity(0.0)

    assert engine.halted
    assert "non-positive" in engine.state.halt_reason


@pytest.mark.parametrize(
    "leverage, allowed",
    [(2.99, True), (3.0, True), (3.01, False)],
    ids=["just-below", "exactly-at", "just-above"],
)
def test_the_leverage_boundary_is_above_the_cap_not_at_it(tmp_path, leverage, allowed):
    engine = build(tmp_path, RiskLimits(max_leverage=3.0, max_position_pct=1.0))

    assert entry(engine, leverage=leverage).allowed is allowed


@pytest.mark.parametrize(
    "liquidation_price, allowed",
    [(49.0, True), (50.0, True), (50.5, False)],
    ids=["further-than", "exactly-at", "closer-than"],
)
def test_the_liquidation_distance_boundary_is_below_the_limit_not_at_it(
    tmp_path, liquidation_price, allowed
):
    """Entry at 100, so a liquidation at 50 is exactly the 50% the limit allows."""
    engine = build(tmp_path, RiskLimits(min_liquidation_distance_pct=0.5))

    assert entry(engine, liquidation_price=liquidation_price).allowed is allowed


# --- the kill switch is configured, never assumed -------------------------
def test_no_kill_switch_is_configured_unless_the_caller_names_one(tmp_path, monkeypatch):
    """A library constructor may not read the working directory.

    ``DEFAULT_KILL_SWITCH_PATH`` is relative, so an engine that applied it as a
    default would resolve it against whatever directory the process started in.
    Every engine in the repository would then depend on an untracked file that no
    committed input names — including the generator behind the frozen
    ``artifacts/futures_dry_run_v1``, whose output must be a function of
    committed inputs alone.
    """
    (tmp_path / DEFAULT_KILL_SWITCH_PATH.parent).mkdir(parents=True, exist_ok=True)
    (tmp_path / DEFAULT_KILL_SWITCH_PATH).write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    engine = RiskEngine(RiskLimits())

    assert not engine.halted
    assert engine.check_kill_switch() is False
    assert entry(engine).allowed


def test_an_engine_given_the_documented_path_still_halts_on_it(tmp_path, monkeypatch):
    """The mirror: the switch works, it just has to be wired to work."""
    (tmp_path / DEFAULT_KILL_SWITCH_PATH.parent).mkdir(parents=True, exist_ok=True)
    (tmp_path / DEFAULT_KILL_SWITCH_PATH).write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    engine = RiskEngine(RiskLimits(), kill_switch_path=DEFAULT_KILL_SWITCH_PATH)

    assert engine.halted
    assert engine.state.halt_reason == "kill_switch"


def test_an_engaged_switch_halts_even_when_the_mirror_is_already_set(tmp_path):
    """The halt is level-triggered, not edge-triggered on its own mirror.

    A state file carrying ``kill_switch: true`` beside ``halted: false`` — what
    hand-editing a halt out of the file produces — used to leave this method
    answering "engaged" while the engine approved entries.
    """
    (tmp_path / "KILL").write_text("", encoding="utf-8")
    build(tmp_path, kill_switch="KILL")
    document = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    assert document["kill_switch"] is True
    document["halted"] = False
    document["halt_reason"] = ""
    (tmp_path / "risk.json").write_text(json.dumps(document), encoding="utf-8")

    revived = build(tmp_path, kill_switch="KILL")

    assert revived.check_kill_switch() is True
    assert revived.halted, "the switch is on disk; answering 'engaged' and trading is worse"
    assert not entry(revived).allowed


# --- a feed whose age cannot be believed ----------------------------------
def test_a_last_close_in_the_future_is_unknown_not_fresh(tmp_path):
    """Clock skew, or one argument in the wrong unit. Neither is a fresh feed."""
    engine = build(tmp_path, RiskLimits(max_data_delay_s=LIMIT_S))
    engine.note_feed(CLOSE_NS, CLOSE_NS + 600 * NS)
    assert engine.state.stale_feed_since is not None

    engine.note_feed(CLOSE_NS + 10_000 * NS, CLOSE_NS)

    assert engine.state.stale_feed_since is not None, "a skewed clock may not lift the mark"
    assert not entry(engine).allowed


def test_a_negative_delay_marks_a_previously_fresh_feed_stale(tmp_path):
    engine = build(tmp_path, RiskLimits(max_data_delay_s=LIMIT_S))
    engine.note_feed(CLOSE_NS, CLOSE_NS + 10 * NS)
    assert entry(engine).allowed

    engine.note_feed(CLOSE_NS + 10_000 * NS, CLOSE_NS)

    assert engine.state.stale_feed_since is not None
    assert not entry(engine).allowed


# --- the funding halt has an operator exit --------------------------------
def test_a_funding_halt_reached_by_following_the_rule_can_be_cleared(tmp_path):
    """Its own remedy is to reduce, and a flat position never settles again.

    Without an operator exit the streak that told the runner to get out would
    then refuse every increase forever, leaving only two ways back: editing the
    state file by hand, or reporting a favourable settlement that never happened
    — putting a fabricated number into a guard.
    """
    limits = RiskLimits(funding_adverse_streak_limit=3)
    engine = build(tmp_path, limits)
    for _ in range(3):
        engine.note_funding_settlement(SYMBOL, PositionSide.LONG, 0.001)
    assert engine.state.funding_halt

    # The position is flattened, so no further settlement can ever arrive.
    revived = build(tmp_path, limits)
    assert revived.state.funding_halt, "it survives a restart; only an operator lifts it"

    revived.resume()

    assert not revived.state.funding_halt
    assert revived.state.funding_adverse_streak == 0
    assert entry(revived).allowed
    assert build(tmp_path, limits).state.funding_halt is False


def test_a_resume_does_not_clear_a_dispute_or_a_stale_feed(tmp_path):
    """The mirror: the funding halt is the exception, not a general amnesty.

    Both of these have a clearing path that still works while the account is
    flat, so a blanket resume that forgot them would be the failure each exists
    to prevent.
    """
    engine = build(tmp_path, RiskLimits(max_data_delay_s=LIMIT_S))
    engine.note_reconciliation("BTC/USDT", "venue reports 2.0, we hold 1.0")
    engine.note_feed(CLOSE_NS, CLOSE_NS + 600 * NS)
    engine.halt("operator")

    engine.resume()

    assert engine.state.reconciliation_disputed == {
        "BTC/USDT": "venue reports 2.0, we hold 1.0"
    }
    assert engine.state.stale_feed_since is not None
    assert not entry(engine, pair="BTC/USDT").allowed
