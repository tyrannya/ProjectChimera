"""R1-l: one valuation price for the perpetual leg, and it is the MARK.

The master roadmap's item: "one documented valuation price (mark) for equity,
drawdown and liquidation; the close-vs-mark inconsistency removed from the
ledger path that R8 will inherit." Before R1-l, `HedgedPosition._perp_pnl`
valued the SHORT perpetual at the minute's kline CLOSE while section 6.7's
liquidation test, standing beside it, priced its threshold at the MARK (high)
and the executor's margin at ``state.mark``. The equity Aegis judged for
drawdown and daily loss was the close-priced one.

Every witness here is two-sided and runs on a minute where the two prices
DIFFER. The synthetic fixture writes ``mark_close == kline_close`` exactly, so
nothing taken from it unaltered can tell a mark-priced equity from a
close-priced one; each test therefore moves one price and holds the other.

What stays on the closes, deliberately: the SPOT leg (spot has no mark price;
its close is what the inventory is worth), ``basis`` and section 6.5's running
identity (spreads the rule reads and the ledger reconciles, not valuations),
and the executors' fill reference. Synthetic worlds only; no market data is
read and no economic claim follows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from decimal import Decimal as D
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import tests.demo_harness as demo_harness
from chimera.carry.factory import PERP_SYMBOL
from chimera.carry.hedge import (
    PERP,
    SPOT,
    HedgeState,
    HedgeTarget,
    ValuationUnknown,
    valuation_mark,
)
from chimera.demo.decision_log import RecordKind
from chimera.demo.fixtures import MinuteShape, SyntheticFeed
from chimera.demo.runner import RunnerState
from chimera.futures.accounting import unrealised_pnl
from chimera.futures.executor import FlattenCause
from chimera.recorder.contract import load_recorder_contract
from tests.demo_harness import (
    CARRY_PARAMS,
    DAY,
    MOMENTUM_PARAMS,
    REPO,
    SHADOW_PARAMS,
    build,
    campaign_config,
)
from tests.test_carry_hedge import open_hedge, position  # noqa: F401 - the fixture
from tools import demo_run, replay_parity

START_NS = 1_700_000_000_000_000_000
MINUTE_NS = 60_000_000_000
T0 = int(pd.Timestamp(DAY, tz="UTC").timestamp() * 1000)
MINUTE_MS = 60_000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Priced:
    """A minute whose closes and mark are set independently. `CarryMarketState`."""

    minute_ns: int
    complete: bool
    spot_close: D
    perp_close: D
    mark: Any
    mark_high: Any = None
    funding_rate_current: D | None = D("0.0001")


def priced(*, spot="30000", perp="30030", mark="30030", high=None, index=1) -> Priced:
    return Priced(
        minute_ns=START_NS + index * MINUTE_NS,
        complete=True,
        spot_close=D(spot),
        perp_close=D(perp),
        mark=D(mark) if isinstance(mark, str) else mark,
        mark_high=D(high) if isinstance(high, str) else high,
    )


def by_hand(position, state, *, perp_price: D) -> D:  # noqa: F811
    """Section 6.6's equity line written out, the perpetual valued at ``perp_price``.

    The oracle is independent of `HedgedPosition`'s own arithmetic: the ledger's
    cash and posted margin, the spot inventory at the spot close, and the SHORT
    perpetual's ``Q * (entry - price)``.
    """
    spot, perp = position.leg(SPOT), position.leg(PERP)
    ledger = position.ledger.state
    return (
        ledger.free_cash
        + spot.quantity * state.spot_close
        + ledger.perp_margin
        + perp.quantity * (perp.entry_price - perp_price)
    )


def patch_minute(monkeypatch, harness, minute_ms: int, **fields) -> None:
    """Make the runner's feed report ``fields`` for one recorded minute.

    `MarketState` is frozen, so the minute is REPLACED, not edited: every other
    field -- books, digests, funding rate -- is the file's own.
    """
    cursor = harness.runner.cursor
    original = cursor.state_for

    def state_for(minute, *args, **kwargs):
        state = original(minute, *args, **kwargs)
        if int(minute) == int(minute_ms):
            return replace(state, **fields)
        return state

    monkeypatch.setattr(cursor, "state_for", state_for)


def hedged_harness(where: Path, **kwargs):
    """A campaign that has opened its hedge in its first three minutes."""
    harness = build(where, **kwargs)
    harness.run(3, start=T0)
    assert harness.runner.position.state is HedgeState.HEDGED
    assert harness.runner.position.leg(PERP).quantity > 0
    return harness


def records_of(harness, kind: RecordKind) -> list[dict]:
    return [r for r in harness.records() if r["kind"] == kind.value]


class MarkOffsetFeed(SyntheticFeed):
    """The synthetic fixture with the perpetual's MARK moved off its close.

    ``mark = close + offset(index)``, an offset that changes sign through the
    day, so the mark sits above the close on some minutes and below it on
    others. Written through the fixture's own `write_day`, so the digest is the
    recorder's over the file as written.
    """

    offset = 40.0

    def _row(self, market, minute_open_ms, index, shape):
        row = super()._row(market, minute_open_ms, index, shape)
        if market == "um" and shape.mark:
            mark = round(row["kline_close"] + self.offset * ((index % 7) - 3), 2)
            row.update(
                {"mark_open": mark, "mark_high": mark, "mark_low": mark, "mark_close": mark}
            )
        return row


class BadMarkFeed(SyntheticFeed):
    """A perpetual row that says its mark arrived and carries no usable one."""

    bad: dict[int, float | None] = {}

    def _row(self, market, minute_open_ms, index, shape):
        row = super()._row(market, minute_open_ms, index, shape)
        if market == "um" and index in self.bad:
            value = self.bad[index]
            row.update(
                {
                    "mark_open": value,
                    "mark_high": value,
                    "mark_low": value,
                    "mark_close": value,
                }
            )
        return row


# ---------------------------------------------------------------------------
# A. the authoritative valuation: mark moves equity, close does not
# ---------------------------------------------------------------------------
def test_moving_the_mark_moves_equity_by_quantity_times_the_move(position):  # noqa: F811
    open_hedge(position)
    quantity = position.leg(PERP).quantity
    assert quantity > 0 and position.leg(PERP).side.value == "SHORT"

    base = position.equity_at(priced(mark="30030"))
    up = position.equity_at(priced(mark="30130"))
    down = position.equity_at(priced(mark="29930"))

    # SHORT: a mark 100 higher is a loss of exactly Q * 100, and 100 lower a gain.
    assert up - base == -quantity * D("100")
    assert down - base == quantity * D("100")
    assert up == by_hand(position, priced(mark="30130"), perp_price=D("30130"))


def test_moving_only_the_perp_close_moves_no_equity(position):  # noqa: F811
    open_hedge(position)
    base = position.equity_at(priced(perp="30030", mark="30030"))
    for close in ("30130", "29930", "35000"):
        state = priced(perp=close, mark="30030")
        assert position.equity_at(state) == base, close
        # Non-vacuous: valuing at that close WOULD have moved the equity.
        assert by_hand(position, state, perp_price=D(close)) != base


# ---------------------------------------------------------------------------
# B. every equity caller reaches the same rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("close", "mark"),
    [("30030", "30530"), ("30530", "30030"), ("30030", "29000"), ("31000", "31000")],
)
def test_mark_to_market_and_equity_at_value_the_same_minute_identically(
    position, close, mark  # noqa: F811
):
    open_hedge(position)
    state = priced(perp=close, mark=mark)
    expected = by_hand(position, state, perp_price=D(mark))

    read_only = position.equity_at(state)
    marked = position.mark_to_market(state)

    assert read_only == expected
    assert marked.equity == expected
    assert position.ledger.state.last_equity == expected


def test_the_carry_perp_term_is_the_futures_layers_unrealised_pnl(position):  # noqa: F811
    """One perpetual valuation function in the codebase, not two.

    The carry leg's unrealised term IS `chimera.futures.accounting.unrealised_pnl`
    of the perpetual executor's position at the mark -- the side-signed function
    the futures layer's own LONG and SHORT tests pin -- so a single-leg runtime
    built on the executor inherits the same valuation without a second copy.
    """
    open_hedge(position)
    held = position.perp.position(PERP_SYMBOL)
    for mark in ("30030", "30530", "29000"):
        state = priced(perp="31000", mark=mark)
        marked = position.mark_to_market(state)
        assert marked.perp_pnl == unrealised_pnl(held, D(mark))
        assert marked.perp_pnl == held.quantity * (held.entry_price - D(mark))


def test_the_carry_mark_documents_the_price_its_perp_term_used(position):  # noqa: F811
    open_hedge(position)
    state = priced(spot="30000", perp="30030", mark="30400")
    marked = position.mark_to_market(state)
    perp = position.leg(PERP)

    assert marked.perp_mark == D("30400")
    assert marked.perp_close == D("30030"), "the close is still reported, beside it"
    assert marked.perp_pnl == perp.quantity * (perp.entry_price - D("30400"))
    assert marked.to_dict()["perp_mark"] == "30400"


# ---------------------------------------------------------------------------
# B'. the valuation price is the mark CLOSE, never the high
# ---------------------------------------------------------------------------
def test_the_equity_is_valued_at_the_mark_close_and_not_at_the_mark_high(
    position,  # noqa: F811
):
    """The high is section 6.7's THRESHOLD price only; the equity is the close's.

    A minute whose mark high sits well above its mark close: the equity (both
    callers) and the documented price are the close's, and the high reaches
    only the touch threshold.
    """
    open_hedge(position)
    state = priced(perp="30030", mark="30030", high="30530")
    expected = by_hand(position, state, perp_price=D("30030"))

    assert position.equity_at(state) == expected
    marked = position.mark_to_market(state)
    assert marked.equity == expected
    assert marked.perp_mark == D("30030")
    assert by_hand(position, state, perp_price=D("30530")) != expected, "non-vacuous"


class DistinctMarkFeed(SyntheticFeed):
    """Every mark column different, and the mark different every minute.

    The fixture writes one value into ``mark_open/high/low/close``; here the
    close is ``kline_close + 10 * (index % 5) + 3`` and the other three sit
    around it, so a reader of the wrong column or the wrong minute is visible.
    """

    def _row(self, market, minute_open_ms, index, shape):
        row = super()._row(market, minute_open_ms, index, shape)
        if market == "um" and shape.mark:
            close = expected_mark_close(row["kline_close"], index)
            row.update(
                {
                    "mark_open": round(close - 7.0, 2),
                    "mark_high": round(close + 5.0, 2),
                    "mark_low": round(close - 9.0, 2),
                    "mark_close": close,
                }
            )
        return row


def expected_mark_close(kline_close: float, index: int) -> float:
    return round(kline_close + 10.0 * (index % 5) + 3.0, 2)


def test_the_feed_reports_each_minutes_own_mark_close(tmp_path, monkeypatch):
    """The valuation price is THIS minute's ``mark_close``: not its open, not
    its high, not a neighbour's."""
    monkeypatch.setattr(demo_harness, "SyntheticFeed", DistinctMarkFeed)
    harness = build(tmp_path)
    fixture = SyntheticFeed(tmp_path / "unused", harness.runner.contract)
    for index in range(6):
        state = harness.runner.cursor.state_for(T0 + index * MINUTE_MS)
        close = expected_mark_close(fixture.perp_close(index), index)
        assert state.mark == D(str(close)), index
        assert state.mark_high == D(str(round(close + 5.0, 2))), index


def test_the_runner_marks_at_the_minutes_own_mark_close(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_harness, "SyntheticFeed", DistinctMarkFeed)
    harness = hedged_harness(tmp_path)
    minute = T0 + 3 * MINUTE_MS
    harness.tick(minute)
    fixture = SyntheticFeed(tmp_path / "unused", harness.runner.contract)
    close = D(str(expected_mark_close(fixture.perp_close(3), 3)))
    state = harness.runner.cursor.state_for(minute)
    carry = harness.runner.position
    assert carry.ledger.state.last_equity == by_hand(carry, state, perp_price=close)
    assert harness.risk.state.equity == float(carry.ledger.state.last_equity)


# ---------------------------------------------------------------------------
# C. the same mark-priced equity reaches Aegis
# ---------------------------------------------------------------------------
def test_a_mark_move_reaches_aegis_equity_and_a_close_move_does_not(tmp_path, monkeypatch):
    minute = T0 + 3 * MINUTE_MS
    plain = hedged_harness(tmp_path / "plain")
    plain.tick(minute)

    moved = hedged_harness(tmp_path / "mark")
    state = moved.runner.cursor.state_for(minute)
    patch_minute(monkeypatch, moved, minute, mark=state.mark + 100, mark_high=state.mark + 100)
    moved.tick(minute)

    closed = hedged_harness(tmp_path / "close")
    patch_minute(monkeypatch, closed, minute, perp_close=state.perp_close + 100)
    closed.tick(minute)

    quantity = moved.runner.position.leg(PERP).quantity
    plain_equity = plain.runner.position.ledger.state.last_equity
    moved_equity = moved.runner.position.ledger.state.last_equity
    assert moved_equity - plain_equity == -quantity * D("100")
    assert moved.risk.state.equity == float(moved_equity)
    assert moved.risk.state.equity != plain.risk.state.equity

    # The close alone changes nothing Aegis sees.
    assert closed.runner.position.ledger.state.last_equity == plain_equity
    assert closed.risk.state.equity == plain.risk.state.equity

    # And the record's `ledger_effect.equity` is that same number.
    decision = records_of(moved, RecordKind.DECISION)[-1]
    assert decision["ledger_effect"]["equity"] == str(moved_equity)
    assert decision["inputs"]["mark"] == str(state.mark + 100)


def _breach_move(harness, fraction: float) -> D:
    """A price move that costs the SHORT perpetual ``fraction`` of its equity."""
    quantity = harness.runner.position.leg(PERP).quantity
    equity = D(str(harness.risk.state.equity))
    return (equity * D(str(fraction)) / quantity).quantize(D("1")) + 1


def test_daily_loss_halts_on_a_mark_move_and_not_on_a_close_move(tmp_path, monkeypatch):
    minute = T0 + 3 * MINUTE_MS
    by_mark = hedged_harness(tmp_path / "mark")
    move = _breach_move(by_mark, 0.025)  # the committed limit is 2%
    state = by_mark.runner.cursor.state_for(minute)
    patch_minute(
        monkeypatch, by_mark, minute, mark=state.mark + move, mark_high=state.mark + move
    )
    by_mark.tick(minute)
    assert by_mark.risk.state.halted
    assert by_mark.risk.state.halt_reason.startswith("max daily loss breached")

    by_close = hedged_harness(tmp_path / "close")
    patch_minute(monkeypatch, by_close, minute, perp_close=state.perp_close + move)
    by_close.tick(minute)
    assert not by_close.risk.state.halted, by_close.risk.state.halt_reason


def test_drawdown_halts_on_a_mark_move_and_not_on_a_close_move(tmp_path, monkeypatch):
    """Daily loss is held out of reach so the drawdown guard is the one that answers."""
    minute = T0 + 3 * MINUTE_MS
    limits = {"max_daily_loss_pct": 0.5}

    def config(where):
        return campaign_config(where / "state", limits=limits)

    by_mark = hedged_harness(tmp_path / "mark", config=config(tmp_path / "mark"))
    move = _breach_move(by_mark, 0.06)  # the committed limit is 5%
    state = by_mark.runner.cursor.state_for(minute)
    patch_minute(
        monkeypatch, by_mark, minute, mark=state.mark + move, mark_high=state.mark + move
    )
    by_mark.tick(minute)
    assert by_mark.risk.state.halted
    assert by_mark.risk.state.halt_reason.startswith("max drawdown breached")

    by_close = hedged_harness(tmp_path / "close", config=config(tmp_path / "close"))
    patch_minute(monkeypatch, by_close, minute, perp_close=state.perp_close + move)
    by_close.tick(minute)
    assert not by_close.risk.state.halted, by_close.risk.state.halt_reason


# ---------------------------------------------------------------------------
# D. liquidation: the threshold between the two candidate equities
# ---------------------------------------------------------------------------
def _place_equity(position, state, *, target: D) -> None:  # noqa: F811
    """Move ``free_cash`` so the mark-priced equity at ``state`` is ``target``.

    `free_cash` is the field funding erodes, which section 6.7 says the check
    exists for; no threshold and no model is changed.
    """
    position.ledger.state.free_cash -= position.equity_at(state) - target


@pytest.mark.parametrize("side", ["mark_above_close", "mark_below_close"])
def test_the_touch_follows_the_mark_priced_equity(position, side):  # noqa: F811
    """Same minute, same threshold, the two candidate equities on either side of it.

    ``mark_above_close``: the SHORT is worse off at the mark, so the mark-priced
    equity is under ``Q * mark_high * rate`` while the close-priced one is over
    it -- TOUCHED, where close pricing said "not touched". ``mark_below_close``
    is the mirror: close pricing touched a position the mark says is safe.
    """
    open_hedge(position)
    mark = "30530" if side == "mark_above_close" else "29530"
    state = priced(perp="30030", mark=mark, high=mark)
    quantity = position.leg(PERP).quantity
    threshold = quantity * D(mark) * position.config.maintenance_margin_rate
    gap = quantity * abs(D(mark) - D("30030"))

    if side == "mark_above_close":
        _place_equity(position, state, target=threshold - gap / 2)
    else:
        _place_equity(position, state, target=threshold + gap / 2)

    by_mark = position.equity_at(state)
    by_close = by_hand(position, state, perp_price=D("30030"))
    assert by_mark == by_hand(position, state, perp_price=D(mark))
    assert min(by_mark, by_close) < threshold <= max(by_mark, by_close)

    assert position.liquidation_touched(state, equity=by_mark) is (side == "mark_above_close")
    assert position.liquidation_touched(state, equity=by_close) is (side != "mark_above_close")


@pytest.mark.parametrize("side", ["mark_above_close", "mark_below_close"])
def test_the_runner_touches_on_the_mark_priced_equity(tmp_path, monkeypatch, side):
    """Through the real tick: LIQUIDATION_TOUCH exactly when the mark equity is under."""
    minute = T0 + 3 * MINUTE_MS
    harness = hedged_harness(tmp_path)
    state = harness.runner.cursor.state_for(minute)
    delta = D("500") if side == "mark_above_close" else D("-500")
    mark = state.perp_close + delta
    patched = replace(state, mark=mark, mark_high=mark)
    patch_minute(monkeypatch, harness, minute, mark=mark, mark_high=mark)

    carry = harness.runner.position
    quantity = carry.leg(PERP).quantity
    threshold = quantity * mark * carry.config.maintenance_margin_rate
    gap = quantity * abs(delta)
    target = threshold - gap / 2 if side == "mark_above_close" else threshold + gap / 2
    _place_equity(carry, patched, target=target)
    by_close = by_hand(carry, patched, perp_price=state.perp_close)
    assert min(target, by_close) < threshold <= max(target, by_close)

    harness.tick(minute)

    touches = records_of(harness, RecordKind.LIQUIDATION_TOUCH)
    if side == "mark_above_close":
        assert len(touches) == 1
        assert f"equity {target}" in touches[0]["veto_or_rejection"]["detail"]
        assert harness.runner.position.state is HedgeState.FLAT
    else:
        assert touches == []
        assert harness.runner.position.leg(PERP).quantity == quantity


# ---------------------------------------------------------------------------
# E. no usable mark
# ---------------------------------------------------------------------------
BAD_MARKS = [
    None,
    D("NaN"),
    D("sNaN"),
    D("Infinity"),
    D("-Infinity"),
    D("0"),
    D("-1"),
    30030.0,  # a binary float is not a price this package accepts
]
BAD_IDS = ["absent", "nan", "snan", "+inf", "-inf", "zero", "negative", "float"]


@pytest.mark.parametrize("bad", BAD_MARKS, ids=BAD_IDS)
def test_the_predicate_refuses_every_unusable_mark(bad):
    assert valuation_mark(bad) is None


def test_the_predicate_returns_a_usable_mark_unchanged():
    assert valuation_mark(D("30030.10")) == D("30030.10")


@pytest.mark.parametrize("bad", BAD_MARKS, ids=BAD_IDS)
def test_a_held_perp_is_never_valued_without_a_usable_mark(position, bad):  # noqa: F811
    """Refused, not valued at the close, and nothing is marked on the way."""
    open_hedge(position)
    position.mark_to_market(priced(mark="30030"))
    before = position.ledger.snapshot()

    blind = priced(perp="30030", mark=bad)
    with pytest.raises(ValuationUnknown):
        position.equity_at(blind)
    with pytest.raises(ValuationUnknown):
        position.mark_to_market(blind)
    assert position.ledger.snapshot() == before
    assert position.state is HedgeState.HEDGED


def test_a_flat_position_needs_no_mark(position):  # noqa: F811
    blind = priced(mark=None)
    assert position.equity_at(blind) == position.ledger.state.free_cash
    marked = position.mark_to_market(blind)
    assert marked.perp_mark is None, "no valuation ran, so none is named"
    assert marked.perp_pnl == 0


def test_a_flat_perpetual_beside_held_spot_needs_no_mark(position):  # noqa: F811
    """A spot-only position is valued on its spot close alone."""
    open_hedge(position)
    position.perp.emergency_flatten(PERP_SYMBOL, FlattenCause.RISK_HALT, D("30030"))
    assert position.leg(PERP).quantity == 0 and position.leg(SPOT).quantity > 0

    blind = priced(spot="30100", mark=None)
    assert position.equity_at(blind) == by_hand(position, blind, perp_price=D("0"))


@pytest.mark.parametrize("bad", [None, 0.0, -5.0], ids=["null", "zero", "negative"])
def test_a_file_minute_with_no_usable_mark_is_incomplete(tmp_path, monkeypatch, bad):
    """``mark_present`` true with no usable ``mark_close``: section 2.2, not a guess.

    Without it a flat position would open on a minute its own mark could not
    value, and the PERSISTENCE mark would then refuse AFTER the legs had filled.
    """

    class Feed(BadMarkFeed):
        pass

    Feed.bad = {0: bad}
    monkeypatch.setattr(demo_harness, "SyntheticFeed", Feed)
    harness = build(tmp_path)
    outcome = harness.tick(T0)
    assert outcome.kind is RecordKind.INCOMPLETE_STATE
    incomplete = records_of(harness, RecordKind.INCOMPLETE_STATE)[0]
    assert "um_mark" in incomplete["veto_or_rejection"]["detail"]
    assert harness.runner.position.state is HedgeState.FLAT
    # The next minute, whose mark is usable, decides and opens.
    assert harness.tick(T0 + MINUTE_MS).kind is RecordKind.DECISION
    assert harness.runner.position.state is HedgeState.HEDGED


@pytest.mark.parametrize("bad", [D("NaN"), D("Infinity")], ids=["nan", "+inf"])
def test_a_held_position_on_a_minute_with_an_unusable_mark_halts_unknown(
    tmp_path, monkeypatch, bad
):
    """Section 7.2: unknown on a non-flat position is refused -- never valued at the close."""
    minute = T0 + 3 * MINUTE_MS
    harness = hedged_harness(tmp_path)
    before = harness.runner.position.ledger.state.last_equity
    patch_minute(monkeypatch, harness, minute, mark=bad, mark_high=None)

    outcome = harness.tick(minute)

    assert outcome.kind is RecordKind.HALT
    assert harness.runner.state is RunnerState.HALT
    assert harness.runner.halt_reason.startswith("liquidation_unknown")
    assert records_of(harness, RecordKind.LIQUIDATION_TOUCH) == []
    assert harness.runner.position.ledger.state.last_equity == before


def _no_book_minutes(first: int, last: int, *, mark: bool) -> dict:
    return {
        f"um:{DAY}": {i: MinuteShape(book=False, mark=mark) for i in range(first, last + 1)}
    }


def test_an_operator_flatten_on_a_minute_with_no_mark_still_completes(tmp_path):
    """The reduction is never made impossible by a valuation it cannot perform.

    The last processed minute has neither a perpetual book nor a mark, so the
    perpetual leg cannot fill and cannot be valued. The spot leg still reduces,
    the ledger is saved, the completion record is written -- and nothing is
    marked: the ledger keeps its last real equity and Aegis is told nothing new.
    """
    harness = build(tmp_path, shapes=_no_book_minutes(3, 6, mark=False))
    harness.run(7, start=T0)
    carry = harness.runner.position
    assert carry.leg(PERP).quantity > 0 and carry.leg(SPOT).quantity > 0
    equity = carry.ledger.state.last_equity
    aegis = harness.risk.state.equity

    outcome = harness.runner.flatten("operator reduces on a minute with no mark")

    assert outcome.kind is RecordKind.OPERATOR
    assert carry.leg(SPOT).quantity == 0, "the spot leg still reduced"
    assert carry.leg(PERP).quantity > 0, "the perpetual had no book to fill against"
    assert carry.ledger.state.last_equity == equity
    assert harness.risk.state.equity == aegis
    completed = [
        r
        for r in records_of(harness, RecordKind.OPERATOR)
        if r["operator"]["phase"] == "completed"
    ]
    assert len(completed) == 1
    assert "ledger_effect" not in completed[0], "no equity was valued, so none is asserted"
    on_disk = json.loads((harness.state_dir / "carry_ledger.json").read_text("utf-8"))
    assert D(on_disk["last_equity"]) == equity


def test_an_operator_flatten_with_a_mark_values_what_it_left(tmp_path):
    """The control: the same book-less minutes WITH a mark are valued at that mark."""
    harness = build(tmp_path, shapes=_no_book_minutes(3, 6, mark=True))
    harness.run(7, start=T0)
    carry = harness.runner.position
    equity = carry.ledger.state.last_equity

    harness.runner.flatten("operator reduces on a minute with a mark")

    assert carry.leg(PERP).quantity > 0
    state = harness.runner.cursor.state_for(T0 + 6 * MINUTE_MS)
    expected = by_hand(carry, state, perp_price=state.mark)
    assert carry.ledger.state.last_equity == expected != equity
    assert harness.risk.state.equity == float(expected)


# ---------------------------------------------------------------------------
# F. spot, basis and the identity stay on the closes
# ---------------------------------------------------------------------------
def test_the_spot_leg_is_valued_at_its_close_and_never_at_the_mark(position):  # noqa: F811
    open_hedge(position)
    spot_quantity = position.leg(SPOT).quantity
    base = position.equity_at(priced(spot="30000", mark="30030"))
    assert position.equity_at(priced(spot="30100", mark="30030")) - base == spot_quantity * 100

    moved = position.mark_to_market(priced(spot="30000", mark="31030"))
    held = position.mark_to_market(priced(spot="30000", mark="30030"))
    assert moved.spot_pnl == held.spot_pnl


def test_basis_and_the_running_identity_stay_on_the_closes(position):  # noqa: F811
    open_hedge(position)
    aligned = position.mark_to_market(priced(spot="30000", perp="30030", mark="30030"))
    residual = aligned.identity_residual
    apart = position.mark_to_market(priced(spot="30000", perp="30030", mark="31030"))

    assert apart.basis == D("30")
    assert position.ledger.state.current_basis == D("30")
    assert apart.identity_residual == residual, "the identity never sees the mark"
    assert position.state is not HedgeState.DISPUTED
    assert position.ledger.disputed is None


def test_the_mark_is_a_hashed_decision_input(tmp_path, monkeypatch):
    """The documented valuation price is already evidence: in the record and its hash."""
    minute = T0 + 3 * MINUTE_MS
    harness = hedged_harness(tmp_path)
    state = harness.runner.cursor.state_for(minute)
    moved = replace(state, mark=state.mark + 1, mark_high=state.mark + 1)
    assert moved.canonical()["mark"] != state.canonical()["mark"]

    from chimera.demo.runner import _inputs_hash

    assert _inputs_hash(moved) != _inputs_hash(state)


# ---------------------------------------------------------------------------
# G. restart and replay
# ---------------------------------------------------------------------------
D1 = DAY
MINUTES = 300
RESTART_AT = 120
FLATTEN_AT = 240
OPERATIONAL_2100 = 4_102_444_800.0
NOTE = "R1-l witness: flatten after the restart on a feed whose mark is off its close"


def _campaign(tmp_path: Path, feed_type: type[SyntheticFeed]) -> dict[str, Any]:
    root = tmp_path / "recorder"
    state_dir = tmp_path / "state"
    feed = feed_type(root, load_recorder_contract("btcusdt-prospective-gen3"))
    end = T0 + MINUTES * MINUTE_MS
    path = feed.write_settlements([D1])
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["funding_time_ms"] <= end
    ]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    base = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text(encoding="utf-8"))
    base.update(
        {
            "profile": "TEST",
            "runner": {"state_dir": str(state_dir)},
            "rules": {
                "R1_carry": dict(CARRY_PARAMS),
                "R2_frozen_logistic": dict(SHADOW_PARAMS),
                "R3_daily_momentum": dict(MOMENTUM_PARAMS),
            },
        }
    )
    config = tmp_path / "campaign.json"
    config.write_text(json.dumps(base), encoding="utf-8")
    argv = ["--config", str(config), "--root", str(root), "--profile", "TEST"]

    def publish(through: int) -> None:
        hidden = {slot: MinuteShape(present=False) for slot in range(through, 1440)}
        for market in ("um", "spot"):
            feed.write_day(market, D1, hidden)

    def cli(*command: str) -> int:
        return demo_run.main(argv + list(command), operational_clock=lambda: OPERATIONAL_2100)

    publish(RESTART_AT)
    assert cli("run", "--once") == demo_run.EXIT_OK
    publish(FLATTEN_AT)
    assert cli("run", "--once") == demo_run.EXIT_OK  # the restart
    assert cli("flatten", "--note", NOTE) == demo_run.EXIT_OK
    publish(MINUTES)
    assert cli("run", "--once") == demo_run.EXIT_OK

    actions = tmp_path / "operator_actions.json"
    after = pd.Timestamp(T0 + (FLATTEN_AT - 1) * MINUTE_MS, unit="ms", tz="UTC").isoformat()
    actions.write_text(
        json.dumps(
            {
                "schema": "chimera.operator-actions/1",
                "actions": [
                    {
                        "id": "r1l-flatten",
                        "after_minute": after,
                        "command": "flatten",
                        "note": NOTE,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",  # the canonical form `replay_parity` requires of this file
        encoding="utf-8",
    )
    records = []
    for log in sorted((state_dir / "decision_log").glob("*.ndjson")):
        records += [json.loads(x) for x in log.read_text("utf-8").splitlines() if x]
    return {
        "records": records,
        "replay_argv": argv
        + [
            "--live-log", str(state_dir / "decision_log"),
            "--days", D1,
            "--scratch", str(tmp_path / "scratch"),
            "--operator-actions", str(actions),
            "--json",
        ],
    }  # fmt: skip


def _equities(records) -> list[str]:
    return [r["ledger_effect"]["equity"] for r in records if r["kind"] == "DECISION"]


def test_a_mark_off_its_close_replays_to_parity_across_a_restart_and_a_flatten(
    tmp_path, capsys
):
    offset = _campaign(tmp_path / "offset", MarkOffsetFeed)
    aligned = _campaign(tmp_path / "aligned", SyntheticFeed)

    # Non-vacuous: the mark reached the recorded equity. A close-priced build
    # records the same equities for both feeds, whose closes are identical.
    assert _equities(offset["records"]) != _equities(aligned["records"])
    kinds = [r["kind"] for r in offset["records"]]
    assert kinds.count("OPERATOR") == 2, "the flatten's request and completion"
    assert kinds.count("STARTUP") >= 3, "three processes before the flatten's: a restart"

    capsys.readouterr()  # the campaigns' own output, not the report
    code = replay_parity.main(
        offset["replay_argv"], operational_clock=lambda: OPERATIONAL_2100
    )
    report = json.loads(capsys.readouterr().out)
    assert code == replay_parity.EXIT_PARITY, report
    assert report["label"] == "PARITY" and report["divergences"] == 0
    assert report["live_only"] == [] and report["replay_only"] == []
    assert report["explained_exclusions"] == []
    assert report["records_compared"] >= MINUTES


# ---------------------------------------------------------------------------
# H. R1-k composition
# ---------------------------------------------------------------------------
def test_the_funding_veto_is_unchanged_on_a_feed_whose_mark_is_off_its_close(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(demo_harness, "SyntheticFeed", MarkOffsetFeed)
    vetoed = build(tmp_path / "adverse", mark_funding_rate=-0.001)
    vetoed.run(3, start=T0)
    perp_orders = list(vetoed.runner.position.perp.store.state.orders.values())
    assert perp_orders and all(
        o.reason.startswith("funding rate would cost this position") for o in perp_orders
    )
    assert vetoed.runner.position.state is HedgeState.FLAT

    allowed = build(tmp_path / "benign", mark_funding_rate=-0.0003)
    allowed.run(3, start=T0)
    assert allowed.runner.position.state is HedgeState.HEDGED


def test_a_mark_driven_halt_blocks_increases_and_still_allows_the_reduction(
    tmp_path, monkeypatch
):
    """Funding-safe throughout; the valuation halt alone refuses the increase."""
    minute = T0 + 3 * MINUTE_MS
    harness = hedged_harness(tmp_path)
    move = _breach_move(harness, 0.025)
    state = harness.runner.cursor.state_for(minute)
    patch_minute(
        monkeypatch, harness, minute, mark=state.mark + move, mark_high=state.mark + move
    )
    harness.tick(minute)
    assert harness.risk.state.halted
    assert state.funding_rate_current is not None and state.funding_rate_current > 0

    held = harness.runner.position.leg(SPOT).quantity
    assert harness.runner.position.plan(HedgeTarget(held * 2), state) == []

    harness.runner.flatten("reduce after a mark-driven loss halt")
    assert harness.runner.position.state is HedgeState.FLAT
