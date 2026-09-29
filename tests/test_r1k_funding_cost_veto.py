"""R1-k: the funding-cost entry veto, wired and witnessed from both sides.

Aegis's side-aware veto -- refuse an exposure increase whose position would PAY
more than ``max_funding_cost_rate`` per settlement, ``cost = sign(side) * rate``
(amendment A10) -- existed and was unreachable: no production caller passed a
rate. R1-k wires the decision minute's rate IN EFFECT (the mark stream's ``r``,
`MarketState.funding_rate_current`) to the PERPETUAL leg's ``execute_target``.

The acceptance cases drive the real path end to end: synthetic recorder files
with a chosen rate per minute -> `FeedCursor` -> `DemoRunner.tick` -> the carry
rule -> `HedgedPosition` -> `FuturesExecutor` -> `RiskEngine.evaluate_entry`.
The carry position holds a SHORT perpetual (it pays a NEGATIVE rate), so the
LONG sign convention is witnessed at the executor boundary, through the real
executor and a real Aegis. Everything is synthetic; no economic claim follows.

The committed campaign limit is 0.0005 per settlement. Sections follow the R1-k
funding acceptance matrix (A..N).
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path

import pandas as pd
import pytest

from chimera.carry.hedge import PERP, SPOT, HedgeError, HedgeState, HedgeTarget
from chimera.demo.feed import MarketState
from chimera.demo.runner import RunnerState
from chimera.futures import OrderState, PositionSide
from chimera.risk import RiskEngine, RiskLimits
from tests.demo_harness import DAY, build, campaign_config
from tests.test_carry_hedge import EQUITY as HEDGE_EQUITY
from tests.test_carry_hedge import book, minute, position  # noqa: F401 - the fixture
from tests.test_demo_cli import written_config
from tests.test_futures_executor import (
    SYMBOL,
    ExplodingVenue,
    dry_run_venue,
    make_executor,
    move_to,
    wide_limits,
)
from tools import replay_parity

LIMIT = 0.0005
T0 = int(pd.Timestamp(DAY, tz="UTC").timestamp() * 1000)
MINUTE_MS = 60_000
FUNDING_REASON = "funding rate would cost this position"

#: Rates for the SHORT perpetual the carry hedge holds. Its cost is ``-rate``.
ADVERSE_ABOVE = -0.001  # cost 0.00100 > 0.00050
ADVERSE_BELOW = -0.0003  # cost 0.00030
AT_LIMIT = -0.0005  # cost 0.00050 == limit
MINUS_EPSILON = -0.00049  # cost 0.00049
PLUS_EPSILON = -0.00051  # cost 0.00051
FAVOURABLE = 0.002  # a short RECEIVES this; |rate| is above the limit on purpose


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class SubmitSpy:
    """Counts what reached each leg's venue. A refused order must never arrive."""

    def __init__(self, harness) -> None:
        self.calls: dict[str, list[str]] = {PERP: [], SPOT: []}
        for leg, executor in (
            (PERP, harness.runner.position.perp),
            (SPOT, harness.runner.position.spot),
        ):
            venue = executor.venue
            original = venue.submit

            def submit(order_id, intent, reference_price, _leg=leg, _original=original):
                self.calls[_leg].append(order_id)
                return _original(order_id, intent, reference_price)

            venue.submit = submit


def run(
    tmp_path: Path, *, rate: float | None = None, rates=None, minutes: int = 3, limits=None
):
    """A fresh campaign over the synthetic day, ``minutes`` decided, venue spied."""
    config = campaign_config(tmp_path / "state", limits=limits) if limits else None
    harness = build(
        tmp_path,
        config=config,
        mark_funding_rate=0.0001 if rate is None else rate,
        mark_funding_rates=rates,
    )
    spy = SubmitSpy(harness)
    outcomes = harness.run(minutes)
    return harness, spy, outcomes


def orders(harness, leg: str):
    executor = harness.runner.position.perp if leg == PERP else harness.runner.position.spot
    return [record for _, record in sorted(executor.store.state.orders.items())]


def decisions(harness) -> list[dict]:
    return [r for r in harness.records() if r["kind"] == "DECISION"]


def refused_on_funding(harness) -> bool:
    return any(
        o.state is OrderState.REJECTED and o.reason.startswith(FUNDING_REASON)
        for o in orders(harness, PERP)
    )


def assert_refused_safely(harness, spy) -> None:
    """The perp increase refused for funding, and nothing unsafe after it."""
    perp = orders(harness, PERP)
    assert perp and all(o.state is OrderState.REJECTED for o in perp), perp
    assert all(o.reason.startswith(FUNDING_REASON) for o in perp), [o.reason for o in perp]
    assert orders(harness, SPOT) == [], "a spot leg was sent after the perpetual was refused"
    assert spy.calls == {PERP: [], SPOT: []}, "a refused order reached a venue"
    assert harness.runner.position.state is HedgeState.FLAT
    assert harness.runner.position.imbalance() == D("0")
    assert not any("RISK_APPROVED" in step for o in perp for step in o.history)


def assert_opened(harness, spy) -> None:
    perp, spot = orders(harness, PERP), orders(harness, SPOT)
    assert [o.state for o in perp] == [OrderState.FILLED], [(o.state, o.reason) for o in perp]
    assert [o.state for o in spot] == [OrderState.FILLED]
    assert len(spy.calls[PERP]) == 1 and len(spy.calls[SPOT]) == 1
    assert harness.runner.position.state is HedgeState.HEDGED
    # Approved by Aegis, not around it: each order passed PLANNED->RISK_APPROVED.
    assert all("PLANNED->RISK_APPROVED" in o.history for o in perp + spot)


# ---------------------------------------------------------------------------
# A. core breach / non-breach, through the real runner
# ---------------------------------------------------------------------------
def test_adverse_cost_above_the_limit_is_refused_on_the_real_path(tmp_path):
    harness, spy, _ = run(tmp_path, rate=ADVERSE_ABOVE)
    assert_refused_safely(harness, spy)
    reason = orders(harness, PERP)[0].reason
    assert reason == (
        "funding rate would cost this position 0.00100 per settlement, over 0.00050"
    ), reason


def test_adverse_cost_below_the_limit_is_allowed_on_the_real_path(tmp_path):
    harness, spy, _ = run(tmp_path, rate=ADVERSE_BELOW)
    assert_opened(harness, spy)


# ---------------------------------------------------------------------------
# B. the exact boundary: `cost > limit` refuses, equality does not
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rate, refused",
    [(MINUS_EPSILON, False), (AT_LIMIT, False), (PLUS_EPSILON, True)],
    ids=["limit-minus-epsilon", "exactly-at-limit", "limit-plus-epsilon"],
)
def test_the_boundary_is_pinned_on_the_real_path(tmp_path, rate, refused):
    """Epsilon is 1e-5 of rate, eleven orders of magnitude above a float64's
    resolution at 5e-4; each rate round-trips parquet -> str -> Decimal -> float
    to the same double as its literal, so no case depends on a rounding accident."""
    assert float(D(str(rate))) == rate
    harness, spy, _ = run(tmp_path, rate=rate)
    if refused:
        assert_refused_safely(harness, spy)
    else:
        assert_opened(harness, spy)


def test_exact_equality_is_the_cost_the_engine_computes(tmp_path):
    """The at-limit case really is equality in the engine's arithmetic: the
    witness above would prove nothing if `-1 * rate` landed a ulp away."""
    assert -1 * float(D(str(AT_LIMIT))) == RiskLimits().max_funding_cost_rate == LIMIT


# ---------------------------------------------------------------------------
# C. sign and side
# ---------------------------------------------------------------------------
def test_short_perpetual_favourable_rate_is_not_refused_even_above_the_limit_in_magnitude(
    tmp_path,
):
    """A short RECEIVES a positive rate. `abs(rate)` would refuse this one."""
    assert abs(FAVOURABLE) > LIMIT
    harness, spy, _ = run(tmp_path, rate=FAVOURABLE)
    assert_opened(harness, spy)


def aegis_spy(executor):
    calls: list[dict] = []
    original = executor.risk.evaluate_entry

    def evaluate_entry(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    executor.risk.evaluate_entry = evaluate_entry
    return calls


@pytest.mark.parametrize(
    "side, rate, refused",
    [
        (PositionSide.LONG, 0.001, True),  # a long PAYS a positive rate
        (PositionSide.LONG, -0.001, False),  # and receives a negative one
        (PositionSide.SHORT, -0.001, True),  # a short pays a negative rate
        (PositionSide.SHORT, 0.001, False),  # and receives a positive one
    ],
    ids=["long-adverse", "long-favourable", "short-adverse", "short-favourable"],
)
def test_the_sign_convention_through_the_real_executor(side, rate, refused):
    venue = dry_run_venue(ExplodingVenue) if refused else None
    executor = make_executor(venue=venue, limits=wide_limits(max_funding_cost_rate=LIMIT))
    calls = aegis_spy(executor)

    (record,) = move_to(executor, side, "0.5", funding_rate=rate)

    # Forwarded unchanged, with the intent's own side.
    assert calls[0]["funding_rate"] == rate
    assert calls[0]["position_side"] is side
    if refused:
        assert record.state is OrderState.REJECTED
        assert record.reason.startswith(FUNDING_REASON)
        assert executor.venue.reached == []
        assert executor.position(SYMBOL).is_flat
    else:
        assert record.state is OrderState.FILLED
        assert executor.position(SYMBOL).side is side


def test_the_sign_blind_fallback_applies_only_without_a_side():
    """No side named: the magnitude is all that can be judged (`max_funding_rate`)."""
    engine = RiskEngine(wide_limits(max_funding_cost_rate=LIMIT, max_funding_rate=LIMIT))
    engine.update_equity(100_000.0)
    args = ("BTC/USDT:USDT", 100_000.0, 60_000.0, 57_000.0)
    blind = engine.evaluate_entry(*args, proposed_stake=100.0, funding_rate=0.001)
    sided = engine.evaluate_entry(
        *args, proposed_stake=100.0, funding_rate=0.001, position_side=PositionSide.SHORT
    )
    assert not blind.allowed and blind.reason.startswith("funding rate 0.00100 exceeds")
    assert sided.allowed


@pytest.mark.parametrize("value", [float("nan"), math.inf, -math.inf])
@pytest.mark.parametrize("side", [PositionSide.LONG, PositionSide.SHORT, None])
def test_aegis_refuses_a_non_finite_funding_rate_on_either_side(value, side):
    """`>` against a NaN is always False and `+inf` is a rebate to a short, so
    without the guard both would be approvals."""
    engine = RiskEngine(wide_limits(max_funding_cost_rate=LIMIT))
    engine.update_equity(100_000.0)
    verdict = engine.evaluate_entry(
        "BTC/USDT:USDT",
        100_000.0,
        60_000.0,
        57_000.0,
        proposed_stake=100.0,
        funding_rate=value,
        position_side=side,
    )
    assert not verdict.allowed and "not a finite number" in verdict.reason


# ---------------------------------------------------------------------------
# D. the perpetual leg only; perp first
# ---------------------------------------------------------------------------
def test_the_rate_goes_to_the_perpetual_and_never_to_the_spot(tmp_path):
    harness = build(tmp_path, mark_funding_rate=ADVERSE_BELOW)
    seen: dict[str, list[dict]] = {PERP: [], SPOT: []}
    for leg, executor in (
        (PERP, harness.runner.position.perp),
        (SPOT, harness.runner.position.spot),
    ):
        original = executor.execute_target

        def execute_target(*args, _leg=leg, _original=original, **kwargs):
            seen[_leg].append(kwargs)
            return _original(*args, **kwargs)

        executor.execute_target = execute_target
    harness.run(1)
    assert [k.get("funding_rate") for k in seen[PERP]] == [ADVERSE_BELOW]
    assert seen[SPOT] and all("funding_rate" not in k for k in seen[SPOT])


def test_a_spot_increase_is_never_funding_gated(position):  # noqa: F811
    """PARTIAL with the perpetual held and the spot missing: the correction's
    spot increase proceeds under a rate the perpetual could not have opened on."""
    bar = minute(0)
    book(position.model, bar)
    position.apply(position.plan(HedgeTarget(D("0.500")), bar)[:1], bar, equity=HEDGE_EQUITY)
    assert position.state is HedgeState.PARTIAL and position.leg(PERP).quantity == D("0.500")

    adverse = replace(minute(1), funding_rate_current=D("-0.01"))
    book(position.model, adverse)
    intents = position.plan(HedgeTarget(D("0.500")), adverse)
    assert [i.leg for i in intents] == [SPOT]
    outcome = position.apply(intents, adverse, equity=HEDGE_EQUITY)
    assert outcome.state is HedgeState.HEDGED
    assert position.leg(SPOT).quantity == D("0.500")


def test_a_perp_funding_veto_leaves_no_naked_spot_and_needs_no_correction(tmp_path):
    harness, spy, outcomes = run(tmp_path, rate=ADVERSE_ABOVE, minutes=5)
    assert_refused_safely(harness, spy)
    assert harness.runner.position.state is not HedgeState.PARTIAL
    assert harness.runner.position.ledger.state.correction is None
    assert [o.kind.value for o in outcomes] == ["DECISION"] * 5
    assert harness.runner.state is RunnerState.READY


# ---------------------------------------------------------------------------
# E. missing, malformed, non-finite input: the minute is incomplete
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, float("nan")], ids=["absent", "nan"])
def test_a_minute_whose_mark_has_no_usable_rate_is_incomplete(tmp_path, bad):
    """A rate the veto cannot judge is never read as zero or as "passed": the
    minute is INCOMPLETE_STATE (section 2.2) and no rule evaluates. The next
    minute, with a rate, decides normally."""
    harness, spy, outcomes = run(tmp_path, rate=ADVERSE_BELOW, rates={T0: bad}, minutes=2)
    assert [o.kind.value for o in outcomes] == ["INCOMPLETE_STATE", "DECISION"]
    incomplete = next(r for r in harness.records() if r["kind"] == "INCOMPLETE_STATE")
    assert "um_funding_rate" in json.dumps(incomplete)
    assert_opened(harness, spy)
    assert (
        decisions(harness)[0]["minute"]
        == pd.Timestamp(T0 + MINUTE_MS, unit="ms", tz="UTC").isoformat()
    )


@pytest.mark.parametrize("bad", [math.inf, -math.inf], ids=["+inf", "-inf"])
def test_an_infinite_rate_in_a_file_halts_before_any_decision(tmp_path, bad):
    """An infinite rate cannot be written through the recorder (its normalizer
    stops the day, and the fixture's frame validation refuses it). Written
    around both, the feed's per-row digest refuses it: the runner halts on
    `feed_unreadable` and nothing is decided or sent. Stronger than the
    incomplete-minute path, and reached before Aegis's own non-finite refusal."""
    harness = build(tmp_path, start=False, mark_funding_rate=ADVERSE_BELOW)
    parquet = harness.feed.normalizer.parquet_path("um", DAY)
    frame = pd.read_parquet(parquet)
    frame.loc[0, "funding_rate_last"] = bad
    frame.to_parquet(parquet, index=False)
    harness.runner.start()
    spy = SubmitSpy(harness)
    outcome = harness.tick(T0)
    assert outcome.kind.value == "HALT" and harness.runner.state is RunnerState.HALT
    assert harness.runner.halt_reason.startswith("feed_unreadable:")
    assert "funding_rate_last" in harness.runner.halt_reason
    assert orders(harness, PERP) == [] and spy.calls == {PERP: [], SPOT: []}


def test_a_present_rate_makes_the_minute_complete(tmp_path):
    """The other side: the same minute with a rate is a DECISION, not incomplete."""
    harness, _, outcomes = run(tmp_path, rate=ADVERSE_BELOW, minutes=1)
    assert [o.kind.value for o in outcomes] == ["DECISION"]
    state = harness.runner.cursor.state_for(T0)
    assert state.complete and state.funding_rate_current == D(str(ADVERSE_BELOW))
    assert "um_funding_rate" not in state.missing


def test_the_hedge_refuses_a_perpetual_increase_on_a_state_without_a_rate(
    position,  # noqa: F811
):
    """Defence in depth for a state that broke the feed's contract: loud, before
    any order, rather than `funding_rate=None`, which Aegis would approve."""
    bar = replace(minute(0), funding_rate_current=None)
    book(position.model, bar)
    with pytest.raises(HedgeError, match="no funding rate in effect"):
        position.apply(position.plan(HedgeTarget(D("0.500")), bar), bar, equity=HEDGE_EQUITY)
    assert position.leg(PERP).quantity == D("0") and position.leg(SPOT).quantity == D("0")
    assert not position.perp.store.state.orders


def test_a_perpetual_reduction_needs_no_rate(position):  # noqa: F811
    bar = minute(0)
    book(position.model, bar)
    position.apply(position.plan(HedgeTarget(D("0.500")), bar), bar, equity=HEDGE_EQUITY)
    later = replace(minute(1), funding_rate_current=None)
    book(position.model, later)
    outcome = position.apply(
        position.plan(HedgeTarget(D("0")), later), later, equity=HEDGE_EQUITY
    )
    assert outcome.state is HedgeState.FLAT


# ---------------------------------------------------------------------------
# F. provenance: this minute's rate, and it is in the evidence
# ---------------------------------------------------------------------------
def test_the_decision_uses_its_own_minutes_rate(tmp_path):
    """Adverse, benign, adverse, adverse: the position opens at minute 1 and
    only there. The previous minute's rate, the next minute's, the realised
    settlement (+0.0001, benign) or a constant would each open it elsewhere."""
    rates = {T0: ADVERSE_ABOVE, T0 + MINUTE_MS: ADVERSE_BELOW}
    rates.update({T0 + k * MINUTE_MS: ADVERSE_ABOVE for k in range(2, 5)})
    harness, spy, _ = run(tmp_path, rates=rates, minutes=4)
    assert_opened_at = [
        (r["minute"], [e["state"] for e in r["execution"]]) for r in decisions(harness)
    ]
    assert assert_opened_at[0][1] == ["REJECTED"]
    assert sorted(assert_opened_at[1][1]) == ["FILLED", "FILLED"]
    assert assert_opened_at[2][1] == [] and assert_opened_at[3][1] == []
    assert decisions(harness)[1]["inputs"]["funding_current"] == str(ADVERSE_BELOW)


def test_the_record_names_the_rate_consumed_and_the_refusal(tmp_path):
    harness, _, _ = run(tmp_path, rate=ADVERSE_ABOVE, minutes=1)
    (record,) = decisions(harness)
    assert record["inputs"]["funding_current"] == str(ADVERSE_ABOVE)
    assert record["inputs"]["funding_last"] != record["inputs"]["funding_current"]
    (perp,) = record["execution"]
    assert perp["leg"] == "perp" and perp["state"] == "REJECTED"
    assert perp["filled_qty"] == "0" and perp["reason"].startswith(FUNDING_REASON)
    assert [a["leg"] for a in record["requested_action"]] == ["perp", "spot"]


def test_allowed_execution_evidence_is_ordinary(tmp_path):
    harness, _, _ = run(tmp_path, rate=ADVERSE_BELOW, minutes=1)
    (record,) = decisions(harness)
    assert sorted(e["state"] for e in record["execution"]) == ["FILLED", "FILLED"]
    assert all(e["reason"] == "" for e in record["execution"])


def test_changing_only_the_rate_moves_the_canonical_input_evidence():
    base = dict(minute_ns=T0 * 1_000_000, minute="m", complete=True)
    one = MarketState(**base, funding_rate_current=D("0.0001"))
    other = MarketState(**base, funding_rate_current=D("-0.0001"))
    assert one.canonical()["funding_rate_current"] == "0.0001"
    assert one.canonical() != other.canonical()


def test_the_inputs_hash_of_a_live_record_moves_with_the_rate_alone(tmp_path):
    one, _, _ = run(tmp_path / "a", rate=ADVERSE_BELOW, minutes=1)
    other, _, _ = run(tmp_path / "b", rate=MINUS_EPSILON, minutes=1)
    a, b = decisions(one)[0]["inputs"], decisions(other)[0]["inputs"]
    assert a["inputs_hash"] != b["inputs_hash"]
    assert {
        k: v
        for k, v in a.items()
        if k not in ("inputs_hash", "funding_current", "um_minute_digest")
    } == {
        k: v
        for k, v in b.items()
        if k not in ("inputs_hash", "funding_current", "um_minute_digest")
    }


# ---------------------------------------------------------------------------
# G/H. reductions stay possible; increases are gated; no-change is no order
# ---------------------------------------------------------------------------
def test_a_held_position_under_an_adverse_rate_is_not_touched(tmp_path):
    """No-change target: holding, and the rate turns adverse. Nothing is
    planned, so nothing is refused -- no fabricated veto."""
    rates = {T0 + k * MINUTE_MS: ADVERSE_ABOVE for k in range(1, 5)}
    harness, spy, _ = run(tmp_path, rate=ADVERSE_BELOW, rates=rates, minutes=5)
    assert harness.runner.position.state is HedgeState.HEDGED
    assert all(o.state is OrderState.FILLED for o in orders(harness, PERP))
    assert all(r["execution"] == [] for r in decisions(harness)[1:])


def test_a_full_flatten_while_the_rate_is_above_the_limit(tmp_path):
    rates = {T0 + k * MINUTE_MS: ADVERSE_ABOVE for k in range(1, 5)}
    harness, _, _ = run(tmp_path, rate=ADVERSE_BELOW, rates=rates, minutes=3)
    assert harness.runner.position.state is HedgeState.HEDGED
    harness.runner.flatten("R1-k: flatten under an adverse rate")
    assert harness.runner.position.leg(PERP).quantity == D("0")
    assert harness.runner.position.leg(SPOT).quantity == D("0")


def test_an_emergency_reduce_while_the_rate_is_above_the_limit(position):  # noqa: F811
    bar = minute(0)
    book(position.model, bar)
    position.apply(position.plan(HedgeTarget(D("0.500")), bar), bar, equity=HEDGE_EQUITY)
    adverse = replace(minute(1), funding_rate_current=D("-0.01"))
    book(position.model, adverse)
    from chimera.futures.executor import FlattenCause

    outcome = position.emergency_reduce(FlattenCause.HEDGE_CORRECTION, adverse)
    assert outcome.state is HedgeState.FLAT


def test_a_partial_reduction_while_the_rate_is_above_the_limit(position):  # noqa: F811
    bar = minute(0)
    book(position.model, bar)
    position.apply(position.plan(HedgeTarget(D("0.500")), bar), bar, equity=HEDGE_EQUITY)
    adverse = replace(minute(1), funding_rate_current=D("-0.01"))
    book(position.model, adverse)
    outcome = position.apply(
        position.plan(HedgeTarget(D("0.200")), adverse), adverse, equity=HEDGE_EQUITY
    )
    assert outcome.state is HedgeState.HEDGED
    assert position.leg(PERP).quantity == position.leg(SPOT).quantity == D("0.200")


def test_an_increase_of_an_open_perpetual_is_refused_above_the_limit():
    executor = make_executor(limits=wide_limits(max_funding_cost_rate=LIMIT))
    (opened,) = move_to(executor, PositionSide.SHORT, "0.5", funding_rate=-0.0001)
    assert opened.state is OrderState.FILLED
    (grown,) = move_to(executor, PositionSide.SHORT, "1.0", funding_rate=-0.001)
    assert grown.state is OrderState.REJECTED and grown.reason.startswith(FUNDING_REASON)
    assert executor.position(SYMBOL).quantity == D("0.5")


def test_a_decrease_of_an_open_perpetual_is_never_asked_about_funding():
    executor = make_executor(limits=wide_limits(max_funding_cost_rate=LIMIT))
    move_to(executor, PositionSide.SHORT, "1.0", funding_rate=-0.0001)
    calls = aegis_spy(executor)
    (reduced,) = move_to(executor, PositionSide.SHORT, "0.4", funding_rate=-0.01)
    (flat,) = move_to(executor, PositionSide.FLAT, "0", funding_rate=-0.01)
    assert reduced.state is OrderState.FILLED and flat.state is OrderState.FILLED
    assert calls == [], "a reduction was sent to the entry gate"
    assert executor.position(SYMBOL).is_flat


def test_a_no_change_target_plans_no_order_and_asks_nothing():
    executor = make_executor(limits=wide_limits(max_funding_cost_rate=LIMIT))
    move_to(executor, PositionSide.SHORT, "0.5", funding_rate=-0.0001)
    calls = aegis_spy(executor)
    assert move_to(executor, PositionSide.SHORT, "0.5", funding_rate=-0.01) == []
    assert calls == []


# ---------------------------------------------------------------------------
# I. the rest of Aegis still composes
# ---------------------------------------------------------------------------
def test_a_benign_rate_does_not_bypass_another_guard(tmp_path):
    harness, spy, _ = run(tmp_path, rate=ADVERSE_BELOW, limits={"max_leverage": 0.5})
    (first, *_rest) = orders(harness, PERP)
    assert first.state is OrderState.REJECTED and first.reason.startswith(
        "leverage 1.00 exceeds"
    )
    assert spy.calls == {PERP: [], SPOT: []}


def test_two_breaches_refuse_in_the_existing_order(tmp_path):
    """Funding is checked before leverage in `evaluate_entry`, as before R1-k."""
    harness, _, _ = run(tmp_path, rate=ADVERSE_ABOVE, limits={"max_leverage": 0.5})
    assert orders(harness, PERP)[0].reason.startswith(FUNDING_REASON)


# ---------------------------------------------------------------------------
# J. restart
# ---------------------------------------------------------------------------
def test_the_same_funding_state_is_judged_the_same_after_a_restart(tmp_path):
    """Refused before the restart; refused, for the same reason, on the next
    minute after it; and the position opens on the first benign minute."""
    rates = {T0 + k * MINUTE_MS: ADVERSE_ABOVE for k in range(0, 4)}
    first = build(tmp_path, mark_funding_rate=ADVERSE_BELOW, mark_funding_rates=rates)
    first.run(2)
    before = [o.reason for o in orders(first, PERP)]
    first.runner.shutdown("R1-k restart witness")

    again = build(
        tmp_path,
        config=first.runner.config,
        start=False,
        mark_funding_rate=ADVERSE_BELOW,
        mark_funding_rates=rates,
    )
    again.runner.start()
    again.tick(T0 + 2 * MINUTE_MS)
    after = [o.reason for o in orders(again, PERP)]
    assert after[: len(before)] == before and len(after) == len(before) + 1
    assert after[-1] == before[-1]
    again.tick(T0 + 3 * MINUTE_MS)
    again.tick(T0 + 4 * MINUTE_MS)
    assert again.runner.position.state is HedgeState.HEDGED
    assert decisions(again)[-1]["inputs"]["funding_current"] == str(ADVERSE_BELOW)


# ---------------------------------------------------------------------------
# K. replay parity, refused and allowed minutes in one log
# ---------------------------------------------------------------------------
def test_live_and_replay_agree_on_refused_and_allowed_minutes(tmp_path, capsys):
    rates = {T0 + k * MINUTE_MS: ADVERSE_ABOVE for k in range(0, 3)}
    live = build(tmp_path / "live", mark_funding_rate=ADVERSE_BELOW, mark_funding_rates=rates)
    live.run(6)
    live.runner.shutdown("done")
    assert refused_on_funding(live) and live.runner.position.state is HedgeState.HEDGED

    argv = [
        "--config", str(written_config(tmp_path, live)),
        "--root", str(live.root),
        "--live-log", str(live.state_dir / "decision_log"),
        "--days", DAY,
        "--profile", live.runner.config.profile.value,
        "--scratch", str(tmp_path / "scratch"),
        "--json",
    ]  # fmt: skip
    code = replay_parity.main(argv)
    report = json.loads(capsys.readouterr().out)
    assert code == replay_parity.EXIT_PARITY and report["parity"] is True, report
    assert report["records_compared"] >= 6
    assert not report["live_only"] and not report["replay_only"]


def test_a_different_rate_is_a_parity_divergence(tmp_path):
    """The negative control: the funding input is part of what parity compares."""
    rates = {T0: ADVERSE_ABOVE}
    live, _, _ = run(tmp_path / "a", rate=ADVERSE_BELOW, rates=rates, minutes=3)
    other, _, _ = run(tmp_path / "b", rate=ADVERSE_BELOW, minutes=3)
    report = replay_parity.compare_logs(live.records(), other.records())
    assert not report.ok
