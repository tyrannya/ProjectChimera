"""Section 7.4's limits, from the campaign file to the guard that enforces them.

The defect this file exists to close: `tools/demo_run.py` built
``RiskLimits(max_position_pct=..., risk_per_trade_pct=0.5)`` and nothing else, so
a campaign was HASHED under `conf/demo/pvc1.json`'s thirteen limits and ENFORCED
under `RiskLimits`' defaults -- a 5% drawdown halt running at 15%, a 1.0 leverage
ceiling running at 3.0, four orders a minute running at ten. `tests/demo_harness.py`
built its own two-literal `RiskLimits` beside it, which is why a suite that
exercised funding, reconciliation, liquidation and replay end to end never saw it.

So a dataclass-equality test is not enough here and is mostly not what is below.
Every limit gets **two witnesses**: that a non-default campaign value arrives in
`RiskLimits`, and that changing it changes what the engine DOES -- a halt that
fires or does not, a veto that fires or does not, a cooldown of a different
length. Each behavioural witness is built from two campaign configurations that
differ in exactly one limit, so the thing being proved is that the limit is
what moved the behaviour, not that some engine somewhere refuses something.

The last section is the one that keeps the defect closed: the CLI and the harness
may not construct `RiskLimits` or `RiskEngine` at all, asserted over their ASTs
with a negative control, and must agree field for field on the same config.
"""

from __future__ import annotations

import ast
import json
from dataclasses import fields, replace
from decimal import Decimal
from pathlib import Path

import pytest

from chimera.demo.config import COUNT_LIMITS, DemoLimits, parse_demo_config
from chimera.demo.risk_wiring import (
    DEMO_RISK_PER_TRADE_PCT,
    DERIVED_LIMITS,
    DIRECT_LIMITS,
    UNCONFIGURED_LIMITS,
    RiskWiringError,
    build_risk_engine,
    check_exhaustive,
    risk_limits,
    sizing_cap_binds,
)
from chimera.futures.domain import PositionSide
from chimera.risk import RiskEngine, RiskLimits
from tests.demo_harness import build, campaign_config

REPO = Path(__file__).resolve().parents[1]
PVC1 = REPO / "conf" / "demo" / "pvc1.json"
CLI = REPO / "tools" / "demo_run.py"
HARNESS = REPO / "tests" / "demo_harness.py"

#: Section 7.4 of the adopted master plan, transcribed from the section and not
#: from the file, so that a limit edited in `conf/demo/pvc1.json` without a plan
#: amendment fails here.
SECTION_7_4 = {
    "max_drawdown_pct": 0.05,
    "max_daily_loss_pct": 0.02,
    "max_open_positions": 2,
    "max_total_exposure_pct": 1.0,
    "max_exposure_per_asset_pct": 1.0,
    "max_leverage": 1.0,
    "max_orders_per_minute": 4,
    "loss_streak_limit": 3,
    "cooldown_seconds": 3600,
    "max_data_delay_s": 180,
    "max_funding_cost_rate": 0.0005,
    "funding_adverse_streak_limit": 3,
    "min_liquidation_distance_pct": 0.5,
}

#: A value for each limit that is neither section 7.4's nor `RiskLimits`' default,
#: so that a mapping which silently kept either would be visible. Counts stay
#: integers; ratios stay inside a range the engine will accept.
PROBE = {
    "funding_adverse_streak_limit": 7,
    "loss_streak_limit": 6,
    "max_open_positions": 5,
    "max_orders_per_minute": 9,
    "cooldown_seconds": 1234.0,
    "max_daily_loss_pct": 0.037,
    "max_data_delay_s": 222.0,
    "max_drawdown_pct": 0.083,
    "max_exposure_per_asset_pct": 0.61,
    "max_funding_cost_rate": 0.00037,
    "max_leverage": 2.5,
    "max_total_exposure_pct": 0.77,
    "min_liquidation_distance_pct": 0.31,
}


def committed() -> DemoLimits:
    """The committed campaign's limits, parsed exactly as the runner parses them."""
    return parse_demo_config(json.loads(PVC1.read_text(encoding="utf-8"))).limits


def demo_limits(**overrides: float | int) -> DemoLimits:
    """The committed limits with some fields moved."""
    return replace(committed(), **overrides)


def engine(tmp_path: Path, limits: DemoLimits, *, equity: float = 1_000_000.0) -> RiskEngine:
    """An engine on `limits`, seeded with equity, in its own state directory."""
    eng = RiskEngine(risk_limits(limits), state_path=tmp_path / "risk.json")
    eng.update_equity(equity)
    return eng


# ---------------------------------------------------------------------------
# 1. The table is total: nothing is dropped, nothing is invented
# ---------------------------------------------------------------------------
def test_every_demo_limit_reaches_a_risk_limit():
    """The thirteen of section 7.4, each with a named destination."""
    assert set(DIRECT_LIMITS) == {f.name for f in fields(DemoLimits)}
    assert set(DIRECT_LIMITS) == set(SECTION_7_4)


def test_every_risk_limit_is_accounted_for_exactly_once():
    claimed = [*DIRECT_LIMITS.values(), *DERIVED_LIMITS, *UNCONFIGURED_LIMITS]
    assert sorted(claimed) == sorted({f.name for f in fields(RiskLimits)})
    assert len(claimed) == len(set(claimed)), "a RiskLimits field is claimed twice"


def test_the_committed_campaign_is_section_7_4():
    """The file and the plan agree. If they stop agreeing, this is the alarm."""
    limits = committed()
    for name, expected in SECTION_7_4.items():
        assert getattr(limits, name) == pytest.approx(expected), name


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"direct": {}}, "reach no RiskLimits field"),
        (
            {"direct": {**DIRECT_LIMITS, "invented": "max_leverage"}},
            "no such DemoLimits field",
        ),
        ({"unconfigured": {}}, "does not account for"),
        (
            {"unconfigured": {**UNCONFIGURED_LIMITS, "not_a_limit": "why"}},
            "no such RiskLimits field",
        ),
        (
            {"derived": {**DERIVED_LIMITS, "max_leverage": "max_leverage"}},
            "claimed twice",
        ),
        (
            {"derived": {**DERIVED_LIMITS, "max_position_pct": "not_a_demo_limit"}},
            "no such DemoLimits field",
        ),
    ],
)
def test_the_exhaustiveness_guard_refuses_a_broken_table(kwargs, message):
    """Every refusal, driven by a table that really is broken.

    The guard runs at import, where it can only ever be seen to pass. A guard
    asserted only against the correct tables proves nothing about the day
    somebody adds a field.
    """
    with pytest.raises(RiskWiringError, match=message):
        check_exhaustive(**kwargs)


def test_a_new_field_on_either_type_is_refused():
    """The case the guard is really for: a limit somebody adds later."""
    with pytest.raises(RiskWiringError, match="reach no RiskLimits field"):
        check_exhaustive(demo_fields={*DIRECT_LIMITS, "max_slippage_pct"})
    with pytest.raises(RiskWiringError, match="does not account for"):
        check_exhaustive(risk_fields={f.name for f in fields(RiskLimits)} | {"max_gap_pct"})


# ---------------------------------------------------------------------------
# 2. Every mapped field carries its value, and carries it alone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("demo_field, risk_field", sorted(DIRECT_LIMITS.items()))
def test_a_campaign_value_reaches_its_risk_limit(demo_field, risk_field):
    """Half one: the value arrives, and it is not the RiskLimits default."""
    probe = PROBE[demo_field]
    mapped = risk_limits(demo_limits(**{demo_field: probe}))
    assert getattr(mapped, risk_field) == pytest.approx(probe)
    assert getattr(mapped, risk_field) != pytest.approx(getattr(RiskLimits(), risk_field)), (
        f"{demo_field} probe collides with the RiskLimits default; the test could not "
        "tell a mapping from an omission"
    )
    assert isinstance(
        getattr(mapped, risk_field), int if demo_field in COUNT_LIMITS else float
    )


@pytest.mark.parametrize("demo_field, risk_field", sorted(DIRECT_LIMITS.items()))
def test_moving_a_campaign_value_moves_only_its_risk_limit(demo_field, risk_field):
    """Half two: the change is visible, and it is confined to one field.

    Confinement is what a per-field mutation needs to mean anything: without it a
    mapping that assigned every limit from `max_leverage` would pass half one.
    """
    base = risk_limits(committed())
    moved = risk_limits(demo_limits(**{demo_field: PROBE[demo_field]}))
    assert getattr(moved, risk_field) != getattr(base, risk_field)

    derived_from = {v for v in DERIVED_LIMITS.values()}
    expected_moved = {risk_field} | {
        target for target, source in DERIVED_LIMITS.items() if source == demo_field
    }
    changed = {
        f.name for f in fields(RiskLimits) if getattr(moved, f.name) != getattr(base, f.name)
    }
    assert changed == expected_moved, (
        f"moving {demo_field} moved {sorted(changed - expected_moved)} as well "
        f"(fields derived from a campaign limit: {sorted(derived_from)})"
    )


def test_the_derived_fields_follow_the_campaign_limit_they_are_derived_from():
    """`max_position_pct` and `max_funding_rate` have no section 7.4 row of their own."""
    moved = risk_limits(
        demo_limits(max_exposure_per_asset_pct=0.42, max_funding_cost_rate=0.00021)
    )
    assert moved.max_position_pct == pytest.approx(0.42)
    assert moved.max_exposure_per_asset_pct == pytest.approx(0.42)
    assert moved.max_funding_rate == pytest.approx(0.00021)
    assert moved.max_funding_cost_rate == pytest.approx(0.00021)


def test_the_sign_blind_funding_ceiling_stays_a_superset_of_the_side_aware_one():
    """Section 7.2: the sign-blind check "cannot approve anything the side-aware
    check would refuse". That holds only while it is no looser."""
    for rate in (0.0005, 0.00037, 0.002, 1e-6):
        mapped = risk_limits(demo_limits(max_funding_cost_rate=rate))
        assert mapped.max_funding_rate <= mapped.max_funding_cost_rate


def test_the_unconfigured_fields_keep_their_risk_limits_default():
    """Named in UNCONFIGURED_LIMITS with a reason, and left exactly where they were."""
    mapped = risk_limits(committed())
    default = RiskLimits()
    for name in UNCONFIGURED_LIMITS:
        if name == "risk_per_trade_pct":
            continue
        assert getattr(mapped, name) == getattr(default, name), name
    assert UNCONFIGURED_LIMITS.keys() >= {"risk_per_trade_pct", "max_inference_staleness_s"}
    assert all(reason.strip() for reason in UNCONFIGURED_LIMITS.values())


def test_the_invented_stop_never_decides_a_size():
    """`risk_per_trade_pct`'s invariant, on the committed campaign and around it.

    A carry hedge has no stop; `FuturesExecutor._implied_stop` fabricates one so
    `position_size`'s arithmetic is defined. The number that may decide a stake is
    therefore the exposure ceiling, and the check is that the cap binds for every
    stop distance the band admits.
    """
    mapped = risk_limits(committed())
    equity = 1_000_000.0
    eng = RiskEngine(mapped)
    for distance in (
        mapped.min_stop_distance_pct,
        (mapped.min_stop_distance_pct + mapped.max_stop_distance_pct) / 2,
        mapped.max_stop_distance_pct,
    ):
        stake = eng.position_size(equity, 100.0, 100.0 * (1 - distance), leverage=1.0)
        assert stake == pytest.approx(equity * mapped.max_position_pct), distance
    # The constant really is what the mapping emits -- `UNCONFIGURED_LIMITS`'
    # default check has to skip this field, so nothing else pins it.
    assert mapped.risk_per_trade_pct == pytest.approx(DEMO_RISK_PER_TRADE_PCT)
    assert sizing_cap_binds(mapped)


def cap_really_binds(limits, equity: float = 1_000_000.0) -> bool:
    """What `position_size` ACTUALLY does at the worst case, computed not asserted.

    The widest stop the band admits, at the highest leverage the engine will
    accept. `sizing_cap_binds` is a claim about this; this is the observation.
    """
    stake = RiskEngine(limits).position_size(
        equity,
        100.0,
        100.0 * (1 - limits.max_stop_distance_pct),
        leverage=limits.max_leverage,
    )
    return bool(stake == pytest.approx(equity * limits.max_position_pct))


@pytest.mark.parametrize(
    "max_leverage, expected",
    [
        # 0.5 >= 1.0 * 0.15 * 1.0 -> the committed campaign, with room to spare.
        (1.0, True),
        # 0.5 >= 1.0 * 0.15 * 3.0 = 0.45 -> still binds, barely.
        (3.0, True),
        # 0.5 >= 1.0 * 0.15 * 4.0 = 0.60 -> it does NOT, and the invented stop
        # starts deciding sizes. WITHOUT the leverage factor the predicate reads
        # `0.5 >= 0.15` and wrongly says True here, so this row is what holds the
        # factor in place.
        (4.0, False),
    ],
)
def test_the_sizing_invariant_counts_leverage(max_leverage, expected):
    """`sizing_cap_binds` must agree with `position_size`, not just with itself.

    `position_size` divides the stop-based notional by
    `max(1.0, min(leverage, max_leverage))`, so a HIGHER ceiling makes the
    stop-based stake smaller and the cap less likely to bind. A predicate that
    omits the factor is right only at 1x -- which is the committed campaign, and
    therefore exactly the case that would never reveal the omission.
    """
    mapped = risk_limits(demo_limits(max_leverage=max_leverage))
    assert mapped.max_leverage == pytest.approx(max_leverage)
    assert sizing_cap_binds(mapped) is expected
    # The claim and the behaviour, on the same limits.
    assert cap_really_binds(mapped) is expected


def test_nothing_on_the_demo_path_supplies_an_inference_age():
    """Why `max_inference_staleness_s` has no campaign source (section 7.2).

    "stale inference | n/a in the demo (no model) | pass None". Asserted over the
    source, because the claim is that no code path CAN reach the guard.
    """
    demo_path = [
        *(REPO / "chimera" / "demo").rglob("*.py"),
        *(REPO / "chimera" / "carry").rglob("*.py"),
        CLI,
    ]
    offenders = []
    for path in sorted(demo_path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "inference_age_s":
                offenders.append(f"{path.relative_to(REPO)}:{node.value.lineno}")
    assert offenders == [], (
        f"{offenders} passes an inference age, so max_inference_staleness_s is reachable "
        "and UNCONFIGURED_LIMITS' reason for leaving it on a default is no longer true"
    )


# ---------------------------------------------------------------------------
# 3. Behavioural witnesses: the guard, not the dataclass
# ---------------------------------------------------------------------------
def test_max_orders_per_minute_halts_on_the_campaign_count(tmp_path):
    tight = engine(tmp_path / "tight", committed())
    loose = engine(tmp_path / "loose", demo_limits(max_orders_per_minute=9))
    for _ in range(SECTION_7_4["max_orders_per_minute"]):
        tight.record_order()
        loose.record_order()
    assert not tight.state.halted, "the campaign's fourth order is still allowed"
    tight.record_order()
    loose.record_order()
    assert tight.state.halted and "4" in tight.state.halt_reason
    assert not loose.state.halted, "the raised limit did not raise the guard"


def test_max_drawdown_pct_halts_where_the_campaign_says(tmp_path):
    # The daily-loss budget is loosened on BOTH sides: a 6% fall from peak is also
    # a 6% fall on the day, and a witness for the drawdown limit must not be able
    # to pass because a different guard fired.
    tight = engine(tmp_path / "tight", demo_limits(max_daily_loss_pct=0.90))
    loose = engine(
        tmp_path / "loose", demo_limits(max_daily_loss_pct=0.90, max_drawdown_pct=0.30)
    )
    # A 6% fall from peak: past section 7.4's 5%, short of the 30% and of the
    # 15% RiskLimits default that the broken wiring left in force.
    for eng in (tight, loose):
        eng.update_equity(940_000.0)
    assert tight.state.halted and "drawdown" in tight.state.halt_reason
    assert not loose.state.halted


def test_max_daily_loss_pct_halts_where_the_campaign_says(tmp_path):
    # 3% down on the day: past 2%, short of the 5% RiskLimits default.
    tight = engine(tmp_path / "tight", demo_limits(max_drawdown_pct=0.90))
    loose = engine(
        tmp_path / "loose", demo_limits(max_drawdown_pct=0.90, max_daily_loss_pct=0.10)
    )
    for eng in (tight, loose):
        eng.update_equity(970_000.0)
    assert tight.state.halted and "daily loss" in tight.state.halt_reason
    assert not loose.state.halted


def test_max_open_positions_vetoes_the_third_pair(tmp_path):
    tight = engine(tmp_path / "tight", committed())
    loose = engine(tmp_path / "loose", demo_limits(max_open_positions=5))
    for eng in (tight, loose):
        eng.state.open_positions = {"BTC/USDT": 1.0, "BTC/USDT:USDT": 1.0}
    verdict = tight.evaluate_entry("ETH/USDT", 1_000_000.0, 100.0, 95.0, proposed_stake=10.0)
    assert not verdict.allowed and "max open positions" in verdict.reason
    assert loose.evaluate_entry(
        "ETH/USDT", 1_000_000.0, 100.0, 95.0, proposed_stake=10.0
    ).allowed


def test_max_total_exposure_pct_vetoes_on_the_campaign_fraction(tmp_path):
    tight = engine(tmp_path / "tight", demo_limits(max_total_exposure_pct=0.20))
    loose = engine(tmp_path / "loose", demo_limits(max_total_exposure_pct=0.80))
    for eng in (tight, loose):
        eng.state.open_positions = {"BTC/USDT": 150_000.0}
    args = ("BTC/USDT:USDT", 1_000_000.0, 100.0, 95.0)
    tight_verdict = tight.evaluate_entry(*args, proposed_stake=100_000.0)
    assert not tight_verdict.allowed and "total exposure" in tight_verdict.reason
    assert loose.evaluate_entry(*args, proposed_stake=100_000.0).allowed


def test_max_exposure_per_asset_pct_vetoes_cumulative_exposure_in_one_pair(tmp_path):
    """The field the broken wiring sent to the wrong guard.

    `max_position_pct` caps ONE order's stake; `max_exposure_per_asset_pct` caps a
    pair's CUMULATIVE exposure. The old runner set the first from the campaign and
    left the second on 0.35, so this witness -- a second order into a pair that is
    already holding -- is the one it could not have passed.
    """
    tight = engine(tmp_path / "tight", demo_limits(max_exposure_per_asset_pct=0.20))
    loose = engine(tmp_path / "loose", demo_limits(max_exposure_per_asset_pct=0.80))
    for eng in (tight, loose):
        eng.state.open_positions = {"BTC/USDT": 150_000.0}
    args = ("BTC/USDT", 1_000_000.0, 100.0, 95.0)
    tight_verdict = tight.evaluate_entry(*args, proposed_stake=100_000.0)
    assert not tight_verdict.allowed and "exposure in BTC/USDT" in tight_verdict.reason
    assert loose.evaluate_entry(*args, proposed_stake=100_000.0).allowed


def test_max_leverage_vetoes_above_the_campaign_ceiling(tmp_path):
    tight = engine(tmp_path / "tight", committed())
    loose = engine(tmp_path / "loose", demo_limits(max_leverage=3.0))
    args = ("BTC/USDT", 1_000_000.0, 100.0, 95.0)
    verdict = tight.evaluate_entry(*args, leverage=2.0, proposed_stake=10_000.0)
    assert not verdict.allowed and "leverage" in verdict.reason
    assert loose.evaluate_entry(*args, leverage=2.0, proposed_stake=10_000.0).allowed
    # And the campaign's own leverage, which the demo executor fixes at 1, passes.
    assert tight.evaluate_entry(*args, leverage=1.0, proposed_stake=10_000.0).allowed


def test_max_funding_cost_rate_vetoes_what_the_position_would_pay(tmp_path):
    tight = engine(tmp_path / "tight", committed())
    loose = engine(tmp_path / "loose", demo_limits(max_funding_cost_rate=0.01))
    args = ("BTC/USDT:USDT", 1_000_000.0, 100.0, 95.0)
    kwargs = dict(proposed_stake=10_000.0, funding_rate=0.001, position_side=PositionSide.LONG)
    verdict = tight.evaluate_entry(*args, **kwargs)
    assert not verdict.allowed and "funding" in verdict.reason
    assert loose.evaluate_entry(*args, **kwargs).allowed
    # A short RECEIVES that rate, so the same number is not a cost to it (A10).
    assert tight.evaluate_entry(
        *args, proposed_stake=10_000.0, funding_rate=0.001, position_side=PositionSide.SHORT
    ).allowed


def test_the_sign_blind_funding_ceiling_moves_with_the_campaign(tmp_path):
    """The derived `max_funding_rate`, witnessed where a caller names no side."""
    tight = engine(tmp_path / "tight", demo_limits(max_funding_cost_rate=0.0001))
    loose = engine(tmp_path / "loose", demo_limits(max_funding_cost_rate=0.01))
    args = ("BTC/USDT:USDT", 1_000_000.0, 100.0, 95.0)
    verdict = tight.evaluate_entry(*args, proposed_stake=10_000.0, funding_rate=-0.001)
    assert not verdict.allowed and "funding rate" in verdict.reason
    assert loose.evaluate_entry(*args, proposed_stake=10_000.0, funding_rate=-0.001).allowed


def test_funding_adverse_streak_limit_halts_on_the_campaign_count(tmp_path):
    tight = engine(tmp_path / "tight", committed())
    loose = engine(tmp_path / "loose", demo_limits(funding_adverse_streak_limit=7))
    for _ in range(SECTION_7_4["funding_adverse_streak_limit"]):
        for eng in (tight, loose):
            eng.note_funding_settlement("BTC/USDT:USDT", PositionSide.LONG, 0.0001)
    assert tight.state.funding_halt
    assert not loose.state.funding_halt
    verdict = tight.evaluate_entry(
        "BTC/USDT:USDT", 1_000_000.0, 100.0, 95.0, proposed_stake=10.0
    )
    assert not verdict.allowed and "funding halt" in verdict.reason


def test_min_liquidation_distance_pct_vetoes_a_near_liquidation(tmp_path):
    tight = engine(tmp_path / "tight", committed())
    loose = engine(tmp_path / "loose", demo_limits(min_liquidation_distance_pct=0.10))
    args = ("BTC/USDT:USDT", 1_000_000.0, 100.0, 95.0)
    # Liquidation 20% away: inside the campaign's 50%, outside the loosened 10%.
    verdict = tight.evaluate_entry(*args, proposed_stake=10_000.0, liquidation_price=80.0)
    assert not verdict.allowed and "liquidation" in verdict.reason
    assert loose.evaluate_entry(*args, proposed_stake=10_000.0, liquidation_price=80.0).allowed


def test_loss_streak_limit_and_cooldown_seconds_are_both_the_campaign_s(tmp_path):
    tight = engine(tmp_path / "tight", committed())
    loose = engine(tmp_path / "loose", demo_limits(loss_streak_limit=6, cooldown_seconds=60.0))
    for _ in range(SECTION_7_4["loss_streak_limit"]):
        for eng in (tight, loose):
            eng.record_trade_result(-1.0)
    assert (
        tight.state.cooldown_until > 0.0
    ), "three losses did not open the campaign's cooldown"
    assert loose.state.cooldown_until == 0.0, "six were configured; three should not be enough"
    verdict = tight.evaluate_entry("BTC/USDT", 1_000_000.0, 100.0, 95.0, proposed_stake=10.0)
    assert not verdict.allowed and "cooldown" in verdict.reason
    # The cooldown's LENGTH is the campaign's too, not just its existence.
    for _ in range(3):
        loose.record_trade_result(-1.0)
    tight_len = tight.state.cooldown_until - tight._clock()
    loose_len = loose.state.cooldown_until - loose._clock()
    assert tight_len == pytest.approx(SECTION_7_4["cooldown_seconds"], abs=5.0)
    assert loose_len == pytest.approx(60.0, abs=5.0)


def test_max_data_delay_s_marks_the_feed_stale_on_the_campaign_delay(tmp_path):
    tight = engine(tmp_path / "tight", committed())
    loose = engine(tmp_path / "loose", demo_limits(max_data_delay_s=600.0))
    close_ns = 1_795_312_800_000_000_000
    # 240s past the close: past the campaign's 180, inside the 300 RiskLimits
    # default the broken wiring left in force, and inside the loosened 600.
    for eng in (tight, loose):
        eng.note_feed(close_ns, close_ns + 240 * 1_000_000_000)
    assert tight.state.stale_feed_since is not None
    assert loose.state.stale_feed_since is None
    verdict = tight.evaluate_entry("BTC/USDT", 1_000_000.0, 100.0, 95.0, proposed_stake=10.0)
    assert not verdict.allowed and "market data late" in verdict.reason


# ---------------------------------------------------------------------------
# 4. The same limits, reached through the runner an operator actually starts
# ---------------------------------------------------------------------------
def test_the_runner_enforces_the_campaign_it_was_configured_with(tmp_path):
    """The end-to-end witness: the engine inside a started runner."""
    harness = build(tmp_path)
    assert harness.runner.risk.limits == risk_limits(harness.runner.config.limits)
    assert harness.runner.risk.limits == risk_limits(committed())
    assert harness.runner.risk.limits != RiskLimits()


def test_the_harness_default_is_the_committed_campaign(tmp_path):
    """No test may run on limits the campaign does not have, unless it says so."""
    assert campaign_config(tmp_path / "state").limits == committed()


def test_a_campaign_limit_changes_what_the_running_runner_refuses(tmp_path):
    """Two campaigns, one limit apart, driven through the real tick loop.

    The order-rate window is the guard that can be moved by a single campaign key
    and observed through `DemoRunner.tick` without touching anything else, so it
    is the one this witness uses.
    """
    tight = build(
        tmp_path / "tight",
        config=campaign_config(
            tmp_path / "tight" / "state", limits={"max_orders_per_minute": 1}
        ),
    )
    tight.run(3)
    assert tight.runner.risk.state.halted
    assert "order rate" in tight.runner.risk.state.halt_reason

    loose = build(
        tmp_path / "loose",
        config=campaign_config(
            tmp_path / "loose" / "state", limits={"max_orders_per_minute": 64}
        ),
    )
    loose.run(3)
    assert not loose.runner.risk.state.halted


def test_the_runner_and_the_cli_seed_the_same_equity(tmp_path):
    """`build_risk_engine` seeds equity as both entry points did by hand."""
    config = campaign_config(tmp_path / "state")
    eng = build_risk_engine(config, capital=Decimal("1000000"))
    assert eng.state.equity == pytest.approx(1_000_000.0)
    assert eng.state.peak_equity == pytest.approx(1_000_000.0)
    assert (tmp_path / "state" / "risk.json").is_file()


def test_build_risk_engine_is_pure_in_its_limits(tmp_path):
    """The mapping reads no clock and no file; two calls agree."""
    limits = committed()
    assert risk_limits(limits) == risk_limits(limits)
    assert risk_limits(limits) is not risk_limits(limits)


# ---------------------------------------------------------------------------
# 5. The two entry points cannot diverge again
# ---------------------------------------------------------------------------
def constructed_names(source: str) -> set[str]:
    """Every name called as a constructor, `Name()` or `mod.Name()`."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            found.add(func.id)
        elif isinstance(func, ast.Attribute):
            found.add(func.attr)
    return found


@pytest.mark.parametrize("path", [CLI, HARNESS], ids=lambda p: p.name)
def test_neither_entry_point_builds_its_own_aegis(path):
    """The regression itself: one mapping, or none of the guarantees above hold.

    A second `RiskLimits(...)` anywhere here is how the CLI and the harness came
    to enforce two different risk regimes while every test passed.
    """
    called = constructed_names(path.read_text(encoding="utf-8"))
    assert "RiskLimits" not in called, f"{path.name} constructs RiskLimits directly again"
    assert "RiskEngine" not in called, f"{path.name} constructs RiskEngine directly again"
    assert (
        "build_risk_engine" in called
    ), f"{path.name} no longer builds Aegis through the mapping"


def test_the_divergence_scan_catches_the_construction_it_forbids():
    """Negative control. The scan above passes trivially if it matches nothing."""
    offending = "from chimera.risk import RiskEngine, RiskLimits\nx = RiskEngine(RiskLimits(max_position_pct=1.0))\n"
    called = constructed_names(offending)
    assert "RiskLimits" in called and "RiskEngine" in called
    assert "build_risk_engine" not in called


def test_the_cli_and_the_harness_agree_field_for_field(tmp_path):
    """Not "both call the mapping" -- both RESULTS, compared."""
    import tools.demo_run as demo_run

    # The harness's config: the committed campaign on a TEST profile, which is
    # the only difference a test profile makes to the limits (none).
    from_harness = build_risk_engine(
        campaign_config(tmp_path / "harness"), capital=Decimal("1000000")
    ).limits
    # The CLI's config: `conf/demo/pvc1.json` itself, parsed as `_load` parses it,
    # with only the state directory pointed somewhere writable.
    payload = json.loads(PVC1.read_text(encoding="utf-8"))
    payload["runner"] = {"state_dir": str(tmp_path / "cli")}
    from_cli = build_risk_engine(parse_demo_config(payload), capital=Decimal("1000000")).limits

    assert from_harness == from_cli == risk_limits(committed())
    assert demo_run.build_risk_engine is build_risk_engine


def test_unstarted_runtime_is_inspectable_but_cannot_mutate(tmp_path):
    """No fake time and no active decision object before recorded startup."""
    from chimera.demo.runner import RunnerError, RunnerState
    from chimera.demo.telemetry import NullTelemetry

    harness = build(tmp_path, start=False, telemetry=NullTelemetry())
    runner = harness.runner

    assert not runner.clock.started
    assert runner.state is RunnerState.STARTUP
    assert runner.inspection.ledger.state.capital == Decimal("1000000")
    assert not harness.state_dir.exists(), "inspection created persisted state"

    for attribute in ("risk", "position"):
        with pytest.raises(RunnerError, match="before the clock starts"):
            getattr(runner, attribute)

    with pytest.raises(RunnerError, match="before start"):
        runner.tick(harness.first_minute_ms())
    with pytest.raises(RunnerError, match="before start"):
        runner.shutdown("must not persist")

    assert not runner.clock.started, "a rejected mutation seeded the clock"
    assert runner.state is RunnerState.STARTUP
    assert not harness.state_dir.exists(), "a rejected mutation persisted state"


def test_kill_switch_present_before_start_halts_before_any_decision(tmp_path):
    """The delayed active engine still checks the switch at construction."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "KILL_SWITCH").write_text("stop\n", encoding="utf-8")
    harness = build(tmp_path, config=campaign_config(state_dir), start=False)

    harness.runner.start()

    assert harness.runner.risk.state.halted
    assert harness.runner.risk.state.halt_reason == "kill_switch"
    assert not [record for record in harness.records() if record["kind"] == "DECISION"]


@pytest.mark.parametrize(
    ("delta_ns", "halted", "live_orders"),
    [
        (59_999_000_000, True, 2),
        (60_000_000_000, False, 1),
        (60_001_000_000, False, 1),
    ],
    ids=("before-60s", "exactly-60s", "after-60s"),
)
def test_order_rate_uses_recorded_time_on_both_sides_of_60_seconds(
    tmp_path, delta_ns, halted, live_orders
):
    """The existing window is ``now - order < 60``; equality expires."""
    state_dir = tmp_path / f"state-{delta_ns}"
    harness = build(
        tmp_path / f"run-{delta_ns}",
        config=campaign_config(state_dir, limits={"max_orders_per_minute": 1}),
    )
    risk = harness.runner.risk
    start_ns = harness.runner.clock.now_ns

    risk.record_order()
    harness.runner.clock.observe(start_ns + delta_ns)
    risk.record_order()

    assert risk.state.halted is halted
    assert len(risk.state.order_times) == live_orders
    assert risk.state.order_times[-1] == pytest.approx((start_ns + delta_ns) / 1e9)


def test_non_demo_constructors_retain_real_time_defaults():
    """Callers that omit the new injection keep the existing ``time.time``."""
    import time

    from chimera.futures.executor import FuturesExecutionConfig, FuturesExecutor
    from chimera.futures.fills import RecordedQuoteFillModel
    from chimera.futures.store import FuturesStore
    from chimera.futures.venue import DryRunFuturesVenue, StaticConstraintSource

    risk = RiskEngine()
    executor = FuturesExecutor(
        venue=DryRunFuturesVenue(StaticConstraintSource({}), RecordedQuoteFillModel()),
        risk=risk,
        store=FuturesStore.open(None),
        config=FuturesExecutionConfig(),
    )

    assert risk._clock is time.time
    assert executor.clock is time.time


def test_cooldown_uses_recorded_time_before_at_and_after_expiry(tmp_path):
    """The gate is active while ``now < cooldown_until`` and open at equality.

    ``record_trade_result`` has no demo production caller today. This witnesses
    the already-existing Aegis path once the demo clock is wired; it does not
    claim that the runner can currently open a cooldown itself.
    """
    harness = build(
        tmp_path,
        config=campaign_config(
            tmp_path / "state",
            limits={"loss_streak_limit": 1, "cooldown_seconds": 300},
        ),
    )
    risk = harness.runner.risk
    start_ns = harness.runner.clock.now_ns
    risk.record_trade_result(-1.0)
    assert risk.state.cooldown_until == pytest.approx(start_ns / 1e9 + 300)

    harness.runner.clock.observe(start_ns + 299_999_000_000)
    before = risk.evaluate_entry("pair", 1000, 100, 90)
    harness.runner.clock.observe(start_ns + 300_000_000_000)
    boundary = risk.evaluate_entry("pair", 1000, 100, 90)
    harness.runner.clock.observe(start_ns + 300_001_000_000)
    after = risk.evaluate_entry("pair", 1000, 100, 90)

    assert not before.allowed and "cooldown active" in before.reason
    assert boundary.allowed
    assert after.allowed


def test_aegis_and_both_executors_persist_the_same_recorded_instant(tmp_path):
    """A behavioral witness for all three consumers, not callable identity."""
    from datetime import datetime, timezone

    from chimera.futures.executor import FlattenCause

    harness = build(tmp_path)
    instant_ns = 1_900_000_000_123_456_789
    harness.runner.clock.observe(instant_ns)
    expected_s = instant_ns / 1e9
    expected_text = datetime.fromtimestamp(expected_s, tz=timezone.utc).isoformat()

    harness.runner.risk.record_order()
    for name, executor, symbol in (
        ("spot", harness.runner.position.spot, "BTC/USDT"),
        ("perp", harness.runner.position.perp, "BTC/USDT:USDT"),
    ):
        executor.emergency_flatten(symbol, FlattenCause.RISK_HALT, Decimal("1"))
        assert executor.store.state.flatten_reasons[-1] == {
            "symbol": symbol,
            "reason": FlattenCause.RISK_HALT.value,
            "at": expected_text,
        }, name

    assert harness.runner.risk.state.order_times[-1] == pytest.approx(expected_s)


def test_recorded_risk_timing_is_independent_of_host_clock_behavior(tmp_path):
    """The same recorded input survives host clocks that move 0s versus 61s.

    A fresh subprocess patches ``time.time`` *before* importing Aegis, so its
    non-injected default binds the hostile host clock. If demo construction ever
    drops the injection, one run keeps two orders inside the window while the
    other expires the first; with RunnerClock both produce the same halt and
    the same persisted order instants.
    """
    import subprocess
    import sys

    program = r"""
import json
import sys
import time
from pathlib import Path

host = [100.0]
time.time = lambda: host[0]

from chimera.demo.telemetry import NullTelemetry
from tests.demo_harness import build, campaign_config

root = Path(sys.argv[1])
harness = build(
    root,
    config=campaign_config(root / "state", limits={"max_orders_per_minute": 1}),
    telemetry=NullTelemetry(),
)
risk = harness.runner.risk
recorded = harness.runner.clock.now_ns
risk.record_order()
host[0] += float(sys.argv[2])
harness.runner.clock.observe(recorded + 30_000_000_000)
risk.record_order()
print(json.dumps({"halted": risk.state.halted, "times": risk.state.order_times}))
"""

    def run(where: Path, host_delta: int) -> dict:
        completed = subprocess.run(
            [sys.executable, "-c", program, str(where), str(host_delta)],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    frozen_host = run(tmp_path / "frozen", 0)
    jumping_host = run(tmp_path / "jumping", 61)
    assert frozen_host == jumping_host
    assert frozen_host["halted"] is True
    assert frozen_host["times"][1] - frozen_host["times"][0] == pytest.approx(30.0)
