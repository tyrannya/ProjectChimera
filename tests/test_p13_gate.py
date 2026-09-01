"""G1-G6, the stresses that gate and the ones that must not, and four result states.

Two-sided by construction: for every condition there is a world in which the gate
MUST pass and one in which it MUST fail, so the decision function is never tested
only against the outcome it happens to produce.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from nn.p13_carry import NOT_DETERMINABLE, BlockResult, LiquidationTouchProvenance
from nn.p13_gate import (
    INVALID,
    NOT_EVALUABLE,
    NOT_VIABLE,
    NOT_YET_RUN,
    VIABLE,
    GateError,
    StressInputs,
    evaluate_gate,
)
from nn.p13_preregistration import (
    BREADTH_OF,
    BREADTH_REQUIRED,
    CURRENT_RESULT_STATE,
    MIN_INCLUDED_BLOCKS,
    MIN_MEAN_NET_RETURN,
    MIN_SETTLEMENTS_PER_BLOCK,
    RESULT_STATES,
    VIABILITY_GATE,
    WORST_BLOCK_FLOOR,
)
from nn.p13_stress import (
    D1_FUNDING_FACTOR,
    S1_FRICTION_FACTOR,
    S3_BASIS_SHIFT,
    higher_friction_costs,
    leave_one_out_means,
)
from nn.p13_screen import frozen_costs

ZERO = Decimal("0")


def result(
    label: str,
    net_return: str,
    *,
    opened: bool = True,
    settlements: int = 500,
    unclosed: bool = False,
) -> BlockResult:
    """A block result carrying only what the gate is entitled to read."""
    value = Decimal(net_return)
    return BlockResult(
        label=label,
        opened=opened,
        reason="synthetic",
        settlements=settlements,
        quantity=Decimal("10"),
        basis_entry=Decimal("30"),
        basis_exit=Decimal("30"),
        basis_pnl=ZERO,
        funding_received=ZERO,
        funding_paid=ZERO,
        fees=ZERO,
        slippage=ZERO,
        rebalance_cost=ZERO,
        net_pnl=value * Decimal("1000000"),
        net_return=NOT_DETERMINABLE if (unclosed or not opened) else value,
        liquidated=False,
        max_adverse_excursion_pnl=ZERO,
        max_adverse_excursion=ZERO,
        thin_sample=settlements < MIN_SETTLEMENTS_PER_BLOCK,
        unclosed=unclosed,
        liquidation_touch_provenance=LiquidationTouchProvenance(mark_high=1),
    )


def six(*returns: str) -> list[BlockResult]:
    return [result(f"b{index}", value) for index, value in enumerate(returns)]


PASSING = ("0.01", "0.01", "0.01", "0.01", "-0.001", "-0.001")


def stresses(s1=None, s3=None) -> StressInputs:
    return StressInputs(s1=list(s1 or six(*PASSING)), s3=list(s3 or six(*PASSING)))


# ---------------------------------------------------------------------------
# The two-sided control
# ---------------------------------------------------------------------------


def test_a_world_built_to_pass_returns_viable():
    verdict = evaluate_gate(six(*PASSING), stresses())
    assert verdict.result_state == VIABLE
    assert verdict.passed
    assert all(condition.passed for condition in verdict.conditions)


def test_a_world_built_to_fail_returns_not_viable():
    verdict = evaluate_gate(
        six("-0.01", "-0.01", "-0.01", "-0.01", "0.001", "0.001"), stresses()
    )
    assert verdict.result_state == NOT_VIABLE
    assert not verdict.passed


# ---------------------------------------------------------------------------
# The conditions, one at a time
# ---------------------------------------------------------------------------


def test_g1_needs_the_frozen_breadth_and_a_zero_block_does_not_count():
    """tie_handling: a block of exactly zero is NOT positive."""
    assert (BREADTH_REQUIRED, BREADTH_OF) == (4, 6)
    three_positive = evaluate_gate(
        six("0.01", "0.01", "0.01", "0", "0.01", "0.01"), stresses()
    )
    assert three_positive.positive_blocks == 5
    zeroed = evaluate_gate(six("0.01", "0.01", "0.01", "0", "0", "0"), stresses())
    assert zeroed.positive_blocks == 3
    assert not _condition(zeroed, "G1").passed


def test_g2_fails_on_a_mean_of_exactly_zero():
    verdict = evaluate_gate(six("0.01", "0.01", "0.01", "0.01", "-0.02", "-0.02"), stresses())
    assert verdict.mean_net_return == 0
    assert not _condition(verdict, "G2").passed


def test_g3_is_inclusive_at_the_floor_and_fails_below_it():
    floor = Decimal(WORST_BLOCK_FLOOR)
    assert floor == Decimal("-0.02")
    at_floor = evaluate_gate(six("0.05", "0.05", "0.05", "0.05", "-0.02", "0.01"), stresses())
    assert _condition(at_floor, "G3").passed
    below = evaluate_gate(six("0.05", "0.05", "0.05", "0.05", "-0.021", "0.01"), stresses())
    assert not _condition(below, "G3").passed


def test_g4_is_inclusive_at_the_settlement_minimum():
    assert MIN_SETTLEMENTS_PER_BLOCK == 200
    exact = [result(f"b{i}", "0.01", settlements=200) for i in range(6)]
    assert _condition(evaluate_gate(exact, stresses()), "G4").passed
    thin = [result(f"b{i}", "0.01", settlements=199) for i in range(6)]
    assert not _condition(evaluate_gate(thin, stresses()), "G4").passed


def test_g6_fails_on_a_mean_of_exactly_the_frozen_floor():
    """tie_handling: the mean must EXCEED 0.0025, not merely reach it."""
    floor = Decimal(MIN_MEAN_NET_RETURN)
    exact = [result(f"b{i}", str(floor)) for i in range(6)]
    verdict = evaluate_gate(exact, stresses())
    assert verdict.mean_net_return == floor
    assert not _condition(verdict, "G6").passed
    above = [result(f"b{i}", "0.0026") for i in range(6)]
    assert _condition(evaluate_gate(above, stresses()), "G6").passed


def test_g5_gates_on_s1_and_s3_and_on_neither_diagnostic():
    failing = six("-0.01", "-0.01", "-0.01", "-0.01", "-0.01", "-0.01")
    assert not _condition(evaluate_gate(six(*PASSING), stresses(s1=failing)), "G5").passed
    assert not _condition(evaluate_gate(six(*PASSING), stresses(s3=failing)), "G5").passed
    # And there is nowhere to put S2 or S4: the gate's input type has two fields.
    assert set(StressInputs.__dataclass_fields__) == {"s1", "s3"}


def test_g5_requires_g3_of_s1_but_not_of_s3():
    """Read literally from the frozen sentence, in both directions.

    The stressed set below passes G1 (five positive) and G2 (mean +0.083) and
    fails G3 alone, so the two assertions differ ONLY on whether G3 is demanded of
    that stress. A set that also failed G2 would prove nothing about G3.
    """
    deep = six("0.2", "0.2", "0.2", "0.2", "-0.5", "0.2")
    breadth, mean, worst = _shape(deep)
    assert breadth == 5 and mean > 0 and worst < Decimal(WORST_BLOCK_FLOOR)

    assert not _condition(evaluate_gate(six(*PASSING), stresses(s1=deep)), "G5").passed
    assert _condition(evaluate_gate(six(*PASSING), stresses(s3=deep)), "G5").passed


def _shape(blocks):
    returns = [block.net_return for block in blocks]
    return (
        sum(1 for value in returns if value > 0),
        sum(returns, Decimal("0")) / Decimal(len(returns)),
        min(returns),
    )


def _condition(verdict, name):
    return next(condition for condition in verdict.conditions if condition.name == name)


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


def test_an_unclosed_block_makes_the_screen_invalid_under_a1():
    blocks = six(*PASSING)
    blocks[2] = result("b2", "0.01", unclosed=True)
    verdict = evaluate_gate(blocks, stresses())
    assert verdict.result_state == INVALID


def test_too_few_included_blocks_is_invalid_rather_than_a_verdict():
    assert MIN_INCLUDED_BLOCKS == 5
    blocks = six(*PASSING)
    for index in (0, 1):
        blocks[index] = result(f"b{index}", "0", opened=False)
    verdict = evaluate_gate(blocks, stresses())
    assert verdict.result_state == INVALID
    assert len(verdict.included_blocks) == 4


def test_an_excluded_block_is_removed_from_the_denominators_not_counted_as_zero():
    blocks = six("0.01", "0.01", "0.01", "0.01", "0.01", "0")
    blocks[5] = result("b5", "0", opened=False)
    verdict = evaluate_gate(blocks, stresses())
    assert verdict.excluded_blocks == ("b5",)
    assert len(verdict.included_blocks) == 5
    assert verdict.mean_net_return == Decimal("0.01")


def test_a_terminal_source_insufficiency_prevents_the_gate_from_running_at_all():
    """Witness 14. NOT EVALUABLE bypasses gate computation entirely."""
    with pytest.raises(GateError, match="terminated NOT EVALUABLE"):
        evaluate_gate(six(*PASSING), stresses(), terminal=object())


def test_not_evaluable_is_not_not_viable():
    """Witness: source insufficiency must never become an economic negative."""
    assert NOT_EVALUABLE != NOT_VIABLE
    assert NOT_EVALUABLE in RESULT_STATES and NOT_VIABLE in RESULT_STATES
    assert "NOT EVALUABLE" in NOT_EVALUABLE and "NOT VIABLE" in NOT_VIABLE
    assert NOT_YET_RUN == CURRENT_RESULT_STATE
    assert NOT_YET_RUN not in RESULT_STATES


def test_a_gate_that_forgot_to_exclude_a_block_would_raise_rather_than_average():
    """NaN is load-bearing: an unmeasured block cannot be averaged silently."""
    from decimal import InvalidOperation

    unopened = result("b0", "0", opened=False)
    assert unopened.net_return.is_nan()
    with pytest.raises(InvalidOperation):
        _ = unopened.net_return > 0


# ---------------------------------------------------------------------------
# The frozen constants, unchanged by any of this
# ---------------------------------------------------------------------------


def test_the_gate_constants_are_the_frozen_ones():
    """Witness 27."""
    assert BREADTH_REQUIRED == 4 and BREADTH_OF == 6
    assert MIN_INCLUDED_BLOCKS == 5
    assert MIN_SETTLEMENTS_PER_BLOCK == 200
    assert WORST_BLOCK_FLOOR == "-0.02"
    assert MIN_MEAN_NET_RETURN == "0.0025"
    assert VIABILITY_GATE["conjunction"].startswith("ALL of G1, G2, G3, G4, G5 and G6")


def test_the_stress_magnitudes_are_the_frozen_ones():
    """Witness 28."""
    assert S1_FRICTION_FACTOR == Decimal("2")
    assert S3_BASIS_SHIFT == Decimal("0.0010")
    assert D1_FUNDING_FACTOR == Decimal("0.5")
    doubled = higher_friction_costs(frozen_costs())
    base = frozen_costs()
    assert doubled.spot_fee == base.spot_fee * 2
    assert doubled.perp_slippage == base.perp_slippage * 2


def test_leave_one_out_reports_one_mean_per_included_block():
    """D2, and it omits blocks that were never measured rather than zeroing them."""
    blocks = six("0.01", "0.02", "0.03", "0.04", "0.05", "0.06")
    means = leave_one_out_means(blocks)
    assert len(means) == 6
    assert dict(means)["b0"] == (
        Decimal("0.02") + Decimal("0.03") + Decimal("0.04") + Decimal("0.05") + Decimal("0.06")
    ) / Decimal(5)
    with_excluded = list(blocks)
    with_excluded[0] = result("b0", "0", opened=False)
    assert len(leave_one_out_means(with_excluded)) == 5
