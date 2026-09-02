"""Behavioural witnesses for amendment A2R2's source-validity semantics.

Every test here is a synthetic world in which the rule under test either fires or
does not, and the assertion is about what the runtime DID — an instant it opened
at, a bar it held, a screen it terminated — rather than about a sentence it
contains. String pinning belongs in ``tests/test_p13_preregistration.py``, which
pins the design; this file pins the behaviour that design demands.

No test in this file touches a Binance archive, a network, or any market
observation. P13's sources have never been obtained.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from nn.p13_alignment import MARK, PERPETUAL, SPOT, AlignedSources, AlignmentError
from nn.p13_blocks import (
    NOT_EVALUABLE,
    BlockError,
    NoValidExit,
    SourceInsufficiency,
    build_quotes,
    calendar_blocks,
    find_opening_instant,
    held_and_exit_instants,
    run_block,
    run_screen,
)
from nn.p13_carry import (
    TOUCH_MARK_CLOSE,
    TOUCH_SPOT_CLOSE,
    CarryError,
    FundingSettlement,
    Quote,
)
from nn.p13_preregistration import (
    MARKLESS_STATE_HELD,
    MARKLESS_STATE_NO_VALID_OPEN,
    MARKLESS_STATE_PRE_OPEN,
    OPENING_DELAY_EXECUTION_ABSENT,
    RESULT_STATES,
    TEMPORAL_PARTITION,
)
from nn.p13_screen import frozen_allocation, frozen_costs, frozen_venue
from tests.p13_synthetic import HOUR, block, funding_row, ns, world

ALLOCATION = frozen_allocation()
COSTS = frozen_costs()
VENUE = frozen_venue()


def _run(aligned: AlignedSources, blk, *, settlements=(), min_settlements: int = 0):
    return run_block(
        aligned,
        blk,
        allocation=ALLOCATION,
        costs=COSTS,
        venue=VENUE,
        min_settlements=min_settlements,
        settlements=settlements,
    )


def _screen(aligned: AlignedSources, blocks, *, settlements=(), min_settlements: int = 0):
    return run_screen(
        aligned,
        allocation=ALLOCATION,
        costs=COSTS,
        venue=VENUE,
        min_settlements=min_settlements,
        settlements=settlements,
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# 1-2. The liquidation ladder: mark HIGH, else mark CLOSE, and nothing else
# ---------------------------------------------------------------------------


def test_the_mark_high_is_used_whenever_the_bar_carries_one():
    """Witness 1. Adverse for a SHORT means HIGH, so the strong touch is preferred."""
    aligned = world(hours=6, mark_high={index: Decimal("31000") for index in range(6)})
    run = _run(aligned, block(hours=6))
    provenance = run.result.liquidation_touch_provenance
    assert provenance.mark_high == run.result.held_bars
    assert provenance.mark_close == 0
    assert provenance.spot_close == 0
    assert provenance.all_mark_high


def test_the_mark_close_answers_when_no_high_is_available():
    """Witness 2, at the level the tier is reachable from.

    Binance's published kline layout always carries a high, so a mark ROW always
    yields one and the close tier cannot be reached through the loader. The tier
    exists for a source that supplies a close without a high, and this asserts the
    frozen ladder on exactly that shape.
    """
    quote = Quote(
        instant_ns=ns("2021-03-01T00:00:00+00:00"),
        spot=Decimal("30000"),
        perp=Decimal("30030"),
        spot_open=Decimal("30000"),
        perp_open=Decimal("30030"),
        mark=Decimal("30010"),
        mark_high=None,
    )
    assert quote.liquidation_touch_source == TOUCH_MARK_CLOSE
    assert quote.liquidation_touch == Decimal("30010")
    assert not quote.liquidation_touch_is_high


def test_a_bar_with_neither_mark_series_refuses_rather_than_using_spot():
    """Witness: there is no third tier, and TOUCH_SPOT_CLOSE stays unreachable."""
    quote = Quote(
        instant_ns=ns("2021-03-01T00:00:00+00:00"),
        spot=Decimal("30000"),
        perp=Decimal("30030"),
        spot_open=Decimal("30000"),
        perp_open=Decimal("30030"),
        mark=None,
        mark_high=None,
    )
    with pytest.raises(CarryError, match="no mark series"):
        _ = quote.liquidation_touch
    with pytest.raises(CarryError, match="no mark series"):
        _ = quote.liquidation_touch_source
    # The name survives as vocabulary and is produced by nothing.
    assert TOUCH_SPOT_CLOSE == "spot_close"


# ---------------------------------------------------------------------------
# 3-4, 16, 19. Held bars terminate; the exit bar is exempt
# ---------------------------------------------------------------------------


def test_a_held_bar_with_no_mark_terminates_the_whole_screen():
    """Witness 3. Not skipped, not jumped, not excluded — terminal and screen-wide."""
    aligned = world(hours=8, missing_mark=[3])
    with pytest.raises(SourceInsufficiency) as raised:
        _run(aligned, block(hours=8))
    failure = raised.value
    assert failure.state == MARKLESS_STATE_HELD
    assert failure.result_state == NOT_EVALUABLE
    assert failure.result_state in RESULT_STATES
    assert failure.instant_ns == ns("2021-03-01T00:00:00+00:00") + 3 * HOUR
    assert failure.missing == (MARK,)
    assert failure.as_dict()["is_economic_failure"] is False


def test_a_missing_mark_on_the_exit_bar_alone_does_not_stop_a_normal_close():
    """Witnesses 4, 16 and 19. Bar N is post-exit, so it needs no liquidation mark."""
    hours = 8
    aligned = world(hours=hours, missing_mark=[hours - 1])
    run = _run(aligned, block(hours=hours))
    assert run.result.opened
    assert not run.result.unclosed
    assert run.result.reason == "closed at block end"
    # Every HELD bar was tested; the exit bar was not, because it was not held.
    assert run.result.held_bars == hours - 1
    assert run.result.liquidation_touch_provenance.tested == hours - 1


def test_bar_zero_is_held_and_its_missing_mark_terminates_rather_than_delaying():
    """**The critical A2R2 witness.** Bar 0 is held; its mark is checked THERE.

    Execution rows exist at bar 0 and the mark row does not. Under A2 and A2R1 the
    opening search advanced past bar 0 — a decision that required knowing whether
    a completed mark row for bar 0 would ever be published, which is not knowable
    at bar 0. Under A2R2 the runtime opens at bar 0 anyway, and then terminates
    because bar 0 is a HELD bar whose liquidation quantity is not computable.

    Both halves matter: opening THERE, and terminating for THAT reason.
    """
    aligned = world(hours=8, missing_mark=[0])
    start = ns("2021-03-01T00:00:00+00:00")

    opening = find_opening_instant(aligned, block(hours=8))
    assert opening.opened_at_ns == start, "the mark moved the entry instant"
    assert not opening.delayed
    assert opening.reason is None

    with pytest.raises(SourceInsufficiency) as raised:
        _run(aligned, block(hours=8))
    assert raised.value.state == MARKLESS_STATE_HELD
    assert raised.value.instant_ns == start
    assert raised.value.result_state == NOT_EVALUABLE
    assert MARK in raised.value.missing

    # And with the mark present at bar 0 the same world opens at the same instant
    # and does NOT terminate — so the mark changed the OUTCOME without ever having
    # changed the ENTRY.
    intact = _run(world(hours=8), block(hours=8))
    assert intact.opening.opened_at_ns == start
    assert intact.result.opened


def test_the_held_window_is_bars_zero_to_n_minus_one():
    """Witness 19, stated as arithmetic rather than inferred from a docstring."""
    hours = 10
    held, exit_ns = held_and_exit_instants(block(hours=hours), ns("2021-03-01T00:00:00+00:00"))
    assert len(held) == hours - 1
    assert exit_ns == ns("2021-03-01T00:00:00+00:00") + (hours - 1) * HOUR
    assert exit_ns not in held
    quotes = build_quotes(world(hours=hours), block(hours=hours), held[0])
    assert len(quotes) == hours
    assert quotes[-1].instant_ns == exit_ns


# ---------------------------------------------------------------------------
# 5-7. The pre-open search, under A2R2
# ---------------------------------------------------------------------------


def test_the_mark_can_never_move_the_chosen_entry_instant():
    """**A2R2 witness 2**, as a comparison rather than a single assertion.

    Four worlds whose execution rows are identical and whose mark coverage differs
    in every way a test can arrange: complete, absent at the boundary, absent for
    a run of hours, absent entirely. The opening instant must be the same in all
    four, because the mark is no longer an input to that decision.
    """
    blk = block(hours=10)
    coverage = (
        (),
        (0,),
        (0, 1, 2),
        range(10),
    )
    chosen = {
        find_opening_instant(world(hours=10, missing_mark=missing), blk).opened_at_ns
        for missing in coverage
    }
    assert chosen == {blk.start_ns}, (
        "mark coverage moved the opening instant, so the retired pre-open mark "
        "filter is back"
    )


def test_only_an_absent_execution_row_delays_an_opening():
    """**A2R2 witness**: the one surviving delay reason, and it is named as such."""
    aligned = world(hours=8, missing_spot=[0])
    run = _run(aligned, block(hours=8))
    assert run.opening.delayed
    assert run.opening.opened_at_ns == run.block.start_ns + HOUR
    assert run.opening.skipped_instants == 1
    assert run.opening.reason == OPENING_DELAY_EXECUTION_ABSENT
    assert run.opening.skipped_ns == HOUR
    # A markless state may never be reported as a delay reason again.
    assert run.opening.reason not in (
        MARKLESS_STATE_PRE_OPEN,
        MARKLESS_STATE_NO_VALID_OPEN,
    )


def test_the_evidence_states_that_the_opening_consulted_no_mark():
    """Reported as a flag, not left to be inferred from the absence of a count."""
    run = _run(world(hours=8, missing_spot=[0]), block(hours=8))
    emitted = run.opening.as_dict()
    assert emitted["opening_consulted_mark"] is False
    assert emitted["reason"] == OPENING_DELAY_EXECUTION_ABSENT
    assert (
        "skipped_for_mark" not in emitted
    ), "a mark-shaped skip count implies the opening search still consults the mark"


def test_several_invalid_pre_open_instants_select_the_first_valid_one():
    """Witness 6. Forward, causal, and it stops at the FIRST fillable instant."""
    aligned = world(hours=10, missing_spot=[0, 1, 2], missing_perp=[3])
    run = _run(aligned, block(hours=10))
    assert run.opening.opened_at_ns == run.block.start_ns + 4 * HOUR
    assert run.opening.skipped_instants == 4
    assert run.opening.reason == OPENING_DELAY_EXECUTION_ABSENT


def test_a_block_whose_mark_never_appears_terminates_through_the_held_bar_branch():
    """Witness 7 under A2R2, and mutation guard: still NOT an excluded block.

    A2R1 refused to open this block at all and terminated with
    ``no_valid_opening_instant_in_block``. A2R2 retires that state: the block opens
    at its first fillable instant, that instant is bar 0 and therefore held, and
    the screen terminates with the SAME label through the branch that is causal.
    """
    aligned = world(hours=8, missing_mark=range(8))
    with pytest.raises(SourceInsufficiency) as raised:
        _run(aligned, block(hours=8))
    failure = raised.value
    assert failure.state == MARKLESS_STATE_HELD
    assert failure.state != MARKLESS_STATE_NO_VALID_OPEN
    assert failure.instant_ns == ns("2021-03-01T00:00:00+00:00")
    assert failure.result_state == NOT_EVALUABLE
    screen = _screen(aligned, [block(hours=8)])
    assert not screen.evaluable
    assert screen.blocks == ()
    assert screen.result_state == NOT_EVALUABLE


def test_a_block_with_no_rows_at_all_stays_the_excluded_block_case():
    """The frozen excluded-block rule is not broadened, and not narrowed either.

    Nothing has been established about the mark when no instant is even fillable,
    so A2R1's terminal branch is deliberately not reached.
    """
    aligned = AlignedSources.build(
        spot=(), perpetual=(), mark=(), funding=(), published_mark_periods=()
    )
    with pytest.raises(BlockError) as raised:
        _run(aligned, block(hours=8))
    assert not isinstance(raised.value, SourceInsufficiency)
    screen = _screen(aligned, [block(hours=8)])
    assert screen.evaluable
    assert len(screen.blocks) == 1
    assert screen.blocks[0].result.opened is False


# ---------------------------------------------------------------------------
# 8-10. What the skipped pre-open interval does, and does not, cost
# ---------------------------------------------------------------------------


def test_nothing_is_attributed_to_the_strategy_before_the_valid_open():
    """Witness 8. No exposure, funding, basis, fee or slippage in the skipped hours."""
    settlement_hour = 1
    settlements = (
        FundingSettlement(
            instant_ns=ns("2021-03-01T00:00:00+00:00") + settlement_hour * HOUR,
            rate=Decimal("0.0005"),
            mark_price=Decimal("30010"),
        ),
    )
    delayed = _run(
        world(hours=10, missing_spot=[0, 1]), block(hours=10), settlements=settlements
    )
    assert delayed.opening.opened_at_ns == delayed.block.start_ns + 2 * HOUR

    # The settlement fell inside the skipped interval, so it is not this
    # position's cash flow at all.
    assert delayed.result.settlements == 0
    assert delayed.result.funding_received == 0
    assert delayed.result.funding_paid == 0
    # Exposure begins at the open: two fewer held bars than an undelayed block.
    prompt = _run(world(hours=10), block(hours=10), settlements=settlements)
    assert prompt.result.held_bars - delayed.result.held_bars == 2
    assert prompt.result.settlements == 1
    # Exactly one round trip either way — a delay adds no fill and no friction.
    assert delayed.result.fees == prompt.result.fees
    assert delayed.result.slippage == prompt.result.slippage


def _return_with_skipped_funding(rate: str) -> tuple[Decimal, Decimal]:
    """Net returns with and without a delayed open, over one skipped settlement."""
    settlements = (
        FundingSettlement(
            instant_ns=ns("2021-03-01T00:00:00+00:00") + HOUR,
            rate=Decimal(rate),
            mark_price=Decimal("30010"),
        ),
    )
    prompt = _run(world(hours=10), block(hours=10), settlements=settlements)
    delayed = _run(
        world(hours=10, missing_spot=[0, 1]), block(hours=10), settlements=settlements
    )
    assert delayed.opening.delayed and not prompt.opening.delayed
    assert prompt.result.settlements == 1 and delayed.result.settlements == 0
    return prompt.result.net_return, delayed.result.net_return


def test_skipping_positive_funding_worsens_the_block():
    """Witness 9. The short RECEIVES on a positive rate, so skipping it costs."""
    prompt, delayed = _return_with_skipped_funding("0.0005")
    assert delayed < prompt


def test_skipping_negative_funding_improves_the_block():
    """Witness 10, and the proof no monotonicity assumption leaked into the runtime.

    The first committed A2 claimed a delayed open "can only reduce accrued
    funding". A2R1 withdrew that as false and A2R2 does not reinstate it. This is
    the synthetic counter-example: the short PAYS on a negative rate, so an hour
    skipped before the open is an hour of payment avoided, and the delayed block
    finishes AHEAD. Under A2R2 the delay is caused by an absent EXECUTION row
    rather than an absent mark, and the arithmetic is identical — which is exactly
    why no monotonicity is claimed for a delayed open of any cause.
    """
    prompt, delayed = _return_with_skipped_funding("-0.0005")
    assert delayed > prompt


def test_the_runtime_states_the_pre_open_direction_as_indeterminate():
    """The evidence must not imply a direction the design explicitly disclaims."""
    run = _run(world(hours=8, missing_spot=[0]), block(hours=8))
    assert run.opening.as_dict()["economic_direction"] == "INDETERMINATE EX ANTE"


# ---------------------------------------------------------------------------
# The look-ahead guard
# ---------------------------------------------------------------------------


def test_the_opening_instant_does_not_depend_on_what_the_mark_says():
    """The entry decision reads PRESENCE, never a price.

    Two worlds identical except for the numeric mark high on every bar. If any
    value ever reached the admissibility decision, the chosen opening instant
    could differ; it must not.
    """
    quiet = world(hours=12, missing_spot=[0, 1])
    violent = world(
        hours=12,
        missing_spot=[0, 1],
        mark_high={index: Decimal("999999") for index in range(12)},
    )
    assert (
        find_opening_instant(quiet, block(hours=12)).opened_at_ns
        == find_opening_instant(violent, block(hours=12)).opened_at_ns
    )


def test_alignment_never_invents_a_row_for_a_missing_instant():
    """Mutation guard: silently forward-filling a missing mark must be impossible."""
    aligned = world(hours=6, missing_mark=[2])
    hole = ns("2021-03-01T00:00:00+00:00") + 2 * HOUR
    assert hole not in aligned.mark
    validity = aligned.instant_validity(hole)
    assert validity.missing == (MARK,)
    assert not validity.valid_for_holding
    with pytest.raises(AlignmentError, match="no mark row"):
        aligned.quote(hole, require_mark=True)


# ---------------------------------------------------------------------------
# 11-12. The funding fallback is separate, in both directions
# ---------------------------------------------------------------------------


def test_the_spot_close_prices_funding_when_the_mark_object_is_unpublished():
    """Witness 11. MARK_PRICE_FALLBACK, per archive object, funding notional only."""
    aligned = world(hours=6, funding=(funding_row(2, "0.0001"),), published_mark_periods=())
    settlements, bases = aligned.settlements(aligned.funding)
    assert len(settlements) == 1
    assert bases[0].source == SPOT
    assert settlements[0].mark_price == Decimal("30000")


def test_the_mark_close_prices_funding_when_the_object_is_published():
    aligned = world(hours=6, funding=(funding_row(2, "0.0001"),))
    settlements, bases = aligned.settlements(aligned.funding)
    assert bases[0].source == MARK
    assert settlements[0].mark_price == Decimal("30010")


def test_the_funding_fallback_does_not_rescue_a_markless_held_bar():
    """Witness 12. The two mechanisms are independent, and this is the direction
    that would flatter a result if they were not."""
    aligned = world(hours=8, missing_mark=[3], published_mark_periods=())
    # The funding fallback is available...
    settlements, bases = aligned.settlements((funding_row(2, "0.0001"),))
    assert bases[0].source == SPOT
    # ...and it does nothing whatsoever for the liquidation test.
    with pytest.raises(SourceInsufficiency) as raised:
        _run(aligned, block(hours=8))
    assert raised.value.state == MARKLESS_STATE_HELD


def test_a_markless_bar_does_not_disable_an_authorised_funding_fallback():
    """The other direction of the same independence."""
    aligned = world(hours=8, missing_mark=[3], published_mark_periods=())
    settlements, bases = aligned.settlements((funding_row(5, "0.0001"),))
    assert bases[0].source == SPOT
    assert settlements[0].mark_price == Decimal("30000")


# ---------------------------------------------------------------------------
# 13-15. Terminal propagation
# ---------------------------------------------------------------------------


def test_a_terminal_failure_discards_every_block_already_computed():
    """Witnesses 13 and 15, and the mutation guard against five-block continuation."""
    blocks = [
        block(start="2021-03-01T00:00:00+00:00", hours=6, label="first"),
        block(start="2021-03-02T00:00:00+00:00", hours=6, label="second"),
    ]
    aligned = AlignedSources.build(
        spot=world(hours=30).spot.values(),
        perpetual=world(hours=30).perpetual.values(),
        mark=[
            row
            for row in world(hours=30).mark.values()
            if row.instant_ns != ns("2021-03-02T02:00:00+00:00")
        ],
        funding=(),
        published_mark_periods={"2021-03"},
    )
    screen = _screen(aligned, blocks)
    assert not screen.evaluable
    assert screen.blocks == ()
    assert screen.terminal is not None
    assert screen.terminal.block_label == "second"
    assert screen.result_state == NOT_EVALUABLE


def test_the_first_block_really_would_have_evaluated_on_its_own():
    """The discarding above is a rule, not an accident of the first block failing."""
    solo = _screen(world(hours=30), [block(hours=6, label="first")])
    assert solo.evaluable
    assert solo.blocks[0].result.opened


# ---------------------------------------------------------------------------
# 17. The exit bar still needs both execution opens
# ---------------------------------------------------------------------------


def test_an_exit_bar_missing_a_leg_becomes_the_a1_unclosed_case():
    """Witness 17. The exemption is from the MARK and from nothing else."""
    hours = 8
    aligned = world(hours=hours, missing_spot=[hours - 1])
    run = _run(aligned, block(hours=hours))
    assert run.result.opened
    assert run.result.unclosed
    assert "UNCLOSED" in run.result.reason
    # A1: the incurred facts are reported, the close-dependent ones are not.
    assert run.result.fees > 0
    assert run.result.net_return.is_nan()
    assert run.result.basis_exit.is_nan()


def test_the_no_valid_exit_case_is_not_a_source_insufficiency():
    """A1's INVALID and A2R1's NOT EVALUABLE are different outcomes."""
    hours = 8
    aligned = world(hours=hours, missing_perp=[hours - 1])
    with pytest.raises(NoValidExit) as raised:
        build_quotes(aligned, block(hours=hours), ns("2021-03-01T00:00:00+00:00"))
    failure = raised.value
    assert not isinstance(failure, SourceInsufficiency)
    # It carries the bars that WERE held, which is what A1 needs to report the
    # funding, fees and slippage actually incurred.
    assert len(failure.held) == hours - 1
    assert failure.instant_ns == ns("2021-03-01T00:00:00+00:00") + (hours - 1) * HOUR


# ---------------------------------------------------------------------------
# 20-22. Chronology and settlement handling, preserved from the A1 repair
# ---------------------------------------------------------------------------


def test_duplicate_and_inverted_quote_instants_are_refused():
    """Witness 20. Causality comes from the instants, never from list order."""
    aligned = world(hours=6)
    quotes = build_quotes(aligned, block(hours=6), ns("2021-03-01T00:00:00+00:00"))
    from nn.p13_carry import evaluate_block

    for broken in ((quotes[0], quotes[0], *quotes[1:]), (quotes[1], quotes[0], *quotes[2:])):
        with pytest.raises(CarryError, match="strictly increasing"):
            evaluate_block("broken", broken, (), ALLOCATION, COSTS, VENUE, 0)


def test_contradictory_settlements_at_one_instant_are_refused():
    """Witness 21."""
    instant = ns("2021-03-01T00:00:00+00:00") + 2 * HOUR
    settlements = (
        FundingSettlement(
            instant_ns=instant, rate=Decimal("0.0001"), mark_price=Decimal("30010")
        ),
        FundingSettlement(
            instant_ns=instant, rate=Decimal("0.0009"), mark_price=Decimal("30010")
        ),
    )
    with pytest.raises(CarryError, match="disagree"):
        _run(world(hours=8), block(hours=8), settlements=settlements)


def test_an_identical_redelivered_settlement_changes_nothing():
    """Witness 22. A row delivered twice is not an ambiguity."""
    instant = ns("2021-03-01T00:00:00+00:00") + 2 * HOUR
    once = (
        FundingSettlement(
            instant_ns=instant, rate=Decimal("0.0001"), mark_price=Decimal("30010")
        ),
    )
    twice = once * 2
    single = _run(world(hours=8), block(hours=8), settlements=once)
    doubled = _run(world(hours=8), block(hours=8), settlements=twice)
    assert single.result.settlements == doubled.result.settlements == 1
    assert single.result.net_return == doubled.result.net_return


# ---------------------------------------------------------------------------
# 25-26. The research boundary
# ---------------------------------------------------------------------------


def test_the_frozen_partition_is_six_blocks_ending_at_the_boundary():
    """Witnesses 25 and 27, at the partition level."""
    blocks = calendar_blocks()
    assert len(blocks) == TEMPORAL_PARTITION["inferential_units"] == 6
    assert [b.label for b in blocks] == list(TEMPORAL_PARTITION["blocks"])
    assert blocks[-1].end_exclusive_ns == ns("2025-05-19T08:00:00+00:00")


def test_no_quote_at_or_after_the_boundary_is_ever_built():
    """Witness 26. The evaluator asserts it too, but nothing should reach it."""
    boundary = ns("2025-05-19T08:00:00+00:00")
    final = calendar_blocks()[-1]
    held, exit_ns = held_and_exit_instants(final, boundary - 6 * HOUR)
    assert all(instant < boundary for instant in held)
    assert exit_ns < boundary
    assert exit_ns == boundary - HOUR

    # And a source that DOES span the boundary cannot get a quote past it: the
    # final block's grid stops one bar short, so no held or exit instant exists
    # at or after the boundary for a quote to be built at.
    aligned = world(start="2025-05-19T04:00:00+00:00", hours=8)
    assert max(aligned.spot) >= boundary
    quotes = build_quotes(aligned, final, boundary - 4 * HOUR)
    assert all(quote.instant_ns < boundary for quote in quotes)
    assert quotes[-1].instant_ns == exit_ns


# ---------------------------------------------------------------------------
# Mutation guards, stated as the behaviour a regression would break
# ---------------------------------------------------------------------------


def test_mutation_restoring_a_spot_liquidation_fallback_would_break_this():
    """If ``Quote`` ever fell back to the spot close, this stops failing."""
    aligned = world(hours=8, missing_mark=[4])
    with pytest.raises(SourceInsufficiency):
        _run(aligned, block(hours=8))


def test_mutation_skipping_bar_zero_would_break_this():
    """If the held loop returned to ``quotes[1:]``, the tested count would drop."""
    hours = 7
    run = _run(world(hours=hours), block(hours=hours))
    assert run.result.held_bars == hours - 1
    assert run.result.liquidation_touch_provenance.tested == hours - 1


def test_mutation_treating_a_terminal_block_as_excluded_would_break_this():
    aligned = world(hours=8, missing_mark=range(8))
    screen = _screen(aligned, [block(hours=8)])
    assert screen.blocks == ()
    assert screen.terminal is not None
    assert screen.terminal.state == MARKLESS_STATE_HELD


def test_mutation_restoring_the_pre_open_mark_filter_would_break_this():
    """**The named behavioural test A2R2 requires.**

    If anyone restores ``mark absence -> advance entry``, this fails: the runtime
    would open at hour 1 instead of hour 0, and it would stop terminating.

    It is stated as two independent assertions because the retired rule had two
    observable consequences, and a partial restoration must not slip through by
    breaking only one of them.
    """
    start = ns("2021-03-01T00:00:00+00:00")
    aligned = world(hours=8, missing_mark=[0])

    # 1. The entry instant does not move.
    assert find_opening_instant(aligned, block(hours=8)).opened_at_ns == start

    # 2. The screen terminates rather than quietly evaluating a later-opened block.
    screen = _screen(aligned, [block(hours=8)])
    assert not screen.evaluable
    assert screen.terminal.state == MARKLESS_STATE_HELD
    assert screen.terminal.instant_ns == start


def test_mutation_reinstating_a_retired_markless_state_would_break_this():
    """The two retired states must never be produced by this runtime again."""
    for missing in ([0], [0, 1], range(8)):
        aligned = world(hours=8, missing_mark=missing)
        with pytest.raises(SourceInsufficiency) as raised:
            _run(aligned, block(hours=8))
        assert raised.value.state == MARKLESS_STATE_HELD
        assert raised.value.state not in (
            MARKLESS_STATE_PRE_OPEN,
            MARKLESS_STATE_NO_VALID_OPEN,
        )


def test_mutation_jumping_the_hole_would_break_this():
    """A runner that filtered invalid hours out would produce a shorter contiguous
    series and evaluate happily. It must terminate instead."""
    aligned = world(hours=12, missing_mark=[5])
    with pytest.raises(SourceInsufficiency) as raised:
        build_quotes(aligned, block(hours=12), ns("2021-03-01T00:00:00+00:00"))
    assert raised.value.instant_ns == ns("2021-03-01T00:00:00+00:00") + 5 * HOUR


def test_a_missing_spot_hour_inside_the_holding_window_also_terminates():
    """The uniform reading: an untested held hour is untested whatever is missing."""
    aligned = world(hours=10, missing_spot=[4])
    with pytest.raises(SourceInsufficiency) as raised:
        _run(aligned, block(hours=10))
    assert raised.value.state == MARKLESS_STATE_HELD
    assert SPOT in raised.value.missing
    assert PERPETUAL not in raised.value.missing
