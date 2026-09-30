"""R1-l: one valuation price, and for a perpetual it is the mark.

> **R1-l Unified valuation price:** one documented valuation price (mark) for
> equity, drawdown and liquidation; the close-vs-mark inconsistency removed from
> the ledger path that R8 will inherit.

The unrealised term in the equity line was measured at ``perp_close`` -- the
price one trade printed at -- while the liquidation test standing beside it
measured its threshold on ``mark_high`` and asked the executor for a liquidation
price at ``state.mark``. Section 6.7's test is
``equity < Q * mark_high * maintenance_margin_rate``, so the two sides were
priced differently: an equity the venue does not compute, against a requirement
the venue does.

Since R1-g there are **two** callers of that equity and not one:
``mark_to_market`` marks the ledger at a minute the runner decides, and
``equity_at`` answers 6.7's question about minutes the runner may *not* mark --
the newer ones a funding deferral watches for a touch. Both now reach the price
through ``_perp_pnl``, so the two cannot disagree about what the position is
worth, and the tests below hold each of them to it.

**The synthetic fixture cannot see any of this**: it writes ``mark = close`` on
every minute, so every existing test passes either way. Each test here therefore
moves one price and holds the other, which is the only arrangement in which the
question has an answer.

ENGINEERING only. Synthetic prices throughout; no economic quantity is asserted.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from chimera.carry.accounting import CarryError
from chimera.carry.hedge import HedgeState
from tests.demo_harness import build

MINUTE_MS = 60_000


def _opened(tmp_path):
    """A harness whose hedge is open, and the state of the minute it opened on."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    minute = None
    for index in range(6):
        minute = first + index * MINUTE_MS
        harness.tick(minute)
        if harness.runner.position.state is HedgeState.HEDGED:
            break
    assert harness.runner.position.state is HedgeState.HEDGED, "the fixture opened nothing"
    state = harness.runner.cursor.state_for(minute, now_ns=harness.runner.clock.now_ns)
    return harness, state


def _perp_quantity(harness) -> Decimal:
    return harness.runner.position.leg("perp").quantity


def _hedged_quantity(harness) -> Decimal:
    """The quantity section 6.7 prices its threshold on: ``min(spot, perp)``."""
    position = harness.runner.position
    return min(position.leg("spot").quantity, position.leg("perp").quantity)


# ---------------------------------------------------------------------------
# which price the equity follows
# ---------------------------------------------------------------------------
def test_equity_follows_the_mark(tmp_path):
    """Move the mark, hold the close: the equity moves, by quantity x the move.

    The position is SHORT the perpetual, so a mark above the close is an
    unrealised loss the old equity line did not admit to.
    """
    harness, state = _opened(tmp_path)
    quantity = _perp_quantity(harness)
    assert quantity > 0

    base = harness.runner.position.mark_to_market(state)
    lifted = harness.runner.position.mark_to_market(
        replace(state, mark=state.mark + Decimal("100"))
    )

    assert lifted.equity == base.equity - quantity * Decimal("100")
    assert lifted.perp_mark == state.mark + Decimal("100")


def test_equity_does_not_follow_the_close(tmp_path):
    """The two-sided half. Move the close, hold the mark: the equity stands.

    This is the assertion that fails against the previous implementation, where
    the unrealised term was computed from ``perp_close``.
    """
    harness, state = _opened(tmp_path)

    base = harness.runner.position.mark_to_market(state)
    moved = harness.runner.position.mark_to_market(
        replace(state, perp_close=state.perp_close + Decimal("100"))
    )

    assert moved.equity == base.equity


def test_the_mark_the_equity_used_is_reported(tmp_path):
    """ "One DOCUMENTED valuation price": the record says which number it was.

    Without it a reader of a mark cannot tell whether the equity beside it was
    priced at the close or at the mark, which is the ambiguity this change
    exists to end.

    Asserted on a minute where the two prices DIFFER. Asking it on a fixture
    minute proves nothing: ``mark == close`` there, so a field reporting either
    one passes. The mutation campaign found this test blind exactly that way.
    """
    harness, state = _opened(tmp_path)
    moved = replace(state, mark=state.mark + Decimal("40"))

    mark = harness.runner.position.mark_to_market(moved)

    assert mark.perp_mark == moved.mark
    assert mark.perp_mark != moved.perp_close
    assert mark.to_dict()["perp_mark"] == str(moved.mark)


# ---------------------------------------------------------------------------
# the second caller: R1-g's deferral safety pass
# ---------------------------------------------------------------------------
def test_equity_at_follows_the_mark(tmp_path):
    """``equity_at`` is the number the deferral pass asks 6.7 with (R1-g).

    It marks nothing -- that is its whole point, a later minute's low must not
    reach an earlier minute's records -- but it must price the position the same
    way the marking path does, or a runner would defer onto a second definition
    of what it is worth.
    """
    harness, state = _opened(tmp_path)
    quantity = _perp_quantity(harness)

    base = harness.runner.position.equity_at(state)
    lifted = harness.runner.position.equity_at(
        replace(state, mark=state.mark + Decimal("100"))
    )

    assert lifted == base - quantity * Decimal("100")


def test_equity_at_does_not_follow_the_close(tmp_path):
    """The two-sided half, on the deferral path."""
    harness, state = _opened(tmp_path)

    base = harness.runner.position.equity_at(state)
    moved = harness.runner.position.equity_at(
        replace(state, perp_close=state.perp_close + Decimal("100"))
    )

    assert moved == base


def test_the_marking_path_computes_no_second_equity(tmp_path):
    """What this can and cannot witness, stated rather than implied.

    Today ``mark_to_market`` asks ``equity_at``, so the two agree BY
    CONSTRUCTION and no test can strengthen that. What the assertion does catch
    is the change that would end it: an equity line inlined back into
    ``mark_to_market``, which is how the two definitions would come apart again.
    That ``equity_at`` itself prices at the mark is held by the two tests above,
    not by this one -- the mutation campaign confirmed it, by surviving here and
    dying there.

    On a minute where the mark and the close differ, because on the fixture's
    own minutes they are equal and any two implementations agree.
    """
    harness, state = _opened(tmp_path)
    moved = replace(state, mark=state.mark + Decimal("75"))

    assert harness.runner.position.equity_at(moved) == (
        harness.runner.position.mark_to_market(moved).equity
    )


def test_a_non_flat_leg_with_no_mark_is_refused_by_equity_at_too(tmp_path):
    """The refusal reaches the deferral pass, which is written to expect it.

    ``_touch_while_deferred`` wraps this call and returns the minute as a touch
    when it raises -- section 7.2's answer, unknown information on a non-flat
    position is refused -- so a minute that cannot be valued stops the runner
    rather than passing its watch silently.
    """
    harness, state = _opened(tmp_path)

    with pytest.raises(CarryError, match="cannot be valued"):
        harness.runner.position.equity_at(replace(state, mark=None))


# ---------------------------------------------------------------------------
# what the liquidation test now compares
# ---------------------------------------------------------------------------
def test_a_touch_the_close_priced_equity_would_have_missed(tmp_path):
    """The defect as an outcome, not as arithmetic: one minute, two verdicts.

    Section 6.7 is ``equity < Q * mark_high * maintenance_margin_rate``. Holding
    ``mark_high`` -- and therefore the threshold -- fixed, and asking the same
    question with each of the two candidate equities, is the only arrangement in
    which "which price" has a visible consequence.

    ``mark_high`` is chosen from the position's own numbers so the threshold
    falls strictly BETWEEN them. It has to be: the fixture capitalises the
    campaign at a million against a maintenance requirement of a thousand, so no
    ordinary price puts the threshold anywhere near either equity. Choosing it
    is legitimate and not a thumb on the scale -- ``mark_high`` is exactly what
    6.7 prices the threshold on, and the test moves it once and then holds it
    while the equity is the only thing that changes.
    """
    harness, state = _opened(tmp_path)
    position = harness.runner.position
    quantity = _hedged_quantity(harness)
    gap = Decimal("1000")

    lifted = replace(state, mark=state.mark + gap)
    at_mark = position.equity_at(lifted)
    at_close = at_mark + quantity * gap  # what the previous implementation reported
    threshold = (at_mark + at_close) / 2
    lifted = replace(
        lifted, mark_high=threshold / (quantity * position.config.maintenance_margin_rate)
    )

    # The mark stays far below the venue's liquidation price, so the second
    # branch of 6.7 answers False and the portfolio test is what decides.
    margin = position.perp.margin(position.config.perp_symbol, lifted.mark)
    assert lifted.mark < margin.liquidation_price

    assert position.liquidation_touched(lifted, equity=at_mark) is True
    assert position.liquidation_touched(lifted, equity=at_close) is False


def test_both_sides_of_the_liquidation_test_are_priced_at_the_mark(tmp_path):
    """The defect, stated as the arithmetic it produced.

    Section 6.7 compares ``equity`` against ``Q * mark_high * rate``. With the
    equity priced at the close and the threshold at the mark, a mark above the
    close made the left side too large by ``Q x (mark - close)`` -- so a touch
    could read as not touched by exactly the amount the two prices differed.
    """
    harness, state = _opened(tmp_path)
    quantity = _perp_quantity(harness)
    gap = Decimal("250")
    lifted = replace(state, mark=state.mark + gap, mark_high=state.mark + gap)

    at_mark = harness.runner.position.mark_to_market(lifted)
    priced_at_close = at_mark.equity + quantity * gap  # what the old line reported

    assert at_mark.equity < priced_at_close
    assert at_mark.perp_mark == lifted.mark
    # And the threshold the test would use is priced on that same number.
    threshold = (
        quantity * lifted.mark_high * harness.runner.position.config.maintenance_margin_rate
    )
    assert threshold > 0


# ---------------------------------------------------------------------------
# what stays on the closes, deliberately
# ---------------------------------------------------------------------------
def test_the_basis_stays_on_the_closes(tmp_path):
    """The basis is a spread the rule reads, not a valuation.

    Repricing it would change a frozen accounting definition -- it is what
    section 6.5's identity reconciles the two legs against -- and R1-l does not
    ask for that.
    """
    harness, state = _opened(tmp_path)

    base = harness.runner.position.mark_to_market(state)
    lifted = harness.runner.position.mark_to_market(
        replace(state, mark=state.mark + Decimal("100"))
    )

    assert lifted.basis == base.basis == state.perp_close - state.spot_close


def test_the_spot_leg_stays_on_its_close(tmp_path):
    """Spot has no mark, so there is no second number to be inconsistent with.

    Both halves, because they are separately reachable: the equity moves with
    the spot close, and the REPORTED ``spot_pnl`` is computed from it too. The
    second is not implied by the first -- ``spot_pnl`` is reported and never
    added to the equity -- so a spot leg quietly repriced onto the perpetual's
    mark moves nothing the first half can see. The mutation campaign found this
    test blind that way.
    """
    harness, state = _opened(tmp_path)

    base = harness.runner.position.mark_to_market(state)
    moved = harness.runner.position.mark_to_market(
        replace(state, spot_close=state.spot_close + Decimal("10"))
    )

    spot_leg = harness.runner.position.leg("spot")
    # Once, not twice: the equity line carries the INVENTORY term
    # (`quantity * spot_close`) and not `spot_pnl`, which `CarryMark` reports
    # but never adds -- the entry price is already inside `free_cash`.
    assert moved.equity == base.equity + spot_leg.quantity * Decimal("10")

    # And the reported term, on a minute where the perpetual's mark is a
    # different number, so "at the spot close" is distinguishable from "at the
    # perpetual's mark".
    apart = harness.runner.position.mark_to_market(
        replace(state, mark=state.mark + Decimal("500"))
    )
    assert apart.spot_pnl == spot_leg.quantity * (state.spot_close - spot_leg.entry_price)


# ---------------------------------------------------------------------------
# no mark
# ---------------------------------------------------------------------------
def test_a_non_flat_leg_with_no_mark_is_refused(tmp_path):
    """Refused, not valued at the close.

    Falling back would reinstate the inconsistency silently, on the one kind of
    minute where nobody would think to look. Same answer section 7.2 already
    gives: unknown information on a non-flat position is refused.
    """
    harness, state = _opened(tmp_path)

    with pytest.raises(CarryError, match="cannot be valued"):
        harness.runner.position.mark_to_market(replace(state, mark=None))


def test_a_flat_position_needs_no_mark(tmp_path):
    """The control: nothing to value, so nothing to refuse.

    And nothing to report either. ``perp_mark`` is None exactly here -- a flat
    leg has no unrealised term, so no price was used -- rather than carrying the
    minute's mark, which would name a valuation that did not happen.
    """
    # Not ticked: the fixture's rule opens on its very first minute, so a
    # position that has traded at all is no longer flat.
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    state = harness.runner.cursor.state_for(first, now_ns=harness.runner.clock.now_ns)
    assert harness.runner.position.leg("perp").quantity == 0

    mark = harness.runner.position.mark_to_market(replace(state, mark=None))

    assert mark.perp_pnl == 0
    assert mark.perp_mark is None
    assert mark.to_dict()["perp_mark"] is None
    # And the same minute with a mark present still reports none: the question
    # is whether a price was USED, not whether one was available.
    assert harness.runner.position.mark_to_market(state).perp_mark is None
