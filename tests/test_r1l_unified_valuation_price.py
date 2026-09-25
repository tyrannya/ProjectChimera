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
    """
    harness, state = _opened(tmp_path)

    mark = harness.runner.position.mark_to_market(state)

    assert mark.perp_mark == state.mark
    assert mark.to_dict()["perp_mark"] == str(state.mark)


# ---------------------------------------------------------------------------
# what the liquidation test now compares
# ---------------------------------------------------------------------------
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
    """Spot has no mark, so there is no second number to be inconsistent with."""
    harness, state = _opened(tmp_path)

    base = harness.runner.position.mark_to_market(state)
    moved = harness.runner.position.mark_to_market(
        replace(state, spot_close=state.spot_close + Decimal("10"))
    )

    spot_quantity = harness.runner.position.leg("spot").quantity
    # Once, not twice: the equity line carries the INVENTORY term
    # (`quantity * spot_close`) and not `spot_pnl`, which `CarryMark` reports
    # but never adds -- the entry price is already inside `free_cash`.
    assert moved.equity == base.equity + spot_quantity * Decimal("10")


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
    """The control: nothing to value, so nothing to refuse."""
    # Not ticked: the fixture's rule opens on its very first minute, so a
    # position that has traded at all is no longer flat.
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    state = harness.runner.cursor.state_for(first, now_ns=harness.runner.clock.now_ns)
    assert harness.runner.position.leg("perp").quantity == 0

    mark = harness.runner.position.mark_to_market(replace(state, mark=None))

    assert mark.perp_pnl == 0
