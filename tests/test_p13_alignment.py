"""Alignment: two validities, no forward fill, and one funding lookup.

The block runner's witnesses exercise most of this indirectly. What is tested here
is the layer's own contract, especially the two places a convenience would become
a fabricated observation.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from nn.p13_alignment import (
    MARK,
    PERPETUAL,
    SPOT,
    AlignedSources,
    AlignmentError,
    grid_instants,
    monthly_period,
)
from nn.p13_carry import CarryError
from tests.p13_synthetic import HOUR, funding_row, ns, world

START = "2021-03-01T00:00:00+00:00"


def test_the_reference_grid_comes_from_the_calendar_not_from_the_rows():
    """A hole must be visible, which means the grid cannot be built from the rows."""
    instants = list(grid_instants(ns(START), ns(START) + 5 * HOUR))
    assert len(instants) == 5
    assert instants[0] == ns(START)
    assert instants[-1] == ns(START) + 4 * HOUR
    aligned = world(hours=5, missing_mark=[2])
    # The grid still names the missing hour; the sources simply do not answer it.
    assert instants[2] not in aligned.mark
    assert instants[2] in aligned.spot


def test_an_instant_reports_exactly_which_sources_are_missing():
    aligned = world(hours=4, missing_spot=[1], missing_mark=[2], missing_perp=[3])
    base = ns(START)
    assert aligned.instant_validity(base).missing == ()
    assert aligned.instant_validity(base + HOUR).missing == (SPOT,)
    assert aligned.instant_validity(base + 2 * HOUR).missing == (MARK,)
    assert aligned.instant_validity(base + 3 * HOUR).missing == (PERPETUAL,)


def test_execution_validity_and_liquidation_validity_are_independent():
    """The exit-bar exemption is only expressible because these are two questions."""
    aligned = world(hours=3, missing_mark=[1], missing_spot=[2])
    base = ns(START)
    mark_gone = aligned.instant_validity(base + HOUR)
    assert mark_gone.has_execution
    assert not mark_gone.has_liquidation_mark
    assert mark_gone.valid_for_exit
    assert not mark_gone.valid_for_holding

    leg_gone = aligned.instant_validity(base + 2 * HOUR)
    assert not leg_gone.has_execution
    assert not leg_gone.valid_for_exit
    assert not leg_gone.valid_for_holding


def test_a_quote_carries_opens_as_fills_and_closes_as_marks():
    """Filling at a close would execute at a price revealed an hour later."""
    aligned = AlignedSources.build(
        spot=world(hours=1).spot.values(),
        perpetual=world(hours=1).perpetual.values(),
        mark=world(hours=1).mark.values(),
        funding=(),
        published_mark_periods={"2021-03"},
    )
    quote = aligned.quote(ns(START))
    assert quote.spot_fill == quote.spot_open
    assert quote.perp_fill == quote.perp_open
    assert quote.spot == Decimal("30000")
    assert quote.perp == Decimal("30030")
    assert quote.mark == Decimal("30010")


def test_a_quote_is_refused_rather_than_filled_in_when_a_leg_is_missing():
    aligned = world(hours=2, missing_perp=[1])
    with pytest.raises(AlignmentError, match="missing"):
        aligned.quote(ns(START) + HOUR)


def test_the_exit_exemption_does_not_smuggle_in_a_liquidation_touch():
    """A quote built without a mark has no authorised touch, and says so."""
    aligned = world(hours=2, missing_mark=[1])
    quote = aligned.quote(ns(START) + HOUR, require_mark=False)
    assert quote.mark is None
    assert quote.mark_high is None
    with pytest.raises(CarryError, match="no mark series"):
        _ = quote.liquidation_touch


def test_the_funding_base_is_the_candle_at_or_immediately_preceding():
    """The frozen phrase includes the settlement hour itself."""
    aligned = world(hours=6, mark_close={2: Decimal("30500")})
    settlements, bases = aligned.settlements((funding_row(2, "0.0001"),))
    assert bases[0].base_instant_ns == ns(START) + 2 * HOUR
    assert settlements[0].mark_price == Decimal("30500")


def test_the_funding_base_walks_back_when_the_settlement_hour_has_no_mark():
    aligned = world(hours=6, missing_mark=[3], mark_close={2: Decimal("30500")})
    _, bases = aligned.settlements((funding_row(3, "0.0001"),))
    assert bases[0].base_instant_ns == ns(START) + 2 * HOUR
    assert bases[0].source == MARK


def test_a_published_mark_month_with_no_preceding_candle_is_refused_not_substituted():
    """MARK_PRICE_FALLBACK is triggered by an unpublished OBJECT, not a row hole."""
    aligned = world(hours=4, missing_mark=[0])
    with pytest.raises(AlignmentError, match="not triggered"):
        aligned.settlements((funding_row(0, "0.0001"),))


def test_the_fallback_is_keyed_on_the_settlements_own_month():
    assert monthly_period(ns("2021-03-01T00:00:00+00:00")) == "2021-03"
    assert monthly_period(ns("2021-12-31T23:00:00+00:00")) == "2021-12"
    assert monthly_period(ns("2022-01-01T00:00:00+00:00")) == "2022-01"


def test_nothing_in_alignment_forward_fills_a_price():
    """Mutation guard. The only backward lookup is the frozen funding base."""
    import ast
    import inspect

    import nn.p13_alignment as module

    source = inspect.getsource(module)
    for banned in ("ffill", "fillna", "pad(", "bfill"):
        assert banned not in source
    # And the only search over instants is the funding one.
    tree = ast.parse(source)
    searches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("bisect")
    ]
    assert len(searches) == 1
