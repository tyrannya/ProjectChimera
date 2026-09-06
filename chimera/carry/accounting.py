"""Two-leg carry accounting: the P13 formulas, ported for the demo runtime.

**A port, not an import.** Section 6.9 of the adopted master plan reuses P13's
pure ``Decimal`` arithmetic and says why it may not simply be imported:
:mod:`nn.p13_carry` pulls ``pandas`` in through ``nn.p13_preregistration``, and
section 16's tier table forbids the demo runtime from importing historical
research code at all. So the formulas are copied, and
``tests/test_carry_accounting_parity.py`` asserts term-by-term equality against
:mod:`nn.p13_carry` on synthetic worlds -- which is what makes this a port
rather than a second, drifting implementation.

**Every quantity is ``Decimal``.** Floats are not admitted anywhere on this
path: replay parity (section 10) requires that the same inputs produce the same
bytes, and binary floating point does not give that for money.

**Signs.** There is one funding convention in this repository and it is stated
in :mod:`chimera.futures.accounting`:

    ``cash_flow = -sign(side) * notional * rate``

A negative cash flow means the position *paid*. The corresponding positive
*cost rate* is the negation of cash-flow-per-notional, ``sign(side) * rate``,
which is amendment **A10** and is what :mod:`chimera.risk` vetoes on. The two
are the same statement read from opposite ends; neither is free to drift from
the other. The superseded ``-sign(side) * rate`` cost formula is not
resurrected here.

**What is deliberately absent.** P13's ``NOT_DETERMINABLE``, its block and
screen machinery, and its data boundary belong to the historical checkpoint and
have no meaning for a forward-running demo. Nothing in this module reads a
historical series, and nothing in it computes a research result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal

from chimera.futures.accounting import liquidation_price
from chimera.futures.domain import PositionSide

ZERO = Decimal("0")
ONE = Decimal("1")

#: Which series answered :attr:`Quote.liquidation_touch`, as a recordable name.
TOUCH_MARK_HIGH = "mark_high"
TOUCH_MARK_CLOSE = "mark_close"
TOUCH_SOURCES: tuple[str, ...] = (TOUCH_MARK_HIGH, TOUCH_MARK_CLOSE)


class CarryError(RuntimeError):
    """A carry quantity cannot be computed from what is known."""


@dataclass(frozen=True)
class Costs:
    """Per-leg frictions, as fractions of notional. Never bps, never percent."""

    spot_fee: Decimal
    spot_slippage: Decimal
    perp_fee: Decimal
    perp_slippage: Decimal

    def __post_init__(self) -> None:
        for name in ("spot_fee", "spot_slippage", "perp_fee", "perp_slippage"):
            rate = getattr(self, name)
            if not isinstance(rate, Decimal):
                raise CarryError(f"{name} must be Decimal, not {type(rate).__name__}")
            if not (ZERO <= rate < ONE):
                raise CarryError(
                    f"{name}={rate} is not a fraction in [0, 1). A rate given in percent or "
                    "basis points would land here rather than silently scaling the result."
                )

    def scaled(self, factor: Decimal) -> "Costs":
        """Every friction multiplied by ``factor``."""
        return Costs(
            spot_fee=self.spot_fee * factor,
            spot_slippage=self.spot_slippage * factor,
            perp_fee=self.perp_fee * factor,
            perp_slippage=self.perp_slippage * factor,
        )


@dataclass(frozen=True)
class Venue:
    """The hard facts the venue imposes. Required, never defaulted."""

    step_size: Decimal
    min_notional: Decimal
    maintenance_margin_rate: Decimal

    def __post_init__(self) -> None:
        if self.step_size <= ZERO:
            raise CarryError(f"step_size={self.step_size} is not positive")
        if self.min_notional < ZERO:
            raise CarryError(f"min_notional={self.min_notional} is negative")
        if not (ZERO < self.maintenance_margin_rate < ONE):
            raise CarryError(
                f"maintenance_margin_rate={self.maintenance_margin_rate} is not in (0, 1)"
            )

    def step_floor(self, quantity: Decimal) -> Decimal:
        """``quantity`` rounded DOWN to the step size, never up.

        Down, because a position one step larger than the capital authorised is a
        position the capital did not authorise.
        """
        return (quantity / self.step_size).to_integral_value(ROUND_DOWN) * self.step_size


@dataclass(frozen=True)
class Allocation:
    """The capital denominator."""

    total_capital: Decimal
    spot: Decimal
    perp: Decimal

    def __post_init__(self) -> None:
        if self.total_capital <= ZERO:
            raise CarryError(f"total_capital={self.total_capital} is not positive")
        if self.spot < ZERO or self.perp < ZERO:
            raise CarryError("an allocation cannot be negative")
        if self.spot + self.perp > self.total_capital:
            raise CarryError(
                f"allocations {self.spot} + {self.perp} exceed total capital "
                f"{self.total_capital}; that would be leverage introduced by arithmetic"
            )


@dataclass(frozen=True)
class FundingSettlement:
    """One realised settlement, as published.

    ``instant_ns`` is the SETTLEMENT instant. The rate is final at it and is not
    knowable before it, which is why this type carries no "predicted" field.
    """

    #: A signed DECIMAL FRACTION of notional, charged once per settlement event
    #: exactly as published. Not a percent, not basis points, not annualised, and
    #: never multiplied by a settlements-per-day count.
    instant_ns: int
    rate: Decimal
    mark_price: Decimal

    #: Far outside any legitimate 8-hourly BTCUSDT settlement. A corruption and
    #: unit detector, not a claim about the venue's funding cap: a rate handed
    #: over in percent is a hundredfold error on the one term that matters, and
    #: it must refuse rather than scale, clip or winsorise.
    MAX_PLAUSIBLE_RATE = Decimal("0.01")

    def __post_init__(self) -> None:
        if self.mark_price <= ZERO:
            raise CarryError(f"funding mark price {self.mark_price} is not positive")
        if abs(self.rate) > self.MAX_PLAUSIBLE_RATE:
            raise CarryError(
                f"funding rate {self.rate} exceeds {self.MAX_PLAUSIBLE_RATE} in magnitude. "
                "A realised 8-hourly BTCUSDT rate is on the order of 1e-4; this is refused as "
                "a unit error rather than clipped, filtered or winsorised."
            )


@dataclass(frozen=True)
class Quote:
    """One grid observation, with fills and marks kept apart.

    ``spot`` and ``perp`` are the CLOSES -- used for marking, basis and the
    funding notional, never for a fill. ``spot_open`` and ``perp_open`` are what
    an order at this instant actually executes against. The open fields have no
    default: filling at a close would execute against a price revealed later,
    and an absent field is not a licence to take that lookahead.
    """

    instant_ns: int
    spot: Decimal
    perp: Decimal
    mark: Decimal | None = None
    spot_open: Decimal | None = None
    perp_open: Decimal | None = None
    #: The bar HIGH of the mark series, used ONLY as the conservative intra-bar
    #: liquidation touch -- never as a fill and never as a mark.
    mark_high: Decimal | None = None

    def _no_mark_series(self) -> CarryError:
        return CarryError(
            f"no mark series at {self.instant_ns}; a spot or perpetual close cannot see a "
            "perpetual mark spike. A held bar without a mark is INVALID rather than testable."
        )

    @property
    def liquidation_touch(self) -> Decimal:
        """The most adverse mark this bar can be shown to have reached.

        Adverse for a SHORT means HIGH, so the conservative test uses the mark
        high where the source provides it and falls back to the mark close.
        There is no third tier: a bar carrying neither is REFUSED.
        """
        if self.mark_high is not None:
            return self.mark_high
        if self.mark is None:
            raise self._no_mark_series()
        return self.mark

    @property
    def liquidation_touch_is_high(self) -> bool:
        """Whether the conservative intra-bar touch was actually available."""
        return self.mark_high is not None

    @property
    def liquidation_touch_source(self) -> str:
        """WHICH series :attr:`liquidation_touch` came from, as a recordable name."""
        if self.mark_high is not None:
            return TOUCH_MARK_HIGH
        if self.mark is None:
            raise self._no_mark_series()
        return TOUCH_MARK_CLOSE

    @property
    def has_execution_opens(self) -> bool:
        """Whether BOTH legs carry the open this design fills against."""
        return self.spot_open is not None and self.perp_open is not None

    @property
    def spot_fill(self) -> Decimal:
        """What a spot order at this instant executes against."""
        if self.spot_open is None:
            raise CarryError(
                f"no spot open at {self.instant_ns}; the frozen design fills at the candle "
                "OPEN and the close is never a fill, so a row without one is INVALID rather "
                "than executable at its close"
            )
        return self.spot_open

    @property
    def perp_fill(self) -> Decimal:
        """What a perpetual order at this instant executes against."""
        if self.perp_open is None:
            raise CarryError(
                f"no perpetual open at {self.instant_ns}; the frozen design fills at the "
                "candle OPEN and the close is never a fill, so a row without one is INVALID "
                "rather than executable at its close"
            )
        return self.perp_open

    @property
    def basis(self) -> Decimal:
        """``perp_close - spot_close``, in quote units per BTC. A mark, not a fill."""
        return self.perp - self.spot

    @property
    def fill_basis(self) -> Decimal:
        """The basis at the prices actually transacted."""
        return self.perp_fill - self.spot_fill


def hedge_quantity(
    entry: Quote, allocation: Allocation, costs: Costs, venue: Venue
) -> Decimal:
    """The largest equal quantity BOTH allocations can fund, floored to the step.

    The minimum over the two legs, not the spot leg alone: sizing from spot only
    makes the perpetual's cash requirement exceed its allocation as soon as the
    perpetual trades a few basis points above spot -- that is, in exactly the
    contango regimes a carry position exists to harvest.
    """
    spot_bound = allocation.spot / (
        entry.spot_fill * (ONE + costs.spot_fee + costs.spot_slippage)
    )
    perp_bound = allocation.perp / (
        entry.perp_fill * (ONE + costs.perp_fee + costs.perp_slippage)
    )
    return venue.step_floor(min(spot_bound, perp_bound))


@dataclass
class CarryPosition:
    """One open delta-hedged position, and its cash.

    ``free_cash`` is the single place fees, slippage and funding reach equity.
    The separate accumulators below exist for REPORTING and are never added to
    equity a second time -- an equity line written as
    ``cash + legs + funding - fees`` double-counts every term cash already holds.
    """

    quantity: Decimal
    spot_entry: Decimal
    perp_entry: Decimal
    perp_margin: Decimal
    free_cash: Decimal
    leverage: Decimal = ONE
    fees: Decimal = ZERO
    slippage: Decimal = ZERO
    funding_received: Decimal = ZERO
    funding_paid: Decimal = ZERO
    #: Cumulative funding attributed to the perpetual leg, tracked separately so
    #: the isolated balance can feel it. Under the portfolio model the same flow
    #: already reached equity through ``free_cash``; this accumulator is never
    #: added to equity, only consulted by the isolated liquidation test.
    perp_funding: Decimal = ZERO
    settled: list[int] = field(default_factory=list)

    @property
    def net_funding(self) -> Decimal:
        return self.funding_received - self.funding_paid

    def unrealised_perp(self, perp_price: Decimal) -> Decimal:
        """SHORT mark-to-market: gains as the perpetual falls."""
        return (self.perp_entry - perp_price) * self.quantity

    def equity_at(self, spot_price: Decimal, perp_price: Decimal) -> Decimal:
        """Total portfolio equity marked at two given prices. Both legs, one pool.

        At the entry instant this returns exactly
        ``total_capital - entry fees - entry slippage``, which is the accounting
        invariant the controls assert.
        """
        return (
            self.free_cash
            + self.quantity * spot_price
            + self.perp_margin
            + (self.perp_entry - perp_price) * self.quantity
        )

    def equity(self, quote: Quote) -> Decimal:
        """Total portfolio equity at ``quote``'s CLOSES -- the end-of-bar mark."""
        return self.equity_at(quote.spot, quote.perp)


def open_carry(
    entry: Quote, allocation: Allocation, costs: Costs, venue: Venue
) -> CarryPosition:
    """LONG spot and SHORT perpetual, equal quantity, at ``entry``.

    Raises rather than shrinking the hedge or borrowing across legs when the
    quantity rounds away or the venue's minimum notional is not met.

    Note the ``perp_margin`` term in ``free_cash``. Section 6.6 of the plan
    writes the cash line without it; that summary omits a term the equity
    identity adds back, so equity would be overstated by exactly the margin.
    The ported arithmetic below is the frozen one, and the parity test is what
    proves the two agree rather than this comment.
    """
    quantity = hedge_quantity(entry, allocation, costs, venue)
    if quantity <= ZERO:
        raise CarryError("hedge quantity rounds to zero at this step size")

    spot_notional = quantity * entry.spot_fill
    perp_notional = quantity * entry.perp_fill
    for name, notional in (("spot", spot_notional), ("perp", perp_notional)):
        if notional < venue.min_notional:
            raise CarryError(
                f"{name} notional {notional} is below min_notional {venue.min_notional}"
            )

    spot_fee = spot_notional * costs.spot_fee
    spot_slip = spot_notional * costs.spot_slippage
    perp_fee = perp_notional * costs.perp_fee
    perp_slip = perp_notional * costs.perp_slippage
    perp_margin = perp_notional / ONE  # leverage is exactly 1x; written, not implied

    free_cash = (
        allocation.total_capital
        - spot_notional
        - spot_fee
        - spot_slip
        - perp_margin
        - perp_fee
        - perp_slip
    )
    if free_cash < ZERO:
        raise CarryError(
            f"opening would need {-free_cash} more than the total capital; the sizing rule "
            "is supposed to make this unreachable, so reaching it is a bug and not a trade"
        )

    return CarryPosition(
        quantity=quantity,
        spot_entry=entry.spot_fill,
        perp_entry=entry.perp_fill,
        perp_margin=perp_margin,
        free_cash=free_cash,
        fees=spot_fee + perp_fee,
        slippage=spot_slip + perp_slip,
    )


def apply_funding(position: CarryPosition, settlement: FundingSettlement) -> Decimal:
    """Charge or credit one settlement. Returns the signed flow; 0 if repeated.

    The sign is the repository's single convention,
    ``cash_flow = -sign(side) * notional * rate``. For a SHORT: a positive rate
    means longs pay shorts, so the short leg RECEIVES. The notional is the
    quantity at the settlement's own MARK, not at entry.

    Idempotent by settlement instant: a duplicated row changes nothing, which is
    what stops a replayed or re-read settlement from being booked twice.
    """
    if settlement.instant_ns in position.settled:
        return ZERO
    notional = position.quantity * settlement.mark_price
    flow = -Decimal(PositionSide.SHORT.sign) * notional * settlement.rate
    position.settled.append(settlement.instant_ns)
    position.free_cash += flow
    position.perp_funding += flow
    if flow < ZERO:
        position.funding_paid += -flow
    elif flow > ZERO:
        position.funding_received += flow
    return flow


def is_liquidated(position: CarryPosition, quote: Quote, venue: Venue, isolated: bool) -> bool:
    """Whether the perpetual leg is liquidated at the adverse mark.

    Two models, both at 1x, and the difference is what collateralises the short:

    *isolated* walls the perpetual off from the spot leg that hedges it, so it is
    liquidated on the venue's own formula.

    *portfolio* recognises that both legs are one book. At equal quantity the
    portfolio is price-invariant, so price alone cannot exhaust it; only funding
    losses and costs can, and the test is total equity against maintenance.
    """
    mark = quote.liquidation_touch
    if isolated:
        # Strict: the walled-off balance carries its own funding and gets no
        # rescue from free cash, and it is marked at the adverse touch rather
        # than the close -- a bar that touched the threshold liquidated the
        # position whatever it closed at.
        balance = position.perp_margin + position.unrealised_perp(mark) + position.perp_funding
        if balance <= position.quantity * mark * venue.maintenance_margin_rate:
            return True
        threshold = liquidation_price(
            PositionSide.SHORT,
            position.perp_entry,
            position.leverage,
            venue.maintenance_margin_rate,
        )
        return mark >= threshold
    # Portfolio: the real spot price, not an assumption that spot tracked perp.
    maintenance = position.quantity * mark * venue.maintenance_margin_rate
    return position.equity(quote) < maintenance


def close_carry(
    position: CarryPosition,
    exit_quote: Quote,
    costs: Costs,
    forfeit_perp_margin: bool = False,
) -> Decimal:
    """Close both legs at ``exit_quote``. Returns final equity, which is all cash.

    Both legs pay an exit fee and exit slippage.

    ``forfeit_perp_margin`` is the isolated-liquidation case: the venue closes
    the perpetual and consumes the margin walled off with it, so that leg returns
    ``max(margin + realised - exit costs, 0)``. The SPOT leg is not forfeited
    with it -- it is still owned, and it has gained roughly what the short lost,
    which is precisely why an isolated liquidation is a hedge failure rather than
    a total loss.
    """
    spot_notional = position.quantity * exit_quote.spot_fill
    perp_notional = position.quantity * exit_quote.perp_fill

    spot_fee = spot_notional * costs.spot_fee
    spot_slip = spot_notional * costs.spot_slippage
    perp_fee = perp_notional * costs.perp_fee
    perp_slip = perp_notional * costs.perp_slippage

    perp_realised = position.unrealised_perp(exit_quote.perp_fill)
    perp_return = position.perp_margin + perp_realised - perp_fee - perp_slip
    if forfeit_perp_margin:
        perp_return = max(perp_return, ZERO)

    position.free_cash += spot_notional - spot_fee - spot_slip + perp_return
    position.fees += spot_fee + perp_fee
    position.slippage += spot_slip + perp_slip
    position.perp_margin = ZERO
    position.quantity = ZERO
    return position.free_cash
