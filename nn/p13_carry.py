"""The P13 delta-hedged carry accounting engine.

Every quantity here is :class:`~decimal.Decimal`. Not for elegance: a carry
result is a small number obtained by subtracting large, nearly equal ones — the
two legs' price PnL cancels to the basis — and binary floating point loses
exactly the digits that survive that cancellation.

The engine implements :mod:`nn.p13_preregistration` and adds nothing to it: it
holds **no threshold, no filter and no parameter of its own**. Sizing, costs, the
allocation and the venue constraints are all **caller-supplied** — they arrive as
:class:`Costs`, :class:`Allocation` and :class:`Venue` — so this module cannot
invent one, but it does not read them from the preregistration either, and a
caller passing the wrong numbers is a caller bug rather than something this file
can prevent. The one thing it does read from the frozen design is the research
boundary, which it asserts rather than trusts.

The viability gate is **not implemented here at all**. Nor is the downloader, the
checksum verification, the loader and its truncating read, the source manifests,
the block runner, the stress runners, the event ledger or the decision writer.
This is the accounting core the rest of that will be built on — P13 is not
implementation-complete, and no economic quantity has ever been computed with it.

**Where the money is.** For an equal-quantity hedge the two legs' price PnL
telescopes to ``Q x (basis_in - basis_out)``: everything the price does in
between cancels. So the whole result is

    net = Q x (basis_in - basis_out)  +  funding  -  fees  -  slippage

and :func:`evaluate_block` returns each of those terms separately rather than
only their sum, because a positive net built from basis convergence is a
different claim about the world than one built from funding.

Nothing in this module reads P4-HOLD or Styx, and nothing in it can place an
order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Iterable, Sequence

from chimera.futures.accounting import liquidation_price
from chimera.futures.domain import PositionSide
from nn.p13_preregistration import DATA_BOUNDARY

ZERO = Decimal("0")
ONE = Decimal("1")

#: The value of an economic quantity that was never measured, used ONLY for the
#: close-dependent fields of an UNCLOSED block (amendment A1).
#:
#: A quiet ``NaN`` rather than a zero, and the choice is load-bearing. Zero is a
#: number: a viability gate that forgot to consult ``BlockResult.unclosed`` would
#: silently average a zero return into G2 and read a harmless worst block into
#: G3, which is exactly the flattering treatment a preregistration exists to
#: forbid. ``Decimal("NaN")`` raises :class:`decimal.InvalidOperation` on every
#: ordering comparison, so the same forgetful gate crashes instead. It fails
#: closed by construction rather than by remembering to.
NOT_DETERMINABLE = Decimal("NaN")

#: The research boundary, read from the frozen preregistration rather than
#: restated. ``DATA_BOUNDARY["enforcement"]`` says "the acquisition AND THE
#: EVALUATOR both assert it"; only the acquisition did, so the frozen sentence
#: was true of half the system. Reading it here rather than copying the literal
#: also means the two cannot drift apart.
RESEARCH_BOUNDARY_NS = int(
    __import__("datetime")
    .datetime.fromisoformat(DATA_BOUNDARY["span_end_exclusive"])
    .timestamp()
    * 1_000_000_000
)


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
        """Every friction multiplied by ``factor`` — the S1 stress, and only that."""
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
    """The capital denominator, frozen before any result."""

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
    """One realised settlement, as published by the archive.

    ``instant_ns`` is the SETTLEMENT instant. The rate is final at it and is not
    knowable before it — which is why this type carries no "predicted" field for
    anything to accidentally read.
    """

    #: A signed DECIMAL FRACTION of notional, charged once per settlement event
    #: exactly as published. Not a percent, not basis points, not annualised, and
    #: never multiplied by a settlements-per-day count.
    instant_ns: int
    rate: Decimal
    mark_price: Decimal

    #: Far outside any legitimate 8-hourly BTCUSDT settlement. A corruption and
    #: unit detector, not a claim about the venue's funding cap: a rate handed
    #: over in percent is a hundredfold error on the one term the whole
    #: checkpoint exists to measure, and it must refuse rather than scale.
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
    """One hourly grid observation, with fills and marks kept apart.

    Both kline sources are stamped by candle OPEN, so at instant ``t`` the only
    price in the row already knowable at ``t`` is the open. ``spot`` and ``perp``
    are therefore the CLOSES — used for marking, basis and the funding notional,
    and never for a fill — while ``spot_open`` and ``perp_open`` are what an order
    at ``t`` actually executes against. Filling at the close of a candle labelled
    ``t`` would execute at a price revealed an hour later, at both ends of the
    position, which is the whole of the price PnL.

    The open fields have no default. They once fell back to the close, which
    meant a production row that simply lacked an open would have executed at a
    price revealed an hour later WITHOUT SAYING SO — the exact lookahead
    ``EXECUTION_PRICE_POLICY.close_is_never_a_fill`` exists to forbid, reached by
    an absent field rather than by a decision. A synthetic witness in a flat
    world states ``spot_open=spot`` and ``perp_open=perp`` explicitly; a
    production row that cannot is INVALID and fails closed.
    """

    instant_ns: int
    spot: Decimal
    perp: Decimal
    mark: Decimal | None = None
    spot_open: Decimal | None = None
    perp_open: Decimal | None = None
    #: The hourly HIGH of the mark series, used ONLY as the conservative
    #: intra-bar liquidation touch — never as a fill and never as a mark. An
    #: hourly grid cannot resolve within-bar price action, so testing
    #: liquidation against the close alone would miss a touch the position
    #: genuinely took. Absent, the close is used and the artifact records which.
    mark_high: Decimal | None = None

    def __post_init__(self) -> None:
        if self.spot <= ZERO or self.perp <= ZERO:
            raise CarryError(
                f"non-positive price at {self.instant_ns}: spot={self.spot} perp={self.perp}"
            )
        if self.mark is not None and self.mark <= ZERO:
            raise CarryError(f"non-positive mark at {self.instant_ns}: {self.mark}")
        for name in ("spot_open", "perp_open", "mark_high"):
            value = getattr(self, name)
            if value is not None and value <= ZERO:
                raise CarryError(f"non-positive {name} at {self.instant_ns}: {value}")

    @property
    def liquidation_touch(self) -> Decimal:
        """The most adverse mark this bar can be shown to have reached.

        Adverse for a SHORT means HIGH, so the conservative test uses the mark
        high where the source provides it and falls back to the mark close, then
        the spot close. The fallback is recorded rather than assumed, so the
        check is never quietly weaker than it claims to be.
        """
        if self.mark_high is not None:
            return self.mark_high
        return self.mark if self.mark is not None else self.spot

    @property
    def liquidation_touch_is_high(self) -> bool:
        """Whether the conservative intra-bar touch was actually available."""
        return self.mark_high is not None

    @property
    def has_execution_opens(self) -> bool:
        """Whether BOTH legs carry the open the frozen design fills against."""
        return self.spot_open is not None and self.perp_open is not None

    @property
    def spot_fill(self) -> Decimal:
        """What a spot order at this instant executes against.

        Refuses when the open is absent. Substituting the close would be a
        one-hour lookahead at an execution instant, and an absent field is not a
        licence to take one: the frozen design says the close is NEVER a fill.
        """
        if self.spot_open is None:
            raise CarryError(
                f"no spot open at {self.instant_ns}; the frozen design fills at the candle "
                "OPEN and the close is never a fill, so a row without one is INVALID rather "
                "than executable at its close"
            )
        return self.spot_open

    @property
    def perp_fill(self) -> Decimal:
        """What a perpetual order at this instant executes against.

        Refuses when the open is absent, for the reason :attr:`spot_fill` gives.
        """
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
        """The basis at the prices actually transacted.

        The telescoping identity holds on what the position PAID, so the realised
        basis PnL is measured here rather than on the close series. In a world
        where open equals close the two coincide, which is why the hand-traced
        witnesses can ignore the distinction.
        """
        return self.perp_fill - self.spot_fill


def hedge_quantity(
    entry: Quote, allocation: Allocation, costs: Costs, venue: Venue
) -> Decimal:
    """The largest equal quantity BOTH allocations can fund, floored to the step.

    The minimum over the two legs, and not the spot leg alone. Sizing from spot
    only makes the perpetual's cash requirement exceed its allocation as soon as
    the perpetual trades a few basis points above spot — that is, in exactly the
    contango regimes a carry position exists to harvest — so the position would
    have been refused in a way correlated with the phenomenon under study.
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
    equity a second time — an equity line written as
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
    #: the S4 isolated balance can feel it. Under the primary portfolio model the
    #: same flow already reached equity through ``free_cash``; this accumulator is
    #: never added to equity, only consulted by the isolated liquidation test.
    perp_funding: Decimal = ZERO
    settled: list[int] = field(default_factory=list)

    @property
    def net_funding(self) -> Decimal:
        return self.funding_received - self.funding_paid

    def unrealised_perp(self, perp_price: Decimal) -> Decimal:
        """SHORT mark-to-market: gains as the perpetual falls."""
        return (self.perp_entry - perp_price) * self.quantity

    def equity(self, quote: Quote) -> Decimal:
        """Total portfolio equity. Both legs, one denominator."""
        return (
            self.free_cash
            + self.quantity * quote.spot
            + self.perp_margin
            + self.unrealised_perp(quote.perp)
        )


def open_carry(
    entry: Quote, allocation: Allocation, costs: Costs, venue: Venue
) -> CarryPosition:
    """LONG spot and SHORT perpetual, equal quantity, at ``entry``.

    Raises rather than shrinking the hedge or borrowing across legs when the
    quantity rounds away or the venue's minimum notional is not met: a block that
    cannot be opened under the frozen rules is recorded as not opened, never
    opened under different ones.
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
    ``cash_flow = -sign(side) x notional x rate``, for a SHORT: positive rate
    means longs pay shorts, so the short leg RECEIVES. It is asserted against
    :func:`chimera.futures.accounting.funding_cash_flow` in the tests rather than
    merely restated here.

    Idempotent by settlement instant: a duplicated archive row changes nothing.
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
    """Whether the perpetual leg is liquidated at ``mark``.

    Two models, both at 1x, and the difference is what collateralises the short:

    *isolated* walls the perpetual off from the spot leg that hedges it, so it is
    liquidated on the venue's own formula — for a SHORT at 1x, roughly a doubling.

    *portfolio* recognises that both legs are one book. At equal quantity the
    portfolio is price-invariant, so price alone cannot exhaust it; only funding
    losses and costs can, and the test is total equity against the maintenance
    requirement.
    """
    mark = quote.liquidation_touch
    if isolated:
        # Strict: the walled-off balance carries its own funding and gets no
        # rescue from free cash. Routing funding to the portfolio instead would
        # make the "unforgiving" bound systematically lenient — a short paying
        # funding for months would never feel it in the balance meant to be
        # isolated.
        # Marked at the adverse touch, not the close: the balance was at its
        # worst when the mark was, and a bar that touched the threshold
        # liquidated the position whatever it closed at.
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

    Both legs pay an exit fee and exit slippage. Omitting either would flatter
    every block by the same amount, which is why the tests assert the total
    friction rather than trusting the call sites.

    ``forfeit_perp_margin`` is the isolated-liquidation case: the venue closes the
    perpetual and consumes the margin that was walled off with it, so that leg
    returns ``max(margin + realised - exit costs, 0)`` rather than the full
    amount. The SPOT leg is not forfeited with it — it is still owned, and it has
    gained roughly what the short lost, which is precisely why an isolated
    liquidation is a hedge failure rather than a total loss.
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


@dataclass(frozen=True)
class BlockResult:
    """One chronological block's economics, decomposed rather than summarised."""

    label: str
    opened: bool
    reason: str
    settlements: int
    quantity: Decimal
    basis_entry: Decimal
    basis_exit: Decimal
    basis_pnl: Decimal
    funding_received: Decimal
    funding_paid: Decimal
    fees: Decimal
    slippage: Decimal
    rebalance_cost: Decimal
    net_pnl: Decimal
    net_return: Decimal
    liquidated: bool
    max_adverse_excursion: Decimal
    thin_sample: bool
    #: The instant the liquidation TRIGGER became observable — the bar whose
    #: adverse touch crossed the threshold. It is not the fill instant: under
    #: ``MARGIN_AND_LIQUIDATION.forced_close_price`` the forced close executes at
    #: the OPEN OF THE FOLLOWING BAR, so the two are recorded separately and an
    #: auditor can see that no fill preceded its own trigger.
    liquidation_instant_ns: int | None = None
    #: The instant of the bar whose OPEN the forced close actually filled at.
    #: Always strictly after :attr:`liquidation_instant_ns`, or ``None`` when no
    #: permitted following bar existed.
    forced_close_instant_ns: int | None = None
    #: Amendment A1: the position was opened but no permitted exit fill exists,
    #: so this block has NO determinable return. The close-dependent economics
    #: below are ``NaN`` rather than zero precisely so that a gate which ignores
    #: this flag raises instead of averaging in a number nobody measured.
    unclosed: bool = False

    @property
    def net_funding(self) -> Decimal:
        return self.funding_received - self.funding_paid


def evaluate_block(
    label: str,
    quotes: Sequence[Quote],
    settlements: Iterable[FundingSettlement],
    allocation: Allocation,
    costs: Costs,
    venue: Venue,
    min_settlements: int,
    isolated: bool = False,
) -> BlockResult:
    """One position, opened at the first quote and closed at the last.

    Funding is applied only for settlements strictly inside the holding window —
    a settlement before the open or after the close is not this position's cash
    flow — and only ever at its own instant, which is the whole of the causality
    story for a strategy that takes no funding signal.

    **Liquidation is a trigger and a fill, and they are different instants.** The
    trigger bar is the one whose adverse intra-bar touch crossed the threshold;
    per ``MARGIN_AND_LIQUIDATION.forced_close_price`` the forced close fills at
    the OPEN OF THE FOLLOWING BAR. Filling at the trigger bar's own open — which
    this function once did — executes at a price stamped an hour BEFORE the high
    that caused the liquidation, so the fill would precede its own cause. The
    position stops accruing at the TRIGGER instant: a settlement after it belongs
    to a position that no longer existed, and is not applied even though the fill
    bar is later.

    When the trigger is the block's final quote there is no following bar inside
    the permitted region, and the exit search may not run past the block end or
    the research boundary to find one. Amendment A1 makes that block UNCLOSED:
    no price is invented, nothing beyond the bound is read, the close-dependent
    fields are :data:`NOT_DETERMINABLE`, and the screen is INVALID.
    """
    for quote in quotes:
        if quote.instant_ns >= RESEARCH_BOUNDARY_NS:
            raise CarryError(
                f"quote at {quote.instant_ns} is at or after the research boundary "
                f"{DATA_BOUNDARY['span_end_exclusive']}. A row past the boundary is a "
                "refusal, not something to filter: reaching here means a loader admitted "
                "one."
            )
        if not quote.has_execution_opens:
            raise CarryError(
                f"quote at {quote.instant_ns} carries no explicit execution open on one or "
                "both legs. The frozen design fills at the candle OPEN and the close is "
                "NEVER a fill, so a row without one is INVALID and is refused here rather "
                "than reported as a block that merely failed to open — reaching here means "
                "a loader admitted a row it should have rejected."
            )
    for settlement in (settlements := list(settlements)):
        if settlement.instant_ns >= RESEARCH_BOUNDARY_NS:
            raise CarryError(
                f"funding settlement at {settlement.instant_ns} is at or after the research "
                f"boundary {DATA_BOUNDARY['span_end_exclusive']}"
            )

    empty = BlockResult(
        label=label,
        opened=False,
        reason="",
        settlements=0,
        quantity=ZERO,
        basis_entry=ZERO,
        basis_exit=ZERO,
        basis_pnl=ZERO,
        funding_received=ZERO,
        funding_paid=ZERO,
        fees=ZERO,
        slippage=ZERO,
        rebalance_cost=ZERO,
        net_pnl=ZERO,
        net_return=ZERO,
        liquidated=False,
        max_adverse_excursion=ZERO,
        thin_sample=True,
    )
    if len(quotes) < 2:
        return BlockResult(**{**empty.__dict__, "reason": "fewer than two quotes in block"})

    entry, final = quotes[0], quotes[-1]
    try:
        position = open_carry(entry, allocation, costs, venue)
    except CarryError as exc:
        return BlockResult(**{**empty.__dict__, "reason": f"not opened: {exc}"})

    due = sorted(
        (s for s in settlements if entry.instant_ns < s.instant_ns <= final.instant_ns),
        key=lambda s: s.instant_ns,
    )

    worst = ZERO
    settled_count = 0
    pending = {s.instant_ns: s for s in due}
    traded_quantity = position.quantity
    trigger_index: int | None = None

    for index, quote in enumerate(quotes[1:], start=1):
        for instant in sorted(k for k in pending if k <= quote.instant_ns):
            apply_funding(position, pending.pop(instant))
            settled_count += 1
        worst = min(worst, position.equity(quote) - allocation.total_capital)
        if is_liquidated(position, quote, venue, isolated):
            trigger_index = index
            break

    liquidated = trigger_index is not None
    liquidation_instant: int | None = None
    forced_close_instant: int | None = None
    if trigger_index is None:
        exit_quote = final
    else:
        # MARGIN_AND_LIQUIDATION.forced_close_price: the trigger is detected from
        # WITHIN-bar action an hourly grid cannot resolve, so the fill is the OPEN
        # OF THE FOLLOWING BAR. Filling at the trigger bar's own open would
        # execute at a price stamped BEFORE the high that caused the liquidation
        # — a fill that precedes its own cause.
        liquidation_instant = quotes[trigger_index].instant_ns
        if trigger_index + 1 >= len(quotes):
            # Amendment A1. There is no following bar inside the block, and the
            # exit search is bounded by the block end and the research boundary,
            # so no permitted fill exists. Nothing is invented and nothing is
            # read past the bound: the block is UNCLOSED.
            return BlockResult(
                **{
                    **empty.__dict__,
                    "opened": True,
                    "reason": (
                        "UNCLOSED: liquidation triggered at the final quote of the block, "
                        "and the preregistered forced-close fill is the OPEN of the "
                        "FOLLOWING bar, which lies outside the permitted region. Per "
                        "amendment A1 the block has no determinable return and the screen "
                        "is INVALID."
                    ),
                    "settlements": settled_count,
                    "quantity": traded_quantity,
                    "basis_entry": entry.fill_basis,
                    "basis_exit": NOT_DETERMINABLE,
                    "basis_pnl": NOT_DETERMINABLE,
                    "funding_received": position.funding_received,
                    "funding_paid": position.funding_paid,
                    "fees": position.fees,
                    "slippage": position.slippage,
                    "net_pnl": NOT_DETERMINABLE,
                    "net_return": NOT_DETERMINABLE,
                    "liquidated": True,
                    "max_adverse_excursion": worst,
                    "thin_sample": settled_count < min_settlements,
                    "liquidation_instant_ns": liquidation_instant,
                    "forced_close_instant_ns": None,
                    "unclosed": True,
                }
            )
        exit_quote = quotes[trigger_index + 1]
        forced_close_instant = exit_quote.instant_ns

    # The identity holds on the prices actually transacted, not on the marks.
    basis_entry = entry.fill_basis
    basis_exit = exit_quote.fill_basis
    final_equity = close_carry(
        position, exit_quote, costs, forfeit_perp_margin=liquidated and isolated
    )
    net_pnl = final_equity - allocation.total_capital
    return BlockResult(
        label=label,
        opened=True,
        reason=(
            "liquidated; forced close at the following bar's open"
            if liquidated
            else "closed at block end"
        ),
        settlements=settled_count,
        quantity=traded_quantity,
        basis_entry=basis_entry,
        basis_exit=basis_exit,
        basis_pnl=(basis_entry - basis_exit) * traded_quantity,
        funding_received=position.funding_received,
        funding_paid=position.funding_paid,
        fees=position.fees,
        slippage=position.slippage,
        rebalance_cost=ZERO,
        net_pnl=net_pnl,
        net_return=net_pnl / allocation.total_capital,
        liquidated=liquidated,
        max_adverse_excursion=worst,
        thin_sample=settled_count < min_settlements,
        liquidation_instant_ns=liquidation_instant,
        forced_close_instant_ns=forced_close_instant,
    )
