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

**What this module refuses.** Four fail-closed rules, each of which existed as a
sentence in the frozen design before it existed as code here:

* the holding window is **bars 0 .. N-1** — the position holds through the
  remainder of the bar it opened at, and through nothing after the open it closed
  at. Liquidation, the intra-bar touch and the maximum adverse excursion share
  that one window, so neither pre-entry nor post-exit price action can reach any
  of them;
* an economic quantity **nobody measured is not a zero**. A block that never
  opened reports :data:`NOT_DETERMINABLE` for every field that would have been a
  measurement of a position, and structural zeroes — no fill, so no fee; no
  position, so no funding — as the numbers they genuinely are;
* **which series the liquidation check used is recorded**, per block and per
  source, so a check run entirely against fallback closes can never present
  itself as one run against hourly mark highs;
* **causality comes from the instants, never from the caller's list order.**
  Quote timestamps must be strictly increasing and unique before any economic
  state exists, and a hole in the hourly grid is measured rather than smoothed
  over.

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


#: The nominal spacing of the frozen hourly grid, in nanoseconds.
#:
#: Not a parameter and not a tolerance: every P13 kline source is the venue's
#: ``1h`` archive and ``DATA_SOURCES`` states the semantics as "the candle OPEN;
#: the candle is complete at open + 1h". It exists so a MULTI-HOUR jump between
#: two adjacent observations cannot look identical to an ordinary +1h
#: transition. Nothing economic is decided from it — it only makes a hole in the
#: grid observable and measurable.
NOMINAL_BAR_NS = 3_600_000_000_000

#: Where an intra-bar liquidation touch came from, strongest first.
#:
#: Named rather than boolean because the two AUTHORISED tiers are not equally
#: strong. ``MARGIN_AND_LIQUIDATION.what_the_simulation_cannot_determine``
#: requires testing "against the hourly HIGH of the mark series where available
#: and the hourly close otherwise, and to RECORD which was used", so the record
#: has to distinguish them.
#:
#: **There is no authorised third tier, and that is why one is named here.**
#: ``Quote.liquidation_touch`` used to fall back once more, to the SPOT close,
#: when no mark series was present at all. ``MARK_PRICE_FALLBACK`` authorises a
#: spot substitution only "as the funding notional base", and
#: ``BASIS_DEFINITION.which_series_plays_which_role`` lists the liquidation test
#: as a SEPARATE use of the mark series without extending the substitution to it.
#: The frozen text therefore does not authorise a spot-close liquidation touch,
#: and it is anti-conservative — a spot close cannot see a perpetual mark spike.
#: The accessor now REFUSES instead, so a mark-less held bar fails closed rather
#: than being tested against a series the design never gave it.
#:
#: ``TOUCH_SPOT_CLOSE`` survives as vocabulary only. It is retained so the
#: provenance record keeps a stable shape and an already-written artifact stays
#: readable, and it is UNREACHABLE from a successful evaluation under the active
#: design: nothing can produce it without the refusal firing first. Whether a
#: mark-less month may be evaluated economically at all is a question for a
#: pre-economic amendment, and deleting the name would hide that the question
#: exists.
TOUCH_MARK_HIGH = "mark_high"
TOUCH_MARK_CLOSE = "mark_close"
TOUCH_SPOT_CLOSE = "spot_close"

#: Strongest first. The order is part of the contract: an artifact reader ranks
#: coverage by it, and a future event ledger copies these names verbatim. The
#: third name is carried for shape, never produced — see above.
TOUCH_SOURCES: tuple[str, ...] = (TOUCH_MARK_HIGH, TOUCH_MARK_CLOSE, TOUCH_SPOT_CLOSE)


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

    def _no_mark_series(self) -> "CarryError":
        """The refusal both liquidation accessors raise, built in ONE place.

        Two accessors answer one question — which series the touch came from, and
        what it was worth — so they must refuse on exactly the same condition. A
        second copy of this message is a second chance for the name and the
        number to disagree about whether a bar is testable at all.
        """
        return CarryError(
            f"no mark series at {self.instant_ns}: the liquidation test takes the mark "
            "HIGH where available and the mark CLOSE otherwise, and this bar carries "
            "neither. MARK_PRICE_FALLBACK authorises the spot close as the FUNDING "
            "NOTIONAL BASE alone, and BASIS_DEFINITION.which_series_plays_which_role "
            "keeps the liquidation test a SEPARATE use of the mark series without "
            "extending that substitution to it. Testing liquidation against the spot "
            "close would therefore be unauthorised, and it is anti-conservative: a spot "
            "close cannot see a perpetual mark spike. A held bar without a mark is "
            "INVALID rather than testable."
        )

    @property
    def liquidation_touch(self) -> Decimal:
        """The most adverse mark this bar can be shown to have reached.

        Adverse for a SHORT means HIGH, so the conservative test uses the mark
        high where the source provides it and falls back to the mark close. There
        is no third tier. A bar carrying neither is REFUSED, because the only
        series left is one the frozen design never authorised for this test.
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
        """WHICH series :attr:`liquidation_touch` came from, as a recordable name.

        The boolean above answers "was the strong touch available"; it cannot
        say which of the two authorised series answered when it was not, and a
        mark CLOSE — the venue's own notional price with the intra-bar extreme
        missing — is a materially weaker test than a mark HIGH. Persisting one
        name per test is what lets a later event ledger state coverage without
        reinterpreting a flag.

        Refuses on exactly the condition :attr:`liquidation_touch` refuses on, so
        a recorded name always has a value behind it and no artifact can claim a
        bar was tested against a series that was not there.
        """
        if self.mark_high is not None:
            return TOUCH_MARK_HIGH
        if self.mark is None:
            raise self._no_mark_series()
        return TOUCH_MARK_CLOSE

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

    def equity_at(self, spot_price: Decimal, perp_price: Decimal) -> Decimal:
        """Total portfolio equity marked at two given prices. Both legs, one pool.

        Split out from :meth:`equity` because the holding period BEGINS at an
        instant whose only knowable prices are the FILLS, not a candle close: at
        the entry instant this returns exactly ``total_capital - entry fees -
        entry slippage``, which is the equity invariant the accounting controls
        assert.
        """
        return (
            self.free_cash
            + self.quantity * spot_price
            + self.perp_margin
            + (self.perp_entry - perp_price) * self.quantity
        )

    def equity(self, quote: Quote) -> Decimal:
        """Total portfolio equity at ``quote``'s CLOSES — the end-of-bar mark."""
        return self.equity_at(quote.spot, quote.perp)


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
class LiquidationTouchProvenance:
    """How many of a block's liquidation tests used each touch source.

    **Counts, not a flag.** ``MARGIN_AND_LIQUIDATION`` requires recording which
    series the check used "so the check is never quietly weaker than it claims to
    be", and a single boolean cannot carry that for a block whose months differ:
    ``MARK_PRICE_FALLBACK`` is explicitly **per archive object**, so partial
    availability is the likely case and a block can legitimately mix all three
    sources. Counts survive that; a flag has to pick a lie.

    Structured this way so a future event ledger can copy the numbers verbatim
    rather than re-deriving coverage from a summary — which is the reinterpretation
    step ``§13`` exists to remove. It holds no threshold and nothing selects on it.
    """

    mark_high: int = 0
    mark_close: int = 0
    spot_close: int = 0

    def __post_init__(self) -> None:
        for name in TOUCH_SOURCES:
            if getattr(self, name) < 0:
                raise CarryError(f"{name} touch count is negative")

    @property
    def tested(self) -> int:
        """How many liquidation tests ran at all."""
        return self.mark_high + self.mark_close + self.spot_close

    @property
    def all_mark_high(self) -> bool:
        """Whether EVERY test used the strong intra-bar touch.

        False when no test ran, deliberately. A block that never opened has not
        established hourly-high coverage, and vacuous truth here would let the
        weakest possible evidence — none — read as the strongest.
        """
        return self.tested > 0 and self.mark_high == self.tested

    @property
    def used_a_weaker_fallback(self) -> bool:
        """Whether any test fell back off the mark high."""
        return self.mark_close > 0 or self.spot_close > 0

    def as_dict(self) -> dict[str, int]:
        """The artifact-facing form: the counts themselves, never a claim about them."""
        return {name: getattr(self, name) for name in TOUCH_SOURCES}


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
    #: The most negative ``equity_t - total_starting_capital`` over the holding
    #: period, in QUOTE UNITS. The absolute half of the pair, exactly as
    #: :attr:`net_pnl` is to :attr:`net_return`.
    max_adverse_excursion_pnl: Decimal
    #: The same excursion AS A FRACTION OF TOTAL CAPITAL, which is what
    #: ``VIABILITY_GATE.maximum_adverse_excursion`` defines it to be: "the most
    #: negative value of (equity_t - total_starting_capital) over the holding
    #: period, AS A FRACTION OF TOTAL CAPITAL — the same base as the block return,
    #: so the two are comparable". It was reported in quote units, so a reader
    #: comparing it against the -0.02 scale of G3 was out by the whole capital
    #: base. The frozen text settles the unit, so this conforms to the
    #: preregistration rather than amending it.
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
    #: Which series each of this block's liquidation tests actually used.
    #: ``Quote.liquidation_touch_is_high`` existed and was thrown away at the end
    #: of the loop, so a block checked entirely against weaker fallback closes was
    #: indistinguishable from one checked against real hourly mark highs. The
    #: frozen contract requires recording which was used; this is where it lands.
    liquidation_touch_provenance: LiquidationTouchProvenance = field(
        default_factory=LiquidationTouchProvenance
    )
    #: How many held intra-bar windows this block was exposed to — bars 0..N-1 of
    #: the quote series when the block closes normally, and 0..trigger when it is
    #: liquidated. Reported so the attribution window is auditable from the
    #: artifact rather than inferred from the code.
    held_bars: int = 0
    #: How many adjacent quote pairs are spaced further apart than one nominal
    #: bar, and — separately — the LARGEST adjacent spacing observed, gap or not.
    #: The second is deliberately not called a "max gap": on a contiguous block it
    #: is one nominal bar, and a field that reported a non-zero "gap" beside a gap
    #: count of zero would be a contradiction sitting in the evidence.
    #:
    #: A hole in the hourly grid is a fact about the block's sources — the frozen
    #: ``SOURCE_FREEZE_FIELDS`` already requires "gaps detected" of the
    #: acquisition — and recording it here too means a multi-hour jump, including
    #: one a forced close filled across, cannot look identical to an ordinary +1h
    #: transition.
    quote_gap_count: int = 0
    max_quote_step_ns: int = 0

    @property
    def net_funding(self) -> Decimal:
        return self.funding_received - self.funding_paid

    @property
    def forced_close_gap_ns(self) -> int | None:
        """How far a forced close's fill lay from the trigger that caused it.

        ``NOMINAL_BAR_NS`` on a contiguous grid. Anything larger means the
        following bar was not the following HOUR — the frozen rule fills at the
        next VALID executable observation, so a gap makes the forced close later
        than an hour without changing the rule, and that is worth seeing.
        """
        if self.liquidation_instant_ns is None or self.forced_close_instant_ns is None:
            return None
        return self.forced_close_instant_ns - self.liquidation_instant_ns


def unclosed_block_result(
    label: str,
    reason: str,
    *,
    settlements: int,
    quantity: Decimal,
    basis_entry: Decimal,
    funding_received: Decimal,
    funding_paid: Decimal,
    fees: Decimal,
    slippage: Decimal,
    max_adverse_excursion_pnl: Decimal,
    total_capital: Decimal,
    thin_sample: bool,
    liquidated: bool = False,
    liquidation_instant_ns: int | None = None,
    liquidation_touch_provenance: LiquidationTouchProvenance | None = None,
    held_bars: int = 0,
    quote_gap_count: int = 0,
    max_quote_step_ns: int = 0,
) -> BlockResult:
    """Amendment A1's encoding of an UNCLOSED block — ONE definition, BOTH causes.

    A1 names two: a liquidation trigger with no permitted following bar, and the
    case ``POSITION_LIFECYCLE.close_instant`` already named, where no valid
    instant exists at or before the block's last hour. A1 says explicitly that
    "defining UNCLOSED once, for both causes, is a narrower commitment than
    defining it only for the case that prompted the question", so it is defined
    once, here, rather than inline at the single site that used to reach it.

    That matters more after the holding-window repair than before it. Correct
    bars-0..N-1 attribution makes the LIQUIDATION cause unreachable from
    :func:`evaluate_block` — a trigger can only land on a bar that has a
    successor — so the surviving cause is the no-valid-exit one, which belongs to
    the block runner that does not exist yet. Leaving A1's only encoding buried in
    an unreachable branch would have left a frozen, hashed rule with no way to
    exercise it; this is the same rule, callable and testable, for whatever
    produces the other cause later.

    What is reported is exactly what A1 requires: the funding, fees and slippage
    actually incurred "as the facts they are", and the close-dependent
    quantities — exit basis, basis PnL, net PnL, net return — as
    :data:`NOT_DETERMINABLE`, never zero, so a gate that ignores
    :attr:`BlockResult.unclosed` raises instead of averaging in a number nobody
    measured.
    """
    return BlockResult(
        label=label,
        opened=True,
        reason=reason,
        settlements=settlements,
        quantity=quantity,
        basis_entry=basis_entry,
        basis_exit=NOT_DETERMINABLE,
        basis_pnl=NOT_DETERMINABLE,
        funding_received=funding_received,
        funding_paid=funding_paid,
        fees=fees,
        slippage=slippage,
        rebalance_cost=ZERO,
        net_pnl=NOT_DETERMINABLE,
        net_return=NOT_DETERMINABLE,
        liquidated=liquidated,
        max_adverse_excursion_pnl=max_adverse_excursion_pnl,
        # Derived here rather than accepted as a second argument, so the absolute
        # and the fraction cannot be handed over disagreeing with each other.
        max_adverse_excursion=max_adverse_excursion_pnl / total_capital,
        thin_sample=thin_sample,
        liquidation_instant_ns=liquidation_instant_ns,
        forced_close_instant_ns=None,
        unclosed=True,
        liquidation_touch_provenance=(
            liquidation_touch_provenance or LiquidationTouchProvenance()
        ),
        held_bars=held_bars,
        quote_gap_count=quote_gap_count,
        max_quote_step_ns=max_quote_step_ns,
    )


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
    """One position, opened at the first quote's OPEN and closed at the last's.

    Funding is applied only for settlements strictly inside the holding window —
    a settlement before the open or after the close is not this position's cash
    flow — and only ever at its own instant, which is the whole of the causality
    story for a strategy that takes no funding signal.

    **The held intra-bar windows are bars 0 .. N-1, and this is the correction
    that matters most.** The position opens at bar 0's OPEN, so it is exposed to
    the REMAINDER OF BAR 0 — its high, its low, its close. It closes at bar N's
    OPEN, so it is exposed to nothing after that open: bar N's own high and close
    happen to a position that no longer exists. This loop once ran over
    ``quotes[1:]``, which made the entry bar's intra-bar action invisible to the
    liquidation model and simultaneously tested the exit bar's post-open window
    against a closed position, in both directions at once. The same window
    governs the maximum adverse excursion, so no pre-entry or post-exit movement
    can reach it either.

    **Liquidation is a trigger and a fill, and they are different instants.** The
    trigger bar is the one whose adverse intra-bar touch crossed the threshold;
    per ``MARGIN_AND_LIQUIDATION.forced_close_price`` the forced close fills at
    the OPEN OF THE FOLLOWING BAR. Filling at the trigger bar's own open — which
    this function once did — executes at a price stamped before the high that
    caused the liquidation, so the fill would precede its own cause. The position
    stops accruing at the TRIGGER instant: a settlement after it belongs to a
    position that no longer existed, and is not applied even though the fill bar
    is later.

    **"Following bar" means the next VALID executable observation**, not the next
    exact hour. That is the frozen text's own resolution rather than a choice made
    here: ``forced_close_price`` calls the next open "the first price an operator
    could actually have transacted at", and all three lifecycle instants in
    ``POSITION_LIFECYCLE`` resolve an invalid grid point the same way — the open
    is "the FIRST hourly grid instant ... at which observations are all present
    and valid", and an invalid close moves to "the FIRST valid instant AT OR AFTER
    it". An operator cannot transact at a hole. ``quotes`` is therefore the series
    of VALID observations and the following bar is the next element of it; the gap
    it may span is measured and reported (:attr:`BlockResult.quote_gap_count`,
    :attr:`BlockResult.max_quote_step_ns`, :attr:`BlockResult.forced_close_gap_ns`)
    so a multi-hour jump can never look like an ordinary +1h transition.

    **Amendment A1 and what corrected attribution does to it.** A1 makes an
    opened-but-unclosable block UNCLOSED, with the close-dependent fields
    :data:`NOT_DETERMINABLE` and the screen INVALID. Under the corrected window a
    liquidation trigger can only fall on bars 0 .. N-1, so ``quotes[trigger + 1]``
    always exists and the LIQUIDATION route into A1 is unreachable from here — the
    apparent "liquidation on the final quote after its open" case was an artefact
    of the off-by-one, since it required testing a bar the position had already
    closed at the open of. The rule is unchanged and is NOT relaxed: A1 says it
    "applies equally to the other unclosed cause", the no-valid-exit-instant case
    ``POSITION_LIFECYCLE.close_instant`` names, which the block runner will
    produce and which this evaluator never sees. The branch below stays as the
    fail-closed encoding of A1 for any caller that reaches it.

    Timestamps are verified before any economic state exists: the causal order of
    this loop is read off the sequence, so a caller whose list order disagreed
    with its own instants would have had its causality invented by list position.
    """
    previous: Quote | None = None
    gap_count = 0
    max_gap = 0
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
        if previous is not None:
            step = quote.instant_ns - previous.instant_ns
            if step <= 0:
                # Strictly increasing and unique, checked BEFORE any economic
                # state exists. Everything below reads causality off this
                # sequence — which bar the position is exposed to, which
                # settlement has happened yet, which bar a forced close fills at
                # — so admitting a duplicate or an inversion would let the CALLER'S
                # LIST ORDER decide the causal order. A duplicate instant is also
                # exactly the ambiguity POSITION_LIFECYCLE.validity_definition
                # calls invalid, and it fails closed here rather than silently
                # double-counting one hour of exposure.
                raise CarryError(
                    f"quote timestamps are not strictly increasing: {quote.instant_ns} "
                    f"follows {previous.instant_ns} in the sequence. P13 reads causality "
                    "from the instants, never from the caller's list order, so a repeated "
                    "or out-of-order instant is INVALID rather than sorted, deduplicated "
                    "or trusted."
                )
            if step > NOMINAL_BAR_NS:
                gap_count += 1
            max_gap = max(max_gap, step)
        previous = quote
    for settlement in (settlements := list(settlements)):
        if settlement.instant_ns >= RESEARCH_BOUNDARY_NS:
            raise CarryError(
                f"funding settlement at {settlement.instant_ns} is at or after the research "
                f"boundary {DATA_BOUNDARY['span_end_exclusive']}"
            )

    # The not-opened template. Every field here is either a STRUCTURAL FACT about
    # a block that has no position — a quantity of zero, no settlement applied, no
    # fee charged, no liquidation — or NOT DETERMINABLE.
    #
    # The economic fields are NaN and this is the R2 repair. They were finite
    # zeroes, and a zero return is a MEASUREMENT: it enters G2's mean as a
    # perfectly ordinary block and G3's worst-block test as a block that lost
    # nothing, when in fact nothing was measured at all. VIABILITY_GATE
    # .excluded_blocks does exclude an unopened block from G1 and G2, so a correct
    # gate never reads these — and a gate that forgets now raises
    # InvalidOperation instead of quietly averaging in a number nobody observed.
    # The distinction is representational: no gate rule changes, the flattering
    # failure mode simply stops being available.
    empty = BlockResult(
        label=label,
        opened=False,
        reason="",
        settlements=0,
        quantity=ZERO,
        basis_entry=NOT_DETERMINABLE,
        basis_exit=NOT_DETERMINABLE,
        basis_pnl=NOT_DETERMINABLE,
        funding_received=ZERO,
        funding_paid=ZERO,
        fees=ZERO,
        slippage=ZERO,
        rebalance_cost=ZERO,
        net_pnl=NOT_DETERMINABLE,
        net_return=NOT_DETERMINABLE,
        liquidated=False,
        max_adverse_excursion_pnl=NOT_DETERMINABLE,
        max_adverse_excursion=NOT_DETERMINABLE,
        thin_sample=True,
        liquidation_touch_provenance=LiquidationTouchProvenance(),
        held_bars=0,
        quote_gap_count=gap_count,
        max_quote_step_ns=max_gap,
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

    # Deduplicated by settlement instant, per FUNDING_SEMANTICS.application: "a
    # redelivered or duplicated archive row changes nothing". That sentence is
    # about a row delivered TWICE, and two rows at one instant carrying DIFFERENT
    # rates are not that — they are the ambiguity
    # POSITION_LIFECYCLE.validity_definition calls invalid: "no duplicate row
    # makes the instant ambiguous. Anything else is invalid and fails closed".
    # Collapsing them silently let the CALLER'S LIST ORDER pick which rate the
    # payoff variable took, which is the defect R4 removed from the quote path
    # and which survived here.
    pending: dict[int, FundingSettlement] = {}
    for settlement in due:
        seen = pending.get(settlement.instant_ns)
        if seen is not None and (
            seen.rate != settlement.rate or seen.mark_price != settlement.mark_price
        ):
            raise CarryError(
                f"two funding settlements at {settlement.instant_ns} disagree: "
                f"rate {seen.rate} / mark {seen.mark_price} against rate {settlement.rate} "
                f"/ mark {settlement.mark_price}. A redelivered IDENTICAL row is "
                "deduplicated and changes nothing; a CONTRADICTORY one makes the instant "
                "ambiguous and is INVALID rather than resolved by whichever row the caller "
                "listed last."
            )
        pending[settlement.instant_ns] = settlement

    # The excursion is measured "over the HOLDING PERIOD"
    # (VIABILITY_GATE.maximum_adverse_excursion), which BEGINS at the entry
    # instant — so it is seeded there rather than at a floor of zero. Seeding at
    # ZERO made an opened block able to report an excursion of exactly 0, which
    # the frozen definition cannot produce: equity at the entry instant is
    # capital minus the two legs' entry frictions, so every opened block's
    # excursion is strictly negative whenever any friction is charged.
    worst = position.equity_at(entry.spot_fill, entry.perp_fill) - allocation.total_capital
    settled_count = 0
    traded_quantity = position.quantity
    trigger_index: int | None = None
    touches: dict[str, int] = {name: 0 for name in TOUCH_SOURCES}

    # Bars 0 .. N-1: every bar whose post-open action the position actually held
    # through. Bar N is excluded because the position closes at ITS OPEN, before
    # any of the rest of it happens.
    for index, quote in enumerate(quotes[:-1]):
        for instant in sorted(k for k in pending if k <= quote.instant_ns):
            apply_funding(position, pending.pop(instant))
            settled_count += 1
        worst = min(worst, position.equity(quote) - allocation.total_capital)
        touches[quote.liquidation_touch_source] += 1
        if is_liquidated(position, quote, venue, isolated):
            trigger_index = index
            break

    held_bars = len(quotes) - 1 if trigger_index is None else trigger_index + 1
    provenance = LiquidationTouchProvenance(**touches)

    liquidated = trigger_index is not None
    liquidation_instant: int | None = None
    forced_close_instant: int | None = None
    if trigger_index is None:
        # The settlements that fall after the last HELD bar's instant and at or
        # before the close. The boundary tie rule applies a settlement whose
        # instant EQUALS the close instant — the position was held through that
        # accrual window — and the held-bar loop stops one bar earlier, so they
        # are applied here rather than dropped. They cannot reach the excursion
        # above, because that is measured over marks the position lived through.
        for instant in sorted(pending):
            apply_funding(position, pending.pop(instant))
            settled_count += 1
        exit_quote = final
    else:
        # MARGIN_AND_LIQUIDATION.forced_close_price: the trigger is detected from
        # WITHIN-bar action an hourly grid cannot resolve, so the fill is the OPEN
        # OF THE FOLLOWING BAR. Filling at the trigger bar's own open would
        # execute at a price stamped BEFORE the high that caused the liquidation
        # — a fill that precedes its own cause.
        liquidation_instant = quotes[trigger_index].instant_ns
        if trigger_index + 1 >= len(quotes):
            # Amendment A1, kept as the fail-closed encoding of the rule.
            #
            # UNREACHABLE from the loop above: it iterates bars 0 .. N-1, so
            # trigger_index <= len(quotes) - 2 and the following bar always
            # exists. It was reachable only while the loop also tested bar N —
            # a bar the position had already closed at the open of — so the case
            # A1 was written against was in part an artefact of that off-by-one.
            # A1 itself is unchanged and still governs the OTHER unclosed cause it
            # names, the no-valid-exit-instant one, which the block runner will
            # produce and this function never sees. Nothing is invented and
            # nothing past the bound is read.
            return unclosed_block_result(
                label,
                (
                    "UNCLOSED: liquidation triggered at the final quote of the block, "
                    "and the preregistered forced-close fill is the OPEN of the "
                    "FOLLOWING bar, which lies outside the permitted region. Per "
                    "amendment A1 the block has no determinable return and the screen "
                    "is INVALID."
                ),
                settlements=settled_count,
                quantity=traded_quantity,
                basis_entry=entry.fill_basis,
                funding_received=position.funding_received,
                funding_paid=position.funding_paid,
                fees=position.fees,
                slippage=position.slippage,
                max_adverse_excursion_pnl=worst,
                total_capital=allocation.total_capital,
                thin_sample=settled_count < min_settlements,
                liquidated=True,
                liquidation_instant_ns=liquidation_instant,
                liquidation_touch_provenance=provenance,
                held_bars=held_bars,
                quote_gap_count=gap_count,
                max_quote_step_ns=max_gap,
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
    # The CLOSE INSTANT is inside the holding period — VIABILITY_GATE.block_net_pnl
    # is "equity AT THE BLOCK'S CLOSE minus total_starting_capital" — so the
    # realised close is an excursion sample like any other, and including it is
    # what makes `max_adverse_excursion <= net_return` hold identically rather
    # than by luck. Without it a block could report a drawdown shallower than the
    # loss it actually finished with, and a funding payment settling exactly at
    # the close instant (the frozen boundary tie rule applies it) reached the
    # result while never reaching the drawdown.
    worst = min(worst, net_pnl)
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
        max_adverse_excursion_pnl=worst,
        max_adverse_excursion=worst / allocation.total_capital,
        thin_sample=settled_count < min_settlements,
        liquidation_instant_ns=liquidation_instant,
        forced_close_instant_ns=forced_close_instant,
        liquidation_touch_provenance=provenance,
        held_bars=held_bars,
        quote_gap_count=gap_count,
        max_quote_step_ns=max_gap,
    )
