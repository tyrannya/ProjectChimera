"""Fills priced from the book that was recorded at the decision minute.

:class:`~chimera.futures.venue.DeterministicFillModel` prices a fill by moving a
*reference* price against the order by a fixed number of basis points. That is
the right model for the validation protocol, where the reference is a synthetic
mark and the property under test is reproducibility. It is the wrong model for a
run on recorded market data, because there the reference is a candle close, and
a candle close is a price one trade printed at rather than a price anyone could
have traded against. Filling a BUY at the close plus a few basis points assumes
the spread was narrower than it was, on precisely the minutes when it was not.

This module fills at the recorded top of book instead. A BUY crosses to the
recorded ask and a SELL to the recorded bid, and the configured slippage is
applied *on top of* that crossing and always adversely, so the spread is a cost
that was measured on the same snapshot the decision was made from rather than a
cost that was assumed.

Everything here fails closed, in three distinguishable ways:

*the book is not fresh.* A missing quote, or one older than
``max_quote_age_ns``, yields the :data:`NO_FRESH_QUOTE` rejection and no fill. A
stale book is not a cheap fill; it is an unknown one, and the size of the error
is exactly the price move nobody recorded. The clock that age is measured
against is the caller's, and the obligation that comes with that is spelled out
on :class:`RecordedQuoteFillModel`.

*the fill is too far from the price the decision was made at.* A fill further
from ``reference_price`` than ``max_reference_deviation_bps`` yields
:data:`QUOTE_REFERENCE_DIVERGENCE`. What that buys depends entirely on where the
caller's reference comes from, and it is worth being exact about which of the two
it is. Section 5.2 of the demo plan has the runner pass ``quote.mid()`` — the mid
of the very snapshot the fill was priced from — and the deviation is then
algebraically half the quoted spread plus the configured slippage and nothing
else, so the rule is a ceiling on how wide a book this model will cross. That is
worth having, but it is blind to a book that is internally consistent and simply
wrong: a tight snapshot from hours earlier passes it. The rule becomes a check
that two recorded series agree only when the reference comes from a *different*
series — the recorded mark close for the same minute, say — and a disagreement
then means one of the two is wrong, with no way here to tell which.

*the inputs are not a market.* A crossed, non-positive or non-``Decimal``
snapshot, a reference price that is not a positive number, and a setting outside
its own domain all raise :class:`~chimera.futures.venue.ConstraintError` instead
of pricing anything. Those are failures of the data path or of the caller, not
market conditions, and reporting them as an ordinary rejection would file a bug
under "the market refused us".

Deliberately absent, per section 5.4 of the demo plan: depth beyond the touch,
market impact, queue position, maker fills and latency between the decision and
the fill. A fill takes the whole quantity at the touch price whatever size was
recorded at that level — ``bid_qty`` and ``ask_qty`` are carried because they
are part of the snapshot and make that optimism auditable, not because anything
here sizes a fill against them. The slippage bps is the only compensation for
it, and a partial fill is reachable only through ``max_fill_ratio``, which is
fault injection rather than a depth model. No fee is computed here: the venue
charges ``taker_fee_rate`` on every fill, and v1 has no maker path for a rebate
to arrive through.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from chimera.futures.domain import ZERO, OrderIntent, OrderSide
from chimera.futures.venue import ConstraintError, FillPlan, SymbolConstraints

ONE = Decimal("1")
BPS = Decimal("10000")

#: The book was missing, or older than the model's freshness bound. Named rather
#: than spelled inline so the runner, the telemetry and the tests all compare
#: against one string.
NO_FRESH_QUOTE = "no_fresh_quote"

#: The book and the price the decision was made from disagree by more than the
#: configured tolerance.
QUOTE_REFERENCE_DIVERGENCE = "quote_reference_divergence"


def _positive_decimal(value: object, field: str, subject: str) -> Decimal:
    """``value`` as a positive, finite :class:`Decimal`, or a refusal.

    Three checks that have to happen in this order. A ``float`` has no
    ``is_finite`` method, so the type check has to come first or a price that
    arrived through JSON as a float would raise ``AttributeError`` from inside
    the arithmetic rather than a named refusal here. A NaN then has to be
    excluded before any ordered comparison, because ``Decimal('NaN') <= 0``
    signals :class:`decimal.InvalidOperation` rather than returning a bool, and
    that exception escapes every caller that catches
    :class:`~chimera.futures.venue.ConstraintError`.
    """
    if not isinstance(value, Decimal):
        raise ConstraintError(
            f"{subject}: {field} is {value!r} ({type(value).__name__}), not a Decimal. "
            "A price routed through float has already lost the precision the rest of "
            "this package is careful about."
        )
    if not value.is_finite():
        raise ConstraintError(
            f"{subject}: {field} is {value}, which is not a finite number. A venue "
            "quotes neither a NaN nor an infinity."
        )
    if value <= ZERO:
        raise ConstraintError(f"{subject}: {field} is {value}, which is not positive")
    return value


def _whole_ns(value: object, field: str, subject: str) -> None:
    """Refuse a clock that is not an exact whole number of nanoseconds.

    The ``float`` case is the one that matters. Around 1.7e18, where a
    nanosecond epoch sits, the spacing between representable floats is 256 ns, so
    an instant that arrived through ``json.load`` or through ``time.time() *
    1e9`` has already been rounded — and that rounding is enough to move an age
    across the freshness bound, in either direction, and so to change whether a
    book may be filled against at all. A module that refuses a ``float`` bid
    price to protect eight decimal places cannot then accept a ``float`` clock
    that decides a fail-closed verdict. ``bool`` is excluded for the same
    fail-closed reason: Python counts it as an ``int``, but ``True`` as an
    instant is a defect rather than one nanosecond past the epoch.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConstraintError(
            f"{subject}: {field} is {value!r} ({type(value).__name__}), not a whole "
            "number of nanoseconds"
        )


@dataclass(frozen=True)
class TopOfBook:
    """One recorded top-of-book snapshot: the two prices a taker could reach.

    Validated at construction, so a snapshot that exists is one a fill may be
    priced from and there is no second place that re-checks it. The rule is that
    both sides are positive, both sizes are positive, and ``bid`` is *strictly*
    below ``ask``.

    That last condition is the load-bearing one. A matching engine cannot leave a
    bid at or above an ask — the two would have traded — so a snapshot where they
    meet or cross was assembled from two different instants, or from a partial
    write, or from a parser reading the wrong field. Filling against it would
    produce a price that no book ever showed, and on a crossed book the fill
    would be *favourable*, which is the one direction a simulator must never err
    in. There is no repair for it either: which of the two sides is the wrong one
    is not knowable from the snapshot, so the snapshot is refused whole.

    ``instant_ns`` is the canonical time of the snapshot — the close of the
    minute it belongs to — and it is what
    :meth:`RecordedQuoteFillModel.plan` measures staleness against. Its *type* is
    checked here, for the same reason the prices' is: a float at nanosecond-epoch
    magnitude is quantized to 256 ns and can therefore decide the freshness
    boundary by itself. Its *value* is not checked, because a snapshot dated
    after the clock is an age question rather than a book question, and the model
    refuses it as an unfresh quote.
    """

    instant_ns: int
    bid: Decimal
    bid_qty: Decimal
    ask: Decimal
    ask_qty: Decimal

    def __post_init__(self) -> None:
        subject = f"top of book at {self.instant_ns}"
        _whole_ns(self.instant_ns, "instant_ns", subject)
        _positive_decimal(self.bid, "bid", subject)
        _positive_decimal(self.bid_qty, "bid_qty", subject)
        _positive_decimal(self.ask, "ask", subject)
        _positive_decimal(self.ask_qty, "ask_qty", subject)
        if self.bid >= self.ask:
            raise ConstraintError(
                f"{subject}: bid {self.bid} is not below ask {self.ask}. A book whose "
                "sides meet or cross is not a book a taker could have crossed; it is a "
                "snapshot assembled from two instants or from a bad read, and there is "
                "no way to tell which side is the wrong one."
            )

    def mid(self) -> Decimal:
        """The midpoint, which is what a decision is priced against.

        Not a price anything fills at — nothing in this module fills at the mid.
        It is the reference the recorded book is *checked against*, and the value
        the runner passes back in as ``reference_price``.
        """
        return (self.bid + self.ask) / Decimal("2")

    def spread_bps(self) -> Decimal:
        """The quoted spread in basis points of the mid.

        Diagnostic: the cost this model charges is the crossing itself, and this
        is the number that says how large that cost was. Inexact by construction
        — a ratio of two decimals rounds at the context precision — so it is for
        reporting and never for a fill price or a comparison that decides one.
        """
        return (self.ask - self.bid) / self.mid() * BPS


@dataclass
class RecordedQuoteFillModel:
    """A :class:`~chimera.futures.venue.FillModel` that fills at the recorded touch.

    Mutable, and deliberately so: the runner installs the decision minute's book
    with :meth:`set_quote` before each cycle, and the same model instance prices
    every leg of that minute. The fields are therefore checked in :meth:`plan`
    rather than at construction — a setting written directly onto the dataclass
    would bypass a constructor check, and "validated once, at a moment that has
    since passed" is not a guarantee.

    The clock is the caller's, and that is an obligation rather than a
    convenience. ``now_ns`` moves only when the caller moves it, so a runner that
    calls :meth:`set_quote` on the minutes a book arrived and does nothing on the
    minutes one did not leaves the previous book installed *together with* the
    previous clock: the age never grows, the freshness rule can never fire, and
    every later order fills against a snapshot of unbounded age while reporting a
    clean fill. **The caller must advance** ``now_ns`` **every decision cycle,
    whether or not a new book arrived.** That is why it is a plain public field
    and not only a :meth:`set_quote` argument — a cycle with no book moves the
    clock by assigning it, and the next order is then refused as unfresh, which
    is the outcome the rule exists to produce.

    ``slippage_bps``
        Adverse movement applied on top of crossing the spread. Zero is allowed
        and means the fill is exactly the touch; negative is refused, because a
        model that can improve on the book has a lower expected cost than a real
        venue and would flatter every strategy measured through it.

    ``max_quote_age_ns``
        How old the installed book may be. Exactly this old is fresh; one
        nanosecond older is not.

    ``max_reference_deviation_bps``
        How far the fill price may sit from ``reference_price`` before the order
        is refused rather than filled. Exactly this far is accepted; further is
        not.

    ``max_fill_ratio``
        Fault injection only. The largest share of an order one fill may take,
        which is how a partial fill — and therefore duplicate-event handling and
        restart recovery — is reachable without a test double. At the default of
        one, every order fills in a single event.
    """

    slippage_bps: Decimal = Decimal("2")
    max_quote_age_ns: int = 120_000_000_000
    max_reference_deviation_bps: Decimal = Decimal("50")
    max_fill_ratio: Decimal = Decimal("1")
    quote: TopOfBook | None = None
    now_ns: int = 0

    def set_quote(self, quote: TopOfBook, now_ns: int) -> None:
        """Install the book to price against, and the instant to measure it from.

        Both together: a book installed without moving ``now_ns`` forward would
        keep passing the freshness check against a clock that stopped, which is
        the failure this model's whole first rule exists to catch.

        The convention this method requires, and which the runner has to meet:
        ``now_ns`` is at or after ``quote.instant_ns``, and ``instant_ns`` is the
        *close* of the minute the snapshot belongs to. A caller that stamps the
        cycle with that minute's open instead hands the model a book from the
        future, and every order in the run is then refused as ``no_fresh_quote``
        — which reads to an operator as a broken feed rather than as the clock
        convention it is. Both fields are whole nanoseconds; the normalized
        minute files have to be converted at the feed boundary rather than passed
        through as floats.
        """
        self.quote = quote
        self.now_ns = now_ns

    def plan(
        self, intent: OrderIntent, reference_price: Decimal, constraints: SymbolConstraints
    ) -> FillPlan:
        """The fills this order produces against the installed book.

        The order of the checks is part of the contract. Settings and the
        reference price are validated first and raise, because a setting outside
        its domain and a missing reference price are defects in the caller; a
        rejection reason is a statement about the market, and turning a bug into
        one hides it in the venue's telemetry. That precedence is deliberate and
        it outranks section 5.2's no-quote rule: a caller that lost its reference
        price gets the exception even when no book is installed and the rule
        alone would have answered ``no_fresh_quote``. Freshness comes next, since
        a book that may not be used cannot be quoted from. Only then is a price
        computed, and the divergence guard is applied to the *fill* price rather
        than to the touch, because the fill price is the number that will be
        booked.
        """
        self._validate_settings()
        reference = _positive_decimal(reference_price, "reference_price", intent.symbol)

        quote = self.quote
        if quote is None:
            return FillPlan(fills=(), rejection=NO_FRESH_QUOTE)
        age_ns = self.now_ns - quote.instant_ns
        # A negative age is a book from the future: the clock went backwards, or
        # the wrong minute's snapshot was installed. It is not fresh evidence
        # either, and treating "age <= max" as the whole test would let it fill.
        if age_ns < 0 or age_ns > self.max_quote_age_ns:
            return FillPlan(fills=(), rejection=NO_FRESH_QUOTE)

        price = self._fill_price(intent.side, quote, constraints)
        # Cross-multiplied rather than divided. `abs(price - reference) /
        # reference` rounds at the context precision, and a boundary case is
        # decided by whichever way that rounding went; the products are exact.
        if abs(price - reference) * BPS > self.max_reference_deviation_bps * reference:
            return FillPlan(fills=(), rejection=QUOTE_REFERENCE_DIVERGENCE)

        return FillPlan(fills=self._chunk(intent.quantity, price, constraints))

    # ------------------------------------------------------------------
    def _validate_settings(self) -> None:
        """Refuse a configuration that would price a fill wrongly rather than not at all."""
        if not isinstance(self.slippage_bps, Decimal) or not self.slippage_bps.is_finite():
            raise ConstraintError(
                f"slippage_bps {self.slippage_bps!r} is not a finite Decimal"
            )
        if self.slippage_bps < ZERO:
            raise ConstraintError(
                f"slippage_bps {self.slippage_bps} is negative, which would fill an order "
                "better than the book it is filling against"
            )
        if self.slippage_bps >= BPS:
            raise ConstraintError(
                f"slippage_bps {self.slippage_bps} is a full 100% of the price or more, "
                "which drives a SELL fill to zero or below"
            )
        _whole_ns(self.max_quote_age_ns, "max_quote_age_ns", "fill model")
        if self.max_quote_age_ns < 0:
            raise ConstraintError(
                f"max_quote_age_ns {self.max_quote_age_ns!r} is not a non-negative "
                "whole number of nanoseconds"
            )
        # The clock itself, checked here rather than in `plan`'s arithmetic: a
        # `now_ns` of the wrong type raises TypeError from the subtraction, and a
        # NaN raises InvalidOperation from the comparison, neither of which is
        # the ConstraintError every caller of this package catches.
        _whole_ns(self.now_ns, "now_ns", "fill model")
        if (
            not isinstance(self.max_reference_deviation_bps, Decimal)
            or not self.max_reference_deviation_bps.is_finite()
            or self.max_reference_deviation_bps < ZERO
        ):
            raise ConstraintError(
                f"max_reference_deviation_bps {self.max_reference_deviation_bps!r} is not "
                "a non-negative finite Decimal"
            )
        # `is_finite` before the comparison, as above: `ZERO < Decimal("NaN")`
        # signals InvalidOperation rather than returning False, and that escapes
        # the executor's `except (ConstraintError, FuturesError)` and takes the
        # run down instead of recording one failed order.
        if (
            not isinstance(self.max_fill_ratio, Decimal)
            or not self.max_fill_ratio.is_finite()
            or not (ZERO < self.max_fill_ratio <= ONE)
        ):
            raise ConstraintError(f"max_fill_ratio {self.max_fill_ratio!r} is not in (0, 1]")

    def _fill_price(
        self, side: OrderSide, quote: TopOfBook, constraints: SymbolConstraints
    ) -> Decimal:
        """The touch on the side the order has to cross, moved against the order.

        The quantization is the venue's own, so the price this returns is one the
        venue would accept. It is then checked back against the touch, because
        rounding to the nearest tick can move a price *down*: a BUY whose
        quantized price lands below the ask it was supposed to cross has been
        filled better than the book it crossed, which is the one direction this
        package will not simulate, so it raises rather than handing back the
        cheaper price.

        That is an invariant about the fill, and deliberately not a claim to
        detect a book paired with the wrong symbol's constraints. A snapshot
        carries no symbol, so pairing the two is the caller's job — one venue and
        one fill model per leg. A book off this symbol's grid trips this check
        only in the direction where the rounding would have been favourable, and
        a book whose prices happen to sit on both grids does not trip it at all.
        """
        drift = self.slippage_bps / BPS
        if side is OrderSide.BUY:
            price = constraints.quantize_price(quote.ask * (ONE + drift))
            touch, direction = quote.ask, "below the ask"
        else:
            price = constraints.quantize_price(quote.bid * (ONE - drift))
            touch, direction = quote.bid, "above the bid"
        improved = price < touch if side is OrderSide.BUY else price > touch
        if improved:
            raise ConstraintError(
                f"{constraints.symbol}: a {side.value} priced from bid {quote.bid} / ask "
                f"{quote.ask} quantizes to {price}, which is {direction} it crossed. A "
                f"recorded book off the {constraints.tick_size} tick grid this symbol "
                "quotes on is not this symbol's book, and a fill better than the touch "
                "is not one this package will simulate."
            )
        return price

    def _chunk(
        self, quantity: Decimal, price: Decimal, constraints: SymbolConstraints
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        """Split one order into the fills it arrives as, exactly as the venue's own model does.

        The rule is
        :meth:`~chimera.futures.venue.DeterministicFillModel.plan`'s, reproduced
        rather than shared because the two models must split identically: a
        scenario replayed under one and then the other has to produce the same
        event sequence, or a difference in event handling would be
        indistinguishable from a difference in chunking. The trailing sliver
        matters for the same reason — a remainder smaller than one step cannot be
        its own fill, because the venue would refuse it for being off the step
        grid, so it is carried into the fill before it.
        """
        chunk = constraints.quantize_quantity(quantity * self.max_fill_ratio)
        if chunk <= ZERO:
            chunk = quantity

        fills: list[tuple[Decimal, Decimal]] = []
        remaining = quantity
        while remaining > ZERO:
            take = chunk if chunk < remaining else remaining
            if remaining - take > ZERO and remaining - take < constraints.step_size:
                take = remaining
            fills.append((take, price))
            remaining -= take
        return tuple(fills)
