"""Venue constraints, and the dry-run simulator that stands in for the exchange.

This module holds the *hard* facts a venue imposes — tick size, step size, the
minimum quantity, the minimum notional, whether the symbol trades at all — and
nothing discretionary. That separation is deliberate: :mod:`chimera.risk` is the
only place in this repository that decides whether a trade is a good idea, and a
second opinion living in the execution layer is exactly the failure this package
is not allowed to introduce. What is here refuses orders the venue would refuse
anyway.

Everything fails closed. A constraint that is missing, non-positive, or
inconsistent with its own precision is not defaulted, guessed or rounded past —
it stops order planning with an explanation. Silently substituting a plausible
tick size is how an order gets rejected in production for a reason nobody can
reconstruct, and how a position ends up a different size than the risk engine
approved.

**There is no live order path in this package, and adding one is not a
configuration change.** :class:`DryRunFuturesVenue` holds no credentials, opens
no socket and imports nothing that could; ``tests/test_futures_no_live_path.py``
asserts that about the package's source rather than about its intentions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import (
    ROUND_DOWN,
    ROUND_HALF_UP,
    Decimal,
    InvalidOperation,
    getcontext,
    localcontext,
)
from typing import Any, Mapping, Protocol

from chimera.futures.domain import (
    ZERO,
    EventKind,
    FuturesError,
    OrderEvent,
    OrderIntent,
    OrderSide,
    Position,
    PositionSide,
)

logger = logging.getLogger(__name__)


class ConstraintError(FuturesError):
    """A venue constraint is missing, contradictory, or would be violated."""


#: Symbol statuses this package will plan an order under. Anything else — HALT,
#: BREAK, SETTLING, PENDING_TRADING, or a status Binance adds tomorrow — is
#: refused, because "not on the list" and "not tradable" have to be the same
#: answer for the list to be worth having.
TRADABLE_STATUS = "TRADING"

#: Order types v1 knows how to simulate. MARKET only: a limit book needs a
#: resting-order model, and pretending to have one would put fake precision into
#: every fill this package reports.
SUPPORTED_ORDER_TYPES = frozenset({"MARKET"})

_REQUIRED_FIELDS = (
    "symbol",
    "status",
    "tick_size",
    "step_size",
    "quantity_precision",
    "price_precision",
    "min_quantity",
    "min_notional",
    "maintenance_margin_rate",
    "taker_fee_rate",
    "maker_fee_rate",
)


def _decimal(value: Any, field: str, symbol: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ConstraintError(
            f"{symbol}: constraint {field!r} is {value!r}, which is not a number. "
            "Venue metadata that cannot be read is not metadata that can be assumed."
        ) from exc
    # NaN and infinity parse. They are worse than unreadable: a NaN comparison
    # signals rather than returning False, and an infinity has exponent 'F', so
    # both escape the checks below as InvalidOperation or TypeError rather than
    # as a ConstraintError — past every caller that catches ConstraintError,
    # naming neither the symbol nor the field.
    if not parsed.is_finite():
        raise ConstraintError(
            f"{symbol}: constraint {field!r} is {value!r}, which is not a finite number. "
            "A venue does not quote NaN or an infinity, and neither is a value an order "
            "can be sized against."
        )
    return parsed


@dataclass(frozen=True)
class SymbolConstraints:
    """Everything about one symbol an order has to respect to be placeable.

    Constructed only through :meth:`from_dict`, which is where the fail-closed
    validation lives. A :class:`SymbolConstraints` that exists has already been
    checked; there is no second place that re-checks it, and no layer below that
    keeps its own copy of a tick size.
    """

    symbol: str
    status: str
    tick_size: Decimal
    step_size: Decimal
    quantity_precision: int
    price_precision: int
    min_quantity: Decimal
    min_notional: Decimal
    #: Binance's maintenance margin rate for the lowest notional tier. Required,
    #: because a liquidation price cannot be estimated without it and an
    #: unestimable liquidation price is not something Aegis may be handed as a
    #: number.
    maintenance_margin_rate: Decimal
    taker_fee_rate: Decimal
    maker_fee_rate: Decimal
    supported_order_types: frozenset[str] = frozenset({"MARKET"})
    supports_reduce_only: bool = True
    #: Binance USD-M is one-way or hedge mode; v1 is one-way, so both position
    #: sides are reachable but only one at a time.
    supported_position_sides: frozenset[str] = frozenset({"LONG", "SHORT"})

    @property
    def tradable(self) -> bool:
        return self.status == TRADABLE_STATUS

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SymbolConstraints":
        """Validate venue metadata, or refuse it. Never repairs it."""
        symbol = str(data.get("symbol", "<unnamed>"))
        missing = [f for f in _REQUIRED_FIELDS if data.get(f) is None]
        if missing:
            raise ConstraintError(
                f"{symbol}: venue metadata is missing {missing}. An order planned "
                "without them would be sized by guesswork; there is no default that "
                "is safer than refusing."
            )

        tick = _decimal(data["tick_size"], "tick_size", symbol)
        step = _decimal(data["step_size"], "step_size", symbol)
        min_qty = _decimal(data["min_quantity"], "min_quantity", symbol)
        min_notional = _decimal(data["min_notional"], "min_notional", symbol)
        mmr = _decimal(data["maintenance_margin_rate"], "maintenance_margin_rate", symbol)
        taker = _decimal(data["taker_fee_rate"], "taker_fee_rate", symbol)
        maker = _decimal(data["maker_fee_rate"], "maker_fee_rate", symbol)
        try:
            qty_precision = int(data["quantity_precision"])
            price_precision = int(data["price_precision"])
        except (TypeError, ValueError) as exc:
            raise ConstraintError(f"{symbol}: a precision field is not an integer") from exc

        if tick <= ZERO or step <= ZERO:
            raise ConstraintError(
                f"{symbol}: tick_size={tick} step_size={step}. A non-positive increment "
                "makes every price or quantity representable, which is the opposite of "
                "what these fields are for."
            )
        if min_qty <= ZERO:
            raise ConstraintError(f"{symbol}: min_quantity={min_qty} is not positive")
        if min_notional < ZERO:
            raise ConstraintError(f"{symbol}: min_notional={min_notional} is negative")
        if not (ZERO < mmr < Decimal("1")):
            raise ConstraintError(
                f"{symbol}: maintenance_margin_rate={mmr} is not a fraction in (0, 1)"
            )
        for name, rate in (("taker_fee_rate", taker), ("maker_fee_rate", maker)):
            if not (ZERO <= rate < Decimal("1")):
                raise ConstraintError(f"{symbol}: {name}={rate} is not a fraction in [0, 1)")
        if qty_precision < 0 or price_precision < 0:
            raise ConstraintError(f"{symbol}: a precision field is negative")

        # A step of 0.001 and a precision of 2 cannot both be true: the venue
        # would be quoting a quantity it says it will not accept. Contradictory
        # metadata is refused rather than reconciled, because either field could
        # be the wrong one and picking is a guess.
        if -step.normalize().as_tuple().exponent > qty_precision:
            raise ConstraintError(
                f"{symbol}: step_size={step} needs more decimals than "
                f"quantity_precision={qty_precision} allows"
            )
        if -tick.normalize().as_tuple().exponent > price_precision:
            raise ConstraintError(
                f"{symbol}: tick_size={tick} needs more decimals than "
                f"price_precision={price_precision} allows"
            )
        if min_qty % step != ZERO:
            raise ConstraintError(
                f"{symbol}: min_quantity={min_qty} is not a multiple of step_size={step}, "
                "so the smallest placeable order is not placeable"
            )

        order_types = frozenset(str(t) for t in data.get("supported_order_types", ["MARKET"]))
        unsupported = order_types - SUPPORTED_ORDER_TYPES
        if not order_types & SUPPORTED_ORDER_TYPES:
            raise ConstraintError(
                f"{symbol}: the venue supports {sorted(order_types)} and this package "
                f"can only simulate {sorted(SUPPORTED_ORDER_TYPES)}"
            )
        if unsupported:
            logger.info(
                "%s: ignoring venue order types this package does not simulate: %s",
                symbol,
                sorted(unsupported),
            )

        return cls(
            symbol=symbol,
            status=str(data["status"]),
            tick_size=tick,
            step_size=step,
            quantity_precision=qty_precision,
            price_precision=price_precision,
            min_quantity=min_qty,
            min_notional=min_notional,
            maintenance_margin_rate=mmr,
            taker_fee_rate=taker,
            maker_fee_rate=maker,
            supported_order_types=order_types & SUPPORTED_ORDER_TYPES,
            supports_reduce_only=bool(data.get("supports_reduce_only", True)),
            supported_position_sides=frozenset(
                str(s) for s in data.get("supported_position_sides", ["LONG", "SHORT"])
            ),
        )

    # ------------------------------------------------------------------
    def quantize_quantity(self, quantity: Decimal) -> Decimal:
        """``quantity`` rounded **down** to the step size.

        Down, never to nearest. Rounding up would hand the venue an order larger
        than the one the risk engine approved, and "larger by one step" is still
        larger than the risk envelope says.

        The division runs in a widened context. At the default 28 significant
        digits a quantity with more digits than that divides to a value one ulp
        *above* an integer, and ``ROUND_DOWN`` then floors to the next step up —
        rounding the order up, which is the one direction this method exists to
        prevent. Nothing in the package produces such a quantity today; the guard
        is here because the failure would be silent and in the wrong direction.
        """
        if quantity <= ZERO:
            return ZERO
        with localcontext() as context:
            context.prec = max(getcontext().prec, len(quantity.as_tuple().digits) + 10)
            steps = (quantity / self.step_size).to_integral_value(rounding=ROUND_DOWN)
        return steps * self.step_size

    def quantize_price(self, price: Decimal) -> Decimal:
        """``price`` rounded to the nearest tick."""
        if price <= ZERO:
            raise ConstraintError(f"{self.symbol}: price {price} is not positive")
        return (price / self.tick_size).to_integral_value(
            rounding=ROUND_HALF_UP
        ) * self.tick_size

    def check_on_grid(self, quantity: Decimal, price: Decimal) -> None:
        """Raise unless a quantity and price land on the venue's own increments.

        The half of :meth:`check_placeable` that is about *representability*
        rather than about a minimum. A simulated fill is checked against this and
        not against the full filter, because ``min_quantity`` and ``min_notional``
        are per-**order** filters: applying them to each fill chunk would let a
        simulator setting — how finely an order is split — manufacture a venue
        rejection that no venue would produce.
        """
        if quantity <= ZERO:
            raise ConstraintError(f"{self.symbol}: quantity {quantity} is not positive")
        if quantity % self.step_size != ZERO:
            raise ConstraintError(
                f"{self.symbol}: quantity {quantity} is not a multiple of step_size "
                f"{self.step_size}"
            )
        if price % self.tick_size != ZERO:
            raise ConstraintError(
                f"{self.symbol}: price {price} is not a multiple of tick_size "
                f"{self.tick_size}"
            )

    def check_placeable(self, quantity: Decimal, price: Decimal, *, reduce_only: bool) -> None:
        """Raise unless an order of this size at this price could be placed.

        The minimum-notional check is skipped for a reduce-only order, and only
        for a reduce-only order: Binance exempts closes from the minimum for the
        obvious reason that a dust position would otherwise be unclosable. That
        exemption is a venue fact, not a discretionary softening, which is why it
        lives here.
        """
        if not self.tradable:
            raise ConstraintError(
                f"{self.symbol}: status is {self.status!r}, not {TRADABLE_STATUS!r}"
            )
        if reduce_only and not self.supports_reduce_only:
            raise ConstraintError(
                f"{self.symbol}: the venue does not accept reduce-only orders, so a "
                "close cannot be guaranteed not to reverse"
            )
        self.check_on_grid(quantity, price)
        if quantity < self.min_quantity:
            raise ConstraintError(
                f"{self.symbol}: quantity {quantity} is below min_quantity "
                f"{self.min_quantity}"
            )
        notional = quantity * price
        if not reduce_only and notional < self.min_notional:
            raise ConstraintError(
                f"{self.symbol}: notional {notional} is below min_notional "
                f"{self.min_notional}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "tick_size": str(self.tick_size),
            "step_size": str(self.step_size),
            "quantity_precision": self.quantity_precision,
            "price_precision": self.price_precision,
            "min_quantity": str(self.min_quantity),
            "min_notional": str(self.min_notional),
            "maintenance_margin_rate": str(self.maintenance_margin_rate),
            "taker_fee_rate": str(self.taker_fee_rate),
            "maker_fee_rate": str(self.maker_fee_rate),
            "supported_order_types": sorted(self.supported_order_types),
            "supports_reduce_only": self.supports_reduce_only,
            "supported_position_sides": sorted(self.supported_position_sides),
        }


class ConstraintSource(Protocol):
    """Where :class:`SymbolConstraints` come from. One per venue, not per layer."""

    def constraints(self, symbol: str) -> SymbolConstraints:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class StaticConstraintSource:
    """Constraints from a mapping — a config file, a fixture, a probe's output.

    v1 reads no exchange metadata endpoint. The values are declared, committed
    and reviewable, which for a dry-run package is strictly better than a live
    fetch: an order planned here is planned against exactly the numbers a reader
    can see.
    """

    table: Mapping[str, SymbolConstraints]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Mapping[str, Any]]) -> "StaticConstraintSource":
        return cls(
            table={
                symbol: SymbolConstraints.from_dict({**fields, "symbol": symbol})
                for symbol, fields in data.items()
            }
        )

    def constraints(self, symbol: str) -> SymbolConstraints:
        try:
            return self.table[symbol]
        except KeyError:
            raise ConstraintError(
                f"no venue metadata for {symbol!r}; known symbols: {sorted(self.table)}. "
                "An unknown symbol is not a tradable one."
            ) from None


# --- simulated fills -------------------------------------------------------


@dataclass(frozen=True)
class FillPlan:
    """The fills a model says an order produces, before any of them is applied."""

    fills: tuple[tuple[Decimal, Decimal], ...]
    #: Set when the model declines the order outright. The order is then REJECTED
    #: rather than partially filled, and the reason reaches the telemetry.
    rejection: str = ""

    @property
    def filled_quantity(self) -> Decimal:
        return sum((q for q, _ in self.fills), ZERO)


class FillModel(Protocol):
    """How a simulated order fills. Replaceable; the executor holds one."""

    def plan(
        self, intent: OrderIntent, reference_price: Decimal, constraints: SymbolConstraints
    ) -> FillPlan:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class DeterministicFillModel:
    """Adverse, deterministic, and never better than the price that was asked for.

    Three properties, each chosen because its opposite is the thing that makes a
    simulation flatter a strategy:

    *slippage is always against the order.* A BUY fills above the reference and a
    SELL below it, by ``slippage_bps``. There is no distribution and no chance of
    a favourable fill; a model that sometimes improves the price is a model whose
    expected cost is lower than a real venue's.

    *fills can be partial.* ``max_fill_ratio`` caps how much of an order one fill
    takes, so an order larger than that arrives as a sequence of events. That is
    the case duplicate-event handling and restart recovery are actually about, so
    it has to be reachable without a special test double.

    *nothing is random.* Same inputs, same fills, forever — which is what makes
    the dry-run validation protocol a check rather than a sample.
    """

    slippage_bps: Decimal = Decimal("5")
    max_fill_ratio: Decimal = Decimal("1")
    #: Below this, an order is rejected outright rather than filled. Models a
    #: venue that will not accept dust; set to zero to disable.
    reject_below_quantity: Decimal = ZERO

    def plan(
        self, intent: OrderIntent, reference_price: Decimal, constraints: SymbolConstraints
    ) -> FillPlan:
        if reference_price <= ZERO:
            raise ConstraintError(
                f"{intent.symbol}: no reference price to simulate a fill against"
            )
        if self.reject_below_quantity > ZERO and intent.quantity < self.reject_below_quantity:
            return FillPlan(fills=(), rejection="below the venue's simulated minimum")

        drift = self.slippage_bps / Decimal("10000")
        signed = drift if intent.side is OrderSide.BUY else -drift
        price = constraints.quantize_price(reference_price * (Decimal("1") + signed))

        ratio = self.max_fill_ratio
        if not (ZERO < ratio <= Decimal("1")):
            raise ConstraintError(f"max_fill_ratio {ratio} is not in (0, 1]")

        chunk = constraints.quantize_quantity(intent.quantity * ratio)
        if chunk <= ZERO:
            chunk = intent.quantity

        fills: list[tuple[Decimal, Decimal]] = []
        remaining = intent.quantity
        while remaining > ZERO:
            take = chunk if chunk < remaining else remaining
            # A trailing sliver smaller than a step cannot be its own fill.
            if remaining - take > ZERO and remaining - take < constraints.step_size:
                take = remaining
            fills.append((take, price))
            remaining -= take
        return FillPlan(fills=tuple(fills))


@dataclass
class DryRunFuturesVenue:
    """A Binance USD-M perpetual venue that exists only in this process.

    It holds the simulated exchange-side position — the thing reconciliation
    compares local state *against* — and it is the only object in this package
    that ever "reports" a position. It has no client, no key, and no method that
    reaches a network, and that is asserted by test rather than promised here.

    ``submit`` returns the events an exchange would deliver. It does not apply
    them: applying is :mod:`chimera.futures.executor`'s job, and keeping the two
    apart is what lets the executor be handed the same event twice.
    """

    source: ConstraintSource
    fill_model: FillModel
    #: The venue's own view of each position, keyed by symbol. Reconciliation
    #: reads this; nothing else writes it.
    positions: dict[str, Position] = None  # type: ignore[assignment]
    _sequence: int = 0

    def __post_init__(self) -> None:
        if self.positions is None:
            self.positions = {}

    def constraints(self, symbol: str) -> SymbolConstraints:
        return self.source.constraints(symbol)

    def reported_position(self, symbol: str) -> Position:
        return self.positions.get(symbol, Position(symbol=symbol))

    def submit(
        self, order_id: str, intent: OrderIntent, reference_price: Decimal
    ) -> list[OrderEvent]:
        """Simulate the venue's response, as the sequence of events it would send.

        The venue applies the fills to *its own* position as it reports them, so a
        caller that drops an event ends up with local and reported state that
        disagree — which is the situation reconciliation exists for, and which
        would be untestable if this method mutated the caller's state too.
        """
        constraints = self.constraints(intent.symbol)
        plan = self.fill_model.plan(intent, reference_price, constraints)
        events = [OrderEvent(event_id=self._next_id(order_id), kind=EventKind.ACKNOWLEDGED)]
        if plan.rejection:
            events.append(
                OrderEvent(
                    event_id=self._next_id(order_id),
                    kind=EventKind.REJECTED,
                    reason=plan.rejection,
                )
            )
            return events

        if not plan.fills:
            # Acknowledged and nothing filled — a resting order, or a model that
            # declines without rejecting. There is nothing to price the venue
            # filters against, and an order that moved no exposure cannot have
            # violated a minimum.
            return events

        # The whole order is checked against the venue's filters once. Each fill
        # chunk is then checked only for grid conformance: min_quantity and
        # min_notional are per-order filters, and applying them per chunk would
        # let `max_fill_ratio` — a simulator setting — refuse an order the venue
        # would have accepted.
        constraints.check_placeable(
            intent.quantity, plan.fills[0][1], reduce_only=intent.reduce_only
        )
        position = self.reported_position(intent.symbol)
        if intent.reduce_only:
            # `reduce_only` is the venue-level restatement of the invariant
            # `Position.apply_fill` enforces locally, and the only venue in this
            # package has to model it or the guarantee is one-sided: whenever the
            # venue's view differs from local in the direction that matters, a
            # close would have opened or grown the opposite exposure *here*, and
            # only a later reconcile would have noticed.
            fill_side = (
                PositionSide.LONG if intent.side is OrderSide.BUY else PositionSide.SHORT
            )
            if position.is_flat or position.side is fill_side:
                events.append(
                    OrderEvent(
                        event_id=self._next_id(order_id),
                        kind=EventKind.REJECTED,
                        reason=(
                            "reduce-only order would open or increase the position: the "
                            f"venue holds {position.side.value} {position.quantity}"
                        ),
                    )
                )
                return events
            if intent.quantity > position.quantity:
                events.append(
                    OrderEvent(
                        event_id=self._next_id(order_id),
                        kind=EventKind.REJECTED,
                        reason=(
                            f"reduce-only order for {intent.quantity} exceeds the "
                            f"{position.quantity} the venue holds"
                        ),
                    )
                )
                return events
        remaining = intent.quantity
        for quantity, price in plan.fills:
            constraints.check_on_grid(quantity, price)
            position, _ = position.apply_fill(intent.side, quantity, price)
            remaining -= quantity
            fee = (quantity * price * constraints.taker_fee_rate).quantize(
                Decimal("0.00000001")
            )
            events.append(
                OrderEvent(
                    event_id=self._next_id(order_id),
                    kind=EventKind.FILL if remaining <= ZERO else EventKind.PARTIAL_FILL,
                    quantity=quantity,
                    price=price,
                    fee=fee,
                )
            )
        self.positions[intent.symbol] = position
        return events

    def apply_settlement(self, symbol: str, position: Position) -> None:
        """Force the venue's view, for tests that need local and reported to differ.

        Named for what it is. Production code has no reason to call it, and a
        reconciliation mismatch that the executor could create for itself would
        not be a mismatch.
        """
        self.positions[symbol] = position

    def _next_id(self, order_id: str) -> str:
        self._sequence += 1
        return f"{order_id}:{self._sequence}"


def default_constraints_table() -> dict[str, dict[str, Any]]:
    """Binance USD-M BTCUSDT, as published, for the dry-run protocol.

    Committed values rather than a live fetch: this is what the validation
    protocol runs against, so it has to be a thing a reviewer can read and a
    later run can reproduce. The numbers are Binance's published BTCUSDT
    perpetual filters and the tier-1 maintenance margin rate; the fee rates are
    the standard USD-M taker and maker rates, which are also the rates
    ``chimera.contracts.TargetSpec`` assumes on the research side.
    """
    return {
        "BTC/USDT:USDT": {
            "status": TRADABLE_STATUS,
            "tick_size": "0.10",
            "step_size": "0.001",
            "quantity_precision": 3,
            "price_precision": 2,
            "min_quantity": "0.001",
            "min_notional": "100",
            "maintenance_margin_rate": "0.004",
            "taker_fee_rate": "0.0005",
            "maker_fee_rate": "0.0002",
            "supported_order_types": ["MARKET"],
            "supports_reduce_only": True,
            "supported_position_sides": ["LONG", "SHORT"],
        }
    }


def load_constraint_source(
    table: Mapping[str, Mapping[str, Any]] | None = None,
) -> StaticConstraintSource:
    return StaticConstraintSource.from_mapping(table or default_constraints_table())
