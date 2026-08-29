"""Position and order semantics for USD-M perpetual futures.

Two things in this module are load-bearing, and both exist because the obvious
shortcut is wrong.

*A position has a side, and a quantity that is never negative.* The cheap way to
support SHORT is to let quantity go negative and let arithmetic sort it out. It
does not sort it out: a "reduce by 3" applied to a position of 2 silently becomes
a SHORT of 1, and nothing in the arithmetic can tell that apart from an intended
reversal. Here :class:`PositionSide` is a first-class value, ``quantity`` is a
magnitude, and the invariant ``side is FLAT <=> quantity == 0`` is checked on
construction.

*Reaching a target position is planned, not arithmetic.* :func:`plan_transition`
turns (current, target) into an explicit list of :class:`OrderIntent`, and it
refuses to express a reversal as one order. A LONG that must become a SHORT is
two legs — close, then open — so no single order can ever carry a quantity larger
than the position it is reducing. That is the property that makes "a close cannot
reverse" structural rather than a comment.

The order lifecycle is an explicit state machine (:class:`OrderState`,
:data:`ALLOWED_TRANSITIONS`) rather than a set of booleans. Invalid transitions
raise; they are not clamped, logged or ignored. Repeated events are absorbed by
identity (:class:`OrderEvent.event_id`), so an exchange that redelivers a fill —
or a process that replays its own journal after a crash — cannot book the same
exposure twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from typing import Any, Iterable, Mapping

ZERO = Decimal("0")


class FuturesError(RuntimeError):
    """Base class for every refusal this package makes."""


class InvalidTransition(FuturesError):
    """An order was asked to move between two states that do not connect."""


class PositionError(FuturesError):
    """An operation would leave the position in a state that cannot be true."""


class PositionSide(str, Enum):
    """Which way a position is exposed. The string values are the wire format."""

    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        """+1 for LONG, -1 for SHORT, 0 for FLAT.

        Used for signed arithmetic (net exposure, funding direction) at the one
        place that needs it, so the sign never has to be inferred from a
        quantity that does not carry it.
        """
        return {PositionSide.LONG: 1, PositionSide.SHORT: -1, PositionSide.FLAT: 0}[self]

    @property
    def opposite(self) -> "PositionSide":
        if self is PositionSide.FLAT:
            return PositionSide.FLAT
        return PositionSide.SHORT if self is PositionSide.LONG else PositionSide.LONG


class OrderSide(str, Enum):
    """Which way an order trades. Not the same thing as a position side."""

    BUY = "BUY"
    SELL = "SELL"

    @classmethod
    def opening(cls, position_side: PositionSide) -> "OrderSide":
        """The order side that *creates or increases* ``position_side``."""
        if position_side is PositionSide.LONG:
            return cls.BUY
        if position_side is PositionSide.SHORT:
            return cls.SELL
        raise PositionError("FLAT is not a side an order can open")

    @classmethod
    def closing(cls, position_side: PositionSide) -> "OrderSide":
        """The order side that *reduces* ``position_side``."""
        return cls.opening(position_side.opposite)


class OrderPurpose(str, Enum):
    """Why an order exists. A bounded label, safe to put on a metric."""

    OPEN = "OPEN"
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    FLATTEN = "FLATTEN"

    @property
    def increases_exposure(self) -> bool:
        """Whether Aegis must approve this order before it may be submitted.

        Only exposure-*increasing* orders are gated. A reduction is the thing a
        halted account most needs to be able to do, and an entry gate that also
        blocked exits would turn a kill switch into a trap.
        """
        return self in (OrderPurpose.OPEN, OrderPurpose.INCREASE)


class OrderState(str, Enum):
    """Every state an order in this package can be in."""

    PLANNED = "PLANNED"
    RISK_APPROVED = "RISK_APPROVED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


#: States from which nothing further happens. An event addressed to one of these
#: is either a duplicate — absorbed by event id — or a bug, and the second is not
#: quietly turned into the first.
TERMINAL_STATES: frozenset[OrderState] = frozenset(
    {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.FAILED,
    }
)

#: The whole lifecycle, written out. A transition absent from this table raises
#: :class:`InvalidTransition`; there is no "unknown transitions are allowed"
#: branch, because the failure mode it would hide — an order that fills after it
#: was cancelled — is exactly the one that duplicates exposure.
ALLOWED_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.PLANNED: frozenset(
        {
            OrderState.RISK_APPROVED,
            OrderState.REJECTED,
            OrderState.CANCELLED,
            OrderState.FAILED,
        }
    ),
    OrderState.RISK_APPROVED: frozenset(
        {OrderState.SUBMITTED, OrderState.CANCELLED, OrderState.FAILED}
    ),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.REJECTED,
            OrderState.FAILED,
            OrderState.RECONCILIATION_REQUIRED,
        }
    ),
    OrderState.ACKNOWLEDGED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.FAILED,
            OrderState.RECONCILIATION_REQUIRED,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.FAILED,
            OrderState.RECONCILIATION_REQUIRED,
        }
    ),
    # Reached only by an explicit resolution, never by another exchange event:
    # the whole point of the state is that nothing automatic may act on the
    # position while local and reported disagree.
    OrderState.RECONCILIATION_REQUIRED: frozenset(
        {OrderState.CANCELLED, OrderState.FAILED, OrderState.FILLED}
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.FAILED: frozenset(),
}


def can_transition(current: OrderState, target: OrderState) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


@dataclass(frozen=True)
class Position:
    """One symbol's exposure. ``quantity`` is a magnitude, never a sign.

    ``entry_price`` is the volume-weighted average of the fills that built the
    current exposure. It is unchanged by a reduction — realising part of a
    position does not re-price the rest — and is reset to zero when the position
    goes flat.
    """

    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: Decimal = ZERO
    entry_price: Decimal = ZERO
    #: Isolated margin at exactly 1x is the whole of v1. Both are recorded rather
    #: than assumed so a stored position cannot be reinterpreted under a
    #: different margin regime by a later version that changed the default.
    leverage: Decimal = Decimal("1")
    margin_mode: str = "ISOLATED"

    def __post_init__(self) -> None:
        if self.quantity < ZERO:
            raise PositionError(
                f"{self.symbol}: quantity {self.quantity} is negative. SHORT is a side, "
                "not a negative number; a signed quantity cannot tell a reduction from "
                "a reversal."
            )
        if (self.side is PositionSide.FLAT) != (self.quantity == ZERO):
            raise PositionError(
                f"{self.symbol}: side {self.side.value} with quantity {self.quantity}. "
                "A flat position has no quantity and a non-flat one has some."
            )
        if self.leverage <= ZERO:
            raise PositionError(f"{self.symbol}: leverage {self.leverage} is not positive")
        if self.side is not PositionSide.FLAT and self.entry_price <= ZERO:
            raise PositionError(
                f"{self.symbol}: a {self.side.value} position of {self.quantity} has "
                f"entry_price {self.entry_price}. An open position was entered at some "
                "price; a zero entry makes `unrealised_pnl` report the whole notional as "
                "profit, and `liquidation_price` refuses the same position outright — so "
                "the two would disagree about whether it can exist at all."
            )

    @property
    def is_flat(self) -> bool:
        return self.side is PositionSide.FLAT

    @property
    def signed_quantity(self) -> Decimal:
        """Quantity with the position's sign. For net exposure and funding only."""
        return self.quantity * self.side.sign

    def notional(self, price: Decimal) -> Decimal:
        """Absolute notional at ``price``. Always non-negative."""
        return self.quantity * price

    def apply_fill(
        self, order_side: OrderSide, quantity: Decimal, price: Decimal
    ) -> tuple["Position", Decimal]:
        """This position after ``quantity`` filled at ``price``, and realised PnL.

        Refuses any fill that would carry the position through zero and out the
        other side. A close is a close: if a venue ever reports a reducing fill
        larger than the position it reduces, that is a reconciliation problem and
        not a licence to open the opposite exposure.
        """
        if quantity <= ZERO:
            raise PositionError(f"{self.symbol}: fill quantity {quantity} is not positive")
        if price <= ZERO:
            raise PositionError(f"{self.symbol}: fill price {price} is not positive")

        fill_side = PositionSide.LONG if order_side is OrderSide.BUY else PositionSide.SHORT

        if self.is_flat:
            return (
                replace(self, side=fill_side, quantity=quantity, entry_price=price),
                ZERO,
            )

        if fill_side is self.side:
            total = self.quantity + quantity
            weighted = (self.entry_price * self.quantity + price * quantity) / total
            return replace(self, quantity=total, entry_price=weighted), ZERO

        if quantity > self.quantity:
            raise PositionError(
                f"{self.symbol}: a reducing fill of {quantity} against a {self.side.value} "
                f"position of {self.quantity} would reverse it. Closing never flips a "
                "position; plan the open as its own order."
            )

        # Realised PnL is signed by the direction being closed, not by the fill.
        realised = (price - self.entry_price) * quantity * self.side.sign
        remaining = self.quantity - quantity
        if remaining == ZERO:
            return (
                replace(self, side=PositionSide.FLAT, quantity=ZERO, entry_price=ZERO),
                realised,
            )
        return replace(self, quantity=remaining), realised

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "leverage": str(self.leverage),
            "margin_mode": self.margin_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Position":
        return cls(
            symbol=str(data["symbol"]),
            side=PositionSide(str(data["side"])),
            quantity=Decimal(str(data["quantity"])),
            entry_price=Decimal(str(data["entry_price"])),
            leverage=Decimal(str(data.get("leverage", "1"))),
            margin_mode=str(data.get("margin_mode", "ISOLATED")),
        )


@dataclass(frozen=True)
class OrderIntent:
    """One order, before it has been sized against the venue's constraints.

    ``reduce_only`` is not decoration. It is the venue-level restatement of the
    invariant :meth:`Position.apply_fill` enforces locally, and every reducing
    intent this package plans carries it.
    """

    symbol: str
    side: OrderSide
    quantity: Decimal
    purpose: OrderPurpose
    reduce_only: bool
    #: The position side this order acts on — the one being opened, increased,
    #: reduced or closed. Recorded so a reduce can be checked against the
    #: position it claims to reduce rather than against whatever is there later.
    position_side: PositionSide

    def __post_init__(self) -> None:
        if self.quantity <= ZERO:
            raise PositionError(
                f"{self.symbol}: order quantity {self.quantity} is not positive"
            )
        if self.reduce_only != (not self.purpose.increases_exposure):
            raise PositionError(
                f"{self.symbol}: purpose {self.purpose.value} and reduce_only="
                f"{self.reduce_only} disagree about whether this order adds exposure"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "purpose": self.purpose.value,
            "reduce_only": self.reduce_only,
            "position_side": self.position_side.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OrderIntent":
        return cls(
            symbol=str(data["symbol"]),
            side=OrderSide(str(data["side"])),
            quantity=Decimal(str(data["quantity"])),
            purpose=OrderPurpose(str(data["purpose"])),
            reduce_only=bool(data["reduce_only"]),
            position_side=PositionSide(str(data["position_side"])),
        )


class EventKind(str, Enum):
    """What an :class:`OrderEvent` reports. A bounded metric label."""

    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class OrderEvent:
    """Something the venue says happened to an order.

    ``event_id`` is the idempotency key and it is required. Two deliveries of one
    fill share an id; two genuinely different fills do not. The simulator derives
    it from the order id and the fill sequence number, which is what a real venue
    also gives you, so the property being tested here is the same one that will
    hold against Binance.
    """

    event_id: str
    kind: EventKind
    quantity: Decimal = ZERO
    price: Decimal = ZERO
    fee: Decimal = ZERO
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.event_id:
            raise FuturesError(
                "an order event with no id cannot be deduplicated, and an event that "
                "cannot be deduplicated may be applied twice"
            )
        if self.kind in (EventKind.PARTIAL_FILL, EventKind.FILL):
            if self.quantity <= ZERO:
                raise FuturesError(f"{self.kind.value} event {self.event_id} fills nothing")
            if self.price <= ZERO:
                raise FuturesError(f"{self.kind.value} event {self.event_id} has no price")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "fee": str(self.fee),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OrderEvent":
        return cls(
            event_id=str(data["event_id"]),
            kind=EventKind(str(data["kind"])),
            quantity=Decimal(str(data.get("quantity", "0"))),
            price=Decimal(str(data.get("price", "0"))),
            fee=Decimal(str(data.get("fee", "0"))),
            reason=str(data.get("reason", "")),
        )


@dataclass
class OrderRecord:
    """An order and everything that has happened to it, in one mutable object.

    ``applied_events`` is the whole of the idempotency guarantee: an event whose
    id is already in it changes nothing at all — not the state, not the filled
    quantity, not the fee. It is persisted with the record, so the property
    survives a restart, which is the case it exists for.
    """

    order_id: str
    intent: OrderIntent
    state: OrderState = OrderState.PLANNED
    filled_quantity: Decimal = ZERO
    #: Volume-weighted average fill price so far. Zero while nothing has filled.
    average_price: Decimal = ZERO
    fees: Decimal = ZERO
    #: Set when a venue reported more fills than the order asked for. The record
    #: keeps the over-delivery visible rather than letting `remaining_quantity`
    #: go negative, which downstream sizing would read as a negative order.
    over_delivered: bool = False
    applied_events: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def remaining_quantity(self) -> Decimal:
        """What is still outstanding. Never negative.

        Clamped at zero rather than allowed to go negative, and
        :attr:`over_delivered` records that it was clamped. A venue that reports
        more fills than the order carried is a reconciliation problem — the
        record-level analogue of the reversal guard
        :meth:`Position.apply_fill` enforces — and a negative remainder read by
        any cancel-the-rest or resize path is a negative order size.
        """
        outstanding = self.intent.quantity - self.filled_quantity
        return outstanding if outstanding > ZERO else ZERO

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def book_fill(self, quantity: Decimal, price: Decimal, fee: Decimal) -> None:
        """Record one fill against this order, refusing an over-delivery.

        The one place ``filled_quantity`` moves, so the invariant
        ``filled_quantity <= intent.quantity`` has somewhere to live.
        """
        if quantity <= ZERO:
            raise PositionError(f"order {self.order_id}: fill quantity {quantity} <= 0")
        if self.filled_quantity + quantity > self.intent.quantity:
            self.over_delivered = True
            raise PositionError(
                f"order {self.order_id}: a fill of {quantity} on top of "
                f"{self.filled_quantity} would exceed the order's {self.intent.quantity}. "
                "A venue that over-delivers is a reconciliation problem, not a larger "
                "order."
            )
        total = self.filled_quantity + quantity
        self.average_price = (
            self.average_price * self.filled_quantity + price * quantity
        ) / total
        self.filled_quantity = total
        self.fees += fee

    def transition(self, target: OrderState, reason: str = "") -> None:
        """Move to ``target``, or raise saying which move was refused."""
        if not can_transition(self.state, target):
            raise InvalidTransition(
                f"order {self.order_id}: {self.state.value} -> {target.value} is not a "
                f"transition this order can make. Allowed: "
                f"{sorted(s.value for s in ALLOWED_TRANSITIONS[self.state])}"
            )
        self.history.append(f"{self.state.value}->{target.value}")
        self.state = target
        if reason:
            self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "intent": self.intent.to_dict(),
            "state": self.state.value,
            "filled_quantity": str(self.filled_quantity),
            "average_price": str(self.average_price),
            "fees": str(self.fees),
            "over_delivered": self.over_delivered,
            "applied_events": list(self.applied_events),
            "history": list(self.history),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OrderRecord":
        return cls(
            order_id=str(data["order_id"]),
            intent=OrderIntent.from_dict(data["intent"]),
            state=OrderState(str(data["state"])),
            filled_quantity=Decimal(str(data["filled_quantity"])),
            average_price=Decimal(str(data["average_price"])),
            fees=Decimal(str(data["fees"])),
            over_delivered=bool(data.get("over_delivered", False)),
            applied_events=[str(e) for e in data.get("applied_events", [])],
            history=[str(h) for h in data.get("history", [])],
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class TargetPosition:
    """What the strategy wants to hold, independent of what it holds now."""

    symbol: str
    side: PositionSide
    quantity: Decimal

    def __post_init__(self) -> None:
        if self.quantity < ZERO:
            raise PositionError(f"{self.symbol}: target quantity {self.quantity} is negative")
        if (self.side is PositionSide.FLAT) != (self.quantity == ZERO):
            raise PositionError(
                f"{self.symbol}: target side {self.side.value} with quantity "
                f"{self.quantity}; a flat target has no quantity"
            )

    @classmethod
    def flat(cls, symbol: str) -> "TargetPosition":
        return cls(symbol=symbol, side=PositionSide.FLAT, quantity=ZERO)


def plan_transition(current: Position, target: TargetPosition) -> list[OrderIntent]:
    """The orders that take ``current`` to ``target``, in the order to send them.

    Every one of the eight transitions v1 supports comes out of this one
    function, and a reversal comes out as **two** intents rather than one:

    ============================  =========================================
    transition                    intents
    ============================  =========================================
    flat -> LONG / SHORT          one OPEN
    increase LONG / SHORT         one INCREASE
    reduce LONG / SHORT           one REDUCE, reduce_only
    LONG / SHORT -> flat          one CLOSE, reduce_only
    LONG <-> SHORT                one CLOSE then one OPEN
    already at target             none
    ============================  =========================================

    The reversal case is the reason this returns a list. Expressing it as a
    single order of ``current.quantity + target.quantity`` is what every
    signed-quantity implementation does, and it is one arithmetic slip away from
    a close that overshoots into a new position. Two orders cannot overshoot:
    the first is reduce-only and exactly the size of what it closes.
    """
    if current.symbol != target.symbol:
        raise PositionError(
            f"cannot plan {current.symbol} against a target for {target.symbol}"
        )

    if current.side is target.side and current.quantity == target.quantity:
        return []

    if target.side is PositionSide.FLAT:
        return [
            OrderIntent(
                symbol=current.symbol,
                side=OrderSide.closing(current.side),
                quantity=current.quantity,
                purpose=OrderPurpose.CLOSE,
                reduce_only=True,
                position_side=current.side,
            )
        ]

    if current.is_flat:
        return [
            OrderIntent(
                symbol=target.symbol,
                side=OrderSide.opening(target.side),
                quantity=target.quantity,
                purpose=OrderPurpose.OPEN,
                reduce_only=False,
                position_side=target.side,
            )
        ]

    if current.side is target.side:
        delta = target.quantity - current.quantity
        if delta > ZERO:
            return [
                OrderIntent(
                    symbol=target.symbol,
                    side=OrderSide.opening(target.side),
                    quantity=delta,
                    purpose=OrderPurpose.INCREASE,
                    reduce_only=False,
                    position_side=target.side,
                )
            ]
        return [
            OrderIntent(
                symbol=target.symbol,
                side=OrderSide.closing(current.side),
                quantity=-delta,
                purpose=OrderPurpose.REDUCE,
                reduce_only=True,
                position_side=current.side,
            )
        ]

    # Reversal: close what is there, then open the other way. Never one order.
    return [
        OrderIntent(
            symbol=current.symbol,
            side=OrderSide.closing(current.side),
            quantity=current.quantity,
            purpose=OrderPurpose.CLOSE,
            reduce_only=True,
            position_side=current.side,
        ),
        OrderIntent(
            symbol=target.symbol,
            side=OrderSide.opening(target.side),
            quantity=target.quantity,
            purpose=OrderPurpose.OPEN,
            reduce_only=False,
            position_side=target.side,
        ),
    ]


def plan_flatten(current: Position) -> list[OrderIntent]:
    """The single reduce-only order that takes ``current`` to zero, or none.

    Separate from :func:`plan_transition` because the *purpose* differs and the
    purpose is what reaches the telemetry and the persisted reason. The quantity
    is exactly the position's own, so a flatten can no more reverse than a close
    can.
    """
    if current.is_flat:
        return []
    return [
        OrderIntent(
            symbol=current.symbol,
            side=OrderSide.closing(current.side),
            quantity=current.quantity,
            purpose=OrderPurpose.FLATTEN,
            reduce_only=True,
            position_side=current.side,
        )
    ]


def _priced(positions: Iterable[Position], prices: Mapping[str, Decimal]) -> list[Position]:
    """The positions, or a refusal naming the ones with no price.

    Fails closed. Skipping an unpriced position silently under-reports exposure,
    and it under-reports it by the most exactly when a symbol's feed is broken —
    which is the moment a risk check reading the number most needs it to be
    right. Everything else in this package refuses missing data; so does this.
    """
    held = [p for p in positions if not p.is_flat]
    missing = sorted({p.symbol for p in held if p.symbol not in prices})
    if missing:
        raise PositionError(
            f"no price for {missing}, which hold open positions. Dropping them would "
            "report an exposure smaller than the one that exists."
        )
    return held


def net_exposure(positions: Iterable[Position], prices: Mapping[str, Decimal]) -> Decimal:
    """Signed notional across positions: LONG adds, SHORT subtracts."""
    return sum(
        (p.signed_quantity * prices[p.symbol] for p in _priced(positions, prices)), ZERO
    )


def gross_exposure(positions: Iterable[Position], prices: Mapping[str, Decimal]) -> Decimal:
    """Absolute notional across positions. LONG and SHORT both add."""
    return sum((p.notional(prices[p.symbol]) for p in _priced(positions, prices)), ZERO)
