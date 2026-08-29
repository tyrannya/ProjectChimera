"""Futures cash flows: trading fees, funding, realised PnL, and margin.

Three cash flows reach a perpetual futures account and they are **not** the same
quantity with different signs:

*trading fees* are always a cost. They are paid on every fill, they never come
back, and they are recorded as a positive magnitude in :attr:`Ledger.trading_fees`.

*funding paid* and *funding received* are two different events, and collapsing
them into one net number loses the thing an operator actually wants to know. A
strategy that pays 12 and receives 10 is not the same as one that pays 2 and
receives nothing, even though both net to −2: the first is running a position the
market is charging it to hold. They are therefore two fields and two counters.

*realised PnL* is booked when a position is reduced, by
:meth:`chimera.futures.domain.Position.apply_fill`, and only then. Unrealised PnL
is derived on demand from a mark price and is never accumulated.

**Funding is an execution cash flow here, not information.** Nothing in this
module may be read by a feature, a label or a model. P4 asked whether funding
*predicts* and answered no; this module is about what funding *costs*, which is a
different question that a negative P4 does not touch.

The funding sign convention, written once so the two places that need it cannot
disagree:

======  =============  ==========================================
side    funding rate   the position holder
======  =============  ==========================================
LONG    positive       **pays** longs-pay-shorts
LONG    negative       **receives**
SHORT   positive       **receives**
SHORT   negative       **pays**
======  =============  ==========================================

which is exactly ``cash_flow = -sign(side) * notional * rate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from chimera.futures.domain import ZERO, FuturesError, Position, PositionSide
from chimera.futures.venue import SymbolConstraints

ONE = Decimal("1")


class AccountingError(FuturesError):
    """A cash flow or a margin figure cannot be computed from what is known."""


@dataclass(frozen=True)
class FundingEvent:
    """One 8-hour funding settlement, as it applies to one position."""

    symbol: str
    rate: Decimal
    mark_price: Decimal
    #: The venue's own settlement id. Funding is deduplicated by it for the same
    #: reason fills are: a redelivered settlement must not be charged twice.
    settlement_id: str

    def __post_init__(self) -> None:
        if not self.settlement_id:
            raise AccountingError("a funding settlement with no id cannot be deduplicated")
        if self.mark_price <= ZERO:
            raise AccountingError(f"{self.symbol}: funding mark price {self.mark_price} <= 0")


def funding_cash_flow(position: Position, event: FundingEvent) -> Decimal:
    """Signed cash flow for ``position`` at ``event``. Negative means paid out.

    A flat position pays and receives nothing, which is stated rather than
    reached by multiplying by a zero sign, so that the answer for FLAT does not
    depend on :attr:`PositionSide.sign` staying zero.

    The symbol is checked **before** the flat short-circuit, and the order
    matters. With the checks the other way round, a settlement for the wrong
    symbol quietly returned zero whenever the position happened to be flat — and
    :meth:`Ledger.book_funding` then recorded that foreign id in
    ``applied_funding``, so the *genuine* settlement carrying the same id was
    later deduplicated away. Ids are venue-scoped and one ledger spans every
    symbol, so a venue numbering settlements per symbol would cross-suppress real
    funding between them.
    """
    if position.symbol != event.symbol:
        raise AccountingError(
            f"funding for {event.symbol} cannot be applied to a {position.symbol} position"
        )
    if position.is_flat:
        return ZERO
    notional = position.notional(event.mark_price)
    return -Decimal(position.side.sign) * notional * event.rate


@dataclass
class Ledger:
    """Every futures cash flow this process has booked, kept apart by kind.

    All four accumulators are non-negative magnitudes except
    :attr:`realised_pnl`, which is genuinely signed. Netting is done by the
    reader, at the point of reading, so nothing here can quietly turn a cost into
    a smaller gain.
    """

    trading_fees: Decimal = ZERO
    funding_paid: Decimal = ZERO
    funding_received: Decimal = ZERO
    realised_pnl: Decimal = ZERO
    turnover: Decimal = ZERO
    #: Settlement ids already booked. Persisted, so a restart cannot re-charge
    #: funding the account has already paid.
    applied_funding: list[str] = field(default_factory=list)

    @property
    def net_funding(self) -> Decimal:
        """Received minus paid. Positive means the account was paid to hold."""
        return self.funding_received - self.funding_paid

    @property
    def net_pnl(self) -> Decimal:
        """Realised PnL after trading fees and net funding."""
        return self.realised_pnl - self.trading_fees + self.net_funding

    def book_fee(self, fee: Decimal) -> None:
        if fee < ZERO:
            raise AccountingError(
                f"a trading fee of {fee} is a rebate, and this package has no maker "
                "rebate path; a negative fee here is a sign error upstream"
            )
        self.trading_fees += fee

    def book_turnover(self, notional: Decimal) -> None:
        if notional < ZERO:
            raise AccountingError(f"turnover {notional} is negative")
        self.turnover += notional

    def book_realised(self, pnl: Decimal) -> None:
        self.realised_pnl += pnl

    def book_funding(self, position: Position, event: FundingEvent) -> Decimal:
        """Charge or credit one settlement. Returns the signed flow, 0 if repeated.

        Idempotent by ``settlement_id``: a second delivery of the same settlement
        changes nothing at all, which is the same guarantee
        :class:`chimera.futures.domain.OrderRecord` gives for fills and for the
        same reason.
        """
        # Validated before the dedup check, not after: an id recorded for a
        # settlement that was never applicable is an id that will silently
        # swallow the real one.
        flow = funding_cash_flow(position, event)
        if event.settlement_id in self.applied_funding:
            return ZERO
        self.applied_funding.append(event.settlement_id)
        if flow < ZERO:
            self.funding_paid += -flow
        else:
            self.funding_received += flow
        return flow

    def to_dict(self) -> dict[str, Any]:
        return {
            "trading_fees": str(self.trading_fees),
            "funding_paid": str(self.funding_paid),
            "funding_received": str(self.funding_received),
            "realised_pnl": str(self.realised_pnl),
            "turnover": str(self.turnover),
            "applied_funding": list(self.applied_funding),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Ledger":
        return cls(
            trading_fees=Decimal(str(data.get("trading_fees", "0"))),
            funding_paid=Decimal(str(data.get("funding_paid", "0"))),
            funding_received=Decimal(str(data.get("funding_received", "0"))),
            realised_pnl=Decimal(str(data.get("realised_pnl", "0"))),
            turnover=Decimal(str(data.get("turnover", "0"))),
            applied_funding=[str(s) for s in data.get("applied_funding", [])],
        )


def unrealised_pnl(position: Position, mark_price: Decimal) -> Decimal:
    """Mark-to-market on an open position. Derived, never accumulated."""
    if position.is_flat:
        return ZERO
    if mark_price <= ZERO:
        raise AccountingError(f"{position.symbol}: mark price {mark_price} is not positive")
    return (mark_price - position.entry_price) * position.quantity * position.side.sign


@dataclass(frozen=True)
class MarginState:
    """What Aegis needs to know about a futures position's solvency.

    This is *reported* to :mod:`chimera.risk`, which decides. Nothing in this
    dataclass rejects a trade, sizes one, or holds a threshold: a second risk
    authority living in the execution layer is precisely what this package is
    not permitted to grow.
    """

    symbol: str
    side: PositionSide
    leverage: Decimal
    #: Margin actually committed, in quote currency. Isolated, so this is all
    #: that is at risk on this position.
    initial_margin: Decimal
    maintenance_margin: Decimal
    liquidation_price: Decimal
    #: ``|mark - liquidation| / mark``. The number
    #: :attr:`chimera.risk.RiskLimits.min_liquidation_distance_pct` is compared
    #: against.
    liquidation_distance: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "leverage": str(self.leverage),
            "initial_margin": str(self.initial_margin),
            "maintenance_margin": str(self.maintenance_margin),
            "liquidation_price": str(self.liquidation_price),
            "liquidation_distance": str(self.liquidation_distance),
        }


def liquidation_price(
    side: PositionSide,
    entry_price: Decimal,
    leverage: Decimal,
    maintenance_margin_rate: Decimal,
) -> Decimal:
    """Isolated-margin liquidation price, to the precision this package has.

    The standard isolated formula with no cross-collateral and no other
    positions::

        LONG :  entry * (1 - 1/leverage + mmr)
        SHORT:  entry * (1 + 1/leverage - mmr)

    At the 1x leverage v1 uses this puts a LONG's liquidation at ``entry * mmr``
    — a 99.6% adverse move on Binance's tier-1 rate — which is the right answer
    and not an oversight: an isolated 1x position can lose essentially all of its
    margin before it is liquidated.

    **What it deliberately does not model:** accrued funding, unrealised PnL from
    other positions (there are none — isolated), the tiered maintenance-margin
    schedule above tier 1, and the maintenance *amount* deduction. Each would
    move the number, none is knowable from what this package is given, and
    inventing them would put precision into a figure Aegis then treats as real.
    The omissions are conservative for a LONG and anti-conservative for a SHORT
    by the same small amount, which is why ``maintenance_margin_rate`` is a
    required venue field rather than a default.
    """
    if entry_price <= ZERO:
        raise AccountingError(f"entry price {entry_price} is not positive")
    if leverage <= ZERO:
        raise AccountingError(f"leverage {leverage} is not positive")
    if not (ZERO < maintenance_margin_rate < ONE):
        raise AccountingError(
            f"maintenance margin rate {maintenance_margin_rate} is not a fraction in (0, 1)"
        )
    if side is PositionSide.FLAT:
        raise AccountingError("a flat position has no liquidation price")
    inverse = ONE / leverage
    if side is PositionSide.LONG:
        return entry_price * (ONE - inverse + maintenance_margin_rate)
    return entry_price * (ONE + inverse - maintenance_margin_rate)


def margin_state(
    position: Position, mark_price: Decimal, constraints: SymbolConstraints
) -> MarginState | None:
    """Margin and liquidation for an open position, or ``None`` when flat.

    Returns ``None`` rather than a zeroed record: a flat position does not have a
    liquidation price that happens to be zero, and handing Aegis a
    ``liquidation_price=0`` would read as "liquidation is 100% away", which is a
    claim about a position that does not exist.
    """
    if position.is_flat:
        return None
    if mark_price <= ZERO:
        raise AccountingError(f"{position.symbol}: mark price {mark_price} is not positive")
    notional = position.notional(mark_price)
    liquidation = liquidation_price(
        position.side,
        position.entry_price,
        position.leverage,
        constraints.maintenance_margin_rate,
    )
    return MarginState(
        symbol=position.symbol,
        side=position.side,
        leverage=position.leverage,
        initial_margin=notional / position.leverage,
        maintenance_margin=notional * constraints.maintenance_margin_rate,
        liquidation_price=liquidation,
        liquidation_distance=abs(mark_price - liquidation) / mark_price,
    )
