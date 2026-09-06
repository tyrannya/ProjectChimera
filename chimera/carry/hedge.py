"""``HedgedPosition``: the two-leg carry position and the state machine over it.

Section 6 of the adopted master plan. One LONG spot leg and one SHORT perpetual
leg of **equal BTC quantity**, held by two :class:`~chimera.futures.executor.FuturesExecutor`
instances, with a :class:`~chimera.carry.ledger.CarryLedger` recording where the
capital went.

**The invariant this state machine exists to protect is the imbalance.** When
the state is :attr:`HedgeState.HEDGED`, the two legs hold the same quantity in
BTC after step rounding. Everything else here -- the entry order, the
correction retry, the rebalance -- is machinery for restoring that equality or
for refusing to pretend it holds.

**Perp first, spot second.** Section 6.2 fixes the entry order and gives the
reason: the perpetual is the leg that can be reversed cheaply and the one whose
rejection is more likely, so it is the leg to discover a refusal on before the
other side has been bought.

**A failed correction flattens with ``HEDGE_CORRECTION``, never ``DATA_LOSS``.**
The causes are not interchangeable: ``DATA_LOSS`` says the runner stopped being
able to see the market, and reading a hedge-correction flatten as a data outage
would misreport why the position closed.

**Two places where the adopted specification could not be followed literally**,
both resolved mechanically rather than by inventing a rule; both are recorded
in the pull request:

1. Section 6.1 types ``plan(target, state: MarketState)`` with a name section
   12.3 assigns to ``chimera/demo/feed.py`` -- a PR-10 module -- while section
   13 makes PR-10 depend on PR-08. Importing it here would invert the adopted
   dependency graph. The market snapshot is therefore typed **structurally**
   by :class:`CarryMarketState`, which PR-10's ``MarketState`` satisfies without
   either package importing the other. Section 12.3 also lists ``HedgeTarget``
   under two owning modules; the dependency direction settles it, so carry owns
   the type and the rule layer above imports it.
2. Section 6.8's stale-leg rule is defined on a store ``updated_at`` field that
   :class:`~chimera.futures.store.FuturesState` does not have. Adding it would
   edit a persisted schema the frozen dry-run protocol exercises, and
   ``chimera/futures/store.py`` is not in PR-08's scope. The carry ledger owns
   the per-leg observation instants instead, which preserves the rule's meaning
   exactly and moves no frozen schema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from chimera.carry.accounting import ZERO, CarryError
from chimera.carry.ledger import CarryLedger
from chimera.futures.domain import OrderState, PositionSide, TargetPosition
from chimera.futures.executor import FlattenCause, FuturesExecutor
from chimera.futures.store import LoadOutcome
from chimera.risk import RiskEngine

logger = logging.getLogger(__name__)

#: The two legs, named once. Used as ledger keys and as log labels.
SPOT = "spot"
PERP = "perp"


class HedgeError(CarryError):
    """The hedge cannot be planned, applied or reconstructed."""


class HedgeState(str, Enum):
    """Section 6.1's states, and nothing else."""

    FLAT = "FLAT"
    OPENING = "OPENING"
    HEDGED = "HEDGED"
    PARTIAL = "PARTIAL"
    REBALANCING = "REBALANCING"
    CLOSING = "CLOSING"
    DISPUTED = "DISPUTED"


@runtime_checkable
class CarryMarketState(Protocol):
    """What this package needs to know about a minute, and nothing more.

    A structural type rather than an import: see the module docstring. PR-10's
    ``MarketState`` satisfies it, and the dependency runs one way only.
    """

    @property
    def minute_ns(self) -> int: ...

    @property
    def complete(self) -> bool: ...

    @property
    def spot_close(self) -> Decimal: ...

    @property
    def perp_close(self) -> Decimal: ...

    @property
    def mark(self) -> Decimal: ...


@dataclass(frozen=True)
class HedgeTarget:
    """What the rule layer wants held, in BTC on each leg."""

    quantity: Decimal

    def __post_init__(self) -> None:
        if self.quantity < ZERO:
            raise HedgeError(f"hedge target {self.quantity} is negative")

    @property
    def is_flat(self) -> bool:
        return self.quantity == ZERO


@dataclass(frozen=True)
class HedgeConfig:
    """The correction policy and the symbols. Values come from config, not here."""

    spot_symbol: str = "BTC/USDT"
    perp_symbol: str = "BTC/USDT:USDT"
    #: Section 6.3: retry the missing leg for at most this many minutes before
    #: reducing the filled leg back to flat.
    max_correction_minutes: int = 3
    #: Section 6.8: a leg whose last observation lags the other by more than this
    #: is DISPUTED. One minute, expressed in nanoseconds.
    stale_leg_tolerance_ns: int = 60_000_000_000
    maintenance_margin_rate: Decimal = Decimal("0.004")

    def __post_init__(self) -> None:
        if self.max_correction_minutes < 0:
            raise HedgeError("max_correction_minutes cannot be negative")
        if self.stale_leg_tolerance_ns <= 0:
            raise HedgeError("stale_leg_tolerance_ns must be positive")


@dataclass(frozen=True)
class LegState:
    """One leg, as the store reports it."""

    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    last_order_id: str = ""

    @property
    def is_flat(self) -> bool:
        return self.quantity == ZERO

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "last_order_id": self.last_order_id,
        }


@dataclass(frozen=True)
class LegIntent:
    """One planned leg change, in the order the entry sequence sends them."""

    leg: str
    symbol: str
    side: PositionSide
    quantity: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "leg": self.leg,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
        }


@dataclass(frozen=True)
class HedgeOutcome:
    """What :meth:`HedgedPosition.apply` did, for the decision log."""

    state: HedgeState
    filled: tuple[str, ...] = ()
    unfilled: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hedge_state": self.state.value,
            "filled": list(self.filled),
            "unfilled": list(self.unfilled),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CarryMark:
    """The position marked at one minute's closes."""

    quantity: Decimal
    spot_close: Decimal
    perp_close: Decimal
    basis: Decimal
    spot_pnl: Decimal
    perp_pnl: Decimal
    equity: Decimal
    identity_residual: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantity": str(self.quantity),
            "spot_close": str(self.spot_close),
            "perp_close": str(self.perp_close),
            "basis": str(self.basis),
            "spot_pnl": str(self.spot_pnl),
            "perp_pnl": str(self.perp_pnl),
            "equity": str(self.equity),
            "identity_residual": str(self.identity_residual),
        }


@dataclass
class HedgedPosition:
    """One LONG spot / SHORT perpetual pair, and the state machine over it."""

    spot: FuturesExecutor
    perp: FuturesExecutor
    risk: RiskEngine
    ledger: CarryLedger
    config: HedgeConfig = field(default_factory=HedgeConfig)
    state: HedgeState = HedgeState.FLAT
    #: How many minutes the current PARTIAL has been under correction.
    correction_minutes: int = 0
    #: The quantity the position is trying to hold while OPENING or correcting.
    pending_quantity: Decimal = ZERO

    # -- observation -------------------------------------------------------

    def leg(self, name: str) -> LegState:
        """One leg's state, read from its executor's store."""
        executor, symbol = self._executor(name)
        position = executor.position(symbol)
        return LegState(
            symbol=symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
        )

    def _executor(self, name: str) -> tuple[FuturesExecutor, str]:
        if name == SPOT:
            return self.spot, self.config.spot_symbol
        if name == PERP:
            return self.perp, self.config.perp_symbol
        raise HedgeError(f"unknown leg {name!r}; a carry position has exactly spot and perp")

    def imbalance(self) -> Decimal:
        """``spot_qty - perp_qty`` in BTC. Zero whenever the state is HEDGED."""
        return self.leg(SPOT).quantity - self.leg(PERP).quantity

    @property
    def is_disputed(self) -> bool:
        return self.state is HedgeState.DISPUTED or self.ledger.disputed is not None

    def dispute(self, reason: str) -> None:
        """Enter DISPUTED, halt Aegis, and record why. Reductions stay possible."""
        self.ledger.dispute(reason)
        self.state = HedgeState.DISPUTED
        self.risk.halt(f"carry_dispute: {reason}")
        logger.critical("Carry position DISPUTED: %s", reason)

    # -- planning ----------------------------------------------------------

    def plan(self, target: HedgeTarget, state: CarryMarketState) -> list[LegIntent]:
        """The legs to send this minute, in send order: perp SHORT, then spot LONG.

        An increase is refused outright while the position is disputed or Aegis
        is halted; a reduction is always planned, because reducing exposure is
        what a halted system is for.
        """
        if not state.complete:
            return []

        spot_leg, perp_leg = self.leg(SPOT), self.leg(PERP)
        held = min(spot_leg.quantity, perp_leg.quantity)
        increasing = target.quantity > held

        if increasing and (self.is_disputed or self.risk.state.halted):
            return []

        intents: list[LegIntent] = []
        if perp_leg.quantity != target.quantity:
            intents.append(
                LegIntent(
                    leg=PERP,
                    symbol=self.config.perp_symbol,
                    side=PositionSide.SHORT if target.quantity > ZERO else PositionSide.FLAT,
                    quantity=target.quantity,
                )
            )
        if spot_leg.quantity != target.quantity:
            intents.append(
                LegIntent(
                    leg=SPOT,
                    symbol=self.config.spot_symbol,
                    side=PositionSide.LONG if target.quantity > ZERO else PositionSide.FLAT,
                    quantity=target.quantity,
                )
            )
        return intents

    # -- execution ---------------------------------------------------------

    def apply(
        self,
        intents: list[LegIntent],
        state: CarryMarketState,
        *,
        equity: float,
    ) -> HedgeOutcome:
        """Send the planned legs and settle the resulting state.

        Stops at the first leg that does not fill: sending the second leg after
        the first was refused is how a one-sided position is created on purpose.
        """
        if not intents:
            return self._settle(detail="nothing to do")

        target_quantity = intents[0].quantity
        self.pending_quantity = target_quantity
        if self.state in (HedgeState.FLAT, HedgeState.HEDGED) and target_quantity > ZERO:
            self.state = HedgeState.OPENING

        filled: list[str] = []
        unfilled: list[str] = []
        frictions: dict[str, tuple[Decimal, Decimal]] = {}
        for intent in intents:
            executor, symbol = self._executor(intent.leg)
            reference = state.spot_close if intent.leg == SPOT else state.perp_close
            records = executor.execute_target(
                TargetPosition(symbol=symbol, side=intent.side, quantity=intent.quantity),
                reference,
                equity=equity,
            )
            self.ledger.note_leg_mark(intent.leg, state.minute_ns)
            if records and all(r.state is OrderState.FILLED for r in records):
                filled.append(intent.leg)
                frictions[intent.leg] = self._frictions(records, reference)
            else:
                unfilled.append(intent.leg)
                break

        outcome = self._settle(filled=tuple(filled), unfilled=tuple(unfilled))
        self._book(outcome, frictions)
        return outcome

    @staticmethod
    def _frictions(records: list[Any], reference: Decimal) -> tuple[Decimal, Decimal]:
        """One leg's ``(fee, slippage)`` in quote currency.

        Slippage is section 6.5's definition: the difference between the fill
        price and the reference price at the decision, per fill, in quote units.
        Taken as a magnitude because a fill on the wrong side of the reference is
        still a cost paid, never a rebate earned.
        """
        fee = ZERO
        slippage = ZERO
        for record in records:
            fee += record.fees
            if record.filled_quantity:
                slippage += abs(record.average_price - reference) * record.filled_quantity
        return fee, slippage

    def _book(
        self, outcome: HedgeOutcome, frictions: dict[str, tuple[Decimal, Decimal]]
    ) -> None:
        """Record this cycle in the ledger: the entry once, the frictions always.

        The entry is booked exactly when the position first becomes HEDGED, which
        is section 6.2 step 4 -- the ledger records the entry basis
        ``perp_fill - spot_fill`` together with fees and slippage per leg. The
        frictions of any later cycle are attributed on their own, so a rebalance
        or a reduction does not look like a second entry.
        """
        first_entry = (
            outcome.state is HedgeState.HEDGED
            and self.ledger.state.entry_basis is None
            and SPOT in frictions
            and PERP in frictions
        )
        if first_entry:
            spot_leg, perp_leg = self.leg(SPOT), self.leg(PERP)
            spot_fee, spot_slip = frictions[SPOT]
            perp_fee, perp_slip = frictions[PERP]
            self.ledger.book_entry(
                quantity=spot_leg.quantity,
                spot_fill=spot_leg.entry_price,
                perp_fill=perp_leg.entry_price,
                spot_fee=spot_fee,
                perp_fee=perp_fee,
                spot_slippage=spot_slip,
                perp_slippage=perp_slip,
                perp_margin=perp_leg.quantity * perp_leg.entry_price,
            )
            return
        for leg, (fee, slippage) in frictions.items():
            if fee or slippage:
                self.ledger.book_costs(leg=leg, fee=fee, slippage=slippage)

    def _settle(
        self, filled: tuple[str, ...] = (), unfilled: tuple[str, ...] = (), detail: str = ""
    ) -> HedgeOutcome:
        """Recompute the state from the legs. The imbalance decides, not the plan."""
        if self.is_disputed:
            self.state = HedgeState.DISPUTED
            return HedgeOutcome(self.state, filled, unfilled, detail or "disputed")

        spot_leg, perp_leg = self.leg(SPOT), self.leg(PERP)
        if spot_leg.is_flat and perp_leg.is_flat:
            self.state = HedgeState.FLAT
            self.correction_minutes = 0
            return HedgeOutcome(self.state, filled, unfilled, detail or "flat")

        if spot_leg.quantity == perp_leg.quantity:
            self.state = HedgeState.HEDGED
            self.correction_minutes = 0
            return HedgeOutcome(self.state, filled, unfilled, detail or "hedged")

        self.state = HedgeState.PARTIAL
        return HedgeOutcome(
            self.state,
            filled,
            unfilled,
            detail or f"imbalance {spot_leg.quantity - perp_leg.quantity}",
        )

    # -- correction --------------------------------------------------------

    def correct(self, state: CarryMarketState, *, equity: float) -> HedgeOutcome:
        """One correction attempt on a PARTIAL position (section 6.3).

        Retries the missing leg for at most ``max_correction_minutes``. On
        exhaustion the filled leg is reduced to flat with
        ``FlattenCause.HEDGE_CORRECTION`` and the position returns to FLAT.
        """
        if self.state is not HedgeState.PARTIAL:
            return self._settle(detail="not partial")

        self.correction_minutes += 1
        if self.correction_minutes > self.config.max_correction_minutes:
            return self.flatten_for_correction(state)

        self.state = HedgeState.REBALANCING
        target = HedgeTarget(quantity=min(self.leg(SPOT).quantity, self.leg(PERP).quantity))
        larger = HedgeTarget(quantity=max(self.leg(SPOT).quantity, self.leg(PERP).quantity))
        # A rebalance reduces the larger leg to the smaller one; only a genuine
        # retry of an unfilled leg increases exposure, and that is what
        # `pending_quantity` remembers.
        wanted = self.pending_quantity if self.pending_quantity > ZERO else target.quantity
        if self.is_disputed or self.risk.state.halted:
            wanted = target.quantity
        _ = larger
        return self.apply(self.plan(HedgeTarget(wanted), state), state, equity=equity)

    def flatten_for_correction(self, state: CarryMarketState) -> HedgeOutcome:
        """Reduce whichever leg is filled back to flat, and say why it closed."""
        for name in (PERP, SPOT):
            executor, symbol = self._executor(name)
            reference = state.spot_close if name == SPOT else state.perp_close
            executor.emergency_flatten(symbol, FlattenCause.HEDGE_CORRECTION, reference)
            self.ledger.note_leg_mark(name, state.minute_ns)
        self.correction_minutes = 0
        self.pending_quantity = ZERO
        return self._settle(detail="hedge correction timed out; flattened")

    def emergency_reduce(self, cause: FlattenCause, state: CarryMarketState) -> HedgeOutcome:
        """Flatten the perpetual first, then the spot (section 6.8)."""
        self.state = HedgeState.CLOSING
        for name in (PERP, SPOT):
            executor, symbol = self._executor(name)
            reference = state.spot_close if name == SPOT else state.perp_close
            executor.emergency_flatten(symbol, cause, reference)
            self.ledger.note_leg_mark(name, state.minute_ns)
        self.pending_quantity = ZERO
        return self._settle(detail=f"emergency reduce: {cause.value}")

    # -- funding -----------------------------------------------------------

    def settle_funding(self, event: Any, *, open_instant_ns: int, now_ns: int) -> Decimal:
        """Book one perpetual settlement, if it falls in ``open < t <= now``.

        Both tie boundaries are pinned: a settlement exactly at the open instant
        is NOT charged (the position did not hold through it) and one exactly at
        ``now`` IS. Funding applies to the perpetual leg only.
        """
        instant = int(getattr(event, "instant_ns", 0) or 0)
        if not (open_instant_ns < instant <= now_ns):
            return ZERO
        flow = self.perp.settle_funding(event)
        return self.ledger.book_funding(instant, flow)

    # -- marking and the identity -----------------------------------------

    def mark_to_market(self, state: CarryMarketState) -> CarryMark:
        """Mark both legs at the minute's closes and check the running identity."""
        spot_leg, perp_leg = self.leg(SPOT), self.leg(PERP)
        quantity = min(spot_leg.quantity, perp_leg.quantity)
        spot_pnl = quantity * (state.spot_close - spot_leg.entry_price) if quantity else ZERO
        perp_pnl = quantity * (perp_leg.entry_price - state.perp_close) if quantity else ZERO
        equity = (
            self.ledger.state.free_cash
            + spot_leg.quantity * state.spot_close
            + self.ledger.state.perp_margin
            + perp_pnl
        )
        self.ledger.mark(
            spot_close=state.spot_close, perp_close=state.perp_close, equity=equity
        )
        residual = self.ledger.check_identity(
            spot_close=state.spot_close, perp_close=state.perp_close
        )
        if self.ledger.disputed is not None and self.state is not HedgeState.DISPUTED:
            self.dispute(self.ledger.disputed)
        return CarryMark(
            quantity=quantity,
            spot_close=state.spot_close,
            perp_close=state.perp_close,
            basis=state.perp_close - state.spot_close,
            spot_pnl=spot_pnl,
            perp_pnl=perp_pnl,
            equity=equity,
            identity_residual=residual,
        )

    # -- liquidation -------------------------------------------------------

    def liquidation_touched(self, state: CarryMarketState, *, equity: Decimal) -> bool:
        """Section 6.7's two checks. A touch is a hedge-level event, not a venue."""
        quantity = min(self.leg(SPOT).quantity, self.leg(PERP).quantity)
        if quantity == ZERO:
            return False
        maintenance = quantity * state.mark * self.config.maintenance_margin_rate
        if equity < maintenance:
            return True
        margin = self.perp.margin(self.config.perp_symbol, state.mark)
        if margin is None:
            # Section 7.2's liquidation rule: an unknown distance on a non-flat
            # position is refused, never read as "far away".
            return True
        distance = getattr(margin, "liquidation_price", None)
        if distance is None:
            return True
        return state.mark >= distance

    # -- restart -----------------------------------------------------------

    def reconstruct(self) -> HedgeOutcome:
        """Rebuild the state from both stores and the ledger (section 6.8).

        Any disagreement disputes. Nothing is healed silently: a restart that
        quietly adopted one side's story would erase the evidence that the two
        sides ever differed.
        """
        for name, executor in ((SPOT, self.spot), (PERP, self.perp)):
            if executor.store.outcome is LoadOutcome.UNREADABLE:
                self.dispute(f"{name}_store_unreadable")
                return HedgeOutcome(self.state, detail="store unreadable")

        for name, executor in ((SPOT, self.spot), (PERP, self.perp)):
            _, symbol = self._executor(name)
            reported = {
                sym: pos
                for sym, pos in executor.store.state.positions.items()
                if sym == symbol
            }
            report = executor.recover(reported)
            if report is not None and not report.agrees:
                self.dispute(f"{name}_reconciliation_mismatch: {report.detail}")
                return HedgeOutcome(self.state, detail="reconciliation mismatch")

        if self.ledger.disputed is not None:
            self.state = HedgeState.DISPUTED
            self.risk.halt(f"carry_dispute: {self.ledger.disputed}")
            return HedgeOutcome(self.state, detail=self.ledger.disputed)

        stale = self.ledger.stale_leg(tolerance_ns=self.config.stale_leg_tolerance_ns)
        if stale is not None:
            self.dispute(f"stale_leg: {stale}")
            return HedgeOutcome(self.state, detail=f"stale leg {stale}")

        spot_leg, perp_leg = self.leg(SPOT), self.leg(PERP)
        held = min(spot_leg.quantity, perp_leg.quantity)
        if held > ZERO and self.ledger.state.quantity != held:
            self.dispute(
                f"ledger_store_mismatch: ledger holds {self.ledger.state.quantity}, the "
                f"legs hold {held}"
            )
            return HedgeOutcome(self.state, detail="ledger disagrees with the stores")

        return self._settle(detail="reconstructed")
