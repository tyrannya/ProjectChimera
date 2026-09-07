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
from typing import Any, Mapping, Protocol, runtime_checkable

from chimera.carry.accounting import ZERO, CarryError, FundingSettlement
from chimera.carry.ledger import CarryLedger
from chimera.futures.domain import OrderState, PositionSide, TargetPosition
from chimera.futures.fills import TopOfBook
from chimera.futures.executor import FlattenCause, FuturesExecutor
from chimera.futures.store import LoadOutcome
from chimera.risk import RiskEngine

logger = logging.getLogger(__name__)

#: The two legs, named once. Used as ledger keys and as log labels.
SPOT = "spot"
PERP = "perp"

#: Milliseconds to nanoseconds. A funding ``settlement_id`` is the settlement
#: instant in MILLISECONDS (section 5.3) and this package deduplicates on
#: NANOSECONDS, so the two are one multiplication apart and the factor is named
#: once rather than written out at each comparison.
_MS_TO_NS = 1_000_000

#: One minute in nanoseconds.
#:
#: A :class:`CarryMarketState`'s ``minute_ns`` is the minute's OPEN, and a leg
#: fills at ``spot_close``/``perp_close`` -- the minute's CLOSE. So a position
#: comes into existence at ``minute_ns + _MINUTE_NS``, and that, not the open, is
#: the instant section 6.9's window opens at. Recording the open instead made the
#: window sixty seconds wider than the position's life at exactly one moment, its
#: own opening: a hedge opened in the minute ending at a settlement was charged
#: that settlement, which is precisely the case "a settlement exactly at the open
#: instant is NOT charged" exists to exclude.
_MINUTE_NS = 60_000_000_000


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

    @property
    def mark_high(self) -> Decimal | None: ...


@dataclass(frozen=True)
class PerpSettlement:
    """One recorded funding settlement, in the shape the perpetual leg books it.

    Two consumers, one object. :meth:`HedgedPosition.settle_funding` needs
    ``instant_ns`` to place the settlement in section 6.9's window
    ``open_instant < settlement <= now``; :meth:`FuturesExecutor.settle_funding`
    needs ``symbol``, ``rate``, ``mark_price`` and ``settlement_id``. Neither
    existing type carries both -- :class:`chimera.carry.accounting.FundingSettlement`
    has the instant and not the identity, and
    :class:`chimera.futures.accounting.FundingEvent` has the identity and not the
    instant -- so before this type the runner had nothing it could hand across.

    ``symbol`` is the **position's** symbol (``BTC/USDT:USDT``), not the venue's
    (``BTCUSDT``). The two differ, and the difference is not cosmetic:
    :func:`chimera.futures.accounting.funding_cash_flow` looks the position up by
    the event's symbol, and a settlement carrying the venue's spelling would find
    no position, read as flat, and book a silent zero -- funding that the log
    would then report as having been settled. Translating at the boundary is the
    caller's job; refusing an untranslated one is this type's.

    The two refusals are section 6.9's reused ones, and they read
    :data:`FundingSettlement.MAX_PLAUSIBLE_RATE` rather than restating it, so the
    carry package cannot come to hold two different opinions about what a
    plausible 8-hourly rate is.
    """

    symbol: str
    rate: Decimal
    mark_price: Decimal
    settlement_id: str
    instant_ns: int

    def __post_init__(self) -> None:
        if not self.symbol:
            raise CarryError("a funding settlement with no symbol cannot be attributed")
        if not self.settlement_id:
            raise CarryError("a funding settlement with no id cannot be deduplicated")
        if self.mark_price <= ZERO:
            raise CarryError(
                f"{self.symbol}: funding mark price {self.mark_price} is not positive. "
                "The notional is the quantity at the settlement's own mark, so a "
                "settlement without one cannot be booked and is refused rather than "
                "priced from a mark this position happens to be holding."
            )
        if abs(self.rate) > FundingSettlement.MAX_PLAUSIBLE_RATE:
            raise CarryError(
                f"funding rate {self.rate} exceeds "
                f"{FundingSettlement.MAX_PLAUSIBLE_RATE} in magnitude. A realised "
                "8-hourly BTCUSDT rate is on the order of 1e-4; this is refused as a "
                "unit error rather than clipped, filtered or winsorised."
            )


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
    #: One fill model per leg, keyed by :data:`SPOT` / :data:`PERP`. Held here
    #: because :meth:`install_quote` is the runner's obligation and the runner
    #: reaches execution through this object; ``None`` leaves a caller that
    #: installs its own quotes exactly as it was.
    #:
    #: Per leg and never shared. ``RecordedQuoteFillModel`` carries one book and
    #: one clock, and :mod:`chimera.futures.fills` states the pairing rule in
    #: terms -- "a snapshot carries no symbol, so pairing the two is the caller's
    #: job: one venue and one fill model per leg". A single shared model priced
    #: the spot leg off whichever book :meth:`install_quote` happened to pick,
    #: which is the perpetual's: the spot leg then filled at a price no spot book
    #: ever showed, and the ledger recorded an entry basis that was wrong by the
    #: whole real basis -- in the synthetic day, ``-13.06`` against a true
    #: ``+30.00``. The basis is the quantity a carry position exists to capture,
    #: so getting it from the wrong book is not a rounding difference.
    fill_models: Mapping[str, Any] | None = None
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

    # -- the book both legs price against ---------------------------------

    def install_quote(self, state: CarryMarketState) -> None:
        """Install each leg's own book on that leg's fill model, and move its clock.

        :class:`~chimera.futures.fills.RecordedQuoteFillModel` states the
        obligation and names the runner as the one who owes it: "the runner
        installs the decision minute's book with ``set_quote`` before each
        cycle", and "**the caller must advance** ``now_ns`` **every decision
        cycle, whether or not a new book arrived**".

        Nothing met it. ``set_quote`` had no caller outside the test harness, so
        through the production entry point the model held no quote and a clock of
        zero, every order was refused ``no_fresh_quote``, and a campaign never
        opened a position at all -- the same defect as a settlement primitive
        with no caller, on the execution path instead of the evidence path.

        The clock moves on every cycle, including a minute that carried no book:
        that is what lets the freshness rule fire on a stale one instead of
        ageing it against a clock that stopped. ``instant_ns`` is the minute's
        CLOSE, which is the convention ``set_quote`` requires and the instant the
        snapshot belongs to; stamping the open would hand the model a book from
        the future and refuse every order in the run.
        """
        models = self.fill_models
        if not models:
            return
        now_ns = int(state.minute_ns) + _MINUTE_NS
        for leg, attribute in ((SPOT, "book_spot"), (PERP, "book_perp")):
            model = models.get(leg)
            if model is None:
                continue
            book = getattr(state, attribute, None) or {}
            bid, ask = book.get("bid"), book.get("ask")
            if bid is None or ask is None:
                # No book for this leg this minute. Its clock still moves, so the
                # previously installed one ages and is refused rather than
                # filling for ever. Per leg: the spot book can be absent on a
                # minute the perpetual's arrived, and ageing both because one is
                # missing would refuse a leg that had a good book.
                model.now_ns = now_ns
                continue
            model.set_quote(
                TopOfBook(
                    instant_ns=now_ns,
                    bid=bid,
                    bid_qty=book.get("bid_qty") or ZERO,
                    ask=ask,
                    ask_qty=book.get("ask_qty") or ZERO,
                ),
                now_ns,
            )

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
        try:
            self._execute(intents, state, equity, filled, unfilled, frictions)
        finally:
            # `execute_target` raises -- ReconciliationRequired, NotBootstrapped
            # -- and it raises per leg, so the second leg can throw after the
            # first has already filled and persisted. Reconciling in `finally`
            # is what stops that cycle's economics from being lost: the fees and
            # the realised PnL would be recovered by a later level comparison,
            # but this cycle's slippage exists nowhere else and would be gone.
            self._reconcile_ledger(frictions)

        outcome = self._settle(filled=tuple(filled), unfilled=tuple(unfilled))
        self.ledger.note_open_instant(
            flat=outcome.state is HedgeState.FLAT,
            instant_ns=state.minute_ns + _MINUTE_NS,
        )
        return outcome

    def _execute(
        self,
        intents: list[LegIntent],
        state: CarryMarketState,
        equity: float,
        filled: list[str],
        unfilled: list[str],
        frictions: dict[str, tuple[Decimal, Decimal]],
    ) -> None:
        """Send the planned legs in order, stopping at the first that does not fill.

        Split out of :meth:`apply` only so that the reconciliation which follows
        it can sit in a ``finally``. The lists it appends to are the caller's, so
        a leg that filled before another raised is still described.
        """
        for intent in intents:
            executor, symbol = self._executor(intent.leg)
            reference = state.spot_close if intent.leg == SPOT else state.perp_close
            records = executor.execute_target(
                TargetPosition(symbol=symbol, side=intent.side, quantity=intent.quantity),
                reference,
                equity=equity,
            )
            self.ledger.note_leg_mark(intent.leg, state.minute_ns)
            # Frictions are taken from whatever came back, filled or not. A
            # partial fill charges a fee and moves a price while ending
            # something other than FILLED, and slippage is the one quantity no
            # executor accumulates -- so a cycle that does not measure it is a
            # cycle that loses it, permanently.
            frictions[intent.leg] = self._frictions(records, reference)
            if records and all(r.state is OrderState.FILLED for r in records):
                filled.append(intent.leg)
            else:
                unfilled.append(intent.leg)
                break

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

    def _reconcile_ledger(
        self, frictions: Mapping[str, tuple[Decimal, Decimal]] | None = None
    ) -> None:
        """Bring the carry ledger to what the two executors now hold. One authority.

        This replaced three separate booking triggers -- ``book_entry`` on the
        first HEDGED cycle, ``book_reduction`` through a close path, and
        ``book_costs`` for everything else. Each was a branch someone had to
        remember to reach, and two transitions had no branch at all:

        * a hedge completed **across two minutes** booked no entry, because the
          entry trigger needed both legs' frictions in ONE cycle. A refused spot
          leg leaves PARTIAL; the next minute
          ``DemoRunner._target_to_act_on`` sees a FLAT spot leg and hands the
          rule's target back, so ``plan`` sends only the missing leg -- and the
          trigger never fires. How much that costs depends on whether the
          re-sized target still equals what the perpetual holds. When it does,
          the position reaches HEDGED with an empty carry ledger and
          ``free_cash`` holds money it never spent; when it does not, the
          perpetual is sent too, the trigger fires, and what is lost is only the
          realised PnL and fees of that leg's own adjustment -- which
          ``book_entry`` had no argument for. The first is the one that reads
          the notional as profit; both are wrong and both are this;
        * an **increase** booked no principal, no margin and no quantity at all.
          ``_target_to_act_on`` holds the entry size while the position is on,
          so the rule cannot re-hedge (section 6.4 forbids it) -- but the same
          two-minute retry above re-sizes, and ``correct()`` retries at
          ``pending_quantity``, so a leg can be asked for MORE than it holds.
          Booking nothing for it is the ~25% phantom drawdown's mirror image.

        Nothing here asks what KIND of cycle this was. Four levels are read off
        the executors -- each leg's inventory at its own VWAP, each leg's
        cumulative fees and cumulative realised PnL -- and the ledger is moved to
        them. A transition nobody thought of is still reconciled, because a
        level does not need to be recognised to be compared.

        The three parts, in this order:

        1. **This cycle's slippage**, per leg, immediately. It is the one
           quantity the executors do not accumulate, so it cannot be recovered
           later and must not be deferred behind a booking that might refuse.
        2. **Fees and realised PnL**, as the difference between each executor's
           cumulative total and what this ledger has already booked for that
           leg. Self-correcting: however many calls a close takes, the ledger
           converges on what the executors actually did.
        3. **The position**, through :meth:`CarryLedger.book_position`, which
           takes each leg's level and moves ``free_cash`` by the change.

        **An asymmetric close still DISPUTES** (section 6.8). ``emergency_flatten``
        returns normally when the venue REJECTS the order, so a close can leave
        one leg flat and the other holding, and this ledger holds one hedged
        quantity for a two-leg position. What has changed is that the dispute no
        longer costs money: the leg that did close really did return its cash and
        that is booked, the leg that did not keeps its principal, and the
        frictions of both are booked either way. Refusing to book was losing the
        closed leg's exit slippage for ever.
        """
        exits = dict(frictions or {})
        state = self.ledger.state
        booked_quantity = state.quantity
        spot_l, perp_l = self.leg(SPOT), self.leg(PERP)

        accruals = {SPOT: state.spot, PERP: state.perp}
        for name, executor in ((SPOT, self.spot), (PERP, self.perp)):
            accrual = accruals[name]
            fee = executor.ledger.trading_fees - accrual.fees
            realised = executor.ledger.realised_pnl - accrual.realised
            slippage = exits.get(name, (ZERO, ZERO))[1]
            if fee < ZERO:
                # `trading_fees` is a magnitude that only ever grows --
                # `Ledger.book_fee` refuses a negative. So a total BELOW what
                # this ledger has booked is not a fee, it is an executor ledger
                # that was reset (``FuturesStore.adopt_after_unreadable`` starts
                # a fresh `FuturesState`, zeroing every accumulator). Booking
                # the difference would CREDIT the campaign its whole fee history
                # as cash. A level reconciliation is only sound while the level
                # it reconciles to is the same history; when it is not, the two
                # ledgers disagree about what this position did and that is a
                # dispute, not an adjustment.
                self.dispute(
                    f"{name}_ledger_regressed: the {name} executor reports "
                    f"{executor.ledger.trading_fees} of cumulative fees and this ledger "
                    f"has already booked {accrual.fees}. Fees never fall, so the "
                    "executor's accumulators were reset and the two ledgers no longer "
                    "describe the same history"
                )
                return
            if fee or realised or slippage:
                self.ledger.book_costs(
                    leg=name, fee=fee, slippage=slippage, realised=realised
                )

        spot_leg, perp_leg = spot_l, perp_l
        self.ledger.book_position(
            spot_quantity=spot_leg.quantity,
            spot_entry=spot_leg.entry_price,
            perp_quantity=perp_leg.quantity,
            perp_entry=perp_leg.entry_price,
        )

        if (
            spot_leg.quantity != perp_leg.quantity
            and min(spot_leg.quantity, perp_leg.quantity) < booked_quantity
        ):
            self.dispute(
                f"asymmetric_close: the spot leg holds {spot_leg.quantity} and the "
                f"perpetual {perp_leg.quantity} after a close of a booked "
                f"{booked_quantity}. One leg did not flatten, so the position is not "
                "reducible by a single quantity; each leg's own cash is booked and the "
                "position is disputed rather than reported as closed"
            )

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
        exits: dict[str, tuple[Decimal, Decimal]] = {}
        try:
            self._flatten_legs(FlattenCause.HEDGE_CORRECTION, state, exits)
        finally:
            self.correction_minutes = 0
            self.pending_quantity = ZERO
            self._reconcile_ledger(exits)
        outcome = self._settle(detail="hedge correction timed out; flattened")
        self.ledger.note_open_instant(
            flat=outcome.state is HedgeState.FLAT,
            instant_ns=state.minute_ns + _MINUTE_NS,
        )
        return outcome

    def _flatten_legs(
        self,
        cause: FlattenCause,
        state: CarryMarketState,
        exits: dict[str, tuple[Decimal, Decimal]],
    ) -> None:
        """Flatten the perpetual first, then the spot (section 6.8).

        Separate from its two callers so that the reconciliation which follows
        can sit in a ``finally``: ``emergency_flatten`` raises ``NotBootstrapped``
        per leg, so the SECOND leg can throw after the first has already
        flattened, and this cycle's slippage for that first leg exists nowhere
        else. ``exits`` is the caller's, so whatever was measured survives.
        """
        for name in (PERP, SPOT):
            executor, symbol = self._executor(name)
            reference = state.spot_close if name == SPOT else state.perp_close
            closing = executor.emergency_flatten(symbol, cause, reference)
            exits[name] = self._frictions([closing] if closing else [], reference)
            self.ledger.note_leg_mark(name, state.minute_ns)

    def emergency_reduce(self, cause: FlattenCause, state: CarryMarketState) -> HedgeOutcome:
        """Flatten the perpetual first, then the spot (section 6.8)."""
        self.state = HedgeState.CLOSING
        exits: dict[str, tuple[Decimal, Decimal]] = {}
        try:
            self._flatten_legs(cause, state, exits)
        finally:
            self.pending_quantity = ZERO
            self._reconcile_ledger(exits)
        outcome = self._settle(detail=f"emergency reduce: {cause.value}")
        self.ledger.note_open_instant(
            flat=outcome.state is HedgeState.FLAT,
            instant_ns=state.minute_ns + _MINUTE_NS,
        )
        return outcome

    # -- funding -----------------------------------------------------------

    def settle_funding(
        self, event: PerpSettlement, *, open_instant_ns: int, now_ns: int
    ) -> Decimal:
        """Book one perpetual settlement, if it falls in ``open < t <= now``.

        Both tie boundaries are pinned: a settlement exactly at the open instant
        is NOT charged (the position did not hold through it) and one exactly at
        ``now`` IS. Funding applies to the perpetual leg only.

        Two dedup layers run, in this order and not the other: the perpetual
        executor's :class:`~chimera.futures.accounting.Ledger` refuses a
        ``settlement_id`` it has already booked and PERSISTS that refusal with the
        store, and this ledger then refuses an ``instant_ns`` it has already
        booked. The order matters because the executor's is the one that moves
        money; a crash between the two is what :meth:`booked_settlement_instants`
        exists to catch on the next start.
        """
        instant = int(event.instant_ns)
        if not (open_instant_ns < instant <= now_ns):
            return ZERO
        flow = self.perp.settle_funding(event)
        return self.ledger.book_funding(instant, flow)

    def booked_settlement_instants(self) -> tuple[int, ...]:
        """The settlement instants the PERPETUAL LEG's ledger has already booked.

        Read back out of the executor's own persisted ``applied_funding`` and
        converted to the instants this package deduplicates on, so the two
        ledgers can be compared on one scale. Section 5.3 fixes the identity as
        ``settlement_id = str(fundingTime)`` in milliseconds; an id that is not
        that is left out of the comparison rather than parsed into a number it
        might not be, because a foreign id is a finding for
        :meth:`reconstruct` and not something to coerce.
        """
        instants: list[int] = []
        for identifier in getattr(self.perp.ledger, "applied_funding", ()):  # noqa: B009
            try:
                instants.append(int(str(identifier)) * _MS_TO_NS)
            except ValueError:
                continue
        return tuple(sorted(instants))

    # -- marking and the identity -----------------------------------------

    def mark_to_market(self, state: CarryMarketState) -> CarryMark:
        """Mark both legs at the minute's closes and check the running identity.

        Section 6.6's identity, per leg. ``perp_margin`` is the margin posted for
        the quantity the PERPETUAL leg holds, so the unrealised term that sits
        beside it in the equity line has to be measured on that same quantity.
        Marking it on ``min(spot, perp)`` instead reported a margin the position
        held and a PnL it did not, and the two disagreed by exactly the
        imbalance -- which is the state an equity reading most needs to be right
        about, because it is the state Aegis is being asked to halt on.

        The reported ``quantity`` stays the HEDGED one: that is what the basis
        and section 6.5's identity are defined on, and it is not an equity term.
        """
        spot_leg, perp_leg = self.leg(SPOT), self.leg(PERP)
        quantity = min(spot_leg.quantity, perp_leg.quantity)
        spot_pnl = (
            spot_leg.quantity * (state.spot_close - spot_leg.entry_price)
            if spot_leg.quantity
            else ZERO
        )
        perp_pnl = (
            perp_leg.quantity * (perp_leg.entry_price - state.perp_close)
            if perp_leg.quantity
            else ZERO
        )
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
        """Section 6.7's two checks. A touch is a hedge-level event, not a venue.

        The portfolio test is section 6.7's, verbatim:
        ``equity < Q * mark_high * maintenance_margin_rate``. The mark HIGH is
        the most adverse mark the minute can be shown to have reached, and using
        the close instead would put the threshold strictly below the adopted one
        -- a spike that section 6.7 calls a touch would read as "not touched".

        The fallback when a minute carries no high is the close, and it is P13's
        own, ported rather than invented: ``chimera.carry.accounting.Quote``'s
        ``liquidation_touch`` uses "the mark high where the source provides it
        and falls back to the mark close. There is no third tier." A non-flat
        position on a minute carrying neither is refused below, with the rest of
        the unknown-information cases.
        """
        quantity = min(self.leg(SPOT).quantity, self.leg(PERP).quantity)
        if quantity == ZERO:
            return False
        adverse = getattr(state, "mark_high", None) or state.mark
        if adverse is None:
            # Section 7.2's liquidation rule: unknown information on a non-flat
            # position is refused, never read as "far away".
            return True
        maintenance = quantity * adverse * self.config.maintenance_margin_rate
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
        # Unconditionally, in both directions. The guard used to read
        # `held > ZERO and ...`, which skipped the comparison in exactly the
        # state the crash window produces: the executors persist inside
        # `execute_target`, this ledger persists later, so a crash between them
        # leaves the legs FLAT and the ledger still holding the whole position.
        # `held` is then zero, the check did not run, `reconstruct` returned
        # READY, and the campaign carried on with `free_cash` short by the
        # principal and the margin of a position it had already closed -- which
        # the next reconciliation would then absorb without saying so. Section
        # 6.8 compares "the two legs' quantities and the ledger file"; it does
        # not make the comparison conditional on the legs being non-flat.
        #
        # The principals are compared as well as the quantity. They are what
        # `free_cash` was moved by, and a torn write can leave them disagreeing
        # while the quantities happen to match -- an increase that reached the
        # stores and not this file moves both legs and the hedged quantity by
        # the same step, so the quantity alone cannot see it.
        mismatch = None
        if self.ledger.state.quantity != held:
            mismatch = (
                f"ledger holds {self.ledger.state.quantity}, the legs hold {held}"
            )
        else:
            for name, leg, booked in (
                (SPOT, spot_leg, self.ledger.state.spot_principal),
                (PERP, perp_leg, self.ledger.state.perp_margin),
            ):
                level = leg.quantity * leg.entry_price
                if level != booked:
                    mismatch = (
                        f"the {name} leg holds {leg.quantity} at {leg.entry_price}, which "
                        f"cost {level}, and the ledger booked {booked}"
                    )
                    break
        if mismatch is not None:
            self.dispute(f"ledger_store_mismatch: {mismatch}")
            return HedgeOutcome(self.state, detail="ledger disagrees with the stores")

        unbooked = self.unbooked_funding_instants()
        if unbooked:
            # The one crash window funding has, and the only place it is visible.
            # `settle_funding` books the perpetual executor's ledger first (which
            # persists with that leg's store) and this ledger second (which
            # persists on the next `save`), so a crash between the two leaves the
            # money moved and the carry ledger not knowing it. Both files are
            # internally consistent, both load, and every other check here passes
            # -- so without this the campaign would run on from a free_cash that
            # is wrong by exactly one settlement, for ever, silently.
            #
            # It is a dispute rather than a repair. The flow can be re-derived,
            # but re-deriving it would be this ledger adopting the other one's
            # story, which is the silent overwrite section 6.8 refuses everywhere
            # else.
            #
            # **This one has no operator clearing path in this build**, and the
            # honest statement is that rather than the one that used to be here.
            # `DemoRunner.resolve` clears only a leg's RECONCILIATION dispute; it
            # used to clear whatever the carry ledger happened to be disputing,
            # which cleared a flag and booked nothing -- the settlement stayed
            # unbooked and the cash stayed short, so the campaign resumed on a
            # ledger it had been told to distrust. Refusing to pretend is the
            # safer half; the missing half is a `resolve` that re-books the torn
            # settlement, and it is recorded in docs/demo_runbook.md rather than
            # improvised here. Until then this dispute is cleared by repairing
            # the ledger file, deliberately by hand.
            self.dispute(
                "funding_booking_torn: the perpetual leg's ledger has booked "
                f"settlement(s) {list(unbooked)} that the carry ledger has not. A crash "
                "between the two bookings leaves this position's free cash short by "
                "exactly those settlements"
            )
            return HedgeOutcome(self.state, detail="funding booking torn")

        outcome = self._settle(detail="reconstructed")
        if outcome.state is HedgeState.FLAT:
            # A restart onto a flat position clears the window. Nothing is held,
            # so no settlement can belong to it, and leaving a stale instant
            # behind would put the next position's lower bound in the past.
            self.ledger.note_open_instant(flat=True, instant_ns=0)
        return outcome

    def unbooked_funding_instants(self) -> tuple[int, ...]:
        """Settlements the perpetual leg has booked and this ledger has not.

        Empty is the only sound answer for a campaign that has not crashed
        mid-settlement. Anything else is the torn window
        :meth:`settle_funding` documents.
        """
        booked = set(self.ledger.state.settled)
        return tuple(
            instant for instant in self.booked_settlement_instants() if instant not in booked
        )
