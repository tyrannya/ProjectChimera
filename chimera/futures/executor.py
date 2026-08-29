"""Hermes for USD-M perpetual futures, in dry-run only.

The path this module implements, and the order of it, is the invariant:

    Pythia signal -> Aegis decision -> Hermes execution

:class:`FuturesExecutor` is the Hermes half. It plans orders, hands every
exposure-*increasing* one to :class:`chimera.risk.RiskEngine`, and submits only
what came back approved. There is no branch in which an order reaches the venue
without that call, which is what "an Aegis veto must make execution impossible"
has to mean to be worth stating: not a check that could be skipped, but the only
route to :meth:`chimera.futures.venue.DryRunFuturesVenue.submit`.

What this module is **not** allowed to be is a second risk authority. It refuses
orders on two grounds only, both of them venue facts rather than opinions: the
constraints in :mod:`chimera.futures.venue` (a size the exchange would not
accept), and a reconciliation mismatch (a state it cannot act on truthfully).
Every discretionary limit — exposure, drawdown, leverage, funding, liquidation
distance, the kill switch — lives in :mod:`chimera.risk` and is reached through
``evaluate_entry``. Margin and liquidation are *computed* here and *reported*
there, which is the only way to give the risk engine futures awareness without
giving the execution layer a veto of its own.

Reductions are deliberately not gated. A halted account must still be able to
close, and an entry gate that also blocked exits turns a kill switch into a trap;
:attr:`chimera.futures.domain.OrderPurpose.increases_exposure` is where that
distinction is written down.

**No live path exists.** The executor refuses to construct outside dry-run,
:class:`chimera.futures.venue.DryRunFuturesVenue` is the only venue in the
package, and neither holds a credential or opens a socket.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Mapping

from chimera import metrics
from chimera.contracts import Signal
from chimera.futures.accounting import (
    FundingEvent,
    Ledger,
    MarginState,
    margin_state,
    unrealised_pnl,
)
from chimera.futures.domain import (
    ZERO,
    EventKind,
    FuturesError,
    InvalidTransition,
    OrderEvent,
    OrderIntent,
    OrderRecord,
    OrderSide,
    OrderState,
    Position,
    PositionSide,
    TargetPosition,
    plan_flatten,
    plan_transition,
)
from chimera.futures.store import FuturesStore, LoadOutcome
from chimera.futures.venue import ConstraintError, DryRunFuturesVenue, SymbolConstraints
from chimera.risk import RiskDecision, RiskEngine

logger = logging.getLogger(__name__)


class LiveFuturesNotImplemented(FuturesError):
    """Someone asked for a live futures path. There is not one to enable."""


class NotBootstrapped(FuturesError):
    """The executor does not yet know what the account holds."""


class ReconciliationRequired(FuturesError):
    """Local and reported state disagree, and nothing may act until they do not."""


class FlattenCause(str, Enum):
    """Why an emergency flatten happened. Bounded: it is a metric label."""

    OPERATOR = "OPERATOR"
    RISK_HALT = "RISK_HALT"
    RECONCILIATION_MISMATCH = "RECONCILIATION_MISMATCH"
    SHUTDOWN = "SHUTDOWN"
    DATA_LOSS = "DATA_LOSS"


class ReconciliationPolicy(str, Enum):
    """What a mismatch does. Chosen in config, before a mismatch exists.

    ``HALT``
        stop. Increases are refused until an operator resolves it. The default,
        because a disagreement about the position is a disagreement about how
        much money is at stake, and acting on either version could be the wrong
        one.

    ``FLATTEN``
        emergency-flatten the *local* position and stop. Appropriate when an
        unattended process must not hold exposure it cannot verify. Still refuses
        to trade afterwards: flattening resolves the exposure, not the
        disagreement.
    """

    HALT = "HALT"
    FLATTEN = "FLATTEN"


class ReconciliationOutcome(str, Enum):
    AGREED = "AGREED"
    MISMATCH = "MISMATCH"


@dataclass(frozen=True)
class ReconciliationReport:
    """What local said, what the venue said, and whether they were the same."""

    symbol: str
    outcome: ReconciliationOutcome
    local: Position
    reported: Position
    detail: str = ""

    @property
    def agrees(self) -> bool:
        return self.outcome is ReconciliationOutcome.AGREED

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "outcome": self.outcome.value,
            "local": self.local.to_dict(),
            "reported": self.reported.to_dict(),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FuturesExecutionConfig:
    """How the executor behaves. Every field is a decision made before a run."""

    #: The only supported value. Present so a config that asks for live trading
    #: is refused with an explanation rather than silently ignored.
    dry_run: bool = True
    #: v1 is isolated margin at exactly 1x. Both are checked rather than assumed,
    #: so a config that sets 3x is stopped at construction instead of producing
    #: positions whose liquidation figures were computed for something else.
    leverage: Decimal = Decimal("1")
    margin_mode: str = "ISOLATED"
    reconciliation_policy: ReconciliationPolicy = ReconciliationPolicy.HALT

    def __post_init__(self) -> None:
        if not self.dry_run:
            raise LiveFuturesNotImplemented(
                "chimera.futures has no live-order path. There is no credential to "
                "supply and no endpoint to enable: the only venue in the package is "
                "DryRunFuturesVenue, which simulates fills in this process. Live "
                "futures execution is a separate piece of work with its own review, "
                "and chimera.safety's acknowledgement gate does not unlock it."
            )
        if self.leverage != Decimal("1"):
            raise FuturesError(
                f"leverage {self.leverage} is not 1. Futures Execution v1 is isolated "
                "margin at exactly 1x; the liquidation model, the margin figures Aegis "
                "is handed and the whole test matrix are written for that."
            )
        if self.margin_mode != "ISOLATED":
            raise FuturesError(
                f"margin mode {self.margin_mode!r} is not ISOLATED. Cross margin shares "
                "collateral between positions, which the margin model here does not "
                "represent."
            )


@dataclass
class FuturesExecutor:
    """Plans, gates, submits and books futures orders against a dry-run venue."""

    venue: DryRunFuturesVenue
    risk: RiskEngine
    store: FuturesStore
    config: FuturesExecutionConfig = field(default_factory=FuturesExecutionConfig)
    #: Injected so a replay is deterministic and a test does not sleep.
    clock: Callable[[], float] = time.time
    _order_seq: int = 0
    #: Peak simulated net PnL, for the drawdown gauge. Descriptive only.
    _peak_net_pnl: Decimal = ZERO

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def ledger(self) -> Ledger:
        return self.store.state.ledger

    def position(self, symbol: str) -> Position:
        return self.store.state.position(symbol)

    def constraints(self, symbol: str) -> SymbolConstraints:
        return self.venue.constraints(symbol)

    def require_ready(self, symbol: str) -> None:
        """Refuse to plan while the account's state is unknown or disputed."""
        if not self.store.state.bootstrapped:
            raise NotBootstrapped(
                "the futures executor has not adopted a starting position. An empty "
                "state file is not a flat account: call recover() with the venue's "
                "reported positions before planning anything."
            )
        dispute = self.store.state.disputed.get(symbol)
        if dispute:
            raise ReconciliationRequired(
                f"{symbol} is disputed: {dispute}. Local and reported state disagree, so "
                "any new order would be sized against a position that may not exist. "
                "Resolve it with resolve_reconciliation(), or flatten."
            )
        blocked = [
            o
            for o in self.store.state.orders.values()
            if o.state is OrderState.RECONCILIATION_REQUIRED and o.intent.symbol == symbol
        ]
        if blocked:
            raise ReconciliationRequired(
                f"{symbol}: {len(blocked)} order(s) are RECONCILIATION_REQUIRED. Local "
                "and reported state disagree, so any new order would be sized against a "
                "position that may not exist. Resolve it or flatten."
            )

    # ------------------------------------------------------------------
    # restart / recovery
    # ------------------------------------------------------------------
    def recover(
        self, reported: Mapping[str, Position] | None = None
    ) -> ReconciliationReport | None:
        """Bring a freshly started process to a state it may act from.

        Idempotent: calling it again on an already-bootstrapped executor
        re-reconciles and changes nothing else. That matters because the five
        boundaries a restart can land on — planned but unsubmitted, acknowledged,
        partially filled, filled but unpersisted, and already recovered — are all
        reached by the same call, and a recovery that behaved differently the
        second time would be a sixth case nobody tested.

        ``reported`` is the venue's own view. It is *adopted* only when there is
        no local state to contradict; where local state exists it is
        **compared**, never overwritten.
        """
        metrics.FUT_RECOVERY.labels(outcome=self.store.outcome.value).inc()
        if self.store.outcome is LoadOutcome.UNREADABLE:
            logger.critical(
                "Recovering from an unreadable futures state file. Nothing will be "
                "planned until an operator adopts a position explicitly."
            )
        reported = dict(reported or {})

        if not self.store.state.bootstrapped:
            self.store.bootstrap(reported)
            logger.warning(
                "Futures executor bootstrapped from the venue-reported position: %s",
                {s: p.to_dict() for s, p in reported.items()} or "flat",
            )
            self._publish_position_metrics()
            return None

        # Already bootstrapped: this is a re-entry, so nothing is adopted. Any
        # order left mid-flight is resolved against what the venue says now.
        #
        # Every symbol is reconciled, and the loop does not stop at the first
        # disagreement. Stopping there left every later symbol unreconciled and
        # therefore still tradable — against state nobody had verified, which is
        # the exact thing the disputed mark exists to prevent. The report handed
        # back is the first disagreement if there was one, because that is what a
        # caller checking `.agrees` is asking about.
        report: ReconciliationReport | None = None
        first_disagreement: ReconciliationReport | None = None
        symbols = set(reported) | set(self.store.state.positions)
        for symbol in sorted(symbols):
            report = self.reconcile(symbol, reported.get(symbol))
            if not report.agrees and first_disagreement is None:
                first_disagreement = report
        report = first_disagreement or report
        self._resolve_stranded_orders()
        self._publish_position_metrics()
        return report

    def _resolve_stranded_orders(self) -> None:
        """Cancel orders that a crash left before submission.

        A PLANNED or RISK_APPROVED order never reached the venue — the venue is
        told about an order in ``submit`` and nowhere else — so cancelling it
        locally cannot orphan anything. An order that *did* reach the venue is
        left alone: reconciliation is what decides those, and guessing here would
        be the silent overwrite this package refuses to do.
        """
        for order in self.store.state.orders.values():
            if order.state in (OrderState.PLANNED, OrderState.RISK_APPROVED):
                order.transition(OrderState.CANCELLED, "not submitted before restart")
        self.store.save()

    # ------------------------------------------------------------------
    # reconciliation
    # ------------------------------------------------------------------
    def reconcile(self, symbol: str, reported: Position | None = None) -> ReconciliationReport:
        """Compare local and reported state. Never replaces one with the other.

        On disagreement the *symbol* is marked disputed and every open order for
        it is moved to ``RECONCILIATION_REQUIRED`` — both of which
        :meth:`require_ready` then refuses to plan through — and the configured
        policy decides whether the position is also flattened. The symbol-level
        mark is the load-bearing one: a mismatch is usually noticed after the
        fills that caused it, so the orders involved are already terminal and a
        stop that only looked at open orders would not fire at all.

        What does *not* happen is the local position quietly becoming the
        reported one: that would make every mismatch invisible in exactly the
        situation where a human needs to see it.
        """
        local = self.position(symbol)
        venue_view = self.venue.reported_position(symbol) if reported is None else reported

        if local.side is venue_view.side and local.quantity == venue_view.quantity:
            metrics.FUT_RECONCILIATION.labels(outcome=ReconciliationOutcome.AGREED.value).inc()
            # An agreement does NOT clear a standing dispute. Two states can come
            # to agree again for a reason nobody has explained — a later fill
            # happening to land on the disputed quantity — and reading that as a
            # resolution would be the silent overwrite this whole path refuses.
            # Only resolve_reconciliation() clears one, and it takes a reason.
            return ReconciliationReport(
                symbol=symbol,
                outcome=ReconciliationOutcome.AGREED,
                local=local,
                reported=venue_view,
            )

        detail = (
            f"local says {local.side.value} {local.quantity}, the venue says "
            f"{venue_view.side.value} {venue_view.quantity}"
        )
        logger.critical("Futures reconciliation mismatch on %s: %s", symbol, detail)
        metrics.FUT_RECONCILIATION.labels(outcome=ReconciliationOutcome.MISMATCH.value).inc()

        # Recorded against the symbol, not only against its open orders. A
        # position can be disputed with every order already terminal — which is
        # the common case, since a mismatch is usually noticed after the fills
        # that caused it — and a stop that only inspected open orders would let
        # the next signal size itself against a position nobody can vouch for.
        self.store.state.disputed[symbol] = detail

        for order in self.store.state.orders.values():
            if order.intent.symbol != symbol or order.is_terminal:
                continue
            if order.state is OrderState.RECONCILIATION_REQUIRED:
                continue
            order.transition(OrderState.RECONCILIATION_REQUIRED, detail)
        self.store.save()

        report = ReconciliationReport(
            symbol=symbol,
            outcome=ReconciliationOutcome.MISMATCH,
            local=local,
            reported=venue_view,
            detail=detail,
        )
        if self.config.reconciliation_policy is ReconciliationPolicy.FLATTEN:
            self.emergency_flatten(
                symbol,
                FlattenCause.RECONCILIATION_MISMATCH,
                reference_price=venue_view.entry_price or local.entry_price,
            )
        return report

    def resolve_reconciliation(self, symbol: str, adopted: Position, note: str) -> None:
        """An operator's explicit decision about a disputed position.

        The only way ``RECONCILIATION_REQUIRED`` is left, and it takes a written
        reason. Automatic resolution is the thing this state exists to prevent.
        """
        if adopted.symbol != symbol:
            raise FuturesError(f"adopted position is for {adopted.symbol}, not {symbol}")
        if not note:
            raise FuturesError("resolving a reconciliation mismatch requires a stated reason")
        self.store.state.disputed.pop(symbol, None)
        for order in self.store.state.orders.values():
            if (
                order.intent.symbol == symbol
                and order.state is OrderState.RECONCILIATION_REQUIRED
            ):
                order.transition(OrderState.CANCELLED, f"resolved by operator: {note}")
        self.store.state.set_position(adopted)
        self.store.save()
        logger.warning("Futures reconciliation on %s resolved by operator: %s", symbol, note)
        self._publish_position_metrics()

    # ------------------------------------------------------------------
    # the signal path
    # ------------------------------------------------------------------
    def target_for(self, signal: Signal, symbol: str, quantity: Decimal) -> TargetPosition:
        """Pythia's signal as a target position. HOLD means flat, not 'do nothing'.

        Stated here rather than left to callers because the two readings differ
        whenever a position is already open, and picking the wrong one is how a
        strategy ends up holding exposure its model stopped asking for.
        """
        if signal is Signal.LONG:
            return TargetPosition(symbol, PositionSide.LONG, quantity)
        if signal is Signal.SHORT:
            return TargetPosition(symbol, PositionSide.SHORT, quantity)
        return TargetPosition.flat(symbol)

    def execute_target(
        self,
        target: TargetPosition,
        reference_price: Decimal,
        *,
        equity: float,
        stop_price: float | None = None,
        data_delay_s: float | None = None,
        inference_age_s: float | None = None,
        funding_rate: float | None = None,
        exchange_healthy: bool = True,
    ) -> list[OrderRecord]:
        """Take the position from where it is to ``target``. The whole path.

        Returns one :class:`OrderRecord` per leg, in the order they were sent.
        A leg that Aegis vetoed comes back ``REJECTED`` and every later leg is
        abandoned: a reversal whose close succeeded and whose open was refused is
        a flat account, which is a safe place to stop, and a reversal whose close
        was refused must not proceed to the open.
        """
        started = self.clock()
        symbol = target.symbol
        self.require_ready(symbol)
        constraints = self.constraints(symbol)

        current = self.position(symbol)
        intents = plan_transition(current, target)
        if not intents:
            metrics.FUT_SIGNALS.labels(outcome="no_change").inc()
            return []

        records: list[OrderRecord] = []
        for intent in intents:
            record = self._run_intent(
                intent,
                constraints,
                reference_price,
                equity=equity,
                stop_price=stop_price,
                data_delay_s=data_delay_s,
                inference_age_s=inference_age_s,
                funding_rate=funding_rate,
                exchange_healthy=exchange_healthy,
            )
            records.append(record)
            if record.state is not OrderState.FILLED:
                break

        metrics.FUT_SIGNALS.labels(
            outcome=(
                "executed" if records and records[-1].state is OrderState.FILLED else "stopped"
            )
        ).inc()
        metrics.FUT_EXECUTION_LATENCY.observe(max(0.0, self.clock() - started))
        self._publish_position_metrics(reference_price)
        return records

    def emergency_flatten(
        self,
        symbol: str,
        cause: FlattenCause,
        reference_price: Decimal,
    ) -> OrderRecord | None:
        """Drive ``symbol`` to zero, whatever else is true.

        Deliberately reachable while the risk engine is halted, while a
        reconciliation mismatch stands, and after a previous flatten: those are
        the situations it is for. The order is reduce-only and exactly the size of
        the position, so it cannot reverse; a flat position produces no order at
        all and still records the reason, because "we tried to flatten and there
        was nothing there" is a thing an operator needs in the log.
        """
        at = datetime.fromtimestamp(self.clock(), tz=timezone.utc).isoformat()
        self.store.record_flatten(symbol, cause.value, at)
        metrics.FUT_EMERGENCY_FLATTEN.labels(cause=cause.value).inc()

        current = self.position(symbol)
        intents = plan_flatten(current)
        if not intents:
            logger.warning("Emergency flatten on %s (%s): already flat", symbol, cause.value)
            return None

        constraints = self.constraints(symbol)
        record = self._run_intent(
            intents[0],
            constraints,
            reference_price,
            equity=0.0,
            stop_price=None,
            bypass_risk_gate=True,
        )
        logger.critical(
            "Emergency flatten on %s (%s): %s %s -> %s",
            symbol,
            cause.value,
            intents[0].side.value,
            intents[0].quantity,
            self.position(symbol).to_dict(),
        )
        self._publish_position_metrics(reference_price)
        return record

    def settle_funding(self, event: FundingEvent) -> Decimal:
        """Book one funding settlement against the current position.

        Returns the signed cash flow: negative paid, positive received, zero for a
        flat position or a settlement already applied. Funding does not move the
        position and cannot fail an order; it is a cash flow, and the executor
        books it exactly once.
        """
        position = self.position(event.symbol)
        flow = self.ledger.book_funding(position, event)
        if flow < ZERO:
            metrics.FUT_FUNDING.labels(direction="paid").inc(float(-flow))
        elif flow > ZERO:
            metrics.FUT_FUNDING.labels(direction="received").inc(float(flow))
        self.store.save()
        self._publish_pnl_metrics()
        return flow

    def margin(self, symbol: str, mark_price: Decimal) -> MarginState | None:
        """What Aegis is told about this position's solvency. Computed, not judged."""
        return margin_state(self.position(symbol), mark_price, self.constraints(symbol))

    def unrealised(self, symbol: str, mark_price: Decimal) -> Decimal:
        return unrealised_pnl(self.position(symbol), mark_price)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _run_intent(
        self,
        intent: OrderIntent,
        constraints: SymbolConstraints,
        reference_price: Decimal,
        *,
        equity: float,
        stop_price: float | None,
        data_delay_s: float | None = None,
        inference_age_s: float | None = None,
        funding_rate: float | None = None,
        exchange_healthy: bool = True,
        bypass_risk_gate: bool = False,
    ) -> OrderRecord:
        record = self._plan(intent, constraints, reference_price)
        if record.state is not OrderState.PLANNED:
            return record

        if intent.purpose.increases_exposure and not bypass_risk_gate:
            # `record.intent`, not `intent`: `_plan` quantized the quantity down
            # to the venue's step, and Aegis must be asked about the order that
            # will actually be sent. Asking about the pre-quantization quantity
            # over-states the stake — conservative, so it can only over-reject,
            # but the approval, the recorded stake and the prospective
            # liquidation figure would all describe a quantity nobody sent.
            decision = self._ask_aegis(
                record.intent,
                constraints,
                reference_price,
                equity=equity,
                stop_price=stop_price,
                data_delay_s=data_delay_s,
                inference_age_s=inference_age_s,
                funding_rate=funding_rate,
                exchange_healthy=exchange_healthy,
            )
            if not decision.allowed:
                record.transition(OrderState.REJECTED, decision.reason)
                self.store.save()
                metrics.FUT_RISK_VETOES.labels(reason=_veto_label(decision.reason)).inc()
                logger.warning(
                    "Aegis vetoed %s %s %s: %s",
                    intent.purpose.value,
                    intent.side.value,
                    intent.quantity,
                    decision.reason,
                )
                return record
            self.risk.record_order()

        record.transition(OrderState.RISK_APPROVED)
        self.store.save()
        return self._submit(record, reference_price)

    def _plan(
        self, intent: OrderIntent, constraints: SymbolConstraints, reference_price: Decimal
    ) -> OrderRecord:
        """Quantize the intent, check it against the venue, and record it."""
        self._order_seq += 1
        order_id = f"{intent.symbol.replace('/', '').replace(':', '')}-{self._order_seq:06d}"
        metrics.FUT_ORDERS_PLANNED.labels(
            side=intent.side.value, purpose=intent.purpose.value
        ).inc()

        quantity = constraints.quantize_quantity(intent.quantity)
        if quantity != intent.quantity:
            logger.info(
                "%s: quantity %s rounded down to the %s step -> %s",
                intent.symbol,
                intent.quantity,
                constraints.step_size,
                quantity,
            )
        if quantity <= ZERO:
            record = OrderRecord(
                order_id=order_id,
                intent=intent,
                state=OrderState.PLANNED,
            )
            self.store.state.orders[order_id] = record
            record.transition(
                OrderState.REJECTED,
                f"quantity {intent.quantity} rounds to zero at step {constraints.step_size}",
            )
            self.store.save()
            metrics.FUT_ORDERS_REJECTED.labels(reason="below_step").inc()
            return record

        sized = OrderIntent(
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            purpose=intent.purpose,
            reduce_only=intent.reduce_only,
            position_side=intent.position_side,
        )
        record = OrderRecord(order_id=order_id, intent=sized)
        self.store.state.orders[order_id] = record

        try:
            price = constraints.quantize_price(reference_price)
            constraints.check_placeable(quantity, price, reduce_only=sized.reduce_only)
        except ConstraintError as exc:
            record.transition(OrderState.REJECTED, str(exc))
            self.store.save()
            metrics.FUT_ORDERS_REJECTED.labels(reason="venue_constraint").inc()
            logger.warning("Venue constraints refuse %s: %s", order_id, exc)
            return record

        self.store.save()
        return record

    def _ask_aegis(
        self,
        intent: OrderIntent,
        constraints: SymbolConstraints,
        reference_price: Decimal,
        *,
        equity: float,
        stop_price: float | None,
        data_delay_s: float | None,
        inference_age_s: float | None,
        funding_rate: float | None,
        exchange_healthy: bool,
    ) -> RiskDecision:
        """The one call. Everything discretionary is decided on the other side.

        The liquidation price handed over is the one this order would create:
        computed from the position the fill *would* produce, not from the position
        that exists now — which for an opening order is flat and has no
        liquidation price at all. Aegis compares it against
        ``min_liquidation_distance_pct`` and decides; nothing here does.
        """
        entry = float(constraints.quantize_price(reference_price))
        stop = (
            float(stop_price)
            if stop_price is not None
            else self._implied_stop(intent.position_side, entry)
        )
        prospective = self.position(intent.symbol).apply_fill(
            intent.side, intent.quantity, constraints.quantize_price(reference_price)
        )[0]
        state = margin_state(
            prospective, constraints.quantize_price(reference_price), constraints
        )
        return self.risk.evaluate_entry(
            pair=intent.symbol,
            equity=equity,
            entry_price=entry,
            stop_price=stop,
            leverage=float(self.config.leverage),
            proposed_stake=float(intent.quantity * constraints.quantize_price(reference_price))
            / float(self.config.leverage),
            data_delay_s=data_delay_s,
            inference_age_s=inference_age_s,
            funding_rate=funding_rate,
            liquidation_price=None if state is None else float(state.liquidation_price),
            exchange_healthy=exchange_healthy,
        )

    def _implied_stop(self, side: PositionSide, entry: float) -> float:
        """A stop the risk engine's sizing band accepts, when none was supplied.

        Not a trading decision: :meth:`chimera.risk.RiskEngine.position_size`
        refuses a stop outside ``[min_stop_distance_pct, max_stop_distance_pct]``,
        so a caller that has no stop still has to name a distance for the sizing
        arithmetic to be defined. The midpoint of the configured band is the
        neutral choice and it is the risk engine's own configuration, not a number
        invented here.
        """
        limits = self.risk.limits
        distance = (limits.min_stop_distance_pct + limits.max_stop_distance_pct) / 2.0
        return entry * (1 - distance) if side is PositionSide.LONG else entry * (1 + distance)

    def _submit(self, record: OrderRecord, reference_price: Decimal) -> OrderRecord:
        record.transition(OrderState.SUBMITTED)
        self.store.save()
        metrics.FUT_ORDERS_SUBMITTED.labels(
            side=record.intent.side.value, purpose=record.intent.purpose.value
        ).inc()
        try:
            events = self.venue.submit(record.order_id, record.intent, reference_price)
        except (ConstraintError, FuturesError) as exc:
            record.transition(OrderState.FAILED, str(exc))
            self.store.save()
            metrics.FUT_ORDERS_REJECTED.labels(reason="venue_error").inc()
            return record

        for event in events:
            self.apply_event(record.order_id, event, reference_price)
        return self.store.state.orders[record.order_id]

    def apply_event(
        self, order_id: str, event: OrderEvent, reference_price: Decimal
    ) -> OrderRecord:
        """Book one venue event against an order. Idempotent by ``event_id``.

        This is the method a redelivered fill, a replayed journal and a restarted
        process all arrive through, so the duplicate guard is here and nowhere
        else: an event whose id is already recorded returns immediately, having
        changed no state, no quantity, no fee and no position.
        """
        record = self.store.state.orders.get(order_id)
        if record is None:
            raise FuturesError(f"no such order {order_id!r}; an event for it cannot be booked")

        if event.event_id in record.applied_events:
            logger.info(
                "Duplicate futures event %s for %s ignored; exposure unchanged",
                event.event_id,
                order_id,
            )
            return record

        try:
            self._apply_new_event(record, event, reference_price)
        except InvalidTransition:
            metrics.FUT_INVALID_TRANSITIONS.labels(from_state=record.state.value).inc()
            raise

        record.applied_events.append(event.event_id)
        self.store.save()
        return record

    def _apply_new_event(
        self, record: OrderRecord, event: OrderEvent, reference_price: Decimal
    ) -> None:
        intent = record.intent

        if event.kind is EventKind.ACKNOWLEDGED:
            record.transition(OrderState.ACKNOWLEDGED)
            return
        if event.kind is EventKind.CANCELLED:
            record.transition(OrderState.CANCELLED, event.reason)
            return
        if event.kind is EventKind.REJECTED:
            record.transition(OrderState.REJECTED, event.reason)
            metrics.FUT_ORDERS_REJECTED.labels(reason="venue_rejected").inc()
            return
        if event.kind is EventKind.FAILED:
            record.transition(OrderState.FAILED, event.reason)
            return

        # A fill. The position moves first: if it refuses the fill — a reducing
        # fill larger than the position — nothing else has been booked yet.
        if event.quantity > record.remaining_quantity:
            record.transition(
                OrderState.RECONCILIATION_REQUIRED,
                f"fill of {event.quantity} exceeds the {record.remaining_quantity} "
                "outstanding on this order",
            )
            metrics.FUT_RECONCILIATION.labels(
                outcome=ReconciliationOutcome.MISMATCH.value
            ).inc()
            return

        before = self.position(intent.symbol)
        after, realised = before.apply_fill(intent.side, event.quantity, event.price)
        self.store.state.set_position(after)

        record.average_price = (
            record.average_price * record.filled_quantity + event.price * event.quantity
        ) / (record.filled_quantity + event.quantity)
        record.filled_quantity += event.quantity
        record.fees += event.fee

        self.ledger.book_fee(event.fee)
        self.ledger.book_turnover(event.quantity * event.price)
        self.ledger.book_realised(realised)
        metrics.FUT_TRADING_FEES.inc(float(event.fee))
        metrics.FUT_TURNOVER.inc(float(event.quantity * event.price))

        if reference_price > ZERO:
            adverse = (event.price - reference_price) * (
                1 if intent.side is OrderSide.BUY else -1
            )
            metrics.FUT_SLIPPAGE_BPS.observe(
                float(adverse / reference_price * Decimal("10000"))
            )

        complete = record.filled_quantity >= intent.quantity
        metrics.FUT_FILLS.labels(
            side=intent.side.value, kind="full" if complete else "partial"
        ).inc()
        record.transition(OrderState.FILLED if complete else OrderState.PARTIALLY_FILLED)

    # ------------------------------------------------------------------
    def _publish_position_metrics(self, mark_price: Decimal | None = None) -> None:
        gross = ZERO
        net = ZERO
        for symbol, position in self.store.state.positions.items():
            price = mark_price if mark_price and mark_price > ZERO else position.entry_price
            for side in (PositionSide.LONG, PositionSide.SHORT):
                metrics.FUT_POSITION_QUANTITY.labels(symbol=symbol, side=side.value).set(
                    float(position.quantity) if position.side is side else 0.0
                )
            gross += position.notional(price)
            net += position.signed_quantity * price
        metrics.FUT_GROSS_EXPOSURE.set(float(gross))
        metrics.FUT_NET_EXPOSURE.set(float(net))
        self._publish_pnl_metrics()

    def _publish_pnl_metrics(self) -> None:
        net = self.ledger.net_pnl
        metrics.FUT_REALISED_PNL.set(float(self.ledger.realised_pnl))
        metrics.FUT_NET_PNL.set(float(net))
        if net > self._peak_net_pnl:
            self._peak_net_pnl = net
        metrics.FUT_DRAWDOWN.set(float(self._peak_net_pnl - net))


def _veto_label(reason: str) -> str:
    """Collapse an Aegis reason to a bounded metric label.

    Deliberately parallel to ``strategies.common.risk_manager._metric_reason``,
    which does the same job for the spot path, so the two dashboards agree about
    what a veto is called. Two prefixes differ from that function and both are
    corrections rather than divergence: ``RiskEngine`` emits ``"market data
    late: ..."`` and the spot collapse matches ``"market data stale"``, so that
    rejection is currently labelled ``other`` there; and ``"order stake ..."``
    has no entry there at all. Neither is this package's to fix — changing the
    spot labels would move a series a live dashboard already queries — but a new
    table repeating a known-wrong prefix would be a defect, not consistency.
    """
    for prefix, label in (
        ("halted", "halted"),
        ("cooldown", "cooldown"),
        ("exchange", "exchange_unhealthy"),
        ("no equity", "no_equity"),
        ("market data late", "stale_data"),
        ("inference stale", "stale_inference"),
        ("max open positions", "max_positions"),
        ("funding rate", "funding"),
        ("leverage", "leverage"),
        ("liquidation", "liquidation"),
        ("position sizing", "sizing"),
        ("order stake", "stake_above_risk"),
        ("total exposure", "total_exposure"),
        ("exposure in", "asset_exposure"),
    ):
        if reason.startswith(prefix):
            return label
    return "other"
