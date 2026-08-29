"""Binance USD-M perpetual futures execution, dry-run only.

Futures Execution v1 is **engineering, not research**. It exists so that a future
strategy can express SHORT exposure safely; it makes no claim about whether any
strategy should. Nothing it measures — simulated PnL, slippage, fees, funding —
may select a model, a feature, a threshold, a horizon or a target. P4's negative
answer was about derivatives as *information*; this package is about futures as
an *instrument*, and the two are independent questions.

Scope, all of it deliberate and all of it enforced rather than documented:

* isolated margin, exactly 1x leverage (:class:`FuturesExecutionConfig` refuses
  anything else at construction);
* LONG and SHORT as first-class position sides, never a signed quantity;
* an explicit order state machine whose invalid transitions raise;
* venue constraints that fail closed on anything missing or contradictory;
* trading fees, funding paid and funding received accounted separately;
* deterministic simulated fills, including partial ones;
* reconciliation that refuses to overwrite local state with a reported one;
* emergency flatten that reaches zero and cannot reverse;
* restart recovery that never assumes flat from an empty memory;
* **no live-order path, no credential, and no network client anywhere in the
  package** — asserted by ``tests/test_futures_no_live_path.py``.

The risk boundary is unchanged: :class:`chimera.risk.RiskEngine` is the sole
portfolio and risk authority, and :class:`FuturesExecutor` reaches the venue only
through it.
"""

from chimera.futures.accounting import (
    AccountingError,
    FundingEvent,
    Ledger,
    MarginState,
    funding_cash_flow,
    liquidation_price,
    margin_state,
    unrealised_pnl,
)
from chimera.futures.domain import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    EventKind,
    FuturesError,
    InvalidTransition,
    OrderEvent,
    OrderIntent,
    OrderPurpose,
    OrderRecord,
    OrderSide,
    OrderState,
    Position,
    PositionError,
    PositionSide,
    TargetPosition,
    can_transition,
    gross_exposure,
    net_exposure,
    plan_flatten,
    plan_transition,
)
from chimera.futures.executor import (
    FlattenCause,
    FuturesExecutionConfig,
    FuturesExecutor,
    LiveFuturesNotImplemented,
    NotBootstrapped,
    ReconciliationOutcome,
    ReconciliationPolicy,
    ReconciliationReport,
    ReconciliationRequired,
)
from chimera.futures.store import (
    STORE_SCHEMA,
    FuturesState,
    FuturesStore,
    LoadOutcome,
    StoreError,
)
from chimera.futures.venue import (
    SUPPORTED_ORDER_TYPES,
    TRADABLE_STATUS,
    ConstraintError,
    ConstraintSource,
    DeterministicFillModel,
    DryRunFuturesVenue,
    FillModel,
    FillPlan,
    StaticConstraintSource,
    SymbolConstraints,
    default_constraints_table,
    load_constraint_source,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "AccountingError",
    "ConstraintError",
    "ConstraintSource",
    "DeterministicFillModel",
    "DryRunFuturesVenue",
    "EventKind",
    "FillModel",
    "FillPlan",
    "FlattenCause",
    "FundingEvent",
    "FuturesError",
    "FuturesExecutionConfig",
    "FuturesExecutor",
    "FuturesState",
    "FuturesStore",
    "InvalidTransition",
    "Ledger",
    "LiveFuturesNotImplemented",
    "LoadOutcome",
    "MarginState",
    "NotBootstrapped",
    "OrderEvent",
    "OrderIntent",
    "OrderPurpose",
    "OrderRecord",
    "OrderSide",
    "OrderState",
    "Position",
    "PositionError",
    "PositionSide",
    "ReconciliationOutcome",
    "ReconciliationPolicy",
    "ReconciliationReport",
    "ReconciliationRequired",
    "STORE_SCHEMA",
    "SUPPORTED_ORDER_TYPES",
    "StaticConstraintSource",
    "StoreError",
    "SymbolConstraints",
    "TERMINAL_STATES",
    "TRADABLE_STATUS",
    "TargetPosition",
    "can_transition",
    "default_constraints_table",
    "funding_cash_flow",
    "gross_exposure",
    "liquidation_price",
    "load_constraint_source",
    "margin_state",
    "net_exposure",
    "plan_flatten",
    "plan_transition",
    "unrealised_pnl",
]
