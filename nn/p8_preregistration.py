"""P8's design, committed as a design. **P8 is NOT OPENED.**

No router exists, no P8 fit has been made, no P8 number exists, and
:data:`OUTCOME` says so. The design is committed now because that is the only
moment at which a routing rule can be fixed without being fitted to the answer.

**Its precondition is unmet, which is why it is not opened.** P6 found no viable
clock, P6-EXT found neither slow clock viable, and P7 found neither measured
consensus supportive — so no trading mode is eligible under
``docs/trading_modes_v1.md`` §2. A router choosing among zero eligible modes
would be choosing among nothing, and a router evaluated on ineligible modes would
be evaluating something the programme has already declined to trade.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CHECKPOINT = "P8"

#: The one thing about P8 that is currently true.
OUTCOME = "NOT OPENED"

QUESTION = (
    "Can a causal automatic router choose among eligible SCALPING, DAY_TRADING and SWING "
    "modes in a way that provides robust cost-aware incremental value over fixed-mode "
    "operation?"
)

NOT_OPENED_BECAUSE = (
    "No mode is eligible. P6 found none of 1m, 5m, 15m, 30m, 1h viable; P6-EXT found "
    "neither 4h nor 1d viable; P7 found neither measured consensus supportive. P8 opens "
    "when at least two modes are eligible under docs/trading_modes_v1.md section 2."
)

OPENING_CONDITION = "at least two trading modes eligible under trading_modes_v1 section 2"

EVIDENCE_CEILING = (
    "Exploratory and adaptive when it opens: the four outer blocks will have been read by "
    "every checkpoint before it. No P8 result could be confirmatory on them."
)

#: Only quantities causally available at the decision timestamp, and only frozen
#: ones. The list is closed: a router needing something outside it is a different
#: checkpoint.
PERMITTED_INPUTS: tuple[str, ...] = (
    "realised volatility state",
    "ATR / range state",
    "trend efficiency and persistence",
    "directional consistency",
    "current cross-timeframe Pythia agreement",
    "calibrated specialist confidence",
    "disagreement between fast and slow specialists",
    "expected edge after the frozen transaction-cost model",
    "turnover burden",
    "liquidity, spread and slippage proxies available at that instant",
)

NO_NEW_FEATURE_FAMILY = (
    "No new predictive feature family may be introduced to make the router look better. A "
    "router that needed one would be a different checkpoint."
)

#: Forbidden permanently, not merely until the first result.
FORBIDDEN_INPUTS: tuple[str, ...] = (
    "recent realised mode PnL used as a winner selector",
    "outer-fold performance",
    "backtest rank",
    "post-hoc best timeframe",
    "post-hoc best horizon",
    "manual knowledge of which mode wins the current period",
)

#: What the router emits at every decision timestamp.
CONTRACT: tuple[str, ...] = (
    "mode",
    "mode_confidence",
    "eligible_modes",
    "reason_code",
    "expected_cost",
    "expected_edge",
    "consensus_state",
)

DETERMINISM = "deterministic given identical input state"

FLAT_IS_SUCCESS = (
    "FLAT is a first-class successful outcome. The system remains FLAT when no eligible "
    "mode has sufficient frozen-model confidence, when cost-adjusted edge is not positive, "
    "when specialists materially disagree, when required context is unavailable, or when "
    "Aegis rejects the intended exposure."
)

ARCHITECTURE = (
    "timeframe specialists -> per-mode Pythia consensus -> AUTO Mode Router -> Aegis -> "
    "Hermes -> Binance USD-M futures dry-run venue -> Argus"
)

#: Unchanged by any router, and not expressible by one.
SAFETY_PROHIBITIONS: tuple[str, ...] = (
    "dynamically increase leverage",
    "switch instrument family",
    "enable margin borrowing",
    "switch exchange",
    "choose a different coin because BTC is difficult",
    "open multiple contradictory modes simultaneously",
    "override emergency flatten",
    "override a reconciliation dispute",
    "override an Aegis veto",
)

ONE_MODE_AT_A_TIME = (
    "One active directional mode at a time in v1. Transitions are explicit and observable "
    "and follow chimera.modes.plan_mode_transition: a mode change with an open position "
    "flattens and reconciles before the new mode may act."
)

#: The fixed-mode controls, evaluated on the same timestamp universe and the same
#: accounting as AUTO.
CONTROLS: tuple[str, ...] = (
    "always-SCALPING when eligible",
    "always-DAY_TRADING when eligible",
    "always-SWING when eligible",
    "the AUTO router",
)

CASH_REFERENCE = "FLAT/cash is included as a reference, not as a required competitor"

BENCHMARK = {
    "definition": (
        "best_fixed_mode_return = max over the eligible fixed-mode controls, within each "
        "outer fold"
    ),
    "delta": "AUTO_delta = AUTO_return - best_fixed_mode_return",
    "forbidden": (
        "comparing AUTO against whichever fixed mode looked best over the whole experiment "
        "without accounting for that selection"
    ),
}

#: The turnover guard, frozen here and not movable afterwards. A router that
#: beats fixed-mode operation by trading materially more has not shown routing
#: helps; it has shown turnover was mispriced.
TURNOVER_GUARD = {
    "rule": (
        "AUTO's total realised turnover across the four outer folds may not exceed 1.25x "
        "the turnover of the fold-wise best fixed-mode control it is compared against, "
        "summed over the same folds"
    ),
    "multiple": 1.25,
    "frozen": "before any evaluation; it may not be moved afterwards",
}

DECISION_RULE = {
    "conditions": [
        "AUTO_delta > 0 in at least 3 of the 4 temporal folds",
        "mean AUTO_delta > 0",
        "AUTO does not obtain its advantage through materially larger turnover or cost "
        "exposure — the frozen turnover guard",
    ],
    "conjunction": "all three",
    "improved_folds_required": 3,
    "total_folds": 4,
}

RESULT_STATES: tuple[str, ...] = (
    "P8 AUTO MODE ROUTER: SUPPORTIVE ADAPTIVE",
    "P8 AUTO MODE ROUTER: NEGATIVE",
    "P8 AUTO MODE ROUTER: INVALID",
    "P8 AUTO MODE ROUTER: NOT OPENED",
)

CURRENT_RESULT_STATE = "P8 AUTO MODE ROUTER: NOT OPENED"

TELEMETRY = (
    "bounded cardinality only: selected mode, eligible-mode mask, transitions, dwell time, "
    "FLAT fraction, router confidence bucket, consensus/disagreement state, Aegis vetoes by "
    "mode, and turnover, fees, funding, slippage, exposure, drawdown and reconciliation "
    "errors by mode. Never a raw free-text reason, an order id, or an unbounded symbol."
)

FORBIDDEN_AFTER_RESULTS: tuple[str, ...] = (
    "changing any router input, or adding a feature family to improve it",
    "changing the fixed-mode controls or the fold-wise best-fixed-mode benchmark",
    "changing any of the three decision conditions, or the 1.25x turnover guard",
    "reporting AUTO against a fixed mode chosen after seeing the experiment",
    "writing auto_router_v2 in the task that produced v1's result",
)

STOPPING_RULE = {
    "on_supportive": (
        "adaptive evidence that routing added value over fixed-mode operation on burned "
        "blocks. Not a deployable strategy and not permission for real money."
    ),
    "on_negative": "recorded as negative; auto_router_v2 is not written in the same task",
    "on_not_opened": (
        "the precondition is unmet and nothing has been measured. This is the current state."
    ),
}


def payload() -> dict[str, Any]:
    return {
        "checkpoint": CHECKPOINT,
        "outcome": OUTCOME,
        "question": QUESTION,
        "not_opened_because": NOT_OPENED_BECAUSE,
        "opening_condition": OPENING_CONDITION,
        "evidence_ceiling": EVIDENCE_CEILING,
        "permitted_inputs": list(PERMITTED_INPUTS),
        "no_new_feature_family": NO_NEW_FEATURE_FAMILY,
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
        "contract": list(CONTRACT),
        "determinism": DETERMINISM,
        "flat_is_success": FLAT_IS_SUCCESS,
        "architecture": ARCHITECTURE,
        "safety_prohibitions": list(SAFETY_PROHIBITIONS),
        "one_mode_at_a_time": ONE_MODE_AT_A_TIME,
        "controls": list(CONTROLS),
        "cash_reference": CASH_REFERENCE,
        "benchmark": BENCHMARK,
        "turnover_guard": TURNOVER_GUARD,
        "decision_rule": DECISION_RULE,
        "result_states": list(RESULT_STATES),
        "current_result_state": CURRENT_RESULT_STATE,
        "telemetry": TELEMETRY,
        "forbidden_after_results": list(FORBIDDEN_AFTER_RESULTS),
        "stopping_rule": STOPPING_RULE,
    }


def preregistration_hash() -> str:
    blob = json.dumps(payload(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def describe() -> dict[str, Any]:
    return {"preregistration_hash": preregistration_hash(), **payload()}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(describe(), indent=2))
