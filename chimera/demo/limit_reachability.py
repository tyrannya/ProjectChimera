"""Which configured limit is enforced where: R1-k's reachability inventory.

A campaign configuration is hashed into every decision record, and a reviewer
reads its limits off the file as the bounds the campaign ran under. That is only
true of a limit some production code path actually enforces. R1-k found three
that none did -- the funding-cost entry veto (no caller passed a rate), the loss
streak and the cooldown (nothing called ``record_trade_result``) -- and this
module is the record that no configured limit is like that any more.

**One row per name the schema has ever carried.** Every field of
:class:`chimera.demo.config.DemoLimits` and every name in
:data:`chimera.demo.config.RETIRED_LIMITS` has exactly one :class:`LimitReach`,
classified as exactly one of:

``REACHABLE``
    The production path supplies the facts that make BOTH sides of the guard's
    branch possible, and a breach refuses or halts something.
``UNREACHABLE``
    Configured, but nothing production reaches can make it bind. Refused in
    the schema: :func:`check_inventory` fails on any configured limit so
    classified, which is the point -- such a limit must be wired or retired.
``RETIRED_FROM_CONFIG``
    No longer configurable; the parser refuses the name and says why.

**Two checks, of different strength.** :func:`check_inventory` is static: the
table is exhaustive, and each REACHABLE row's ``chain`` -- the functions from
the limit's enforcement to the runner, each with the source token that links it
to the next -- is present in the code. It catches a link being deleted or
renamed, and it cannot prove behaviour. That is the unsafe-default detector's
job (``tests/test_r1k_unsafe_default_detector.py``, its own CI job): for every
configured limit it runs the real :class:`chimera.demo.runner.DemoRunner`
twice over the same synthetic minutes, once on the committed value and once on
a breaching one, and fails unless the committed value is allowed and the breach
is refused or halted with a reason naming that limit. A limit no code path
enforces changes nothing when its value changes, and fails there.

This module opens no file for write, makes no decision and reads no clock.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping

from chimera.demo.config import LIMIT_FIELDS, RETIRED_LIMITS


class Reachability(str, Enum):
    """The three classes a configured-or-retired limit can be in."""

    REACHABLE = "REACHABLE"
    UNREACHABLE = "UNREACHABLE"
    RETIRED_FROM_CONFIG = "RETIRED_FROM_CONFIG"


class LimitReachabilityError(ValueError):
    """The inventory no longer describes the code."""


@dataclass(frozen=True)
class LimitReach:
    """One limit: who enforces it, from what input, supplied by whom, to what effect."""

    config_field: str
    classification: Reachability
    #: The ``RiskLimits`` field it maps to, or ``None`` when retired.
    risk_field: str | None
    #: ``module:Qualname`` of the function that compares against it.
    enforcement: str
    #: The runtime fact the comparison needs.
    runtime_input: str
    #: ``module:Qualname`` of the production code that supplies that fact.
    production_caller: str
    #: Whether a DemoRunner tick (or the READY gate the service runs) reaches it.
    reachable_from_runner: bool
    #: The state mutation a breach makes.
    breach_mutation: str
    #: What a breach refuses or halts.
    refused: str
    #: Whether reductions stay possible while it binds.
    reductions_allowed: bool
    #: Whether a replay of the same files meets it on the same path.
    replay_same_path: bool
    #: What a restart does to the guard.
    restart: str
    #: ``(module:Qualname, token)`` pairs, enforcement first, runner last: each
    #: function's source must contain its token. Empty for a retired limit.
    chain: tuple[tuple[str, str], ...] = ()


_EVALUATE_ENTRY = "chimera.risk:RiskEngine.evaluate_entry"
_ASK_AEGIS = (
    "chimera.futures.executor:FuturesExecutor._ask_aegis",
    "self.risk.evaluate_entry(",
)
_RUN_INTENT = ("chimera.futures.executor:FuturesExecutor._run_intent", "self._ask_aegis(")
_EXECUTE_TARGET = (
    "chimera.futures.executor:FuturesExecutor.execute_target",
    "self._run_intent(",
)
_HEDGE_EXECUTE = ("chimera.carry.hedge:HedgedPosition._execute", "executor.execute_target(")
_HEDGE_APPLY = ("chimera.carry.hedge:HedgedPosition.apply", "self._execute(")
_RUNNER_DECIDE = ("chimera.demo.runner:DemoRunner._decide", "self.position.apply(")
#: The executor-to-runner half every per-order Aegis veto shares.
_ENTRY_PATH = (
    _ASK_AEGIS,
    _RUN_INTENT,
    _EXECUTE_TARGET,
    _HEDGE_EXECUTE,
    _HEDGE_APPLY,
    _RUNNER_DECIDE,
)


def _entry_veto(
    name: str,
    token: str,
    runtime_input: str,
    *,
    refused: str = "the exposure-increasing order",
) -> LimitReach:
    return LimitReach(
        config_field=name,
        classification=Reachability.REACHABLE,
        risk_field=name,
        enforcement=_EVALUATE_ENTRY,
        runtime_input=runtime_input,
        production_caller=_ASK_AEGIS[0],
        reachable_from_runner=True,
        breach_mutation="none: the order is REJECTED before the venue; only the rejected "
        "order record is persisted",
        refused=refused,
        reductions_allowed=True,
        replay_same_path=True,
        restart="stateless comparison; re-evaluated on every order",
        chain=((_EVALUATE_ENTRY, token),) + _ENTRY_PATH,
    )


#: The inventory. Keyed by config name; see the module docstring.
INVENTORY: Mapping[str, LimitReach] = {
    "funding_adverse_streak_limit": LimitReach(
        config_field="funding_adverse_streak_limit",
        classification=Reachability.REACHABLE,
        risk_field="funding_adverse_streak_limit",
        enforcement="chimera.risk:RiskEngine.note_funding_settlement",
        runtime_input="each booked settlement's rate and the perpetual's side",
        production_caller="chimera.demo.runner:DemoRunner._settle_funding",
        reachable_from_runner=True,
        breach_mutation="funding_halt = True (persisted)",
        refused="every exposure increase, until a received settlement or an operator resume",
        reductions_allowed=True,
        replay_same_path=True,
        restart="funding_adverse_streak and funding_halt persist in risk.json",
        chain=(
            (
                "chimera.risk:RiskEngine.note_funding_settlement",
                "self.limits.funding_adverse_streak_limit",
            ),
            ("chimera.demo.runner:DemoRunner._settle_funding", "note_funding_settlement("),
            ("chimera.demo.runner:DemoRunner._safety_checks", "self._settle_funding("),
        ),
    ),
    "max_daily_loss_pct": LimitReach(
        config_field="max_daily_loss_pct",
        classification=Reachability.REACHABLE,
        risk_field="max_daily_loss_pct",
        enforcement="chimera.risk:RiskEngine._record_equity",
        runtime_input="the minute's marked equity and the day's starting equity",
        production_caller="chimera.demo.runner:DemoRunner._decide",
        reachable_from_runner=True,
        breach_mutation="halted = True with the daily-loss reason (persisted)",
        refused="every exposure increase until an operator resume",
        reductions_allowed=True,
        replay_same_path=True,
        restart="day and day_start_equity persist; the day rolls on the decision clock",
        chain=(
            ("chimera.risk:RiskEngine._record_equity", "self.limits.max_daily_loss_pct"),
            ("chimera.risk:RiskEngine.update_equity", "self._record_equity("),
            ("chimera.demo.runner:DemoRunner._decide", "self.risk.update_equity("),
        ),
    ),
    "max_data_delay_s": LimitReach(
        config_field="max_data_delay_s",
        classification=Reachability.REACHABLE,
        risk_field="max_data_delay_s",
        enforcement="chimera.demo.runner:DemoRunner.check_feed",
        runtime_input="the operational clock and the recorder heartbeat's vouched instant",
        production_caller="tools.demo_run:_daemon",
        reachable_from_runner=True,
        breach_mutation="runner state FEED_STALLED (a FEED_STALLED record)",
        refused="every decision until a fresh minute; the held position is still checked",
        reductions_allowed=True,
        replay_same_path=False,
        restart="re-evaluated on the first pass; nothing to restore",
        chain=(
            ("chimera.demo.runner:DemoRunner.check_feed", "self.risk.limits.max_data_delay_s"),
            ("tools.demo_run:_daemon", "runner.check_feed("),
        ),
    ),
    "max_drawdown_pct": LimitReach(
        config_field="max_drawdown_pct",
        classification=Reachability.REACHABLE,
        risk_field="max_drawdown_pct",
        enforcement="chimera.risk:RiskEngine._record_equity",
        runtime_input="the minute's marked equity and the persisted peak",
        production_caller="chimera.demo.runner:DemoRunner._decide",
        reachable_from_runner=True,
        breach_mutation="halted = True with the drawdown reason (persisted)",
        refused="every exposure increase until an operator resume",
        reductions_allowed=True,
        replay_same_path=True,
        restart="peak_equity persists (R1-b)",
        chain=(
            ("chimera.risk:RiskEngine._record_equity", "self.limits.max_drawdown_pct"),
            ("chimera.risk:RiskEngine.update_equity", "self._record_equity("),
            ("chimera.demo.runner:DemoRunner._decide", "self.risk.update_equity("),
        ),
    ),
    "max_exposure_per_asset_pct": _entry_veto(
        "max_exposure_per_asset_pct",
        "lim.max_exposure_per_asset_pct",
        "the pair's persisted exposure plus the order's stake (and, through the derived "
        "max_position_pct, the per-order stake cap)",
    ),
    "max_funding_cost_rate": replace(
        _entry_veto("max_funding_cost_rate", "lim.max_funding_cost_rate", ""),
        runtime_input="MarketState.funding_rate_current (the mark stream's r at the decision "
        "minute's close) and the PERPETUAL intent's position_side",
        production_caller="chimera.carry.hedge:HedgedPosition._funding_input",
        refused="the perpetual leg's exposure-increasing order; perp first, so the spot leg "
        "is never sent after it. A minute without a usable rate is incomplete",
        chain=(
            (_EVALUATE_ENTRY, "lim.max_funding_cost_rate"),
            (_ASK_AEGIS[0], "funding_rate=funding_rate"),
            (_RUN_INTENT[0], "funding_rate=funding_rate"),
            (_EXECUTE_TARGET[0], "funding_rate=funding_rate"),
            (_HEDGE_EXECUTE[0], "self._funding_input(intent, state)"),
            (
                "chimera.carry.hedge:HedgedPosition._funding_input",
                "state.funding_rate_current",
            ),
            ("chimera.demo.feed:FeedCursor.state_for", "funding_rate_current="),
            _HEDGE_APPLY,
            _RUNNER_DECIDE,
        ),
    ),
    "max_leverage": _entry_veto(
        "max_leverage", "lim.max_leverage", "the executor's configured leverage"
    ),
    "max_open_positions": _entry_veto(
        "max_open_positions",
        "lim.max_open_positions",
        "the persisted open pairs, reported by the executor after every fill",
    ),
    "max_orders_per_minute": LimitReach(
        config_field="max_orders_per_minute",
        classification=Reachability.REACHABLE,
        risk_field="max_orders_per_minute",
        enforcement="chimera.risk:RiskEngine.record_order",
        runtime_input="the approval instants in the rolling 60 s window (decision clock)",
        production_caller="chimera.futures.executor:FuturesExecutor._run_intent",
        reachable_from_runner=True,
        breach_mutation="halted = True with the order-rate reason (persisted)",
        refused="the order that tripped it, and every increase after, until a resume. "
        "A backstop: a tick approves at most two increases and minutes are 60 s apart, so "
        "only a limit of 1 binds on today's cadence",
        reductions_allowed=True,
        replay_same_path=True,
        restart="order_times persist (the window is restored)",
        chain=(
            ("chimera.risk:RiskEngine.record_order", "self.limits.max_orders_per_minute"),
            (_RUN_INTENT[0], "self.risk.record_order()"),
            _EXECUTE_TARGET,
            _HEDGE_EXECUTE,
            _HEDGE_APPLY,
            _RUNNER_DECIDE,
        ),
    ),
    "max_total_exposure_pct": _entry_veto(
        "max_total_exposure_pct",
        "lim.max_total_exposure_pct",
        "every pair's persisted exposure plus the order's stake",
    ),
    "min_liquidation_distance_pct": _entry_veto(
        "min_liquidation_distance_pct",
        "lim.min_liquidation_distance_pct",
        "the liquidation price of the position the order would leave (liquidation "
        "unknown is itself a veto)",
    ),
    "loss_streak_limit": LimitReach(
        config_field="loss_streak_limit",
        classification=Reachability.RETIRED_FROM_CONFIG,
        risk_field=None,
        enforcement="chimera.risk:RiskEngine.record_trade_result",
        runtime_input="a per-trade realised result",
        production_caller="none: no caller in chimera/ or tools/",
        reachable_from_runner=False,
        breach_mutation="(would be) cooldown_until = now + cooldown_seconds",
        refused="(would be) every entry until the cooldown expires",
        reductions_allowed=True,
        replay_same_path=True,
        restart="consecutive_losses persists, and stays 0",
    ),
    "cooldown_seconds": LimitReach(
        config_field="cooldown_seconds",
        classification=Reachability.RETIRED_FROM_CONFIG,
        risk_field=None,
        enforcement="chimera.risk:RiskEngine.record_trade_result",
        runtime_input="a loss streak reaching loss_streak_limit",
        production_caller="none: no caller in chimera/ or tools/",
        reachable_from_runner=False,
        breach_mutation="(would be) cooldown_until",
        refused="(would be) every entry until it expires",
        reductions_allowed=True,
        replay_same_path=True,
        restart="cooldown_until persists, and stays 0.0",
    ),
}


def _resolve(qualified: str) -> object:
    module_name, _, attribute_path = qualified.partition(":")
    target: object = importlib.import_module(module_name)
    for part in attribute_path.split("."):
        target = getattr(target, part)
    return target


def check_inventory(
    inventory: Mapping[str, LimitReach] | None = None,
    *,
    configured: Iterable[str] | None = None,
    retired: Iterable[str] | None = None,
) -> None:
    """Refuse while the inventory and the code disagree. Every input injectable,
    so the tests can drive each refusal with a table that really is broken."""
    inventory = INVENTORY if inventory is None else inventory
    configured = set(LIMIT_FIELDS if configured is None else configured)
    retired = set(RETIRED_LIMITS if retired is None else retired)

    if configured & retired:
        raise LimitReachabilityError(
            f"retired limits are still configurable: {sorted(configured & retired)}"
        )
    if missing := (configured | retired) - set(inventory):
        raise LimitReachabilityError(
            f"limits with no reachability row: {sorted(missing)}. Every configured and "
            "every retired limit is classified here"
        )
    if unknown := set(inventory) - (configured | retired):
        raise LimitReachabilityError(f"rows for no such limit: {sorted(unknown)}")

    for name, row in inventory.items():
        if row.config_field != name:
            raise LimitReachabilityError(f"{name}: row names {row.config_field!r}")
        if name in retired:
            if row.classification is not Reachability.RETIRED_FROM_CONFIG:
                raise LimitReachabilityError(
                    f"{name} is retired and classified {row.classification.value}"
                )
            continue
        if row.classification is not Reachability.REACHABLE:
            raise LimitReachabilityError(
                f"{name} is configurable and classified {row.classification.value}. A "
                "configured limit no production path enforces must be wired or retired "
                "from the schema"
            )
        if not row.chain:
            raise LimitReachabilityError(f"{name} is REACHABLE with no chain to the runner")
        for qualified, token in row.chain:
            try:
                source = inspect.getsource(_resolve(qualified))
            except (ImportError, AttributeError, TypeError, OSError) as exc:
                raise LimitReachabilityError(
                    f"{name}: {qualified} cannot be read: {exc}"
                ) from exc
            if token not in source:
                raise LimitReachabilityError(
                    f"{name}: {qualified} no longer contains {token!r}, the link that makes "
                    "this limit reachable. Re-establish the wiring or reclassify the limit"
                )


#: Where the production code lives. A caller anywhere in these trees is one the
#: demo can reach; `strategies/` (the retired Freqtrade path) cannot.
PRODUCTION_TREES: tuple[str, ...] = ("chimera", "tools")


def record_trade_result_callers(
    root: Path | str | None = None, trees: Iterable[str] = PRODUCTION_TREES
) -> list[str]:
    """``path:line`` of every call of ``record_trade_result`` in the production trees.

    Empty is what retiring the loss streak and the cooldown rests on. A caller
    appearing here means a loss streak could open a cooldown on the
    ``RiskLimits`` DEFAULT, a limit nobody configured, so the two have to come
    back into the campaign schema as configured limits first.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    found: list[str] = []
    for tree in trees:
        for path in sorted((base / tree).rglob("*.py")):
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(module):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                )
                if name == "record_trade_result":
                    found.append(f"{path.relative_to(base).as_posix()}:{node.lineno}")
    return found


__all__ = [
    "INVENTORY",
    "PRODUCTION_TREES",
    "LimitReach",
    "LimitReachabilityError",
    "Reachability",
    "check_inventory",
    "record_trade_result_callers",
]
