"""Freqtrade adapter for :mod:`chimera.risk`.

:class:`RiskAwareStrategy` is the base class every ProjectChimera strategy
inherits. It wires the risk engine into the Freqtrade callbacks that are
actually on the execution path in the installed version (verified against
Freqtrade 2026.7):

``bot_loop_start``
    refresh equity, recompute drawdown, publish metrics.

``custom_stake_amount``
    ask the risk engine to size the position from the stop distance.

``confirm_trade_entry``
    the gate. Returning False here stops the order before it is placed, so a
    halted or over-exposed account cannot open a trade no matter what the
    signal says.

``order_filled``
    track exposure, loss streaks and the order rate.

The engine's ``halted`` flag is local state checked synchronously in
``confirm_trade_entry`` — no HTTP call, no external service, nothing that can
fail open.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from freqtrade.persistence import Order, Trade
from freqtrade.strategy import IStrategy

from chimera import metrics
from chimera.notify import TelegramNotifier
from chimera.risk import RiskEngine, RiskLimits

logger = logging.getLogger(__name__)


class RiskAwareStrategy(IStrategy):
    """Base strategy with central risk controls and optional notifications.

    Reads its limits from the Freqtrade config under a ``risk`` key::

        "risk": {
            "max_drawdown_pct": 0.15,
            "risk_per_trade_pct": 0.01,
            "state_file": "user_data/risk_state.json"
        }
    """

    #: Freqtrade rejects a `protections` key in the config with
    #: ConfigurationError ("DEPRECATED: Setting 'protections' in the
    #: configuration is deprecated"), so they belong on the strategy. These
    #: complement the risk engine rather than replacing it: Freqtrade enforces
    #: them per-pair inside its own execution loop.
    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 2},
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 48,
            "trade_limit": 10,
            "stop_duration_candles": 12,
            "max_allowed_drawdown": 0.1,
        },
    ]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        risk_config = dict(config.get("risk", {}))
        state_file = risk_config.pop("state_file", None)
        self.risk = RiskEngine(RiskLimits.from_dict(risk_config), state_path=state_file)
        self.notifier = TelegramNotifier()
        self._starting_equity: float | None = None
        #: Leverage last granted per pair by the leverage() callback. Freqtrade
        #: does not pass leverage to confirm_trade_entry, but the gate needs it
        #: to turn the order's `amount` back into a stake.
        self._leverage: dict[str, float] = {}
        logger.info("Risk engine active: %s", self.risk.limits.to_dict())
        self._start_metrics_server(config)

    def _start_metrics_server(self, config: dict) -> None:
        """Expose /metrics so the trading gauges reach Prometheus.

        Only in live and dry-run: a backtest would bind the port once per
        process for metrics nobody scrapes, and hyperopt runs many processes.
        Failure to bind is logged, never fatal — monitoring is not load-bearing.
        """
        if config.get("runmode") not in ("live", "dry_run"):
            return
        port = int(config.get("metrics_port", 9101))
        try:
            metrics.serve_metrics(port)
        except OSError as exc:
            logger.warning("Could not start the metrics server on port %d: %s", port, exc)

    # ------------------------------------------------------------------
    def _equity(self) -> float:
        """Current total equity in the stake currency, 0.0 if unavailable."""
        wallets = getattr(self, "wallets", None)
        if wallets is None:
            return 0.0
        try:
            return float(wallets.get_total_stake_amount())
        except Exception as exc:  # noqa: BLE001 - wallet access must not kill the loop
            logger.warning("Could not read wallet equity: %s", exc)
            return 0.0

    def bot_loop_start(self, current_time: datetime, **kwargs: Any) -> None:
        equity = self._equity()
        if equity > 0:
            if self._starting_equity is None:
                self._starting_equity = equity
            self.risk.update_equity(equity, now=current_time)

        metrics.EQUITY.set(equity)
        metrics.DRAWDOWN.set(self.risk.current_drawdown())
        metrics.EXPOSURE.set(self.risk.total_exposure)
        metrics.OPEN_POSITIONS.set(len(self.risk.state.open_positions))
        metrics.RISK_HALTED.set(1 if self.risk.halted else 0)
        if self._starting_equity:
            metrics.PNL_TOTAL.set(equity - self._starting_equity)

        if self.risk.halted:
            # Deduplicated inside the notifier, so a persistent halt sends one
            # message rather than one per bot iteration.
            self.notifier.send_error(
                "Trading halted by risk engine", self.risk.state.halt_reason
            )

    # ------------------------------------------------------------------
    def _stop_price(self, rate: float, side: str) -> float:
        """Price the configured stoploss implies for an entry at ``rate``."""
        distance = abs(float(self.stoploss))
        return rate * (1 - distance) if side == "long" else rate * (1 + distance)

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        """Risk-based sizing, clamped to what the exchange will accept.

        Note what this is *not*: ``risk_per_trade_pct`` is not the fraction of
        the wallet to spend. It is the fraction of equity lost if the stop is
        hit, so the stake is ``risk_capital / stop_distance``.
        """
        # Freqtrade passes the leverage it will actually use here, and does not
        # pass it to confirm_trade_entry. Record it so the gate can convert that
        # callback's `amount` back into a stake.
        self._record_leverage(pair, leverage)

        equity = self._equity()
        if equity <= 0:
            return proposed_stake

        stake = self.risk.position_size(
            equity, current_rate, self._stop_price(current_rate, side), leverage
        )
        if stake <= 0:
            # Zero means "do not size this trade"; confirm_trade_entry will
            # reject it. Returning 0 here can trip Freqtrade's min-stake check
            # with a confusing message, so hand back the proposal and let the
            # gate below make the real decision.
            return proposed_stake

        stake = min(stake, max_stake)
        if min_stake is not None:
            stake = max(stake, min_stake)
        return stake

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> bool:
        """The single gate every entry passes through.

        Judges the order Freqtrade is actually about to send. ``custom_stake_amount``
        proposes a size, but ``FreqtradeBot.get_valid_enter_price_and_stake`` may
        then raise it to the exchange minimum, cap it, or round it — and only
        the ``amount``/``rate`` arriving here reflect that outcome.
        """
        equity = self._equity()
        stake = self._committed_stake(amount, rate, pair)
        decision = self.risk.evaluate_entry(
            pair=pair,
            equity=equity,
            entry_price=rate,
            stop_price=self._stop_price(rate, side),
            leverage=self._leverage_for(pair),
            proposed_stake=stake,
            data_delay_s=self.data_delay_seconds(pair, current_time),
            inference_age_s=self.inference_age_seconds(pair),
        )
        if not decision.allowed:
            logger.info(
                "Entry on %s rejected by risk engine (stake %.2f): %s",
                pair,
                stake,
                decision.reason,
            )
            metrics.REJECTED_ENTRIES.labels(reason=_metric_reason(decision.reason)).inc()
            return False

        # Counts orders this gate has approved for submission, which is the rate
        # the exchange actually sees. Rejected signals are not orders, and
        # counting fills instead would let a burst of unfilled orders through.
        self.risk.record_order()
        if self.risk.halted:
            # record_order can trip the rate limit; re-check before approving.
            metrics.REJECTED_ENTRIES.labels(reason="order_rate").inc()
            return False
        return True

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        """Clamp leverage to the risk limit and remember it for the entry gate.

        Freqtrade does not pass leverage to ``confirm_trade_entry``, but it does
        use the value returned here when computing
        ``amount = (stake / rate) * leverage``. Recording it is what lets the
        gate turn ``amount`` back into a stake. (The previous code read
        ``self.config["leverage"]``, which is not a Freqtrade config key and so
        was always the 1.0 default even on a futures config.)
        """
        allowed = max(1.0, min(proposed_leverage, max_leverage, self.risk.limits.max_leverage))
        self._record_leverage(pair, allowed)
        return allowed

    def _record_leverage(self, pair: str, leverage: float) -> None:
        self._leverage[pair] = float(leverage)

    def _leverage_for(self, pair: str) -> float:
        """Leverage last granted for ``pair``; 1.0 on spot, where it never runs."""
        return self._leverage.get(pair, 1.0)

    def _committed_stake(self, amount: float, rate: float, pair: str) -> float:
        """Quote-currency stake implied by a Freqtrade entry order.

        ``FreqtradeBot.execute_entry`` computes ``amount = (stake / rate) *
        leverage``, so the margin actually committed is ``amount * rate /
        leverage``. On spot, leverage is 1 and this is simply the notional.
        """
        leverage = max(1.0, self._leverage_for(pair))
        return float(amount) * float(rate) / leverage

    def order_filled(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs: Any
    ) -> None:
        """Track exposure and trade outcomes as orders complete.

        Freqtrade calls this for every order reaching a closed state — including
        partial fills, position adjustments and partial exits — and
        ``trade.stake_amount`` is recomputed as the trade's *total* stake each
        time. So exposure is **set**, never accumulated.
        """
        super().order_filled(pair, trade, order, current_time, **kwargs)
        try:
            if trade.is_open:
                self.risk.set_position_exposure(pair, float(trade.stake_amount or 0.0))
                return

            self.risk.close_position(pair)
            profit = float(trade.close_profit_abs or 0.0)
            self.risk.record_trade_result(profit)
            metrics.TRADES_TOTAL.labels(result="win" if profit >= 0 else "loss").inc()
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not break trading
            logger.warning("Risk bookkeeping failed for %s: %s", pair, exc)

    # ------------------------------------------------------------------
    # Overridable freshness hooks. Subclasses that consume market data or
    # model predictions report their freshness here.
    # ------------------------------------------------------------------
    def data_delay_seconds(self, pair: str, current_time: datetime) -> float | None:
        """Seconds the newest candle is late past its close, or None if unknown."""
        return None

    def inference_age_seconds(self, pair: str) -> float | None:
        return None


def _metric_reason(reason: str) -> str:
    """Collapse a human reason to a low-cardinality metric label.

    Prometheus labels must not carry free text: embedding numbers would create
    a new time series per rejection.
    """
    for prefix, label in (
        ("halted", "halted"),
        ("cooldown", "cooldown"),
        ("exchange", "exchange_unhealthy"),
        ("no equity", "no_equity"),
        ("market data stale", "stale_data"),
        ("inference stale", "stale_inference"),
        ("max open positions", "max_positions"),
        ("funding rate", "funding"),
        ("leverage", "leverage"),
        ("liquidation", "liquidation"),
        ("position sizing", "sizing"),
        ("total exposure", "total_exposure"),
        ("exposure in", "asset_exposure"),
    ):
        if reason.startswith(prefix):
            return label
    return "other"
