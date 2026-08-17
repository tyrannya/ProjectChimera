"""Central risk engine.

Every entry in the system passes through :meth:`RiskEngine.evaluate_entry`.
The engine owns account state (equity, drawdown, exposure, open positions),
trade sizing, and the kill switch.

Design notes
------------

*The kill switch is local state, not a network call.* The previous
implementation fired ``requests.post("http://localhost:8080/api/v1/stop")`` and
treated that as the guarantee. A guard that depends on an HTTP round trip
succeeding is not a guard. Here, :attr:`halted` is checked synchronously at the
top of ``evaluate_entry``, so a halted engine cannot approve an entry even if
every network path in the process is down. The halt is additionally persisted
to disk so a process restart does not silently clear it.

*Sizing is risk-based, not wallet-fraction-based.* ``risk_per_trade_pct`` is the
fraction of equity lost **if the stop is hit**, which is what "1% risk" means.
Converting that to a position size requires dividing by the stop distance; using
1% of the wallet as the position size instead would mean the actual risk is
0.01 * stop_distance, i.e. 20x smaller than intended for a 5% stop.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class RiskViolation(Exception):
    """Raised by the guards that are meant to abort an operation outright."""


@dataclass(frozen=True)
class RiskLimits:
    """Configured limits. All fractions are of current equity unless noted."""

    # --- account limits -------------------------------------------------
    max_drawdown_pct: float = 0.15
    max_daily_loss_pct: float = 0.05
    max_open_positions: int = 3
    max_total_exposure_pct: float = 1.0
    max_exposure_per_asset_pct: float = 0.35

    # --- per-trade limits -----------------------------------------------
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.25
    max_leverage: float = 3.0
    min_stop_distance_pct: float = 0.005
    max_stop_distance_pct: float = 0.15

    # --- operational guards ---------------------------------------------
    max_orders_per_minute: int = 10
    loss_streak_limit: int = 3
    cooldown_seconds: float = 3600.0
    #: How late the newest candle may be *past its close* before entries are
    #: blocked. Not the candle's age: an OHLCV timestamp is the candle's OPEN
    #: time, so a just-closed 1h candle is already 3600s "old" while being
    #: perfectly fresh. Freqtrade measures the same way in
    #: ``IStrategy.ignore_expired_candle``. Because this is a delay past close,
    #: one value is meaningful across every timeframe.
    max_data_delay_s: float = 300.0
    max_inference_staleness_s: float = 300.0

    # --- futures guards --------------------------------------------------
    max_funding_rate: float = 0.0005
    min_liquidation_distance_pct: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RiskLimits":
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in data.items() if k in fields})


@dataclass
class RiskDecision:
    """Result of an entry evaluation."""

    allowed: bool
    reason: str
    #: Stake in quote currency. Zero when ``allowed`` is False.
    stake: float = 0.0

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class RiskState:
    """Mutable account state tracked by the engine."""

    equity: float = 0.0
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day: str = ""
    open_positions: dict[str, float] = field(default_factory=dict)
    order_times: list[float] = field(default_factory=list)
    consecutive_losses: int = 0
    cooldown_until: float = 0.0
    halted: bool = False
    halt_reason: str = ""


class RiskEngine:
    """Stateful risk controller. Not thread-safe; one per bot process."""

    def __init__(
        self,
        limits: RiskLimits | None = None,
        state_path: str | Path | None = None,
        clock=time.time,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.state = RiskState()
        self._clock = clock
        self._state_path = Path(state_path) if state_path else None
        self._load_halt()

    # ------------------------------------------------------------------
    # kill switch
    # ------------------------------------------------------------------
    @property
    def halted(self) -> bool:
        return self.state.halted

    def halt(self, reason: str) -> None:
        """Block all new entries. Idempotent: re-halting does not re-alert."""
        if self.state.halted:
            return
        self.state.halted = True
        self.state.halt_reason = reason
        logger.critical("RISK HALT: %s", reason)
        self._persist_halt()

    def resume(self) -> None:
        """Clear the halt. Only ever called by an explicit operator action."""
        self.state.halted = False
        self.state.halt_reason = ""
        logger.warning("Risk halt cleared by operator")
        self._persist_halt()

    def _persist_halt(self) -> None:
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(
                    {
                        "halted": self.state.halted,
                        "halt_reason": self.state.halt_reason,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            )
        except OSError as exc:
            # A halt that cannot be written down is still a halt in memory; log
            # loudly rather than letting the write failure unwind the halt.
            logger.error("Could not persist risk halt state: %s", exc)

    def _load_halt(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text())
        except (OSError, ValueError) as exc:
            logger.error("Could not read risk halt state, assuming halted: %s", exc)
            self.state.halted = True
            self.state.halt_reason = "unreadable persisted risk state"
            return
        if data.get("halted"):
            self.state.halted = True
            self.state.halt_reason = str(data.get("halt_reason", "persisted halt"))
            logger.critical(
                "Starting in HALTED state from %s: %s",
                self._state_path,
                self.state.halt_reason,
            )

    # ------------------------------------------------------------------
    # account state
    # ------------------------------------------------------------------
    def update_equity(self, equity: float, now: datetime | None = None) -> None:
        """Record current equity and trip account-level guards if breached."""
        if equity <= 0:
            self.halt(f"equity is non-positive: {equity}")
            return

        today = (now or datetime.now(timezone.utc)).date().isoformat()
        if self.state.day != today:
            self.state.day = today
            self.state.day_start_equity = equity

        self.state.equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)

        drawdown = (self.state.peak_equity - equity) / self.state.peak_equity
        if drawdown >= self.limits.max_drawdown_pct:
            self.halt(
                f"max drawdown breached: {drawdown:.2%} >= "
                f"{self.limits.max_drawdown_pct:.2%}"
            )
            return

        if self.state.day_start_equity > 0:
            daily_loss = (self.state.day_start_equity - equity) / self.state.day_start_equity
            if daily_loss >= self.limits.max_daily_loss_pct:
                self.halt(
                    f"max daily loss breached: {daily_loss:.2%} >= "
                    f"{self.limits.max_daily_loss_pct:.2%}"
                )

    def current_drawdown(self) -> float:
        if self.state.peak_equity <= 0:
            return 0.0
        return (self.state.peak_equity - self.state.equity) / self.state.peak_equity

    def record_order(self) -> None:
        """Register an order for rate limiting. Halts if the rate is exceeded."""
        now = self._clock()
        self.state.order_times = [t for t in self.state.order_times if now - t < 60.0]
        self.state.order_times.append(now)
        if len(self.state.order_times) > self.limits.max_orders_per_minute:
            self.halt(
                f"order rate limit exceeded: {len(self.state.order_times)} orders/min > "
                f"{self.limits.max_orders_per_minute}"
            )

    def record_trade_result(self, profit_abs: float) -> None:
        """Track the loss streak and start a cooldown when it is hit."""
        if profit_abs < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.limits.loss_streak_limit:
                self.state.cooldown_until = self._clock() + self.limits.cooldown_seconds
                logger.warning(
                    "%d consecutive losses, entries paused for %.0fs",
                    self.state.consecutive_losses,
                    self.limits.cooldown_seconds,
                )
        else:
            self.state.consecutive_losses = 0

    def set_position_exposure(self, pair: str, stake: float) -> None:
        """Record ``pair``'s *current total* exposure, replacing any previous value.

        Assignment, not accumulation. Freqtrade calls ``order_filled`` once per
        order that reaches a closed state — entries, partial fills, position
        adjustments and partial exits alike — and
        ``LocalTrade.recalc_trade_from_orders`` rewrites ``trade.stake_amount``
        to the trade's total stake each time. Adding that total on every
        callback double-counted: two fills of one 200 position reported 400.

        Keyed by pair because Freqtrade opens at most one trade per pair — it
        removes pairs with an open trade from the candidate whitelist in
        ``FreqtradeBot.enter_positions``. Position adjustments extend that one
        trade rather than creating a second.
        """
        if stake > 0:
            self.state.open_positions[pair] = stake
        else:
            self.state.open_positions.pop(pair, None)

    def close_position(self, pair: str) -> None:
        self.state.open_positions.pop(pair, None)

    @property
    def total_exposure(self) -> float:
        return sum(self.state.open_positions.values())

    # ------------------------------------------------------------------
    # sizing
    # ------------------------------------------------------------------
    def position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        leverage: float = 1.0,
    ) -> float:
        """Stake (in quote currency) such that hitting the stop costs
        ``risk_per_trade_pct`` of equity.

        Returns 0.0 when the stop distance is outside the configured band: too
        tight a stop implies an unbounded position, too wide a stop means the
        trade's risk cannot be controlled.

        This deliberately does *not* shrink the stake to fit the remaining
        exposure headroom. Sizing answers "how large should this trade be?";
        whether there is room for it is :meth:`evaluate_entry`'s decision. Were
        sizing to clamp instead, a portfolio at 97% of its exposure cap would
        quietly open a position 3% of the intended size — all of the fees, none
        of the edge — and the exposure limit would never actually reject
        anything.
        """
        if equity <= 0 or entry_price <= 0:
            return 0.0

        stop_distance = abs(entry_price - stop_price) / entry_price
        if not (
            self.limits.min_stop_distance_pct
            <= stop_distance
            <= self.limits.max_stop_distance_pct
        ):
            logger.warning(
                "Stop distance %.4f outside [%.4f, %.4f], refusing to size",
                stop_distance,
                self.limits.min_stop_distance_pct,
                self.limits.max_stop_distance_pct,
            )
            return 0.0

        risk_capital = equity * self.limits.risk_per_trade_pct
        # Notional such that a `stop_distance` adverse move loses risk_capital.
        notional = risk_capital / stop_distance

        capped_leverage = max(1.0, min(leverage, self.limits.max_leverage))
        # Stake is the margin the account actually commits.
        stake = notional / capped_leverage

        return max(0.0, min(stake, equity * self.limits.max_position_pct))

    # ------------------------------------------------------------------
    # entry gate
    # ------------------------------------------------------------------
    def evaluate_entry(
        self,
        pair: str,
        equity: float,
        entry_price: float,
        stop_price: float,
        leverage: float = 1.0,
        proposed_stake: float | None = None,
        data_delay_s: float | None = None,
        inference_age_s: float | None = None,
        funding_rate: float | None = None,
        liquidation_price: float | None = None,
        exchange_healthy: bool = True,
    ) -> RiskDecision:
        """The one gate every entry must pass.

        Checks run cheapest-and-most-fatal first. The returned reason is what
        gets logged and exported as a ``rejected_entries`` metric label, so it
        is written to be readable in a dashboard.

        ``proposed_stake`` is the stake the caller is *actually about to
        commit*, in quote currency. When it is given, every limit is applied to
        that number and the trade is additionally rejected if it exceeds what
        risk-based sizing would have allowed — so an exchange minimum, a
        rounding step or any other Freqtrade adjustment cannot quietly inflate a
        position past the risk envelope. When it is omitted the engine falls
        back to its own sizing, which is what a caller that has not yet built an
        order (a pre-trade check) needs.

        ``data_delay_s`` is how late the newest candle is *past its close*, not
        its age since opening. See :attr:`RiskLimits.max_data_delay_s`.
        """
        lim = self.limits

        if self.state.halted:
            return RiskDecision(False, f"halted: {self.state.halt_reason}")

        if self._clock() < self.state.cooldown_until:
            remaining = self.state.cooldown_until - self._clock()
            return RiskDecision(False, f"cooldown active for another {remaining:.0f}s")

        if not exchange_healthy:
            return RiskDecision(False, "exchange/API reported unhealthy")

        if equity <= 0:
            return RiskDecision(False, "no equity")

        if data_delay_s is not None and data_delay_s > lim.max_data_delay_s:
            return RiskDecision(
                False,
                f"market data late: {data_delay_s:.0f}s past candle close > "
                f"{lim.max_data_delay_s:.0f}s",
            )

        if inference_age_s is not None and inference_age_s > lim.max_inference_staleness_s:
            return RiskDecision(
                False,
                f"inference stale: {inference_age_s:.0f}s > "
                f"{lim.max_inference_staleness_s:.0f}s",
            )

        if pair not in self.state.open_positions and (
            len(self.state.open_positions) >= lim.max_open_positions
        ):
            return RiskDecision(
                False,
                f"max open positions reached: {len(self.state.open_positions)} >= "
                f"{lim.max_open_positions}",
            )

        if funding_rate is not None and abs(funding_rate) > lim.max_funding_rate:
            return RiskDecision(
                False,
                f"funding rate {funding_rate:.5f} exceeds {lim.max_funding_rate:.5f}",
            )

        if leverage > lim.max_leverage:
            return RiskDecision(
                False, f"leverage {leverage:.2f} exceeds max {lim.max_leverage:.2f}"
            )

        if liquidation_price is not None and entry_price > 0:
            distance = abs(entry_price - liquidation_price) / entry_price
            if distance < lim.min_liquidation_distance_pct:
                return RiskDecision(
                    False,
                    f"liquidation only {distance:.2%} away, need "
                    f"{lim.min_liquidation_distance_pct:.2%}",
                )

        allowed_stake = self.position_size(equity, entry_price, stop_price, leverage)
        if allowed_stake <= 0:
            return RiskDecision(False, "position sizing returned zero stake")

        stake = allowed_stake if proposed_stake is None else proposed_stake
        if stake <= 0:
            return RiskDecision(False, "order stake is zero")

        # An exchange minimum, a precision step or any other Freqtrade
        # adjustment may have raised the order above what risk-based sizing
        # permits. Getting filled at a size we would never have chosen is the
        # failure this check exists to stop; taking the trade anyway would mean
        # the stated risk-per-trade is simply untrue.
        if stake > allowed_stake + 1e-9:
            return RiskDecision(
                False,
                f"order stake {stake:.2f} exceeds the risk-based maximum "
                f"{allowed_stake:.2f}",
            )

        projected = self.total_exposure + stake
        if projected > equity * lim.max_total_exposure_pct:
            return RiskDecision(
                False,
                f"total exposure {projected:.2f} would exceed "
                f"{equity * lim.max_total_exposure_pct:.2f}",
            )

        pair_exposure = self.state.open_positions.get(pair, 0.0) + stake
        if pair_exposure > equity * lim.max_exposure_per_asset_pct:
            return RiskDecision(
                False,
                f"exposure in {pair} would reach {pair_exposure:.2f}, over "
                f"{equity * lim.max_exposure_per_asset_pct:.2f}",
            )

        return RiskDecision(True, "ok", stake)
