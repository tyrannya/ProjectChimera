"""
CommonRiskManager — универсальный риск-менеджер для Freqtrade-бота.
- MaxDrawdown: глобальный лимит по equity — не более 8% просадки;
- RateLimitGuard: если от биржи 5 и более ошибок HTTP 429 за 60 секунд — временно стопим торговлю;
- FundingGuard: если расходы на финансирование превышают 30% от реализованной прибыли — пауза.
"""

from freqtrade.exceptions import TemporaryStopException

class CommonRiskManager:
    def __init__(self):
        self.max_drawdown = 0.08  # 8%
        self.http_429_errors = []
        self.funding_cost = 0
        self.realized_pnl = 0
        self.pause_triggered = False

    def check_drawdown(self, equity_curve):
        """Проверка просадки equity."""
        peak = max(equity_curve)
        min_val = min(equity_curve)
        drawdown = (peak - min_val) / peak
        if drawdown > self.max_drawdown:
            raise TemporaryStopException(f"Drawdown limit exceeded: {drawdown:.2%}")

    def log_http_429(self, timestamp):
        """Логируем HTTP 429 ошибки (Rate Limit)."""
        self.http_429_errors.append(timestamp)
        # Оставляем только события за последние 60 секунд
        self.http_429_errors = [t for t in self.http_429_errors if timestamp - t <= 60]
        if len(self.http_429_errors) >= 5:
            raise TemporaryStopException("Rate limit: too many HTTP 429 errors in 60s.")

    def update_funding_and_pnl(self, funding_cost, realized_pnl):
        """Обновить значения финансирования и прибыли."""
        self.funding_cost = funding_cost
        self.realized_pnl = realized_pnl

    def funding_guard(self):
        """Финансовая защита: если cost > 0.3*PNL — пауза."""
        if self.funding_cost > abs(self.realized_pnl) * 0.3:
            self.trigger_pause("Funding cost exceeds threshold.")

    def trigger_pause(self, reason):
        """Триггерим ручную паузу торгов."""
        self.pause_triggered = True
        raise TemporaryStopException(f"Trading paused: {reason}")
