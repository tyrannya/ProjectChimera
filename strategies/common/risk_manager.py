import time
import requests
from freqtrade.exceptions import TemporaryStopException


class RiskManager:
    def __init__(self, drawdown_limit: float = 8.0, max_orders_per_minute: int = 60, funding_threshold: float = 0.0):
        self.drawdown_limit = drawdown_limit / 100
        self.max_orders = max_orders_per_minute
        self.funding_threshold = funding_threshold
        self.order_times = []
        self.equity_curve = []

    def update_equity(self, equity: float):
        self.equity_curve.append(equity)
        peak = max(self.equity_curve)
        drawdown = (peak - equity) / peak
        if drawdown >= self.drawdown_limit:
            raise TemporaryStopException("drawdown_limit reached")

    def register_order(self):
        now = time.time()
        self.order_times.append(now)
        self.order_times = [t for t in self.order_times if now - t < 60]
        if len(self.order_times) > self.max_orders:
            raise TemporaryStopException("order rate limit")

    def funding_guard(self, avg_funding_rate: float):
        if avg_funding_rate > self.funding_threshold:
            requests.post("http://localhost:8080/api/v1/stop")
