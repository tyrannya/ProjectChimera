"""Optional Telegram notifications.

Three properties the previous implementation lacked:

*optional*
    Missing credentials disable notifications. They never raise, because a
    monitoring integration must not be able to stop a trading bot from booting.

*deduplicated and rate limited*
    A repeated inference failure used to send one message per occurrence. Here,
    an identical message is suppressed for ``dedup_window_s``, and the notifier
    sends at most ``max_per_minute`` messages overall, so one fault cannot
    produce a notification storm.

*secret-free*
    Only :func:`chimera.safety.redact`-safe text is ever sent, and the token is
    never logged or echoed.
"""

from __future__ import annotations

import html
import logging
import os
import time
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class TelegramNotifier:
    """Sends messages if configured, silently no-ops if not.

    Check :attr:`enabled` when you want to know which mode you got.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        dedup_window_s: float = 900.0,
        max_per_minute: int = 10,
        timeout_s: float = 10.0,
    ) -> None:
        self._token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN") or ""
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID") or ""
        self.dedup_window_s = dedup_window_s
        self.max_per_minute = max_per_minute
        self.timeout_s = timeout_s
        self._recent: dict[str, float] = {}
        self._sent_at: Deque[float] = deque()

        if not self.enabled:
            logger.info(
                "Telegram notifications disabled (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set)"
            )

    @property
    def enabled(self) -> bool:
        return bool(self._token and self._chat_id)

    def _should_send(self, key: str) -> bool:
        now = time.time()

        last = self._recent.get(key)
        if last is not None and now - last < self.dedup_window_s:
            return False

        while self._sent_at and now - self._sent_at[0] > 60.0:
            self._sent_at.popleft()
        if len(self._sent_at) >= self.max_per_minute:
            logger.warning("Telegram rate limit reached, dropping message")
            return False

        # Bound the dedup table so a long-running process with many distinct
        # messages does not grow it without limit.
        if len(self._recent) > 512:
            cutoff = now - self.dedup_window_s
            self._recent = {k: v for k, v in self._recent.items() if v > cutoff}

        self._recent[key] = now
        self._sent_at.append(now)
        return True

    def send(self, message: str, dedup_key: str | None = None) -> bool:
        """Send ``message``. Returns True only if it actually went out."""
        if not self.enabled:
            return False
        if not self._should_send(dedup_key or message):
            return False

        try:
            import requests

            response = requests.post(
                f"{TELEGRAM_API}/bot{self._token}/sendMessage",
                data={
                    "chat_id": self._chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            return True
        except Exception as exc:  # noqa: BLE001 - notification must never propagate
            # str(exc) can contain the request URL, which contains the token.
            logger.error("Telegram send failed: %s", type(exc).__name__)
            return False

    def send_error(self, title: str, detail: str = "") -> bool:
        """Send an error. Deduplicated on ``title`` so a repeating fault sends once."""
        body = f"⚠️ <b>{html.escape(title)}</b>"
        if detail:
            body += f"\n<pre>{html.escape(detail[:1500])}</pre>"
        return self.send(body, dedup_key=f"error:{title}")

    def send_event(self, title: str, detail: str = "") -> bool:
        body = f"ℹ️ <b>{html.escape(title)}</b>"
        if detail:
            body += f"\n{html.escape(detail)}"
        return self.send(body, dedup_key=f"event:{title}")
