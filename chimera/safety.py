"""Startup safety gate.

ProjectChimera is dry-run only unless the operator takes two independent,
deliberate actions:

1. set ``ENABLE_LIVE_TRADING`` to the exact token ``I_UNDERSTAND_THE_RISK``;
2. pass a config that explicitly sets ``dry_run`` to ``false``.

Either one alone is not enough, and the presence of exchange API keys is never
sufficient on its own. If the environment variable is absent or holds anything
else, :func:`resolve_trading_mode` forces dry-run regardless of what the config
file says.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

LIVE_TRADING_ENV_VAR = "ENABLE_LIVE_TRADING"
LIVE_TRADING_ACK = "I_UNDERSTAND_THE_RISK"

logger = logging.getLogger(__name__)

#: Substrings that mark an environment variable as secret. Values of matching
#: variables are never logged or echoed back.
_SECRET_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "APIKEY")


class LiveTradingBlocked(RuntimeError):
    """Raised when a config asks for live trading without the explicit ack."""


def is_secret_name(name: str) -> bool:
    """Return True if ``name`` looks like it holds a credential."""
    upper = name.upper()
    return any(hint in upper for hint in _SECRET_HINTS)


def redact(name: str, value: str | None) -> str:
    """Render ``value`` for logs, masking it when ``name`` looks secret."""
    if value is None:
        return "<unset>"
    if is_secret_name(name):
        return "<set>" if value else "<empty>"
    return value


def live_trading_acknowledged() -> bool:
    """True only when the live-trading acknowledgement token is set exactly."""
    return os.environ.get(LIVE_TRADING_ENV_VAR, "") == LIVE_TRADING_ACK


@dataclass(frozen=True)
class TradingMode:
    """Outcome of the safety gate."""

    dry_run: bool
    reason: str

    @property
    def live(self) -> bool:
        return not self.dry_run


def resolve_trading_mode(config: Mapping[str, Any]) -> TradingMode:
    """Decide whether this process may trade live.

    ``config`` is a parsed Freqtrade config. A missing ``dry_run`` key is
    treated as dry-run, so a truncated or unexpected config fails closed.
    """
    wants_live = config.get("dry_run") is False
    if not wants_live:
        return TradingMode(dry_run=True, reason="config requests dry-run")
    if not live_trading_acknowledged():
        raise LiveTradingBlocked(
            f"config sets dry_run=false but {LIVE_TRADING_ENV_VAR} is not set to "
            f"{LIVE_TRADING_ACK!r}. Refusing to start: live trading requires an "
            "explicit acknowledgement in the environment."
        )
    return TradingMode(
        dry_run=False,
        reason=(
            f"config requests live trading and {LIVE_TRADING_ENV_VAR} " "acknowledges the risk"
        ),
    )


def enforce_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    """Return ``config`` with ``dry_run`` set to the value the gate permits.

    Mutates and returns the mapping so it can be handed straight to Freqtrade.
    """
    mode = resolve_trading_mode(config)
    config["dry_run"] = mode.dry_run
    if mode.live:
        logger.warning("LIVE TRADING ENABLED: %s", mode.reason)
    else:
        logger.info("Dry-run mode: %s", mode.reason)
    return config


def require_env(names: Iterable[str]) -> dict[str, str]:
    """Return the named environment variables, or raise listing the missing ones.

    Values are never included in the error message.
    """
    names = list(names)
    found: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if value:
            found[name] = value
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(sorted(missing))
            + ". Copy .env.example to .env and fill them in."
        )
    return found
