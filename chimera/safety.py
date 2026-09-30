"""Startup safety gate.

ProjectChimera is dry-run only. Since R1-m there is no live path at all: the
Freqtrade engine, its strategies, its launcher and every exchange config were
deleted, so nothing in this repository can place a real order however this
module answers.

What survives here is the gate itself, and it survives on purpose. It is still
imported by ``tools/futures_dry_run.py`` and named by
``chimera/futures/executor.py``, it still refuses live trading unless
``ENABLE_LIVE_TRADING`` is set to the exact token ``I_UNDERSTAND_THE_RISK``, and
its redaction helpers (:func:`redact`, :func:`is_secret_name`) are used by the
notifier and the telemetry. A gate that guards nothing today is the right thing
to keep standing when the thing it guards is what R6 will eventually build:
deleting it and rewriting it later is how a safety check comes back weaker.

**Historical, for reading the tests that still name it.** The gate used to take
two independent actions -- the token above, and asking the launcher for live
mode. No committed config was ever independently live-capable: every file in
``conf/`` kept ``dry_run: true``, including the ``*.live.json`` profiles, which
carried ``"chimera_live_intent": true`` instead, and :func:`enforce_dry_run` was
the only place in the repository that ever set ``dry_run`` to ``False`` -- into
a private generated config, never back into the tree. Those configs and that
launcher are gone; :func:`enforce_dry_run` remains, and a caller that hands it a
mapping asking for live trading is still refused without the acknowledgement.
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


def resolve_trading_mode(config: Mapping[str, Any], request_live: bool = False) -> TradingMode:
    """Decide whether this process may trade live.

    Live trading needs a deliberate request *and* the environment
    acknowledgement. There are two ways a request can arrive:

    ``request_live``
        the operator asked the launcher for live mode (``--mode live``). This is
        the supported path, and the reason no committed config carries
        ``dry_run: false``.

    ``config["dry_run"] is False``
        a config already asks for real orders. No committed file does, but a
        hand-edited or generated one might, and it must still pass the gate
        rather than sneak past it — defence in depth for anything that reaches
        this function without going through the launcher.

    A missing ``dry_run`` key is treated as dry-run, so a truncated or
    unexpected config fails closed.
    """
    wants_live = request_live or config.get("dry_run") is False
    if not wants_live:
        return TradingMode(dry_run=True, reason="dry-run requested")
    if not live_trading_acknowledged():
        raise LiveTradingBlocked(
            f"live trading was requested but {LIVE_TRADING_ENV_VAR} is not set to "
            f"{LIVE_TRADING_ACK!r}. Refusing to start: live trading requires an "
            "explicit acknowledgement in the environment."
        )
    return TradingMode(
        dry_run=False,
        reason=f"live trading requested and {LIVE_TRADING_ENV_VAR} acknowledges the risk",
    )


def enforce_dry_run(config: dict[str, Any], request_live: bool = False) -> dict[str, Any]:
    """Return ``config`` with ``dry_run`` set to the value the gate permits.

    Mutates and returns the mapping so it can be handed straight to Freqtrade.
    This is the only place ``dry_run`` is ever set to ``False``.
    """
    mode = resolve_trading_mode(config, request_live)
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
