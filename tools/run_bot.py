"""Safe Freqtrade launcher.

    python -m tools.run_bot --exchange binance --mode test --strategy NNPredictorStrategy

Replaces the old ``tools/start.sh``. Three things it does that the shell script
did not:

**It resolves credentials from the environment.** ``conf/*.json`` used to
contain ``"key": "${BINANCE_KEY}"``. Freqtrade does not expand ``${...}`` inside
JSON, so the literal string was passed to the exchange as an API key. Here the
placeholders are substituted before Freqtrade ever sees the config, and missing
credentials produce a clear error naming the variable — never its value.

**It enforces the dry-run gate.** ``chimera.safety.enforce_dry_run`` runs on the
merged config. A config asking for live trading without
``ENABLE_LIVE_TRADING=I_UNDERSTAND_THE_RISK`` in the environment aborts the
launch. Nothing downstream can re-enable live trading, because the merged file
this script writes already has ``dry_run`` forced to the permitted value.

**It writes the merged config to a private file.** The old script wrote to a
predictable ``/tmp`` path that any user on the host could read — and the merged
config contains API keys.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from chimera.safety import (
    LIVE_TRADING_ACK,
    LIVE_TRADING_ENV_VAR,
    LiveTradingBlocked,
    enforce_dry_run,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"
_PLACEHOLDER = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge; ``override`` wins. Neither input is mutated."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_placeholders(node: Any, missing: list[str]) -> Any:
    """Replace ``${VAR}`` strings with environment values, recursively.

    Unset variables are collected in ``missing`` and their placeholders removed,
    so a partially configured deployment fails with a named error instead of
    sending the literal ``${BINANCE_KEY}`` to an exchange.
    """
    if isinstance(node, dict):
        return {k: resolve_placeholders(v, missing) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_placeholders(v, missing) for v in node]
    if isinstance(node, str):
        match = _PLACEHOLDER.match(node)
        if match:
            name = match.group(1)
            value = os.environ.get(name)
            if not value:
                missing.append(name)
                return ""
            return value
    return node


def load_config(exchange: str, mode: str) -> dict[str, Any]:
    base_path = CONF_DIR / "base.json"
    override_path = CONF_DIR / f"{exchange}.{mode}.json"
    for path in (base_path, override_path):
        if not path.exists():
            raise SystemExit(f"config not found: {path}")
    return deep_merge(json.loads(base_path.read_text()), json.loads(override_path.read_text()))


def write_private(config: dict[str, Any], path: Path) -> Path:
    """Write the merged config readable by its owner only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Freqtrade safely.")
    parser.add_argument("--exchange", required=True, choices=["binance", "bybit", "okx"])
    parser.add_argument("--mode", required=True, choices=["test", "live"])
    parser.add_argument("--strategy", default="NNPredictorStrategy")
    parser.add_argument(
        "--command",
        default="trade",
        choices=["trade", "backtesting", "lookahead-analysis"],
    )
    parser.add_argument(
        "--config-out",
        default=None,
        help="Where to write the merged config (default: user_data/merged.json).",
    )
    parser.add_argument(
        "--dry-run-only-check",
        action="store_true",
        help="Merge, validate and print the effective mode, then exit without trading.",
    )
    parser.add_argument("extra", nargs="*", help="Extra arguments passed to freqtrade.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    config = load_config(args.exchange, args.mode)
    config["strategy"] = args.strategy

    missing: list[str] = []
    config = resolve_placeholders(config, missing)

    # The committed live profile keeps dry_run=true and marks its intent, so
    # that running Freqtrade directly against any file in conf/ cannot trade for
    # real. --mode live is what asks; the gate below is what grants.
    request_live = args.mode == "live" or config.get("chimera_live_intent") is True
    config.pop("chimera_live_intent", None)

    try:
        config = enforce_dry_run(config, request_live=request_live)
    except LiveTradingBlocked as exc:
        logger.error("%s", exc)
        logger.error(
            "To trade live deliberately: export %s=%s", LIVE_TRADING_ENV_VAR, LIVE_TRADING_ACK
        )
        return 2

    if config["dry_run"]:
        # Dry-run does not place orders, so absent credentials are fine; say so
        # rather than failing a perfectly valid paper-trading launch.
        if missing:
            logger.info(
                "Running dry-run without credentials for: %s", ", ".join(sorted(set(missing)))
            )
    elif missing:
        logger.error(
            "Live trading requires these environment variables: %s",
            ", ".join(sorted(set(missing))),
        )
        return 2

    config_path = Path(args.config_out or REPO_ROOT / "user_data" / "merged.json")
    write_private(config, config_path)
    logger.info(
        "Merged config written to %s (dry_run=%s, strategy=%s)",
        config_path,
        config["dry_run"],
        config["strategy"],
    )

    if args.dry_run_only_check:
        print(json.dumps({"dry_run": config["dry_run"], "strategy": config["strategy"]}))
        return 0

    freqtrade = shutil.which("freqtrade")
    if freqtrade is None:
        logger.error("freqtrade is not on PATH. Install it or use docker compose.")
        return 127

    command = [
        freqtrade,
        args.command,
        "--config",
        str(config_path),
        "--strategy",
        args.strategy,
        "--userdir",
        str(REPO_ROOT / "user_data"),
        # Explicit, because Freqtrade otherwise looks under <userdir>/strategies.
        # user_data is a mounted volume in Docker, so the strategies shipped in
        # the image would be shadowed by whatever the host directory contains.
        "--strategy-path",
        str(REPO_ROOT / "strategies"),
        *args.extra,
    ]
    # Freqtrade's IResolver imports strategy modules by file path, so the
    # repository root is not on sys.path in the child process and every
    # `from chimera...` / `from strategies...` import in a strategy fails with
    # "No module named". Verified with `freqtrade list-strategies`, which
    # reports LOAD FAILED for all four strategies without this.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)

    logger.info("Starting: %s", " ".join(command))
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    sys.exit(main())
