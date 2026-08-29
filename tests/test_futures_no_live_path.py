"""The safety barrier around ``chimera.futures``, asserted about the source.

Every other futures test asks whether the package behaves correctly. This one
asks whether it *could* place a real order, and it answers by reading the
package's own source with :mod:`ast` and by driving it with a scrubbed
environment. Nothing here trusts a docstring, a naming convention or an
intention: a promise that a module holds no exchange client is worth exactly as
much as the parse that confirms it.

If a test in this file starts failing, assume the futures package has grown a
live path — or the beginning of one — until proven otherwise. The three ways
that happens, in order of how easy they are to do by accident:

* an import creeps in (``requests`` for "just a price", ``hmac`` for "just a
  signature") and the package acquires the ability to reach an exchange;
* a credential is read from the environment, so the package starts depending on
  something an operator can supply;
* someone reads :class:`chimera.futures.LiveFuturesNotImplemented`, concludes
  that the missing piece is ``ENABLE_LIVE_TRADING``, sets it, and expects a live
  run. It is asserted below that the spot acknowledgement token does not unlock
  a futures path, because the futures path does not exist to be unlocked.

The scope is ``chimera/futures/*.py`` itself, and the limit is worth naming:
``executor.py`` imports :mod:`chimera.metrics`, which imports
``prometheus_client``, which puts ``http.client``, ``ssl`` and ``socket`` in the
process — an inbound scrape endpoint that nothing here starts, not an outbound
client. Importing the package makes no network call, but that is not asserted
here: a transitive barrier would have to follow the package's ``chimera.*``
imports one level out, and this file deliberately stops at the package.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

import chimera.futures as futures
from chimera.futures import (
    DeterministicFillModel,
    DryRunFuturesVenue,
    FuturesExecutionConfig,
    FuturesExecutor,
    FuturesStore,
    LiveFuturesNotImplemented,
    OrderState,
    PositionSide,
    TargetPosition,
    load_constraint_source,
)
from chimera.risk import RiskEngine, RiskLimits
from chimera.safety import (
    LIVE_TRADING_ACK,
    LIVE_TRADING_ENV_VAR,
    LiveTradingBlocked,
    is_secret_name,
    live_trading_acknowledged,
    resolve_trading_mode,
)

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path(futures.__file__).resolve().parent
SOURCES = sorted(PACKAGE.glob("*.py"))
SOURCE_TEXT = {path: path.read_text(encoding="utf-8") for path in SOURCES}

#: Modules that would give the package a way to reach an exchange, sign a
#: request, or open a socket. ``hmac`` and ``hashlib`` are here because a
#: Binance request is authenticated by an HMAC-SHA256 signature: importing them
#: is the first line of a live client even when no URL is present yet.
#: ``importlib`` and ``subprocess`` are here because they are how any of the
#: others gets reached without being named.
FORBIDDEN_IMPORTS = frozenset(
    {
        "aiohttp",
        "binance",
        "ccxt",
        "freqtrade",
        "hashlib",
        "hmac",
        "http",
        "httpx",
        "importlib",
        "requests",
        "socket",
        "ssl",
        "subprocess",
        "urllib",
        "urllib3",
        "websocket",
        "websockets",
    }
)

#: Matched case-insensitively against the raw source. Any of these appearing at
#: all — in code, a comment or a docstring — means the package has started
#: talking about credentials, which is one step from holding one.
CREDENTIAL_TOKENS = (
    "api_key",
    "apikey",
    "api_secret",
    "secret_key",
    "x-mbx-apikey",
    "signature=",
    "private_key",
)

#: Exchange endpoints, scoped deliberately. A ``https://`` link to a spec or a
#: doc is not a live path, so the literals below name Binance hosts and the two
#: API path prefixes rather than URLs in general; the host check in
#: :func:`test_no_exchange_endpoint_appears_in_the_package_source` covers the
#: rest without pretending every URL is an order.
ENDPOINT_TOKENS = (
    "fapi.binance.com",
    "api.binance.com",
    "binance.com",
    "/fapi/",
    "/api/v3/",
    "wss://",
    "ws://",
)

SYMBOL = "BTC/USDT:USDT"
EQUITY = 100_000.0


def imported_top_level_modules(source: str) -> set[str]:
    """Every top-level module name the source imports, by parsing it.

    Parsed rather than matched. A regex over the text cannot tell ``import
    socket`` from the word "socket" in a docstring, and — more to the point —
    an import written in a shape the regex does not cover would pass a textual
    check while still being an import.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def dynamic_execution_calls(source: str) -> set[str]:
    """Calls to builtins that could import or run code the import scan cannot see.

    ``import requests`` is a node the scan reads; ``__import__("requests")`` is a
    call whose argument it never looks at. Forbidding the four builtins is the
    cheap way to keep the parse above complete rather than merely careful.
    """
    watched = {"__import__", "eval", "exec", "compile"}
    return {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in watched
    }


def defined_class_names(source: str) -> set[str]:
    """Every class the source defines, nested ones included."""
    return {n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)}


def build_executor() -> FuturesExecutor:
    """A bootstrapped dry-run executor whose risk limits let an open through.

    The default :class:`chimera.risk.RiskLimits` are tight enough to veto this
    order, which is correct and is asserted elsewhere; widening them here is
    what makes this file's subject — the absence of a live path — the only
    reason a cycle could fail.
    """
    venue = DryRunFuturesVenue(
        source=load_constraint_source(), fill_model=DeterministicFillModel()
    )
    risk = RiskEngine(RiskLimits(max_position_pct=1.0, risk_per_trade_pct=0.5))
    risk.update_equity(EQUITY)
    executor = FuturesExecutor(venue=venue, risk=risk, store=FuturesStore.open(None))
    executor.recover({})
    return executor


def scrub_credentials(monkeypatch) -> None:
    """Delete every credential-shaped variable, and the live-trading ack with it.

    ``chimera.safety.is_secret_name`` decides what counts, so this and the spot
    safety gate cannot disagree about what a credential looks like. Written as a
    function rather than only a fixture so that a test can plant a credential and
    watch this remove it.
    """
    for name in list(os.environ):
        if is_secret_name(name):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)


@pytest.fixture
def scrubbed_env(monkeypatch):
    """The environment with every credential-shaped variable and the ack gone."""
    scrub_credentials(monkeypatch)


# --- the source itself ------------------------------------------------------
def test_the_source_scan_covers_every_module_in_the_package():
    """A glob that matched nothing would make every assertion below vacuous."""
    found = {path.name for path in SOURCES}
    expected = {
        "__init__.py",
        "accounting.py",
        "domain.py",
        "executor.py",
        "store.py",
        "venue.py",
    }
    assert expected <= found, (
        f"scanning {PACKAGE} found {sorted(found)}, which is missing "
        f"{sorted(expected - found)}. The barrier assertions in this file only "
        "cover the files this scan returns."
    )
    assert all(text.strip() for text in SOURCE_TEXT.values()), "a scanned module read empty"


def test_the_import_scan_can_actually_see_a_forbidden_import():
    """Catches the scanner going blind, which would pass all six modules vacuously.

    A scan that returned nothing would make every module below look clean. The
    sample covers each shape a live client could arrive in — plain, dotted,
    aliased, ``from``, and imported lazily inside a function, which is exactly
    where someone would hide one — and a mention in a string, which is not an
    import and must not be read as one.
    """
    sample = "\n".join(
        [
            "'A module docstring that mentions requests, socket and ccxt.'",
            "import socket",
            "import urllib.parse",
            "import httpx as _client",
            "from hmac import new",
            "from chimera.futures.domain import ZERO",
            "from .sibling import thing",
            'NOT_AN_IMPORT = "import ccxt"',
            "def f():",
            "    import ssl",
            "    return ssl",
        ]
    )
    found = imported_top_level_modules(sample)
    assert found & FORBIDDEN_IMPORTS == {"socket", "urllib", "httpx", "hmac", "ssl"}
    assert "ccxt" not in found, "a name inside a string literal is not an import"
    assert "chimera" in found

    assert dynamic_execution_calls("x = __import__('requests')") == {"__import__"}
    assert dynamic_execution_calls("import ast\nast.parse('')") == set()


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_no_module_in_the_package_imports_a_network_or_exchange_client(source):
    """Catches the package acquiring the *ability* to reach an exchange at all.

    A live path starts as an import. No credential, no URL and no order are
    needed for this test to be the one that fires first.
    """
    imported = imported_top_level_modules(SOURCE_TEXT[source])
    offending = sorted(imported & FORBIDDEN_IMPORTS)
    assert not offending, (
        f"{source} imports {offending}, which chimera.futures may not import. "
        "The package is dry-run only: it holds no client, signs nothing and "
        "opens no socket, and an import of any of these is how that stops being "
        f"true. Forbidden set: {sorted(FORBIDDEN_IMPORTS)}."
    )
    dynamic = sorted(dynamic_execution_calls(SOURCE_TEXT[source]))
    assert not dynamic, (
        f"{source} calls {dynamic}. An import spelled as a call is still an "
        "import, and it is one the scan above cannot read; the forbidden set is "
        "only a barrier while every import in the package is a literal one."
    )


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_no_module_in_the_package_reads_an_environment_variable(source):
    """Catches the package starting to depend on something an operator supplies.

    A package that reads no environment variable cannot be configured into a
    live one, and cannot pick up a credential that happens to be exported.
    """
    text = SOURCE_TEXT[source]
    for token in ("os.environ", "getenv"):
        assert token not in text, (
            f"{source} contains {token!r}. chimera.futures reads no environment: "
            "every input it has is passed to it, so there is no variable an "
            "operator could set that changes what it does."
        )


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_no_credential_shaped_identifier_appears_in_the_package_source(source):
    """Catches a credential being named — the step before one is held."""
    lowered = SOURCE_TEXT[source].lower()
    present = [token for token in CREDENTIAL_TOKENS if token in lowered]
    assert not present, (
        f"{source} mentions {present}. There is no credential in this package, "
        "no field to put one in, and nothing that would know what to do with "
        "one; a name like this is the first half of a live client."
    )


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_no_exchange_endpoint_appears_in_the_package_source(source):
    """Catches an exchange REST or websocket address appearing anywhere.

    Scoped to Binance hosts and to the ``/fapi/`` and ``/api/v3/`` prefixes on
    purpose: a ``https://`` link to a specification is not an order path, and a
    test that failed on any URL would be one people learn to work around.
    """
    text = SOURCE_TEXT[source]
    lowered = text.lower()
    present = [token for token in ENDPOINT_TOKENS if token in lowered]
    assert not present, (
        f"{source} contains {present}. chimera.futures addresses no exchange: "
        "the only venue in the package simulates fills in this process."
    )
    hosts = [url.split("/")[2].lower() for url in re.findall(r"https?://[^\s\"'<>)]+", text)]
    exchange_hosts = [host for host in hosts if "binance" in host]
    assert not exchange_hosts, f"{source} names the exchange host(s) {exchange_hosts}"


def test_the_package_defines_exactly_one_venue_class_and_it_is_the_dry_run_one():
    """Catches a second venue being added beside the simulator rather than instead.

    Asserted about the source and about the export list together: a live venue
    that existed but was left out of ``__all__`` would still be importable.
    """
    defined = set()
    for source in SOURCES:
        defined |= {n for n in defined_class_names(SOURCE_TEXT[source]) if n.endswith("Venue")}
    assert defined == {"DryRunFuturesVenue"}, (
        f"the package defines the venue class(es) {sorted(defined)}. "
        "DryRunFuturesVenue is the only venue chimera.futures has."
    )
    exported = {name for name in futures.__all__ if name.endswith("Venue")}
    assert exported == {"DryRunFuturesVenue"}


def test_the_dry_run_venue_has_no_field_that_could_hold_a_client_or_a_key():
    """Catches a credential or a session being added to the venue's own state."""
    fields = set(DryRunFuturesVenue.__dataclass_fields__)
    assert fields == {"source", "fill_model", "positions", "_sequence"}, (
        f"DryRunFuturesVenue now holds {sorted(fields)}. Its whole state is the "
        "constraint source, the fill model and the simulated positions."
    )


def test_the_only_exported_name_that_mentions_live_is_the_refusal_itself():
    """Catches a live-capable name being exported from the package root.

    ``LiveFuturesNotImplemented`` is the one exception to the rule and it is the
    rule's own enforcement: an error type raised to refuse, never a thing that
    trades. Anything else with "Live" in its name would be a capability.
    """
    mentions_live = {name for name in futures.__all__ if "live" in name.lower()}
    assert mentions_live == {"LiveFuturesNotImplemented"}, (
        f"chimera.futures.__all__ exports {sorted(mentions_live)}. The only name "
        "in this package allowed to mention live trading is the exception that "
        "refuses it."
    )
    assert issubclass(LiveFuturesNotImplemented, Exception)


# --- the configuration gate --------------------------------------------------
def test_asking_the_config_for_a_live_run_raises_instead_of_enabling_one():
    """Catches ``dry_run=False`` being accepted, ignored, or silently coerced.

    Refusing at construction is what makes "there is no live path" a fact about
    the object graph rather than about a branch nobody took.
    """
    with pytest.raises(LiveFuturesNotImplemented, match="has no live-order path"):
        FuturesExecutionConfig(dry_run=False)


def test_the_default_config_is_dry_run_at_exactly_one_times_isolated_margin():
    """Catches the safe default drifting; the refusal above only guards the else."""
    config = FuturesExecutionConfig()
    assert config.dry_run is True
    assert config.leverage == Decimal("1")
    assert config.margin_mode == "ISOLATED"


def test_the_live_trading_acknowledgement_does_not_unlock_the_futures_path(monkeypatch):
    """The test that stops someone concluding the missing piece is the ack.

    With ``ENABLE_LIVE_TRADING`` set to the exact token, the *spot* gate really
    is open — asserted here so the test cannot pass by the token being wrong —
    and the futures config still refuses, because there is no futures live path
    for an acknowledgement to unlock.
    """
    monkeypatch.setenv(LIVE_TRADING_ENV_VAR, LIVE_TRADING_ACK)
    assert live_trading_acknowledged() is True
    assert resolve_trading_mode({"dry_run": False}, request_live=True).live is True

    with pytest.raises(LiveFuturesNotImplemented) as excinfo:
        FuturesExecutionConfig(dry_run=False)
    assert "acknowledgement gate does not unlock it" in str(excinfo.value)


# --- the package under a scrubbed environment ---------------------------------
def test_the_scrub_deletes_a_credential_that_is_really_there(monkeypatch):
    """Catches the scrub doing nothing, which would make the tests below vacuous.

    The credential is planted here so the assertion has something to be about: on
    a machine that exports no keys, "the scrubbed environment holds none" passes
    whether the scrub ran or not, and so does anything that depends on it.
    """
    monkeypatch.setenv("BINANCE_API_KEY", "planted-by-this-test")
    monkeypatch.setenv(LIVE_TRADING_ENV_VAR, LIVE_TRADING_ACK)
    assert is_secret_name("BINANCE_API_KEY")
    assert live_trading_acknowledged() is True

    scrub_credentials(monkeypatch)

    leaked = sorted(name for name in os.environ if is_secret_name(name))
    assert leaked == [], f"the scrub left credential-shaped variables behind: {leaked}"
    assert LIVE_TRADING_ENV_VAR not in os.environ
    assert live_trading_acknowledged() is False


def test_a_full_open_and_close_cycle_needs_no_credential_in_the_environment(scrubbed_env):
    """Catches a credential becoming load-bearing anywhere on the execution path.

    The whole path runs — plan, risk gate, submit, fill, book, close — with
    every credential-shaped variable deleted from the environment. The exact
    fills are asserted so that a cycle which "worked" by doing nothing cannot
    pass: 5 bps of adverse slippage each way on a 0.5 BTC round trip is a
    realised loss of exactly 30 quote units.
    """
    executor = build_executor()

    opened = executor.execute_target(
        TargetPosition(SYMBOL, PositionSide.LONG, Decimal("0.5")),
        Decimal("60000"),
        equity=EQUITY,
    )
    assert [record.state for record in opened] == [OrderState.FILLED]
    assert opened[0].filled_quantity == Decimal("0.5")
    assert opened[0].average_price == Decimal("60030.00")

    position = executor.position(SYMBOL)
    assert position.side is PositionSide.LONG
    assert position.quantity == Decimal("0.5")
    assert position.entry_price == Decimal("60030.00")

    closed = executor.execute_target(
        TargetPosition.flat(SYMBOL), Decimal("60000"), equity=EQUITY
    )
    assert [record.state for record in closed] == [OrderState.FILLED]
    assert closed[0].average_price == Decimal("59970.00")

    assert executor.position(SYMBOL).side is PositionSide.FLAT
    assert executor.position(SYMBOL).quantity == Decimal("0")
    assert executor.ledger.realised_pnl == Decimal("-30")
    assert executor.ledger.trading_fees == Decimal("30.00")
    assert executor.ledger.turnover == Decimal("60000")

    assert [name for name in os.environ if is_secret_name(name)] == []


def test_building_the_whole_package_leaves_the_live_trading_gate_shut(scrubbed_env):
    """Catches ``chimera.futures`` opening the gate, at import or as it is built.

    The gate is read after the venue, the risk engine, the store and the executor
    all exist, so a write to ``ENABLE_LIVE_TRADING`` from anywhere in that path —
    or from importing the package, which this module already did — shows up here.
    """
    build_executor()

    assert LIVE_TRADING_ENV_VAR not in os.environ
    assert live_trading_acknowledged() is False
    assert resolve_trading_mode({}).dry_run is True
    binance = {"dry_run": True, "exchange": {"name": "binance"}}
    assert resolve_trading_mode(binance).dry_run is True
    with pytest.raises(LiveTradingBlocked, match=LIVE_TRADING_ENV_VAR):
        resolve_trading_mode({"dry_run": False})


def test_a_fresh_interpreter_imports_the_package_with_no_credential_available():
    """Catches an import-time read of a credential that this session already has.

    A fresh process, an environment with every credential-shaped variable and
    the ack removed, and nothing but ``import chimera.futures``. If the package
    needed anything from the environment, this is where it would fail rather
    than where it would quietly succeed on a developer's exported keys.
    """
    env = {k: v for k, v in os.environ.items() if not is_secret_name(k)}
    env.pop(LIVE_TRADING_ENV_VAR, None)
    env["PYTHONPATH"] = str(ROOT)
    script = (
        "import json, os\n"
        "import chimera.futures\n"
        "from chimera.safety import LIVE_TRADING_ENV_VAR as V\n"
        "from chimera.safety import live_trading_acknowledged, resolve_trading_mode\n"
        "print(json.dumps({\n"
        "    'live_var_set': V in os.environ,\n"
        "    'acknowledged': live_trading_acknowledged(),\n"
        "    'empty_config_is_dry_run': resolve_trading_mode({}).dry_run,\n"
        "    'venues': [n for n in chimera.futures.__all__ if n.endswith('Venue')],\n"
        "}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "importing chimera.futures in a scrubbed environment failed:\n" f"{result.stderr}"
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "live_var_set": False,
        "acknowledged": False,
        "empty_config_is_dry_run": True,
        "venues": ["DryRunFuturesVenue"],
    }
