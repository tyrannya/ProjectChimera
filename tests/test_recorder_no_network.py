"""The barrier around the recorder's offline core, asserted about the source.

Every other recorder test asks whether the package behaves correctly. This one
asks whether it *could* reach an exchange, and it answers by parsing the
package's own source with :mod:`ast`, by watching what importing it puts in
``sys.modules``, and by driving the whole pipeline with every socket entry point
replaced by something that raises. Nothing here trusts a docstring or an
intention: a promise that a module holds no client is worth exactly as much as
the parse that confirms it.

This is the counterpart of ``tests/test_futures_no_live_path.py`` for the
recorder, with one deliberate difference. That file forbids ``hashlib``, because
in the futures package a hash is the first line of a request signature. Here a
hash is the *point* — the contract's identity, the day manifest's checksums, the
normalized day's value digest — so ``hashlib`` is allowed and ``hmac``, which
has no honest use in a recorder, is not.

**The limit of the barrier, stated rather than implied.**
``chimera.recorder.normalize`` imports pandas, and pandas puts ``socket``,
``urllib`` and ``subprocess`` in the process on its own. So the import test below
measures what importing the recorder adds *on top of* pandas rather than what is
in ``sys.modules`` afterwards, and the source scan stops at the package
boundary. What is asserted is exactly what can be: the recorder's own modules
import nothing that can reach a network, name no endpoint, hold no credential,
read no clock, and complete a full record-and-normalize cycle with every socket
call denied.

**When PR-05 arrives**, ``test_the_package_holds_only_the_offline_core`` fails.
That is the intended behaviour: the websocket clients and REST pollers belong to
a package that *does* open sockets, and moving a module out of
:data:`OFFLINE_CORE` must be a deliberate, reviewed edit rather than a silent
widening of this file's scope.
"""

from __future__ import annotations

import ast
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import chimera.recorder as recorder

PACKAGE = Path(recorder.__file__).resolve().parent
REPO = Path(__file__).resolve().parent.parent

#: The modules PR-04 delivers. Everything in the package is one of these, and a
#: later module that opens a socket must be added to the package's own barrier
#: rather than inheriting this one.
OFFLINE_CORE: tuple[str, ...] = (
    "__init__.py",
    "contract.py",
    "events.py",
    "normalize.py",
    "sink.py",
)

SOURCES = sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)
SOURCE_TEXT = {path: path.read_text(encoding="utf-8") for path in SOURCES}

#: Modules that would give the package a way to reach an exchange, sign a
#: request, open a socket, or run something the import scan cannot see.
#: ``hashlib`` is deliberately absent — see the module docstring.
FORBIDDEN_IMPORTS = frozenset(
    {
        "aiohttp",
        "asyncio",
        "binance",
        "ccxt",
        "ftplib",
        "freqtrade",
        "hmac",
        "http",
        "httpx",
        "importlib",
        "requests",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "subprocess",
        "telnetlib",
        "urllib",
        "urllib3",
        "webbrowser",
        "websocket",
        "websockets",
        "xmlrpc",
    }
)

#: Matched case-insensitively against the raw source. Any of these appearing at
#: all means the package has started talking about credentials, which is one
#: step from holding one. The recorder reads public market data and there is no
#: version of it that needs a key.
CREDENTIAL_TOKENS = (
    "api_key",
    "apikey",
    "api_secret",
    "secret_key",
    "x-mbx-apikey",
    "signature=",
    "private_key",
    "listenkey",
)

#: Endpoints, scoped deliberately: the word "Binance" is documentation and the
#: docstrings use it, so these name hosts, schemes and API path prefixes rather
#: than pretending every mention is a client.
ENDPOINT_TOKENS = (
    "fapi.binance.com",
    "api.binance.com",
    "stream.binance.com",
    "fstream.binance.com",
    "data.binance.vision",
    "/fapi/",
    "/api/v3/",
    "wss://",
    "ws://",
    "http://",
    "https://",
)

#: Reading a clock would make the offline core non-deterministic and would put a
#: value into a record that no test could pin. Receipt timestamps are arguments.
CLOCK_CALLS = (
    ("time", "time"),
    ("time", "time_ns"),
    ("time", "monotonic"),
    ("time", "monotonic_ns"),
    ("time", "perf_counter"),
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("date", "today"),
)


def imported_top_level_modules(source: str) -> set[str]:
    """Every top-level module name the source imports, by parsing it.

    Parsed rather than matched. A regex over the text cannot tell ``import
    socket`` from the word "socket" in a docstring, and an import written in a
    shape the regex does not cover would pass a textual check while still being
    an import.
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
    """Calls to builtins that could import or run code the import scan cannot see."""
    watched = {"__import__", "eval", "exec", "compile"}
    return {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in watched
    }


def clock_reads(source: str) -> set[str]:
    """Every ``module.attribute`` call in the source that reads a clock."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        value = node.func.value
        owner = value.id if isinstance(value, ast.Name) else None
        for module, attribute in CLOCK_CALLS:
            if node.func.attr == attribute and owner == module:
                found.add(f"{module}.{attribute}")
    return found


# --- A. the scan itself --------------------------------------------------------
def test_the_package_holds_only_the_offline_core():
    """A tripwire, and the message says what to do when it fires."""
    found = tuple(sorted(path.name for path in SOURCES))
    assert found == OFFLINE_CORE, (
        f"{PACKAGE} holds {list(found)}, not {list(OFFLINE_CORE)}. Every assertion in this "
        "file covers exactly the modules it lists. A new module that opens a socket — the "
        "websocket clients, the REST pollers, the service — belongs to that package's own "
        "barrier: move it out of OFFLINE_CORE here and give it one, rather than letting it "
        "inherit an offline guarantee it does not keep."
    )
    assert all(text.strip() for text in SOURCE_TEXT.values()), "a scanned module read empty"


def test_the_import_scan_can_actually_see_a_forbidden_import():
    """Catches the scanner going blind, which would pass every module vacuously."""
    sample = "\n".join(
        [
            "'A module docstring that mentions requests, socket and websockets.'",
            "import socket",
            "import urllib.parse",
            "import httpx as _client",
            "from hmac import new",
            "from chimera.recorder.events import RawEvent",
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


def test_the_clock_scan_can_actually_see_a_clock_read():
    sample = (
        "import time\nfrom datetime import datetime\nx = time.time_ns()\ny = datetime.now()"
    )
    assert clock_reads(sample) == {"time.time_ns", "datetime.now"}


# --- B. the source -------------------------------------------------------------
@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_module_imports_anything_that_can_reach_a_network(path):
    found = imported_top_level_modules(SOURCE_TEXT[path]) & FORBIDDEN_IMPORTS
    assert not found, (
        f"{path.name} imports {sorted(found)}. The offline core has no client, no session "
        "and no event loop; acquisition is a later package"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_module_reaches_a_module_by_running_a_string(path):
    found = dynamic_execution_calls(SOURCE_TEXT[path])
    assert not found, (
        f"{path.name} calls {sorted(found)}, which is how the import scan above is made "
        "incomplete"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_module_names_a_credential(path):
    lowered = SOURCE_TEXT[path].lower()
    found = [token for token in CREDENTIAL_TOKENS if token in lowered]
    assert not found, (
        f"{path.name} mentions {found}. The recorder reads public market data and there is "
        "no version of it that needs a key"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_module_names_an_endpoint(path):
    lowered = SOURCE_TEXT[path].lower()
    found = [token for token in ENDPOINT_TOKENS if token in lowered]
    assert not found, f"{path.name} names {found}; there is nothing here to connect with"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_module_reads_a_clock(path):
    found = clock_reads(SOURCE_TEXT[path])
    assert not found, (
        f"{path.name} calls {sorted(found)}. Receipt timestamps are arguments to the offline "
        "core, which is what makes a record reproducible and a test able to pin every byte"
    )


def test_the_committed_contract_declares_streams_without_naming_a_url():
    """The contract may describe what will be acquired. It may not carry a client."""
    text = (PACKAGE / "contracts" / "btcusdt-prospective-gen3.json").read_text(
        encoding="utf-8"
    )
    document = json.loads(text)
    assert document["streams"], "the contract describes what will be recorded"
    lowered = text.lower()
    for token in ("wss://", "ws://", "https://", "/fapi/", "/api/v3/"):
        assert token not in lowered, f"the contract carries {token!r}"
    for token in CREDENTIAL_TOKENS:
        assert token not in lowered


# --- C. what importing it costs ------------------------------------------------
def test_importing_the_recorder_adds_no_network_module_of_its_own():
    """Measured in a fresh interpreter, against pandas rather than against nothing.

    ``chimera.recorder.normalize`` imports pandas, and pandas imports ``socket``,
    ``urllib`` and ``subprocess`` by itself. Asserting that they are absent
    afterwards would be a false claim; asserting that the recorder adds none of
    them is the true one, and it is the one that fails if a client creeps in.
    """
    program = """
import json, sys
import numpy, pandas, pyarrow, pyarrow.parquet
before = set(sys.modules)
import chimera.recorder
print(json.dumps(sorted(set(sys.modules) - before)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    added = json.loads(completed.stdout.strip().splitlines()[-1])
    assert added, "the probe imported nothing and would pass vacuously"
    offenders = sorted(name for name in added if name.split(".")[0] in FORBIDDEN_IMPORTS)
    assert not offenders, f"importing chimera.recorder pulled in {offenders}"
    assert all(
        name.split(".")[0] in {"chimera"} for name in added
    ), f"importing chimera.recorder pulled in more than itself: {added}"


# --- D. the pipeline with the network taken away -------------------------------
@pytest.fixture
def no_network(monkeypatch):
    """Every way a Python process opens a connection, replaced by a refusal."""
    denied: list[str] = []

    def refuse(name):
        def blocked(*args, **kwargs):
            denied.append(name)
            raise AssertionError(f"the offline recorder core called {name}")

        return blocked

    # The class's own methods first: replacing ``socket.socket`` with a function
    # would make ``socket.socket.connect`` unreachable, and a socket object that
    # already existed would still be able to connect through it.
    real_socket = socket.socket
    monkeypatch.setattr(real_socket, "connect", refuse("socket.socket.connect"))
    monkeypatch.setattr(real_socket, "connect_ex", refuse("socket.socket.connect_ex"))
    for name in (
        "socket",
        "create_connection",
        "socketpair",
        "getaddrinfo",
        "gethostbyname",
        "create_server",
    ):
        monkeypatch.setattr(socket, name, refuse(f"socket.{name}"), raising=False)
    return denied


def test_the_whole_pipeline_runs_with_every_socket_call_denied(no_network, tmp_path):
    """Contract, events, sink and normalizer, end to end, with no network available."""
    from tests.recorder_synthetic import DAY, funding_day, um_day

    contract = recorder.load_recorder_contract()
    assert len(contract.contract_hash) == 64
    assert contract.activated is False

    root = contract.storage_root(tmp_path / "data")
    for stream, events in {**um_day(range(3)), recorder.UM_FUNDING: funding_day(DAY)}.items():
        with recorder.RawSink(root, stream, contract=contract) as sink:
            for event in events:
                sink.append(event)
            sink.sync()
            sink.freeze_day(DAY)

    normalizer = recorder.MinuteNormalizer(root, contract)
    day = normalizer.build_day("um", DAY)
    settlements = normalizer.build_settlements("um")
    normalizer.freeze_day("um", DAY)

    assert day.rows == 3
    assert len(day.digest) == 64
    assert settlements.rows == 3
    assert day.parquet_path.exists()
    assert no_network == [], f"the offline core reached for {no_network}"


def test_the_denial_fixture_would_actually_catch_a_connection(no_network):
    """Catches the fixture going blind, which would pass the test above vacuously."""
    with pytest.raises(AssertionError, match="called socket.create_connection"):
        socket.create_connection(("127.0.0.1", 9))
    assert no_network == ["socket.create_connection"]
