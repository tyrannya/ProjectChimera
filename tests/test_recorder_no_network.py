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

**Two barriers, because there are now two kinds of module.** PR-05 added the
websocket clients, the REST poller, the service and the health writer, and those
exist precisely to open sockets. Applying the offline core's rules to them would
have meant weakening the rules; inheriting them would have meant a guarantee that
was not kept. So the package is split:

* :data:`OFFLINE_CORE` — the contract, the parsers, the sink and the normalizer —
  keeps every rule in sections B, C and D: no network import, no endpoint, no
  credential, no clock, and a full record-and-normalize cycle with every socket
  call denied.
* :data:`LIVE_LAYER` — the acquisition modules — is held to section E instead:
  public market-data endpoints from a reviewed list and nothing else, no
  credential, no signature, no private path, no exchange SDK, no environment
  read, no research dependency, and no way to write the prospective boundary.

``chimera/recorder/__init__.py`` imports only the first group, so importing the
recorder's data model still cannot pull a socket into the process. Section E
asserts that too.
"""

from __future__ import annotations

import ast
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import chimera.recorder as recorder

PACKAGE = Path(recorder.__file__).resolve().parent
REPO = Path(__file__).resolve().parent.parent

#: The modules PR-04 delivers: the data model, which holds no client and opens
#: nothing. Everything in sections B, C and D is asserted about exactly these.
OFFLINE_CORE: tuple[str, ...] = (
    "__init__.py",
    "contract.py",
    "events.py",
    # The incremental normalizer folds a day from a cursor instead of re-reading
    # it. It opens nothing, reads no clock and is a pure function of the raw
    # files, so it is held to the offline core's rules rather than the live
    # layer's weaker ones — which is also the strongest available statement that
    # a performance cache cannot reach a network.
    "incremental.py",
    "normalize.py",
    "sink.py",
)

#: The modules PR-05 delivers: live collection. They open sockets by design, and
#: section E is the barrier they are held to instead.
LIVE_LAYER: tuple[str, ...] = (
    "health.py",
    "rest.py",
    "service.py",
    "streams.py",
)

ALL_SOURCES = sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)
SOURCES = [path for path in ALL_SOURCES if path.name in OFFLINE_CORE]
LIVE_SOURCES = [path for path in ALL_SOURCES if path.name in LIVE_LAYER]
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
def test_every_module_in_the_package_is_covered_by_one_of_the_two_barriers():
    """A tripwire, and the message says what to do when it fires."""
    found = tuple(sorted(path.name for path in ALL_SOURCES))
    assert found == tuple(sorted(OFFLINE_CORE + LIVE_LAYER)), (
        f"{PACKAGE} holds {list(found)}, not {sorted(OFFLINE_CORE + LIVE_LAYER)}. Every "
        "assertion in this file covers exactly the modules it lists, so a new module is "
        "covered by neither barrier until it is named here. Put it in OFFLINE_CORE if it "
        "holds no client and opens nothing; put it in LIVE_LAYER if it acquires, and read "
        "section E for what it then has to keep."
    )
    assert set(SOURCES).isdisjoint(LIVE_SOURCES), "a module cannot be in both barriers"
    assert all(
        path.read_text(encoding="utf-8").strip() for path in ALL_SOURCES
    ), "a scanned module read empty"


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


# --- E. the live layer's own barrier -------------------------------------------
#
# PR-05's modules exist to open sockets, so the offline core's guarantee cannot
# apply to them and pretending otherwise would be worse than having no guarantee
# at all. What is asserted here instead is everything that is still true of a
# module whose whole job is acquisition:
#
#   * it reaches public market-data endpoints, and only the ones named below;
#   * it holds no credential, computes no signature and reads no secret;
#   * it never touches an authenticated or private path;
#   * it imports no exchange SDK and runs no string;
#   * it depends on no research code and computes no economic quantity.
#
# The list of allowed endpoints is exhaustive on purpose. Adding a host or a
# path here is a reviewed edit, which is exactly the property the offline core's
# "no endpoint at all" rule buys for the layer beneath.

#: Every endpoint PR-05 may name, and nothing else. Public market data on both
#: hosts; no signed endpoint, no user-data stream, no account path.
ALLOWED_ENDPOINTS = frozenset(
    {
        "wss://fstream.binance.com/market/ws",
        "wss://fstream.binance.com/public/ws",
        "wss://stream.binance.com:9443/ws",
        "https://fapi.binance.com",
        "https://api.binance.com",
        "/fapi/v1/klines",
        "/fapi/v1/markPriceKlines",
        "/fapi/v1/indexPriceKlines",
        "/fapi/v1/fundingRate",
        "/fapi/v1/premiumIndex",
        "/api/v3/klines",
        # The two public API version prefixes, which the modules name in prose
        # when saying which market lives where. A prefix is not an endpoint:
        # "/fapi/v1/order" is still an offender, here and in
        # PRIVATE_PATH_TOKENS below.
        "/fapi/v1",
        "/api/v3",
    }
)

#: Paths and tokens that only exist on an authenticated Binance API. None of
#: them has a public counterpart, so any of them appearing is a private path.
PRIVATE_PATH_TOKENS = (
    "/sapi/",
    "/fapi/v1/order",
    "/fapi/v2/account",
    "/fapi/v2/balance",
    "/fapi/v1/positionside",
    "/fapi/v1/leverage",
    "/api/v3/order",
    "/api/v3/account",
    "userdatastream",
    "listenkey",
    "x-mbx-apikey",
)

#: Modules that would let the live layer sign a request, drive an exchange SDK,
#: shell out or import something the AST scan cannot see. ``socket``, ``ssl``,
#: ``asyncio`` and ``websockets`` are deliberately absent — they are the layer's
#: job — and ``hmac`` is deliberately present, because a recorder that needed a
#: MAC would be a recorder that had started authenticating.
LIVE_FORBIDDEN_IMPORTS = frozenset(
    {
        "binance",
        "ccxt",
        "freqtrade",
        "ftplib",
        "hmac",
        "importlib",
        "smtplib",
        "subprocess",
        "telnetlib",
        "webbrowser",
        "xmlrpc",
    }
)

#: How a process finds a secret without importing anything suspicious.
ENVIRONMENT_READS = (
    ("os", "getenv"),
    ("os", "environ"),
    ("environ", "get"),
    ("dotenv", "load_dotenv"),
)

URL_PATTERN = re.compile(r"(?:wss?|https?)://[^\s\"'`)\]<>]+")
API_PATH_PATTERN = re.compile(r"/(?:fapi|dapi|sapi|api)/v\d[A-Za-z0-9_/]*")

LIVE_SOURCE_TEXT = {path: path.read_text(encoding="utf-8") for path in LIVE_SOURCES}
#: The CLI and the network preflight are part of PR-05's reachable surface, so
#: they are held to the same rules. The preflight deliberately imports nothing
#: from this repository — that is what makes its answer about the venue rather
#: than about the recorder — but it still names hosts, and a host it named that
#: the allow-list did not would be exactly as much of a finding there.
CLI_SOURCE = REPO / "tools" / "recorder.py"
PREFLIGHT_SOURCE = REPO / "tools" / "recorder_preflight.py"
for _tool in (CLI_SOURCE, PREFLIGHT_SOURCE):
    LIVE_SOURCE_TEXT[_tool] = _tool.read_text(encoding="utf-8")
LIVE_PATHS = sorted(LIVE_SOURCE_TEXT)


def endpoints_in(source: str) -> set[str]:
    """Every URL and API path named anywhere in the source, prose included.

    Deliberately textual rather than AST-based: a host mentioned in a docstring
    is still a host this repository has written down, and the point of the
    allow-list is that adding one is visible in review.
    """
    found = set(URL_PATTERN.findall(source))
    found |= set(API_PATH_PATTERN.findall(source))
    return {token.rstrip(".,;:") for token in found}


def attribute_reads(source: str) -> set[str]:
    """Every ``owner.attribute`` the source names, as ``owner.attribute``."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            found.add(f"{node.value.id}.{node.attr}")
    return found


def test_the_live_layer_is_exactly_the_modules_pr_05_delivers():
    """The second half of the tripwire: what claims the weaker guarantee."""
    found = tuple(sorted(path.name for path in LIVE_SOURCES))
    assert found == LIVE_LAYER, (
        f"{PACKAGE} holds live modules {list(found)}, not {list(LIVE_LAYER)}. Every "
        "assertion in section E covers exactly the modules it lists; a new one that "
        "opens a socket must be added here deliberately."
    )
    assert LIVE_SOURCES, "the live-layer scan found nothing and would pass vacuously"


def test_the_endpoint_scan_can_actually_see_one():
    """Catches the scanner going blind, which would allow any host at all."""
    sample = (
        "URL = 'wss://evil.example.com/ws'\n"
        "'A docstring naming GET /sapi/v1/capital/config and https://api.example.com/x'\n"
    )
    found = endpoints_in(sample)
    assert "wss://evil.example.com/ws" in found
    assert "https://api.example.com/x" in found
    assert "/sapi/v1/capital/config" in found
    assert not found <= ALLOWED_ENDPOINTS, "the sample must not be mistaken for allowed"


@pytest.mark.parametrize("path", LIVE_PATHS, ids=lambda p: p.name)
def test_the_live_layer_names_only_public_market_data_endpoints(path):
    found = endpoints_in(LIVE_SOURCE_TEXT[path])
    offenders = sorted(token for token in found if token not in ALLOWED_ENDPOINTS)
    assert not offenders, (
        f"{path.name} names {offenders}, which is not in the reviewed list of public "
        "market-data endpoints. Adding one is a deliberate edit to ALLOWED_ENDPOINTS, "
        "not something that happens by writing a URL"
    )


@pytest.mark.parametrize("path", LIVE_PATHS, ids=lambda p: p.name)
def test_the_live_layer_never_names_a_private_or_authenticated_path(path):
    lowered = LIVE_SOURCE_TEXT[path].lower()
    found = [token for token in PRIVATE_PATH_TOKENS if token in lowered]
    assert not found, (
        f"{path.name} names {found}. Every one of those exists only behind an API key, "
        "and the recorder reads public market data"
    )


@pytest.mark.parametrize("path", LIVE_PATHS, ids=lambda p: p.name)
def test_the_live_layer_names_no_credential(path):
    lowered = LIVE_SOURCE_TEXT[path].lower()
    found = [token for token in CREDENTIAL_TOKENS if token in lowered]
    assert not found, (
        f"{path.name} mentions {found}. Opening a socket does not make a key necessary: "
        "there is no version of this recorder that authenticates"
    )


@pytest.mark.parametrize("path", LIVE_PATHS, ids=lambda p: p.name)
def test_the_live_layer_imports_no_exchange_sdk_and_signs_nothing(path):
    found = imported_top_level_modules(LIVE_SOURCE_TEXT[path]) & LIVE_FORBIDDEN_IMPORTS
    assert not found, (
        f"{path.name} imports {sorted(found)}. The live layer speaks HTTP and websockets "
        "to public endpoints; an exchange SDK, a MAC or a subprocess is a different thing"
    )


@pytest.mark.parametrize("path", LIVE_PATHS, ids=lambda p: p.name)
def test_the_live_layer_reaches_no_module_by_running_a_string(path):
    found = dynamic_execution_calls(LIVE_SOURCE_TEXT[path])
    assert not found, f"{path.name} calls {sorted(found)}, which defeats the import scan"


@pytest.mark.parametrize("path", LIVE_PATHS, ids=lambda p: p.name)
def test_the_live_layer_reads_no_environment_variable(path):
    """A secret that is never imported can still be read out of the environment."""
    found = attribute_reads(LIVE_SOURCE_TEXT[path])
    offenders = sorted(
        name for name in found if name in {f"{a}.{b}" for a, b in ENVIRONMENT_READS}
    )
    assert not offenders, (
        f"{path.name} reads {offenders}. Nothing the recorder needs is configured by a "
        "secret, and reading the environment is how a credential arrives without being "
        "named"
    )


def test_the_environment_scan_can_actually_see_a_read():
    sample = "import os\nkey = os.getenv('BINANCE_KEY')\nother = os.environ['X']\n"
    found = attribute_reads(sample)
    assert {"os.getenv", "os.environ"} <= found


@pytest.mark.parametrize("path", LIVE_PATHS, ids=lambda p: p.name)
def test_the_live_layer_depends_on_no_research_code(path):
    """Live collection is infrastructure. It must not import a checkpoint's code.

    ``tools/recorder.py`` is the one deliberate exception and it is asserted
    rather than waived: the CLI stamps ``nn.source_identity``'s revision onto the
    heartbeat, which is provenance, and it is the *only* thing it takes from
    ``nn``. The recorder package itself takes nothing.
    """
    text = LIVE_SOURCE_TEXT[path]
    assert "nn.evaluate" not in text, f"{path.name} imports the research evaluator"
    assert "chimera.carry" not in text, f"{path.name} imports the carry accounting"
    imports = set(re.findall(r"^\s*(?:from|import)\s+(nn[\w.]*)", text, re.MULTILINE))
    if path == PREFLIGHT_SOURCE:
        assert not imports, "the preflight imports nothing from this repository"
    elif path == CLI_SOURCE:
        assert imports <= {"nn.source_identity"}, (
            f"the CLI imports {sorted(imports)} from nn; provenance is the only thing it "
            "may take from research code"
        )
    else:
        assert not imports, (
            f"{path.name} imports {sorted(imports)}; the recorder package is "
            "infrastructure and must not depend on a research checkpoint's code"
        )


def test_the_live_layer_never_writes_the_prospective_boundary():
    """PR-05 records. It does not decide that what it recorded is evidence.

    ``with_prospective_from`` is the only way a boundary is ever set, it is pure,
    and setting it is a reviewed commit in a later pull request. A collector that
    called it would turn its own engineering data into scientific evidence while
    nobody was looking.
    """
    for path in LIVE_PATHS:
        text = LIVE_SOURCE_TEXT[path]
        assert "with_prospective_from" not in text, (
            f"{path.name} sets the prospective boundary. Recording is PR-05's; deciding "
            "that a recording is evidence is a reviewed commit, not a runtime action"
        )


def test_the_live_layer_implements_no_reconciliation_or_coverage_gate():
    """PR-06's names must not appear as PR-05's code.

    The plan's long-term CLI has ``reconcile`` and ``coverage`` subcommands and
    the contract's coverage rule names the arithmetic. Neither is implemented
    here, and a stub carrying the name would be read as the thing itself.
    """
    forbidden = ("published_coverage", "wallclock_coverage", "settlement_coverage")
    for path in LIVE_PATHS:
        names = module_identifiers_of(LIVE_SOURCE_TEXT[path])
        offenders = sorted(name for name in names if name in forbidden)
        assert not offenders, f"{path.name} computes {offenders}, which belongs to PR-06"
    cli = LIVE_SOURCE_TEXT[CLI_SOURCE]
    for subcommand in ('"reconcile"', "'reconcile'", '"coverage"', "'coverage'"):
        assert subcommand not in cli, (
            f"tools/recorder.py registers {subcommand}; a subcommand that exists but "
            "cannot work reads as a capability this build does not have"
        )


def module_identifiers_of(source: str) -> set[str]:
    """Every name the source defines, reads or calls an attribute of."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_importing_the_recorder_package_still_does_not_import_the_live_layer():
    """The offline core keeps its guarantee because ``__init__`` does not widen it.

    ``chimera.recorder`` re-exports the contract, the parsers, the sink and the
    normalizer. If it also imported the streams module, every consumer of the
    data model — a replay, a test, a report — would pull a websocket client and
    an event loop into its process, and section C's import-cost test would be
    measuring something else.
    """
    text = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    for module in ("streams", "rest", "service", "health"):
        assert f"chimera.recorder.{module}" not in text, (
            f"chimera/recorder/__init__.py imports {module}. The live layer is imported "
            "explicitly by whoever wants it, so that holding the data model never means "
            "holding a socket"
        )
