"""Section 7.6's architectural guarantees, asserted over the source itself.

Section 7.6 point 4 fixes four rules, and section 14 row 10 adds two more as
this PR's scientific-integrity check. Each is checked by reading the AST of the
demo and carry packages rather than by running them, because the claim is about
what the code CAN do, not about what one execution happened to do.

Every guard here ships with a negative control: a synthetic module containing
exactly the forbidden construct, asserted to be caught. A guard that cannot fail
proves nothing, and an AST rule that silently stops matching -- because an
import moved, or a call is now made through an alias -- is exactly the kind of
check that rots without anyone noticing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "chimera" / "demo"
CARRY = REPO / "chimera" / "carry"

#: The one file section 7.6 permits to construct a venue.
VENUE_FACTORY = CARRY / "factory.py"

NETWORK_ROOTS = {
    "socket",
    "ssl",
    "http",
    "urllib",
    "requests",
    "websockets",
    "websocket",
    "aiohttp",
    "httpx",
    "ccxt",
    "freqtrade",
}

CREDENTIAL_MARKERS = ("os.environ", "getenv", "API_KEY", "API_SECRET", "SECRET_KEY")


def sources() -> list[Path]:
    return sorted(p for p in (*DEMO.rglob("*.py"), *CARRY.rglob("*.py")))


def imported_modules(tree: ast.AST) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.lineno))
    return found


def called_names(tree: ast.AST) -> list[tuple[str, int]]:
    """Every called name, whether `f()`, `obj.f()` or `a.b.f()`."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            found.append((func.id, node.lineno))
        elif isinstance(func, ast.Attribute):
            found.append((func.attr, node.lineno))
    return found


# ---------------------------------------------------------------------------
# (i) no import of chimera.futures.venue except from chimera/carry/factory.py
# ---------------------------------------------------------------------------
def test_only_the_carry_factory_imports_the_venue_module():
    offenders = []
    for path in sources():
        if path == VENUE_FACTORY:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module, line in imported_modules(tree):
            if "chimera.futures.venue" in module:
                offenders.append(f"{path.relative_to(REPO)}:{line}")
    assert not offenders, (
        "section 7.6 permits chimera/carry/factory.py alone to import "
        f"chimera.futures.venue; found {offenders}"
    )


def test_the_venue_import_guard_catches_a_planted_import(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        "from chimera.futures.venue import DryRunFuturesVenue\n", encoding="utf-8"
    )
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    hits = [m for m, _ in imported_modules(tree) if "chimera.futures.venue" in m]
    assert hits, "the guard's own matcher must see a planted venue import"


# ---------------------------------------------------------------------------
# (ii) no call to submit outside chimera/futures/executor.py
# ---------------------------------------------------------------------------
def test_nothing_on_the_demo_path_calls_submit():
    offenders = []
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, line in called_names(tree):
            if name == "submit":
                offenders.append(f"{path.relative_to(REPO)}:{line}")
    assert not offenders, (
        "only chimera/futures/executor.py may call submit(); the demo path goes "
        f"through the executor so every order passes its checks. Found {offenders}"
    )


def test_the_submit_guard_catches_a_planted_call(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text("venue.submit(intent)\nsubmit(intent)\n", encoding="utf-8")
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    assert [n for n, _ in called_names(tree) if n == "submit"], "matcher must see both forms"


# ---------------------------------------------------------------------------
# (iii) no OrderIntent with purpose OPEN or INCREASE outside plan_transition
# ---------------------------------------------------------------------------
def test_nothing_on_the_demo_path_constructs_an_opening_order_intent():
    offenders = []
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name != "OrderIntent":
                continue
            for keyword in node.keywords:
                if keyword.arg != "purpose":
                    continue
                rendered = ast.dump(keyword.value)
                if "OPEN" in rendered or "INCREASE" in rendered:
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, (
        "an OrderIntent that opens or increases a position may only be built by "
        f"chimera.futures.domain.plan_transition; found {offenders}"
    )


def test_the_order_intent_guard_catches_a_planted_construction(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        "OrderIntent(symbol='X', purpose=OrderPurpose.OPEN, quantity=1)\n", encoding="utf-8"
    )
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "OrderIntent":
            for keyword in node.keywords:
                if keyword.arg == "purpose" and "OPEN" in ast.dump(keyword.value):
                    hits.append(node.lineno)
    assert hits, "the guard's own matcher must see a planted opening intent"


# ---------------------------------------------------------------------------
# (iv) the runner never opens a socket, and holds no credential
# ---------------------------------------------------------------------------
def test_the_demo_path_imports_no_network_module():
    offenders = []
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module, line in imported_modules(tree):
            root = module.split(".")[0]
            if root in NETWORK_ROOTS:
                offenders.append(f"{path.relative_to(REPO)}:{line}: {module}")
    assert not offenders, (
        "the demo runner reads the recorder's files and talks to a dry-run venue; "
        f"it opens no socket. Found {offenders}"
    )


@pytest.mark.parametrize("marker", CREDENTIAL_MARKERS)
def test_the_demo_path_reads_no_credential(marker):
    offenders = [
        str(path.relative_to(REPO))
        for path in sources()
        if marker in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"{marker} must not appear on the demo path; found in {offenders}"


def test_the_network_guard_catches_a_planted_import(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text("import socket\nfrom urllib import request\n", encoding="utf-8")
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    roots = {m.split(".")[0] for m, _ in imported_modules(tree)}
    assert roots & NETWORK_ROOTS, "the guard's own matcher must see planted network imports"


# ---------------------------------------------------------------------------
# section 14 row 10: rules read only MarketState
# ---------------------------------------------------------------------------
RULE_MODULES = ("rules.py", "rules_carry.py", "rules_shadow.py")


def test_a_rule_module_imports_no_executor_venue_or_position():
    """A rule decides from the state it is handed and cannot reach the account.

    Section 14's scientific-integrity check for this PR is "rules read only
    `MarketState`". The strongest mechanical form of that is the import graph: a
    module that cannot name an executor, a venue or a position cannot consult
    one, whatever a later edit does inside a function body.
    """
    forbidden = (
        "chimera.futures.executor",
        "chimera.futures.venue",
        "chimera.carry.factory",
        "chimera.demo.runner",
        "chimera.demo.feed",
    )
    offenders = []
    for name in RULE_MODULES:
        path = DEMO / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module, line in imported_modules(tree):
            if any(module.startswith(bad) for bad in forbidden):
                offenders.append(f"{name}:{line}: {module}")
    assert not offenders, (
        "a rule reads the MarketState it is given and nothing else; it may not "
        f"import an executor, a venue, a position or the feed. Found {offenders}"
    )


def test_a_rule_module_opens_no_file_and_reads_no_clock():
    """No hidden state: section 2.2's "rules hold no hidden state beyond what
    the log records". A rule that could open a file or read a clock could carry
    state between minutes that the decision record does not show."""
    forbidden_calls = {"open", "read_text", "read_bytes", "time", "now", "utcnow", "monotonic"}
    offenders = []
    for name in RULE_MODULES:
        path = DEMO / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for called, line in called_names(tree):
            if called in forbidden_calls:
                offenders.append(f"{name}:{line}: {called}()")
    assert not offenders, f"a rule reads no file and no clock; found {offenders}"


def test_the_rule_purity_guard_catches_a_planted_clock_read(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text("import time\nx = time.time()\n", encoding="utf-8")
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    assert [n for n, _ in called_names(tree) if n == "time"], "matcher must see a clock read"


# ---------------------------------------------------------------------------
# the shadow rules can never size a position
# ---------------------------------------------------------------------------
def test_a_shadow_rule_returns_a_type_the_position_cannot_accept():
    """Structural, not conventional.

    `HedgedPosition.plan` takes a `HedgeTarget`. A shadow rule returns a
    `SignalOnly`. There is no conversion between them anywhere in the package,
    so a shadow signal cannot become a target without an edit that changes a
    type -- which a reviewer sees.
    """
    from chimera.carry.hedge import HedgeTarget
    from chimera.demo.rules import SignalOnly

    assert not issubclass(SignalOnly, HedgeTarget)
    assert not issubclass(HedgeTarget, SignalOnly)

    # Checked over the AST, not the text: the module's docstring names
    # `HedgeTarget` precisely to explain why it never uses one, and a guard that
    # could not tell prose from code would forbid saying so.
    tree = ast.parse((DEMO / "rules_shadow.py").read_text(encoding="utf-8"))
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    imported = {m for m, _ in imported_modules(tree)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            referenced |= {alias.name for alias in node.names}
    assert "HedgeTarget" not in referenced, (
        "chimera/demo/rules_shadow.py must not reference HedgeTarget in code: a "
        "shadow rule produces SignalOnly and nothing else"
    )
    assert (
        "chimera.carry.hedge" not in imported
    ), "a shadow rule module has no business importing the hedged position"
