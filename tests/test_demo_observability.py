"""Section 11.1's runner metrics: that they exist, and that they cannot decide anything.

This file is `tests/test_demo_telemetry.py`'s job under the name the PR-12 task
gave it, and it is deliberately separate from `tests/test_observability.py`. That
file answers a question about *configuration* — does every dashboard panel and
alert reference a series this repository exports — over `conf/` and `grafana/`.
This one answers a question about *code*: which module may emit, what a label may
contain, and whether an observation can reach the thing it observes. Merging them
would put an AST walk of three packages into a file whose fixtures are YAML.

Two failures are guarded against, and they pull in opposite directions.

*A series nothing writes.* An alert on an unwritten series reads as healthy for
ever, which is the exact failure `tests/test_observability.py`'s docstring says
that file exists to prevent. So the behavioural half of this file drives real
campaigns through `tests/demo_harness.py` and reads the counters before and
after: a deleted `.inc()` fails a delta, not a docstring.

*An observation that changes what it observes.* A metric that fed a rule, a
target, an Aegis input, a fill price or the decision log would make a campaign's
evidence a function of its own monitoring. Three independent guards:

1. **Emission surface.** An AST walk proves that the only file under
   ``chimera/demo/`` or ``chimera/carry/`` importing :mod:`chimera.metrics` is
   ``chimera/demo/telemetry.py``, and that no module in the runtime packages
   reads a metric value back.
2. **Position.** Within ``chimera/demo/runner.py`` every ``self.telemetry.…``
   call is a bare statement, none sits inside the rule-evaluation loop, and none
   precedes the DECISION record inside ``_decide`` — so the risk check, the
   execution, the reconciliation and the persistence are all behind it.
3. **Behaviour.** Three campaigns over the same synthetic day — one with the real
   emitter, one with :class:`NullTelemetry`, one with a wall clock frozen three
   hundred million seconds away — must write byte-identical decision logs. This
   is the load-bearing one; 1 and 2 are lexical and would not by themselves catch
   an emitter reached through an alias or a callback.

**Limits of the structural guards, stated plainly.** They read the source of
named files. They say nothing about a metric read through ``getattr``, through
the Prometheus registry from a module they do not scan, or in a package added
later; and the positional guard is lexical, so it constrains where a call is
WRITTEN in ``runner.py``, not what the functions it calls go on to do. Every one
of them ships with a negative control, because a matcher that has silently
stopped matching proves nothing at all.

Every expectation here is written out literally. Nothing is derived from the
constant or the function under test: a rename in :mod:`chimera.metrics` must fail
a comparison against a tuple typed out from section 11.1, not agree with itself.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter as CountBy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import demo_harness
from chimera import metrics
from chimera.carry.hedge import HedgeState
from chimera.demo import telemetry as demo_telemetry
from chimera.demo.decision_log import DecisionLog, RecordKind
from chimera.demo.runner import RunnerState
from chimera.demo.telemetry import NullTelemetry, RunnerTelemetry

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "chimera" / "demo"
CARRY = REPO / "chimera" / "carry"
FUTURES = REPO / "chimera" / "futures"

requires_prometheus = pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE, reason="prometheus_client is not installed"
)

# --- literal oracles -------------------------------------------------------
#
# Typed out from section 11.1 of the adopted demo plan and from the task's own
# list. If a series is renamed in chimera/metrics.py, this tuple does not follow
# it, which is the whole point.
SECTION_11_1_NAMES: tuple[str, ...] = (
    "chimera_demo_up",
    "chimera_demo_state",
    "chimera_demo_heartbeat_timestamp",
    "chimera_demo_ticks_total",
    "chimera_demo_decisions_total",
    "chimera_demo_last_minute_age_seconds",
    "chimera_demo_feed_age_seconds",
    "chimera_demo_hedge_state",
    "chimera_demo_hedge_imbalance_btc",
    "chimera_demo_equity",
    "chimera_demo_net_pnl",
    "chimera_demo_funding_total",
    "chimera_demo_basis",
    "chimera_demo_liquidation_distance",
    "chimera_demo_log_records_total",
    "chimera_demo_log_write_errors_total",
    "chimera_demo_disk_free_bytes",
    "chimera_demo_funding_adverse_streak",
)

#: Attribute name -> the label tuple that attribute is allowed to carry. Adding
#: a series, or widening one's labels, has to be a deliberate edit here.
EXPECTED_LABELS: dict[str, tuple[str, ...]] = {
    "DEMO_UP": (),
    "DEMO_STATE": ("state",),
    "DEMO_HEARTBEAT": (),
    "DEMO_TICKS": (),
    "DEMO_DECISIONS": ("rule", "kind"),
    "DEMO_LAST_MINUTE_AGE": (),
    "DEMO_FEED_AGE": ("market",),
    "DEMO_HEDGE_STATE": ("state",),
    "DEMO_HEDGE_IMBALANCE": (),
    "DEMO_EQUITY": (),
    "DEMO_NET_PNL": (),
    "DEMO_FUNDING": ("direction",),
    "DEMO_BASIS": (),
    "DEMO_LIQUIDATION_DISTANCE": ("leg",),
    "DEMO_LOG_RECORDS": ("kind",),
    "DEMO_LOG_WRITE_ERRORS": (),
    "DEMO_DISK_FREE": (),
    "DEMO_FUNDING_ADVERSE_STREAK": (),
}

#: The only label names the demo family may carry, written out rather than read
#: off EXPECTED_LABELS, so a widened label has to break two independent oracles.
DEMO_LABEL_NAMES: frozenset[str] = frozenset(
    {"state", "rule", "kind", "market", "direction", "leg"}
)

#: Section 8.1's states, typed out. Thirteen.
RUNNER_STATE_VALUES: tuple[str, ...] = (
    "STARTUP",
    "SELF_CHECK",
    "RECOVER",
    "READY",
    "DATA_READY",
    "RULE_EVALUATION",
    "RISK_CHECK",
    "EXECUTION",
    "RECONCILIATION",
    "PERSISTENCE",
    "REPORTING",
    "HALT",
    "SHUTDOWN",
)
#: Section 6.1's hedge states, typed out. Seven.
HEDGE_STATE_VALUES: tuple[str, ...] = (
    "FLAT",
    "OPENING",
    "HEDGED",
    "PARTIAL",
    "REBALANCING",
    "CLOSING",
    "DISPUTED",
)
#: Section 9.1's record kinds, typed out. Twelve.
RECORD_KIND_VALUES: tuple[str, ...] = (
    "DECISION",
    "FUNDING",
    "RECONCILIATION",
    "OPERATOR",
    "HALT",
    "RESUME",
    "STARTUP",
    "SHUTDOWN",
    "INCOMPLETE_STATE",
    "SKIPPED_STALE",
    "LIQUIDATION_TOUCH",
    "RECOVERY",
)

#: The mode family, by the names `chimera.metrics` gives it. None of these may be
#: reachable from the demo path: a mode metric on a single-mode campaign would
#: advertise a mode selection that is not happening.
MODE_NAMES: tuple[str, ...] = (
    "MODE_SELECTED",
    "MODE_ELIGIBLE",
    "MODE_DECISIONS",
    "MODE_TRANSITIONS",
    "MODE_TRANSITION_FLATTENS",
    "MODE_CONSENSUS_STATE",
    "MODE_RISK_VETOES",
    "mark_mode_decision",
    "mark_mode_transition",
    "mark_mode_veto",
)

#: prometheus_client internals a caller uses to READ a series back. A runtime
#: module naming one of these is reading its own monitoring.
METRIC_READ_ATTRS: frozenset[str] = frozenset(
    {"_value", "_sum", "_buckets", "_metrics", "_child_samples"}
)
METRIC_READ_CALLS: frozenset[str] = frozenset({"collect", "get_sample_value"})

#: Methods that write, plan, persist, permit or decide. The emitter may call none
#: of them: it observes state that already exists.
MUTATING_CALLS: frozenset[str] = frozenset(
    {
        "append",
        "apply",
        "book_costs",
        "book_entry",
        "book_funding",
        "check_kill_switch",
        "correct",
        "dispute",
        "emergency_reduce",
        "evaluate",
        "evaluate_entry",
        "execute_target",
        "flatten",
        "halt",
        "mark",
        "mark_processed",
        "mark_to_market",
        "mkdir",
        "note_feed",
        "note_funding_settlement",
        "note_reconciliation",
        "observe",
        "plan",
        "recover",
        "reconstruct",
        "record_order",
        "resolve",
        "resume",
        "save",
        "settle_funding",
        "touch",
        "unlink",
        "update_equity",
        "write",
        "write_text",
    }
)


# --- reading the declarations from the source ------------------------------
def _declared_string(node: ast.AST) -> str:
    """The literal a metric's name/documentation argument evaluates to."""
    if isinstance(node, ast.Constant):
        return str(node.value)
    assert isinstance(node, ast.JoinedStr), f"unreadable metric argument: {ast.dump(node)}"
    parts = []
    for piece in node.values:
        if isinstance(piece, ast.Constant):
            parts.append(str(piece.value))
            continue
        assert isinstance(piece.value, ast.Name) and piece.value.id == "_PREFIX", (
            "a demo metric name interpolates something other than _PREFIX, so its series "
            "name cannot be read without importing prometheus_client"
        )
        parts.append(metrics._PREFIX)
    return "".join(parts)


def declared_demo_metrics() -> dict[str, dict[str, object]]:
    """``{attribute: {kind, name, labels}}`` for every ``DEMO_*`` declaration.

    Read from the source rather than from the objects, so the structural half of
    this file still runs on a machine where ``prometheus_client`` is missing and
    every metric in the module is an unnamed no-op stub. ``DEMO_METRIC_NAMES`` is
    an annotated tuple assignment rather than a call and is skipped by the shape
    of the walk, which is what lets the two be compared against each other.
    """
    tree = ast.parse(Path(metrics.__file__).read_text(encoding="utf-8"))
    declared: dict[str, dict[str, object]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("DEMO_"):
            continue
        call = node.value
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        unreadable = set(keywords) - {"buckets", "labelnames"}
        assert not unreadable, (
            f"{target.id} is declared with {sorted(map(str, unreadable))}, which this "
            "reader does not understand, so its labels cannot be reviewed"
        )
        if "labelnames" in keywords:
            labels = tuple(ast.literal_eval(keywords["labelnames"]))
        elif len(call.args) > 2:
            labels = tuple(ast.literal_eval(call.args[2]))
        else:
            labels = ()
        declared[target.id] = {
            "kind": call.func.id,
            "name": _declared_string(call.args[0]),
            "labels": labels,
        }
    return declared


def python_sources(*roots: Path) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        out.extend(root.rglob("*.py") if root.is_dir() else [root])
    return sorted(out)


def imports_metrics(tree: ast.AST) -> bool:
    """Whether this module can name :mod:`chimera.metrics` at all.

    Catches all three spellings: ``import chimera.metrics``, ``from chimera
    import metrics`` and ``from chimera.metrics import X``.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("chimera.metrics") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("chimera.metrics"):
                return True
            if module == "chimera" and any(a.name == "metrics" for a in node.names):
                return True
    return False


def metric_importers(*roots: Path) -> list[str]:
    return [
        path.relative_to(REPO).as_posix()
        for path in python_sources(*roots)
        if imports_metrics(ast.parse(path.read_text(encoding="utf-8")))
    ]


def metric_reads(tree: ast.AST) -> list[str]:
    """Every place a module reaches for a metric's stored value."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in METRIC_READ_ATTRS:
            found.append(f"{node.attr}@{node.lineno}")
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in METRIC_READ_CALLS:
                found.append(f"{name}@{node.lineno}")
    return found


def telemetry_calls(node: ast.AST) -> list[ast.Call]:
    """Every ``self.telemetry.<method>(...)`` at or below ``node``."""
    found: list[ast.Call] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "telemetry"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "self"
        ):
            found.append(child)
    return found


def named_identifiers(tree: ast.AST) -> set[str]:
    """Every name a module can refer to: bare names, attributes and import aliases."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])
    return names


def function_named(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is no longer a function in chimera/demo/runner.py")


def runner_tree() -> ast.Module:
    return ast.parse((DEMO / "runner.py").read_text(encoding="utf-8"))


# --- reading the values ----------------------------------------------------
def value_of(metric, **labels):
    """One series' current value, creating the child at zero if it is new."""
    return (metric.labels(**labels) if labels else metric)._value.get()


def mode_children() -> dict[str, set]:
    """The label children each mode series currently holds."""
    return {name: set(getattr(metrics, name)._metrics) for name in MODE_NAMES[:7]}


# --------------------------------------------------------------------------- #
# A. the declarations
# --------------------------------------------------------------------------- #
def test_the_demo_metric_names_are_section_11_1s_exactly():
    assert len(metrics.DEMO_METRIC_NAMES) == len(set(metrics.DEMO_METRIC_NAMES))
    assert set(metrics.DEMO_METRIC_NAMES) == set(SECTION_11_1_NAMES)
    assert len(SECTION_11_1_NAMES) == 18
    for name in metrics.DEMO_METRIC_NAMES:
        assert name.startswith("chimera_demo_"), name


def test_every_pinned_demo_name_is_a_declaration_in_the_module():
    """The pinned tuple and the objects must not be able to drift apart.

    One of them alone is half a guard: a name deleted from both would pass the
    test above, and a metric declared but left out of the tuple would leave a
    dashboard panel with nothing behind it.
    """
    declared = declared_demo_metrics()
    assert {info["name"] for info in declared.values()} == set(metrics.DEMO_METRIC_NAMES)
    assert set(declared) == {attr for attr in dir(metrics) if attr.startswith("DEMO_")} - {
        "DEMO_METRIC_NAMES"
    }


def test_the_declared_demo_metrics_carry_exactly_the_reviewed_labels():
    declared = declared_demo_metrics()
    assert set(declared) == set(EXPECTED_LABELS)
    for attr, expected in EXPECTED_LABELS.items():
        assert declared[attr]["labels"] == expected, attr
        assert set(declared[attr]["labels"]) <= DEMO_LABEL_NAMES, attr


def test_every_demo_label_domain_is_a_bounded_enum():
    """A label whose values grow with traffic is one time series per event, kept
    for ever. Each domain below is fixed by a committed Enum or by this module."""
    assert tuple(state.value for state in RunnerState) == RUNNER_STATE_VALUES
    assert tuple(state.value for state in HedgeState) == HEDGE_STATE_VALUES
    assert tuple(kind.value for kind in RecordKind) == RECORD_KIND_VALUES
    assert demo_telemetry.MARKETS == ("um", "spot")
    assert demo_telemetry.LEGS == ("spot", "perp")
    assert demo_telemetry.SIGNAL_KINDS == ("target", "signal_only")
    assert demo_telemetry.VETO_LABELS == frozenset({"no_intent"})


def test_the_demo_family_lives_in_its_own_banner_section():
    """Three committed tests slice chimera/metrics.py on its ``# --- `` banners.

    A demo series placed inside the trading-modes span would be read there as a
    mode metric with an unbounded label — which has happened once already, to the
    recorder family. The banners are asserted here so it cannot happen again.
    """
    source = Path(metrics.__file__).read_text(encoding="utf-8")
    banners = [
        line.split("---")[1].strip()
        for line in source.splitlines()
        if line.startswith("# --- ")
    ]
    assert banners[-1] == "the demo runner", banners
    block = source.split("# --- the demo runner")[1]
    assert "DEMO_UP" in block, "the section was sliced wrongly and would pass vacuously"
    labels = set(re.findall(r'"(\w+)"\]', block)) | set(re.findall(r'\["(\w+)",', block))
    assert labels <= DEMO_LABEL_NAMES, f"the demo family gained {sorted(labels)}"
    recorder = source.split("# --- prospective recorder")[1].split("# --- trading modes")[0]
    modes = source.split("# --- trading modes")[1].split("# --- system")[0]
    assert "_demo_" not in recorder
    assert "_demo_" not in modes


# --------------------------------------------------------------------------- #
# B. the emission surface
# --------------------------------------------------------------------------- #
def test_only_the_telemetry_module_imports_metrics_on_the_demo_path():
    assert metric_importers(DEMO, CARRY) == ["chimera/demo/telemetry.py"]


def test_only_the_executor_imports_metrics_in_the_futures_package():
    assert metric_importers(FUTURES) == ["chimera/futures/executor.py"]


def test_the_runner_names_no_metric_object():
    """The runner publishes through the emitter or not at all."""
    source = (DEMO / "runner.py").read_text(encoding="utf-8")
    assert "chimera.metrics" not in source
    assert "DEMO_" not in source


def test_the_import_guard_catches_all_three_spellings(tmp_path):
    """The negative control. A matcher that stopped matching proves nothing."""
    for spelling in (
        "import chimera.metrics\n",
        "from chimera import metrics\n",
        "from chimera.metrics import EQUITY\n",
    ):
        planted = tmp_path / "planted.py"
        planted.write_text(spelling, encoding="utf-8")
        assert imports_metrics(ast.parse(spelling)), spelling
    planted.write_text("from chimera.risk import RiskEngine\n", encoding="utf-8")
    assert not imports_metrics(ast.parse(planted.read_text(encoding="utf-8")))


def test_no_runtime_module_reads_a_metric_value():
    """Including the CLIs, which is where PR-12 put the only new metrics import.

    `tools/demo_run.py` is the module this PR newly gives a `chimera.metrics`
    import -- for `serve_metrics` -- and it is also the module that decides
    `--allow-dirty` and hands the runner its telemetry. Leaving it out of this
    scan left the one obvious place a metric could be read back into a decision
    unguarded, and a planted `if DEMO_EQUITY._value.get() >= 0.0` there passed
    the whole suite.
    """
    offenders = {}
    active_clis = (
        REPO / "tools" / "demo_run.py",
        REPO / "tools" / "demo_report.py",
        REPO / "tools" / "replay_parity.py",
    )
    for path in python_sources(
        DEMO, CARRY, FUTURES, REPO / "chimera" / "risk.py", *active_clis
    ):
        hits = metric_reads(ast.parse(path.read_text(encoding="utf-8")))
        if hits:
            offenders[path.relative_to(REPO).as_posix()] = hits
    hits = metric_reads(ast.parse(Path(metrics.__file__).read_text(encoding="utf-8")))
    if hits:
        offenders["chimera/metrics.py"] = hits
    assert not offenders, (
        "a runtime module reads a metric back; observability would then be an "
        f"input to the thing it observes. Found {offenders}"
    )


def test_the_metric_read_guard_catches_a_planted_read(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        "def f(metrics, registry):\n"
        "    if metrics.EQUITY._value.get() < 0:\n"
        "        return registry.get_sample_value('chimera_equity')\n"
        "    return list(metrics.EQUITY.collect())\n",
        encoding="utf-8",
    )
    hits = metric_reads(ast.parse(planted.read_text(encoding="utf-8")))
    assert {h.split("@")[0] for h in hits} == {"_value", "get_sample_value", "collect"}


def test_every_telemetry_observation_method_returns_none():
    """A method that returned something could be read; one that cannot, cannot."""
    tree = ast.parse((DEMO / "telemetry.py").read_text(encoding="utf-8"))
    seen = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("on_"):
            continue
        seen += 1
        assert (
            isinstance(node.returns, ast.Constant) and node.returns.value is None
        ), f"{node.name} is not annotated -> None"
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                assert child.value is None, f"{node.name} returns a value"
    assert seen == 16, f"expected eight methods on each of the two classes, saw {seen}"


def test_the_telemetry_module_calls_no_mutating_method():
    """The emitter reads state that already exists. It writes nothing but metrics."""
    tree = ast.parse((DEMO / "telemetry.py").read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name in MUTATING_CALLS:
            offenders.append(f"{name}@{node.lineno}")
    assert not offenders, f"chimera/demo/telemetry.py calls {offenders}"


def test_the_mutating_call_guard_catches_a_planted_call(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text("position.ledger.save()\nrisk.halt('x')\n", encoding="utf-8")
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    hits = {getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert hits & MUTATING_CALLS == {"save", "halt"}


def test_the_demo_path_names_no_mode_series():
    """`chimera.modes` is not on the demo deployment, in any spelling."""
    offenders = {}
    for path in python_sources(DEMO, CARRY, REPO / "tools" / "demo_run.py"):
        names = named_identifiers(ast.parse(path.read_text(encoding="utf-8")))
        hit = names & set(MODE_NAMES)
        if hit:
            offenders[path.relative_to(REPO).as_posix()] = sorted(hit)
    assert not offenders, f"a mode series is reachable from the demo path: {offenders}"


def test_the_mode_name_guard_catches_a_planted_reference(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        "from chimera.metrics import MODE_SELECTED\nmetrics.mark_mode_veto('a', 'b')\n",
        encoding="utf-8",
    )
    names = named_identifiers(ast.parse(planted.read_text(encoding="utf-8")))
    assert names & set(MODE_NAMES) == {"MODE_SELECTED", "mark_mode_veto"}


# --------------------------------------------------------------------------- #
# C. where the runner may emit
# --------------------------------------------------------------------------- #
def test_the_runner_never_uses_a_telemetry_call_as_a_value():
    """Every emission is a statement whose result is discarded.

    Lexical, and that is the limit: it says the runner does not WRITE a telemetry
    call into an expression. What the emitter does once called is covered by the
    read guard above and by the byte-identity runs below.
    """
    tree = runner_tree()
    statements = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr)}
    calls = telemetry_calls(tree)
    # Eight, named: on_state, on_log_write_error, on_record, on_minute,
    # on_reporting, on_halt, on_position, on_shutdown. The count is written out
    # so that an emission added without a reader thinking about where it sits in
    # the tick is a failing test rather than a silent ninth call.
    assert len(calls) == 8, f"expected eight emission points, found {len(calls)}"
    assert {call.func.attr for call in calls} == {
        "on_state",
        "on_log_write_error",
        "on_record",
        "on_minute",
        "on_reporting",
        "on_halt",
        "on_position",
        "on_shutdown",
    }
    for call in calls:
        method = call.func.attr
        assert id(call) in statements, (
            f"self.telemetry.{method}() at line {call.lineno} is used as a value; a "
            "metric would then be an input to the runner"
        )


def test_the_call_as_a_value_guard_catches_a_planted_assignment():
    planted = ast.parse("self.halt_reason = self.telemetry.on_state(state.value)\n")
    statements = {id(n.value) for n in ast.walk(planted) if isinstance(n, ast.Expr)}
    calls = telemetry_calls(planted)
    assert len(calls) == 1 and id(calls[0]) not in statements


def test_no_telemetry_call_sits_inside_the_rule_evaluation_loop():
    """Section 8.1's RULE_EVALUATION must be able to name nothing observational."""
    tick = function_named(runner_tree(), "tick")
    loops = [
        node
        for node in ast.walk(tick)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Attribute)
        and node.iter.attr == "rules"
    ]
    assert len(loops) == 1, "the rule-evaluation loop moved; this guard is now vacuous"
    assert telemetry_calls(loops[0]) == []


def test_no_telemetry_call_precedes_the_decision_record():
    """Everything decision-derived is emitted after the record is on disk.

    ``_decide`` runs the risk check, the execution, the reconciliation and the
    persistence before it appends the DECISION record; asserting that no emission
    appears before that append covers all four in one statement, and asserting
    that one appears after it keeps the check from passing vacuously.
    """
    decide = function_named(runner_tree(), "_decide")
    index = _decision_append_index(decide)
    before = [c for stmt in decide.body[: index + 1] for c in telemetry_calls(stmt)]
    after = [c for stmt in decide.body[index + 1 :] for c in telemetry_calls(stmt)]
    assert before == [], f"an emission precedes the DECISION record at {before[0].lineno}"
    assert [c.func.attr for c in after] == ["on_reporting"]


def _decision_append_index(function: ast.FunctionDef) -> int:
    """Index in ``function.body`` of the statement appending the DECISION record."""
    for index, statement in enumerate(function.body):
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", None) != "_append" or not node.args:
                continue
            if "DECISION" in ast.dump(node.args[0]):
                return index
    raise AssertionError("_decide no longer appends a DECISION record at its top level")


def test_the_ordering_guard_catches_a_planted_early_emission():
    planted = ast.parse(
        "def _decide(self):\n"
        "    self.telemetry.on_reporting(position=self.position)\n"
        "    record_hash = self._append(RecordKind.DECISION, 0, {})\n"
    )
    decide = function_named(planted, "_decide")
    index = _decision_append_index(decide)
    before = [c for stmt in decide.body[: index + 1] for c in telemetry_calls(stmt)]
    assert [c.func.attr for c in before] == ["on_reporting"]


# --------------------------------------------------------------------------- #
# D. what a real run actually moves
# --------------------------------------------------------------------------- #
@requires_prometheus
def test_a_tick_moves_the_tick_counter(tmp_path):
    harness = demo_harness.build(tmp_path)
    before = value_of(metrics.DEMO_TICKS)
    harness.run(30)
    assert value_of(metrics.DEMO_TICKS) - before == 30


@requires_prometheus
def test_every_record_moves_its_kind_counter(tmp_path):
    """The oracle is the log file, counted independently of the emitter."""
    before = {
        kind: value_of(metrics.DEMO_LOG_RECORDS, kind=kind) for kind in RECORD_KIND_VALUES
    }
    harness = demo_harness.build(tmp_path)
    harness.run(12)
    harness.runner.shutdown("done")
    written = CountBy(record["kind"] for record in harness.records())
    assert written["DECISION"] == 12 and written["STARTUP"] == 1
    for kind in RECORD_KIND_VALUES:
        moved = value_of(metrics.DEMO_LOG_RECORDS, kind=kind) - before[kind]
        assert moved == written[kind], kind


@requires_prometheus
def test_a_log_write_error_is_counted_and_still_raised(tmp_path, monkeypatch):
    harness = demo_harness.build(tmp_path)
    before = value_of(metrics.DEMO_LOG_WRITE_ERRORS)

    def boom(self, record):
        raise OSError("no space left on device")

    monkeypatch.setattr(DecisionLog, "append", boom)
    with pytest.raises(OSError):
        harness.runner.shutdown("done")
    assert value_of(metrics.DEMO_LOG_WRITE_ERRORS) - before == 1


@requires_prometheus
def test_the_hedge_state_gauge_shows_exactly_one_state(tmp_path):
    """Two states at 1 is what a label left behind by the previous state looks like."""
    harness = demo_harness.build(tmp_path)
    harness.run(10)
    values = {
        state: value_of(metrics.DEMO_HEDGE_STATE, state=state) for state in HEDGE_STATE_VALUES
    }
    assert sorted(values.values()) == [0.0] * 6 + [1.0], values
    live = [state for state, value in values.items() if value == 1.0]
    assert live == [harness.runner.position.state.value]


@requires_prometheus
def test_the_runner_state_gauge_shows_exactly_one_state(tmp_path):
    """The twin of the hedge sweep above, and it was missing.

    `set_demo_state` sweeps every state to 0 before setting one to 1, and
    `RunnerHalted` reads `chimera_demo_state{state="HALT"}`. Replacing the sweep
    with a single `labels(state=state).set(1.0)` leaves the previous state at 1
    for ever -- a runner that looks halted long after it resumed, or one that
    never looks halted at all -- and no test noticed.
    """
    harness = demo_harness.build(tmp_path)
    harness.run(4)
    values = {
        state: value_of(metrics.DEMO_STATE, state=state) for state in RUNNER_STATE_VALUES
    }
    live = [state for state, value in values.items() if value == 1.0]
    assert live == [harness.runner.state.value], values
    assert sorted(values.values()) == [0.0] * (len(RUNNER_STATE_VALUES) - 1) + [1.0]


@requires_prometheus
def test_the_gauge_values_are_the_quantities_they_are_named_for(tmp_path):
    """Read against the position and the ledger, never against the gauge itself.

    Every one of these was mutated to a constant, or to a neighbouring quantity,
    against the suite as it stood, and none of it failed: the writer check is an
    AST scan, so it proves the module reaches for the object and says nothing
    about what it stores. `chimera_demo_net_pnl` set to absolute equity is the
    one that matters most -- the dashboard line called "net PnL" would then be
    the account balance, off by the whole starting capital.
    """
    harness = demo_harness.build(tmp_path)
    outcomes = harness.run(6)
    last_minute = harness.first_minute_ms() + (len(outcomes) - 1) * 60_000
    position = harness.runner.position
    ledger = position.ledger.state
    # Re-derive the mark from the same minute the last tick reported on, through
    # the position's own accounting rather than from anything the emitter did.
    mark = position.mark_to_market(harness.runner.cursor.state_for(last_minute))

    assert value_of(metrics.DEMO_UP) == 1.0
    assert value_of(metrics.DEMO_EQUITY) == pytest.approx(float(mark.equity))
    assert value_of(metrics.DEMO_NET_PNL) == pytest.approx(float(mark.equity - ledger.capital))
    assert value_of(metrics.DEMO_NET_PNL) != pytest.approx(
        float(mark.equity)
    ), "net PnL equals absolute equity; the starting capital was not subtracted"
    assert value_of(metrics.DEMO_BASIS) == pytest.approx(float(mark.basis))
    assert value_of(metrics.DEMO_HEARTBEAT) > 0.0


@requires_prometheus
def test_the_decision_counter_tells_an_actionable_rule_from_a_shadow_one(tmp_path):
    """`kind` is the label the counter exists for, and nothing read it.

    Hard-coding it to "signal_only" reports every real decision as a shadow
    signal, which is the difference between a rule that traded and one that
    watched.
    """
    harness = demo_harness.build(tmp_path)
    harness.run(6)
    seen = {
        (labels[0], labels[1]): metrics.DEMO_DECISIONS.labels(
            rule=labels[0], kind=labels[1]
        )._value.get()
        for labels in metrics.DEMO_DECISIONS._metrics
    }
    assert seen, "no decision was counted at all"
    kinds = {kind for (_rule, kind) in seen}
    assert kinds <= set(demo_telemetry.SIGNAL_KINDS)
    # The fixture's rule registry holds the carry rule, which is actionable, so
    # the actionable label must actually have been used -- a counter that only
    # ever wrote the shadow label would satisfy the subset check above.
    actionable = demo_telemetry.SIGNAL_KINDS[0]
    assert any(
        value > 0 for (_rule, kind), value in seen.items() if kind == actionable
    ), f"no actionable decision was counted; saw {seen}"


@requires_prometheus
def test_a_flat_leg_reports_nan_rather_than_zero_liquidation_distance(tmp_path):
    """Zero would read as "liquidation is 100% away"; NaN is unorderable, so no
    alert can conclude anything at all about a leg that has no position."""
    harness = demo_harness.build(tmp_path)
    assert harness.runner.position.state is HedgeState.FLAT
    _report(harness)
    for leg in ("spot", "perp"):
        assert math.isnan(value_of(metrics.DEMO_LIQUIDATION_DISTANCE, leg=leg)), leg
    harness.run(5)
    for leg in ("spot", "perp"):
        distance = value_of(metrics.DEMO_LIQUIDATION_DISTANCE, leg=leg)
        assert not math.isnan(distance) and 0.0 < distance < 1.0, leg


@requires_prometheus
def test_a_paid_settlement_moves_only_the_paid_direction(tmp_path):
    """Amendment A10's convention: a negative flow is funding PAID."""
    harness = demo_harness.build(tmp_path)
    _report(harness)  # the first report adopts the ledger's totals as the baseline
    paid_before = value_of(metrics.DEMO_FUNDING, direction="paid")
    received_before = value_of(metrics.DEMO_FUNDING, direction="received")

    harness.runner.position.ledger.book_funding(1, Decimal("-4.20"))
    _report(harness)

    assert value_of(metrics.DEMO_FUNDING, direction="paid") - paid_before == 4.20
    assert value_of(metrics.DEMO_FUNDING, direction="received") == received_before


@requires_prometheus
def test_a_received_settlement_moves_only_the_received_direction(tmp_path):
    """The other side of the control. The two are never netted into one series."""
    harness = demo_harness.build(tmp_path)
    _report(harness)
    paid_before = value_of(metrics.DEMO_FUNDING, direction="paid")
    received_before = value_of(metrics.DEMO_FUNDING, direction="received")

    harness.runner.position.ledger.book_funding(2, Decimal("1.75"))
    _report(harness)

    assert value_of(metrics.DEMO_FUNDING, direction="received") - received_before == 1.75
    assert value_of(metrics.DEMO_FUNDING, direction="paid") == paid_before


@requires_prometheus
def test_an_unknown_veto_label_collapses_instead_of_creating_a_series(tmp_path):
    """The two-sided control on the one label the runner supplies itself."""
    harness = demo_harness.build(tmp_path)
    known_before = value_of(metrics.REJECTED_ENTRIES, reason="no_intent")
    other_before = value_of(metrics.REJECTED_ENTRIES, reason="other")

    _report(harness, veto={"stage": "risk", "label": "no_intent"})
    _report(harness, veto={"stage": "risk", "label": "a reason invented at 03:00"})

    assert value_of(metrics.REJECTED_ENTRIES, reason="no_intent") - known_before == 1
    assert value_of(metrics.REJECTED_ENTRIES, reason="other") - other_before == 1
    assert "a reason invented at 03:00" not in {
        labels[0] for labels in metrics.REJECTED_ENTRIES._metrics
    }


@requires_prometheus
def test_the_runner_writes_aegis_shared_series(tmp_path):
    """Ruling D8: section 11.1 names these three for the demo deployment too.

    Read against the risk engine, not against the metric, so a deleted `.set()`
    fails rather than agreeing with itself. The halted flag is asserted against a
    LITERAL 0.0 here and against a literal 1.0 in the halt test below, because
    comparing it to `1.0 if risk.snapshot()["halted"] else 0.0` on a clean run is
    `0.0 == 0.0` whatever the emitter does -- the tautology this pair replaces.
    """
    harness = demo_harness.build(tmp_path)
    harness.run(8)
    risk = harness.runner.risk
    assert risk.snapshot()["halted"] is False, "the clean fixture must not halt"
    assert value_of(metrics.RISK_HALTED) == 0.0
    assert value_of(metrics.DRAWDOWN) == pytest.approx(risk.current_drawdown())
    assert value_of(metrics.DEMO_FUNDING_ADVERSE_STREAK) == float(
        risk.snapshot()["funding_adverse_streak"]
    )


@requires_prometheus
def test_a_halt_raises_the_shared_halted_gauge(tmp_path):
    """The other side of the gauge, and the one `RunnerHalted` is named for.

    Every `_halt(...)` path returns before REPORTING, so before `on_halt` existed
    the demo runner could only ever write 0 here and the first disjunct of
    `RunnerHalted` was dead on the one deployment it is named for -- leaving the
    retired Freqtrade container, which writes the same series, as the only thing
    that could raise it. Hence the `job="demo"` matcher on that disjunct and
    hence this test.
    """
    harness = demo_harness.build(tmp_path)
    harness.run(2)
    assert value_of(metrics.RISK_HALTED) == 0.0

    (harness.state_dir / "KILL_SWITCH").touch()
    outcome = harness.tick(harness.runner.cursor.next_minute_ms())

    assert outcome.kind is RecordKind.HALT
    assert harness.runner.state is RunnerState.HALT
    assert harness.runner.risk.snapshot()["halted"] is True
    assert value_of(metrics.RISK_HALTED) == 1.0
    assert value_of(metrics.DEMO_STATE.labels(state="HALT")) == 1.0


@requires_prometheus
def test_the_disk_gauge_and_the_ages_are_finite_and_non_negative(tmp_path):
    """`_age_seconds` floors at zero because the fixture's minutes are in the
    future relative to this host's wall clock, and a negative staleness reads as
    "fresher than possible" rather than as the clock difference it is."""
    harness = demo_harness.build(tmp_path)
    harness.run(4)
    assert value_of(metrics.DEMO_DISK_FREE) > 0
    assert value_of(metrics.DEMO_LAST_MINUTE_AGE) >= 0.0
    for market in ("um", "spot"):
        assert value_of(metrics.DEMO_FEED_AGE, market=market) >= 0.0


def test_the_age_helper_is_two_sided():
    """A minute an hour old is an hour stale; one whose close is ahead is zero."""
    assert demo_telemetry._age_seconds(3_600_000_000_000, 0) == 3600.0
    assert demo_telemetry._age_seconds(0, 3_600_000_000_000) == 0.0


@requires_prometheus
def test_the_mode_series_are_untouched_by_a_demo_run(tmp_path):
    before = mode_children()
    harness = demo_harness.build(tmp_path)
    harness.run(20)
    harness.runner.shutdown("done")
    assert mode_children() == before


# --------------------------------------------------------------------------- #
# E. the load-bearing one: observability cannot reach the evidence
# --------------------------------------------------------------------------- #
def _log_bytes(state_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted((state_dir / "decision_log").glob("*.ndjson"))
    }


def _campaign(tmp_path: Path, name: str, telemetry) -> dict[str, bytes]:
    """One complete synthetic campaign over the same day, into its own directory."""
    harness = demo_harness.build(tmp_path / name, telemetry=telemetry)
    harness.run(120)
    harness.runner.shutdown("done")
    return _log_bytes(harness.state_dir)


def test_the_decision_log_is_byte_identical_with_and_without_telemetry(tmp_path):
    """The control arm. If any metric could reach a decision, these would differ."""
    live = _campaign(tmp_path, "live", None)
    dark = _campaign(tmp_path, "dark", NullTelemetry())
    assert sorted(live) == sorted(dark)
    assert live == dark
    assert sum(len(b) for b in live.values()) > 0, "the campaign wrote nothing"


def test_the_decision_log_is_byte_identical_under_a_frozen_wall_clock(tmp_path):
    """The emitter reads `time.time_ns`; the log must not be able to see it.

    Run C's clock is frozen three hundred million seconds away from run A's, so
    every age, every heartbeat and every derived second differs — and every byte
    of evidence still has to match.
    """
    live = _campaign(tmp_path, "live", None)
    frozen = _campaign(
        tmp_path,
        "frozen",
        RunnerTelemetry(
            state_dir=tmp_path / "frozen",
            states=RUNNER_STATE_VALUES,
            rules=("R1_carry", "R2_frozen_logistic", "R3_daily_momentum"),
            wall_ns=lambda: 1_700_000_000_000_000_000,
        ),
    )
    assert live == frozen


@requires_prometheus
def test_the_reporting_metrics_are_emitted_after_the_record_is_on_disk(tmp_path):
    """The positional guard, checked by running rather than by reading.

    Every REPORTING emission must be preceded by an append, so the record bytes
    exist before any decision-derived series moves.
    """
    order: list[str] = []
    real_append = DecisionLog.append
    real_reporting = RunnerTelemetry.on_reporting

    def append(self, record):
        result = real_append(self, record)
        order.append(f"append:{record['kind']}")
        return result

    def on_reporting(self, **kwargs):
        order.append("report")
        return real_reporting(self, **kwargs)

    DecisionLog.append = append
    RunnerTelemetry.on_reporting = on_reporting
    try:
        harness = demo_harness.build(tmp_path)
        harness.run(5)
    finally:
        DecisionLog.append = real_append
        RunnerTelemetry.on_reporting = real_reporting

    assert order.count("report") == 5
    for index, event in enumerate(order):
        if event == "report":
            assert order[index - 1] == "append:DECISION", order[max(0, index - 3) : index + 1]


# --- shared helper ---------------------------------------------------------
def _report(harness, *, veto=None) -> None:
    """Drive one REPORTING emission with stand-in market values.

    The mark and the market state are written here rather than taken from
    `mark_to_market`, which would move the ledger: the point of these tests is
    what the emitter does with numbers, and a test that mutated the position to
    read a gauge would be doing the very thing the emitter may not.
    """
    harness.runner.telemetry.on_reporting(
        position=harness.runner.position,
        risk=harness.runner.risk,
        mark=SimpleNamespace(
            equity=Decimal("1000000"), basis=Decimal("30"), spot_close=Decimal("60000")
        ),
        market=SimpleNamespace(mark=Decimal("60030")),
        decisions=(),
        veto=veto,
    )


def test_the_harness_records_helper_reads_what_this_file_asserts_on(tmp_path):
    """A guard on the oracle itself: `records()` must see the log on disk."""
    harness = demo_harness.build(tmp_path)
    harness.run(3)
    harness.runner.shutdown("done")
    on_disk = sum(
        len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
        for path in sorted((harness.state_dir / "decision_log").glob("*.ndjson"))
    )
    assert on_disk == len(harness.records()) == 5
    assert json.loads(json.dumps(harness.records()[0]))["kind"] == "STARTUP"
