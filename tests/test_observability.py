"""Every dashboard panel and alert must reference a metric this code exports.

The previous repository had five dashboards querying ten series (`equity`,
`drift_score`, `sharpe_live`, `retrain_trigger`, ...), not one of which was
produced by any code in it. These tests make that impossible to reintroduce
silently.

The demo deployment raises the bar, because `chimera/metrics.py` DECLARING a
series is not the same as some process WRITING one, and an alert on a series
nobody writes is indistinguishable from an alert on a healthy system: it never
fires, for ever. So the demo alert and dashboard checks below carry a
hand-written table naming, per series, the module that writes it, and assert
that module really references it. The tables are literal on purpose. An oracle
collected from the file under test would pass whatever that file happened to
say, which is the shape of tautology a previous sprint shipped.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

from chimera import metrics

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "grafana" / "provisioning" / "dashboards"
CONF_DIR = ROOT / "conf"

#: Both committed rule files. Collected by glob so a third one cannot be added
#: without the per-file checks below picking it up; pinned by the literal test
#: immediately after so deleting one cannot silently shrink the parametrisation
#: to nothing, which would turn a green run into no run at all.
ALERT_FILES = sorted(CONF_DIR.glob("alerts*.yml"))

#: Series Prometheus itself provides, plus PromQL function names we should not
#: mistake for metrics.
_BUILTIN = {"up", "time", "histogram_quantile", "rate", "sum", "by", "le", "avg", "max", "min"}


def exported_metric_names() -> set[str]:
    """Base names plus the suffixed series a histogram actually produces."""
    names = set()
    for attr in dir(metrics):
        value = getattr(metrics, attr)
        name = getattr(value, "_name", None)
        if not name or not str(name).startswith("chimera_"):
            continue
        name = str(name)
        names.add(name)
        # prometheus_client strips the _seconds suffix into _name; restore the
        # series names a scrape would actually show.
        for suffix in ("_bucket", "_count", "_sum", "_total", "_seconds", "_seconds_bucket"):
            names.add(name + suffix)
    return names


def referenced_metrics(expression: str) -> set[str]:
    """Identifiers in a PromQL expression that look like metric names."""
    found = set()
    for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expression):
        if token in _BUILTIN or token.startswith(("$", "on", "ignoring")):
            continue
        if token.startswith("chimera_") or token == "up":
            found.add(token)
    return found


@pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE, reason="prometheus_client is not installed"
)
def test_metrics_module_defines_the_expected_families():
    names = exported_metric_names()
    for expected in (
        "chimera_equity",
        "chimera_drawdown",
        "chimera_exposure",
        "chimera_open_positions",
        "chimera_rejected_entries_total",
        "chimera_risk_halted",
        "chimera_inference_latency_seconds",
        "chimera_inference_errors_total",
        "chimera_predictions_total",
        "chimera_prediction_confidence",
        "chimera_model_info",
        "chimera_last_successful_inference_timestamp",
        "chimera_service_up",
        "chimera_data_delay_seconds",
    ):
        assert expected in names, f"{expected} is not exported"


@pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE, reason="prometheus_client is not installed"
)
@pytest.mark.parametrize("path", sorted(DASHBOARD_DIR.glob("*.json")))
def test_every_dashboard_panel_queries_an_exported_metric(path):
    exported = exported_metric_names()
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    assert dashboard["panels"], f"{path.name} has no panels"

    for panel in dashboard["panels"]:
        assert panel.get("targets"), f"{path.name}: panel '{panel['title']}' has no query"
        for target in panel["targets"]:
            for name in referenced_metrics(target["expr"]):
                assert name in exported, (
                    f"{path.name}: panel '{panel['title']}' queries {name}, "
                    "which nothing in this repository exports"
                )


def test_the_alert_file_set_is_exactly_the_two_committed_ones():
    """A literal name check, so a deleted file cannot empty the parametrisation."""
    assert [path.name for path in ALERT_FILES] == ["alerts.yml", "alerts_demo.yml"]


@pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE, reason="prometheus_client is not installed"
)
@pytest.mark.parametrize("path", ALERT_FILES, ids=lambda path: path.name)
def test_every_alert_rule_queries_an_exported_metric(path):
    exported = exported_metric_names()
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    for group in rules["groups"]:
        for rule in group["rules"]:
            for name in referenced_metrics(rule["expr"]):
                assert name in exported, f"alert {rule['alert']} queries unknown {name}"


def test_prometheus_scrapes_only_services_that_exist():
    """And only ones the DEFAULT topology starts.

    Existing in the `services:` map is not enough. A service behind a
    `profiles:` key is absent from a bare `docker compose up`, so scraping it
    yields a job that is down for ever -- the same defect as scraping a service
    that was never declared, and the one the header of conf/prometheus.yml
    records. Reading the profile here is what makes the check mean "reachable"
    rather than "mentioned".
    """
    prometheus = yaml.safe_load((CONF_DIR / "prometheus.yml").read_text(encoding="utf-8"))
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    default = {
        name for name, service in compose["services"].items() if not service.get("profiles")
    }

    for job in prometheus["scrape_configs"]:
        for static in job["static_configs"]:
            for target in static["targets"]:
                host = target.split(":")[0]
                if host == "localhost":
                    continue
                assert host in default, (
                    f"prometheus scrapes '{host}', which the default compose "
                    "topology does not start"
                )


def test_alertmanager_config_contains_no_placeholder_webhook():
    """A placeholder URL silently swallows every critical alert."""
    text = (CONF_DIR / "alertmanager.yml").read_text(encoding="utf-8")
    assert "example.com" not in text


def test_grafana_datasource_is_provisioned():
    datasource = yaml.safe_load(
        (ROOT / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text(
            encoding="utf-8"
        )
    )
    assert datasource["datasources"][0]["url"].startswith("http://prometheus:")


def test_dashboards_are_mounted_into_grafana():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    mounts = compose["services"]["grafana"]["volumes"]
    assert any(
        "grafana/provisioning" in m for m in mounts
    ), "dashboards in the repository must actually reach the container"


def test_inference_metric_helpers_move_the_counters():
    before_ok = metrics.PREDICTIONS.labels(signal="LONG")._value.get()
    metrics.mark_inference_success("LONG", 0.8, 0.01)
    assert metrics.PREDICTIONS.labels(signal="LONG")._value.get() == before_ok + 1

    before_err = metrics.INFERENCE_ERRORS.labels(kind="timeout")._value.get()
    metrics.mark_inference_failure("timeout")
    assert metrics.INFERENCE_ERRORS.labels(kind="timeout")._value.get() == before_err + 1


# --- the demo deployment -------------------------------------------------
#
# Section 11.2's alerts, section 11.1's dashboard and the two adopted scrape
# jobs. The oracles below are written out by hand and compared against the
# committed files; none of them is derived from the file, constant or function
# it is checking.


#: Section 11.2's ten adopted alerts, sorted. Equality against the sorted names
#: in the file is two-sided by construction: an eleventh alert fails it and a
#: deleted one fails it.
DEMO_ALERT_NAMES: tuple[str, ...] = (
    "ClockSkew",
    "DataGapToday",
    "DiskLow",
    "FundingAdverseStreak",
    "HedgeImbalance",
    "ReconciliationMismatch",
    "RecorderDown",
    "RecorderStreamStale",
    "RunnerDown",
    "RunnerHalted",
)

#: Every series `conf/alerts_demo.yml` may name, and the module that writes it.
#: `chimera/metrics.py` is not an acceptable answer for the second half: it
#: declares series, it does not write them, and a declared-but-unwritten series
#: is exactly the failure this table exists to catch.
DEMO_ALERT_SERIES: dict[str, tuple[str, str]] = {
    "chimera_demo_disk_free_bytes": ("DEMO_DISK_FREE", "chimera/demo/telemetry.py"),
    "chimera_demo_funding_adverse_streak": (
        "DEMO_FUNDING_ADVERSE_STREAK",
        "chimera/demo/telemetry.py",
    ),
    "chimera_demo_heartbeat_timestamp": ("DEMO_HEARTBEAT", "chimera/demo/telemetry.py"),
    "chimera_demo_hedge_imbalance_btc": ("DEMO_HEDGE_IMBALANCE", "chimera/demo/telemetry.py"),
    "chimera_demo_state": ("set_demo_state", "chimera/demo/telemetry.py"),
    "chimera_futures_reconciliation_total": (
        "FUT_RECONCILIATION",
        "chimera/futures/executor.py",
    ),
    "chimera_recorder_clock_skew_ms": ("RECORDER_CLOCK_SKEW", "chimera/recorder/health.py"),
    "chimera_recorder_disk_free_bytes": ("RECORDER_DISK_FREE", "chimera/recorder/health.py"),
    "chimera_recorder_heartbeat_timestamp": (
        "RECORDER_HEARTBEAT",
        "chimera/recorder/health.py",
    ),
    "chimera_recorder_last_event_age_seconds": (
        "RECORDER_LAST_EVENT_AGE",
        "chimera/recorder/health.py",
    ),
    "chimera_recorder_missing_minutes_total": (
        "RECORDER_MISSING_MINUTES",
        "chimera/recorder/health.py",
    ),
    "chimera_risk_halted": ("RISK_HALTED", "chimera/demo/telemetry.py"),
}

#: The eight panels of section 11.1's dashboard, in the order they are laid out.
DEMO_PANEL_TITLES: tuple[str, ...] = (
    "Runner state and heartbeat",
    "Recorder stream ages",
    "Equity and net PnL",
    "Hedge state and imbalance",
    "Funding paid and received",
    "Fees and slippage (both legs combined)",
    "Vetoes by reason",
    "Reconciliation outcomes",
)

#: The same writer table for every series the demo dashboard queries. The two
#: tables overlap but are kept apart: an alert and a panel are allowed to watch
#: different things, and merging them would let one file's oracle cover the
#: other's gap.
DEMO_DASHBOARD_SERIES: dict[str, tuple[str, str]] = {
    "chimera_demo_equity": ("DEMO_EQUITY", "chimera/demo/telemetry.py"),
    "chimera_demo_funding_total": ("DEMO_FUNDING", "chimera/demo/telemetry.py"),
    "chimera_demo_heartbeat_timestamp": ("DEMO_HEARTBEAT", "chimera/demo/telemetry.py"),
    "chimera_demo_hedge_imbalance_btc": ("DEMO_HEDGE_IMBALANCE", "chimera/demo/telemetry.py"),
    "chimera_demo_hedge_state": ("set_hedge_state", "chimera/demo/telemetry.py"),
    "chimera_demo_net_pnl": ("DEMO_NET_PNL", "chimera/demo/telemetry.py"),
    "chimera_demo_state": ("set_demo_state", "chimera/demo/telemetry.py"),
    "chimera_futures_reconciliation_total": (
        "FUT_RECONCILIATION",
        "chimera/futures/executor.py",
    ),
    "chimera_futures_risk_vetoes_total": ("FUT_RISK_VETOES", "chimera/futures/executor.py"),
    "chimera_futures_slippage_bps_bucket": ("FUT_SLIPPAGE_BPS", "chimera/futures/executor.py"),
    "chimera_futures_trading_fees_total": ("FUT_TRADING_FEES", "chimera/futures/executor.py"),
    "chimera_recorder_last_event_age_seconds": (
        "RECORDER_LAST_EVENT_AGE",
        "chimera/recorder/health.py",
    ),
    "chimera_rejected_entries_total": ("REJECTED_ENTRIES", "chimera/demo/telemetry.py"),
}

#: The adopted scrape endpoints. 9102 for the recorder and 9103 for the runner,
#: both from the master plan's deployment table, both written out here rather
#: than read back out of conf/prometheus.yml.
#: Every demo alert's EXACT expression and `for` clause, hand-written here.
#:
#: The series-name checks below cannot see any of what actually decides whether
#: an alert fires: a label matcher, a threshold, or the direction of a
#: comparison. Each of these mutations was tried against the suite as it stood
#: and none of them failed a test -- `state="HALT"` to `state="HALTED"`, a
#: label value `RunnerState` cannot produce; the stale-stream matcher to
#: `um.kline_5m|spot.kline_5m`, two stream ids the gen3 contract does not have,
#: with the threshold moved from 180 seconds to 18000; and the funding panel
#: netted with `sum(...)`. An alert disarmed that way still parses, still names
#: only exported series, and never fires again.
#:
#: So the expression is pinned literally. Editing one here is meant to be
#: deliberate: change the rule file and this table together, and say in the
#: commit message why the alert should now fire differently.
DEMO_ALERT_EXPRESSIONS: dict[str, tuple[str, str]] = {
    "RecorderStreamStale": (
        'chimera_recorder_last_event_age_seconds{stream=~"um.kline_1m|spot.kline_1m'
        '|um.markPrice|um.bookTicker"} > 180',
        "2m",
    ),
    "RecorderDown": ("time() - chimera_recorder_heartbeat_timestamp > 120", "0m"),
    "DataGapToday": ("chimera_recorder_missing_minutes_total > 7", "0m"),
    "ClockSkew": ("abs(chimera_recorder_clock_skew_ms) > 5000", "5m"),
    "RunnerDown": ("time() - chimera_demo_heartbeat_timestamp > 120", "0m"),
    # `job="demo"` is load-bearing: chimera_risk_halted is written by the retired
    # Freqtrade path too, and without the matcher an alert named for the runner
    # can be satisfied by a series the legacy container wrote.
    "RunnerHalted": (
        'chimera_risk_halted{job="demo"} == 1 or chimera_demo_state{state="HALT"} == 1',
        "0m",
    ),
    "ReconciliationMismatch": (
        'increase(chimera_futures_reconciliation_total{outcome="MISMATCH"}[10m]) > 0',
        "0m",
    ),
    "HedgeImbalance": ("abs(chimera_demo_hedge_imbalance_btc) > 0", "5m"),
    "FundingAdverseStreak": ("chimera_demo_funding_adverse_streak >= 2", "0m"),
    # Section 11.2 writes this as `min(a, b) < 10e9`, which is not valid PromQL:
    # `min` aggregates one instant vector and takes no second argument, and
    # `promtool check rules` -- an acceptance criterion for this PR -- rejects it.
    # The disjunction has the same firing semantics and names which disk fired.
    "DiskLow": (
        "chimera_recorder_disk_free_bytes < 10e9 or chimera_demo_disk_free_bytes < 10e9",
        "5m",
    ),
}

#: Every demo panel's EXACT queries, for the same reason and against the same
#: demonstrated mutation: `chimera_demo_funding_total` netted to
#: `sum(chimera_demo_funding_total)` passes a metric-name oracle, because `sum`
#: is filtered out as a PromQL builtin, while collapsing the `direction` label
#: that amendment A10 exists to keep apart.
DEMO_PANEL_EXPRESSIONS: dict[str, tuple[str, ...]] = {
    "Runner state and heartbeat": (
        "chimera_demo_state",
        "time() - chimera_demo_heartbeat_timestamp",
    ),
    "Recorder stream ages": ("chimera_recorder_last_event_age_seconds",),
    "Equity and net PnL": ("chimera_demo_equity", "chimera_demo_net_pnl"),
    "Hedge state and imbalance": (
        "chimera_demo_hedge_state",
        "chimera_demo_hedge_imbalance_btc",
    ),
    "Funding paid and received": ("chimera_demo_funding_total",),
    "Fees and slippage (both legs combined)": (
        "chimera_futures_trading_fees_total",
        "histogram_quantile(0.9, sum(rate(chimera_futures_slippage_bps_bucket[1h])) by (le))",
    ),
    "Vetoes by reason": (
        "sum(increase(chimera_futures_risk_vetoes_total[1h])) by (reason)",
        'sum(increase(chimera_rejected_entries_total{job="demo"}[1h])) by (reason)',
    ),
    "Reconciliation outcomes": (
        "sum(increase(chimera_futures_reconciliation_total[1h])) by (outcome)",
    ),
}

#: The only sources the demo containers may mount. `./conf` is read-only and
#: holds the campaign configuration; the rest are docker-managed named volumes.
#: Anything else -- `./.env`, `./user_data`, an absolute host path -- is a way to
#: put a credential inside a container that must not have one.
ALLOWED_BIND_SOURCES: frozenset[str] = frozenset({"./conf", "recorder_data", "demo_state"})

#: Tokens that must appear in no deployment artifact for the demo path. The
#: acknowledgement token is the one that actually unlocks live trading in
#: `chimera/safety.py`; the rest are the shapes a credential arrives in.
FORBIDDEN_DEPLOYMENT_TOKENS: tuple[str, ...] = (
    "I_UNDERSTAND_THE_RISK",
    "ENABLE_LIVE_TRADING",
    "API_KEY",
    "API_SECRET",
    "BINANCE_KEY",
    "BINANCE_SECRET",
    "EnvironmentFile",
    "run_bot",
)

#: Every supervision artifact this repository ships for the demo path.
DEPLOYMENT_ARTIFACTS: tuple[str, ...] = (
    "deploy/systemd/chimera-demo.service",
    "deploy/systemd/chimera-recorder.service",
    "deploy/docker/Dockerfile.demo",
    "deploy/docker/Dockerfile.recorder",
)

DEMO_SCRAPE_TARGETS: dict[str, str] = {"recorder": "recorder:9102", "demo": "demo:9103"}

#: Every compose service, so the set fails both when one is deleted and when a
#: new one -- a live-trading one, say -- is added without anybody noticing.
COMPOSE_SERVICES: frozenset[str] = frozenset(
    {"freqtrade", "nn_infer", "prometheus", "grafana", "alertmanager", "recorder", "demo"}
)


def demo_alert_rules() -> list[dict]:
    """Every rule in conf/alerts_demo.yml, flattened out of its groups."""
    rules = yaml.safe_load((CONF_DIR / "alerts_demo.yml").read_text(encoding="utf-8"))
    return [rule for group in rules["groups"] for rule in group["rules"]]


def dashboards() -> list[tuple[Path, dict]]:
    """Every provisioned dashboard, as (path, parsed JSON)."""
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(DASHBOARD_DIR.glob("*.json"))
    ]


def panel_expressions(board: dict) -> list[str]:
    """Every PromQL expression a dashboard's top-level panels query.

    Top level only, deliberately, and asserted to be the whole story by
    ``test_exactly_one_provisioned_dashboard_covers_the_demo_campaign``: a row
    panel carries its children in its own nested ``panels`` key, where this walk
    -- and the one in ``test_every_dashboard_panel_queries_an_exported_metric``
    -- would not see them.
    """
    return [target["expr"] for panel in board["panels"] for target in panel.get("targets", [])]


def _references(source: str, attribute: str) -> bool:
    """True when ``source`` does something *to* ``attribute`` in executable code.

    Deliberately an AST walk rather than a text search. A text search was tried
    first and a mutation defeated it: commenting the emitting line out left the
    name in the file, so the guard went on reporting a series as written when
    nothing wrote it any more. The AST has neither comments nor strings in it,
    and an imported-but-unused name appears in it only as an ``alias``, so
    neither survives here.

    Two shapes count. ``NAME.something`` covers ``DEMO_EQUITY.set(...)``,
    ``DEMO_FUNDING.labels(...).inc()`` and ``metrics.FUT_RECONCILIATION.labels(...)``;
    ``NAME(...)`` covers the helpers, ``set_demo_state(...)`` and
    ``set_hedge_state(...)``, which are how the two swept gauges are written.
    What this proves is that the module reaches for the object, not that the
    line runs on any given campaign -- tests/test_demo_observability.py runs
    whole campaigns for that question.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Name) and base.id == attribute:
                return True
            if isinstance(base, ast.Attribute) and base.attr == attribute:
                return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == attribute:
                return True
    return False


def writer_references(attribute: str, module_path: str) -> bool:
    """``_references`` against a module of this repository."""
    return _references((ROOT / module_path).read_text(encoding="utf-8"), attribute)


def test_the_writer_check_can_tell_a_used_metric_from_an_unused_one():
    """The negative half of the two checks that lean on ``writer_references``.

    Without it, a helper that returned True for everything would make both
    "is it emitted" assertions vacuous. ``MODE_SELECTED`` is a real series in
    chimera/metrics.py that the demo emitter deliberately does not write -- the
    demo deployment runs no trading-mode machinery -- so it is the honest
    counter-example.
    """
    assert writer_references("DEMO_EQUITY", "chimera/demo/telemetry.py")
    assert writer_references("set_demo_state", "chimera/demo/telemetry.py")
    assert writer_references("RECORDER_HEARTBEAT", "chimera/recorder/health.py")
    assert not writer_references("MODE_SELECTED", "chimera/demo/telemetry.py")


def test_the_writer_check_is_not_satisfied_by_a_name_that_only_looks_used():
    """The three ways a series can appear in a module without being written.

    Each of these defeated an earlier text-matching version of the check, so
    they are pinned as cases rather than left to a reviewer's memory.
    """
    imported_only = "from chimera.metrics import DEMO_EQUITY\n\nX = 1\n"
    commented_out = "from chimera.metrics import DEMO_EQUITY\n\n# DEMO_EQUITY.set(1.0)\npass\n"
    named_in_a_string = (
        'NAMES = ["DEMO_EQUITY"]\nHELP = "DEMO_EQUITY.set() used to live here"\n'
    )
    for source in (imported_only, commented_out, named_in_a_string):
        assert not _references(source, "DEMO_EQUITY"), source
    assert _references("DEMO_EQUITY.set(1.0)\n", "DEMO_EQUITY")


def test_the_demo_alerts_are_exactly_section_11_2s_ten():
    names = sorted(rule["alert"] for rule in demo_alert_rules())
    assert names == sorted(DEMO_ALERT_NAMES)
    assert len(names) == len(set(names)), "two alerts share a name"


def test_every_demo_alert_carries_a_severity_and_both_annotations():
    """An alert with no severity does not route, and one with no description
    tells the operator on call nothing about what to do next."""
    for rule in demo_alert_rules():
        assert "for" in rule, f"{rule['alert']} has no `for` clause"
        assert rule["labels"]["severity"] in {"critical", "warning"}
        annotations = rule["annotations"]
        assert annotations["summary"].strip(), f"{rule['alert']} has an empty summary"
        assert annotations["description"].strip(), f"{rule['alert']} has no description"


def test_every_demo_alert_series_is_written_by_a_named_module():
    referenced = set()
    for rule in demo_alert_rules():
        referenced |= {
            name for name in referenced_metrics(rule["expr"]) if name.startswith("chimera_")
        }
    assert referenced == set(DEMO_ALERT_SERIES), "the alert file's series set moved"
    for series, (attribute, module_path) in DEMO_ALERT_SERIES.items():
        assert writer_references(
            attribute, module_path
        ), f"{series} is declared but {module_path} never writes it"


@pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE, reason="prometheus_client is not installed"
)
def test_every_demo_alert_series_is_also_declared_in_metrics():
    exported = exported_metric_names()
    for series in DEMO_ALERT_SERIES:
        assert (
            series in exported
        ), f"{series} is written but chimera/metrics.py declares no such name"


def test_exactly_one_provisioned_dashboard_covers_the_demo_campaign():
    """ "Exactly one" is a property of the provisioned SET, not of a filename.

    The dashboard that covers the campaign is identified by what it queries --
    any ``chimera_demo_*`` series -- so renaming demo.json keeps this passing
    while adding a second demo dashboard, or copying this one, fails it. That is
    the invariant worth holding: Grafana provisions the whole directory, so a
    second board is a second board however it is named, and two boards sharing
    a uid means Grafana loads one of them and silently drops the other.
    """
    demo = [
        (path, board)
        for path, board in dashboards()
        if any("chimera_demo_" in expr for expr in panel_expressions(board))
    ]
    assert len(demo) == 1, f"expected one demo dashboard, found {[p.name for p, _ in demo]}"

    path, board = demo[0]
    assert board["uid"] == "chimera-demo"
    assert board["title"] == "Chimera / Demo campaign"
    assert [panel["title"] for panel in board["panels"]] == list(DEMO_PANEL_TITLES)
    assert all(
        "panels" not in panel for panel in board["panels"]
    ), "a row panel hides its children from the exported-metric check"

    uids = [board["uid"] for _, board in dashboards()]
    assert len(uids) == len(set(uids)), f"two dashboards share a uid: {uids}"


def test_the_provisioned_dashboard_directory_is_the_one_in_this_repository():
    """Makes "provisioned" mean something: the provider must point at the
    directory the compose mount puts these files in, or the count above is a
    count of files nothing loads."""
    provider = yaml.safe_load((DASHBOARD_DIR / "dashboards.yml").read_text(encoding="utf-8"))[
        "providers"
    ][0]
    assert provider["options"]["path"] == "/etc/grafana/provisioning/dashboards"
    mounts = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))[
        "services"
    ]["grafana"]["volumes"]
    assert "./grafana/provisioning:/etc/grafana/provisioning:ro" in mounts


def test_every_demo_dashboard_series_is_written_by_a_named_module():
    board = json.loads((DASHBOARD_DIR / "demo.json").read_text(encoding="utf-8"))
    referenced = set()
    for expr in panel_expressions(board):
        referenced |= {
            name for name in referenced_metrics(expr) if name.startswith("chimera_")
        }
    assert referenced == set(DEMO_DASHBOARD_SERIES), "the dashboard's series set moved"
    for series, (attribute, module_path) in DEMO_DASHBOARD_SERIES.items():
        assert writer_references(
            attribute, module_path
        ), f"panel series {series} is declared but {module_path} never writes it"


def test_every_demo_panel_asks_a_question():
    """No empty panel, and no panel whose targets were left behind by an edit."""
    board = json.loads((DASHBOARD_DIR / "demo.json").read_text(encoding="utf-8"))
    for panel in board["panels"]:
        assert panel["targets"], f"panel '{panel['title']}' has no query"
        assert panel["datasource"] == {"type": "prometheus", "uid": "prometheus"}
        refs = [target["refId"] for target in panel["targets"]]
        assert len(refs) == len(set(refs)), f"panel '{panel['title']}' repeats a refId"


def test_the_demo_profile_scrapes_the_adopted_ports():
    prometheus = yaml.safe_load((CONF_DIR / "prometheus.yml").read_text(encoding="utf-8"))
    targets = {
        job["job_name"]: job["static_configs"][0]["targets"][0]
        for job in prometheus["scrape_configs"]
    }
    for job_name, endpoint in DEMO_SCRAPE_TARGETS.items():
        assert targets.get(job_name) == endpoint
    assert "/etc/prometheus/alerts_demo.yml" in prometheus["rule_files"]
    # PR-12 is additive: taking the legacy jobs off the runtime is the
    # disconnect change's, and removing them here first would leave
    # conf/alerts.yml's InferenceServiceDown pointing at a job nothing scrapes.
    assert "nn_infer" in targets and "freqtrade" in targets


def test_every_loaded_rule_file_is_mounted_into_prometheus():
    """A rule file listed but not mounted stops Prometheus from starting at all."""
    prometheus = yaml.safe_load((CONF_DIR / "prometheus.yml").read_text(encoding="utf-8"))
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    mounted = {
        mount.split(":")[1]
        for mount in compose["services"]["prometheus"]["volumes"]
        if ":" in mount
    }
    for rule_file in prometheus["rule_files"]:
        assert rule_file in mounted, f"{rule_file} is loaded but never mounted"


def test_the_demo_services_carry_no_credential_and_no_live_flag():
    """The two new containers must be unable to trade, not merely configured not to."""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert set(compose["services"]) == set(COMPOSE_SERVICES)
    for name in ("recorder", "demo"):
        service = compose["services"][name]
        assert "env_file" not in service, f"{name} takes an env file"
        assert "environment" not in service, f"{name} takes an environment block"
        # No profile: these two ARE the demo deployment, so they are what a bare
        # `docker compose up` starts. Behind a profile they would be absent from
        # the default topology while conf/prometheus.yml scraped them anyway,
        # which is the permanently-down job the header of that file exists to
        # forbid. The disconnect change relies on this too: it profiles the
        # retired services out and states that a demo-path service carries no
        # profile, and this assertion is what keeps that statement true.
        assert "profiles" not in service, f"{name} is behind a compose profile"
        # `volumes` too, and this is the gap the first version left: the most
        # direct way to hand a container a credential is to bind-mount the host
        # path that holds one. `./.env`, `./user_data` (which is where the kill
        # switch and any operator secret live) and any absolute host path are
        # refused; the named volumes and the read-only `./conf` mount are what
        # these two services legitimately need.
        for mount in service.get("volumes", []):
            source = mount.split(":")[0]
            assert not source.startswith("/"), f"{name} bind-mounts host path {source}"
            assert source in ALLOWED_BIND_SOURCES, f"{name} mounts {source}"
    # freqtrade is the one service the live flag belongs to, and it sets it
    # blank. Any other service naming it would be a second way to reach a live
    # venue from this file, which is the thing that must not appear.
    for name, service in compose["services"].items():
        if name == "freqtrade":
            continue
        assert "ENABLE_LIVE_TRADING" not in yaml.safe_dump(
            service
        ), f"{name} names the live flag"


def test_every_demo_alert_expression_is_the_one_that_was_reviewed():
    """Matchers, thresholds and comparison directions, pinned literally.

    Everything else in this file checks that an alert names a series something
    exports. None of it can tell an armed alert from a disarmed one.
    """
    actual = {rule["alert"]: (rule["expr"], rule["for"]) for rule in demo_alert_rules()}
    assert actual == DEMO_ALERT_EXPRESSIONS


def test_every_demo_alert_label_value_is_one_the_code_can_produce():
    """A matcher on a label value nothing emits is an alert that cannot fire.

    Read out of the code and the committed contract rather than restated, so a
    renamed state or a changed stream set fails here instead of going quiet.
    """
    from chimera.demo.runner import RunnerState
    from chimera.recorder.contract import load_recorder_contract

    states = {state.value for state in RunnerState}
    streams = set(load_recorder_contract("btcusdt-prospective-gen3").streams)

    text = (CONF_DIR / "alerts_demo.yml").read_text(encoding="utf-8")
    for value in re.findall(r'state="([^"]+)"', text):
        assert value in states, f"no RunnerState is {value!r}"
    for matcher in re.findall(r'stream=~"([^"]+)"', text):
        for value in matcher.split("|"):
            assert value in streams, f"{value!r} is not a gen3 contract stream"
    for value in re.findall(r'job="([^"]+)"', text):
        prometheus = yaml.safe_load((CONF_DIR / "prometheus.yml").read_text(encoding="utf-8"))
        jobs = {job["job_name"] for job in prometheus["scrape_configs"]}
        assert value in jobs, f"no prometheus job is named {value!r}"


def test_the_label_value_guard_catches_a_state_nothing_can_emit():
    from chimera.demo.runner import RunnerState

    assert "HALTED" not in {state.value for state in RunnerState}
    assert "HALT" in {state.value for state in RunnerState}


def test_every_demo_panel_query_is_the_one_that_was_reviewed():
    """Panel semantics, not just panel metric names.

    A panel that aggregates away the label a reader needs -- `direction` on the
    funding panel above all, which amendment A10 exists to keep apart -- still
    names an exported series.
    """
    board = json.loads((DASHBOARD_DIR / "demo.json").read_text(encoding="utf-8"))
    actual = {
        panel["title"]: tuple(target["expr"] for target in panel["targets"])
        for panel in board["panels"]
    }
    assert actual == DEMO_PANEL_EXPRESSIONS


def test_the_shared_risk_series_are_scoped_to_the_demo_job():
    """chimera_risk_halted and chimera_rejected_entries_total have two writers.

    The retired Freqtrade path writes both, and until it is deleted a demo alert
    or panel that selects them unqualified reads whichever container answered.
    """
    board = json.loads((DASHBOARD_DIR / "demo.json").read_text(encoding="utf-8"))
    expressions = [(rule["alert"], rule["expr"]) for rule in demo_alert_rules()]
    expressions += [
        (panel["title"], target["expr"])
        for panel in board["panels"]
        for target in panel["targets"]
    ]
    # Only the queries, never the surrounding prose: the rule file's header
    # names chimera_risk_halted while explaining exactly this, and a raw text
    # scan would read that sentence as an unqualified selector.
    for series in ("chimera_risk_halted", "chimera_rejected_entries_total"):
        for where, expr in expressions:
            for occurrence in re.findall(re.escape(series) + r"(\{[^}]*\})?", expr):
                assert "job=" in (occurrence or ""), (
                    f"{where} selects {series} without a job matcher; the retired "
                    "Freqtrade container writes the same series"
                )


def test_no_demo_deployment_artifact_can_be_given_a_credential():
    """The compose guard above reads one file; this reads the other four.

    A unit file and a Dockerfile are deployment surface exactly as compose is,
    and both were unguarded: `EnvironmentFile=-/etc/chimera/exchange.env` plus
    `Environment=ENABLE_LIVE_TRADING=I_UNDERSTAND_THE_RISK` in the unit, and
    `ENV BINANCE_KEY=... ENABLE_LIVE_TRADING=...` in the image, both shipped
    green. The demo path has no authenticated endpoint to reach, so there is
    nothing here that a credential could legitimately be for.
    """
    for relative in DEPLOYMENT_ARTIFACTS:
        path = ROOT / relative
        assert path.is_file(), f"{relative} is missing"
        # Comments are where these tokens are legitimately discussed -- the unit
        # explains at length why it carries no EnvironmentFile -- so the scan is
        # over directives only.
        directives = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for token in FORBIDDEN_DEPLOYMENT_TOKENS:
            offenders = [line for line in directives if token in line]
            assert not offenders, f"{relative} names {token}: {offenders}"


def test_every_demo_image_runs_as_a_named_non_root_user():
    """A root container that also mounts the recorder's data is a wider blast
    radius than anything the demo path needs."""
    for relative in ("deploy/docker/Dockerfile.demo", "deploy/docker/Dockerfile.recorder"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        users = re.findall(r"^USER\s+(\S+)", text, re.MULTILINE)
        assert users, f"{relative} never drops privileges"
        assert users[-1] not in {"root", "0"}, f"{relative} ends as {users[-1]}"


def test_no_demo_image_copies_the_retired_live_capable_launcher():
    """`tools/run_bot.py` is the only live-capable entrypoint in the tree.

    `COPY tools/ ./tools/` put it inside both dry-run images. An image that
    cannot import it cannot be argued into running it.
    """
    for relative in ("deploy/docker/Dockerfile.demo", "deploy/docker/Dockerfile.recorder"):
        copied = re.findall(
            r"^COPY\s+(\S+)", (ROOT / relative).read_text(encoding="utf-8"), re.MULTILINE
        )
        assert "tools/" not in copied, f"{relative} copies the whole tools tree"
        assert "tools/run_bot.py" not in copied, f"{relative} copies the launcher"
        assert "strategies/" not in copied, f"{relative} copies the retired strategies"
