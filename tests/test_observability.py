"""Every dashboard panel and alert must reference a metric this code exports.

The previous repository had five dashboards querying ten series (`equity`,
`drift_score`, `sharpe_live`, `retrain_trigger`, ...), not one of which was
produced by any code in it. These tests make that impossible to reintroduce
silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from chimera import metrics

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "grafana" / "provisioning" / "dashboards"
CONF_DIR = ROOT / "conf"

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
        "chimera_data_staleness_seconds",
    ):
        assert expected in names, f"{expected} is not exported"


@pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE, reason="prometheus_client is not installed"
)
@pytest.mark.parametrize("path", sorted(DASHBOARD_DIR.glob("*.json")))
def test_every_dashboard_panel_queries_an_exported_metric(path):
    exported = exported_metric_names()
    dashboard = json.loads(path.read_text())
    assert dashboard["panels"], f"{path.name} has no panels"

    for panel in dashboard["panels"]:
        assert panel.get("targets"), f"{path.name}: panel '{panel['title']}' has no query"
        for target in panel["targets"]:
            for name in referenced_metrics(target["expr"]):
                assert name in exported, (
                    f"{path.name}: panel '{panel['title']}' queries {name}, "
                    "which nothing in this repository exports"
                )


@pytest.mark.skipif(
    not metrics.PROMETHEUS_AVAILABLE, reason="prometheus_client is not installed"
)
def test_every_alert_rule_queries_an_exported_metric():
    exported = exported_metric_names()
    rules = yaml.safe_load((CONF_DIR / "alerts.yml").read_text())
    for group in rules["groups"]:
        for rule in group["rules"]:
            for name in referenced_metrics(rule["expr"]):
                assert name in exported, f"alert {rule['alert']} queries unknown {name}"


def test_prometheus_scrapes_only_services_that_exist():
    prometheus = yaml.safe_load((CONF_DIR / "prometheus.yml").read_text())
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = set(compose["services"])

    for job in prometheus["scrape_configs"]:
        for static in job["static_configs"]:
            for target in static["targets"]:
                host = target.split(":")[0]
                if host == "localhost":
                    continue
                assert (
                    host in services
                ), f"prometheus scrapes '{host}', which is not a compose service"


def test_alertmanager_config_contains_no_placeholder_webhook():
    """A placeholder URL silently swallows every critical alert."""
    text = (CONF_DIR / "alertmanager.yml").read_text()
    assert "example.com" not in text


def test_grafana_datasource_is_provisioned():
    datasource = yaml.safe_load(
        (ROOT / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text()
    )
    assert datasource["datasources"][0]["url"].startswith("http://prometheus:")


def test_dashboards_are_mounted_into_grafana():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
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
