"""The P4 derivatives snapshot: what it must be, and what the verifier must refuse.

Every rejection below is a claim about a *file*, and the public Binance archive
cannot be asked to serve a corrupt one — so the fixture writes a well-formed
snapshot through the real exporter and then breaks exactly one thing about it.
That is the same shape ``tests/test_trade_snapshot.py`` uses for P3, and it is the
only way the fail-closed reader is exercised at all.

The spine underneath is the **committed** one, so the boundary tests are about
the geometry P4 will actually run on: the snapshot stops at row 45802's hour, and
P4-HOLD begins at the next.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nn.derivatives_sources import HOUR_NS, UNAVAILABLE_AGE_NS, staleness_bound_ns
from nn.p4_holdout import HoldoutError, check_holdout_boundary, holdout_first_instant
from nn.p4_preregistration import HOLDOUT_ROWS, preregistration_hash
from nn.research_contract import load_contract
from tools.export_derivatives_snapshot import (
    DERIVATIVES_SNAPSHOT_SCHEMA,
    MANIFEST_NAME,
    DerivativesExportError,
    acquisition_window,
    check_hourly_table,
    describe_plan,
    plan,
    source_spec_hash,
    write_snapshot,
)
from tools.verify_derivatives_snapshot import (
    DerivativesSnapshotVerificationError,
    cross_source_consistency,
    verify_derivatives_snapshot,
)

CONTRACT = load_contract("btc-usdt-1h-gen1")

FIXTURE_ARCHIVES = [
    {
        "field": "funding_rate",
        "name": "SYNTHETIC-FIXTURE-fundingRate.zip",
        "url": "fixture://synthetic",
        "kind": "monthly",
        "period_start": "2019-12-01T00:00:00+00:00",
        "period_end": "2020-01-01T00:00:00+00:00",
        "bytes": 1,
        "sha256": "0" * 64,
        "published_sha256": None,
        "checksum_verified": False,
        "checksum_mismatch": False,
        "layout": {"layout": "fundingTime/fundingRate"},
    }
]


@pytest.fixture(scope="module")
def exported(p4_tree, p4_spine, p4_hourly):
    """A snapshot written by the real exporter, into a copy of the research tree."""
    start, end, alignment = acquisition_window(p4_spine["manifest"], CONTRACT)
    manifest = write_snapshot(
        p4_hourly,
        FIXTURE_ARCHIVES,
        [],
        contract=CONTRACT,
        out_dir=p4_tree / "data" / "research",
        ohlcv_manifest=p4_spine["manifest"],
        alignment_source=alignment,
        requested_from=start,
        requested_through=end,
        acquired_at="1970-01-01T00:00:00Z",
    )
    return {
        "root": p4_tree,
        "path": p4_tree / "data" / "research" / MANIFEST_NAME,
        "payload": manifest,
    }


def _repoint(exported, tmp_path, mutate):
    """A copy of the snapshot with one thing changed, for the verifier to refuse."""
    import shutil

    root = tmp_path / "tree"
    shutil.copytree(exported["root"], root)
    path = root / "data" / "research" / MANIFEST_NAME
    payload = json.loads(path.read_text())
    mutate(payload, root)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# --- the plan is networkless and bounded before both regions ------------------
def test_the_plan_touches_no_network_and_stops_before_styx_and_the_holdout(p4_spine):
    start, end, alignment = acquisition_window(p4_spine["manifest"], CONTRACT)
    described = describe_plan(plan(start, end), start, end, CONTRACT, alignment)
    assert described["network_accessed"] is False
    assert described["bounded_before_styx"] is True
    assert described["bounded_before_p4_hold"] is True
    assert described["preregistration_hash"] == preregistration_hash()


def test_the_plan_requests_the_daily_metrics_archive_and_no_monthly_one(p4_spine):
    start, end, _ = acquisition_window(p4_spine["manifest"], CONTRACT)
    fields = plan(start, end)
    assert all(a.kind == "daily" for a in fields["open_interest"])
    assert all("/daily/metrics/" in a.url for a in fields["open_interest"])


def test_the_acquisition_window_ends_exactly_at_the_first_holdout_instant(p4_spine):
    _, end, _ = acquisition_window(p4_spine["manifest"], CONTRACT)
    assert end == pd.Timestamp(holdout_first_instant()).tz_convert("UTC")


def test_the_ledgers_holdout_instant_is_the_committed_spines_own_next_hour(p4_spine):
    boundary = check_holdout_boundary(p4_spine["manifest"])
    assert boundary["p4_hold_first_instant"] == "2025-05-19T08:00:00+00:00"
    assert boundary["stage_1_last_instant"] == "2025-05-19T07:00:00+00:00"


def test_a_snapshot_cut_one_block_longer_is_refused_before_a_window_is_planned(
    p4_spine, tmp_path
):
    """The one-row overrun, at the level the acquisition can still stop it."""
    payload = json.loads(p4_spine["manifest"].read_text())
    payload["processed_outer_coverage"]["row_range"] = [0, HOLDOUT_ROWS[0] + 1]
    longer = tmp_path / "manifest.json"
    longer.write_text(json.dumps(payload))
    with pytest.raises(HoldoutError, match=f"past {HOLDOUT_ROWS[0]}"):
        acquisition_window(longer, CONTRACT)


# --- the exported snapshot ----------------------------------------------------
def test_the_exported_snapshot_verifies_in_full(exported):
    checks = verify_derivatives_snapshot(exported["path"])
    assert len(checks) >= 12
    names = {check.name for check in checks}
    assert {"schema", "preregistration", "semantic hash", "styx", "p4-hold"} <= names


def test_the_snapshot_stops_before_the_first_holdout_hour(exported):
    payload = exported["payload"]
    assert payload["contains_p4_hold"] is False
    assert payload["safety_checks"]["p4_hold_rows_exported"] == 0
    assert pd.Timestamp(payload["hourly"]["end"]) < pd.Timestamp(
        holdout_first_instant()
    ).tz_convert("UTC")
    assert payload["hourly"]["end"] == "2025-05-19T07:00:00+00:00"


def test_the_snapshot_declares_no_rest_row_and_the_verifier_checks_it(exported, tmp_path):
    assert exported["payload"]["safety_checks"]["rest_rows_exported"] == 0
    path = _repoint(
        exported, tmp_path, lambda p, _: p["safety_checks"].update(rest_rows_exported=1)
    )
    with pytest.raises(DerivativesSnapshotVerificationError, match="no REST row"):
        verify_derivatives_snapshot(path)


def test_no_module_on_the_p4_path_can_build_a_rest_request():
    """§3.0a forbids REST standing in for an archive day; there is no path to it.

    Checked with the parser rather than with a grep, and against the *fetchable*
    form of the endpoint rather than its name. The name appears all over this
    code — in the prohibition itself, and in the manifest field that records why
    no REST row is present — and a test that banned the name would ban the
    documentation. What must not exist is a string an expression could turn into
    a request: a scheme plus the host, outside a docstring.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in (
        "nn/derivatives.py",
        "nn/derivatives_sources.py",
        "nn/p4_universe.py",
        "nn/p4_stage1.py",
        "nn/p4_holdout.py",
        "tools/export_derivatives_snapshot.py",
        "tools/verify_derivatives_snapshot.py",
    ):
        tree = ast.parse((root / name).read_text())
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                lowered = node.value.lower()
                assert "https://fapi" not in lowered, f"{name}:{node.lineno}"
                assert "http://fapi" not in lowered, f"{name}:{node.lineno}"


def test_every_url_the_acquisition_builds_points_at_the_public_archive():
    """The three preregistered paths, and no fourth host to substitute from."""
    from nn.derivatives_sources import (
        BASE_URL,
        FUNDING_TEMPLATE,
        KLINE_TEMPLATE,
        METRICS_TEMPLATE,
    )

    assert BASE_URL == "https://data.binance.vision"
    for template in (FUNDING_TEMPLATE, METRICS_TEMPLATE, KLINE_TEMPLATE):
        assert template.startswith("{base}/data/futures/um/")
    assert "/daily/metrics/" in METRICS_TEMPLATE
    assert "/monthly/fundingRate/" in FUNDING_TEMPLATE
    assert "/monthly/klines/" in KLINE_TEMPLATE


def test_the_manifest_records_the_preregistration_the_source_was_acquired_under(exported):
    assert exported["payload"]["preregistration_hash"] == preregistration_hash()
    assert exported["payload"]["snapshot_schema"] == DERIVATIVES_SNAPSHOT_SCHEMA
    assert exported["payload"]["hourly"]["source_spec_hash"] == source_spec_hash()


def test_a_snapshot_under_another_preregistration_is_refused(exported, tmp_path):
    path = _repoint(exported, tmp_path, lambda p, _: p.update(preregistration_hash="0" * 64))
    with pytest.raises(DerivativesSnapshotVerificationError, match="under preregistration"):
        verify_derivatives_snapshot(path)


def test_a_snapshot_under_another_source_spec_is_refused(exported, tmp_path):
    """An unknown identity scheme fails closed rather than being compared across."""
    path = _repoint(
        exported, tmp_path, lambda p, _: p["hourly"].update(source_spec_hash="1" * 64)
    )
    with pytest.raises(DerivativesSnapshotVerificationError, match="Two staleness rules"):
        verify_derivatives_snapshot(path)


def test_a_snapshot_under_another_schema_is_refused(exported, tmp_path):
    path = _repoint(exported, tmp_path, lambda p, _: p.update(snapshot_schema="other/1"))
    with pytest.raises(DerivativesSnapshotVerificationError, match="declares schema"):
        verify_derivatives_snapshot(path)


def test_an_edited_source_file_moves_the_digest_and_is_refused(exported, tmp_path):
    def edit(payload, root):
        path = root / payload["hourly"]["path"]
        frame = pd.read_parquet(path)
        frame.loc[frame.index[10_000], "perp_close"] *= 1.0001
        frame.to_parquet(path, index=False)

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(DerivativesSnapshotVerificationError, match="hashes to"):
        verify_derivatives_snapshot(path)


def test_an_edited_source_file_also_moves_the_semantic_hash(exported, tmp_path):
    """Falsification of the semantic identity, not just of the byte digest."""
    import hashlib

    def edit(payload, root):
        path = root / payload["hourly"]["path"]
        frame = pd.read_parquet(path)
        frame.loc[frame.index[10_000], "oi_contracts"] *= 1.05
        frame.to_parquet(path, index=False)
        payload["hourly"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(DerivativesSnapshotVerificationError, match="semantic hash"):
        verify_derivatives_snapshot(path)


def test_a_manifest_that_overstates_its_coverage_is_refused(exported, tmp_path):
    def edit(payload, _root):
        payload["coverage"]["open_interest"]["hours_available"] += 1

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(DerivativesSnapshotVerificationError, match="hours_available"):
        verify_derivatives_snapshot(path)


def test_a_missing_day_recorded_for_an_unclassified_reason_is_refused(exported, tmp_path):
    """§3.0a names three reasons. Anything else must have stopped the acquisition."""

    def edit(payload, _root):
        payload["coverage"]["missing_metrics_days"] = [
            {"day": "2023-01-01", "reason": "network_error"}
        ]
        payload["acquisition"]["missing_metrics_days"] = 1

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(
        DerivativesSnapshotVerificationError, match="recorded missing for reason"
    ):
        verify_derivatives_snapshot(path)


def test_a_manifest_that_miscounts_its_missing_days_is_refused(exported, tmp_path):
    def edit(payload, _root):
        payload["acquisition"]["missing_metrics_days"] = 3

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(DerivativesSnapshotVerificationError, match="missing metrics day"):
        verify_derivatives_snapshot(path)


def test_a_snapshot_bound_to_other_candles_is_refused(exported, tmp_path):
    def edit(payload, _root):
        payload["alignment"]["ohlcv_processed_semantic_prefix_hash"] = "9" * 64

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(DerivativesSnapshotVerificationError, match="different OHLCV"):
        verify_derivatives_snapshot(path)


# --- the exporter's own structural checks -------------------------------------
def test_a_table_with_a_hole_in_its_hourly_grid_is_refused(p4_hourly):
    holed = p4_hourly.drop(index=p4_hourly.index[5_000]).reset_index(drop=True)
    with pytest.raises(DerivativesExportError, match="has a hole"):
        check_hourly_table(holed)


def test_an_age_past_the_staleness_bound_is_refused(p4_hourly):
    broken = p4_hourly.copy()
    # A row the source says it *has* an observation for: an age past the bound on
    # a row already marked unavailable is a different refusal.
    row = broken.index[broken["oi_available"].to_numpy() == 1][100]
    broken.loc[row, "oi_age_ns"] = staleness_bound_ns("open_interest") + HOUR_NS
    with pytest.raises(DerivativesExportError, match="staleness bound"):
        check_hourly_table(broken)


def test_an_unavailable_hour_carrying_an_age_is_refused(p4_hourly):
    broken = p4_hourly.copy()
    row = broken.index[5_000]
    broken.loc[row, "oi_available"] = 0
    broken.loc[row, "oi_age_ns"] = 0
    with pytest.raises(DerivativesExportError, match="has no observation"):
        check_hourly_table(broken)


def test_an_available_hour_with_no_age_is_refused(p4_hourly):
    broken = p4_hourly.copy()
    row = broken.index[5_000]
    broken.loc[row, "perp_available"] = 1
    broken.loc[row, "perp_age_ns"] = UNAVAILABLE_AGE_NS
    with pytest.raises(DerivativesExportError, match="negative on an available hour"):
        check_hourly_table(broken)


def test_a_non_positive_perpetual_close_is_refused(p4_hourly):
    broken = p4_hourly.copy()
    broken.loc[broken.index[5_000], "perp_close"] = 0.0
    with pytest.raises(DerivativesExportError, match="non-positive on an available hour"):
        check_hourly_table(broken)


# --- §3.5's value-level cross-source check ------------------------------------
def test_the_cross_source_check_passes_on_a_basis_inside_its_own_clip(p4_hourly, p4_spine):
    result = cross_source_consistency(p4_hourly, p4_spine["raw"])
    assert result["hours_compared"] > 40_000
    assert result["worst_relative_deviation"] < result["tolerance"]


def test_a_perpetual_series_at_the_wrong_scale_is_caught(p4_hourly, p4_spine):
    broken = p4_hourly.copy()
    broken["perp_close"] = broken["perp_close"] * 10.0
    with pytest.raises(DerivativesSnapshotVerificationError, match="not a cost of"):
        cross_source_consistency(broken, p4_spine["raw"])
