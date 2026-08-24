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

from nn.derivatives_sources import (
    HOUR_NS,
    UNAVAILABLE_AGE_NS,
    funding_archive,
    kline_archive,
    staleness_bound_ns,
)
from nn.p4_holdout import HoldoutError, check_holdout_boundary, holdout_first_instant
from nn.p4_preregistration import (
    FUNDING_ARCHIVE_INCEPTION_POLICY,
    HOLDOUT_ROWS,
    PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY,
    preregistration_hash,
)
from nn.research_contract import load_contract
from tools.export_derivatives_snapshot import (
    DERIVATIVES_SNAPSHOT_SCHEMA,
    MANIFEST_NAME,
    DerivativesExportError,
    acquisition_window,
    check_hourly_table,
    acquire,
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
    },
    # The 2020-09-01 shape: 576 raw rows, 288 logical observations, every
    # instant duplicated once with the paired rows identical across the whole
    # CSV row. Present so the manifest's source-integrity accounting is written
    # from something other than zeros and the verifier has arithmetic to redo.
    {
        "field": "open_interest",
        "name": "SYNTHETIC-FIXTURE-metrics-2020-09-01.zip",
        "url": "fixture://synthetic",
        "kind": "daily",
        "period_start": "2020-09-01T00:00:00+00:00",
        "period_end": "2020-09-02T00:00:00+00:00",
        "bytes": 1,
        "sha256": "0" * 64,
        "published_sha256": None,
        "checksum_verified": False,
        "checksum_mismatch": False,
        "rows": 284,
        "normalisation": {
            "rows_read": 576,
            "observations_retained": 288,
            "exact_duplicate_rows_collapsed": 288,
            "duplicate_instants": 288,
        },
        # A4 runs on what A1 left behind: 288 logical rows, four of which carry a
        # zero in a consumed field and are rejected as observations. Non-zero on
        # purpose, so the manifest's validity accounting and the verifier's
        # arithmetic are written from something other than a row of zeros.
        "validity": {
            "logical_observations": 288,
            "valid_positive_observations": 284,
            "invalid_zero_observations": 4,
            "invalid_both_zero_observations": 2,
            "invalid_zero_contracts_only": 1,
            "invalid_zero_notional_only": 1,
            "negative_observations": 0,
            "nonfinite_observations": 0,
        },
    },
    {
        "field": "open_interest",
        "name": "SYNTHETIC-FIXTURE-metrics-2020-09-03.zip",
        "url": "fixture://synthetic",
        "kind": "daily",
        "period_start": "2020-09-03T00:00:00+00:00",
        "period_end": "2020-09-04T00:00:00+00:00",
        "bytes": 1,
        "sha256": "0" * 64,
        "published_sha256": None,
        "checksum_verified": False,
        "checksum_mismatch": False,
        "rows": 288,
        "normalisation": {
            "rows_read": 288,
            "observations_retained": 288,
            "exact_duplicate_rows_collapsed": 0,
            "duplicate_instants": 0,
        },
        "validity": {
            "logical_observations": 288,
            "valid_positive_observations": 288,
            "invalid_zero_observations": 0,
            "invalid_both_zero_observations": 0,
            "invalid_zero_contracts_only": 0,
            "invalid_zero_notional_only": 0,
            "negative_observations": 0,
            "nonfinite_observations": 0,
        },
    },
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


# --- amendment A2: the funding archive's inception, in the real plan ----------
def test_the_real_plan_asks_for_the_warmup_month_and_gets_the_inception_month(p4_spine):
    """The committed spine's own numbers, which is where §3.4b came from.

    The generic warm-up planner asks for 2019-12; the published archive begins at
    2020-01. Both are in the plan's account of itself, so the clamped month is
    visibly clamped rather than quietly absent.
    """
    start, end, alignment = acquisition_window(p4_spine["manifest"], CONTRACT)
    assert start == pd.Timestamp("2019-12-01", tz="UTC")
    described = describe_plan(plan(start, end), start, end, CONTRACT, alignment)

    boundary = described["funding_source_boundary"]
    assert boundary["amendment"] == "A2"
    assert boundary["generic_requested_from"] == "2019-12"
    assert boundary["source_inception_month"] == "2020-01"
    assert boundary["effective_from"] == "2020-01"
    assert boundary["months_clamped"] == 1
    assert boundary["not_an_internal_gap"] is True

    funding = described["fields"]["funding_rate"]
    assert funding["period_from"] == "2020-01-01T00:00:00+00:00"
    assert "BTCUSDT-fundingRate-2020-01.zip" in funding["first"]
    assert described["network_accessed"] is False


def test_the_generic_window_and_warmup_are_not_moved_by_the_clamp(p4_spine):
    """§3.4b clamps the funding iterator and nothing else.

    The research spine boundary and the requested feature warm-up are unchanged,
    and the other two fields plan from the same start they always did.
    """
    start, end, alignment = acquisition_window(p4_spine["manifest"], CONTRACT)
    described = describe_plan(plan(start, end), start, end, CONTRACT, alignment)
    assert described["window"]["from"] == "2019-12-01T00:00:00+00:00"
    assert described["spine"]["warmup_hours_requested"] == 822
    # The perpetual plan is now clamped too — by amendment A3 (§3.4c), on its own
    # archive and its own evidence, not by A2 reaching further than it said.
    assert described["fields"]["perpetual_price"]["period_from"] == "2020-01-01T00:00:00+00:00"
    # Open interest keeps its own §3.0a first day; neither amendment touched it.
    assert described["earliest_intended_metrics_day"] == "2020-09-01"


def test_the_pre_inception_month_is_never_requested_and_so_is_never_a_missing_month(p4_spine):
    """The two halves of §3.4b's distinction, in one place.

    An unplanned month cannot be fetched, so it cannot reach the fail-closed rule
    that stops the acquisition on an absent funding archive. That is what "outside
    the source, not an internal gap" means operationally.
    """
    start, end, _ = acquisition_window(p4_spine["manifest"], CONTRACT)
    names = [archive.name for archive in plan(start, end)["funding_rate"]]
    assert "BTCUSDT-fundingRate-2019-12.zip" not in names
    assert names[0] == "BTCUSDT-fundingRate-2020-01.zip"
    assert not any(name.startswith("BTCUSDT-fundingRate-2019") for name in names)


def test_a_missing_funding_month_at_the_inception_stops_the_acquisition(monkeypatch, tmp_path):
    """§3.4b relaxes nothing at or after 2020-01: the first month is mandatory."""
    import tools.export_derivatives_snapshot as exporter

    monkeypatch.setattr(exporter, "download_archive", lambda *a, **k: None)
    start = pd.Timestamp("2019-12-01", tz="UTC")
    end = pd.Timestamp("2020-04-01", tz="UTC")
    plan_by_field = plan(start, end, oi_first_day=pd.Timestamp("2020-09-01", tz="UTC"))
    with pytest.raises(DerivativesExportError, match="BTCUSDT-fundingRate-2020-01.zip"):
        acquire(plan_by_field, start=start, end=end, timeout=1, workdir=tmp_path)


@pytest.mark.parametrize("year,month", [(2020, 1), (2020, 2), (2022, 7), (2025, 5)])
def test_every_expected_funding_month_from_the_inception_onward_is_mandatory(
    monkeypatch, tmp_path, year, month
):
    """A gap *after* inception is still a hard failure, and never a skipped month."""
    import tools.export_derivatives_snapshot as exporter

    monkeypatch.setattr(exporter, "download_archive", lambda *a, **k: None)
    archive = funding_archive(year, month)
    assert archive.period_start >= pd.Timestamp(
        f"{FUNDING_ARCHIVE_INCEPTION_POLICY['first_protocol_month']}-01", tz="UTC"
    )
    with pytest.raises(DerivativesExportError, match="is not published"):
        exporter._require(archive, tmp_path, timeout=1)


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


def test_the_manifest_records_the_funding_source_boundary(exported):
    """§3.4b's provenance requirement: the clamped month does not disappear."""
    boundary = exported["payload"]["acquisition"]["funding_source_boundary"]
    for key in FUNDING_ARCHIVE_INCEPTION_POLICY["provenance_required"]:
        assert key in boundary, key
    assert boundary["generic_requested_from"] == "2019-12"
    assert boundary["effective_from"] == "2020-01"
    assert boundary["months_clamped"] == 1
    assert boundary["no_substitution"].startswith("never")
    # And the hashed rule itself travels with the table, not a paraphrase of it.
    assert exported["payload"]["source"]["funding_inception_rule"] == dict(
        FUNDING_ARCHIVE_INCEPTION_POLICY
    )


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


def test_the_manifest_accounts_for_every_exact_duplicate_row_it_collapsed(exported):
    """Collapsed rows leave a trace, per archive and in aggregate."""
    integrity = exported["payload"]["acquisition"]["open_interest_source_integrity"]
    assert integrity["archives_read"] == 2
    assert integrity["raw_rows_read"] == 576 + 288
    assert integrity["logical_observations_retained"] == 288 + 288
    assert integrity["exact_duplicate_rows_collapsed"] == 288
    assert integrity["duplicate_instants_collapsed"] == 288
    assert integrity["archives_with_exact_duplicates"] == 1
    assert integrity["conflicting_duplicate_instants"] == 0
    # The rule itself lives once — in the hashed preregistration, which the
    # source spec reads and the manifest copies — so the manifest's copy is the
    # preregistered policy rather than a paraphrase of it.
    from nn.p4_preregistration import OPEN_INTEREST_DUPLICATE_POLICY

    assert exported["payload"]["source"]["duplicate_rule"] == dict(
        OPEN_INTEREST_DUPLICATE_POLICY
    )


def test_the_raw_archive_identity_survives_the_collapse(exported):
    """Normalisation must not pretend the upstream bytes were different.

    The per-archive record still carries the ZIP's own sha256 and the raw row
    count it was read from, so the logical table and the bytes it came from stay
    separately identified. Three counts, three different statements: 576 rows
    published, 288 logical rows after A1 collapsed the identical repeats, and 284
    valid observations after A4 declined the four zero-valued ones.
    """
    records = [
        a
        for a in exported["payload"]["acquisition"]["archives"]
        if a["field"] == "open_interest"
    ]
    duplicated = next(a for a in records if a["normalisation"]["rows_read"] == 576)
    assert duplicated["sha256"], "the archive's own digest is still recorded"
    assert duplicated["normalisation"]["rows_read"] == 576
    assert duplicated["normalisation"]["observations_retained"] == 288
    assert duplicated["validity"]["logical_observations"] == 288
    assert duplicated["rows"] == 284 == duplicated["validity"]["valid_positive_observations"]


def test_a_manifest_whose_duplicate_aggregate_disagrees_with_its_records_is_refused(
    exported, tmp_path
):
    def edit(payload, _root):
        payload["acquisition"]["open_interest_source_integrity"][
            "exact_duplicate_rows_collapsed"
        ] = 0

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(
        DerivativesSnapshotVerificationError, match="exact_duplicate_rows_collapsed"
    ):
        verify_derivatives_snapshot(path)


def test_a_manifest_declaring_a_conflicting_duplicate_instant_is_refused(exported, tmp_path):
    """A conflict stops the acquisition, so no written snapshot may report one."""

    def edit(payload, _root):
        payload["acquisition"]["open_interest_source_integrity"][
            "conflicting_duplicate_instants"
        ] = 1

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(DerivativesSnapshotVerificationError, match="conflicting"):
        verify_derivatives_snapshot(path)


def test_a_per_archive_normalisation_that_does_not_add_up_is_refused(exported, tmp_path):
    def edit(payload, _root):
        for archive in payload["acquisition"]["archives"]:
            if archive.get("normalisation", {}).get("rows_read") == 576:
                archive["normalisation"]["observations_retained"] = 400

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(DerivativesSnapshotVerificationError, match="do not add up"):
        verify_derivatives_snapshot(path)


def test_a_manifest_claiming_rows_it_did_not_retain_is_refused(exported, tmp_path):
    def edit(payload, _root):
        for archive in payload["acquisition"]["archives"]:
            if "normalisation" in archive:
                archive["rows"] = 999

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(DerivativesSnapshotVerificationError, match="claims it contributed"):
        verify_derivatives_snapshot(path)


def test_rows_removed_without_a_duplicate_instant_behind_them_are_refused(exported, tmp_path):
    """Every collapsed row came from an instant; a count without one is nonsense."""

    def edit(payload, _root):
        for archive in payload["acquisition"]["archives"]:
            if archive.get("normalisation", {}).get("duplicate_instants") == 288:
                archive["normalisation"]["duplicate_instants"] = 0
                payload["acquisition"]["open_interest_source_integrity"][
                    "duplicate_instants_collapsed"
                ] = 0

    path = _repoint(exported, tmp_path, edit)
    with pytest.raises(DerivativesSnapshotVerificationError, match="at least one row"):
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


# --- amendment A3: the perpetual kline archive's inception, in the real plan ---
def test_the_real_plan_asks_for_the_warmup_kline_month_and_gets_the_inception_month(p4_spine):
    """The committed spine's own numbers, which is where §3.4c came from.

    The generic warm-up planner asks for 2019-12; the published kline archive
    begins at 2020-01. Both are in the plan's account of itself, so the clamped
    month is visibly clamped rather than quietly absent.
    """
    start, end, alignment = acquisition_window(p4_spine["manifest"], CONTRACT)
    assert start == pd.Timestamp("2019-12-01", tz="UTC")
    described = describe_plan(plan(start, end), start, end, CONTRACT, alignment)

    boundary = described["perpetual_source_boundary"]
    assert boundary["amendment"] == "A3"
    assert boundary["generic_requested_from"] == "2019-12"
    assert boundary["source_inception_month"] == "2020-01"
    assert boundary["effective_from"] == "2020-01"
    assert boundary["months_clamped"] == 1
    assert boundary["not_an_internal_gap"] is True

    perpetual = described["fields"]["perpetual_price"]
    assert perpetual["period_from"] == "2020-01-01T00:00:00+00:00"
    assert "BTCUSDT-1h-2020-01.zip" in perpetual["first"]
    assert described["network_accessed"] is False


def test_the_two_boundaries_are_reported_separately_in_the_same_plan(p4_spine):
    """A2 and A3 travel as two provenance objects, not one merged claim."""
    start, end, alignment = acquisition_window(p4_spine["manifest"], CONTRACT)
    described = describe_plan(plan(start, end), start, end, CONTRACT, alignment)
    assert described["funding_source_boundary"]["amendment"] == "A2"
    assert described["funding_source_boundary"]["field"] == "funding_rate"
    assert described["perpetual_source_boundary"]["amendment"] == "A3"
    assert described["perpetual_source_boundary"]["field"] == "perpetual_price"


def test_the_pre_inception_kline_month_is_never_requested_and_so_is_never_missing(p4_spine):
    """The two halves of §3.4c's distinction, in one place.

    An unplanned month cannot be fetched, so it cannot reach the fail-closed rule
    that stops the acquisition on an absent perpetual archive. That is what
    "outside the source, not an internal gap" means operationally.
    """
    start, end, _ = acquisition_window(p4_spine["manifest"], CONTRACT)
    names = [archive.name for archive in plan(start, end)["perpetual_price"]]
    assert "BTCUSDT-1h-2019-12.zip" not in names
    assert names[0] == "BTCUSDT-1h-2020-01.zip"
    assert not any(name.startswith("BTCUSDT-1h-2019") for name in names)
    # Every later expected month is there: the clamp removed the pre-inception
    # months and nothing else.
    assert names[1] == "BTCUSDT-1h-2020-02.zip"
    assert len(names) == len({*names})


def test_a_missing_kline_month_at_the_inception_stops_the_acquisition(monkeypatch, tmp_path):
    """§3.4c relaxes nothing at or after 2020-01: the first month is mandatory."""
    import tools.export_derivatives_snapshot as exporter

    monkeypatch.setattr(exporter, "download_archive", lambda *a, **k: None)
    start = pd.Timestamp("2019-12-01", tz="UTC")
    end = pd.Timestamp("2020-04-01", tz="UTC")
    plan_by_field = plan(start, end, oi_first_day=pd.Timestamp("2020-09-01", tz="UTC"))
    with pytest.raises(DerivativesExportError, match="BTCUSDT-1h-2020-01.zip"):
        exporter._require(plan_by_field["perpetual_price"][0], tmp_path, timeout=1)


@pytest.mark.parametrize("year,month", [(2020, 1), (2020, 2), (2022, 7), (2025, 5)])
def test_every_expected_kline_month_from_the_inception_onward_is_mandatory(
    monkeypatch, tmp_path, year, month
):
    """A gap *after* inception is still a hard failure, and never a skipped month."""
    import tools.export_derivatives_snapshot as exporter

    monkeypatch.setattr(exporter, "download_archive", lambda *a, **k: None)
    archive = kline_archive(year, month)
    assert archive.period_start >= pd.Timestamp(
        f"{PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY['first_protocol_month']}-01", tz="UTC"
    )
    with pytest.raises(DerivativesExportError, match="is not published"):
        exporter._require(archive, tmp_path, timeout=1)


def test_no_rest_or_spot_fallback_exists_for_pre_inception_perpetual_data():
    """There is no code path that fetches the months the archive does not publish.

    Asserted structurally rather than by reading prose: the acquisition's only
    perpetual fetch is `_require` over a planned archive, the planner never emits
    a pre-inception month, and no module in the acquisition path names the REST
    kline endpoint at all.
    """
    import inspect

    import nn.derivatives_sources as sources
    import tools.export_derivatives_snapshot as exporter

    for module in (sources, exporter):
        text = inspect.getsource(module)
        assert "fapi.binance.com/fapi/v1/klines" not in text
        assert "futures/data/klines" not in text
    # The one place a perpetual archive is fetched takes an Archive the planner
    # built, so the clamp is upstream of every fetch rather than beside one.
    assert "download_archive" in inspect.getsource(exporter._require)
    start = pd.Timestamp("2019-01-01", tz="UTC")
    end = pd.Timestamp("2019-12-01", tz="UTC")
    with pytest.raises(Exception, match="no published month"):
        plan(start, end)


def test_the_manifest_records_the_perpetual_source_boundary(exported):
    """§3.4c's provenance requirement: the clamped month does not disappear."""
    boundary = exported["payload"]["acquisition"]["perpetual_source_boundary"]
    for key in PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["provenance_required"]:
        assert key in boundary, key
    assert boundary["amendment"] == "A3"
    assert boundary["source_inception_month"] == "2020-01"
    assert boundary["no_substitution"].startswith("never")
    # And the hashed rule itself travels with the table, not a paraphrase of it.
    assert exported["payload"]["source"]["perpetual_inception_rule"] == dict(
        PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY
    )
    # A2's boundary is still there beside it, as its own object.
    assert exported["payload"]["acquisition"]["funding_source_boundary"]["amendment"] == "A2"
