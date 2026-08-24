"""What P4's acquisition is allowed to read, and everything it must refuse.

``docs/p4_preregistration.md`` §3.0b fixes the funding column mapping *before any
archive was opened*, because "we looked at the file and mapped the columns that
worked" is a researcher degree of freedom wearing an implementation detail's
clothes. These tests are that allow-list exercised against real first lines,
which is only possible because :mod:`nn.derivatives_sources` resolves the layout
from a string rather than from a downloaded ZIP.
"""

from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from nn.derivatives_sources import (
    BASE_URL,
    EARLIEST_METRICS_DAY,
    FUNDING,
    OPEN_INTEREST,
    PERPETUAL,
    DerivativesSourceError,
    canonical_member,
    check_strictly_increasing,
    funding_archive,
    funding_inception,
    funding_source_boundary,
    kline_archive,
    metrics_archive,
    perpetual_inception,
    perpetual_source_boundary,
    parse_instants,
    plan_archives,
    resolve_funding_columns,
    source_spec,
    staleness_bound_ns,
)
from nn.p4_preregistration import (
    FUNDING_ARCHIVE_INCEPTION_POLICY,
    FUNDING_CSV_COLUMN_POLICY,
    MAX_STALENESS_HOURS,
    PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY,
)
from nn.trade_aggregates import TradeAggregationError

import numpy as np

ARCHIVE = funding_archive(2023, 5)


# --- the funding allow-list of §3.0b -----------------------------------------
@pytest.mark.parametrize("entry", FUNDING_CSV_COLUMN_POLICY["allowed_header_maps"])
def test_every_preregistered_layout_is_accepted(entry):
    layout = resolve_funding_columns(",".join(entry["columns"]) + "\n", ARCHIVE)
    assert layout.settlement_instant == entry["settlement_instant"]
    assert layout.realised_funding_rate == entry["realised_funding_rate"]
    assert layout.headerless is False


def test_a_recognised_layout_may_carry_columns_the_policy_does_not_name():
    """§3.0b: extra columns are ignored. An unrecognised column *set* is not."""
    layout = resolve_funding_columns("fundingTime,fundingRate,symbol,markPrice\n", ARCHIVE)
    assert layout.settlement_instant == "fundingTime"
    assert layout.realised_funding_rate == "fundingRate"


def test_the_three_column_layout_is_named_by_its_most_specific_match():
    """`calc_time`/`last_funding_rate` is a subset of the three-column layout.

    Both entries match, they agree on the mapping, and the recorded name is the
    specific one — so a manifest says which layout the archive actually was
    rather than which subset of it happened to be checked first.
    """
    layout = resolve_funding_columns(
        "calc_time,funding_interval_hours,last_funding_rate\n", ARCHIVE
    )
    assert layout.layout == "calc_time/funding_interval_hours/last_funding_rate"


@pytest.mark.parametrize(
    "header",
    [
        "time,rate\n",
        "fundingTime,rate\n",
        "calc_time,fundingRate\n",
        "open_time,close\n",
        "fundingRate,fundingTime_ms\n",
    ],
)
def test_an_unrecognised_column_set_is_refused_rather_than_inferred(header):
    with pytest.raises(DerivativesSourceError, match="matches none of the preregistered"):
        resolve_funding_columns(header, ARCHIVE)


def test_the_refusal_quotes_the_preregistration_rather_than_paraphrasing_it():
    with pytest.raises(DerivativesSourceError) as excinfo:
        resolve_funding_columns("time,rate\n", ARCHIVE)
    assert FUNDING_CSV_COLUMN_POLICY["on_unrecognised_layout"] in str(excinfo.value)


# --- headerless -------------------------------------------------------------
def test_a_headerless_two_column_archive_is_accepted_inside_its_own_period():
    inside = int(pd.Timestamp("2023-05-14T08:00:00Z").value // 1_000_000)
    layout = resolve_funding_columns(f"{inside},0.0001\n", ARCHIVE)
    assert layout.headerless is True
    assert (layout.settlement_instant, layout.realised_funding_rate) == (0, 1)


def test_a_headerless_archive_with_three_columns_is_refused():
    """§3.0b accepts headerless data in *one* unambiguous shape and no other."""
    inside = int(pd.Timestamp("2023-05-14T08:00:00Z").value // 1_000_000)
    with pytest.raises(DerivativesSourceError, match="single unambiguous 2-column shape"):
        resolve_funding_columns(f"{inside},8,0.0001\n", ARCHIVE)


def test_a_headerless_archive_whose_instant_leaves_its_period_is_refused():
    """The archive's own calendar period decides the unit; nothing else may."""
    outside = int(pd.Timestamp("2019-01-01T00:00:00Z").value // 1_000_000)
    with pytest.raises(TradeAggregationError):
        resolve_funding_columns(f"{outside},0.0001\n", ARCHIVE)


# --- instants ----------------------------------------------------------------
def test_an_epoch_column_resolves_its_unit_from_the_archive_period():
    day = metrics_archive(pd.Timestamp("2023-05-14", tz="UTC"))
    millis = pd.Series(
        [int(pd.Timestamp(f"2023-05-14T{h:02d}:00:00Z").value // 1_000_000) for h in range(4)]
    )
    stamps = parse_instants(millis, day, what="create_time")
    assert (
        pd.Timestamp(int(stamps[0]), unit="ns", tz="UTC").isoformat().startswith("2023-05-14")
    )


def test_a_wall_clock_column_is_read_as_utc():
    day = metrics_archive(pd.Timestamp("2023-05-14", tz="UTC"))
    text = pd.Series(["2023-05-14 00:00:00", "2023-05-14 00:05:00"])
    stamps = parse_instants(text, day, what="create_time")
    assert int(stamps[1] - stamps[0]) == 5 * 60 * 1_000_000_000


def test_an_unrecognised_instant_representation_is_refused():
    day = metrics_archive(pd.Timestamp("2023-05-14", tz="UTC"))
    with pytest.raises(DerivativesSourceError, match="neither an integer epoch"):
        parse_instants(pd.Series(["14/05/2023 00:00"]), day, what="create_time")


def test_an_instant_outside_the_archives_own_day_is_refused():
    day = metrics_archive(pd.Timestamp("2023-05-14", tz="UTC"))
    with pytest.raises(DerivativesSourceError, match="outside the archive's own period"):
        parse_instants(pd.Series(["2023-05-15 00:00:00"]), day, what="create_time")


def test_a_duplicate_instant_is_a_rejection_and_not_a_de_duplication():
    """§3.4: two rows claiming one instant means the reader cannot tell which is it."""
    stamps = np.array([0, 3_600_000_000_000, 3_600_000_000_000], dtype=np.int64)
    with pytest.raises(DerivativesSourceError, match="two rows at"):
        check_strictly_increasing(stamps, ARCHIVE, what="funding settlements")


def test_an_archive_that_goes_backwards_is_refused():
    stamps = np.array([7_200_000_000_000, 3_600_000_000_000], dtype=np.int64)
    with pytest.raises(DerivativesSourceError, match="goes backwards"):
        check_strictly_increasing(stamps, ARCHIVE, what="funding settlements")


# --- ZIP members -------------------------------------------------------------
def test_the_single_root_member_is_the_member():
    assert canonical_member(
        ["BTCUSDT-metrics-2023-05-14.csv"], "BTCUSDT-metrics-2023-05-14.zip"
    )


def test_a_nested_single_member_is_refused_rather_than_read():
    with pytest.raises(DerivativesSourceError, match="nested rather than"):
        canonical_member(
            ["data/BTCUSDT-metrics-2023-05-14.csv"], "BTCUSDT-metrics-2023-05-14.zip"
        )


def test_two_members_resolve_only_to_the_canonical_name():
    names = ["BTCUSDT-metrics-2023-05-14.csv", "readme.txt"]
    assert canonical_member(names, "BTCUSDT-metrics-2023-05-14.zip") == names[0]


def test_two_ambiguous_members_are_refused_rather_than_chosen_between():
    with pytest.raises(DerivativesSourceError, match="will not guess"):
        canonical_member(["a.csv", "b.csv"], "BTCUSDT-metrics-2023-05-14.zip")


def test_a_zip_holding_only_directories_is_refused(tmp_path):
    path = tmp_path / "BTCUSDT-metrics-2023-05-14.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("nested/", b"")
    with zipfile.ZipFile(path) as zf:
        with pytest.raises(DerivativesSourceError, match="holds no files"):
            canonical_member(zf.namelist(), path.name)


# --- the plan ----------------------------------------------------------------
def test_open_interest_is_planned_daily_and_never_monthly():
    """§3.0a: Binance publishes no monthly metrics archive, so none is constructed."""
    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp("2023-01-04", tz="UTC")
    archives = plan_archives(OPEN_INTEREST, start, end)
    assert [a.kind for a in archives] == ["daily"] * 3
    for archive in archives:
        assert "/daily/metrics/" in archive.url
        assert "/monthly/metrics/" not in archive.url


def test_open_interest_is_never_requested_before_the_preregistered_first_day():
    archives = plan_archives(
        OPEN_INTEREST,
        pd.Timestamp("2019-01-01", tz="UTC"),
        pd.Timestamp("2020-09-05", tz="UTC"),
    )
    assert archives[0].period_start == pd.Timestamp(EARLIEST_METRICS_DAY, tz="UTC")


def test_a_probe_may_narrow_the_open_interest_window_and_not_widen_it():
    later = pd.Timestamp("2021-01-01", tz="UTC")
    archives = plan_archives(
        OPEN_INTEREST,
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2021-01-04", tz="UTC"),
        oi_first_day=later,
    )
    assert archives[0].period_start == later


@pytest.mark.parametrize("field", [FUNDING, PERPETUAL])
def test_funding_and_klines_are_planned_monthly(field):
    archives = plan_archives(
        field, pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-04-01", tz="UTC")
    )
    assert [a.kind for a in archives] == ["monthly"] * 3


def test_an_empty_window_is_refused():
    with pytest.raises(DerivativesSourceError, match="is empty"):
        plan_archives(
            FUNDING, pd.Timestamp("2023-02-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")
        )


# --- amendment A2: where the funding archive is allowed to begin --------------
#
# The generic warm-up planner asks for a month before the research spine so the
# 30-settlement funding window has somewhere to accumulate. For the committed
# spine that month is 2019-12, and the published monthly archive does not have
# it. §3.4b makes that a *source boundary* rather than a gap, and these tests are
# the boundary as the planner sees it.
INCEPTION_MONTH = FUNDING_ARCHIVE_INCEPTION_POLICY["first_protocol_month"]


def test_the_funding_plan_starts_at_the_source_inception_when_an_earlier_month_is_asked_for():
    """§3.4b: the plan begins at max(requested start, inception). Nothing earlier."""
    archives = plan_archives(
        FUNDING,
        pd.Timestamp("2019-12-01", tz="UTC"),
        pd.Timestamp("2020-04-01", tz="UTC"),
    )
    assert archives[0].period_start == funding_inception()
    assert archives[0].period_start == pd.Timestamp(f"{INCEPTION_MONTH}-01", tz="UTC")


def test_the_pre_inception_month_is_never_listed():
    """The specific archive that does not exist, named.

    It is not requested, so it never reaches the fail-closed rule that stops the
    acquisition on an absent funding month — which is exactly the point of §3.4b:
    a month before the source begins is outside the source, not a hole in it.
    """
    archives = plan_archives(
        FUNDING,
        pd.Timestamp("2019-09-01", tz="UTC"),
        pd.Timestamp("2020-04-01", tz="UTC"),
    )
    names = [archive.name for archive in archives]
    assert "BTCUSDT-fundingRate-2019-12.zip" not in names
    for month in ("2019-09", "2019-10", "2019-11", "2019-12"):
        assert f"BTCUSDT-fundingRate-{month}.zip" not in names


def test_the_inception_month_and_every_later_month_are_still_planned():
    """Clamping the start removes the months before the source, and nothing else."""
    archives = plan_archives(
        FUNDING,
        pd.Timestamp("2019-12-01", tz="UTC"),
        pd.Timestamp("2020-06-01", tz="UTC"),
    )
    assert [archive.name for archive in archives] == [
        f"BTCUSDT-fundingRate-2020-{month:02d}.zip" for month in range(1, 6)
    ]


@pytest.mark.parametrize("requested", ["2020-01-01", "2020-01-15", "2021-03-01", "2024-11-01"])
def test_a_start_at_or_after_the_inception_is_not_clamped(requested):
    """A2 is a floor, not a rewrite: it may narrow the window and never move it later."""
    start = pd.Timestamp(requested, tz="UTC")
    archives = plan_archives(FUNDING, start, start + pd.offsets.MonthBegin(3))
    assert archives[0].period_start == start.normalize().replace(day=1)


def test_a_window_entirely_before_the_inception_is_refused_rather_than_invented():
    with pytest.raises(DerivativesSourceError, match="no published month"):
        plan_archives(
            FUNDING,
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2019-06-01", tz="UTC"),
        )


def test_the_clamped_month_stays_visible_in_provenance():
    """§3.4b forbids the month simply disappearing from the plan's account of itself."""
    boundary = funding_source_boundary(pd.Timestamp("2019-12-01", tz="UTC"))
    assert boundary["amendment"] == "A2"
    assert boundary["generic_requested_from"] == "2019-12"
    assert boundary["source_inception_month"] == INCEPTION_MONTH
    assert boundary["effective_from"] == INCEPTION_MONTH
    assert boundary["months_clamped"] == 1
    assert boundary["not_an_internal_gap"] is True
    assert boundary["reason"] == FUNDING_ARCHIVE_INCEPTION_POLICY["pre_inception_behaviour"]


def test_provenance_reports_no_clamp_when_none_happened():
    boundary = funding_source_boundary(pd.Timestamp("2021-05-15", tz="UTC"))
    assert boundary["generic_requested_from"] == "2021-05"
    assert boundary["effective_from"] == "2021-05"
    assert boundary["months_clamped"] == 0
    assert boundary["not_an_internal_gap"] is False


def test_the_planner_reads_the_preregistered_month_rather_than_a_literal(monkeypatch):
    """The anti-drift property: move the policy, and the plan moves with it.

    A checkout where the preregistration says one month and the planner asks for
    an earlier one is what this makes unconstructible.
    """
    import nn.p4_preregistration as prereg

    with monkeypatch.context() as patch:
        patch.setattr(
            prereg,
            "FUNDING_ARCHIVE_INCEPTION_POLICY",
            {**FUNDING_ARCHIVE_INCEPTION_POLICY, "first_protocol_month": "2021-06"},
        )
        archives = plan_archives(
            FUNDING,
            pd.Timestamp("2019-12-01", tz="UTC"),
            pd.Timestamp("2021-09-01", tz="UTC"),
        )
        assert archives[0].period_start == pd.Timestamp("2021-06-01", tz="UTC")
        assert funding_inception() == pd.Timestamp("2021-06-01", tz="UTC")
    assert funding_inception() == pd.Timestamp(f"{INCEPTION_MONTH}-01", tz="UTC")


def test_a_policy_month_the_planner_cannot_read_is_refused_rather_than_guessed(monkeypatch):
    import nn.p4_preregistration as prereg

    with monkeypatch.context() as patch:
        patch.setattr(
            prereg,
            "FUNDING_ARCHIVE_INCEPTION_POLICY",
            {**FUNDING_ARCHIVE_INCEPTION_POLICY, "first_protocol_month": "January 2020"},
        )
        with pytest.raises(DerivativesSourceError, match="not a YYYY-MM month"):
            funding_inception()


def test_no_planned_funding_archive_leaves_the_public_archive():
    """The pre-inception region is not sought from the REST fallback or anywhere else."""
    archives = plan_archives(
        FUNDING,
        pd.Timestamp("2019-12-01", tz="UTC"),
        pd.Timestamp("2020-06-01", tz="UTC"),
    )
    assert archives
    for archive in archives:
        assert archive.url.startswith(f"{BASE_URL}/data/futures/um/monthly/fundingRate/")


def test_the_source_spec_reads_the_inception_policy_rather_than_restating_it():
    assert source_spec()["funding_inception_rule"] == dict(FUNDING_ARCHIVE_INCEPTION_POLICY)


# --- the staleness bounds, read rather than restated --------------------------
@pytest.mark.parametrize("field,hours", sorted(MAX_STALENESS_HOURS.items()))
def test_the_staleness_bounds_come_from_the_preregistration(field, hours):
    assert staleness_bound_ns(field) == hours * 3_600_000_000_000


def test_the_source_spec_records_the_rules_a_number_depends_on():
    spec = source_spec()
    assert spec["max_staleness_hours"] == dict(MAX_STALENESS_HOURS)
    assert "never" in spec["rest_substitution"]
    assert "never interpolated" in spec["missing_day_rule"]


# --- amendment A3: where the perpetual kline archive is allowed to begin ------
#
# The same generic warm-up month, a different archive. The acquisition plan asks
# for perpetual-price history from 2019-12 to give the basis a pre-spine overlap
# between its two legs, and the published monthly 1h kline archive does not have
# it. §3.4c makes that a *source boundary* rather than a gap, on its own evidence
# and with its own object, and these tests are that boundary as the planner sees
# it.
PERP_INCEPTION_MONTH = PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["first_protocol_month"]


def test_the_kline_plan_starts_at_the_source_inception_when_an_earlier_month_is_asked_for():
    """§3.4c: the plan begins at max(requested start, inception). Nothing earlier."""
    archives = plan_archives(
        PERPETUAL,
        pd.Timestamp("2019-12-01", tz="UTC"),
        pd.Timestamp("2020-04-01", tz="UTC"),
    )
    assert archives[0].period_start == perpetual_inception()
    assert archives[0].period_start == pd.Timestamp(f"{PERP_INCEPTION_MONTH}-01", tz="UTC")


def test_the_pre_inception_kline_month_is_never_listed():
    """The specific archive that does not exist, named.

    It is not requested, so it never reaches the fail-closed rule that stops the
    acquisition on an absent perpetual month — which is exactly the point of
    §3.4c: a month before the source begins is outside the source, not a hole in
    it.
    """
    archives = plan_archives(
        PERPETUAL,
        pd.Timestamp("2019-09-01", tz="UTC"),
        pd.Timestamp("2020-04-01", tz="UTC"),
    )
    names = [archive.name for archive in archives]
    assert "BTCUSDT-1h-2019-12.zip" not in names
    for month in ("2019-09", "2019-10", "2019-11", "2019-12"):
        assert f"BTCUSDT-1h-{month}.zip" not in names


def test_the_kline_inception_month_and_every_later_month_are_still_planned():
    """Clamping the start removes the months before the source, and nothing else."""
    archives = plan_archives(
        PERPETUAL,
        pd.Timestamp("2019-12-01", tz="UTC"),
        pd.Timestamp("2020-06-01", tz="UTC"),
    )
    assert [archive.name for archive in archives] == [
        f"BTCUSDT-1h-2020-{month:02d}.zip" for month in range(1, 6)
    ]
    assert archives[0].name == "BTCUSDT-1h-2020-01.zip"


@pytest.mark.parametrize("requested", ["2020-01-01", "2020-01-15", "2021-03-01", "2024-11-01"])
def test_a_kline_start_at_or_after_the_inception_is_not_clamped(requested):
    """A3 is a floor, not a rewrite: it may narrow the window and never move it later."""
    start = pd.Timestamp(requested, tz="UTC")
    archives = plan_archives(PERPETUAL, start, start + pd.offsets.MonthBegin(3))
    assert archives[0].period_start == start.normalize().replace(day=1)
    assert perpetual_source_boundary(start)["months_clamped"] == 0


def test_a_kline_window_entirely_before_the_inception_is_refused_rather_than_invented():
    with pytest.raises(DerivativesSourceError, match="no published month"):
        plan_archives(
            PERPETUAL,
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2019-06-01", tz="UTC"),
        )


def test_the_clamped_kline_month_stays_visible_in_provenance():
    """§3.4c forbids the month simply disappearing from the plan's account of itself."""
    boundary = perpetual_source_boundary(pd.Timestamp("2019-12-01", tz="UTC"))
    for key in PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["provenance_required"]:
        assert key in boundary, key
    assert boundary["amendment"] == "A3"
    assert boundary["field"] == "perpetual_price"
    assert boundary["generic_requested_from"] == "2019-12"
    assert boundary["source_inception_month"] == PERP_INCEPTION_MONTH
    assert boundary["effective_from"] == PERP_INCEPTION_MONTH
    assert boundary["months_clamped"] == 1
    assert boundary["not_an_internal_gap"] is True
    assert boundary["reason"] == (
        PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY["pre_inception_behaviour"]
    )


def test_kline_provenance_reports_no_clamp_when_none_happened():
    boundary = perpetual_source_boundary(pd.Timestamp("2021-05-15", tz="UTC"))
    assert boundary["generic_requested_from"] == "2021-05"
    assert boundary["effective_from"] == "2021-05"
    assert boundary["months_clamped"] == 0
    assert boundary["not_an_internal_gap"] is False


def test_the_kline_planner_reads_the_preregistered_month_rather_than_a_literal(monkeypatch):
    """The anti-drift property: move the policy, and the plan moves with it.

    A checkout where the preregistration says one month and the planner asks for
    an earlier one is what this makes unconstructible.
    """
    import nn.p4_preregistration as prereg

    with monkeypatch.context() as patch:
        patch.setattr(
            prereg,
            "PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY",
            {**PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY, "first_protocol_month": "2021-06"},
        )
        archives = plan_archives(
            PERPETUAL,
            pd.Timestamp("2019-12-01", tz="UTC"),
            pd.Timestamp("2021-09-01", tz="UTC"),
        )
        assert archives[0].period_start == pd.Timestamp("2021-06-01", tz="UTC")
        assert perpetual_inception() == pd.Timestamp("2021-06-01", tz="UTC")
        # A2's archive is a different source and is not dragged along.
        assert funding_inception() == pd.Timestamp(
            f"{FUNDING_ARCHIVE_INCEPTION_POLICY['first_protocol_month']}-01", tz="UTC"
        )
    assert perpetual_inception() == pd.Timestamp(f"{PERP_INCEPTION_MONTH}-01", tz="UTC")


def test_a_kline_policy_month_the_planner_cannot_read_is_refused_rather_than_guessed(
    monkeypatch,
):
    import nn.p4_preregistration as prereg

    with monkeypatch.context() as patch:
        patch.setattr(
            prereg,
            "PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY",
            {
                **PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY,
                "first_protocol_month": "January 2020",
            },
        )
        with pytest.raises(DerivativesSourceError, match="not a YYYY-MM month"):
            perpetual_inception()


def test_no_planned_kline_archive_leaves_the_public_kline_archive():
    """The pre-inception region is not sought from REST, from spot, or anywhere else.

    Every planned perpetual URL is under the USD-M monthly kline prefix. Nothing
    in the plan reaches `fapi.binance.com`, and nothing reaches the spot archive —
    which is the substitution §3.4c calls out by name, because the spot leg is
    already in the repository and already in the basis formula as its denominator.
    """
    archives = plan_archives(
        PERPETUAL,
        pd.Timestamp("2019-12-01", tz="UTC"),
        pd.Timestamp("2020-06-01", tz="UTC"),
    )
    assert archives
    for archive in archives:
        assert archive.url.startswith(f"{BASE_URL}/data/futures/um/monthly/klines/BTCUSDT/1h/")
        assert "fapi" not in archive.url
        assert "/spot/" not in archive.url


def test_the_source_spec_reads_the_kline_inception_policy_rather_than_restating_it():
    assert source_spec()["perpetual_inception_rule"] == dict(
        PERPETUAL_KLINE_ARCHIVE_INCEPTION_POLICY
    )
    # And it is a distinct object from A2's, not the same rule read twice.
    assert source_spec()["perpetual_inception_rule"] != source_spec()["funding_inception_rule"]


def test_the_spot_denominator_is_still_not_acquired_and_still_not_a_perpetual_proxy():
    """A3 forbids the one substitution that would otherwise look convenient."""
    spec = source_spec()
    assert "not acquired" in spec["spot_source"]
    assert spec["rest_substitution"].startswith("never")
    assert spec["perpetual_rule"] == (
        "perp_close is the close of the 1h perpetual candle opening at date, or of "
        "the candle one hour earlier when that candle is absent"
    )
    assert "spot" in spec["perpetual_inception_rule"]["no_substitution"]


def test_every_kline_archive_the_planner_builds_covers_exactly_its_own_month():
    """The clamp changes where the plan starts and nothing about what a month is."""
    archive = kline_archive(2020, 1)
    assert archive.field == PERPETUAL
    assert archive.kind == "monthly"
    assert archive.period_start == pd.Timestamp("2020-01-01", tz="UTC")
    assert archive.period_end == pd.Timestamp("2020-02-01", tz="UTC")
