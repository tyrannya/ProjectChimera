"""The loader, tested on synthetic archive bytes. No network, no Binance object.

Every payload here is constructed by the test. Nothing in this file has ever been
downloaded, and the module under test has no code path that could download one.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from nn.p13_sources import (
    CHECKSUM_NOT_SUPPLIED,
    CHECKSUM_STATES,
    CHECKSUM_UNVERIFIED,
    CHECKSUM_VERIFIED,
    KLINE_COLUMNS,
    RESEARCH_BOUNDARY_NS,
    ObjectProvenance,
    SourceError,
    extract_single_member,
    period_bounds,
    read_funding_object,
    read_kline_object,
    straddles_boundary,
    verify_published_checksum,
)
from tests.p13_synthetic import kline_csv, kline_row_fields, ms, ns, zip_bytes

PERIOD = "2021-03"
START = "2021-03-01T00:00:00+00:00"


def _klines(count: int = 4, *, header: bool = False, price: str = "30000") -> bytes:
    rows = [kline_row_fields(ms(START) + index * 3_600_000, price) for index in range(count)]
    return kline_csv(rows, header=header)


def _read(payload: bytes, *, period: str = PERIOD, field: str = "spot_price"):
    return read_kline_object(
        payload, field=field, object_name=f"BTCUSDT-1h-{period}.zip", period=period
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_a_headerless_published_layout_reads():
    table = _read(_klines(4))
    assert len(table.rows) == 4
    assert table.rows[0].instant_ns == ns(START)
    assert table.rows[0].close == Decimal("30000")


def test_a_headered_published_layout_reads_and_the_header_is_not_a_row():
    table = _read(_klines(4, header=True))
    assert len(table.rows) == 4
    assert table.provenance.rows_read == 4


def test_prices_stay_decimal_and_never_pass_through_float():
    """A price with more digits than a double can hold must survive intact."""
    exact = "30000.123456789012345678"
    table = _read(_klines(1, price=exact))
    assert table.rows[0].close == Decimal(exact)
    assert str(table.rows[0].close) == exact


def test_an_unrecognised_header_is_refused_rather_than_mapped_by_position():
    payload = kline_csv(
        [kline_row_fields(ms(START), "30000")],
        header=True,
        columns=["when", "o", "h", "l", "c", *KLINE_COLUMNS[5:]],
    )
    with pytest.raises(SourceError):
        _read(payload)


def test_a_short_headerless_row_is_refused():
    payload = kline_csv([[ms(START), "30000", "30000"]])
    with pytest.raises(SourceError, match="fewer than"):
        _read(payload)


# ---------------------------------------------------------------------------
# Timestamp units
# ---------------------------------------------------------------------------


def test_the_unit_is_resolved_against_the_objects_own_period():
    """Milliseconds for a 2021 futures object, resolved rather than assumed."""
    table = _read(_klines(4))
    assert table.provenance.resolved_epoch_unit == "ms"


def test_microsecond_spot_timestamps_from_2025_resolve_as_microseconds():
    """The unit change this checkpoint straddles, read from the archive's period."""
    period = "2025-02"
    start = "2025-02-01T00:00:00+00:00"
    rows = [
        kline_row_fields(ns(start) // 1_000 + index * 3_600_000_000, "95000")
        for index in range(3)
    ]
    table = read_kline_object(
        kline_csv(rows),
        field="spot_price",
        object_name="BTCUSDT-1h-2025-02.zip",
        period=period,
    )
    assert table.provenance.resolved_epoch_unit == "us"
    assert table.rows[0].instant_ns == ns(start)


def test_a_timestamp_that_fits_no_unit_is_refused_rather_than_guessed():
    payload = kline_csv([kline_row_fields(1, "30000")])
    with pytest.raises(SourceError, match="epoch unit"):
        _read(payload)


# ---------------------------------------------------------------------------
# Validity: unusable rows withhold their instant
# ---------------------------------------------------------------------------


def test_a_non_positive_price_withholds_its_instant_and_is_counted():
    rows = [
        kline_row_fields(ms(START), "30000"),
        kline_row_fields(ms(START) + 3_600_000, "0"),
        kline_row_fields(ms(START) + 7_200_000, "30000"),
    ]
    table = _read(kline_csv(rows))
    assert [row.instant_ns for row in table.rows] == [ns(START), ns(START) + 7_200_000_000_000]
    assert table.provenance.non_positive_instants == 1
    assert table.provenance.rows_read == 2


def test_two_disagreeing_rows_at_one_instant_withhold_that_instant_entirely():
    """Both copies go. Keeping either would let row order pick the candle."""
    rows = [
        kline_row_fields(ms(START), "30000"),
        kline_row_fields(ms(START), "31000"),
        kline_row_fields(ms(START) + 3_600_000, "30000"),
    ]
    table = _read(kline_csv(rows))
    assert [row.instant_ns for row in table.rows] == [ns(START) + 3_600_000_000_000]
    assert table.provenance.ambiguous_instants == 1


def test_an_identical_duplicate_row_is_collapsed_and_decides_nothing():
    rows = [kline_row_fields(ms(START), "30000")] * 2
    table = _read(kline_csv(rows))
    assert len(table.rows) == 1
    assert table.provenance.ambiguous_instants == 0


# ---------------------------------------------------------------------------
# The research boundary
# ---------------------------------------------------------------------------


def test_only_the_straddling_month_may_be_truncated():
    assert straddles_boundary("2025-05")
    assert not straddles_boundary("2025-04")
    assert not straddles_boundary("2025-06")


def test_the_straddling_month_is_truncated_and_the_drop_is_recorded():
    period = "2025-05"
    boundary_ms = RESEARCH_BOUNDARY_NS // 1_000_000
    rows = [
        kline_row_fields(boundary_ms - 3_600_000, "95000"),
        kline_row_fields(boundary_ms, "95000"),
        kline_row_fields(boundary_ms + 3_600_000, "95000"),
    ]
    table = read_kline_object(
        kline_csv(rows),
        field="spot_price",
        object_name="BTCUSDT-1h-2025-05.zip",
        period=period,
    )
    assert len(table.rows) == 1
    assert table.provenance.rows_dropped_at_boundary == 2
    assert table.provenance.last_instant_ns < RESEARCH_BOUNDARY_NS


def test_any_other_month_carrying_a_boundary_crossing_row_is_refused_not_filtered():
    """DATA_BOUNDARY.enforcement: a row past the boundary is a refusal.

    June 2025 lies wholly after the boundary and does not straddle it, so its rows
    are refused outright rather than truncated to nothing.
    """
    june = ms("2025-06-01T00:00:00+00:00")
    rows = [kline_row_fields(june + index * 3_600_000, "95000") for index in range(3)]
    with pytest.raises(SourceError, match="REFUSAL"):
        read_kline_object(
            kline_csv(rows),
            field="spot_price",
            object_name="BTCUSDT-1h-2025-06.zip",
            period="2025-06",
        )


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------


def _funding(payload_rows, header=None) -> bytes:
    lines = []
    if header:
        lines.append(",".join(header))
    lines.extend(",".join(str(cell) for cell in row) for row in payload_rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_the_frozen_funding_header_layouts_are_accepted():
    payload = _funding(
        [[ms(START), "0.0001"], [ms(START) + 28_800_000, "-0.0002"]],
        header=["fundingTime", "fundingRate"],
    )
    table = read_funding_object(
        payload, object_name="BTCUSDT-fundingRate-2021-03.zip", period=PERIOD
    )
    assert [row.rate for row in table.rows] == [Decimal("0.0001"), Decimal("-0.0002")]


def test_the_calc_time_layout_carries_the_settlement_interval():
    payload = _funding(
        [[ms(START), 8, "0.0001"]],
        header=["calc_time", "funding_interval_hours", "last_funding_rate"],
    )
    table = read_funding_object(
        payload, object_name="BTCUSDT-fundingRate-2021-03.zip", period=PERIOD
    )
    assert table.rows[0].interval_hours == Decimal("8")


def test_an_unrecognised_funding_layout_is_refused():
    """A column-name SET the frozen policy does not list is a refusal, not a guess."""
    payload = _funding([[ms(START), 8, "0.0001"]], header=["when", "interval", "rate"])
    with pytest.raises(SourceError, match="unrecognised fundingRate layout"):
        read_funding_object(
            payload, object_name="BTCUSDT-fundingRate-2021-03.zip", period=PERIOD
        )


def test_a_two_column_header_is_still_refused_rather_than_read_as_data():
    """The frozen positional layout is admitted "only when the first column parses
    as an epoch instant inside the archive's own calendar period", so a header row
    sitting in that shape fails that condition instead of becoming a settlement."""
    payload = _funding([[ms(START), "0.0001"]], header=["t", "r"])
    with pytest.raises(SourceError):
        read_funding_object(
            payload, object_name="BTCUSDT-fundingRate-2021-03.zip", period=PERIOD
        )


def test_a_repeated_funding_instant_is_counted_and_passed_through():
    """Resolved once, in the accounting engine — not twice, in two places."""
    payload = _funding([[ms(START), "0.0001"], [ms(START), "0.0001"]])
    table = read_funding_object(
        payload, object_name="BTCUSDT-fundingRate-2021-03.zip", period=PERIOD
    )
    assert len(table.rows) == 2
    assert table.provenance.repeated_instants == 1


# ---------------------------------------------------------------------------
# Objects and provenance
# ---------------------------------------------------------------------------


def test_a_published_object_holds_exactly_one_member():
    payload = _klines(2)
    raw = zip_bytes("BTCUSDT-1h-2021-03.csv", payload)
    name, member = extract_single_member(raw)
    assert name.endswith(".csv")
    assert member == payload


def test_an_archive_with_two_members_is_refused_rather_than_chosen_between():
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.csv", _klines(1))
        archive.writestr("b.csv", _klines(1))
    with pytest.raises(SourceError, match="exactly one"):
        extract_single_member(buffer.getvalue())


def test_the_recorded_digest_is_of_the_whole_object_not_the_member():
    """The digest checkable against Binance's .CHECKSUM is the ZIP's."""
    payload = _klines(2)
    raw = zip_bytes("BTCUSDT-1h-2021-03.csv", payload)
    _, member = extract_single_member(raw)
    table = read_kline_object(
        member,
        field="spot_price",
        object_name="BTCUSDT-1h-2021-03.zip",
        period=PERIOD,
        raw_object=raw,
    )
    assert table.provenance.sha256 == hashlib.sha256(raw).hexdigest()
    assert table.provenance.byte_size == len(raw)


def test_an_unverified_checksum_is_reported_as_unverified_not_as_matched():
    table = _read(_klines(2))
    assert table.provenance.published_checksum is None
    assert table.provenance.as_dict()["checksum_verified"] is False
    assert table.provenance.checksum_state == CHECKSUM_NOT_SUPPLIED


# ---------------------------------------------------------------------------
# BOTH digests, and a checksum state that cannot claim what it did not check
# ---------------------------------------------------------------------------


def _object(count: int = 3, member: str = "BTCUSDT-1h-2021-03.csv"):
    """One published object and its single member, as bytes."""
    payload = _klines(count)
    return zip_bytes(member, payload), payload, member


def test_the_archive_and_member_digests_are_both_recorded_and_are_different():
    """SOURCE_FREEZE_FIELDS requires both, and they attest to different bytes."""
    raw, payload, member = _object()
    table = read_kline_object(
        payload,
        field="spot_price",
        object_name="BTCUSDT-1h-2021-03.zip",
        period=PERIOD,
        raw_object=raw,
        member_name=member,
    )
    provenance = table.provenance
    # Recomputed here, independently of the module under test.
    assert provenance.sha256 == hashlib.sha256(raw).hexdigest()
    assert provenance.byte_size == len(raw)
    assert provenance.member_sha256 == hashlib.sha256(payload).hexdigest()
    assert provenance.member_byte_size == len(payload)
    assert provenance.member_name == member
    # A zip frames and compresses its member, so the two digests are of different
    # bytes and must differ. Their SIZES are not asserted to differ: compression
    # can coincidentally land on the payload's own length, and it does here.
    assert provenance.sha256 != provenance.member_sha256


def test_the_evidence_serialises_both_digests_without_conflating_them():
    raw, payload, member = _object()
    table = read_kline_object(
        payload,
        field="spot_price",
        object_name="BTCUSDT-1h-2021-03.zip",
        period=PERIOD,
        raw_object=raw,
        member_name=member,
    )
    emitted = table.provenance.as_dict()
    assert emitted["archive_sha256"] == hashlib.sha256(raw).hexdigest()
    assert emitted["member_sha256"] == hashlib.sha256(payload).hexdigest()
    assert emitted["archive_byte_size"] == len(raw)
    assert emitted["member_byte_size"] == len(payload)
    assert emitted["member_name"] == member
    assert emitted["archive_sha256"] != emitted["member_sha256"]


def test_a_member_parsed_without_its_archive_records_no_archive_digest():
    """The member digest must never stand in for the publisher-checkable one."""
    _, payload, _ = _object()
    table = _read(payload)
    assert table.provenance.sha256 is None
    assert table.provenance.byte_size is None
    assert table.provenance.member_sha256 == hashlib.sha256(payload).hexdigest()


def test_a_matching_published_checksum_is_verified_by_an_actual_comparison():
    raw, payload, member = _object()
    table = read_kline_object(
        payload,
        field="spot_price",
        object_name="BTCUSDT-1h-2021-03.zip",
        period=PERIOD,
        raw_object=raw,
        member_name=member,
        published_checksum=hashlib.sha256(raw).hexdigest(),
    )
    assert table.provenance.checksum_state == CHECKSUM_VERIFIED
    assert table.provenance.checksum_verified is True
    assert table.provenance.as_dict()["checksum_verified"] is True


def test_binances_published_companion_format_is_parsed_not_rejected():
    """Binance publishes ``<digest>  <filename>``, not a bare digest."""
    raw, payload, member = _object()
    companion = hashlib.sha256(raw).hexdigest() + "  BTCUSDT-1h-2021-03.zip\n"
    table = read_kline_object(
        payload,
        field="spot_price",
        object_name="BTCUSDT-1h-2021-03.zip",
        period=PERIOD,
        raw_object=raw,
        member_name=member,
        published_checksum=companion,
    )
    assert table.provenance.checksum_state == CHECKSUM_VERIFIED


def test_a_deliberate_checksum_mismatch_is_a_refusal_not_a_flag():
    """The whole point: a disagreeing object is not read at all."""
    raw, payload, member = _object()
    with pytest.raises(SourceError, match="does not match"):
        read_kline_object(
            payload,
            field="spot_price",
            object_name="BTCUSDT-1h-2021-03.zip",
            period=PERIOD,
            raw_object=raw,
            member_name=member,
            published_checksum="0" * 64,
        )


def test_a_checksum_supplied_without_the_archive_bytes_is_not_called_verified():
    """The state that used to be reported as verified merely for existing."""
    _, payload, _ = _object()
    table = read_kline_object(
        payload,
        field="spot_price",
        object_name="BTCUSDT-1h-2021-03.zip",
        period=PERIOD,
        published_checksum="0" * 64,
    )
    assert table.provenance.published_checksum is not None
    assert table.provenance.checksum_state == CHECKSUM_UNVERIFIED
    assert table.provenance.checksum_verified is False
    assert table.provenance.as_dict()["checksum_verified"] is False


def test_the_four_checksum_facts_are_distinguishable():
    """Absent, supplied-but-unchecked, matched and mismatched are four states."""
    assert len(set(CHECKSUM_STATES)) == 4
    raw, _, _ = _object()
    digest = hashlib.sha256(raw).hexdigest()
    assert verify_published_checksum(raw, None) == CHECKSUM_NOT_SUPPLIED
    assert verify_published_checksum(None, digest) == CHECKSUM_UNVERIFIED
    assert verify_published_checksum(raw, digest) == CHECKSUM_VERIFIED
    with pytest.raises(SourceError, match="REFUSAL"):
        verify_published_checksum(raw, "1" * 64)


def test_provenance_cannot_be_constructed_claiming_a_verification_it_lacks():
    """The state is re-derived by the constructor, so it cannot simply be typed in."""
    common = dict(
        field="spot_price",
        object_name="BTCUSDT-1h-2021-03.zip",
        period=PERIOD,
        member_sha256="ab" * 32,
        member_byte_size=10,
        resolved_epoch_unit="ms",
        rows_read=1,
        rows_dropped_at_boundary=0,
        first_instant_ns=1,
        last_instant_ns=1,
    )
    # Verified, but the digests disagree.
    with pytest.raises(SourceError, match="differ"):
        ObjectProvenance(
            byte_size=10,
            sha256="aa" * 32,
            published_checksum="bb" * 32,
            checksum_state=CHECKSUM_VERIFIED,
            **common,
        )
    # Verified, with no archive digest to have verified against.
    with pytest.raises(SourceError, match="requires the whole-object digest"):
        ObjectProvenance(
            byte_size=None,
            sha256=None,
            published_checksum="bb" * 32,
            checksum_state=CHECKSUM_VERIFIED,
            **common,
        )
    # Verified, with no published checksum at all.
    with pytest.raises(SourceError, match="claims a published checksum"):
        ObjectProvenance(
            byte_size=10,
            sha256="aa" * 32,
            published_checksum=None,
            checksum_state=CHECKSUM_VERIFIED,
            **common,
        )
    # A mismatch is a refusal and may never be recorded as provenance.
    with pytest.raises(SourceError, match="REFUSAL"):
        ObjectProvenance(
            byte_size=10,
            sha256="aa" * 32,
            published_checksum="aa" * 32,
            checksum_state="mismatch_refused",
            **common,
        )
    # And the honest construction is accepted, case-insensitively.
    ok = ObjectProvenance(
        byte_size=10,
        sha256="aa" * 32,
        published_checksum="AA" * 32,
        checksum_state=CHECKSUM_VERIFIED,
        **common,
    )
    assert ok.checksum_verified is True


def test_the_funding_reader_records_both_digests_and_verifies_too():
    payload = b"1614556800000,0.0001\n"
    raw = zip_bytes("BTCUSDT-fundingRate-2021-03.csv", payload)
    table = read_funding_object(
        payload,
        object_name="BTCUSDT-fundingRate-2021-03.zip",
        period=PERIOD,
        raw_object=raw,
        member_name="BTCUSDT-fundingRate-2021-03.csv",
        published_checksum=hashlib.sha256(raw).hexdigest(),
    )
    assert table.provenance.checksum_state == CHECKSUM_VERIFIED
    assert table.provenance.sha256 == hashlib.sha256(raw).hexdigest()
    assert table.provenance.member_sha256 == hashlib.sha256(payload).hexdigest()
    assert table.provenance.sha256 != table.provenance.member_sha256


def test_period_bounds_are_the_objects_own_utc_month():
    start, end = period_bounds("2021-12")
    assert start.isoformat() == "2021-12-01T00:00:00+00:00"
    assert end.isoformat() == "2022-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# No network
# ---------------------------------------------------------------------------


def test_the_loader_module_imports_nothing_that_could_reach_a_network():
    """Checked on the IMPORT GRAPH, not on the prose.

    The docstring names ``data.binance.vision`` in order to say the module cannot
    reach it, so a substring scan would fail on its own disclaimer. What matters is
    that no networking module is imported, which the AST answers exactly.
    """
    import ast
    import inspect

    import nn.p13_sources as module

    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("requests", "urllib", "urllib3", "http", "socket", "ftplib", "aiohttp"):
        assert forbidden not in imported, f"{forbidden} is imported by the offline loader"
