"""The one source defect P4's metrics acquisition normalises, and everything else.

**The condition this exists for.** The official Binance USD-M daily metrics
archives for BTCUSDT dated **2020-09-01** and **2020-09-02** were observed to
hold 576 data rows for 288 five-minute ``create_time`` instants: every
observation appears exactly twice, and the paired rows are identical across the
full CSV row. The fail-closed reader refused them, correctly — under §3.4 a
duplicate instant means the reader cannot tell which row is the observation.

The narrowing is that when there is nothing to tell apart, there is nothing to
choose between. Rows repeating an instant collapse to one logical observation
**if and only if every source field in them is identical**; a single differing
field — including in a column ``derivatives_v1`` never reads — is a conflict and
still stops the acquisition.

Nothing here claims other days are duplicated. Two archives were measured; the
rule is written for whatever the source actually serves.

Every test below goes through :func:`tools.export_derivatives_snapshot.read_metrics`
on a real ZIP, because that is the production path — the collapse, the schema
refusal and the strictly-increasing check are only in the right order there.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from nn.derivatives_sources import (
    HOUR_NS,
    DerivativesSourceError,
    check_strictly_increasing,
    funding_archive,
    metrics_archive,
)
from tools.export_derivatives_snapshot import read_metrics

DAY = pd.Timestamp("2020-09-01", tz="UTC")
ARCHIVE = metrics_archive(DAY)

#: The real archive's header. ``symbol`` and the long/short ratios are columns
#: §3.0a does not read — and they still decide whether two rows are the same row.
HEADER = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)


def _row(minute: int, *, contracts: str = "10000.000", notional: str = "1.0E8", **over):
    fields = {
        "create_time": f"2020-09-01 {minute // 60:02d}:{minute % 60:02d}:00",
        "symbol": "BTCUSDT",
        "sum_open_interest": contracts,
        "sum_open_interest_value": notional,
        "count_toptrader_long_short_ratio": "1.5000",
        "sum_taker_long_short_vol_ratio": "0.9000",
    }
    fields.update(over)
    return tuple(fields[name] for name in HEADER)


def _archive(tmp_path, rows, *, header=HEADER, name=ARCHIVE.name):
    """A real metrics ZIP holding exactly ``rows``, in the order given."""
    path = tmp_path / name
    body = ",".join(header) + "\n" + "".join(",".join(row) + "\n" for row in rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name.removesuffix(".zip") + ".csv", body)
    return path


# --- 1. one exact duplicate pair ---------------------------------------------
def test_one_exact_duplicate_pair_is_accepted_as_one_observation(tmp_path):
    rows = [_row(0), _row(5), _row(5), _row(10)]
    instants, contracts, notional, norm = read_metrics(_archive(tmp_path, rows), ARCHIVE)

    assert len(instants) == 3, "the repeated instant is one logical observation"
    assert norm.rows_read == 4
    assert norm.observations_retained == 3
    assert norm.exact_duplicate_rows_collapsed == 1
    assert norm.duplicate_instants == 1
    assert [pd.Timestamp(int(v), unit="ns", tz="UTC").minute for v in instants] == [0, 5, 10]
    assert contracts.tolist() == [10000.0] * 3
    assert notional.tolist() == [1.0e8] * 3


# --- 2. every observation duplicated: the observed real-source shape ----------
def test_a_wholly_duplicated_archive_halves_to_its_logical_observations(tmp_path):
    """The 2020-09-01/2020-09-02 shape: 2N rows, N instants, paired rows equal."""
    minutes = list(range(0, 60, 5))
    rows = [_row(m, contracts=f"{10000 + m}.000") for m in minutes for _ in (0, 1)]
    instants, contracts, _, norm = read_metrics(_archive(tmp_path, rows), ARCHIVE)

    assert norm.rows_read == 2 * len(minutes)
    assert norm.observations_retained == len(minutes)
    assert norm.exact_duplicate_rows_collapsed == len(minutes)
    assert norm.duplicate_instants == len(minutes)
    assert [
        pd.Timestamp(int(v), unit="ns", tz="UTC").minute for v in instants
    ] == minutes, "order and timestamps survive the collapse"
    assert contracts.tolist() == [float(10000 + m) for m in minutes]


# --- 3. three identical rows at one instant ----------------------------------
def test_three_identical_rows_at_one_instant_collapse_to_one(tmp_path):
    rows = [_row(0), _row(5), _row(5), _row(5), _row(10)]
    instants, _, _, norm = read_metrics(_archive(tmp_path, rows), ARCHIVE)

    assert len(instants) == 3
    assert norm.exact_duplicate_rows_collapsed == 2, "two removed, not one"
    assert norm.duplicate_instants == 1
    assert norm.rows_read - norm.exact_duplicate_rows_collapsed == norm.observations_retained


# --- 4. a consumed field disagrees -------------------------------------------
def test_the_same_instant_with_different_open_interest_is_still_refused(tmp_path):
    rows = [_row(0), _row(5), _row(5, contracts="10001.000")]
    with pytest.raises(DerivativesSourceError) as excinfo:
        read_metrics(_archive(tmp_path, rows), ARCHIVE)

    message = str(excinfo.value)
    assert ARCHIVE.name in message, "the refusal names the archive"
    assert "2020-09-01T00:05:00+00:00" in message, "the refusal names the instant"
    assert "disagree" in message
    assert "sum_open_interest=" in message


# --- 5. an unconsumed field disagrees ----------------------------------------
def test_the_same_instant_agreeing_on_open_interest_but_not_elsewhere_is_refused(tmp_path):
    """§3.0a reads three columns. The equality test reads the whole row.

    Both rows carry the same ``create_time``, ``sum_open_interest`` and
    ``sum_open_interest_value``, so a de-duplication keyed on what P4 consumes
    would silently accept them. A source disagreeing with itself about a field
    this design ignores is still a source disagreeing with itself.
    """
    rows = [_row(0), _row(5), _row(5, sum_taker_long_short_vol_ratio="0.9500")]
    with pytest.raises(DerivativesSourceError, match="disagree"):
        read_metrics(_archive(tmp_path, rows), ARCHIVE)


# --- 6. schema identity is enforced before normalisation ----------------------
def test_a_duplicate_under_an_unrecognised_schema_is_refused_for_the_schema(tmp_path):
    """Normalisation must not become a way past the header check."""
    header = ("create_time", "symbol", "sum_open_interest")
    rows = [
        ("2020-09-01 00:00:00", "BTCUSDT", "10000.000"),
        ("2020-09-01 00:05:00", "BTCUSDT", "10000.000"),
        ("2020-09-01 00:05:00", "BTCUSDT", "10000.000"),
    ]
    with pytest.raises(DerivativesSourceError) as excinfo:
        read_metrics(_archive(tmp_path, rows, header=header), ARCHIVE)

    message = str(excinfo.value)
    assert "sum_open_interest_value" in message and "no column" in message
    assert "disagree" not in message, "the schema refusal wins, not the duplicate rule"


def test_an_instant_outside_the_archives_own_day_still_stops_a_duplicated_archive(tmp_path):
    rows = [_row(0), _row(0)]
    other_day = metrics_archive(pd.Timestamp("2020-09-02", tz="UTC"))
    with pytest.raises(DerivativesSourceError, match="outside the archive's own period"):
        read_metrics(_archive(tmp_path, rows, name=other_day.name), other_day)


# --- 7. the ordinary archive is untouched ------------------------------------
def test_an_archive_without_duplicates_reads_exactly_as_before(tmp_path):
    minutes = list(range(0, 60, 5))
    rows = [_row(m, contracts=f"{10000 + m}.000") for m in minutes]
    instants, contracts, notional, norm = read_metrics(_archive(tmp_path, rows), ARCHIVE)

    assert len(instants) == len(minutes)
    assert norm.rows_read == norm.observations_retained == len(minutes)
    assert norm.exact_duplicate_rows_collapsed == 0
    assert norm.duplicate_instants == 0
    assert contracts.tolist() == [float(10000 + m) for m in minutes]
    assert notional.tolist() == [1.0e8] * len(minutes)


def test_the_other_two_sources_still_reject_a_duplicate_instant_outright(tmp_path):
    """The narrowing is metrics-only. Funding and klines keep §3.4 as written.

    Exercised on the shared check rather than on a second synthetic archive,
    because that check is the *whole* of the duplicate policy for those two
    sources: nothing in :func:`tools.export_derivatives_snapshot.read_funding` or
    ``iter_klines`` collapses anything before reaching it.
    """
    funding = funding_archive(2020, 9)
    stamps = np.array([0, HOUR_NS, HOUR_NS], dtype=np.int64) + int(funding.period_start.value)
    with pytest.raises(DerivativesSourceError, match="two rows at"):
        check_strictly_increasing(stamps, funding, what="funding settlements")


# --- 8. logical identity is stable; raw identity is not pretended away --------
def test_collapsing_yields_the_same_logical_data_as_the_unduplicated_archive(tmp_path):
    """The point of the normalisation, stated as an equality.

    A duplicated archive and the clean archive it duplicates must reduce to the
    same observations — same instants, same values, same order — so the hourly
    table and every fingerprint downstream of it are identical. What must *not*
    become identical is the raw archive: the bytes differ, the row count read
    differs, and the provenance records both.
    """
    minutes = list(range(0, 60, 5))
    clean_rows = [_row(m, contracts=f"{10000 + m}.000") for m in minutes]
    duplicated_rows = [row for row in clean_rows for _ in (0, 1)]

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    clean_path = _archive(tmp_path / "a", clean_rows)
    duplicated_path = _archive(tmp_path / "b", duplicated_rows)

    clean_out = read_metrics(clean_path, ARCHIVE)
    duplicated_out = read_metrics(duplicated_path, ARCHIVE)

    assert np.array_equal(clean_out[0], duplicated_out[0])
    assert np.array_equal(clean_out[1], duplicated_out[1])
    assert np.array_equal(clean_out[2], duplicated_out[2])

    assert clean_out[3].observations_retained == duplicated_out[3].observations_retained
    assert (
        clean_out[3].rows_read != duplicated_out[3].rows_read
    ), "the raw source identity must not be normalised away with the duplicates"
    assert clean_path.read_bytes() != duplicated_path.read_bytes()
