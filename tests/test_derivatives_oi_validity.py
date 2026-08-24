"""Which published metrics rows are open-interest observations, and which are not.

**The condition this exists for.** The first real full derivatives acquisition
after amendment A3 completed all 1722 planned open-interest days and then failed
during construction and validation of the hourly table, on the pre-write check
that refuses ``open interest is non-positive on an available hour``. Direct
inspection of the source — before any P4 fit, Stage-1 result or outcome — found
that the daily metrics archive publishes rows whose consumed open-interest
metrics are exactly zero, in two shapes:

- both consumed fields zero, on multiple dates, in runs as long as 102
  consecutive five-minute observations (510 minutes, 8.5 hours);
- ``sum_open_interest > 0`` with ``sum_open_interest_value == 0`` — exactly 12
  such rows in the whole planned range, all on 2023-04-10.

The same complete scan found no negative consumed value and finished with
``errors=0``.

Amendment A4 is the rule those observations forced, and it is deliberately
narrow. It does **not** say what the true open interest was during such a row,
and it does not claim the venue documents zero as a sentinel: it says only that a
non-positive consumed metric is not a valid *positive* open-interest state, so it
is not admitted as an observation. The row stays counted, the hours it leaves
without a valid observation inside the unchanged 1-hour staleness bound become
unavailable for every arm, and nothing is interpolated, averaged, substituted or
carried backwards to hide that.

Every test goes through the production path —
:func:`tools.export_derivatives_snapshot.read_metrics` on a real ZIP, and the
exporter's own reducer — because the ordering the rule depends on (schema, then
A1's exact-duplicate collapse, then the numeric parse, then A4) only exists there.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from nn.derivatives_sources import (
    HOUR_NS,
    OPEN_INTEREST,
    DerivativesSourceError,
    metrics_archive,
    source_spec,
    staleness_bound_ns,
)
from nn.p4_preregistration import (
    FEATURES,
    MAX_STALENESS_HOURS,
    OPEN_INTEREST_OBSERVATION_VALIDITY_POLICY as A4,
)
from tools.export_derivatives_snapshot import (
    HourlyLastObservation,
    read_metrics,
    resolve_availability,
)

DAY = pd.Timestamp("2020-09-01", tz="UTC")
ARCHIVE = metrics_archive(DAY)

#: The real archive's header, including the columns §3.0a does not consume.
HEADER = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

ZERO = "0.00000000"


def _row(minute: int, *, contracts: str = "10000.000", notional: str = "1.0E8", **over):
    """One published row, at ``minute`` minutes past 2020-09-01T00:00Z."""
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


def _both_zero(minute: int, **over):
    """The observed paired-zero shape."""
    return _row(minute, contracts=ZERO, notional=ZERO, **over)


def _zero_notional(minute: int, **over):
    """The observed 2023-04-10 shape: positive contracts, zero notional."""
    return _row(minute, contracts="106675.406", notional=ZERO, **over)


def _archive(tmp_path, rows, *, header=HEADER, name=ARCHIVE.name):
    path = tmp_path / name
    body = ",".join(header) + "\n" + "".join(",".join(row) + "\n" for row in rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name.removesuffix(".zip") + ".csv", body)
    return path


def _instant(minute: int) -> int:
    return int(DAY.value) + minute * 60 * 1_000_000_000


def _hour(hour: int) -> int:
    return int(DAY.value) + hour * HOUR_NS


def _reduce(tmp_path, rows, *, hours):
    """One archive through the production reducer, onto a fixed hourly grid.

    Returns the availability the exporter would write, the ages behind it, the
    reducer itself — so that what *entered* the causal sequence can be inspected
    directly — and A4's per-archive accounting.
    """
    instants, contracts, notional, normalisation, validity = read_metrics(
        _archive(tmp_path, rows), ARCHIVE
    )
    reducer = HourlyLastObservation(hours=np.asarray(hours, dtype=np.int64), width=2)
    reducer.add(instants, np.column_stack((contracts, notional)))
    reducer.finish()
    available, age = resolve_availability(reducer, OPEN_INTEREST)
    return available, age, reducer, normalisation, validity


# --- 1-4. the four value combinations ----------------------------------------
def test_positive_contracts_and_positive_notional_is_a_valid_observation(tmp_path):
    rows = [_row(0), _row(5), _row(10)]
    instants, contracts, notional, _, validity = read_metrics(
        _archive(tmp_path, rows), ARCHIVE
    )

    assert len(instants) == 3, "every row is a valid observation"
    assert contracts.tolist() == [10000.0] * 3
    assert notional.tolist() == [1.0e8] * 3
    assert validity.logical_observations == 3
    assert validity.valid_positive_observations == 3
    assert validity.invalid_zero_observations == 0


def test_zero_contracts_and_zero_notional_is_invalid_and_never_reaches_the_reducer(
    tmp_path,
):
    """The observed paired-zero shape, which is what the failing run tripped on."""
    rows = [_row(0), _both_zero(5), _row(10)]
    available, _, reducer, _, validity = _reduce(tmp_path, rows, hours=[_hour(0), _hour(1)])

    assert validity.logical_observations == 3
    assert validity.valid_positive_observations == 2
    assert validity.invalid_zero_observations == 1
    assert validity.invalid_both_zero_observations == 1
    assert validity.invalid_zero_contracts_only == 0
    assert validity.invalid_zero_notional_only == 0
    # The rejected row is not in the causal sequence at all: the instants the two
    # grid hours resolve to are the two valid ones, and 00:05 is not among them.
    assert set(np.unique(reducer.at[reducer.at >= 0]).tolist()) == {
        _instant(0),
        _instant(10),
    }
    assert bool(available[1]), "the valid 00:10 observation is what 01:00 sees"
    assert int(reducer.at[1]) == _instant(10)
    assert reducer.values[1].tolist() == [10000.0, 1.0e8]


def test_positive_contracts_with_zero_notional_is_invalid(tmp_path):
    """The 2023-04-10 shape. A positive contract count does not rescue it."""
    rows = [_row(0), _zero_notional(5), _row(10)]
    _, _, reducer, _, validity = _reduce(tmp_path, rows, hours=[_hour(0), _hour(1)])

    assert validity.valid_positive_observations == 2
    assert validity.invalid_zero_observations == 1
    assert validity.invalid_zero_notional_only == 1
    assert validity.invalid_both_zero_observations == 0
    assert validity.invalid_zero_contracts_only == 0
    assert int(reducer.at[1]) == _instant(10), "01:00 resolves past the rejected row"


def test_zero_contracts_with_positive_notional_is_invalid(tmp_path):
    """Not a shape the scan found. The rule is symmetric anyway, and says so."""
    rows = [_row(0), _row(5, contracts=ZERO), _row(10)]
    _, _, reducer, _, validity = _reduce(tmp_path, rows, hours=[_hour(0), _hour(1)])

    assert validity.valid_positive_observations == 2
    assert validity.invalid_zero_observations == 1
    assert validity.invalid_zero_contracts_only == 1
    assert validity.invalid_both_zero_observations == 0
    assert validity.invalid_zero_notional_only == 0
    assert int(reducer.at[1]) == _instant(10), "01:00 resolves past the rejected row"


# --- 5-7. negative and non-finite are hard failures, not invalid observations --
def test_negative_contracts_stop_the_acquisition(tmp_path):
    rows = [_row(0), _row(5, contracts="-1.000")]
    with pytest.raises(DerivativesSourceError) as excinfo:
        read_metrics(_archive(tmp_path, rows), ARCHIVE)

    message = str(excinfo.value)
    assert ARCHIVE.name in message, "the refusal names the archive"
    assert "2020-09-01T00:05:00+00:00" in message, "and the row"
    assert "sum_open_interest" in message and "negative" in message
    assert "HARD FAIL" in message, "the refusal quotes the policy rather than paraphrasing"


def test_negative_notional_stops_the_acquisition(tmp_path):
    rows = [_row(0), _row(5, notional="-1.0E8")]
    with pytest.raises(DerivativesSourceError, match="sum_open_interest_value"):
        read_metrics(_archive(tmp_path, rows), ARCHIVE)


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "not-a-number", ""])
def test_a_non_finite_or_unparseable_consumed_value_stops_the_acquisition(tmp_path, value):
    """Infinite, NaN and unparseable all arrive here as "not a finite number".

    None of these was observed by the scan behind A4, so none has a preregistered
    meaning. Classifying one as an ordinary zero-invalid observation would be
    inventing a rule for a case nobody measured.
    """
    rows = [_row(0), _row(5, contracts=value)]
    with pytest.raises(DerivativesSourceError, match="not a\n?\\s*finite number"):
        read_metrics(_archive(tmp_path, rows), ARCHIVE)


def test_a_negative_value_is_never_counted_as_an_ordinary_zero_invalid_row(tmp_path):
    """The two cases must not collapse into one another.

    A run that meets a negative value has met something the source inspection did
    not measure, and the protocol's answer is to stop — not to fold it into the
    zero-invalid count and carry on.
    """
    assert "HARD FAIL" in A4["on_negative_or_nonfinite"]
    assert "never classified as an ordinary zero-invalid observation" in (
        A4["on_negative_or_nonfinite"]
    )
    rows = [_row(0), _both_zero(5), _row(10, contracts="-1.000")]
    with pytest.raises(DerivativesSourceError, match="negative"):
        read_metrics(_archive(tmp_path, rows), ARCHIVE)


# --- 8. provenance keeps the rejected rows visible ---------------------------
def test_invalid_rows_stay_visible_in_the_provenance_counts(tmp_path):
    """A rejected observation must not look like a row the archive never served.

    The accounting identity the policy states, checked on an archive carrying all
    three invalid shapes at once.
    """
    rows = [
        _row(0),
        _both_zero(5),
        _row(10, contracts=ZERO),
        _zero_notional(15),
        _row(20),
        _both_zero(25),
    ]
    _, _, _, normalisation, validity = _reduce(tmp_path, rows, hours=[_hour(0)])

    assert normalisation.rows_read == 6
    assert normalisation.observations_retained == 6
    assert validity.logical_observations == normalisation.observations_retained
    assert validity.valid_positive_observations == 2
    assert validity.invalid_zero_observations == 4
    assert validity.invalid_both_zero_observations == 2
    assert validity.invalid_zero_contracts_only == 1
    assert validity.invalid_zero_notional_only == 1
    assert validity.negative_observations == 0
    assert validity.nonfinite_observations == 0

    # The identity, exactly, and the partition inside it.
    assert validity.logical_observations == (
        validity.valid_positive_observations + validity.invalid_zero_observations
    )
    assert validity.invalid_zero_observations == (
        validity.invalid_both_zero_observations
        + validity.invalid_zero_contracts_only
        + validity.invalid_zero_notional_only
    )
    assert set(A4["provenance_required"]) == set(validity.to_dict())


# --- 9-10. A4 runs after A1, never before ------------------------------------
def test_conflicting_zero_rows_are_still_refused_by_the_duplicate_rule(tmp_path):
    """Proof of order: classifying first would drop these before A1 ever saw them.

    Two rows at one instant, both zero in the consumed fields, disagreeing in a
    column ``derivatives_v1`` never reads. A4 alone would call each of them an
    invalid observation and discard them quietly. A1 sees a source disagreeing
    with itself, and that still stops the acquisition.
    """
    rows = [
        _row(0),
        _both_zero(5),
        _both_zero(5, sum_taker_long_short_vol_ratio="0.9500"),
    ]
    with pytest.raises(DerivativesSourceError, match="disagree"):
        read_metrics(_archive(tmp_path, rows), ARCHIVE)


def test_identical_duplicate_zero_rows_collapse_under_a1_then_classify_once(tmp_path):
    """The published shape both amendments touch, in the order they touch it.

    A1 sees two identical rows at 00:05 and collapses them to one logical
    observation — which is what the archive published, and what its duplicate
    accounting must keep describing. A4 then classifies that one logical row
    once, as one invalid observation and not two.
    """
    rows = [_row(0), _both_zero(5), _both_zero(5), _row(10)]
    _, _, _, normalisation, validity = _reduce(tmp_path, rows, hours=[_hour(0)])

    assert normalisation.rows_read == 4, "A1 still describes what the archive published"
    assert normalisation.observations_retained == 3
    assert normalisation.exact_duplicate_rows_collapsed == 1
    assert normalisation.duplicate_instants == 1

    assert validity.logical_observations == 3, "A4 runs on what A1 left behind"
    assert validity.invalid_zero_observations == 1, "classified once, not twice"
    assert validity.invalid_both_zero_observations == 1
    assert validity.valid_positive_observations == 2


def test_the_policy_states_the_order_it_depends_on(tmp_path):
    assert "A1 exact-duplicate normalisation" in A4["applies_after"]
    assert A4["applies_after"].index("schema") < A4["applies_after"].index("duplicate")


# --- 11. an invalid row does not make a missing day ---------------------------
def test_one_invalid_row_does_not_make_its_day_a_missing_day(tmp_path):
    """A day with one rejected row is a day the archive published, in full.

    288 five-minute rows, one of them zero-valued. The rejected row costs its own
    slot and nothing else: the 02:55 observation is five minutes old at 03:00, so
    every hour of the day still has open interest inside the staleness bound. A
    §3.0a missing day is an archive that 404s, fails its checksum or arrives
    short, and this archive did none of those.
    """
    minutes = list(range(0, 24 * 60, 5))
    rows = [_both_zero(m) if m == 180 else _row(m) for m in minutes]
    hours = [_hour(h) for h in range(24)]
    available, _, _, normalisation, validity = _reduce(tmp_path, rows, hours=hours)

    assert normalisation.rows_read == len(minutes)
    assert validity.logical_observations == len(minutes)
    assert validity.invalid_zero_observations == 1
    assert available.all(), "every hour of the day still has a valid observation"

    # And the rule that decides missing days is unchanged and does not mention it.
    assert "not thereby a §3.0a missing day" in A4["missing_day_relationship"].replace(
        "NOT", "not"
    )
    for reason in ("404", "checksum", "short"):
        assert reason in source_spec()["missing_day_rule"] or reason in (
            A4["missing_day_relationship"]
        )


# --- 12-14. the causal consequence, on the observed 8.5-hour run --------------
@pytest.fixture
def long_run(tmp_path):
    """The observed worst case: 102 consecutive paired-zero five-minute rows.

    Valid observations at 07:40 and 07:45, then the run from 07:50 to 16:15
    inclusive — 510 minutes, 8.5 hours — then a valid observation at 16:20. The
    grid is the whole UTC day.
    """
    run = list(range(7 * 60 + 50, 7 * 60 + 50 + 102 * 5, 5))
    assert len(run) == 102 and run[-1] == 16 * 60 + 15
    rows = (
        [_row(7 * 60 + 40), _row(7 * 60 + 45)]
        + [_both_zero(m) for m in run]
        + [_row(16 * 60 + 20, contracts="20000.000", notional="2.0E8")]
    )
    hours = [_hour(h) for h in range(24)]
    available, age, reducer, _, validity = _reduce(tmp_path, rows, hours=hours)
    return {
        "available": available,
        "age": age,
        "reducer": reducer,
        "validity": validity,
        "run": run,
    }


def test_a_prior_valid_observation_is_visible_only_inside_the_staleness_bound(long_run):
    """07:45 is visible at 08:00 and not at 09:00. The bound is unchanged."""
    available, age = long_run["available"], long_run["age"]

    assert bool(available[8]), "08:00 is 15 minutes after the last valid observation"
    assert int(age[8]) == 15 * 60 * 1_000_000_000
    assert int(age[8]) <= staleness_bound_ns(OPEN_INTEREST)
    assert not bool(available[9]), "09:00 is 1h15m after it, past the 1-hour bound"
    assert MAX_STALENESS_HOURS[OPEN_INTEREST] == 1, "A4 does not move the bound"


def test_the_long_invalid_run_produces_unavailable_hours_rather_than_a_carry(long_run):
    """The whole point. 8.5 hours of zeros must not become 8.5 hours of 07:45."""
    available = long_run["available"]

    assert long_run["validity"].invalid_zero_observations == 102
    assert long_run["validity"].valid_positive_observations == 3
    assert not available[9:17].any(), "09:00 through 16:00 have no valid observation"
    assert int((~available[:24]).sum()) >= 8, "at least the 8 hours the run covers"
    # And not because the observation was replaced by a zero: the reducer's last
    # state through the run is still the real 07:45 one, and availability is what
    # removes it.
    reducer = long_run["reducer"]
    assert reducer.values[12].tolist() == [10000.0, 1.0e8]
    assert int(reducer.at[12]) == _instant(7 * 60 + 45)


def test_a_later_valid_observation_restores_availability_only_forwards(long_run):
    """16:20 is visible at 17:00 and invisible at 16:00. Never backwards."""
    available, reducer = long_run["available"], long_run["reducer"]

    assert bool(available[17]), "17:00 sees the 16:20 observation"
    assert reducer.values[17].tolist() == [20000.0, 2.0e8]
    assert not bool(available[16]), "16:00 precedes it and stays unavailable"
    assert int(reducer.at[16]) == _instant(7 * 60 + 45), "no future row reached back"
    assert "PROSPECTIVELY" in A4["later_valid_observation"]


# --- 15. nothing is repaired -------------------------------------------------
def test_no_invalid_row_is_interpolated_substituted_or_backfilled(long_run):
    """Only valid instants ever entered the causal sequence.

    Stated as an equality over what the reducer saw rather than as a list of
    things that did not happen: an interpolation, a REST substitution, a carried
    zero or a backfill from 16:20 would each put an instant or a value in here
    that no valid observation published.
    """
    reducer = long_run["reducer"]
    seen = set(np.unique(reducer.at[reducer.at >= 0]).tolist())
    valid = {_instant(7 * 60 + 40), _instant(7 * 60 + 45), _instant(16 * 60 + 20)}

    assert seen <= valid, "every hour resolves to a valid observation or to none"
    # The two an hourly grid can land on: 07:40 is superseded by 07:45 before 08:00.
    assert seen == {_instant(7 * 60 + 45), _instant(16 * 60 + 20)}
    for minute in long_run["run"]:
        assert _instant(minute) not in seen
    # No zero was ever written as a state, either: every value the reducer holds
    # on an hour it has an observation for is one of the two real ones.
    held = reducer.values[reducer.at >= 0]
    assert {tuple(row) for row in held.tolist()} == {(10000.0, 1.0e8), (20000.0, 2.0e8)}
    for forbidden in ("interpolat", "averag", "REST", "future observation"):
        assert any(forbidden in entry for entry in A4["invalid_row_handling"]), forbidden


# --- 16. the features are untouched ------------------------------------------
def test_the_open_interest_feature_definitions_and_windows_are_unchanged():
    """A4 decides which rows are observations. It does not touch what is computed."""
    by_name = {feature["name"]: feature for feature in FEATURES}
    assert by_name["drv_oi_log_change_24h"]["window"] == 24
    assert by_name["drv_oi_log_change_24h"]["clip"] == [-1.0, 1.0]
    assert by_name["drv_oi_notional_ratio"]["window"] == 168
    assert by_name["drv_oi_notional_ratio"]["clip"] == [0.0, 10.0]
    assert by_name["drv_oi_price_divergence"]["window"] == 24
    assert by_name["drv_oi_price_divergence"]["clip"] == [-1.0, 1.0]
    assert "No epsilon is added to open interest" in A4["windows_unchanged"]
    assert "no zero is replaced by a tiny positive value" in A4["windows_unchanged"]


def test_the_amendment_asserts_nothing_about_the_true_open_interest():
    """The one claim A4 must not make, checked as a claim it does not make."""
    assert "does not infer the true economic open interest" in A4["no_economic_inference"]
    assert "no claim is made" in A4["no_sentinel_claim"].lower()
    assert "sentinel" in A4["no_sentinel_claim"]
    # The wording the amendment is required to avoid, checked as an absence. It
    # would assert a cause nobody has documented.
    text = " ".join(str(value) for value in A4.values()).lower()
    assert "binance outage sentinel" not in text
    assert "published zero-valued observation" in A4["exactly_zero_is_invalid"]
