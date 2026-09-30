"""The 30-day coverage gate, against records this file writes and can break.

The gate is a pure function of the reconciliation records on disk, so the
material here is those records — written directly, one JSON document per day, so
that a test can put a day exactly on a threshold, remove one from the middle of
a streak, or corrupt one in a single field. Building them through the
reconciliation instead would have made every arithmetic test depend on a fake
venue and would have made "0.995 passes" impossible to state exactly.

**The two thresholds are pinned exactly, and pinned at the predicate.** ``1440``
admits no whole number of minutes at either boundary — ``1440 * 0.995`` is
``1432.8`` and ``1440 * 0.990`` is ``1425.6`` — so a test that could only speak
in minutes could never assert the numbers the specification writes down. The
predicates take a numerator and a denominator for exactly that reason, and the
day-level tests then pin the behaviour at a real day's scale on both sides.

**Two-sided everywhere.** Every failing case below has a passing partner one
count away, every broken record has the intact one it was made from, and the
gate's pass is asserted as well as its refusals — a gate that never passed would
satisfy half of this file.

**One test closes the loop.** :func:`test_a_record_the_reconciliation_wrote_is_a_record_the_gate_reads`
runs the real reconciliation over synthetic archives, persists what it produced,
and judges it here. Without it, this file and
``tests/test_recorder_reconcile.py`` could each be internally consistent about a
different document.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.coverage import (
    DAY_FAIL,
    DAY_MISSING,
    DAY_PASS,
    DAY_UNJUDGEABLE,
    FUNDING_SCHEDULE_UNAVAILABLE,
    GATE_BOUNDARY_UNSET,
    GATE_FAIL,
    GATE_PASS,
    MAX_OUTAGE_FLAGGED_DAYS,
    RECONCILIATION_SCHEMA,
    RECORDER_OUTAGE,
    RecorderCoverageError,
    available_reconciliation_days,
    coverage_for_day,
    gate,
    gate_path,
    published_coverage_passes,
    reconciliation_path,
    wallclock_flags_outage,
    write_gate,
)
from chimera.recorder.events import MINUTES_PER_DAY

CONTRACT = load_recorder_contract()
BOUNDARY = datetime(2026, 10, 1, tzinfo=timezone.utc)
ACTIVATED = CONTRACT.with_prospective_from(BOUNDARY)
FIRST_DAY = "2026-10-01"


def by_stream(coverage) -> dict:
    """One day's stream coverages, looked up by name rather than by position."""
    return {entry.stream: entry for entry in coverage.streams}


def day_at(offset: int, first: str = FIRST_DAY) -> str:
    return (date.fromisoformat(first) + timedelta(days=offset)).isoformat()


def stream_entry(
    *,
    published: int = MINUTES_PER_DAY,
    agreeing: int = MINUTES_PER_DAY,
    judged: bool = True,
    reason: str | None = None,
) -> dict:
    """One stream's section of a record: only the fields the gate reads."""
    return {
        "index_kind": "minute",
        "judged": judged,
        "reason": reason,
        "published_minutes": published,
        "agreeing_minutes": agreeing,
    }


def funding_entry(
    *,
    established: bool = True,
    complete: bool = True,
    outcome: str = "OK",
    scheduled: int = 3,
    captured: int = 3,
) -> dict:
    return {
        "index_kind": "settlement",
        "schedule_established": established,
        "funding_complete": complete,
        "outcome": outcome,
        "scheduled": scheduled,
        "captured": captured,
    }


def record(
    day: str,
    *,
    contract=ACTIVATED,
    streams: dict | None = None,
    funding: dict | None = None,
) -> dict:
    """A whole record, judged and passing unless a caller says otherwise."""
    return {
        "reconciliation_schema": RECONCILIATION_SCHEMA,
        "day": day,
        "contract_id": contract.contract_id,
        "contract_hash": contract.contract_hash,
        "prospective_from": (
            None
            if contract.prospective_from is None
            else contract.prospective_from.isoformat()
        ),
        "streams": (
            streams
            if streams is not None
            else {name: stream_entry() for name in contract.minute_indexed_required()}
        ),
        "funding": funding if funding is not None else funding_entry(),
    }


def write_record(root: Path, day: str, **kwargs) -> Path:
    path = reconciliation_path(root, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record(day, **kwargs), indent=2, sort_keys=True), "utf-8")
    return path


def write_streak(root: Path, days: int, *, first: str = FIRST_DAY, **kwargs) -> list[str]:
    written = [day_at(offset, first) for offset in range(days)]
    for day in written:
        write_record(root, day, **kwargs)
    return written


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "root"


# --- A. the two thresholds, exactly ------------------------------------------------
def test_published_coverage_passes_at_0_995_and_fails_at_0_9949():
    """The specification's two numbers, decided in integers rather than in floats."""
    assert published_coverage_passes(995, 1000) is True, "exactly 0.995 passes"
    assert published_coverage_passes(9950, 10000) is True
    assert published_coverage_passes(9949, 10000) is False, "0.9949 fails"
    assert published_coverage_passes(994, 1000) is False
    assert published_coverage_passes(1000, 1000) is True
    assert published_coverage_passes(0, 1000) is False


def test_published_coverage_refuses_a_denominator_of_zero():
    """No archive minute is no denominator, which is neither a zero nor a pass."""
    with pytest.raises(RecorderCoverageError, match="unjudgeable"):
        published_coverage_passes(0, 0)


def test_the_outage_threshold_is_not_flagged_at_exactly_0_990():
    assert wallclock_flags_outage(990, 1000) is False, "exactly 0.990 is not an outage"
    assert wallclock_flags_outage(9899, 10000) is True, "just below 0.990 is"
    assert wallclock_flags_outage(9900, 10000) is False


def test_the_outage_threshold_at_a_real_day_s_scale():
    """``1440 * 0.99`` is ``1425.6``, so the first flagged minute count is 1425."""
    assert wallclock_flags_outage(1426) is False
    assert wallclock_flags_outage(1425) is True
    assert wallclock_flags_outage(MINUTES_PER_DAY) is False


# --- B. one day --------------------------------------------------------------------
def test_a_complete_day_passes(root):
    write_record(root, FIRST_DAY)
    coverage = coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)
    assert coverage.verdict == DAY_PASS
    assert coverage.passed is True
    assert coverage.outage_flagged is False
    assert coverage.reasons == ()
    assert set(by_stream(coverage)) == set(ACTIVATED.minute_indexed_required())


def test_one_stream_below_the_bar_fails_the_whole_day(root):
    """Exactly at 0.995 the day passes; one agreeing minute fewer and it does not."""
    for agreeing, expected in ((995, DAY_PASS), (994, DAY_FAIL)):
        streams = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
        streams["spot.kline_1m"] = stream_entry(published=1000, agreeing=agreeing)
        write_record(root, FIRST_DAY, streams=streams)
        coverage = coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)
        assert coverage.verdict == expected, agreeing
        spot = by_stream(coverage)["spot.kline_1m"]
        assert spot.published_coverage == agreeing / 1000
        assert spot.passes is (expected == DAY_PASS)


def test_a_day_can_pass_and_still_be_flagged_as_a_recorder_outage(root):
    """The flag is not a failure. Three of them in the window are."""
    streams = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
    streams["um.kline_1m"] = stream_entry(published=1430, agreeing=1425)
    write_record(root, FIRST_DAY, streams=streams)
    coverage = coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)
    assert coverage.verdict == DAY_PASS
    assert coverage.outage_flagged is True
    assert coverage.to_dict()["flags"] == [RECORDER_OUTAGE]

    streams["um.kline_1m"] = stream_entry(published=1430, agreeing=1426)
    write_record(root, FIRST_DAY, streams=streams)
    assert coverage_for_day(root, FIRST_DAY, contract=ACTIVATED).outage_flagged is False


def test_an_unjudged_stream_makes_the_day_unjudgeable_rather_than_failed(root):
    streams = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
    streams["um.markPrice"] = stream_entry(judged=False, reason="ABSENT: not published")
    write_record(root, FIRST_DAY, streams=streams)
    coverage = coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)
    assert coverage.verdict == DAY_UNJUDGEABLE
    assert coverage.passed is False
    assert coverage.outage_flagged is False, (
        "a day with no denominator has no wall-clock coverage either, so it must not "
        "enter the three-flagged-days count"
    )
    assert "ABSENT" in coverage.reasons[0]


def test_a_published_set_of_zero_minutes_is_unjudgeable_and_never_a_vacuous_pass(root):
    streams = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
    streams["um.kline_1m"] = stream_entry(published=0, agreeing=0)
    write_record(root, FIRST_DAY, streams=streams)
    coverage = coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)
    assert coverage.verdict == DAY_UNJUDGEABLE
    entry = by_stream(coverage)["um.kline_1m"]
    assert entry.published_coverage is None
    assert entry.judged is False
    assert "no denominator" in entry.reason


def test_a_schedule_that_could_not_be_established_neither_passes_nor_flags(root):
    """Amendments A2 and A9: missing evidence is not a zero and is not an outage."""
    write_record(
        root,
        FIRST_DAY,
        funding=funding_entry(
            established=False,
            complete=False,
            outcome=FUNDING_SCHEDULE_UNAVAILABLE,
            scheduled=0,
            captured=0,
        ),
    )
    coverage = coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)
    assert coverage.verdict == DAY_UNJUDGEABLE
    assert coverage.schedule_established is False
    assert coverage.funding_outcome == FUNDING_SCHEDULE_UNAVAILABLE
    assert coverage.scheduled_settlements is None, "an unestablished schedule has no count"
    assert coverage.outage_flagged is False
    assert FUNDING_SCHEDULE_UNAVAILABLE in coverage.reasons[0]


def test_an_established_empty_schedule_passes_the_day(root):
    """The universal holds over the empty set, and no quotient is formed."""
    write_record(root, FIRST_DAY, funding=funding_entry(scheduled=0, captured=0))
    coverage = coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)
    assert coverage.verdict == DAY_PASS
    assert coverage.schedule_established is True
    assert coverage.scheduled_settlements == 0
    assert coverage.captured_settlements == 0


def test_a_missing_scheduled_settlement_fails_the_day_outright(root):
    write_record(root, FIRST_DAY, funding=funding_entry(complete=False, captured=2))
    coverage = coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)
    assert coverage.verdict == DAY_FAIL
    assert coverage.outage_flagged is False


def test_completeness_without_an_established_schedule_is_refused(root):
    write_record(root, FIRST_DAY, funding=funding_entry(established=False, complete=True))
    with pytest.raises(RecorderCoverageError, match="quantifier"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


def test_funding_is_never_divided_by_1440(root):
    """It is settlement-indexed: no wall-clock coverage, and the 0.990 bar is not its.

    A day with three of three settlements captured passes on funding while its
    settlement count is a thousandth of the minute-stream denominator. If the
    gate divided it by 1440 the day could not pass at all, which is the exact
    arithmetic failure amendment A1 corrected.
    """
    write_record(root, FIRST_DAY)
    coverage = coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)
    assert coverage.verdict == DAY_PASS
    assert coverage.captured_settlements == 3
    assert "um.funding" not in {entry.stream for entry in coverage.streams}
    assert (
        wallclock_flags_outage(3) is True
    ), "three settlements out of 1440 would be flagged if the rule applied to funding"


def test_the_gated_streams_are_read_from_the_contract(root):
    """Amendment A5: remove one from the contract and the gate stops counting it."""
    streams = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
    streams["um.markPrice"] = stream_entry(published=1000, agreeing=1)
    write_record(root, FIRST_DAY, streams=streams)
    assert coverage_for_day(root, FIRST_DAY, contract=ACTIVATED).verdict == DAY_FAIL

    without_mark = dataclasses.replace(
        ACTIVATED,
        required_for_coverage=tuple(
            name for name in ACTIVATED.required_for_coverage if name != "um.markPrice"
        ),
    )
    write_record(root, FIRST_DAY, contract=without_mark, streams=streams)
    coverage = coverage_for_day(root, FIRST_DAY, contract=without_mark)
    assert coverage.verdict == DAY_PASS
    assert "um.markPrice" not in {entry.stream for entry in coverage.streams}


def test_a_stream_the_contract_requires_and_the_record_omits_is_unjudgeable(root):
    streams = {
        name: stream_entry()
        for name in ACTIVATED.minute_indexed_required()
        if name != "spot.kline_1m"
    }
    write_record(root, FIRST_DAY, streams=streams)
    with pytest.raises(RecorderCoverageError, match="says nothing about required stream"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


# --- C. malformed evidence fails closed ----------------------------------------------
def test_a_record_that_is_not_json_is_refused(root):
    write_record(root, FIRST_DAY)
    assert coverage_for_day(root, FIRST_DAY, contract=ACTIVATED).verdict == DAY_PASS
    reconciliation_path(root, FIRST_DAY).write_text("{ truncated", encoding="utf-8")
    with pytest.raises(RecorderCoverageError, match="not readable JSON"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


def test_a_record_from_another_schema_is_refused(root):
    path = write_record(root, FIRST_DAY)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["reconciliation_schema"] = "chimera.recorder-reconciliation/2"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RecorderCoverageError, match="schema"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


def test_a_record_of_another_day_is_refused(root):
    path = write_record(root, FIRST_DAY)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["day"] = day_at(1)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RecorderCoverageError, match="record of day"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


def test_a_record_written_under_another_contract_hash_is_refused(root):
    """A storage root never mixes hashes, and judging across one would relabel it."""
    write_record(root, FIRST_DAY, contract=CONTRACT)
    with pytest.raises(RecorderCoverageError, match="contract hash"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


@pytest.mark.parametrize(
    "field,value",
    [
        ("published_minutes", -1),
        ("published_minutes", "1440"),
        ("agreeing_minutes", None),
        ("agreeing_minutes", True),
        ("judged", "yes"),
    ],
)
def test_a_record_whose_counts_are_not_counts_is_refused(root, field, value):
    streams = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
    streams["um.kline_1m"][field] = value
    write_record(root, FIRST_DAY, streams=streams)
    with pytest.raises(RecorderCoverageError):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


def test_more_agreeing_minutes_than_published_ones_is_refused(root):
    streams = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
    streams["um.kline_1m"] = stream_entry(published=1400, agreeing=1440)
    write_record(root, FIRST_DAY, streams=streams)
    with pytest.raises(RecorderCoverageError, match="cannot exceed the denominator"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


def test_a_denominator_larger_than_the_day_it_counts_is_refused(root):
    """A day holds 1440 minutes, so it cannot publish 1441. Two-sided at the boundary."""
    streams = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
    streams["um.kline_1m"] = stream_entry(published=MINUTES_PER_DAY, agreeing=MINUTES_PER_DAY)
    write_record(root, FIRST_DAY, streams=streams)
    assert coverage_for_day(root, FIRST_DAY, contract=ACTIVATED).verdict == DAY_PASS

    streams["um.kline_1m"] = stream_entry(
        published=MINUTES_PER_DAY + 1, agreeing=MINUTES_PER_DAY + 1
    )
    write_record(root, FIRST_DAY, streams=streams)
    with pytest.raises(RecorderCoverageError, match="larger than the day"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


def test_more_captured_settlements_than_scheduled_ones_is_refused(root):
    """The funding counts get the cross-check the minute counts already had."""
    write_record(root, FIRST_DAY, funding=funding_entry(scheduled=3, captured=3))
    assert coverage_for_day(root, FIRST_DAY, contract=ACTIVATED).verdict == DAY_PASS

    write_record(root, FIRST_DAY, funding=funding_entry(scheduled=2, captured=3))
    with pytest.raises(RecorderCoverageError, match="cannot exceed the schedule"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


def test_completeness_that_contradicts_the_counts_in_the_same_record_is_refused(root):
    """``funding_complete`` is every scheduled settlement captured, not a claim to trust.

    A record saying the schedule was established, that three settlements were
    scheduled, that none were captured and that the day was nonetheless complete
    is a document this reader does not understand, and taking the boolean's word
    for it would pass a day on evidence that says the opposite.
    """
    write_record(
        root, FIRST_DAY, funding=funding_entry(scheduled=3, captured=0, complete=False)
    )
    assert coverage_for_day(root, FIRST_DAY, contract=ACTIVATED).verdict == DAY_FAIL

    write_record(
        root, FIRST_DAY, funding=funding_entry(scheduled=3, captured=0, complete=True)
    )
    with pytest.raises(RecorderCoverageError, match="claims completeness"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


def test_a_missing_record_is_refused_rather_than_defaulted(root):
    with pytest.raises(RecorderCoverageError, match="no reconciliation record"):
        coverage_for_day(root, FIRST_DAY, contract=ACTIVATED)


def test_a_file_that_is_not_a_day_is_not_a_day_record(root):
    write_record(root, FIRST_DAY)
    (reconciliation_path(root, FIRST_DAY).parent / "notes.json").write_text("{}", "utf-8")
    assert available_reconciliation_days(root) == [FIRST_DAY]


# --- D. the gate --------------------------------------------------------------------
def test_thirty_consecutive_passing_days_pass_the_gate(root):
    days = write_streak(root, 30)
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.verdict == GATE_PASS
    assert verdict.gate_passed is True
    assert verdict.official is True
    assert verdict.streak == 30
    assert list(verdict.window_days) == days
    assert verdict.outage_flagged_days == ()
    assert verdict.prospective_from == FIRST_DAY


def test_twenty_nine_days_do_not(root):
    write_streak(root, 29)
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.verdict == GATE_FAIL
    assert verdict.gate_passed is False
    assert verdict.streak == 29
    assert verdict.current_streak == 29
    assert verdict.window_days == ()
    assert "is 29 of 30" in verdict.reasons[0]


def test_a_missing_day_breaks_the_streak(root):
    """A hole splits one run of 31 into two, and neither half reaches the window."""
    write_streak(root, 31)
    reconciliation_path(root, day_at(10)).unlink()
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.verdict == GATE_FAIL
    assert verdict.streak == 20, "the longer of the two runs the hole left"
    assert verdict.current_streak == 20, "and it is the one ending at the newest day"
    assert {entry.day: entry.verdict for entry in verdict.days}[day_at(10)] == DAY_MISSING


def test_a_failed_day_resets_the_count_to_zero(root):
    write_streak(root, 30)
    write_record(root, day_at(29), funding=funding_entry(complete=False, captured=1))
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.current_streak == 0, "the newest day did not pass, so nothing is running"
    assert verdict.streak == 29, "and the run it ended was one day short of the window"
    assert verdict.verdict == GATE_FAIL


def test_a_run_that_reached_the_window_is_not_undone_by_a_later_unjudgeable_day(root):
    """Amendment A9's expected evidence latency must not un-claim an achieved gate.

    The funding schedule source is a monthly object that does not exist while its
    month is open, so every day of the current month reconciles to
    FUNDING_SCHEDULE_UNAVAILABLE and is unjudgeable — and the daily job writes
    those records like any others. A gate measured only backwards from the newest
    record would therefore pass on one day and fail on the next with no recorded
    minute having changed, no threshold having moved and no recorder fault having
    occurred, and the operator's only way to hold a pass would be to stop
    reconciling. The 30 days that passed still passed.
    """
    write_streak(root, 30)
    assert gate(root, 30, contract=ACTIVATED).verdict == GATE_PASS
    unavailable = funding_entry(
        established=False, complete=False, outcome=FUNDING_SCHEDULE_UNAVAILABLE
    )
    for offset in (30, 31):
        write_record(root, day_at(offset), funding=unavailable)
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.verdict == GATE_PASS, "the demonstrated window is still on the disk"
    assert verdict.streak == 30
    assert verdict.current_streak == 0, "and the reader still sees that nothing is running"
    assert list(verdict.window_days) == [day_at(offset) for offset in range(30)]
    latest = {entry.day: entry.verdict for entry in verdict.days}
    assert latest[day_at(31)] == DAY_UNJUDGEABLE, "the tail is reported, not hidden"


def test_a_window_that_never_qualified_does_not_become_one_by_deleting_records(root):
    """The two-sided partner: the rule is existential over the calendar, not over files.

    Twenty-nine passing days followed by a failure is not a qualifying window,
    and it does not become one because a later run of days exists somewhere else
    on the disk. Every candidate window is a run of consecutive calendar days.
    """
    write_streak(root, 29)
    write_record(root, day_at(29), funding=funding_entry(complete=False, captured=1))
    write_streak(root, 29, first=day_at(30))
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.verdict == GATE_FAIL
    assert verdict.streak == 29, "the longest run is still one day short"
    assert verdict.window_days == ()


def test_a_malformed_day_inside_the_window_breaks_the_streak_and_does_not_raise(root):
    write_streak(root, 30)
    reconciliation_path(root, day_at(5)).write_text("{ not json", encoding="utf-8")
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.verdict == GATE_FAIL
    assert verdict.streak == 24
    entry = {day.day: day for day in verdict.days}[day_at(5)]
    assert entry.verdict == DAY_UNJUDGEABLE
    assert "not readable JSON" in entry.reasons[0]


def test_three_flagged_days_in_the_window_fail_a_gate_every_day_of_which_passed(root):
    """The two-sided pin: two flagged days pass, and the third fails the gate."""
    flagged = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
    flagged["um.kline_1m"] = stream_entry(published=1430, agreeing=1425)

    write_streak(root, 30)
    for offset in (3, 17):
        write_record(root, day_at(offset), streams=flagged)
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.verdict == GATE_PASS
    assert len(verdict.outage_flagged_days) == MAX_OUTAGE_FLAGGED_DAYS

    write_record(root, day_at(22), streams=flagged)
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.verdict == GATE_FAIL
    assert verdict.streak == 30, "every day still passed; the flags failed the gate"
    assert list(verdict.outage_flagged_days) == [day_at(3), day_at(17), day_at(22)]
    assert "three flagged days" in verdict.reasons[0]


def test_a_thirty_first_passing_day_offers_a_window_the_three_flags_do_not_all_fall_in(root):
    """A flag fails a window it is *in*; the gate asks whether a clean window exists.

    Three flagged days inside the only candidate window fail the gate, and the
    recorder's remedy is to record more days rather than fewer: the thirty-first
    consecutive passing day offers a second candidate window, and the gate passes
    only because that window really does contain no more than two flagged days.
    Pinned in both directions, because the alternative readings are not
    hypothetical — carrying the flags forward for ever would mean a recorder had
    to *fail* a day to clear them, and counting flags outside the window would
    fail a window that meets the bar.
    """
    flagged = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
    flagged["um.kline_1m"] = stream_entry(published=1430, agreeing=1425)

    write_streak(root, 30)
    for offset in (0, 1, 2):
        write_record(root, day_at(offset), streams=flagged)
    assert gate(root, 30, contract=ACTIVATED).verdict == GATE_FAIL

    write_record(root, day_at(30))
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.verdict == GATE_PASS
    assert verdict.streak == 31
    assert list(verdict.window_days) == [day_at(offset) for offset in range(1, 31)]
    assert list(verdict.outage_flagged_days) == [
        day_at(1),
        day_at(2),
    ], "the window that qualified carries two flags, and both are inside it"


def test_a_fourth_flagged_day_fails_every_window_a_longer_streak_offers(root):
    """The flags do not simply age out: a window has to exist that carries at most two."""
    flagged = {name: stream_entry() for name in ACTIVATED.minute_indexed_required()}
    flagged["um.kline_1m"] = stream_entry(published=1430, agreeing=1425)

    write_streak(root, 33)
    for offset in (0, 10, 20, 30):
        write_record(root, day_at(offset), streams=flagged)
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.streak == 33, "every one of the 33 days passed"
    assert verdict.verdict == GATE_FAIL
    assert len(verdict.outage_flagged_days) == MAX_OUTAGE_FLAGGED_DAYS + 1
    assert "three flagged days" in verdict.reasons[0]


def test_days_before_the_boundary_do_not_count_toward_the_streak(root):
    write_streak(root, 30, first="2026-09-02")
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.streak == 1, "only 2026-10-01 is at or after the boundary"
    assert verdict.verdict == GATE_FAIL
    assert all(entry.day >= FIRST_DAY for entry in verdict.days)


def test_the_verdict_is_recomputed_and_holds_no_state(root):
    """A day that takes its real verdict later moves the gate, in both directions."""
    write_streak(root, 30)
    assert gate(root, 30, contract=ACTIVATED).verdict == GATE_PASS
    write_record(
        root,
        day_at(4),
        funding=funding_entry(
            established=False, complete=False, outcome=FUNDING_SCHEDULE_UNAVAILABLE
        ),
    )
    assert gate(root, 30, contract=ACTIVATED).verdict == GATE_FAIL
    write_record(root, day_at(4))
    assert (
        gate(root, 30, contract=ACTIVATED).verdict == GATE_PASS
    ), "the archive arrived late and the day took its real verdict; nothing had to be reset"


def test_an_unset_boundary_can_never_produce_an_official_pass(root):
    """Whatever the records say. Reconciliation while null is engineering work."""
    write_streak(root, 40, contract=CONTRACT)
    verdict = gate(root, 30, contract=CONTRACT)
    assert verdict.verdict == GATE_BOUNDARY_UNSET
    assert verdict.gate_passed is False
    assert verdict.official is False
    assert verdict.prospective_from is None
    assert "no streak of engineering days is an S1 pass" in verdict.reasons[0]
    document = verdict.to_dict()
    assert document["verdict"] == GATE_BOUNDARY_UNSET
    assert document["gate_passed"] is False
    assert document["official"] is False
    assert CONTRACT.prospective_from is None, "the committed contract is still unactivated"


def test_the_committed_contract_is_the_unactivated_one():
    """A guard on this file's own premise: activation happens here and nowhere else."""
    assert CONTRACT.prospective_from is None
    assert CONTRACT.activated is False
    assert ACTIVATED.contract_hash != CONTRACT.contract_hash


def test_the_verdict_is_written_where_gitignore_re_includes_it(root):
    write_streak(root, 30)
    path = write_gate(root, gate(root, 30, contract=ACTIVATED))
    assert path == gate_path(root)
    assert path.parent.name == "coverage" and path.name == "GATE.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["verdict"] == GATE_PASS
    assert document["contract_hash"] == ACTIVATED.contract_hash
    assert len(document["days"]) == 30
    for economic in ("return", "pnl", "profit", "basis", "carry", "alpha"):
        assert economic not in json.dumps(document["days"]).lower()

    # Section 4.9 names what this file carries: the verdict, the streak, the
    # per-stream coverages, schedule_established, scheduled_settlements and
    # captured_settlements, and the list of days. It is the artefact a reviewer
    # reads to check the claim without re-running anything, so a day entry that
    # carried only a verdict would send them to thirty separate reconciliation
    # records for the numbers that decided it.
    for entry in document["days"]:
        assert entry["schedule_established"] is True
        assert entry["scheduled_settlements"] == 3
        assert entry["captured_settlements"] == 3
        assert {stream["stream"] for stream in entry["streams"]} == set(
            ACTIVATED.minute_indexed_required()
        )
        for stream in entry["streams"]:
            assert stream["published_minutes"] == MINUTES_PER_DAY
            assert stream["published_coverage"] == 1.0
            assert stream["wallclock_coverage"] == 1.0
            assert stream["passes"] is True


def test_a_day_the_gate_could_not_judge_carries_no_invented_numbers(root):
    """The other side of it: absent evidence is written as null, never as a zero."""
    write_streak(root, 3)
    reconciliation_path(root, day_at(1)).unlink()
    document = json.loads(
        write_gate(root, gate(root, 30, contract=ACTIVATED)).read_text(encoding="utf-8")
    )
    entry = {day["day"]: day for day in document["days"]}[day_at(1)]
    assert entry["verdict"] == DAY_MISSING
    assert entry["streams"] is None
    assert entry["schedule_established"] is None
    assert entry["scheduled_settlements"] is None
    assert entry["captured_settlements"] is None


def test_a_window_shorter_than_one_day_is_refused(root):
    with pytest.raises(RecorderCoverageError, match="at least one day"):
        gate(root, 0, contract=ACTIVATED)


def test_an_empty_root_is_a_failure_and_not_a_crash(root):
    verdict = gate(root, 30, contract=ACTIVATED)
    assert verdict.verdict == GATE_FAIL
    assert verdict.streak == 0
    assert verdict.days == ()


# --- E. the writer and the reader agree ------------------------------------------------
def test_a_record_the_reconciliation_wrote_is_a_record_the_gate_reads(tmp_path):
    """The loop closed: the real reconciliation's document, judged by the real gate.

    Both files could otherwise be internally consistent about different
    documents — this one fails the moment the writer and the reader disagree
    about a field name, a type or a schema string.
    """
    from tests.test_recorder_reconcile import CONTRACT as RECONCILE_CONTRACT
    from tests.test_recorder_reconcile import FakeVenue, publish_day
    from tests.recorder_synthetic import DAY, funding_day, spot_day, um_day
    from chimera.recorder.events import UM_FUNDING
    from chimera.recorder.normalize import MinuteNormalizer
    from chimera.recorder.reconcile import reconcile_day, write_reconciliation
    from chimera.recorder.sink import RawSink

    storage = RECONCILE_CONTRACT.storage_root(tmp_path / "data")
    material = {**um_day(range(3)), **spot_day(range(3)), UM_FUNDING: funding_day(DAY)}
    for stream, events in material.items():
        with RawSink(storage, stream, contract=RECONCILE_CONTRACT) as sink:
            for event in events:
                sink.append(event)
            sink.sync()
    normalizer = MinuteNormalizer(storage, RECONCILE_CONTRACT)
    for market in RECONCILE_CONTRACT.market_keys():
        normalizer.build_day(market, DAY)
    normalizer.build_settlements("um")

    report = reconcile_day(storage, DAY, publish_day(FakeVenue()), contract=RECONCILE_CONTRACT)
    write_reconciliation(storage, report)

    coverage = coverage_for_day(storage, DAY, contract=RECONCILE_CONTRACT)
    assert coverage.funding_complete is True
    assert coverage.schedule_established is True
    assert {entry.stream for entry in coverage.streams} == set(
        RECONCILE_CONTRACT.minute_indexed_required()
    )
    for entry in coverage.streams:
        assert entry.judged is True
        assert entry.published == 3 and entry.agreeing == 3
        assert entry.passes is True
    assert coverage.verdict == DAY_PASS, "three published minutes, all three agreeing"
    assert coverage.outage_flagged is True, (
        "three captured minutes out of 1440 is a real recorder outage on a real day, and "
        "the synthetic day is deliberately not a full one"
    )

    verdict = gate(storage, 30, contract=RECONCILE_CONTRACT)
    assert verdict.verdict == GATE_BOUNDARY_UNSET, "the committed contract is unactivated"
    assert verdict.gate_passed is False
