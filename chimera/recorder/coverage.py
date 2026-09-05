"""The 30-day coverage gate, computed from the reconciliation records and nothing else.

This module is the reader of record. The archive reconciliation writes one
deterministic JSON document per UTC day; everything here is a pure function of
those documents and of the committed contract, recomputed from scratch on every
call. There is no streak file, no counter, no cursor and no cached verdict
anywhere: a day whose funding archive was published late takes its real verdict
the moment its record is rewritten, and a gate that had been claimed on a stale
count could never be un-claimed. Recomputing is cheap; a hidden accumulator that
disagrees with the evidence is not.

**Pure and offline, which is why it lives with the offline core.** Nothing here
opens a socket, names an endpoint, holds a credential or reads a clock. The
acquisition lives in :mod:`chimera.recorder.reconcile`, which is held to the
acquiring layer's barrier instead; this module only ever reads what that one
persisted. That split is what lets the gate's arithmetic be audited without
trusting anything about a network, and ``tests/test_recorder_no_network.py``
asserts it about the source rather than taking this paragraph's word for it.

**It computes nothing economic.** Two counts divided by a third, compared
against two frozen thresholds. No price, no return, no funding flow, no basis,
no profit: coverage is a statement about how much of what the venue published
this recorder actually holds, and about nothing else.

**The arithmetic is integer arithmetic.** ``published_coverage >= 0.995`` is
evaluated as ``agreeing * 1000 >= published * 995`` and the outage threshold as
``agreeing * 1000 < minutes * 990``, so the two frozen boundaries are decided
exactly rather than by a float comparison that is a few ulps away from the
number the specification writes down. The float ratios are computed too, and
they are reported — but nothing is decided by them.

**The two index kinds are counted in different units** (amendment A1). Minute-
indexed streams are divided by their published-minute count and by 1440;
``um.funding`` is settlement-indexed, has no wall-clock coverage, is never
divided by 1440, and the ``0.990`` outage threshold does not apply to it. Which
streams are of which kind is read from
:meth:`chimera.recorder.contract.RecorderContract.minute_indexed_required` and
is never a literal here (amendment A5), so a change to the required set cannot
leave a stale partition behind in this module.

**Funding completeness is a quantifier, not a quotient** (amendment A2). A day
whose established schedule is empty is complete because the universal holds over
the empty set; a schedule that could not be established is not an empty schedule
and never becomes one, is reported ``FUNDING_SCHEDULE_UNAVAILABLE``, does not
pass, and is deliberately **not** a ``RECORDER_OUTAGE`` flag: it is missing
evidence rather than a recorder fault, and a day that cannot be judged is not a
day that passed (amendment A9).

**The gate is a window that exists, not a window that is still running.** Section
4.9 asks for 30 consecutive passing UTC days counted from the first passing day
at or after the boundary, with three flagged days *in a window* failing it. So a
qualifying window is 30 consecutive days at or after the boundary, all of them
passing, with at most two flagged — and the gate passes exactly when the records
hold one. Measuring only the run that ends at the newest record would make an
honestly achieved gate un-claim itself as soon as the next day's record landed,
which under amendment A9 is guaranteed to happen: the current month's funding
archive does not exist yet, so every day of it is unjudgeable, and the operator's
only way to keep a pass would be to stop reconciling. Recording more days can
never take a demonstrated period away.

**Everything fails closed.** A missing record, a record from another contract, a
record whose schema this build does not know, a record whose counts are absurd,
a stream with no published denominator: every one of them makes the day
unjudgeable, and an unjudgeable day does not pass and breaks the streak. There
is no path through this module on which absent or damaged evidence produces a
pass.

**And while ``prospective_from`` is null, nothing official can pass at all.** The
committed contract carries no boundary, so no recorded minute is scientific
evidence and no streak of engineering days is an S1 claim. The verdict in that
state is :data:`BOUNDARY_UNSET`, ``gate_passed`` is false, and the written
``coverage/GATE.json`` says so in three separate fields rather than leaving a
reader to infer it from a number that happens to be 30.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from chimera.recorder.contract import (
    GEN3_CONTRACT_ID,
    RecorderContract,
    load_recorder_contract,
)
from chimera.recorder.events import MINUTES_PER_DAY, NS_PER_DAY, day_start_ns, utc_day
from chimera.recorder.sink import RecorderSinkError, require_day, write_json_atomic

#: Names the shape of one day's reconciliation record. The reconciliation writes
#: it and this module reads it, and the constant lives here — with the reader
#: that has to fail closed on a document it does not understand — so that there
#: is one definition of the record's identity rather than a writer's and a
#: reader's that can drift apart.
RECONCILIATION_SCHEMA = "chimera.recorder-reconciliation/1"

#: Where those records live under the storage root. ``.gitignore`` re-includes
#: exactly this directory's JSON: the records are claims *about* the data and
#: are small enough to review as commits, while the data itself stays ignored.
RECONCILIATION_DIRECTORY = "reconciliation"

#: Names the shape of the gate verdict document.
COVERAGE_GATE_SCHEMA = "chimera.recorder-coverage-gate/1"

#: Where the verdict is written, and its file name. One file, overwritten
#: atomically, because the verdict is a function of the records and keeping a
#: history of stale verdicts would invite reading the wrong one.
COVERAGE_DIRECTORY = "coverage"
GATE_FILE = "GATE.json"

#: ``published_coverage(s, D) >= 0.995``, as an exact integer comparison.
PUBLISHED_COVERAGE_NUMERATOR = 995
PUBLISHED_COVERAGE_DENOMINATOR = 1000

#: ``wallclock_coverage(s, D) < 0.990`` flags ``RECORDER_OUTAGE``.
WALLCLOCK_OUTAGE_NUMERATOR = 990
WALLCLOCK_OUTAGE_DENOMINATOR = 1000

#: Three flagged days in the window fail the gate, so two are survivable and the
#: third is not.
MAX_OUTAGE_FLAGGED_DAYS = 2

#: The default streak the S1 gate requires.
DEFAULT_WINDOW = 30

#: The flag a day earns when a minute-indexed required stream's wall-clock
#: coverage falls below the threshold. It is a flag and not a failure: the day
#: can still pass, and three flagged days in the window fail the gate.
RECORDER_OUTAGE = "RECORDER_OUTAGE"

#: What a day is reported as when its funding schedule could not be established.
#: Neither a pass nor an outage flag — see the module docstring.
FUNDING_SCHEDULE_UNAVAILABLE = "FUNDING_SCHEDULE_UNAVAILABLE"

#: The four states a day can be in. ``UNJUDGEABLE`` and ``MISSING`` are kept
#: apart from ``FAIL`` because "the recorder missed too much" and "there is no
#: evidence either way" are different findings, and only the first is a
#: statement about the recorder.
DAY_PASS = "PASS"
DAY_FAIL = "FAIL"
DAY_MISSING = "MISSING"
DAY_UNJUDGEABLE = "UNJUDGEABLE"

#: The three gate verdicts. ``BOUNDARY_UNSET`` is not a failure of the recorder
#: and is not a pass: it says the question the gate answers has not been asked
#: yet, because no prospective boundary has been committed.
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_BOUNDARY_UNSET = "BOUNDARY_UNSET"


class RecorderCoverageError(RuntimeError):
    """A day's evidence cannot be read as a verdict about that day."""


def published_coverage_passes(agreeing: int, published: int) -> bool:
    """Whether ``agreeing / published >= 0.995``, decided in integers.

    ``published == 0`` is not a coverage of zero and not a coverage of one: it
    is the absence of a denominator, and the caller must have refused the day
    before reaching here. Refused again rather than answered, because either
    answer would be an invention.
    """
    if published <= 0:
        raise RecorderCoverageError(
            f"published minute count {published} cannot be a coverage denominator; a day "
            "whose archive publishes no minute is unjudgeable, never a coverage of zero "
            "and never a vacuous pass"
        )
    return (
        agreeing * PUBLISHED_COVERAGE_DENOMINATOR >= published * PUBLISHED_COVERAGE_NUMERATOR
    )


def wallclock_flags_outage(agreeing: int, minutes: int = MINUTES_PER_DAY) -> bool:
    """Whether ``agreeing / minutes < 0.990``, decided in integers.

    ``minutes`` is a parameter only so that the threshold itself can be pinned
    at exactly ``0.990`` by a test — ``1440 * 0.99`` is ``1425.6`` and no whole
    number of minutes lands on the boundary, so a test that could only speak in
    minutes could never assert what the specification writes down.
    """
    if minutes <= 0:
        raise RecorderCoverageError(
            f"wall-clock denominator {minutes} is not a day of minutes"
        )
    return agreeing * WALLCLOCK_OUTAGE_DENOMINATOR < minutes * WALLCLOCK_OUTAGE_NUMERATOR


def reconciliation_path(root: str | Path, day: str) -> Path:
    """Where one day's reconciliation record lives under a storage root."""
    return Path(root) / RECONCILIATION_DIRECTORY / f"{require_day(day)}.json"


def gate_path(root: str | Path) -> Path:
    """Where the gate verdict is written under a storage root."""
    return Path(root) / COVERAGE_DIRECTORY / GATE_FILE


def available_reconciliation_days(root: str | Path) -> list[str]:
    """Every day a reconciliation record exists for, sorted, ill-named files ignored.

    A file whose name is not a UTC day is not a record of one, and skipping it is
    the honest reading: it cannot be a day that passed, and treating it as a
    corrupt day would let an unrelated file break somebody's streak.
    """
    directory = Path(root) / RECONCILIATION_DIRECTORY
    if not directory.is_dir():
        return []
    days: list[str] = []
    for path in sorted(directory.glob("*.json")):
        candidate = path.name[: -len(".json")]
        try:
            require_day(candidate)
        except RecorderSinkError:
            continue
        days.append(candidate)
    return sorted(days)


def read_reconciliation(root: str | Path, day: str) -> dict[str, Any]:
    """One day's record, structurally validated, or a refusal.

    Validated on the way in rather than trusted: a document that is not JSON, is
    not an object, carries another schema, names another day or lacks the two
    sections the gate is computed from cannot be read as a verdict about this
    day, and reading it half-way would produce a number rather than an error.
    """
    path = reconciliation_path(root, day)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RecorderCoverageError(f"no reconciliation record at {path}") from exc
    except OSError as exc:
        raise RecorderCoverageError(f"{path} cannot be read: {exc}") from exc
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise RecorderCoverageError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise RecorderCoverageError(f"{path} holds {type(document).__name__}, not an object")
    schema = document.get("reconciliation_schema")
    if schema != RECONCILIATION_SCHEMA:
        raise RecorderCoverageError(
            f"{path} carries schema {schema!r}, not {RECONCILIATION_SCHEMA!r}. A record from "
            "a schema this build does not know is refused rather than read with today's "
            "field meanings"
        )
    if document.get("day") != day:
        raise RecorderCoverageError(
            f"{path} is a record of day {document.get('day')!r}, not of {day!r}"
        )
    for section in ("streams", "funding"):
        if not isinstance(document.get(section), Mapping):
            raise RecorderCoverageError(f"{path} has no {section!r} object")
    return dict(document)


@dataclass(frozen=True)
class StreamCoverage:
    """One minute-indexed required stream's coverage on one UTC day."""

    stream: str
    judged: bool
    published: int
    agreeing: int
    published_coverage: float | None
    wallclock_coverage: float | None
    passes: bool
    outage_flagged: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "judged": self.judged,
            "published_minutes": self.published,
            "agreeing_minutes": self.agreeing,
            "published_coverage": self.published_coverage,
            "wallclock_coverage": self.wallclock_coverage,
            "passes": self.passes,
            "outage_flagged": self.outage_flagged,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DayCoverage:
    """Whether one UTC day qualifies, and every number that decided it."""

    day: str
    verdict: str
    streams: tuple[StreamCoverage, ...]
    schedule_established: bool
    funding_complete: bool
    funding_outcome: str
    scheduled_settlements: int | None
    captured_settlements: int | None
    outage_flagged: bool
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True only for :data:`DAY_PASS`. Every other state is not a pass."""
        return self.verdict == DAY_PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "verdict": self.verdict,
            "passed": self.passed,
            "streams": [entry.to_dict() for entry in self.streams],
            "schedule_established": self.schedule_established,
            "funding_complete": self.funding_complete,
            "funding_outcome": self.funding_outcome,
            "scheduled_settlements": self.scheduled_settlements,
            "captured_settlements": self.captured_settlements,
            "outage_flagged": self.outage_flagged,
            "flags": [RECORDER_OUTAGE] if self.outage_flagged else [],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class GateDay:
    """One day as the gate saw it: the verdict, the flag, the numbers and why.

    The numbers are carried rather than dropped because section 4.9 names what
    the verdict document must hold — the per-stream coverages,
    ``schedule_established``, ``scheduled_settlements`` and
    ``captured_settlements`` — and ``coverage/GATE.json`` is the artefact a
    reviewer reads to check an S1 claim without re-running anything. A verdict
    file that made them open thirty separate reconciliation records to find the
    coverage that decided each day would be a verdict nobody could check in the
    form it was published in.

    :attr:`coverage` is ``None`` only when there was no verdict to compute one
    from: a day with no record, or one whose record this build refuses.
    """

    day: str
    verdict: str
    outage_flagged: bool
    reasons: tuple[str, ...]
    coverage: DayCoverage | None = None

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "day": self.day,
            "verdict": self.verdict,
            "outage_flagged": self.outage_flagged,
            "reasons": list(self.reasons),
        }
        if self.coverage is None:
            document.update(
                {
                    "streams": None,
                    "schedule_established": None,
                    "funding_complete": None,
                    "scheduled_settlements": None,
                    "captured_settlements": None,
                }
            )
            return document
        document.update(
            {
                "streams": [entry.to_dict() for entry in self.coverage.streams],
                "schedule_established": self.coverage.schedule_established,
                "funding_complete": self.coverage.funding_complete,
                "funding_outcome": self.coverage.funding_outcome,
                "scheduled_settlements": self.coverage.scheduled_settlements,
                "captured_settlements": self.coverage.captured_settlements,
            }
        )
        return document


@dataclass(frozen=True)
class GateVerdict:
    """The 30-day gate, recomputed from the records every time it is asked for."""

    verdict: str
    official: bool
    gate_passed: bool
    window: int
    #: The longest run of consecutive passing days at or after the boundary. This
    #: is the number the window is measured against.
    streak: int
    #: The run that ends at the newest day a record exists for, which is zero
    #: whenever that day did not pass. Reported beside :attr:`streak` because the
    #: two answer different questions — "has the recorder ever demonstrated the
    #: period" and "is it demonstrating one right now" — and collapsing them is
    #: what made an achieved gate un-claim itself.
    current_streak: int
    contract_id: str
    contract_hash: str
    prospective_from: str | None
    window_days: tuple[str, ...]
    outage_flagged_days: tuple[str, ...]
    days: tuple[GateDay, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_gate_schema": COVERAGE_GATE_SCHEMA,
            "verdict": self.verdict,
            "official": self.official,
            "gate_passed": self.gate_passed,
            "window": self.window,
            "streak": self.streak,
            "current_streak": self.current_streak,
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "prospective_from": self.prospective_from,
            "window_days": list(self.window_days),
            "outage_flagged_days": list(self.outage_flagged_days),
            "outage_flagged_in_window": len(self.outage_flagged_days),
            "max_outage_flagged_days": MAX_OUTAGE_FLAGGED_DAYS,
            "days": [entry.to_dict() for entry in self.days],
            "reasons": list(self.reasons),
            "note": (
                "Coverage is how much of what the venue published this recorder holds. It "
                "is not a result, not a performance statement and not a promotion: no "
                "price, return, funding flow, basis or profit is computed anywhere in this "
                "verdict."
            ),
        }


def _require_contract(contract: RecorderContract | None) -> RecorderContract:
    return load_recorder_contract(GEN3_CONTRACT_ID) if contract is None else contract


def _count(section: Mapping[str, Any], key: str, where: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecorderCoverageError(
            f"{where}.{key} is {value!r}; a count of minutes is a non-negative integer"
        )
    return value


def _flag(section: Mapping[str, Any], key: str, where: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise RecorderCoverageError(f"{where}.{key} is {value!r}, not a boolean")
    return value


def _stream_coverage(stream: str, section: Any, where: str) -> StreamCoverage:
    """One stream's record turned into a verdict, or a refusal."""
    if not isinstance(section, Mapping):
        raise RecorderCoverageError(f"{where} holds no object for stream {stream!r}")
    judged = _flag(section, "judged", f"{where}.{stream}")
    reason = section.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise RecorderCoverageError(f"{where}.{stream}.reason is {reason!r}, not a string")
    if not judged:
        return StreamCoverage(
            stream=stream,
            judged=False,
            published=0,
            agreeing=0,
            published_coverage=None,
            wallclock_coverage=None,
            passes=False,
            outage_flagged=False,
            reason=reason or "the archive object could not be established",
        )
    published = _count(section, "published_minutes", f"{where}.{stream}")
    agreeing = _count(section, "agreeing_minutes", f"{where}.{stream}")
    if agreeing > published:
        raise RecorderCoverageError(
            f"{where}.{stream} claims {agreeing} agreeing minutes out of {published} "
            "published; an agreeing minute is a published minute the recorder also holds, "
            "so it cannot exceed the denominator"
        )
    if published > MINUTES_PER_DAY:
        raise RecorderCoverageError(
            f"{where}.{stream} claims {published} published minutes and a UTC day holds "
            f"{MINUTES_PER_DAY}. A denominator larger than the day it is a denominator for "
            "is a document this reader does not understand, and it is refused rather than "
            "divided by"
        )
    if published == 0:
        return StreamCoverage(
            stream=stream,
            judged=False,
            published=0,
            agreeing=agreeing,
            published_coverage=None,
            wallclock_coverage=None,
            passes=False,
            outage_flagged=False,
            reason="the archive published no minute for this day, so there is no denominator",
        )
    return StreamCoverage(
        stream=stream,
        judged=True,
        published=published,
        agreeing=agreeing,
        published_coverage=agreeing / published,
        wallclock_coverage=agreeing / MINUTES_PER_DAY,
        passes=published_coverage_passes(agreeing, published),
        outage_flagged=wallclock_flags_outage(agreeing),
        reason=reason,
    )


def coverage_for_day(
    root: str | Path, day: str, *, contract: RecorderContract | None = None
) -> DayCoverage:
    """Whether one UTC day qualifies, computed from its reconciliation record.

    A day passes when **both** conditions of section 4.9 hold: every
    minute-indexed required stream reaches ``published_coverage >= 0.995``, and
    the day is funding-complete — its schedule established and every settlement
    it lists captured with exact agreement. Wall-clock coverage decides no
    stream's pass; it only flags ``RECORDER_OUTAGE``, and three flagged days in
    the gate's window fail the gate.

    The required streams are read from the contract, never listed here, so
    removing one from ``required_for_coverage`` removes it from the gate and
    cannot leave this module counting a stream the contract stopped requiring.

    Raises :class:`RecorderCoverageError` when the record is missing or cannot be
    read as a verdict about this day. The caller decides what an unjudgeable day
    means; :func:`gate` treats it as a day that did not pass, which is what
    breaks a streak rather than silently extending one.
    """
    resolved = _require_contract(contract)
    require_day(day)
    document = read_reconciliation(root, day)
    recorded_hash = document.get("contract_hash")
    if recorded_hash != resolved.contract_hash:
        raise RecorderCoverageError(
            f"the record for {day} was written under contract hash {recorded_hash!r} and is "
            f"being judged under {resolved.contract_hash!r}. A storage root never mixes "
            "contract hashes, and judging a day under an identity it was not recorded "
            "under would be relabelling evidence"
        )
    streams_section = document["streams"]
    funding_section = document["funding"]

    reasons: list[str] = []
    coverages: list[StreamCoverage] = []
    for stream in resolved.minute_indexed_required():
        if stream not in streams_section:
            raise RecorderCoverageError(
                f"the record for {day} says nothing about required stream {stream!r}. A "
                "required stream with no record is unjudgeable, never a pass"
            )
        entry = _stream_coverage(stream, streams_section[stream], f"{day} streams")
        coverages.append(entry)
        if not entry.judged:
            reasons.append(f"{stream}: {entry.reason}")
        elif not entry.passes:
            reasons.append(
                f"{stream}: published_coverage {entry.agreeing}/{entry.published} is below "
                "0.995"
            )

    established = _flag(funding_section, "schedule_established", f"{day} funding")
    complete = _flag(funding_section, "funding_complete", f"{day} funding")
    outcome = funding_section.get("outcome")
    if not isinstance(outcome, str) or not outcome:
        raise RecorderCoverageError(f"{day} funding.outcome is {outcome!r}, not a label")
    if complete and not established:
        raise RecorderCoverageError(
            f"{day} funding claims completeness without an established schedule. "
            "Completeness is a quantifier over an established set; an unestablished "
            "schedule is not an empty one and never becomes one"
        )
    where = f"{day} funding"
    scheduled = _count(funding_section, "scheduled", where) if established else None
    captured = _count(funding_section, "captured", where) if established else None
    if scheduled is not None and captured is not None:
        # The counts are read straight afterwards, so they are cross-checked
        # against the boolean that was supposed to have been derived from them.
        # ``funding_complete`` is "every scheduled settlement is captured", so a
        # record claiming completeness while capturing fewer than it scheduled is
        # self-contradictory, and this reader has to fail closed on a document it
        # does not understand rather than take the boolean's word for it.
        if captured > scheduled:
            raise RecorderCoverageError(
                f"{where} claims {captured} captured settlements out of {scheduled} "
                "scheduled; a captured settlement is a scheduled settlement the recorder "
                "also holds, so it cannot exceed the schedule"
            )
        if complete and captured != scheduled:
            raise RecorderCoverageError(
                f"{where} claims completeness with {captured} of {scheduled} scheduled "
                "settlements captured. Completeness is every scheduled settlement being "
                "captured, so the boolean and the counts in the same record disagree"
            )
    if not established:
        reasons.append(f"um.funding: {FUNDING_SCHEDULE_UNAVAILABLE}")
    elif not complete:
        reasons.append("um.funding: a scheduled settlement is missing or disagrees")

    judged = all(entry.judged for entry in coverages)
    minutes_pass = judged and all(entry.passes for entry in coverages)
    if not coverages:
        raise RecorderCoverageError(
            f"recorder contract {resolved.label} requires no minute-indexed stream, so a "
            "day would pass on funding alone. The gate is not defined without a "
            "minute-indexed denominator"
        )
    if minutes_pass and complete:
        verdict = DAY_PASS
    elif judged and established:
        verdict = DAY_FAIL
    else:
        verdict = DAY_UNJUDGEABLE
    return DayCoverage(
        day=day,
        verdict=verdict,
        streams=tuple(coverages),
        schedule_established=established,
        funding_complete=complete,
        funding_outcome=outcome,
        scheduled_settlements=scheduled,
        captured_settlements=captured,
        # A stream that could not be judged has no wall-clock coverage either,
        # so an unjudgeable day is never *also* an outage flag: it would put a
        # day with no evidence into the three-flagged-days count.
        outage_flagged=judged and any(entry.outage_flagged for entry in coverages),
        reasons=tuple(reasons),
    )


def _next_day(day: str) -> str:
    return utc_day(day_start_ns(day) + NS_PER_DAY)


def _judge(root: str | Path, day: str, contract: RecorderContract) -> GateDay:
    """One day's gate entry: a verdict that exists whatever the evidence does.

    A missing record and a malformed one are different findings and are reported
    as such, and neither raises: the gate has to be computable over a window
    that contains a damaged day, and a day it could not judge is a day that did
    not pass.
    """
    try:
        coverage = coverage_for_day(root, day, contract=contract)
    except RecorderCoverageError as exc:
        missing = not reconciliation_path(root, day).exists()
        return GateDay(
            day=day,
            verdict=DAY_MISSING if missing else DAY_UNJUDGEABLE,
            outage_flagged=False,
            reasons=(str(exc),),
        )
    return GateDay(
        day=day,
        verdict=coverage.verdict,
        outage_flagged=coverage.outage_flagged,
        reasons=coverage.reasons,
        coverage=coverage,
    )


def _walk(root: str | Path, contract: RecorderContract, days: Sequence[str]) -> list[GateDay]:
    return [_judge(root, day, contract) for day in days]


def _calendar(first: str, last: str) -> list[str]:
    """Every UTC day from ``first`` to ``last`` inclusive, gaps included.

    Built from the calendar rather than from the files, because a day with no
    record is exactly the case the streak has to notice: iterating the records
    would skip it and silently join the days on either side of a hole.
    """
    if day_start_ns(first) > day_start_ns(last):
        return []
    walked = [first]
    while walked[-1] != last:
        walked.append(_next_day(walked[-1]))
    return walked


def _passing_runs(entries: Sequence[GateDay]) -> list[tuple[str, ...]]:
    """The maximal runs of consecutive passing days, in calendar order.

    A day that failed, a day that could not be judged and a day with no record at
    all each end the run they are in, which is the whole point: the gate is a
    statement about an unbroken period of complete recording, and a hole in the
    evidence is not a period of complete recording.
    """
    runs: list[tuple[str, ...]] = []
    current: list[str] = []
    for entry in entries:
        if entry.verdict == DAY_PASS:
            current.append(entry.day)
            continue
        if current:
            runs.append(tuple(current))
            current = []
    if current:
        runs.append(tuple(current))
    return runs


def gate(
    root: str | Path,
    window: int = DEFAULT_WINDOW,
    *,
    contract: RecorderContract | None = None,
) -> GateVerdict:
    """The gate's verdict, recomputed from every reconciliation record on disk.

    A **qualifying window** is ``window`` consecutive UTC days, at or after
    ``prospective_from``, every one of which passed, carrying no more than
    :data:`MAX_OUTAGE_FLAGGED_DAYS` days flagged ``RECORDER_OUTAGE``. The gate
    passes exactly when one exists. That is section 4.9 read literally — 30
    consecutive days pass, counted from the first passing day at or after the
    boundary, with any non-passing day resetting the count, and three flagged
    days *in a window* failing the gate — and it is deliberately **not** measured
    only backwards from the newest record.

    The difference matters because of amendment A9. The funding schedule source
    is a monthly object that is not published while its month is open, so every
    day of the current month reconciles to ``FUNDING_SCHEDULE_UNAVAILABLE`` and
    is unjudgeable; the daily job writes those records like any others. A gate
    measured as the run *ending at the newest record* would therefore un-claim
    itself the day after it was honestly achieved, destroyed by expected evidence
    latency that A9 states is "not a recorder outage" — no recorded minute having
    changed, no threshold having moved, and the only way to hold a pass being to
    stop reconciling. Counting a run wherever the calendar puts it means more
    evidence can only ever help: a day recorded later never removes a period the
    recorder has already demonstrated, and a day whose verdict genuinely changes
    still moves the answer, because everything here is recomputed from the
    records every time.

    Both numbers are reported. :attr:`GateVerdict.streak` is the longest run and
    is what the window is measured against; :attr:`GateVerdict.current_streak`
    is the run ending at the newest record, so a reader can still see at a glance
    whether the recorder is in an unbroken period right now.

    While the contract's ``prospective_from`` is null the verdict is
    :data:`GATE_BOUNDARY_UNSET` and can never be a pass, whatever the records
    say. Reconciliation and coverage are still computed — that is engineering
    work and it is how the pipeline is exercised before the boundary exists —
    but no run of engineering days is an S1 claim, and this function will not
    return one.
    """
    resolved = _require_contract(contract)
    if window < 1:
        raise RecorderCoverageError(f"window must be at least one day, got {window}")
    recorded = available_reconciliation_days(root)
    boundary = None if resolved.prospective_from is None else resolved.prospective_from.date()
    boundary_day = None if boundary is None else boundary.isoformat()

    considered = (
        recorded
        if boundary_day is None
        else [day for day in recorded if day_start_ns(day) >= day_start_ns(boundary_day)]
    )
    if considered:
        first = considered[0] if boundary_day is None else boundary_day
        entries = _walk(root, resolved, _calendar(first, considered[-1]))
    else:
        entries = []

    runs = _passing_runs(entries)
    streak = max((len(run) for run in runs), default=0)
    current_streak = (
        len(runs[-1]) if runs and entries and runs[-1][-1] == entries[-1].day else 0
    )

    flagged_lookup = {entry.day: entry.outage_flagged for entry in entries}
    candidates = [
        run[start : start + window] for run in runs for start in range(len(run) - window + 1)
    ]
    scored = [
        (candidate, tuple(day for day in candidate if flagged_lookup.get(day)))
        for candidate in candidates
    ]
    qualifying = [entry for entry in scored if len(entry[1]) <= MAX_OUTAGE_FLAGGED_DAYS]
    if qualifying:
        # The earliest window that clears the bar, so the answer is a fact about
        # the calendar rather than about the order the records were examined in.
        window_days, flagged = qualifying[0]
    elif scored:
        # None qualifies: show the closest one so the reader sees which flags
        # stood in the way, still chosen deterministically.
        window_days, flagged = min(scored, key=lambda entry: (len(entry[1]), entry[0][0]))
    else:
        window_days, flagged = (), ()

    reasons: list[str] = []
    if boundary_day is None:
        verdict = GATE_BOUNDARY_UNSET
        reasons.append(
            "prospective_from is null in the committed contract, so no recorded minute is "
            "scientific evidence and no streak of engineering days is an S1 pass. The "
            "boundary is written by a separate reviewed commit, never by this tool."
        )
    elif not scored:
        verdict = GATE_FAIL
        reasons.append(
            f"the longest run of consecutive passing day(s) is {streak} of {window}"
        )
    elif not qualifying:
        verdict = GATE_FAIL
        reasons.append(
            f"{len(flagged)} day(s) in every candidate window are flagged {RECORDER_OUTAGE}; "
            f"the closest window is {window_days[0]}..{window_days[-1]}, and three flagged "
            "days in a window fail the gate"
        )
    else:
        verdict = GATE_PASS

    passed = verdict == GATE_PASS
    return GateVerdict(
        verdict=verdict,
        official=boundary_day is not None,
        gate_passed=passed,
        window=window,
        streak=streak,
        current_streak=current_streak,
        contract_id=resolved.contract_id,
        contract_hash=resolved.contract_hash,
        prospective_from=boundary_day,
        window_days=window_days,
        outage_flagged_days=flagged,
        days=tuple(entries),
        reasons=tuple(reasons),
    )


def write_gate(root: str | Path, verdict: GateVerdict) -> Path:
    """Write ``coverage/GATE.json`` atomically, and return where it went."""
    path = gate_path(root)
    write_json_atomic(path, verdict.to_dict())
    return path


def summarise(verdict: GateVerdict) -> Iterable[str]:
    """The verdict as lines a person reads, in the CLI's voice."""
    yield f"contract      {verdict.contract_id}  {verdict.contract_hash}"
    boundary = verdict.prospective_from or "not set"
    kind = "prospective" if verdict.official else "engineering"
    yield f"boundary      prospective_from={boundary}  ({kind} data)"
    yield f"verdict       {verdict.verdict}  gate_passed={str(verdict.gate_passed).lower()}"
    yield f"streak        {verdict.streak} of {verdict.window} consecutive passing day(s)"
    yield f"current run   {verdict.current_streak} day(s) ending at the newest record"
    window = (
        f"{verdict.window_days[0]}..{verdict.window_days[-1]}"
        if verdict.window_days
        else "none yet"
    )
    yield f"window        {window}"
    yield (
        f"outage flags  {len(verdict.outage_flagged_days)} in window "
        f"(more than {MAX_OUTAGE_FLAGGED_DAYS} fails the gate)"
    )
    for reason in verdict.reasons:
        yield f"  ! {reason}"
