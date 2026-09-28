"""R1-i: ``resolve --ledger funding_booking_torn`` after a RESUME, and interrupted.

The independent review of PR #110 (head ``bd1ea99``) reproduced two failures of
the torn-funding resolve. Both came from a proof anchored to the log's last
restated ``risk.state_hash``, which a ``RESUME`` does not restate:

A. **double count** -- halt, operator ``resume`` (which clears the adverse
   streak by design), the next settlement booked, the process killed after
   Aegis noted it and before the carry ledger was saved: the resolve "proved"
   Aegis had missed it and noted it again;
B. **no clearing path** -- the same, killed before Aegis noted it: the resolve
   could prove neither and refused, and nothing else clears the dispute.

A kill INSIDE the resolve had the same dead end.

Every scenario here is compared with an UNINTERRUPTED reference: the same base
state, the same settlement minute ticked without a crash. Cash alone is not the
comparison -- the original defect kept cash exact -- so Aegis's funding streak,
its whole hashed state (halt fields aside), the settlements and the number of
times the log books each settlement are compared too.

The settlements are fixture data at hours 1 and 2 (minutes 59 and 119), not
the venue's 8-hourly cadence: what is tested is the order of the writes, and
two settlements are what a RESUME between them needs.
"""

from __future__ import annotations

import contextlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

import pytest

from chimera.carry.ledger import LEDGER_SCHEMA, CarryLedger
from chimera.demo.risk_continuity import risk_state_hash
from chimera.demo.runner import DemoRunner, RunnerError, RunnerState
from chimera.futures.executor import FuturesExecutor

from demo_harness import DAY, build
from r1i_crash import (
    Faults,
    Killed,
    World,
    check_log,
    consistency,
    copy_state,
    economics,
    excluded_minutes,
    instrumented,
    kill_at_the_note,
    records,
    state_bytes,
)

NOTE = "funding witness: the operator checked the files the runbook names"
PAID = "-0.0003"  # the SHORT perpetual pays: the direction that moves the streak
RECEIVED = "0.0001"  # the SHORT perpetual receives: a rebate that resets it
#: The minute (index) whose close is 02:00, the second settlement.
SETTLEMENT = 119
#: Where both runs are continued to after the settlement.
CONTINUE_TO = 123
TORN = "funding_booking_torn"
D = Decimal


# ---------------------------------------------------------------------------
# worlds and base states, built once
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def worlds(tmp_path_factory) -> dict[str, tuple[World, int]]:
    """``paid``: both settlements paid. ``received``: paid, then received."""
    made: dict[str, tuple[World, int]] = {}
    for name, second in (("paid", PAID), ("received", RECEIVED)):
        base = tmp_path_factory.mktemp(f"funding_world_{name}")
        harness = build(base, start=False)
        harness.feed.write_settlements(
            [DAY], hours=(1, 2), rates={(DAY, 1): PAID, (DAY, 2): second}
        )
        made[name] = (World(root=harness.root), harness.first_minute_ms())
    return made


@pytest.fixture(scope="module")
def bases(worlds, tmp_path_factory) -> dict[tuple[str, str], Path]:
    """The state before the settlement minute, per (world, origin).

    ``plain``: minutes 0 .. 118 decided in one process; the first (paid)
    settlement has left Aegis's streak at 1. ``resumed``: then halted by the
    kill switch and resumed by the operator in fresh processes, so the log's
    newest risk statement is a ``RESUME`` that restates no hash and the streak
    is 0.
    """
    made: dict[tuple[str, str], Path] = {}
    for name, (world, first_ms) in worlds.items():
        plain = tmp_path_factory.mktemp(f"base_{name}_plain")
        runner = world.runner(plain)
        runner.start()
        runner.catch_up(now_ms=first_ms + (SETTLEMENT - 1) * 60_000)
        assert runner.cursor.last_minute_processed == first_ms + (SETTLEMENT - 1) * 60_000
        assert (runner.risk.state.funding_adverse_streak, runner.risk.state.funding_halt) == (
            1,
            False,
        )
        runner.shutdown("prepared")
        made[(name, "plain")] = plain

        resumed = copy_state(plain, tmp_path_factory.mktemp(f"base_{name}_resumed") / "s")
        (resumed / "KILL_SWITCH").write_text("drill\n", encoding="utf-8")
        halted = world.runner(resumed)
        assert halted.start() is RunnerState.HALT
        halted.shutdown("halted")
        (resumed / "KILL_SWITCH").unlink()
        operator = world.runner(resumed)
        operator.start()
        operator.resume(NOTE)
        assert (
            operator.risk.state.funding_adverse_streak,
            operator.risk.state.funding_halt,
        ) == (
            0,
            False,
        )
        operator.shutdown("resumed")
        kinds = [r["kind"] for r in records(resumed)]
        restated = max(i for i, r in enumerate(records(resumed)) if "risk" in r)
        assert (
            max(i for i, k in enumerate(kinds) if k == "RESUME") > restated
        ), "the log's newest risk statement must be the RESUME, which restates no hash"
        made[(name, "resumed")] = resumed
    return made


# ---------------------------------------------------------------------------
# the uninterrupted reference
# ---------------------------------------------------------------------------
def _aegis(runner: DemoRunner) -> dict[str, Any]:
    """Aegis's funding guard and its hashed state, halt fields aside (a torn
    campaign is halted on its dispute until the operator resumes it)."""
    snapshot = runner.risk.snapshot()
    return {
        "funding": (snapshot["funding_adverse_streak"], snapshot["funding_halt"]),
        "hash": risk_state_hash({**snapshot, "halted": False, "halt_reason": ""}),
    }


def _settlement_ns(first_ms: int, hour: int) -> int:
    return (first_ms + hour * 3_600_000) * 1_000_000


def _reference(world: World, base: Path, tmp: Path, first_ms: int) -> dict[str, Any]:
    state_dir = copy_state(base, tmp / "reference")
    runner = world.runner(state_dir)
    runner.start()
    runner.tick(first_ms + SETTLEMENT * 60_000)
    assert runner.state is not RunnerState.HALT, runner.halt_reason
    at_settlement = _aegis(runner)
    runner.catch_up(now_ms=first_ms + CONTINUE_TO * 60_000)
    result = {"aegis": at_settlement, "economics": economics(runner)}
    runner.shutdown("reference")
    return result


def _verdict(world_name: str, origin: str, noted: bool) -> str:
    """What the resolve must find, by hand: a rebate on a streak a RESUME left at
    zero moves nothing (``unmoved``); otherwise whether Aegis noted it."""
    if world_name == "received" and origin == "resumed":
        return "unmoved"
    return "counted" if noted else "missed"


def _expected_streak(world_name: str, origin: str) -> tuple[int, bool]:
    """What the second settlement leaves the streak at, by hand.

    The first (paid) settlement left 1; a RESUME cleared it to 0. A paid
    settlement adds one (limit 3: no funding halt); a received one resets it.
    """
    prior = 1 if origin == "plain" else 0
    return (prior + 1, False) if world_name == "paid" else (0, False)


# ---------------------------------------------------------------------------
# kills: the tick's settlement, and the resolve itself
# ---------------------------------------------------------------------------
def _tear(world: World, base: Path, tmp: Path, first_ms: int, when: str) -> Path:
    """The settlement minute, killed at the note: a torn booking on disk."""
    state_dir = copy_state(base, tmp / f"torn_{when}")
    faults = Faults()
    runner = world.runner(state_dir)
    with instrumented(faults), kill_at_the_note(faults, when), pytest.raises(Killed):
        runner.start()
        runner.tick(first_ms + SETTLEMENT * 60_000)
    assert faults.killed and not faults.uninventoried
    halted = world.runner(state_dir)
    assert halted.start() is RunnerState.HALT
    assert (halted.position.ledger.disputed or "").startswith(TORN), halted.position.ledger
    assert ("--ledger " + TORN) in [s for s, _ in halted.standing_disputes()]
    halted.shutdown("torn")
    return state_dir


@contextlib.contextmanager
def _kill_inside_the_resolve(faults: Faults, at: str) -> Iterator[None]:
    """``after_request``: the OPERATOR ``requested`` record is durable and
    nothing else. ``after_note``: the resolve's own note to Aegis persisted and
    the ledger was not saved."""
    if at == "after_request":
        original = DemoRunner._operator_requested

        def requested(self, *args: Any, **kwargs: Any) -> int:
            original(self, *args, **kwargs)
            faults.killed = True
            raise Killed("killed after the resolve's request")

        DemoRunner._operator_requested = requested  # type: ignore[method-assign]
        try:
            yield
        finally:
            DemoRunner._operator_requested = original  # type: ignore[method-assign]
    else:
        with kill_at_the_note(faults, "after"):
            yield


def _resolve(world: World, state_dir: Path) -> None:
    runner = world.runner(state_dir)
    assert runner.start() is RunnerState.HALT
    runner.resolve_ledger(TORN, NOTE)
    assert not [s for s, _ in runner.standing_disputes()], runner.standing_disputes()
    runner.shutdown("resolved")


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def _bookings(state_dir: Path, instant_ns: int) -> int:
    """How many times the log books the settlement: its FUNDING record, or a
    COMPLETED re-book of it. Exactly once, whichever path booked it."""
    count = 0
    for record in records(state_dir):
        funding = record.get("funding")
        if record["kind"] == "FUNDING" and isinstance(funding, dict):
            count += int(funding.get("settlement_instant_ns") == instant_ns)
        operator = record.get("operator")
        if (
            isinstance(operator, dict)
            and operator.get("phase") == "completed"
            and operator.get("kind") == TORN
        ):
            count += int(operator["rebook"]["instant_ns"] == instant_ns)
    return count


def _operator_records_are_truthful(state_dir: Path) -> list[str]:
    """Every ``requested`` is completed or reported unfinished; every
    completion cites a request."""
    problems: list[str] = []
    requested: set[int] = set()
    closed: set[int] = set()
    for record in records(state_dir):
        operator = record.get("operator")
        if isinstance(operator, dict):
            if operator.get("phase") == "requested":
                requested.add(int(record["seq"]))
            elif isinstance(operator.get("request_seq"), int):
                if operator["request_seq"] not in requested:
                    problems.append(f"seq {record['seq']} completes an unknown request")
                closed.add(int(operator["request_seq"]))
        recovery = record.get("recovery")
        if isinstance(recovery, dict) and recovery.get("cause") == "OPERATOR_INCOMPLETE":
            closed.add(int(recovery["request_seq"]))
    if requested - closed:
        problems.append(
            f"requests neither completed nor reported: {sorted(requested - closed)}"
        )
    return problems


def _converges(
    world: World,
    state_dir: Path,
    reference: dict[str, Any],
    *,
    first_ms: int,
    expected: tuple[int, bool],
    verdict: str,
) -> None:
    """The resolved campaign against the uninterrupted one, then a second restart."""
    instant = _settlement_ns(first_ms, 2)
    # Before the operator's resume, which clears the streak by design.
    runner = world.runner(state_dir)
    assert runner.start() is RunnerState.HALT
    assert runner.standing_disputes() == [], runner.standing_disputes()
    aegis = _aegis(runner)
    assert aegis["funding"] == expected == reference["aegis"]["funding"], (
        f"Aegis's funding streak {aegis['funding']}, reference "
        f"{reference['aegis']['funding']}, by hand {expected}"
    )
    assert (
        aegis["hash"] == reference["aegis"]["hash"]
    ), "Aegis's hashed state is not the reference's"
    assert instant in runner.position.ledger.state.settled
    runner.resume(NOTE)
    runner.shutdown("resumed")

    runner = world.runner(state_dir)
    assert runner.start() is RunnerState.READY, runner.halt_reason
    runner.catch_up(now_ms=first_ms + CONTINUE_TO * 60_000)
    result = economics(runner)
    assert not consistency(runner), consistency(runner)
    runner.shutdown("continued")
    # Exact recovery is required here; no minute may be excluded to pass.
    assert excluded_minutes(state_dir) == []
    wanted = dict(reference["economics"])
    got = dict(result)
    # The resume cleared the streak; the comparison above is the one that counts.
    wanted.pop("funding_streak")
    got.pop("funding_streak")
    assert got == wanted, {k: (wanted[k], got[k]) for k in wanted if wanted[k] != got[k]}
    assert _bookings(state_dir, instant) == 1
    assert check_log(state_dir) == []
    assert _operator_records_are_truthful(state_dir) == []
    completions = [
        r["operator"]
        for r in records(state_dir)
        if r["kind"] == "OPERATOR"
        and r["operator"].get("phase") == "completed"
        and r["operator"].get("kind") == TORN
    ]
    assert [c["rebook"]["aegis"] for c in completions] == [verdict]

    # A second restart: the same interpretation, nothing new.
    before = len(records(state_dir))
    again = world.runner(state_dir)
    assert again.start() is RunnerState.READY, again.halt_reason
    assert "RECOVERY" not in [r["kind"] for r in records(state_dir)[before:]]
    assert economics(again) == result
    assert _bookings(state_dir, instant) == 1
    again.shutdown("converged")


CASES = [
    (world, origin, when)
    for world in ("paid", "received")
    for origin in ("resumed", "plain")
    for when in ("after", "before")
]


def _id(case: tuple[str, ...]) -> str:
    world, origin, when = case[:3]
    tail = f"-{case[3]}" if len(case) > 3 else ""
    counted = "counted" if when == "after" else "missed"
    return f"{world}-{origin}-{counted}{tail}"


# ---------------------------------------------------------------------------
# 1, 2 and 5: a torn settlement, counted or missed, after a RESUME or not
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=_id)
def test_a_torn_settlement_is_counted_by_aegis_exactly_once(worlds, bases, tmp_path, case):
    """``after``: Aegis noted it before the kill (proved COUNTED, not noted
    again). ``before``: the kill came first (proved MISSED, noted once). With a
    RESUME as the log's newest risk statement, each is the review's A and B."""
    name, origin, when = case
    world, first_ms = worlds[name]
    base = bases[(name, origin)]
    reference = _reference(world, base, tmp_path, first_ms)
    state_dir = _tear(world, base, tmp_path, first_ms, when)
    _resolve(world, state_dir)
    _converges(
        world,
        state_dir,
        reference,
        first_ms=first_ms,
        expected=_expected_streak(name, origin),
        verdict=_verdict(name, origin, noted=(when == "after")),
    )


# ---------------------------------------------------------------------------
# 3, 4 and 5: a kill inside the resolve, then the resolve run again
# ---------------------------------------------------------------------------
RESOLVE_CASES = [
    (world, "resumed", when, at)
    for world in ("paid", "received")
    for when in ("after", "before")
    for at in ("after_request", "after_note")
    # The resolve notes Aegis only for a settlement proved MISSED (a rebate on
    # the zero streak a RESUME left is UNMOVED: there is nothing to note).
    if not (at == "after_note" and (when == "after" or world == "received"))
] + [("paid", "plain", "before", "after_request"), ("paid", "plain", "before", "after_note")]


@pytest.mark.parametrize("case", RESOLVE_CASES, ids=_id)
def test_a_kill_inside_the_resolve_is_finished_by_running_it_again(
    worlds, bases, tmp_path, case
):
    """The request is durable (and, for ``after_note``, the resolve's own note
    to Aegis); the ledger is not. The dispute still stands, and the same
    resolve, run again, finishes it -- never counting the settlement a second
    time and never refusing -- and the interrupted request is reported."""
    name, origin, when, at = case
    world, first_ms = worlds[name]
    base = bases[(name, origin)]
    reference = _reference(world, base, tmp_path, first_ms)
    state_dir = _tear(world, base, tmp_path, first_ms, when)

    faults = Faults()
    runner = world.runner(state_dir)
    with instrumented(faults), _kill_inside_the_resolve(faults, at), pytest.raises(Killed):
        assert runner.start() is RunnerState.HALT
        runner.resolve_ledger(TORN, NOTE)
    assert faults.killed and not faults.uninventoried

    # The request is durable and nothing completes it. (This start halts on the
    # standing dispute before RECOVER, so the OPERATOR_INCOMPLETE report comes
    # from the first start after the dispute is cleared; `_converges` requires
    # it.) The dispute still stands: the ledger was never saved.
    requested = [
        r
        for r in records(state_dir)
        if r["kind"] == "OPERATOR" and r["operator"].get("phase") == "requested"
    ]
    assert requested and requested[-1]["operator"]["kind"] == TORN
    restarted = world.runner(state_dir)
    assert restarted.start() is RunnerState.HALT
    assert ("--ledger " + TORN) in [s for s, _ in restarted.standing_disputes()]
    restarted.shutdown("seen")

    _resolve(world, state_dir)
    _converges(
        world,
        state_dir,
        reference,
        first_ms=first_ms,
        expected=_expected_streak(name, origin),
        # The completing run judged Aegis as the interrupted one left it: a
        # note that persisted before the kill is found counted.
        verdict=_verdict(name, origin, noted=(when == "after" or at == "after_note")),
    )


# ---------------------------------------------------------------------------
# the record itself: chimera.carry-ledger/3's `aegis_funding_before`
# ---------------------------------------------------------------------------
GUARD = {"instant_ns": 7_200_000_000_000, "funding_adverse_streak": 2, "funding_halt": False}


def test_the_guard_round_trips_and_older_versions_read_as_unknown(tmp_path):
    """/3 writes and reads it. /2 keeps its correction (still read, not
    expired) and, like /1, reads as no guard: the resolve then refuses."""
    path = tmp_path / "carry_ledger.json"
    ledger = CarryLedger.open(path, capital=D("1000000"))
    ledger.note_aegis_funding_before(
        GUARD["instant_ns"], funding_adverse_streak=2, funding_halt=False
    )
    assert ledger.begin_correction(123_000_000_000, D("0.5"))
    ledger.save()
    written = json.loads(path.read_text("utf-8"))
    assert written["schema"] == LEDGER_SCHEMA == "chimera.carry-ledger/3"
    assert written["aegis_funding_before"] == GUARD
    assert CarryLedger.open(path, capital=D("1000000")).state.aegis_funding_before == GUARD

    for version in ("chimera.carry-ledger/2", "chimera.carry-ledger/1"):
        old = dict(written, schema=version)
        if version.endswith("/1"):
            old.pop("correction")
        path.write_text(json.dumps(old), encoding="utf-8")
        state = CarryLedger.open(path, capital=D("1000000")).state
        assert not state.disputed, (version, state.disputed)
        assert state.aegis_funding_before is None, version
        if version.endswith("/2"):
            assert state.correction == {"started_ns": 123_000_000_000, "target": "0.5"}


@pytest.mark.parametrize(
    "value",
    [
        "yes",
        {**GUARD, "funding_adverse_streak": -1},
        {**GUARD, "funding_adverse_streak": True},
        {**GUARD, "funding_halt": "false"},
        {**GUARD, "instant_ns": "7200"},
        {k: v for k, v in GUARD.items() if k != "funding_halt"},
        {**GUARD, "extra": 1},
    ],
)
def test_a_malformed_guard_is_refused_never_defaulted(tmp_path, value):
    path = tmp_path / "carry_ledger.json"
    path.write_text(
        json.dumps(
            {"schema": LEDGER_SCHEMA, "capital": "1000000", "aegis_funding_before": value}
        ),
        encoding="utf-8",
    )
    disputed = CarryLedger.open(path, capital=D("1000000")).state.disputed
    assert (disputed or "").startswith("ledger_unreadable"), disputed


def test_booking_a_settlement_clears_its_own_guard_and_no_other(tmp_path):
    ledger = CarryLedger.open(tmp_path / "carry_ledger.json", capital=D("1000000"))
    ledger.note_aegis_funding_before(
        GUARD["instant_ns"], funding_adverse_streak=2, funding_halt=False
    )
    ledger.book_funding(GUARD["instant_ns"] + 1, D("-1"))
    assert ledger.state.aegis_funding_before == GUARD
    ledger.book_funding(GUARD["instant_ns"], D("-1"))
    assert ledger.state.aegis_funding_before is None


def test_the_tick_saves_the_guard_before_the_executor_books(worlds, bases, tmp_path):
    """The order the proof rests on, read off disk at the executor's booking:
    the ledger already holds the guard as it was before the note, for this
    exact settlement, and Aegis has not moved."""
    world, first_ms = worlds["paid"]
    state_dir = copy_state(bases[("paid", "resumed")], tmp_path / "order")
    seen: list[dict[str, Any]] = []
    original = FuturesExecutor.settle_funding

    def settle(self, *args: Any, **kwargs: Any) -> Any:
        ledger = json.loads((state_dir / "carry_ledger.json").read_text("utf-8"))
        risk = json.loads((state_dir / "risk.json").read_text("utf-8"))
        seen.append(
            {
                "ledger": ledger["aegis_funding_before"],
                "streak": risk["funding_adverse_streak"],
            }
        )
        return original(self, *args, **kwargs)

    FuturesExecutor.settle_funding = settle  # type: ignore[method-assign]
    try:
        runner = world.runner(state_dir)
        runner.start()
        runner.tick(first_ms + SETTLEMENT * 60_000)
    finally:
        FuturesExecutor.settle_funding = original  # type: ignore[method-assign]
    assert seen == [
        {
            "ledger": {
                "instant_ns": _settlement_ns(first_ms, 2),
                "funding_adverse_streak": 0,
                "funding_halt": False,
            },
            "streak": 0,
        }
    ]
    # And the save that books it clears it.
    assert (
        json.loads((state_dir / "carry_ledger.json").read_text("utf-8"))[
            "aegis_funding_before"
        ]
        is None
    )
    runner.shutdown("done")


# ---------------------------------------------------------------------------
# UNKNOWABLE: refused, and nothing changed
# ---------------------------------------------------------------------------
def _edit_ledger(state_dir: Path, edit) -> None:
    path = state_dir / "carry_ledger.json"
    payload = json.loads(path.read_text("utf-8"))
    edit(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _older_version(payload: dict[str, Any]) -> None:
    payload["schema"] = "chimera.carry-ledger/2"
    payload.pop("aegis_funding_before")


def _another_instant(payload: dict[str, Any]) -> None:
    payload["aegis_funding_before"]["instant_ns"] += 60_000_000_000


def _neither_state(payload: dict[str, Any]) -> None:
    # A guard Aegis is in neither before nor after this settlement.
    payload["aegis_funding_before"]["funding_adverse_streak"] = 7


@pytest.mark.parametrize(
    "edit", [_older_version, _another_instant, _neither_state], ids=lambda f: f.__name__
)
@pytest.mark.parametrize("when", ["after", "before"])
def test_an_unprovable_verdict_refuses_and_changes_nothing(
    worlds, bases, tmp_path, edit, when
):
    """No guessed funding state: without the guard for THIS settlement, or with
    Aegis in neither state it allows, the resolve refuses and every file is
    left byte for byte as it was."""
    world, first_ms = worlds["paid"]
    state_dir = _tear(world, bases[("paid", "resumed")], tmp_path, first_ms, when)
    _edit_ledger(state_dir, edit)
    runner = world.runner(state_dir)
    assert runner.start() is RunnerState.HALT
    before = state_bytes(state_dir)
    with pytest.raises(RunnerError, match="not provable"):
        runner.resolve_ledger(TORN, NOTE)
    assert state_bytes(state_dir) == before
    assert ("--ledger " + TORN) in [s for s, _ in runner.standing_disputes()]
    runner.shutdown("refused")
