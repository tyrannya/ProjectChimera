"""The runner clock, and the one property it exists to have.

Section 2.4's requirement is not "the runner has a clock" but "the runner's
decisions do not depend on when they were taken". So the tests here are mostly
about what *cannot* change the answer: reordering the observations, re-running
them, and moving the host's clock while they run. Each is asserted from both
sides — the sequence that should advance the clock does, the sequence that
should not does not — and the wall-clock independence is asserted twice, once by
denying every clock in the process and once by parsing the module's own imports,
because a monkeypatch only covers the calls somebody thought to patch.
"""

from __future__ import annotations

import ast
import time
from fractions import Fraction
from itertools import accumulate
from pathlib import Path

import pytest

from chimera.demo.clock import RunnerClock, RunnerClockError
from chimera.recorder import events

MODULE_SOURCE = Path(__file__).resolve().parents[1] / "chimera" / "demo" / "clock.py"

#: A recorded receipt sequence with everything that actually happens in one: a
#: run of increasing stamps, a duplicate, a reconnect that replays two earlier
#: frames, and a jump forward afterwards.
RECEIPTS = [
    1_795_312_800_000_000_000,
    1_795_312_860_000_000_000,
    1_795_312_860_000_000_000,
    1_795_312_830_000_000_000,
    1_795_312_800_000_000_001,
    1_795_312_920_123_456_789,
    1_795_312_920_123_456_788,
]


def test_clock_starts_with_no_instant() -> None:
    clock = RunnerClock()
    assert clock.started is False
    with pytest.raises(RunnerClockError, match="observed nothing"):
        _ = clock.now_ns
    with pytest.raises(RunnerClockError, match="observed nothing"):
        clock.time()


def test_start_ns_seeds_the_clock_for_a_restart() -> None:
    clock = RunnerClock(start_ns=1_795_312_800_000_000_000)
    assert clock.started is True
    assert clock.now_ns == 1_795_312_800_000_000_000
    # A seeded clock is still a floor: an earlier observation does not move it.
    assert clock.observe(1_795_312_700_000_000_000) == 1_795_312_800_000_000_000


def test_advances_to_a_later_receipt_and_never_to_an_earlier_one() -> None:
    clock = RunnerClock()
    assert clock.observe(1_000_000_000_000_000_000) == 1_000_000_000_000_000_000
    # Just above: advances by exactly one nanosecond.
    assert clock.observe(1_000_000_000_000_000_001) == 1_000_000_000_000_000_001
    # Exactly at: unchanged.
    assert clock.observe(1_000_000_000_000_000_001) == 1_000_000_000_000_000_001
    # Just below: unchanged, and the *clock* is returned, not the argument.
    assert clock.observe(1_000_000_000_000_000_000) == 1_000_000_000_000_000_001
    assert clock.now_ns == 1_000_000_000_000_000_001


def test_the_sequence_is_the_running_maximum_of_the_receipts() -> None:
    clock = RunnerClock()
    observed = [clock.observe(receipt) for receipt in RECEIPTS]
    # An independent oracle rather than the implementation restated: the clock
    # after k observations is the maximum of the first k receipts.
    assert observed == list(accumulate(RECEIPTS, max))
    assert clock.now_ns == max(RECEIPTS)


def test_the_same_recording_replays_to_the_same_sequence() -> None:
    live = RunnerClock()
    first = [live.observe(receipt) for receipt in RECEIPTS]
    replay = RunnerClock()
    second = [replay.observe(receipt) for receipt in RECEIPTS]
    assert first == second
    assert live.now_ns == replay.now_ns
    assert live.time() == replay.time()


def test_the_wall_clock_cannot_change_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deny every clock in the process, then run the recording again."""
    reference = RunnerClock()
    expected = [reference.observe(receipt) for receipt in RECEIPTS]

    def denied(*args: object, **kwargs: object) -> float:
        raise AssertionError("the runner clock read a wall clock")

    for name in (
        "time",
        "time_ns",
        "monotonic",
        "monotonic_ns",
        "perf_counter",
        "perf_counter_ns",
        "gmtime",
        "localtime",
    ):
        monkeypatch.setattr(time, name, denied)

    clock = RunnerClock()
    denied_run = [clock.observe(receipt) for receipt in RECEIPTS]
    assert denied_run == expected
    assert clock.time() == pytest.approx(max(RECEIPTS) / 1e9)


def test_the_module_imports_no_clock_at_all() -> None:
    """A monkeypatch only covers the calls somebody thought of; this covers all.

    The clock module is allowed exactly one dependency. Anything else — ``time``,
    ``datetime``, ``os`` — would be a way for the host's state to reach a
    decision, so the import list is pinned rather than spot-checked.
    """
    tree = ast.parse(MODULE_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {"__future__", "chimera.recorder.events"}, imported

    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"now", "utcnow", "today", "time_ns", "monotonic"}


def test_seconds_form_matches_the_nanosecond_form() -> None:
    exact = RunnerClock(start_ns=1_500_000_000_000_000_000)
    assert exact.time() == 1_500_000_000.0

    clock = RunnerClock(start_ns=1_795_312_920_123_456_789)
    # Correctly rounded, computed a different way from the implementation.
    assert clock.time() == float(Fraction(clock.now_ns, 1_000_000_000))
    # And the loss is bounded by the float resolution at this magnitude, not by
    # anything the clock chose to drop.
    assert abs(Fraction(clock.time()) - Fraction(clock.now_ns, 1_000_000_000)) < Fraction(
        1, 1_000_000
    )


def test_seconds_form_is_non_decreasing_across_the_recording() -> None:
    clock = RunnerClock()
    seconds = []
    for receipt in RECEIPTS:
        clock.observe(receipt)
        seconds.append(clock.time())
    assert seconds == sorted(seconds)


@pytest.mark.parametrize(
    "bad",
    [
        True,
        1.0,
        "1795312920123456789",
        None,
        -1,
        # Milliseconds read as nanoseconds: an instant past 2100, which is the
        # unit mistake the recorder's bound exists to catch.
        4_102_444_800 * 1_000_000_000 + 1,
    ],
)
def test_a_receipt_that_is_not_a_nanosecond_instant_is_refused(bad: object) -> None:
    clock = RunnerClock()
    with pytest.raises(RunnerClockError):
        clock.observe(bad)  # type: ignore[arg-type]
    # And the refusal left the clock exactly as it was.
    assert clock.started is False


def test_a_start_that_is_not_a_nanosecond_instant_is_refused() -> None:
    with pytest.raises(RunnerClockError, match="start_ns"):
        RunnerClock(start_ns=1.5)  # type: ignore[arg-type]


def test_repr_says_whether_the_clock_has_started() -> None:
    assert repr(RunnerClock()) == "RunnerClock(unstarted)"
    assert repr(RunnerClock(start_ns=7_000_000_000)) == "RunnerClock(7000000000)"


def test_the_one_dependency_reads_no_clock_either() -> None:
    """The import barrier above is only as strong as what it lets through.

    ``clock.py`` is pinned to exactly one dependency,
    ``chimera.recorder.events``, and every property the clock has then rests on
    ``require_canonical_ns``. That function bounds a receipt against two fixed
    constants today, so section 2.4 holds — but the obvious future edit, bounding
    a receipt against "now" instead of against a fixed 2100 ceiling, is a natural
    thing for a recorder to want and would silently make ``RunnerClock`` outputs
    depend on the host clock without failing anything in this file. So the
    dependency is pinned as well as the importer: the bounds are constants, and
    the module reads no clock.
    """
    assert events.MIN_CANONICAL_NS == 0
    assert events.MAX_CANONICAL_NS == 4_102_444_800 * events.NS_PER_SECOND

    source = Path(events.__file__)
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not imported & {"time", "calendar"}, imported

    # `datetime` is imported for rendering, so the barrier there is on the calls:
    # nothing in the module may ask any of them what time it is now.
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {
        "now",
        "utcnow",
        "today",
        "fromtimestamp",
        "utcfromtimestamp",
        "time_ns",
        "monotonic",
        "monotonic_ns",
        "perf_counter",
    }, attributes
