"""The runner's clock, derived from what was recorded rather than from the wall.

A demo runner that read ``time.time()`` to decide whether its feed was stale
would produce a decision that cannot be replayed: run the same recorded minutes
again tomorrow and the staleness veto fires differently, so the log written
today and the log written by the replay disagree about a value nobody can
reconstruct. Section 2.4 of the adopted plan removes the wall clock from that
path entirely. The decision clock is this class, and its whole definition is

    ``now_ns = max(receipt_ns seen so far)``

advanced monotonically as records are read from the recorder's files. Feed
staleness becomes ``now_ns - canonical_ns`` of the last complete minute, which is
a difference between two numbers that are both in the recording, so the same
files replayed produce the same sequence of vetoes for ever.

**Monotone means a lower receipt is ignored, not an error.** Observations do not
arrive in receipt order — a reconnect replays frames, a REST poll backfills a gap
the websocket missed — and a clock that stepped backwards for one of them would
make a minute that was already judged fresh become stale again. So
:meth:`RunnerClock.observe` takes the maximum and returns the clock's value,
never the argument's.

**The clock starts with no time, and says so.** Before the first observation
there is no recorded instant to be "now", and inventing one — the epoch, or the
host's clock — would be exactly the wall-clock dependency this class exists to
remove. :attr:`RunnerClock.now_ns` raises until the clock has been started,
either by an observation or by an explicit ``start_ns`` the caller carries over
from persisted state. A runner that has read nothing has not decided anything
yet, so there is nothing for the missing value to break.

**This module reads no clock at all.** Not ``time.time``, not ``time.monotonic``,
not ``datetime.now``. Scheduling the next wake-up and stamping the heartbeat file
are wall-clock work and they belong to the runner and to operations; nothing
here can be given a different answer by changing the host's time, which is what
``tests/test_demo_clock.py`` asserts by denying every clock in the process and
running the same recording twice.
"""

from __future__ import annotations

from chimera.recorder.events import NS_PER_SECOND, RecorderEventError, require_canonical_ns


class RunnerClockError(ValueError):
    """The runner clock was given, or asked for, a time it cannot stand behind."""


def _require_receipt_ns(value: object, *, field: str) -> int:
    """An integer UTC nanosecond instant, under the recorder's own bounds.

    Delegated rather than re-derived: the recorder already refuses a bool, a
    float, a negative instant and a magnitude far enough from now to mean a unit
    was mistaken, and two modules disagreeing about what a nanosecond instant is
    would be a way for a value the recorder rejected to become the runner's
    decision time.
    """
    try:
        return require_canonical_ns(value, field=field)
    except RecorderEventError as exc:
        raise RunnerClockError(str(exc)) from exc


class RunnerClock:
    """The decision clock: the highest receipt timestamp observed so far.

    Not thread-safe, deliberately: the runner's tick loop is one thread
    (section 8.2), and a clock that could be advanced from a background thread
    would make the decision time depend on scheduling.
    """

    __slots__ = ("_now_ns",)

    def __init__(self, start_ns: int | None = None) -> None:
        """Start unset, or at an instant the caller is carrying over.

        ``start_ns`` exists for a restart: the runner knows the instant its last
        persisted record was written and can hand it back, so the clock resumes
        where the campaign left it instead of appearing to jump backwards for
        every observation already consumed before the crash.
        """
        self._now_ns: int | None = (
            None if start_ns is None else _require_receipt_ns(start_ns, field="start_ns")
        )

    @property
    def started(self) -> bool:
        """Whether the clock has an instant. False until the first observation."""
        return self._now_ns is not None

    @property
    def now_ns(self) -> int:
        """The decision instant, in integer nanoseconds since the UTC epoch."""
        if self._now_ns is None:
            raise RunnerClockError(
                "the runner clock has observed nothing, so it has no instant. A decision "
                "time that was never recorded is not a value to guess at: read a record "
                "first, or construct the clock with the start_ns carried over from "
                "persisted state"
            )
        return self._now_ns

    def observe(self, receipt_ns: int) -> int:
        """Advance the clock to ``receipt_ns`` if that is later, and return it.

        Returns the clock, not the argument. A caller that used the return value
        as "the time of this observation" would otherwise silently get the
        clock's value for an in-order record and the record's own value for a
        late one, which is two different meanings for one expression.
        """
        candidate = _require_receipt_ns(receipt_ns, field="receipt_ns")
        if self._now_ns is None or candidate > self._now_ns:
            self._now_ns = candidate
        return self._now_ns

    def time(self) -> float:
        """The clock in seconds, for the injected-clock APIs that take one.

        ``RiskEngine(clock=...)`` and ``FuturesExecutor(clock=...)`` both take a
        zero-argument callable returning seconds as a float, so the runner
        injects this bound method and those two components stop reading the wall
        clock as well.

        The conversion is lossy and that is stated rather than hidden: a float
        holds about 238 ns of resolution at present-day epoch seconds, so the
        seconds form cannot distinguish two instants closer together than that.
        It is deterministic — the same ``now_ns`` gives the same float on every
        platform — which is what replay needs. :attr:`now_ns` remains the
        authoritative value and is what the decision log records.
        """
        return self.now_ns / NS_PER_SECOND

    def __repr__(self) -> str:
        state = "unstarted" if self._now_ns is None else str(self._now_ns)
        return f"RunnerClock({state})"


__all__ = ["RunnerClock", "RunnerClockError"]
