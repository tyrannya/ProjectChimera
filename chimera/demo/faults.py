"""Fault schedules: deliberate breakage, applied by tests and drills only.

Section 12.3 line 1443: "`FaultSchedule`, applied by tests and soak drills
only". `chimera/demo/config.py`'s own docstring already points here -- "The
block's own schema belongs to `chimera/demo/faults.py`, which is PR-10's" --
and `parse_demo_config` carries the block opaquely into the config hash without
interpreting it, so this module owns the meaning and the parser owns nothing.

Two properties matter and are enforced rather than documented:

**A schedule is inert unless a caller applies it.** Nothing in the runner reads
a schedule. The runner is handed already-broken inputs by a test, exactly as it
would be handed genuinely broken inputs by a degraded recorder, so no
fault-injection branch exists on the production path to be left switched on.

**A schedule is data, not code.** Faults are named by an enum and scheduled by
minute; there is no callable to smuggle behaviour through, so a config file can
describe a drill but cannot execute anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

__all__ = ["Fault", "FaultSchedule", "FaultError"]


class FaultError(ValueError):
    """A fault schedule that cannot be read is refused rather than ignored."""


class Fault(str, Enum):
    """The failures section 10's acceptance run has to survive.

    Each names a real thing the demo path can meet: a minute the recorder never
    wrote, a minute without a book, a stale feed, a venue that refuses, a leg
    that fills while its partner does not, a store that will not save, and an
    operator pulling the kill switch.
    """

    MISSING_MINUTE = "missing_minute"
    MISSING_BOOK = "missing_book"
    STALE_FEED = "stale_feed"
    VENUE_REJECT = "venue_reject"
    PARTIAL_FILL = "partial_fill"
    STORE_ERROR = "store_error"
    KILL_SWITCH = "kill_switch"

    @classmethod
    def parse(cls, value: Any) -> "Fault":
        try:
            return cls(str(value))
        except ValueError as exc:
            known = ", ".join(sorted(f.value for f in cls))
            raise FaultError(f"unknown fault {value!r}; known faults are {known}") from exc


@dataclass(frozen=True)
class ScheduledFault:
    """One fault, at one minute offset from the run's first minute."""

    minute_offset: int
    fault: Fault
    detail: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.minute_offset < 0:
            raise FaultError("a fault offset is measured forward from the first minute")
        object.__setattr__(self, "detail", dict(self.detail or {}))


class FaultSchedule:
    """Which faults happen at which minute offsets.

    Offsets rather than wall-clock instants, so the same schedule describes the
    same drill whichever day the fixture generates -- and so a schedule read
    from a config file cannot smuggle in a date.
    """

    def __init__(self, faults: Iterable[ScheduledFault] = ()) -> None:
        self._by_offset: dict[int, list[ScheduledFault]] = {}
        for scheduled in faults:
            self._by_offset.setdefault(scheduled.minute_offset, []).append(scheduled)

    @classmethod
    def from_config(cls, block: Mapping[str, Any] | None) -> "FaultSchedule":
        """Read the `faults` block `parse_demo_config` carried through opaquely."""
        if not block:
            return cls()
        entries = block.get("schedule")
        if entries is None:
            return cls()
        if not isinstance(entries, (list, tuple)):
            raise FaultError("faults.schedule must be a list of {minute_offset, fault}")
        scheduled = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise FaultError(
                    f"a fault entry must be a mapping, got {type(entry).__name__}"
                )
            if "minute_offset" not in entry or "fault" not in entry:
                raise FaultError("a fault entry needs both minute_offset and fault")
            scheduled.append(
                ScheduledFault(
                    minute_offset=int(entry["minute_offset"]),
                    fault=Fault.parse(entry["fault"]),
                    detail=entry.get("detail") or {},
                )
            )
        return cls(scheduled)

    # --- reading ----------------------------------------------------------
    def at(self, minute_offset: int) -> tuple[ScheduledFault, ...]:
        return tuple(self._by_offset.get(int(minute_offset), ()))

    def faults_at(self, minute_offset: int) -> frozenset[Fault]:
        return frozenset(s.fault for s in self.at(minute_offset))

    def __contains__(self, fault: object) -> bool:
        return any(s.fault == fault for group in self._by_offset.values() for s in group)

    def __len__(self) -> int:
        return sum(len(group) for group in self._by_offset.values())

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_offset))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": [
                {
                    "minute_offset": s.minute_offset,
                    "fault": s.fault.value,
                    "detail": dict(s.detail),
                }
                for offset in self.offsets
                for s in self._by_offset[offset]
            ]
        }

    # --- applying, for fixtures -------------------------------------------
    def minute_shapes(self, market: str) -> dict[int, Any]:
        """Translate feed-shaped faults into `chimera.demo.fixtures.MinuteShape`.

        Only the two faults that are properties OF THE FILES are expressible
        this way. The rest are conditions a test arranges around the runner --
        a venue that refuses, a store that will not save -- and they are not
        smuggled in here, because a fixture that could fake them would be
        testing itself rather than the runner.
        """
        from chimera.demo.fixtures import MinuteShape

        shapes: dict[int, Any] = {}
        for offset in self.offsets:
            for scheduled in self._by_offset[offset]:
                if scheduled.fault is Fault.MISSING_MINUTE:
                    shapes[offset] = MinuteShape(present=False)
                elif scheduled.fault is Fault.MISSING_BOOK:
                    shapes[offset] = MinuteShape(book=False)
        return shapes
