"""The pre-allocated emergency record: the trace a persistence failure leaves (R1-i).

The adopted roadmap's R1-i, last clause:

    on any persistence failure (ENOSPC included) the runner exits non-zero
    after attempting to append the halt reason to a pre-allocated emergency
    record, so a restart never lacks a trace.

**Why a separate file, and why pre-allocated.** When the disk is full, every
ordinary writer here needs NEW blocks: the atomic writers create a temporary
file, and the decision log grows. A trace written the same way fails for the
same reason as the write it is trying to report. So this file is created, at
its full fixed size, by :func:`arm` at the start of every writing run -- before
anything that could fail has run -- and a failure is written by :func:`trip` as
an in-place rewrite of bytes that already exist (``r+b``, seek, write, flush,
``fsync``). Nothing grows, nothing is created, nothing is renamed.

What that does and does not guarantee is stated, not implied. On a filesystem
that rewrites blocks in place (ext4, XFS, NTFS), rewriting allocated bytes needs
no new block, so the trace can be written when the disk is otherwise full. On a
copy-on-write filesystem (btrfs, ZFS) even an in-place rewrite may need a fresh
block, and it can fail; so can a disk that has failed outright. :func:`trip`
therefore returns whether it succeeded, and the caller exits non-zero either
way.

**Two slots, 2048 bytes each.** The FIRST trip since the record was last armed
goes to slot 0 and is never overwritten; any later one goes to slot 1, which
holds the latest. So a second failure -- in a restart, or in the command an
operator runs to look -- can neither erase the first nor be silently dropped.
Only :func:`rearm`, the one operator path (the CLI's ``resolve --emergency``),
clears them, and the runner copies the whole trace into its decision log before
it does.

**Format.** Each slot is one canonical JSON object, then spaces up to the slot
size, then a newline. A slot that does not parse, a file of the wrong size, or a
file missing its header is UNREADABLE, which is treated as a tripped record: a
partial trace is still evidence that something went wrong, and "the record could
not be read" is never read as "nothing happened".

**What it is not.** Not scientific evidence, not a decision record, not part of
any hash the campaign's identity or replay depends on, and never read by a
decision. It carries the failure's errno symbol, the file's ROLE (its name,
never a host path), the runner's state and halt reason, and where the log had
got to -- enough to name the failure class and the transition, and nothing a
host or platform would spell differently.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "EMERGENCY_NAME",
    "EMERGENCY_SCHEMA",
    "FILE_BYTES",
    "SLOT_BYTES",
    "EmergencyState",
    "EmergencyTrace",
    "arm",
    "read_trace",
    "rearm",
    "trip",
]

#: The file, under the runner's state directory.
EMERGENCY_NAME = "emergency_record.bin"
#: Named in every slot, so a file of the right size holding something else is
#: not mistaken for this record.
EMERGENCY_SCHEMA = "chimera.demo-emergency/1"
#: One slot. Big enough for the bounded payload with room to spare.
SLOT_BYTES = 2048
SLOTS = 2
FILE_BYTES = SLOT_BYTES * SLOTS
#: Every free-text field is cut to this, so a payload always fits a slot.
_TEXT_LIMIT = 400

_ARMED = "ARMED"
_TRIPPED = "TRIPPED"


class EmergencyState(str, Enum):
    """What :func:`read_trace` found."""

    #: No file: never armed in this state directory.
    ABSENT = "ABSENT"
    #: Armed and untouched: no failure recorded since it was last armed.
    ARMED = "ARMED"
    #: At least one failure is recorded.
    TRIPPED = "TRIPPED"
    #: Present and not a well-formed record. Treated as TRIPPED.
    UNREADABLE = "UNREADABLE"


@dataclass(frozen=True)
class EmergencyTrace:
    """The record as read, read-only. ``slots`` holds the tripped payloads."""

    state: EmergencyState
    slots: tuple[Mapping[str, Any], ...] = ()
    #: sha256 of the file's bytes, so a restart can tell whether it already
    #: recorded THIS trace in the decision log. Empty when there is no file.
    digest: str = ""
    detail: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def needs_attention(self) -> bool:
        """Whether a failure (or an unreadable record) is waiting for an operator."""
        return self.state in (EmergencyState.TRIPPED, EmergencyState.UNREADABLE)

    def summary(self) -> str:
        """One line for a halt reason: the first failure's class and transition."""
        if self.state is EmergencyState.UNREADABLE:
            return f"the emergency record is unreadable ({self.detail})"
        if not self.slots:
            return "no failure recorded"
        first = self.slots[0]
        failure = first.get("failure") or {}
        return (
            f"{failure.get('errno', '?')} on {failure.get('operation', '?')} of "
            f"{failure.get('role', '?')} during {first.get('runner_state', '?')}"
        )

    def status_block(self) -> dict[str, Any]:
        """For the CLI's ``status``."""
        return {
            "state": self.state.value,
            "digest": self.digest or None,
            "slots": [dict(slot) for slot in self.slots],
            **({"detail": self.detail} if self.detail else {}),
        }


def path_for(state_dir: str | Path) -> Path:
    return Path(state_dir) / EMERGENCY_NAME


def _slot(payload: Mapping[str, Any]) -> bytes:
    """One slot's exact bytes: canonical JSON, space padding, newline."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    raw = body.encode("ascii")
    if len(raw) > SLOT_BYTES - 1:
        raise ValueError(f"emergency payload of {len(raw)} bytes does not fit a slot")
    return raw + b" " * (SLOT_BYTES - 1 - len(raw)) + b"\n"


def _armed_slot(index: int) -> bytes:
    return _slot({"schema": EMERGENCY_SCHEMA, "slot": index, "state": _ARMED})


def _armed_bytes() -> bytes:
    return b"".join(_armed_slot(i) for i in range(SLOTS))


def _bounded(value: Any) -> Any:
    """Cut free text so that a payload always fits its slot."""
    if isinstance(value, str):
        return value if len(value) <= _TEXT_LIMIT else value[: _TEXT_LIMIT - 3] + "..."
    if isinstance(value, Mapping):
        return {str(k): _bounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded(v) for v in value]
    return value


def read_trace(state_dir: str | Path) -> EmergencyTrace:
    """Read the record. Writes nothing, under any outcome."""
    location = path_for(state_dir)
    try:
        data = location.read_bytes()
    except FileNotFoundError:
        return EmergencyTrace(EmergencyState.ABSENT)
    except OSError as exc:
        return EmergencyTrace(
            EmergencyState.UNREADABLE, detail=f"could not be read: {type(exc).__name__}"
        )
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if len(data) != FILE_BYTES:
        return EmergencyTrace(
            EmergencyState.UNREADABLE,
            digest=digest,
            detail=f"{len(data)} bytes where {FILE_BYTES} were allocated",
        )
    tripped: list[Mapping[str, Any]] = []
    for index in range(SLOTS):
        chunk = data[index * SLOT_BYTES : (index + 1) * SLOT_BYTES]
        try:
            payload = json.loads(chunk.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError):
            return EmergencyTrace(
                EmergencyState.UNREADABLE, digest=digest, detail=f"slot {index} does not parse"
            )
        if not isinstance(payload, dict) or payload.get("schema") != EMERGENCY_SCHEMA:
            return EmergencyTrace(
                EmergencyState.UNREADABLE,
                digest=digest,
                detail=f"slot {index} is not a {EMERGENCY_SCHEMA} slot",
            )
        if payload.get("state") == _TRIPPED:
            tripped.append(payload)
        elif payload.get("state") != _ARMED:
            return EmergencyTrace(
                EmergencyState.UNREADABLE,
                digest=digest,
                detail=f"slot {index} is in no known state",
            )
    if tripped:
        return EmergencyTrace(EmergencyState.TRIPPED, slots=tuple(tripped), digest=digest)
    return EmergencyTrace(EmergencyState.ARMED, digest=digest)


def _fsync_directory(directory: Path) -> None:
    try:
        handle = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory handles
        return
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _write_fresh(location: Path) -> None:
    """Create the armed record at full size: temporary, fsync, replace, dir fsync."""
    location.parent.mkdir(parents=True, exist_ok=True)
    temporary = location.with_suffix(location.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(_armed_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, location)
    _fsync_directory(location.parent)


def arm(state_dir: str | Path) -> bool:
    """Make sure the reserve exists before anything that may fail. True if written.

    Writes only when there is NO record. An armed record is left exactly as it
    is (nothing to do), and a tripped or unreadable one is left exactly as it is
    too: that is the evidence of the last failure, and only :func:`rearm` -- the
    operator's acknowledgement -- replaces it. Raises ``OSError`` when the
    reserve cannot be created; the caller must not go on to process anything.
    """
    location = path_for(state_dir)
    if read_trace(state_dir).state is not EmergencyState.ABSENT:
        return False
    _write_fresh(location)
    return True


def trip(state_dir: str | Path, payload: Mapping[str, Any]) -> bool:
    """Write one failure into the reserve, in place. True only if it was fsynced.

    Never raises: it is called on the way out of a process whose disk has
    already failed, and an exception here would replace the report of the real
    failure with a report of this one. Refuses to write at all -- returns False
    -- when the file is not the full-size record :func:`arm` created, because
    writing into a shorter file would have to GROW it, which is the one thing a
    full disk cannot do.
    """
    location = path_for(state_dir)
    trace = read_trace(state_dir)
    if trace.state is EmergencyState.ABSENT:
        return False
    # Slot 0 keeps the FIRST failure since the record was armed; everything
    # after it goes to slot 1. An unreadable record is not overwritten at slot
    # 0 either: it may be a partial first trace.
    index = 0 if trace.state is EmergencyState.ARMED else 1
    body = {
        **{str(k): _bounded(v) for k, v in payload.items()},
        "schema": EMERGENCY_SCHEMA,
        "slot": index,
        "state": _TRIPPED,
    }
    try:
        data = _slot(body)
    except ValueError:
        data = _slot(
            {
                "schema": EMERGENCY_SCHEMA,
                "slot": index,
                "state": _TRIPPED,
                "failure": _bounded(dict(payload.get("failure") or {})),
                "truncated": True,
            }
        )
    try:
        with open(location, "r+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() != FILE_BYTES:
                return False
            handle.seek(index * SLOT_BYTES)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        return False
    return True


def rearm(state_dir: str | Path) -> None:
    """The operator's acknowledgement: replace a tripped record with an armed one.

    Only the CLI's ``resolve --emergency`` calls this, and only after the trace
    has been copied into the decision log. Raises ``OSError`` on failure.
    """
    _write_fresh(path_for(state_dir))
