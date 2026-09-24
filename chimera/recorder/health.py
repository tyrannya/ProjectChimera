"""What the recorder reports about itself: metrics, and the heartbeat file.

Two audiences, one set of facts. Prometheus scrapes the series section 4.8
names; a human — or a supervisor, or a reviewer checking a 60-minute run —
reads ``health/heartbeat.json``. Both are written from the same
:class:`RecorderHealth` snapshot, so the dashboard and the file cannot disagree
about whether the recorder is up.

**Reporting never decides anything.** Nothing in this module is read back by the
sink, the parsers, the normalizer or the service's control flow. It observes
counters that already exist, computes ages and a median, and writes them out. A
metric that fed back into a recording decision would make the recorder's output
a function of its own monitoring, which is a thing that cannot be reproduced.

**No economics, by construction.** Every field below counts observations,
connections, files, bytes and clocks. There is no price, no return, no funding
flow, no basis and no profitability anywhere in the heartbeat or in the metric
family — a recorder that published how a recorded price had *moved* would be
computing an economic quantity, and this one computes none.

**The heartbeat is written atomically**, with
:func:`chimera.recorder.sink.write_json_atomic`: a supervisor that reads the
file while it is being replaced sees the previous heartbeat, never half of the
next one. A truncated heartbeat would read as "the recorder is broken" at
exactly the moments it is busiest.
"""

from __future__ import annotations

import json
import os
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from chimera import metrics
from chimera.recorder.contract import RecorderContract
from chimera.recorder.events import NS_PER_SECOND, iso_utc
from chimera.recorder.sink import write_json_atomic

logger = logging.getLogger(__name__)


class RecorderHealthError(RuntimeError):
    """A health document on disk cannot be read as what it claims to be."""


#: Schema of the document below. Versioned like every other persisted shape in
#: this package, so a reader can refuse one it does not understand.
HEARTBEAT_SCHEMA = "chimera.recorder-heartbeat/1"

#: Where it is written, under the storage root.
HEALTH_DIRECTORY = "health"

#: The recorder's own append-only log of starting and stopping (R1-h).
#:
#: **Why the heartbeat is not enough.** ``heartbeat.json`` is replaced in place
#: and says what the recorder is doing now. It therefore cannot say that the
#: recorder ever stopped: a process that dies leaves its last heartbeat behind,
#: and a reader coming back later sees a stale timestamp and has to guess
#: whether the recorder died, the host clock moved, or the file was copied from
#: somewhere else. The gap in the data is real, and nothing in the recording
#: names it.
#:
#: This file names it. One line per lifecycle transition, appended and never
#: rewritten, so a stop leaves a record even when the stop was not the
#: recorder's idea: a ``recorder.up`` with no matching ``recorder.down`` after
#: it is exactly the signature of a kill, and the next start says so in its own
#: line rather than leaving a reader to infer it.
#:
#: Engineering evidence about the RECORDER, never about the market. No price, no
#: economic quantity, nothing a coverage gate or a reconciliation reads, and
#: outside every value digest: deleting it loses operational history and changes
#: no recorded value.
LIFECYCLE_FILE = "lifecycle.ndjson"

#: The lifecycle events, and what each one asserts.
LIFECYCLE_UP = "recorder.up"
LIFECYCLE_DOWN = "recorder.down"
#: Written by a START that found the previous line was an ``up``. It asserts
#: only what can be established from the file -- the previous run wrote no
#: ``down`` -- and never why, because this process was not there.
LIFECYCLE_UNCLEAN = "recorder.unclean_stop"

#: The lifecycle document's schema string.
LIFECYCLE_SCHEMA = "chimera.recorder-lifecycle/1"
HEARTBEAT_FILE = "heartbeat.json"

#: The adopted cadence, section 4.3's storage layout: "rewritten every 30 s".
HEARTBEAT_INTERVAL_S = 30.0

#: Skew above this is worth an alert, per section 4.1. Reported, never applied.
SKEW_ALERT_MS = 5_000.0


@dataclass
class StreamHealth:
    """Everything known about one stream, and nothing derived from a price."""

    stream: str
    connected: bool = False
    events: int = 0
    duplicates: int = 0
    late: int = 0
    reconnects: int = 0
    gapfill_rows: int = 0
    write_errors: int = 0
    out_of_order: int = 0
    decode_errors: int = 0
    missing_minutes: int | None = None
    last_event_ns: int | None = None
    #: Set once, by a storage failure, and never cleared inside a run. Section
    #: 2.2's A -> B arrow: "write error -> recorder halts that stream, health
    #: metric flips, alert". A recorder that had lost the ability to write one
    #: stream durably and kept reporting it up would be presenting silence as
    #: data, so the halt is latched rather than retried: the connection may be
    #: perfectly healthy and the storage still gone, and only a restart — which
    #: runs the whole recovery path — may lift it.
    halted: bool = False
    halt_reason: str | None = None
    halted_at_ns: int | None = None
    #: Observations refused because the stream was already halted.
    dropped_after_halt: int = 0

    @property
    def up(self) -> bool:
        """Whether this stream is both receiving *and* able to store what it receives.

        ``connected`` is a fact about a socket and stays truthful about one.
        This is the operational answer, and it is what
        ``chimera_recorder_up`` publishes: a stream whose sink has failed is not
        up, however well its connection is doing.
        """
        return self.connected and not self.halted

    def halt(self, reason: str, *, now_ns: int | None = None) -> bool:
        """Latch a storage failure. Returns whether this call was the one that did.

        Idempotent in the flag and in the reason — the first failure is the one
        worth reading, because everything after it is a consequence — while the
        write-error counter still counts every distinct failure.
        """
        self.write_errors += 1
        if self.halted:
            return False
        self.halted = True
        self.halt_reason = reason
        self.halted_at_ns = now_ns
        return True

    def age_seconds(self, now_ns: int) -> float | None:
        """Now minus the canonical time of the last observation.

        ``None`` while nothing has been seen: a stream that has never delivered
        an event has no age, and reporting zero would read as "perfectly fresh".
        """
        if self.last_event_ns is None:
            return None
        return max(0.0, (now_ns - self.last_event_ns) / NS_PER_SECOND)

    def to_dict(self, now_ns: int) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "up": self.up,
            "connected": self.connected,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "halted_at_ns": self.halted_at_ns,
            "dropped_after_halt": self.dropped_after_halt,
            "events": self.events,
            "duplicates": self.duplicates,
            "late": self.late,
            "reconnects": self.reconnects,
            "gapfill_rows": self.gapfill_rows,
            "write_errors": self.write_errors,
            "out_of_order": self.out_of_order,
            "decode_errors": self.decode_errors,
            "missing_minutes": self.missing_minutes,
            "last_event_ns": self.last_event_ns,
            "last_event_utc": (
                None if self.last_event_ns is None else iso_utc(self.last_event_ns)
            ),
            "last_event_age_seconds": self.age_seconds(now_ns),
        }


@dataclass
class RecorderHealth:
    """One snapshot of the recorder's operational state.

    Built by the service on every heartbeat, published to Prometheus and written
    to disk. It carries provenance — which contract is in force, whether the
    prospective boundary has been fixed — because an operator looking at a
    running recorder must be able to tell engineering data from evidence without
    reading the code.
    """

    contract_id: str
    contract_hash: str
    prospective_from: str | None
    streams: dict[str, StreamHealth] = field(default_factory=dict)
    clock_skew_ms: float | None = None
    disk_free_bytes: int | None = None
    started_ns: int | None = None
    heartbeat_ns: int | None = None
    open_day: str | None = None
    normalized_day: str | None = None
    source_revision: str | None = None
    errors: tuple[str, ...] = ()

    def stream(self, name: str) -> StreamHealth:
        """The record for one stream, created on first use."""
        health = self.streams.get(name)
        if health is None:
            health = StreamHealth(stream=name)
            self.streams[name] = health
        return health

    @property
    def write_errors(self) -> int:
        """Failed writes across every stream. The acceptance criterion reads this."""
        return sum(health.write_errors for health in self.streams.values())

    @property
    def halted_streams(self) -> tuple[str, ...]:
        """Streams whose storage failed and which are recording nothing."""
        return tuple(sorted(name for name, health in self.streams.items() if health.halted))

    @property
    def evidence_class(self) -> str:
        """What a minute recorded right now would be.

        ``engineering`` until the contract carries a committed
        ``prospective_from``. Written into the heartbeat so that the answer is
        visible on the running host and does not have to be inferred.
        """
        return "prospective" if self.prospective_from else "engineering"

    def to_document(self, now_ns: int) -> dict[str, Any]:
        """The heartbeat, exactly as it is persisted."""
        return {
            "schema": HEARTBEAT_SCHEMA,
            "heartbeat_ns": now_ns,
            "heartbeat_utc": iso_utc(now_ns),
            "started_ns": self.started_ns,
            "uptime_seconds": (
                None if self.started_ns is None else (now_ns - self.started_ns) / NS_PER_SECOND
            ),
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "prospective_from": self.prospective_from,
            "evidence_class": self.evidence_class,
            "source_revision": self.source_revision,
            "open_day": self.open_day,
            "normalized_day": self.normalized_day,
            "clock_skew_ms": self.clock_skew_ms,
            "clock_skew_alert": (
                self.clock_skew_ms is not None and abs(self.clock_skew_ms) > SKEW_ALERT_MS
            ),
            "disk_free_bytes": self.disk_free_bytes,
            "write_errors": self.write_errors,
            "halted_streams": list(self.halted_streams),
            "errors": list(self.errors),
            "streams": [self.streams[name].to_dict(now_ns) for name in sorted(self.streams)],
        }


def read_heartbeat(root: str | Path) -> dict[str, Any] | None:
    """The last heartbeat written under ``root``, or ``None`` if there is none."""
    path = heartbeat_path(root)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecorderHealthError(f"{path} is not readable as a heartbeat: {exc}") from exc
    if not isinstance(document, Mapping):
        raise RecorderHealthError(f"{path} does not hold a heartbeat object")
    if document.get("schema") != HEARTBEAT_SCHEMA:
        raise RecorderHealthError(
            f"{path} declares schema {document.get('schema')!r}; this build reads "
            f"{HEARTBEAT_SCHEMA!r}"
        )
    return dict(document)


def read_status(
    contract: RecorderContract, root: str | Path, *, now_ns: int
) -> dict[str, Any]:
    """What is on disk under ``root``, without starting anything.

    Strictly read-only: it opens no socket, starts no task and writes no file.
    Everything below is a fact about files this host already holds — which days
    exist, which are frozen, when the last heartbeat was written, how many
    minutes of the newest normalized day have no row. Nothing is fetched to
    answer it and nothing is repaired by asking.
    """
    from chimera.recorder.sink import RawSink, available_days

    base = Path(root)
    heartbeat = read_heartbeat(base)
    provenance = contract.provenance()
    streams: list[dict[str, Any]] = []
    for name in contract.streams:
        sink = RawSink(base, name, contract=contract)
        days = available_days(base, name)
        streams.append(
            {
                "stream": name,
                "days": days,
                "last_day": days[-1] if days else None,
                "frozen_days": [day for day in days if sink.is_frozen(day)],
            }
        )
    markets: list[dict[str, Any]] = []
    for key in contract.market_keys():
        directory = base / "normalized" / key / "1m"
        metas = sorted(directory.glob("*.meta.json")) if directory.is_dir() else []
        newest = metas[-1] if metas else None
        summary: dict[str, Any] = {
            "market": key,
            "normalized_days": [path.name.split(".")[0] for path in metas],
            "last_day": None if newest is None else newest.name.split(".")[0],
            "rows": None,
            "missing_minutes": None,
            "frozen": None,
        }
        if newest is not None:
            try:
                document = json.loads(newest.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RecorderHealthError(f"{newest} is not readable: {exc}") from exc
            summary["rows"] = document.get("rows")
            summary["missing_minutes"] = len(document.get("missing") or [])
            summary["frozen"] = newest.with_suffix("").with_suffix(".sha256").exists()
        markets.append(summary)
    settlements = base / "funding" / "um" / "settlements.ndjson"
    return {
        "root": str(base),
        "exists": base.is_dir(),
        "contract_id": contract.contract_id,
        "contract_hash": contract.contract_hash,
        "prospective_from": provenance.get("prospective_from"),
        "evidence_class": (
            "prospective" if provenance.get("prospective_from") else "engineering"
        ),
        "heartbeat": heartbeat,
        "heartbeat_age_seconds": (
            None
            if heartbeat is None or not isinstance(heartbeat.get("heartbeat_ns"), int)
            else max(0.0, (now_ns - int(heartbeat["heartbeat_ns"])) / NS_PER_SECOND)
        ),
        "streams": streams,
        "markets": markets,
        "settlements_rows": (
            sum(1 for line in settlements.read_bytes().splitlines() if line.strip())
            if settlements.exists()
            else 0
        ),
        "disk_free_bytes": disk_free_bytes(base),
    }


def heartbeat_path(root: str | Path) -> Path:
    """``<root>/health/heartbeat.json``."""
    return Path(root) / HEALTH_DIRECTORY / HEARTBEAT_FILE


def lifecycle_path(root: str | Path) -> Path:
    """``<root>/health/lifecycle.ndjson``."""
    return Path(root) / HEALTH_DIRECTORY / LIFECYCLE_FILE


def disk_free_bytes(root: str | Path) -> int | None:
    """Free bytes on the filesystem holding ``root``, or ``None`` if unknowable."""
    try:
        return int(shutil.disk_usage(Path(root)).free)
    except OSError as exc:  # pragma: no cover - platform dependent
        logger.warning("could not read free disk space for %s: %s", root, exc)
        return None


def initial_health(
    contract: RecorderContract, *, source_revision: str | None = None
) -> RecorderHealth:
    """A snapshot with the contract's provenance and one entry per stream."""
    provenance = contract.provenance()
    health = RecorderHealth(
        contract_id=contract.contract_id,
        contract_hash=contract.contract_hash,
        prospective_from=provenance.get("prospective_from"),
        source_revision=source_revision,
    )
    for name in contract.streams:
        health.stream(name)
    return health


def publish(
    health: RecorderHealth, *, now_ns: int, previous: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Push one snapshot into the Prometheus series and return the new totals.

    Counters are monotonic in Prometheus, so this publishes the *increment*
    since the previous call rather than the absolute count, and returns the
    totals to pass back next time. Gauges are set outright.
    """
    totals: dict[str, Any] = {}
    seen = previous or {}
    for name in sorted(health.streams):
        stream = health.streams[name]
        # `up`, not `connected`: a stream that cannot store what it receives is
        # not up, and section 2.2 requires the health metric to flip on a write
        # error rather than on a disconnection alone.
        metrics.RECORDER_UP.labels(stream=name).set(1 if stream.up else 0)
        age = stream.age_seconds(now_ns)
        if age is not None:
            metrics.RECORDER_LAST_EVENT_AGE.labels(stream=name).set(age)
        if stream.missing_minutes is not None:
            metrics.RECORDER_MISSING_MINUTES.labels(stream=name).set(stream.missing_minutes)
        for field_name, counter in (
            ("events", metrics.RECORDER_EVENTS),
            ("duplicates", metrics.RECORDER_DUPLICATES),
            ("late", metrics.RECORDER_LATE),
            ("reconnects", metrics.RECORDER_RECONNECTS),
            ("gapfill_rows", metrics.RECORDER_GAPFILL_ROWS),
            ("write_errors", metrics.RECORDER_WRITE_ERRORS),
        ):
            total = int(getattr(stream, field_name))
            key = f"{name}/{field_name}"
            delta = total - int(seen.get(key, 0))
            if delta > 0:
                counter.labels(stream=name).inc(delta)
            totals[key] = total
    if health.clock_skew_ms is not None:
        metrics.RECORDER_CLOCK_SKEW.set(health.clock_skew_ms)
    if health.disk_free_bytes is not None:
        metrics.RECORDER_DISK_FREE.set(health.disk_free_bytes)
    metrics.RECORDER_HEARTBEAT.set(now_ns / NS_PER_SECOND)
    return totals


class HeartbeatWriter:
    """Writes ``health/heartbeat.json`` atomically, and publishes the metrics.

    Holds the counter totals between calls so that :func:`publish` can turn
    absolute counts into the increments Prometheus counters take.
    """

    def __init__(self, root: str | Path, *, wall_ns: Any = time.time_ns) -> None:
        self.root = Path(root)
        self.path = heartbeat_path(self.root)
        self._wall_ns = wall_ns
        self._totals: dict[str, Any] = {}
        self.writes = 0

    def write(self, health: RecorderHealth, *, now_ns: int | None = None) -> dict[str, Any]:
        """One heartbeat: read the disk, publish the metrics, replace the file."""
        stamp = self._wall_ns() if now_ns is None else int(now_ns)
        health.heartbeat_ns = stamp
        health.disk_free_bytes = disk_free_bytes(self.root)
        document = health.to_document(stamp)
        self._totals = publish(health, now_ns=stamp, previous=self._totals)
        write_json_atomic(self.path, document)
        self.writes += 1
        return document


class LifecycleLog:
    """The recorder's own append-only record of going up and coming down (R1-h).

    Three methods and no policy: :meth:`up` on start, :meth:`down` on a stop
    this process performed, and :meth:`last_event` so a start can see how the
    previous run ended. Nothing here is read by the sink, the parsers, the
    normalizer or any control flow, and nothing it records can change a recorded
    value.

    **Appended, fsynced, never rewritten.** The file's whole purpose is to hold
    a line the process that wrote it did not live to follow up, so a writer that
    buffered would lose exactly the record worth having. Each line is flushed
    and ``fsync``-ed before the call returns.

    **A failure to write is logged and swallowed.** This is the one place in the
    recorder where that is right: the log is evidence about the recorder, not
    part of the recording, and a full disk must not stop the recorder from
    starting -- or, worse, raise out of a shutdown path and lose the clean stop
    it was trying to record. The heartbeat's ``disk_free_bytes`` is what
    surfaces the underlying fault.
    """

    def __init__(self, root: str | Path, *, wall_ns: Any = time.time_ns) -> None:
        self.root = Path(root)
        self.path = lifecycle_path(self.root)
        self._wall_ns = wall_ns

    def up(
        self, *, source_revision: str | None = None, **fields: Any
    ) -> dict[str, Any] | None:
        """Record this process starting, and any unclean stop before it.

        The unclean-stop line is written FIRST and as its own record, so the
        chronology reads in the order the facts were established: the previous
        run's end was unwitnessed, and then this run began. Folding it into the
        ``up`` record instead would make one line assert two things, one of them
        about a process this one never saw.
        """
        previous = self.last_event()
        if previous is not None and previous.get("event") == LIFECYCLE_UP:
            self._append(
                LIFECYCLE_UNCLEAN,
                previous_up_ns=previous.get("wall_ns"),
                detail=(
                    "the previous run wrote no recorder.down; it was killed, the host "
                    "stopped, or the write failed. This line records that the stop was "
                    "unwitnessed and asserts nothing about its cause"
                ),
            )
        return self._append(LIFECYCLE_UP, source_revision=source_revision, **fields)

    def down(self, reason: str, **fields: Any) -> dict[str, Any] | None:
        """Record a stop this process carried out, with the reason it had."""
        return self._append(LIFECYCLE_DOWN, reason=str(reason), **fields)

    def last_event(self) -> dict[str, Any] | None:
        """The last well-formed line, or ``None`` if there is none.

        A torn last line -- the tail of a write interrupted by the kill this
        file exists to witness -- is skipped rather than raising: the earlier
        lines are still evidence, and refusing the whole file because its last
        byte is missing would discard the history to protect the guess.
        """
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except ValueError:
                continue
            if isinstance(document, dict):
                return document
        return None

    def _needs_newline(self) -> bool:
        """Whether the file ends mid-line, which a killed write can leave it doing."""
        try:
            with open(self.path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    return False
                handle.seek(-1, os.SEEK_END)
                return handle.read(1) != b"\n"
        except OSError:
            return False

    def _append(self, event: str, **fields: Any) -> dict[str, Any] | None:
        stamp = int(self._wall_ns())
        document: dict[str, Any] = {
            "schema": LIFECYCLE_SCHEMA,
            "event": event,
            "wall_ns": stamp,
            "wall_utc": iso_utc(stamp),
        }
        document.update({key: value for key, value in fields.items() if value is not None})
        line = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                # Close an unterminated line before appending to it.
                #
                # The interruption this file exists to record is also what can
                # leave its last line half-written, and appending straight onto
                # that fragment concatenates the two into one unparseable line --
                # destroying the new record as well as the old one. The record
                # lost that way is the worst possible one: the `recorder.unclean_stop`
                # a start writes on finding an unmatched `up`, which is precisely
                # the evidence the torn line is a symptom of.
                #
                # The newline isolates the fragment. It stays on disk as its own
                # unreadable line -- a reader skips it and keeps the history --
                # and every record after it is well formed.
                if self._needs_newline():
                    handle.write("\n")
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            logger.warning("could not record %s in %s: %s", event, self.path, exc)
            return None
        return document
