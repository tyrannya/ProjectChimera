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
