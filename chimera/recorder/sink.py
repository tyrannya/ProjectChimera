"""The append-only raw sink: what the exchange said, in the order it said it.

One sink owns one stream. It writes one file per UTC day, appends and never
rewrites, and it is the only thing in the recorder allowed to touch those files.
Everything downstream — the minute normalizer, the archive reconciliation, the
coverage gate — is a function of what is here, so this layer's job is to lose
nothing and to invent nothing.

Layout, storage layout version 1, under ``<root>/raw/<stream>/<YYYY-MM-DD>/``::

    events.ndjson              the day, while it is open
    events.ndjson.gz           the day, once frozen
    events.late.ndjson         observations that arrived after the day closed
    events.ndjson.truncated    bytes recovered from a torn tail
    manifest.json              what the day holds, and its checksums

**Five outcomes, none of them silent.** An observation offered to
:meth:`RawSink.append` is exactly one of:

``ACCEPTED`` / ``DUPLICATE`` / ``LATE`` / ``LATE_DUPLICATE``
    returned as an :class:`AppendResult` and counted per day, because the
    reconciliation and the coverage report both need to know a duplicate or a
    late arrival happened, not merely that the file has one fewer line than the
    exchange sent.

*malformed*
    raised, and raised early: :class:`chimera.recorder.events.RawEvent` refuses
    a payload it cannot serialise deterministically at construction, before a
    file is open, and :meth:`RawSink.recover_tail` reports every record it had
    to remove from a torn file.

*write failure*
    raised as :class:`RecorderSinkError`. It is not a return value on purpose: a
    stream whose writes are failing must halt, and a code path that can ignore
    the result is a code path that will.

**What the durability guarantee actually is.** Records are appended in binary
mode, so a line is the same bytes on Windows and on Linux and no newline is ever
translated. Every record is flushed as it is written, so a reader of the open
day sees everything the sink has accepted; ``fsync`` — which is what makes a
record survive a power loss rather than merely a process death — is
:meth:`RawSink.sync`, called on demand. The service that drives that cadence is
a later package, and there is deliberately no background thread here. Between
syncs, a machine-level crash can leave a torn final line;
:meth:`RawSink.recover_tail` finds it, moves the bytes to a ``.truncated``
companion **before** shortening the file, and counts what it removed. Nothing is
discarded, and a torn tail can never quietly become a valid record.

**What duplicate detection actually is.** The sink holds the last
:data:`DEFAULT_DEDUP_WINDOW` deduplication keys per file, rebuilt from the tail
on reopen. Inside that window a re-delivered observation is recognised exactly;
outside it, it is not, and the archive reconciliation is the backstop. That
bound is stated rather than hidden because an unbounded set would be a memory
leak on a stream that runs for six months.

This module opens no socket, makes no request and reads no clock.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

from chimera.recorder.contract import STORAGE_LAYOUT_VERSION, RecorderContract
from chimera.recorder.events import (
    RawEvent,
    RecorderEventError,
    day_start_ns,
    iso_utc,
    minute_open_ms,
)

#: Names the shape of a day manifest. A manifest from a schema this build does
#: not know is refused rather than read with today's field meanings.
DAY_MANIFEST_SCHEMA = "chimera.recorder-day-manifest/1"

RAW_DIRECTORY = "raw"
EVENTS_FILE = "events.ndjson"
LATE_FILE = "events.late.ndjson"
MANIFEST_FILE = "manifest.json"
GZIP_SUFFIX = ".gz"
TRUNCATED_SUFFIX = ".truncated"

#: Deduplication keys held per file. See the module docstring: the bound is the
#: honest limit of what this layer detects, not an implementation detail.
DEFAULT_DEDUP_WINDOW = 4096

#: How much of a file's end :meth:`RawSink.recover_tail` reads. One mebibyte is
#: several thousand records of every stream in the gen3 contract, so the tail a
#: crash can tear is comfortably inside it.
TAIL_BYTES = 1 << 20

#: How many unreadable records at the end of a file are treated as a torn tail.
#: Beyond this the file is not a tail that was cut short, it is a file that has
#: been damaged, and the difference matters enough to refuse rather than trim.
MAX_TRUNCATED_RECORDS = 64

_COPY_CHUNK = 1 << 20


class RecorderSinkError(RuntimeError):
    """The raw sink cannot write, or cannot honestly read what it wrote."""


class AppendOutcome(str, Enum):
    """What became of one observation. A bounded, reported label."""

    #: Written to the open day's file.
    ACCEPTED = "ACCEPTED"
    #: Recognised as an observation already on disk, and not written twice.
    DUPLICATE = "DUPLICATE"
    #: Its canonical time falls in a day that has already closed, so it was
    #: appended to that day's late file and the finalised day was not touched.
    LATE = "LATE"
    #: Late, and already present in that day's late file.
    LATE_DUPLICATE = "LATE_DUPLICATE"

    @property
    def accepted(self) -> bool:
        """Whether the observation was written somewhere."""
        return self in (AppendOutcome.ACCEPTED, AppendOutcome.LATE)


@dataclass(frozen=True)
class AppendResult:
    """What :meth:`RawSink.append` did, and where."""

    outcome: AppendOutcome
    day: str
    #: Repository-style forward-slash path relative to the sink's root. Never an
    #: absolute path: a manifest or a log line that named one would describe the
    #: machine that wrote it rather than the data.
    path: str
    bytes_written: int

    @property
    def accepted(self) -> bool:
        return self.outcome.accepted


@dataclass(frozen=True)
class TailRecovery:
    """What reopening a file found at its end."""

    path: str
    records_scanned: int
    dedup_keys_loaded: int
    truncated_records: int
    truncated_bytes: int
    truncated_path: str | None

    @property
    def clean(self) -> bool:
        return self.truncated_bytes == 0


@dataclass(frozen=True)
class DayCounters:
    """What one day of one stream saw. Written into the day manifest."""

    accepted: int = 0
    duplicates: int = 0
    late: int = 0
    late_duplicates: int = 0
    bytes_written: int = 0
    truncated_records: int = 0
    truncated_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "late": self.late,
            "late_duplicates": self.late_duplicates,
            "bytes_written": self.bytes_written,
            "truncated_records": self.truncated_records,
            "truncated_bytes": self.truncated_bytes,
        }


class _Counters:
    """The mutable per-day tally. Snapshotted into :class:`DayCounters`."""

    __slots__ = (
        "accepted",
        "duplicates",
        "late",
        "late_duplicates",
        "bytes_written",
        "truncated_records",
        "truncated_bytes",
    )

    def __init__(self) -> None:
        self.accepted = 0
        self.duplicates = 0
        self.late = 0
        self.late_duplicates = 0
        self.bytes_written = 0
        self.truncated_records = 0
        self.truncated_bytes = 0

    def snapshot(self) -> DayCounters:
        return DayCounters(
            accepted=self.accepted,
            duplicates=self.duplicates,
            late=self.late,
            late_duplicates=self.late_duplicates,
            bytes_written=self.bytes_written,
            truncated_records=self.truncated_records,
            truncated_bytes=self.truncated_bytes,
        )


_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def require_stream_id(stream: Any) -> str:
    """A stream id that is safe to use as a directory name on every platform."""
    if not isinstance(stream, str) or not stream:
        raise RecorderSinkError(f"stream must be a non-empty string, got {stream!r}")
    market, _, rest = stream.partition(".")
    if not market or not rest or "." in rest:
        raise RecorderSinkError(f"stream {stream!r} is not a <market>.<stream> id")
    if set(market) - _NAME_CHARS or set(rest) - _NAME_CHARS:
        raise RecorderSinkError(
            f"stream {stream!r} carries a character outside [A-Za-z0-9_]; a stream id is "
            "also a directory name"
        )
    return stream


def require_day(day: Any) -> str:
    """A ``YYYY-MM-DD`` UTC day, checked by parsing it rather than by its shape."""
    if not isinstance(day, str):
        raise RecorderSinkError(f"a day must be a YYYY-MM-DD string, got {day!r}")
    try:
        day_start_ns(day)
    except RecorderEventError as exc:
        raise RecorderSinkError(str(exc)) from exc
    return day


def _relative(root: Path, path: Path) -> str:
    """``path`` under ``root``, with forward slashes and no drive letter.

    Every path this package writes into a manifest goes through here. A manifest
    that recorded ``F:\\Projects\\...`` would not compare equal to the same
    manifest produced on the Linux host that reviews it, and the difference would
    be a fact about two machines rather than about the data.
    """
    return path.relative_to(root).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_COPY_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_atomic(path: Path, body: bytes) -> None:
    """Write bytes so that a crash leaves the previous file rather than half of one.

    Temp file, ``fsync``, ``os.replace`` — the discipline
    :meth:`chimera.futures.store.FuturesStore.save` uses, and for the same
    reason. Binary throughout, so no newline is translated and the file is the
    same bytes on Windows and on Linux.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise RecorderSinkError(f"could not write {path}: {exc}") from exc


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON document atomically, sorted and UTF-8, with a trailing newline."""
    body = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    write_bytes_atomic(path, body.encode("utf-8"))


def _gzip_deterministically(source: Path, destination: Path) -> None:
    """Compress ``source`` to ``destination`` with a byte-stable gzip header.

    ``mtime=0`` and an empty ``filename``: the default header embeds the current
    time and the source file's name, which would make the same events compress
    to different bytes on every run and on every machine. The deflate stream
    itself is a function of the zlib build, so the gz digest identifies the
    stored bytes; the digest of the uncompressed NDJSON, recorded beside it, is
    what identifies the events.
    """
    try:
        with open(source, "rb") as reader, open(destination, "wb") as writer:
            with gzip.GzipFile(
                fileobj=writer, mode="wb", compresslevel=9, filename="", mtime=0
            ) as compressor:
                while True:
                    chunk = reader.read(_COPY_CHUNK)
                    if not chunk:
                        break
                    compressor.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    except OSError as exc:
        raise RecorderSinkError(
            f"could not compress {source} to {destination}: {exc}"
        ) from exc


def iter_raw_lines(path: Path) -> Iterable[bytes]:
    """Every complete line of a raw file, plain or gzipped."""
    opener = gzip.open if path.suffix == GZIP_SUFFIX else open
    with opener(path, "rb") as handle:  # type: ignore[operator]
        for line in handle:
            if line.strip():
                yield line


def read_raw_events(
    root: str | Path, stream: str, day: str, *, include_late: bool = True
) -> list[RawEvent]:
    """Every recorded observation of one stream on one UTC day, in file order.

    The main file first and the late file after it, which is the order the
    normalizer's tie-break depends on: for two observations carrying the same
    canonical time, the one that was on disk when the day closed comes first.
    """
    sink_root = Path(root)
    directory = sink_root / RAW_DIRECTORY / require_stream_id(stream) / day
    plain = directory / EVENTS_FILE
    packed = directory / (EVENTS_FILE + GZIP_SUFFIX)
    if plain.exists() and packed.exists():
        raise RecorderSinkError(
            f"{directory} holds both {EVENTS_FILE} and {EVENTS_FILE}{GZIP_SUFFIX}. Freezing "
            "removes the plain file only after the compressed one is verified, so having "
            "both means an interrupted freeze that must be looked at, not read past"
        )
    events: list[RawEvent] = []
    for source in (packed if packed.exists() else plain, directory / LATE_FILE):
        if not source.exists():
            continue
        if source == directory / LATE_FILE and not include_late:
            continue
        for line in iter_raw_lines(source):
            events.append(RawEvent.from_line(line))
    return events


def available_days(root: str | Path, stream: str) -> list[str]:
    """The UTC days this stream has a directory for, sorted."""
    directory = Path(root) / RAW_DIRECTORY / require_stream_id(stream)
    if not directory.is_dir():
        return []
    return sorted(entry.name for entry in directory.iterdir() if entry.is_dir())


class RawSink:
    """Append-only storage for one stream, one UTC day at a time."""

    def __init__(
        self,
        root: str | Path,
        stream: str,
        *,
        contract: RecorderContract | None = None,
        dedup_window: int = DEFAULT_DEDUP_WINDOW,
    ) -> None:
        self.root = Path(root)
        self.stream = require_stream_id(stream)
        self.contract = contract
        if contract is not None and stream not in contract.streams:
            raise RecorderSinkError(
                f"recorder contract {contract.label} declares streams "
                f"{list(contract.streams)}, not {stream!r}. A file written for a stream the "
                "contract does not name is data nothing will ever read"
            )
        if dedup_window < 1:
            raise RecorderSinkError(f"dedup_window must be >= 1, got {dedup_window}")
        self.dedup_window = int(dedup_window)
        self._day: str | None = None
        self._handle: BinaryIO | None = None
        self._seen: dict[tuple[str, str], None] = {}
        self._counters: dict[str, _Counters] = {}
        self._recovered: set[Path] = set()
        self._rotations = 0

    # --- paths ------------------------------------------------------------
    def day_dir(self, day: str) -> Path:
        return self.root / RAW_DIRECTORY / self.stream / day

    def events_path(self, day: str) -> Path:
        return self.day_dir(day) / EVENTS_FILE

    def gz_path(self, day: str) -> Path:
        return self.day_dir(day) / (EVENTS_FILE + GZIP_SUFFIX)

    def late_path(self, day: str) -> Path:
        return self.day_dir(day) / LATE_FILE

    def manifest_path(self, day: str) -> Path:
        return self.day_dir(day) / MANIFEST_FILE

    # --- state ------------------------------------------------------------
    @property
    def open_day(self) -> str | None:
        """The UTC day currently accepting appends, if any."""
        return self._day

    @property
    def rotations(self) -> int:
        """How many times this sink has crossed a UTC day boundary."""
        return self._rotations

    def is_frozen(self, day: str) -> bool:
        """Whether ``day`` has a manifest, and is therefore finalised."""
        return self.manifest_path(day).exists()

    def counters(self, day: str) -> DayCounters:
        """What this sink has seen for one day."""
        return self._counters.get(day, _Counters()).snapshot()

    def days_seen(self) -> list[str]:
        """Days this sink has counted anything for, sorted."""
        return sorted(self._counters)

    # --- writing ----------------------------------------------------------
    def append(self, event: RawEvent) -> AppendResult:
        """Record one observation, or say why it was not recorded.

        Never raises for a duplicate or a late arrival: those are conditions the
        coverage report has to be able to see, so they are outcomes. It does
        raise for an event belonging to another stream — that is a caller
        defect, not a property of the data — and for a failed write.
        """
        if not isinstance(event, RawEvent):
            raise RecorderSinkError(f"append expects a RawEvent, got {type(event).__name__}")
        if event.stream != self.stream:
            raise RecorderSinkError(
                f"this sink owns {self.stream!r} and was offered a {event.stream!r} event. "
                "One writer per stream is what makes the file append-only"
            )
        day = event.day
        late = self.is_frozen(day) or (self._day is not None and day < self._day)
        if late:
            return self._write(self.late_path(day), day, event, late=True)
        if self._day is None:
            self._open_day(day)
        elif day > self._day:
            self._rotate_to(day)
        return self._write(self.events_path(day), day, event, late=False)

    def rotate(self, day: str) -> None:
        """Close the open day and open ``day``. Idempotent for the open day."""
        require_day(day)
        if self._day == day:
            return
        if self.is_frozen(day):
            raise RecorderSinkError(
                f"{self.stream} on {day} is frozen; rotating into it would append to a "
                "finalised file. A late observation for a closed day belongs in its late "
                "file, which append() routes to on its own"
            )
        if self._day is not None:
            self._rotate_to(day)
            return
        self._open_day(day)

    def sync(self) -> None:
        """Flush and ``fsync`` the open day's file.

        The cadence — the plan's "at most once per second, and on rotation" — is
        the caller's to drive. Putting a timer in here would make the offline
        core own a background task, which is exactly what the later service
        package is for.
        """
        if self._handle is None:
            return
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except OSError as exc:
            raise RecorderSinkError(
                f"could not fsync the {self.stream} raw file for {self._day}: {exc}"
            ) from exc

    def close(self) -> None:
        """Sync and close the open day. The sink can be reopened by appending."""
        self._close_handle()
        self._day = None

    def __enter__(self) -> "RawSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- recovery ---------------------------------------------------------
    def recover_tail(self, day: str | None = None, *, late: bool = False) -> TailRecovery:
        """Validate the end of a day's file and repair a torn tail.

        Reads the last :data:`TAIL_BYTES`, removes any trailing bytes that are
        not a complete record — preserving them in a ``.truncated`` companion
        first, so a crash between the two loses nothing — and loads the
        surviving deduplication keys so a restart recognises a re-delivered
        observation.

        A line that has no terminating newline is treated as torn even when the
        JSON it holds happens to parse: without the newline there is no evidence
        the writer finished, and a record that might have been about to grow is
        not evidence.
        """
        target = day if day is not None else self._day
        if target is None:
            raise RecorderSinkError("recover_tail needs a day; this sink has none open")
        require_day(target)
        path = self.late_path(target) if late else self.events_path(target)
        return self._recover_file(path, target, force=True)

    # --- freezing ---------------------------------------------------------
    def freeze_day(self, day: str, *, provenance: Mapping[str, Any] | None = None) -> Path:
        """Close a day for ever: verify, compress, checksum, and write the manifest.

        Every record is parsed on the way through, so a manifest is never
        written over a file the sink cannot read back. The plain NDJSON is
        removed only after the compressed copy has been decompressed and found
        to hash to the same bytes.

        The late file is deliberately **not** compressed and the manifest says
        so: a late event can still arrive for a closed day, and a file that
        claimed to be frozen while still growing would be a false claim. What
        the manifest records for it is its state at this instant.
        """
        require_day(day)
        manifest_path = self.manifest_path(day)
        if manifest_path.exists():
            raise RecorderSinkError(
                f"{manifest_path} already exists, so {self.stream} on {day} is frozen. A "
                "frozen day is never rewritten; a correction is a new file with a note, and "
                "the reconciliation report says which version was used"
            )
        if self._day == day:
            self._close_handle()
            self._day = None

        plain = self.events_path(day)
        packed = self.gz_path(day)
        if plain.exists() and packed.exists():
            raise RecorderSinkError(
                f"{self.day_dir(day)} holds both the plain and the compressed events file. "
                "That is an interrupted freeze; look at it rather than freeze over it"
            )
        if not plain.exists() and not packed.exists():
            raise RecorderSinkError(
                f"no raw events for {self.stream} on {day}. Freezing a day that was never "
                "written would produce a manifest asserting an empty day was recorded"
            )

        if plain.exists():
            self._recover_file(plain, day)
            scan = self._scan(plain, day)
            ndjson_sha = _sha256_file(plain)
            _gzip_deterministically(plain, packed)
            if _sha256_of_gz(packed) != ndjson_sha:
                raise RecorderSinkError(
                    f"{packed} does not decompress to the bytes of {plain}; the plain file "
                    "is left in place"
                )
            gz_sha = _sha256_file(packed)
            ndjson_bytes = plain.stat().st_size
            plain.unlink()
        else:
            scan = self._scan(packed, day)
            ndjson_sha = _sha256_of_gz(packed)
            gz_sha = _sha256_file(packed)
            ndjson_bytes = scan.byte_length

        late_block: dict[str, Any] | None = None
        late = self.late_path(day)
        if late.exists():
            self._recover_file(late, day)
            late_scan = self._scan(late, day)
            late_block = {
                "path": _relative(self.root, late),
                "rows": late_scan.rows,
                "sha256": _sha256_file(late),
                "open": True,
                "note": (
                    "recorded as of this manifest and not compressed: a late event may "
                    "still arrive for a closed day, and the reconciliation reads this file "
                    "as it then stands"
                ),
            }

        truncated = plain.with_name(plain.name + TRUNCATED_SUFFIX)
        truncated_block = (
            {"path": _relative(self.root, truncated), "bytes": truncated.stat().st_size}
            if truncated.exists()
            else None
        )

        manifest = {
            "manifest_schema": DAY_MANIFEST_SCHEMA,
            "stream": self.stream,
            "day": day,
            "storage_layout_version": STORAGE_LAYOUT_VERSION,
            "contract": None if self.contract is None else self.contract.provenance(),
            "raw": {
                "path": _relative(self.root, packed),
                "rows": scan.rows,
                "ndjson_bytes": ndjson_bytes,
                "sha256_ndjson": ndjson_sha,
                "sha256_gz": gz_sha,
                "note": (
                    "sha256_ndjson identifies the events and reproduces from any host; "
                    "sha256_gz identifies the stored bytes, which depend on the zlib build "
                    "that wrote them"
                ),
            },
            "late": late_block,
            "truncated": truncated_block,
            "first_canonical_ns": scan.first_ns,
            "last_canonical_ns": scan.last_ns,
            "first_canonical_utc": None if scan.first_ns is None else iso_utc(scan.first_ns),
            "last_canonical_utc": None if scan.last_ns is None else iso_utc(scan.last_ns),
            "first_minute_open_ms": scan.first_minute,
            "last_minute_open_ms": scan.last_minute,
            "counters": self.counters(day).to_dict(),
            "provenance": None if provenance is None else dict(provenance),
        }
        write_json_atomic(manifest_path, manifest)
        return manifest_path

    # --- internals --------------------------------------------------------
    def _counters_for(self, day: str) -> _Counters:
        return self._counters.setdefault(day, _Counters())

    def _remember(self, key: tuple[str, str]) -> None:
        self._seen.pop(key, None)
        self._seen[key] = None
        while len(self._seen) > self.dedup_window:
            self._seen.pop(next(iter(self._seen)))

    def _open_day(self, day: str) -> None:
        path = self.events_path(day)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RecorderSinkError(f"could not create {path.parent}: {exc}") from exc
        # Tail recovery runs before the handle exists, because it may shorten
        # the file, and appending through a handle opened over a torn tail would
        # write the next record after bytes that are not a record.
        self._recover_file(path, day)
        try:
            self._handle = open(path, "ab")
        except OSError as exc:
            raise RecorderSinkError(f"could not open {path} for append: {exc}") from exc
        self._day = day

    def _rotate_to(self, day: str) -> None:
        self._close_handle()
        self._rotations += 1
        self._open_day(day)

    def _close_handle(self) -> None:
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            handle.flush()
            os.fsync(handle.fileno())
        except OSError as exc:
            handle.close()
            raise RecorderSinkError(f"could not fsync the raw file on close: {exc}") from exc
        handle.close()

    def _write(self, path: Path, day: str, event: RawEvent, *, late: bool) -> AppendResult:
        counters = self._counters_for(day)
        if late:
            self._recover_file(path, day)
        key = (day, event.dedup_key)
        if key in self._seen:
            if late:
                counters.late_duplicates += 1
                outcome = AppendOutcome.LATE_DUPLICATE
            else:
                counters.duplicates += 1
                outcome = AppendOutcome.DUPLICATE
            return AppendResult(outcome, day, _relative(self.root, path), 0)

        line = event.canonical_line()
        if late:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "ab") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise RecorderSinkError(
                    f"could not append a late event to {path}: {exc}"
                ) from exc
            counters.late += 1
            outcome = AppendOutcome.LATE
        else:
            if self._handle is None:  # pragma: no cover - _open_day always sets it
                raise RecorderSinkError(f"no open file for {self.stream} on {day}")
            try:
                self._handle.write(line)
                # Flushed on every record, so a reader of the open day — the
                # normalizer re-deriving today from raw — always sees every
                # record the sink has accepted. Durability is a separate
                # question and is what `sync` answers.
                self._handle.flush()
            except OSError as exc:
                raise RecorderSinkError(f"could not append to {path}: {exc}") from exc
            counters.accepted += 1
            outcome = AppendOutcome.ACCEPTED
        counters.bytes_written += len(line)
        self._remember(key)
        return AppendResult(outcome, day, _relative(self.root, path), len(line))

    def _recover_file(self, path: Path, day: str, *, force: bool = False) -> TailRecovery:
        if not force and path in self._recovered:
            return TailRecovery(_relative(self.root, path), 0, 0, 0, 0, None)
        self._recovered.add(path)
        if not path.exists() or path.stat().st_size == 0:
            return TailRecovery(_relative(self.root, path), 0, 0, 0, 0, None)

        size = path.stat().st_size
        start = max(0, size - TAIL_BYTES)
        try:
            with open(path, "rb") as handle:
                handle.seek(start)
                chunk = handle.read()
        except OSError as exc:
            raise RecorderSinkError(f"could not read the tail of {path}: {exc}") from exc

        offset = start
        if start > 0:
            newline = chunk.find(b"\n")
            if newline == -1:
                raise RecorderSinkError(
                    f"the last {TAIL_BYTES} bytes of {path} hold no line break, so the end "
                    "of the file cannot be bounded. This is damage, not a torn tail"
                )
            offset = start + newline + 1
            chunk = chunk[newline + 1 :]

        cut = chunk.rfind(b"\n")
        complete = b"" if cut == -1 else chunk[: cut + 1]
        fragment_bytes = len(chunk) - len(complete)

        lines = complete.split(b"\n")[:-1]
        valid_end = offset + len(complete)
        removed_records = 1 if fragment_bytes else 0
        while lines:
            try:
                RawEvent.from_line(lines[-1])
            except RecorderEventError:
                if removed_records >= MAX_TRUNCATED_RECORDS:
                    raise RecorderSinkError(
                        f"{path} ends with more than {MAX_TRUNCATED_RECORDS} records this "
                        "build cannot read. That is a damaged file, not a tail a crash cut "
                        "short, and it is left exactly as it is"
                    ) from None
                valid_end -= len(lines[-1]) + 1
                lines.pop()
                removed_records += 1
                continue
            break

        truncated_bytes = size - valid_end
        truncated_path: str | None = None
        if truncated_bytes > 0:
            companion = path.with_name(path.name + TRUNCATED_SUFFIX)
            try:
                with open(path, "rb") as reader:
                    reader.seek(valid_end)
                    salvage = reader.read()
                with open(companion, "ab") as writer:
                    writer.write(salvage)
                    writer.flush()
                    os.fsync(writer.fileno())
                os.truncate(path, valid_end)
            except OSError as exc:
                raise RecorderSinkError(
                    f"could not preserve the torn tail of {path} into {companion}: {exc}"
                ) from exc
            truncated_path = _relative(self.root, companion)
            counters = self._counters_for(day)
            counters.truncated_records += removed_records
            counters.truncated_bytes += truncated_bytes

        loaded = 0
        for line in lines[-self.dedup_window :]:
            self._remember((day, RawEvent.from_line(line).dedup_key))
            loaded += 1
        return TailRecovery(
            path=_relative(self.root, path),
            records_scanned=len(lines),
            dedup_keys_loaded=loaded,
            truncated_records=removed_records if truncated_bytes else 0,
            truncated_bytes=truncated_bytes,
            truncated_path=truncated_path,
        )

    def _scan(self, path: Path, day: str) -> "_Scan":
        rows = 0
        byte_length = 0
        first_ns: int | None = None
        last_ns: int | None = None
        first_minute: int | None = None
        last_minute: int | None = None
        for line in iter_raw_lines(path):
            byte_length += len(line)
            event = RawEvent.from_line(line)
            if event.stream != self.stream:
                raise RecorderSinkError(
                    f"{path} holds a {event.stream!r} record in the {self.stream!r} file"
                )
            if event.day != day:
                raise RecorderSinkError(
                    f"{path} holds a record whose canonical day is {event.day}, not {day}"
                )
            rows += 1
            if first_ns is None or event.canonical_ns < first_ns:
                first_ns = event.canonical_ns
            if last_ns is None or event.canonical_ns > last_ns:
                last_ns = event.canonical_ns
        if first_ns is not None:
            first_minute = minute_open_ms(first_ns)
        if last_ns is not None:
            last_minute = minute_open_ms(last_ns)
        return _Scan(rows, byte_length, first_ns, last_ns, first_minute, last_minute)


@dataclass(frozen=True)
class _Scan:
    rows: int
    byte_length: int
    first_ns: int | None
    last_ns: int | None
    first_minute: int | None
    last_minute: int | None


def _sha256_of_gz(path: Path) -> str:
    """SHA-256 of what a gzip file decompresses to, not of the file itself."""
    digest = hashlib.sha256()
    try:
        with gzip.open(path, "rb") as handle:
            while True:
                chunk = handle.read(_COPY_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise RecorderSinkError(f"could not read {path}: {exc}") from exc
    return digest.hexdigest()
