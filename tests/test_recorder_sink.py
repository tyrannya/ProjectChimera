"""The append-only raw sink: what it writes, what it refuses, and what a crash leaves.

Four properties carry the weight, and each of them is asserted against bytes
rather than against a return value:

* **append-only.** Once a record is on disk its bytes never change, a finalised
  day is never touched again, and a late observation for a closed day goes to
  that day's late file with the frozen one left byte-for-byte alone.
* **a torn tail can never become evidence.** A crash between writes leaves an
  incomplete final line; reopening moves those bytes to a ``.truncated``
  companion *before* shortening the file, counts them, and leaves a file whose
  every line parses.
* **five outcomes, none of them silent.** Accepted, duplicate, late, late
  duplicate, and — as exceptions, because they must halt rather than be
  ignored — malformed and failed writes.
* **restart is an ordinary event.** A new sink over the same directory rebuilds
  its deduplication horizon from the file, so a re-delivered observation after a
  restart is still recognised as one.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.events import (
    SPOT_KLINE_1M,
    UM_BOOK_TICKER,
    UM_KLINE_1M,
    EventSource,
    RawEvent,
)
from chimera.recorder.sink import (
    DAY_MANIFEST_SCHEMA,
    EVENTS_FILE,
    GZIP_SUFFIX,
    LATE_FILE,
    MAX_TRUNCATED_RECORDS,
    TRUNCATED_SUFFIX,
    AppendOutcome,
    RawSink,
    RecorderSinkError,
    available_days,
    read_raw_events,
    require_day,
    require_stream_id,
    write_json_atomic,
)

from tests.recorder_synthetic import (
    DAY,
    NEXT_DAY,
    book_event,
    kline_event,
    minute_ms,
)

CONTRACT = load_recorder_contract()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A storage root shaped the way the contract says, under a temp directory."""
    return CONTRACT.storage_root(tmp_path / "data")


@pytest.fixture
def sink(root: Path) -> RawSink:
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        yield writer


def lines(path: Path) -> list[bytes]:
    return [line for line in path.read_bytes().split(b"\n") if line]


# --- A. construction -----------------------------------------------------------
def test_a_sink_only_accepts_a_stream_the_contract_declares(root):
    with pytest.raises(RecorderSinkError, match="declares streams"):
        RawSink(root, "um.aggTrade", contract=CONTRACT)


def test_a_stream_id_must_be_safe_as_a_directory_name():
    assert require_stream_id(UM_KLINE_1M) == UM_KLINE_1M
    for bad in ("umkline", "um.kline/1m", "um..kline", "", None, "um.kline.1m"):
        with pytest.raises(RecorderSinkError):
            require_stream_id(bad)


def test_a_day_must_parse_as_a_utc_day():
    assert require_day(DAY) == DAY
    for bad in ("2026-13-01", "19-09-2026", 20260919, None):
        with pytest.raises(RecorderSinkError):
            require_day(bad)


def test_a_sink_refuses_an_event_belonging_to_another_stream(sink):
    with pytest.raises(RecorderSinkError, match="One writer per stream"):
        sink.append(kline_event(minute_ms(0), stream=SPOT_KLINE_1M))
    with pytest.raises(RecorderSinkError, match="expects a RawEvent"):
        sink.append({"stream": UM_KLINE_1M})  # type: ignore[arg-type]


# --- B. append-only ------------------------------------------------------------
def test_a_record_is_written_as_exactly_the_bytes_the_event_defines(sink, root):
    event = kline_event(minute_ms(0))
    result = sink.append(event)
    sink.sync()

    assert result.outcome is AppendOutcome.ACCEPTED
    assert result.accepted
    assert result.day == DAY
    assert result.path == f"raw/{UM_KLINE_1M}/{DAY}/{EVENTS_FILE}"
    assert "\\" not in result.path
    assert sink.events_path(DAY).read_bytes() == event.canonical_line()
    assert result.bytes_written == len(event.canonical_line())


def test_appending_never_rewrites_what_is_already_on_disk(sink):
    first = kline_event(minute_ms(0))
    sink.append(first)
    sink.sync()
    prefix = sink.events_path(DAY).read_bytes()

    for index in range(1, 5):
        sink.append(kline_event(minute_ms(index)))
    sink.sync()

    body = sink.events_path(DAY).read_bytes()
    assert body.startswith(prefix), "an earlier record's bytes changed"
    assert len(lines(sink.events_path(DAY))) == 5
    assert body.endswith(b"\n")
    assert b"\r\n" not in body, "a translated newline would differ between platforms"


def test_the_same_observation_twice_is_a_duplicate_and_is_not_written(sink):
    event = kline_event(minute_ms(0))
    assert sink.append(event).outcome is AppendOutcome.ACCEPTED
    second = sink.append(event)
    sink.sync()

    assert second.outcome is AppendOutcome.DUPLICATE
    assert second.accepted is False
    assert second.bytes_written == 0
    assert len(lines(sink.events_path(DAY))) == 1
    assert sink.counters(DAY).duplicates == 1
    assert sink.counters(DAY).accepted == 1


def test_a_partial_frame_and_the_closed_candle_of_a_minute_are_both_kept(sink):
    minute = minute_ms(0)
    assert sink.append(kline_event(minute, closed=False)).outcome is AppendOutcome.ACCEPTED
    assert sink.append(kline_event(minute, closed=True)).outcome is AppendOutcome.ACCEPTED
    sink.sync()
    assert len(lines(sink.events_path(DAY))) == 2


def test_a_rest_gapfill_of_a_minute_the_websocket_already_closed_is_kept(sink):
    """Both records stay so the reconciliation can compare them."""
    minute = minute_ms(0)
    sink.append(kline_event(minute))
    result = sink.append(kline_event(minute, source=EventSource.REST_GAPFILL))
    sink.sync()
    assert result.outcome is AppendOutcome.ACCEPTED
    stored = read_raw_events(sink.root, UM_KLINE_1M, DAY)
    assert {event.source for event in stored} == {
        EventSource.WEBSOCKET,
        EventSource.REST_GAPFILL,
    }


# --- C. day rotation -----------------------------------------------------------
def test_crossing_a_utc_midnight_opens_a_new_file_and_closes_the_old_one(sink):
    sink.append(kline_event(minute_ms(1439)))
    sink.sync()
    before = sink.events_path(DAY).read_bytes()

    result = sink.append(kline_event(minute_ms(0, day=NEXT_DAY)))
    sink.sync()

    assert result.day == NEXT_DAY
    assert sink.open_day == NEXT_DAY
    assert sink.rotations == 1
    assert sink.events_path(DAY).read_bytes() == before, "the closed day was written to"
    assert len(lines(sink.events_path(NEXT_DAY))) == 1
    assert sorted(available_days(sink.root, UM_KLINE_1M)) == [DAY, NEXT_DAY]


def test_the_day_a_record_lands_in_is_its_canonical_time_not_its_arrival(sink):
    """Rotation is a property of the data, so the same events always split the same way."""
    sink.append(kline_event(minute_ms(0, day=NEXT_DAY), receipt_wall_ns=1))
    sink.append(kline_event(minute_ms(1, day=NEXT_DAY), receipt_wall_ns=2))
    sink.sync()
    assert sink.events_path(DAY).exists() is False
    assert len(lines(sink.events_path(NEXT_DAY))) == 2


def test_an_explicit_rotation_is_idempotent_for_the_open_day(sink):
    sink.append(kline_event(minute_ms(0)))
    sink.rotate(DAY)
    assert sink.rotations == 0
    sink.rotate(NEXT_DAY)
    assert sink.rotations == 1
    assert sink.open_day == NEXT_DAY


# --- D. late events ------------------------------------------------------------
def test_a_late_observation_goes_to_the_late_file_and_never_into_the_closed_day(sink):
    sink.append(kline_event(minute_ms(1439)))
    sink.append(kline_event(minute_ms(0, day=NEXT_DAY)))
    sink.sync()
    closed_day = sink.events_path(DAY).read_bytes()

    late = kline_event(minute_ms(100))
    result = sink.append(late)

    assert result.outcome is AppendOutcome.LATE
    assert result.accepted is True
    assert result.path == f"raw/{UM_KLINE_1M}/{DAY}/{LATE_FILE}"
    assert sink.events_path(DAY).read_bytes() == closed_day
    assert sink.late_path(DAY).read_bytes() == late.canonical_line()
    assert sink.counters(DAY).late == 1


def test_a_repeated_late_observation_is_reported_as_a_late_duplicate(sink):
    sink.append(kline_event(minute_ms(1439)))
    sink.append(kline_event(minute_ms(0, day=NEXT_DAY)))
    late = kline_event(minute_ms(100))
    sink.append(late)
    result = sink.append(late)

    assert result.outcome is AppendOutcome.LATE_DUPLICATE
    assert result.accepted is False
    assert len(lines(sink.late_path(DAY))) == 1
    assert sink.counters(DAY).late_duplicates == 1


def test_a_frozen_day_takes_late_events_without_being_reopened(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.freeze_day(DAY)
        frozen = writer.gz_path(DAY).read_bytes()
        manifest = writer.manifest_path(DAY).read_bytes()

        result = writer.append(kline_event(minute_ms(5)))

        assert result.outcome is AppendOutcome.LATE
        assert writer.gz_path(DAY).read_bytes() == frozen
        assert writer.manifest_path(DAY).read_bytes() == manifest
        assert writer.events_path(DAY).exists() is False


def test_rotating_into_a_frozen_day_is_refused(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.freeze_day(DAY)
        with pytest.raises(RecorderSinkError, match="is frozen"):
            writer.rotate(DAY)


# --- E. crash, torn tails and restart ------------------------------------------
def test_a_torn_final_line_is_preserved_and_removed_before_it_can_be_read(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.append(kline_event(minute_ms(1)))
        writer.sync()
    path = root / "raw" / UM_KLINE_1M / DAY / EVENTS_FILE
    good = path.read_bytes()

    torn = kline_event(minute_ms(2)).canonical_line()[:40]
    with open(path, "ab") as handle:
        handle.write(torn)

    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        recovery = writer.recover_tail(DAY)

    assert recovery.clean is False
    assert recovery.truncated_bytes == len(torn)
    assert recovery.truncated_records == 1
    assert (
        recovery.truncated_path == f"raw/{UM_KLINE_1M}/{DAY}/{EVENTS_FILE}{TRUNCATED_SUFFIX}"
    )
    assert path.read_bytes() == good, "the surviving records were altered"
    assert (path.with_name(path.name + TRUNCATED_SUFFIX)).read_bytes() == torn
    assert all(RawEvent.from_line(line) for line in lines(path))


def test_a_complete_json_line_with_no_newline_is_still_treated_as_torn(root):
    """Without the newline there is no evidence the writer finished the record."""
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.sync()
    path = root / "raw" / UM_KLINE_1M / DAY / EVENTS_FILE
    unterminated = kline_event(minute_ms(1)).canonical_line().rstrip(b"\n")
    with open(path, "ab") as handle:
        handle.write(unterminated)

    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        recovery = writer.recover_tail(DAY)

    assert recovery.truncated_bytes == len(unterminated)
    assert len(lines(path)) == 1


def test_a_complete_but_unreadable_final_record_is_removed_too(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.sync()
    path = root / "raw" / UM_KLINE_1M / DAY / EVENTS_FILE
    with open(path, "ab") as handle:
        handle.write(b'{"schema":"chimera.recorder-raw-event/1"}\n')

    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        recovery = writer.recover_tail(DAY)

    assert recovery.truncated_records == 1
    assert len(lines(path)) == 1


def test_a_file_that_is_damaged_rather_than_torn_is_left_alone(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.sync()
    path = root / "raw" / UM_KLINE_1M / DAY / EVENTS_FILE
    with open(path, "ab") as handle:
        handle.write(b"garbage\n" * (MAX_TRUNCATED_RECORDS + 2))
    before = path.read_bytes()

    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        with pytest.raises(RecorderSinkError, match="damaged file"):
            writer.recover_tail(DAY)

    assert path.read_bytes() == before


def test_appending_after_a_crash_writes_after_the_last_complete_record(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.sync()
    path = root / "raw" / UM_KLINE_1M / DAY / EVENTS_FILE
    with open(path, "ab") as handle:
        handle.write(kline_event(minute_ms(1)).canonical_line()[:30])

    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(2)))
        writer.sync()

    stored = read_raw_events(root, UM_KLINE_1M, DAY)
    assert [event.minute_open_ms for event in stored] == [minute_ms(0), minute_ms(2)]
    assert writer.counters(DAY).truncated_bytes == 30


def test_a_restart_rebuilds_the_deduplication_horizon_from_the_file(root):
    event = kline_event(minute_ms(0))
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(event)
        writer.sync()

    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as restarted:
        result = restarted.append(event)
        restarted.sync()

    assert result.outcome is AppendOutcome.DUPLICATE
    assert len(lines(root / "raw" / UM_KLINE_1M / DAY / EVENTS_FILE)) == 1


def test_the_deduplication_window_is_bounded_and_says_so(root):
    """The bound is documented rather than hidden; the reconciliation is the backstop."""
    with RawSink(root, UM_BOOK_TICKER, contract=CONTRACT, dedup_window=2) as writer:
        first = book_event(1, event_ms=minute_ms(0))
        writer.append(first)
        writer.append(book_event(2, event_ms=minute_ms(0) + 1_000))
        writer.append(book_event(3, event_ms=minute_ms(0) + 2_000))
        assert (
            writer.append(first).outcome is AppendOutcome.ACCEPTED
        ), "an observation older than the window is not detected, which is the stated limit"
        assert writer.append(book_event(3, event_ms=minute_ms(0) + 2_000)).outcome is (
            AppendOutcome.DUPLICATE
        )


def test_a_dedup_window_below_one_is_refused(root):
    with pytest.raises(RecorderSinkError, match="dedup_window"):
        RawSink(root, UM_KLINE_1M, contract=CONTRACT, dedup_window=0)


# --- F. write failure ----------------------------------------------------------
def test_a_failed_write_raises_rather_than_returning_an_outcome_that_can_be_ignored(sink):
    """A full disk must halt the stream, not return a value a caller can drop.

    Injected at the handle because that is where a real ``ENOSPC`` arrives, and
    because making a directory unwritable is not portable between the two
    platforms this has to pass on.
    """
    sink.append(kline_event(minute_ms(0)))
    real = sink._handle

    class Failing:
        def write(self, _data: bytes) -> int:
            raise OSError("no space left on device")

        def flush(self) -> None:  # pragma: no cover - not reached
            return None

    sink._handle = Failing()  # type: ignore[assignment]
    try:
        with pytest.raises(RecorderSinkError, match="could not append"):
            sink.append(kline_event(minute_ms(1)))
    finally:
        sink._handle = real
    assert len(lines(sink.events_path(DAY))) == 1


def test_a_failed_late_write_raises(root, monkeypatch):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(1439)))
        writer.append(kline_event(minute_ms(0, day=NEXT_DAY)))

        real_open = open

        def refuse(path, mode="r", *args, **kwargs):
            if str(path).endswith(LATE_FILE) and "a" in mode:
                raise OSError("read-only file system")
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", refuse)
        with pytest.raises(RecorderSinkError, match="late event"):
            writer.append(kline_event(minute_ms(100)))


# --- G. freezing ---------------------------------------------------------------
def test_freezing_compresses_verifies_and_records_both_digests(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        for index in range(3):
            writer.append(kline_event(minute_ms(index)))
        plain = writer.events_path(DAY).read_bytes()
        manifest_path = writer.freeze_day(DAY, provenance={"host_hash": "abc123"})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed = root / "raw" / UM_KLINE_1M / DAY / (EVENTS_FILE + GZIP_SUFFIX)

    assert manifest["manifest_schema"] == DAY_MANIFEST_SCHEMA
    assert manifest["stream"] == UM_KLINE_1M
    assert manifest["day"] == DAY
    assert manifest["storage_layout_version"] == 1
    assert manifest["contract"]["contract_hash"] == CONTRACT.contract_hash
    assert manifest["contract"]["prospective_from"] is None
    assert manifest["raw"]["rows"] == 3
    assert manifest["raw"]["path"] == f"raw/{UM_KLINE_1M}/{DAY}/{EVENTS_FILE}{GZIP_SUFFIX}"
    assert manifest["provenance"] == {"host_hash": "abc123"}
    assert manifest["counters"]["accepted"] == 3
    assert manifest["first_canonical_utc"] == "2026-09-19T00:00:00+00:00"
    assert manifest["last_canonical_utc"] == "2026-09-19T00:02:00+00:00"
    assert manifest["late"] is None

    assert gzip.decompress(packed.read_bytes()) == plain
    assert not (root / "raw" / UM_KLINE_1M / DAY / EVENTS_FILE).exists()
    import hashlib

    assert manifest["raw"]["sha256_ndjson"] == hashlib.sha256(plain).hexdigest()
    assert manifest["raw"]["sha256_gz"] == hashlib.sha256(packed.read_bytes()).hexdigest()


def test_the_compressed_bytes_do_not_carry_a_timestamp_or_a_file_name(root):
    """Otherwise the same events would compress differently on every run."""
    digests = set()
    for run in range(2):
        run_root = root / f"run{run}"
        with RawSink(run_root, UM_KLINE_1M, contract=CONTRACT) as writer:
            writer.append(kline_event(minute_ms(0)))
            writer.freeze_day(DAY)
        packed = run_root / "raw" / UM_KLINE_1M / DAY / (EVENTS_FILE + GZIP_SUFFIX)
        header = packed.read_bytes()[:10]
        assert header[4:8] == b"\x00\x00\x00\x00", "the gzip header carries an mtime"
        assert header[3] & 0x08 == 0, "the gzip header carries the source file name"
        digests.add(packed.read_bytes())
    assert len(digests) == 1


def test_a_frozen_day_is_never_frozen_again(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.freeze_day(DAY)
        with pytest.raises(RecorderSinkError, match="already exists"):
            writer.freeze_day(DAY)


def test_a_day_that_was_never_written_cannot_be_frozen(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        with pytest.raises(RecorderSinkError, match="never written"):
            writer.freeze_day(DAY)


def test_freezing_records_the_late_file_as_open_rather_than_final(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(1439)))
        writer.append(kline_event(minute_ms(0, day=NEXT_DAY)))
        writer.append(kline_event(minute_ms(100)))
        manifest_path = writer.freeze_day(DAY)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["late"]["rows"] == 1
    assert manifest["late"]["open"] is True
    assert manifest["late"]["path"] == f"raw/{UM_KLINE_1M}/{DAY}/{LATE_FILE}"


def test_freezing_refuses_a_directory_holding_both_the_plain_and_the_packed_file(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.sync()
        writer.gz_path(DAY).write_bytes(gzip.compress(b"{}\n"))
        with pytest.raises(RecorderSinkError, match="interrupted freeze"):
            writer.freeze_day(DAY)


def test_freezing_resumes_when_a_crash_left_the_compressed_file_without_a_manifest(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.append(kline_event(minute_ms(1)))
        writer.freeze_day(DAY)
    manifest_path = root / "raw" / UM_KLINE_1M / DAY / "manifest.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.unlink()

    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.freeze_day(DAY)
    again = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert again["raw"]["sha256_ndjson"] == expected["raw"]["sha256_ndjson"]
    assert again["raw"]["rows"] == expected["raw"]["rows"]


def test_a_manifest_names_no_absolute_path(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        manifest_path = writer.freeze_day(DAY)
    text = manifest_path.read_text(encoding="utf-8")
    assert "\\" not in text
    assert str(root) not in text
    assert ":" not in json.loads(text)["raw"]["path"]


def test_a_manifest_records_the_absence_of_a_contract_rather_than_inventing_one(root):
    with RawSink(root, UM_KLINE_1M) as writer:
        writer.append(kline_event(minute_ms(0)))
        manifest_path = writer.freeze_day(DAY)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["contract"] is None
    assert manifest["provenance"] is None


# --- H. reading back -----------------------------------------------------------
def test_reading_a_day_returns_the_main_file_then_the_late_file(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(1439)))
        writer.append(kline_event(minute_ms(0, day=NEXT_DAY)))
        writer.append(kline_event(minute_ms(100)))

    both = read_raw_events(root, UM_KLINE_1M, DAY)
    assert [event.minute_open_ms for event in both] == [minute_ms(1439), minute_ms(100)]
    main_only = read_raw_events(root, UM_KLINE_1M, DAY, include_late=False)
    assert [event.minute_open_ms for event in main_only] == [minute_ms(1439)]


def test_reading_a_frozen_day_reads_the_compressed_file(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        before = read_raw_events(root, UM_KLINE_1M, DAY)
        writer.freeze_day(DAY)
    assert read_raw_events(root, UM_KLINE_1M, DAY) == before


def test_reading_refuses_a_directory_holding_both_forms(root):
    with RawSink(root, UM_KLINE_1M, contract=CONTRACT) as writer:
        writer.append(kline_event(minute_ms(0)))
        writer.sync()
        writer.gz_path(DAY).write_bytes(gzip.compress(b""))
    with pytest.raises(RecorderSinkError, match="interrupted freeze"):
        read_raw_events(root, UM_KLINE_1M, DAY)


def test_a_day_with_no_directory_reads_as_nothing(root):
    assert read_raw_events(root, UM_KLINE_1M, DAY) == []
    assert available_days(root, UM_KLINE_1M) == []


# --- I. atomic metadata --------------------------------------------------------
def test_a_metadata_write_leaves_the_previous_file_when_it_fails(tmp_path, monkeypatch):
    target = tmp_path / "meta.json"
    write_json_atomic(target, {"first": 1})
    before = target.read_bytes()

    real_replace = __import__("os").replace

    def refuse(src, dst):
        raise OSError("crash between the temp file and the rename")

    monkeypatch.setattr("os.replace", refuse)
    with pytest.raises(RecorderSinkError, match="could not write"):
        write_json_atomic(target, {"second": 2})
    monkeypatch.setattr("os.replace", real_replace)

    assert target.read_bytes() == before
    assert json.loads(before) == {"first": 1}
    assert before.endswith(b"\n")
    assert b"\r\n" not in before
