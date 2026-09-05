"""The decision log: the bytes it writes, the chain it keeps, and what it refuses.

The log is the campaign's evidence, so the tests that matter are the ones that
would fail if the evidence could be changed without that being visible. Three
groups:

* **the bytes.** Section 9.2's serialization is pinned against literals written
  out by hand, and the hashes are recomputed here with :mod:`hashlib` over those
  literals rather than by calling the module's own helper — so a change to the
  canonical form fails these tests instead of moving with them. The
  OS-neutrality claim is checked the way a Windows runner would break it: no
  ``\\r`` anywhere, one ``\\n`` per record, and the whole file decodable as ASCII.
* **the chain.** Every way it can be wrong — a tampered field, a wrong
  ``record_hash``, a broken link, a duplicate or decreasing ``seq``, a tail that
  disagrees with the runner's persisted hash — is built deliberately and asserted
  to be found, and each is asserted *not* to be found in the sound file it was
  built from.
* **the refusals.** ``NaN``, the infinities, a raw ``Decimal``, a ``Z``-suffixed
  timestamp, a caller-supplied ``seq``: each has a passing case beside it, so the
  test says where the boundary is rather than that one exists.

Nothing here writes a real campaign, and nothing here computes an economic
quantity: the records are shaped like section 9.1's and their contents are
arbitrary.
"""

from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

from chimera.demo.decision_log import (
    CORE_FIELDS,
    DECISION_RECORD_SCHEMA,
    EVIDENCE_KINDS,
    OPERATIONAL_KINDS,
    UNCLASSIFIED_KINDS,
    ZERO_PREV_HASH,
    ChainFault,
    DecisionLog,
    DecisionLogError,
    DecisionLogTailError,
    RecordKind,
    canonical_json,
    canonical_line,
    compute_record_hash,
    day_files,
    decimal_str,
    is_evidence,
    is_hash,
    iso_minute,
    read_records,
    recover_tail,
    require_iso_minute,
    require_iso_utc,
    verify_chain,
    verify_log,
)

#: 2026-11-23T00:02:00.123456789Z, inside the UTC day 2026-11-23.
NOW_NS = 1_795_392_120_123_456_789
#: One minute later, same UTC day.
NEXT_NS = NOW_NS + 60_000_000_000
#: The first instant of 2026-11-24, for the day-rotation tests.
NEXT_DAY_NS = 1_795_478_400_000_000_000
#: The first instant of 2026-11-25, so a whole day can be deleted from between.
THIRD_DAY_NS = 1_795_564_800_000_000_000

A_HASH = "sha256:" + "a" * 64


def payload(**overrides: object) -> dict[str, object]:
    """A minimal, valid record payload. Overridden per test."""
    record: dict[str, object] = {
        "kind": RecordKind.STARTUP.value,
        "minute": None,
        "runner_now_ns": NOW_NS,
    }
    record.update(overrides)
    return record


# --- canonical bytes --------------------------------------------------------
#: The first record of a campaign, written out by hand in section 9.2's form:
#: keys sorted, no spaces, ASCII only, and without `record_hash`, which is what
#: the hash is taken over.
FIRST_BODY = (
    b'{"kind":"STARTUP",'
    b'"minute":null,'
    b'"prev_hash":"sha256:' + b"0" * 64 + b'",'
    b'"runner_now_ns":1795392120123456789,'
    b'"schema":"chimera.decision-record/1",'
    b'"seq":1}'
)
FIRST_HASH = "sha256:" + hashlib.sha256(FIRST_BODY).hexdigest()
#: The same record as it is written to the file: `record_hash` sorts between
#: `prev_hash` and `runner_now_ns`, and the line ends in exactly one newline.
FIRST_LINE = (
    b'{"kind":"STARTUP",'
    b'"minute":null,'
    b'"prev_hash":"sha256:' + b"0" * 64 + b'",'
    b'"record_hash":"' + FIRST_HASH.encode("ascii") + b'",'
    b'"runner_now_ns":1795392120123456789,'
    b'"schema":"chimera.decision-record/1",'
    b'"seq":1}\n'
)


def test_the_first_record_is_exactly_these_bytes(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        appended = log.append(payload())
    assert appended.seq == 1
    assert appended.prev_hash == ZERO_PREV_HASH
    assert appended.record_hash == FIRST_HASH
    assert appended.line == FIRST_LINE
    assert appended.path.read_bytes() == FIRST_LINE
    assert appended.path.name == "2026-11-23.ndjson"


def test_the_record_hash_is_sha256_over_the_record_without_it() -> None:
    body = json.loads(FIRST_BODY.decode("ascii"))
    assert compute_record_hash(body) == FIRST_HASH
    # And hashing a record that still carries its own digest is refused, because
    # doing it would make every later verification pass while proving nothing.
    with pytest.raises(DecisionLogError, match="without record_hash"):
        compute_record_hash({**body, "record_hash": FIRST_HASH})


def test_the_hash_is_independent_of_the_order_the_fields_were_written_in() -> None:
    forward = {"a": 1, "b": {"x": 1, "y": 2}, "c": [1, 2]}
    backward = {"c": [1, 2], "b": {"y": 2, "x": 1}, "a": 1}
    assert canonical_json(forward) == canonical_json(backward)
    assert compute_record_hash(forward) == compute_record_hash(backward)
    # Two-sided: a genuine difference in content still moves it.
    assert compute_record_hash(forward) != compute_record_hash({**forward, "a": 2})


def test_key_order_in_the_payload_does_not_change_the_written_bytes(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    with DecisionLog.open(first) as log:
        one = log.append({"runner_now_ns": NOW_NS, "minute": None, "kind": "STARTUP"})
    with DecisionLog.open(second) as log:
        two = log.append({"kind": "STARTUP", "runner_now_ns": NOW_NS, "minute": None})
    assert one.line == two.line == FIRST_LINE


def test_the_bytes_are_os_neutral(tmp_path: Path) -> None:
    """The claim a Windows runner would break: no CRLF, ASCII only, one \\n each."""
    with DecisionLog.open(tmp_path) as log:
        log.append(payload(note="a line\r\nand a path C:\\demo\\state"))
        log.append(payload(runner_now_ns=NEXT_NS, note="\u00e9\u4e2d"))
    data = (tmp_path / "decision_log" / "2026-11-23.ndjson").read_bytes()
    assert b"\r" not in data
    assert data.count(b"\n") == 2
    assert data.endswith(b"\n")
    # ensure_ascii=True, so the whole file is ASCII whatever the record carried.
    text = data.decode("ascii")
    assert "\\r\\n" in text and "\\u00e9" in text and "\\u4e2d" in text
    # The backslash of a Windows path is escaped, not emitted raw, so the same
    # record is the same bytes on both platforms.
    assert "C:\\\\demo\\\\state" in text


def test_a_non_ascii_record_is_written_as_escapes_not_as_utf8() -> None:
    line = canonical_line({"detail": "\u20ac"})
    assert line == b'{"detail":"\\u20ac"}\n'
    assert line.decode("ascii")


def test_a_file_written_with_crlf_endings_is_refused(tmp_path: Path) -> None:
    """What a Windows host opening the file in text mode would produce.

    The record still parses — JSON tolerates a trailing carriage return — and it
    still hashes correctly, so only a byte comparison catches it. That is the
    whole reason the verifier makes one.
    """
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    assert verify_chain(first.path).ok
    first.path.write_bytes(first.line.replace(b"\n", b"\r\n"))
    verification = verify_chain(first.path)
    assert verification.faults == (ChainFault.NON_CANONICAL_BYTES,)
    assert verification.is_forged and not verification.is_torn


# --- the chain --------------------------------------------------------------
def test_a_multi_record_chain_links_and_verifies(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        second = log.append(
            payload(
                kind="DECISION",
                minute="2026-11-23T00:03:00+00:00",
                runner_now_ns=NEXT_NS,
                config_hash=A_HASH,
            )
        )
        third = log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))

    assert [first.seq, second.seq, third.seq] == [1, 2, 3]
    assert first.prev_hash == ZERO_PREV_HASH
    assert second.prev_hash == first.record_hash
    assert third.prev_hash == second.record_hash

    verification = verify_chain(first.path)
    assert verification.ok
    assert verification.records == 3
    assert verification.first_seq == 1 and verification.last_seq == 3
    assert verification.last_record_hash == third.record_hash
    assert verification.last_runner_now_ns == NEXT_NS
    assert verification.torn_tail_bytes == 0
    assert verification.summary() == "3 record(s), chain intact"


def test_an_empty_log_verifies_and_holds_nothing(tmp_path: Path) -> None:
    log = DecisionLog.open(tmp_path)
    root = tmp_path / "decision_log"
    assert day_files(root) == []
    verification = verify_log(root)
    assert verification.ok and verification.records == 0
    assert verification.last_record_hash is None
    assert log.next_seq == 1
    assert log.last_record_hash == ZERO_PREV_HASH


def test_editing_a_field_breaks_that_record_and_nothing_reports_it_as_torn(
    tmp_path: Path,
) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))
    path = first.path
    assert verify_chain(path).ok

    lines = path.read_bytes().splitlines(keepends=True)
    tampered = lines[0].replace(b'"kind":"STARTUP"', b'"kind":"HALT"   ')
    assert len(tampered) == len(lines[0])
    path.write_bytes(tampered + lines[1])

    verification = verify_chain(path)
    assert not verification.ok
    assert verification.is_forged and not verification.is_torn
    assert ChainFault.RECORD_HASH_MISMATCH in verification.faults
    assert verification.defects[0].line == 1


def test_a_reformatted_record_is_refused_even_though_it_means_the_same(
    tmp_path: Path,
) -> None:
    """Section 10 compares bytes, so a semantically equal record is not enough."""
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    record = json.loads(first.line.decode("ascii"))
    first.path.write_bytes(
        json.dumps(record, sort_keys=True, separators=(", ", ": ")).encode("ascii") + b"\n"
    )
    verification = verify_chain(first.path)
    assert not verification.ok
    assert verification.faults == (ChainFault.NON_CANONICAL_BYTES,)
    assert verification.is_forged


def test_a_broken_link_is_found_even_when_every_record_hashes_correctly(
    tmp_path: Path,
) -> None:
    """Two sound records from two different chains, spliced together."""
    left = tmp_path / "left"
    right = tmp_path / "right"
    with DecisionLog.open(left) as log:
        one = log.append(payload())
    with DecisionLog.open(right) as log:
        log.append(payload(note="a different first record"))
        two = log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))

    spliced = tmp_path / "spliced.ndjson"
    spliced.write_bytes(one.line + two.line)
    verification = verify_chain(spliced)
    assert ChainFault.PREV_HASH_MISMATCH in verification.faults
    # Each record on its own is untouched, which is exactly why the link matters.
    assert ChainFault.RECORD_HASH_MISMATCH not in verification.faults
    assert verification.is_forged


@pytest.mark.parametrize(
    ("second_seq", "expected"),
    [(1, ChainFault.DUPLICATE_SEQ), (0, ChainFault.SEQ_NOT_MONOTONE)],
)
def test_a_repeated_or_decreasing_seq_is_found(
    tmp_path: Path, second_seq: int, expected: ChainFault
) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    body = json.loads(first.line.decode("ascii"))
    forged = {k: v for k, v in body.items() if k != "record_hash"}
    forged["seq"] = second_seq
    forged["prev_hash"] = first.record_hash
    forged["record_hash"] = compute_record_hash(
        {k: v for k, v in forged.items() if k != "record_hash"}
    )
    first.path.write_bytes(first.line + canonical_line(forged))

    verification = verify_chain(first.path)
    assert expected in verification.faults
    # The forged record hashes correctly and links correctly; only its seq is wrong.
    assert ChainFault.RECORD_HASH_MISMATCH not in verification.faults
    assert ChainFault.PREV_HASH_MISMATCH not in verification.faults


def test_a_seq_that_still_increases_is_not_a_fault(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        log.append(payload())
        appended = log.append(payload(runner_now_ns=NEXT_NS))
    assert verify_chain(appended.path).ok


def test_a_torn_tail_is_reported_as_torn_and_not_as_forged(tmp_path: Path) -> None:
    """A crash between the write and the fsync: the last line has no newline."""
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        second = log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))
    path = first.path
    torn = second.line[:40]
    path.write_bytes(first.line + torn)

    verification = verify_chain(path)
    assert not verification.ok
    assert verification.is_torn and not verification.is_forged
    assert verification.faults == (ChainFault.TORN_TAIL,)
    assert verification.torn_tail_bytes == len(torn)
    # The complete records before the tear are still counted and still sound.
    assert verification.records == 1
    assert verification.last_record_hash == first.record_hash


def test_a_tail_that_parses_but_has_no_newline_is_still_torn(tmp_path: Path) -> None:
    """Without the newline there is no evidence the writer finished the record."""
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    first.path.write_bytes(first.line.rstrip(b"\n"))
    verification = verify_chain(first.path)
    assert verification.is_torn and not verification.is_forged
    assert verification.records == 0


@pytest.mark.parametrize(
    "line",
    [
        b"not json at all\n",
        b"[1,2,3]\n",
        b"\n",
        b'{"schema":"chimera.decision-record/2","seq":1}\n',
        b'{"seq":1,"prev_hash":"x","record_hash":"y","kind":"STARTUP","runner_now_ns":1}\n',
    ],
)
def test_a_malformed_line_is_reported_and_never_ignored(tmp_path: Path, line: bytes) -> None:
    path = tmp_path / "2026-11-23.ndjson"
    path.write_bytes(line)
    verification = verify_chain(path)
    assert verification.faults == (ChainFault.MALFORMED_RECORD,)
    assert verification.is_forged and not verification.is_torn
    assert verification.records == 0


def test_a_line_carrying_nan_is_malformed_although_python_would_decode_it(
    tmp_path: Path,
) -> None:
    """Python's decoder accepts ``NaN``; no other reader does, so it is refused."""
    assert json.loads('{"x": NaN}')["x"] != json.loads('{"x": NaN}')["x"]
    path = tmp_path / "2026-11-23.ndjson"
    path.write_bytes(
        b'{"kind":"STARTUP","minute":null,"prev_hash":"' + ZERO_PREV_HASH.encode() + b'",'
        b'"record_hash":"' + A_HASH.encode() + b'","runner_now_ns":1,"schema":'
        b'"chimera.decision-record/1","seq":1,"x":NaN}\n'
    )
    verification = verify_chain(path)
    assert verification.faults == (ChainFault.MALFORMED_RECORD,)
    assert "NaN" in verification.defects[0].detail


def test_verification_stops_at_a_line_it_cannot_read(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        second = log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))
    first.path.write_bytes(first.line + b"garbage\n" + second.line)
    verification = verify_chain(first.path)
    assert verification.faults == (ChainFault.MALFORMED_RECORD,)
    assert verification.records == 1
    assert verification.defects[0].line == 2


def test_the_persisted_tail_hash_is_compared_and_the_disagreement_is_reported(
    tmp_path: Path,
) -> None:
    """Section 9.3's ``last_record_hash``: reported, never resolved here."""
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        second = log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))

    assert verify_chain(first.path, expected_last_hash=second.record_hash).ok
    behind = verify_chain(first.path, expected_last_hash=first.record_hash)
    assert behind.faults == (ChainFault.TAIL_HASH_MISMATCH,)
    assert first.record_hash in behind.defects[0].detail
    assert second.record_hash in behind.defects[0].detail
    # A state file naming a hash for a log that holds nothing is the same fault.
    empty = verify_log(tmp_path / "nothing", expected_last_hash=A_HASH)
    assert empty.faults == (ChainFault.TAIL_HASH_MISMATCH,)
    assert verify_log(tmp_path / "nothing", expected_last_hash=ZERO_PREV_HASH).ok


def test_the_first_record_of_a_file_can_be_chained_to_the_previous_day(
    tmp_path: Path,
) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        second = log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_DAY_NS))
    assert first.path != second.path
    assert second.path.name == "2026-11-24.ndjson"

    assert verify_chain(second.path, expected_prev_hash=first.record_hash).ok
    broken = verify_chain(second.path, expected_prev_hash=A_HASH)
    assert broken.faults == (ChainFault.PREV_HASH_MISMATCH,)


def test_verify_log_checks_the_link_between_days(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        log.append(payload())
        middle = log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_DAY_NS))
        last = log.append(payload(kind="STARTUP", runner_now_ns=THIRD_DAY_NS))
    root = tmp_path / "decision_log"
    whole = verify_log(root, expected_last_hash=last.record_hash)
    assert whole.ok
    assert whole.records == 3
    assert [path.name for path in whole.files] == [
        "2026-11-23.ndjson",
        "2026-11-24.ndjson",
        "2026-11-25.ndjson",
    ]

    # Deleting a whole day leaves two files that each verify on their own; only
    # the link between them says a day went missing.
    middle.path.unlink()
    torn_out = verify_log(root)
    assert ChainFault.PREV_HASH_MISMATCH in torn_out.faults
    assert verify_chain(root / "2026-11-23.ndjson").ok
    assert verify_chain(root / "2026-11-25.ndjson").ok


def test_read_records_yields_what_was_written_and_stops_at_a_bad_line(
    tmp_path: Path,
) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))
    assert [record["seq"] for record in read_records(first.path)] == [1, 2]
    first.path.write_bytes(first.line + b"half a record")
    assert [record["seq"] for record in read_records(first.path)] == [1]


# --- append semantics -------------------------------------------------------
def test_every_committed_record_is_fsynced_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = os.fsync
    monkeypatch.setattr(
        "chimera.demo.decision_log.os.fsync", lambda fd: (calls.append(fd), real(fd))[1]
    )
    log = DecisionLog.open(tmp_path)
    log.append(payload())
    assert len(calls) == 1
    log.append(payload(runner_now_ns=NEXT_NS))
    assert len(calls) == 2
    log.close()
    # close() syncs the handle once more before releasing it.
    assert len(calls) == 3


def test_the_file_is_appended_to_and_never_rewritten(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    before = first.path.read_bytes()

    resumed = DecisionLog.open(tmp_path)
    assert resumed.next_seq == 2
    assert resumed.last_record_hash == first.record_hash
    assert resumed.last_runner_now_ns == NOW_NS
    second = resumed.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))
    resumed.close()

    after = first.path.read_bytes()
    assert after.startswith(before)
    assert after == before + second.line
    assert second.prev_hash == first.record_hash
    assert verify_chain(first.path).ok


def test_open_refuses_to_append_onto_a_tail_that_does_not_verify(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    first.path.write_bytes(first.line + b"a torn half-record")
    with pytest.raises(DecisionLogError, match="TORN_TAIL"):
        DecisionLog.open(tmp_path)


def test_a_day_earlier_than_one_already_on_disk_is_refused(tmp_path: Path) -> None:
    """Reachable only through the constructor, which carries no last instant."""
    with DecisionLog.open(tmp_path) as log:
        later = log.append(payload(runner_now_ns=NEXT_DAY_NS))
    resumed = DecisionLog(
        tmp_path / "decision_log", seq=later.seq, prev_hash=later.record_hash
    )
    with pytest.raises(DecisionLogError, match="read in name order"):
        resumed.append(payload(runner_now_ns=NOW_NS))
    # The same log accepts a record for the day that is already there.
    assert resumed.append(payload(runner_now_ns=NEXT_DAY_NS + 1)).day == "2026-11-24"
    resumed.close()


def test_a_clock_that_goes_backwards_is_refused(tmp_path: Path) -> None:
    log = DecisionLog.open(tmp_path)
    log.append(payload(runner_now_ns=NEXT_NS))
    # Exactly at is allowed: several records can share one tick.
    log.append(payload(runner_now_ns=NEXT_NS))
    with pytest.raises(DecisionLogError, match="precedes the last committed record"):
        log.append(payload(runner_now_ns=NEXT_NS - 1))
    log.close()
    assert verify_chain(log.path_for_day("2026-11-23")).ok


def test_a_failed_write_is_raised_and_not_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = DecisionLog.open(tmp_path)
    log.append(payload())
    monkeypatch.setattr(
        "chimera.demo.decision_log.os.fsync",
        lambda fd: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(DecisionLogError, match="could not commit decision record"):
        log.append(payload(runner_now_ns=NEXT_NS))
    # And the log is closed for writing: continuing from an in-memory chain head
    # that may already be behind the file is how one seq gets written twice.
    monkeypatch.undo()
    with pytest.raises(DecisionLogError, match="accepts no further records"):
        log.append(payload(runner_now_ns=NEXT_NS))


# --- refusals ---------------------------------------------------------------
@pytest.mark.parametrize("reserved", ["schema", "seq", "prev_hash", "record_hash"])
def test_a_caller_cannot_stamp_the_fields_the_log_owns(tmp_path: Path, reserved: str) -> None:
    with DecisionLog.open(tmp_path) as log:
        with pytest.raises(DecisionLogError, match="stamps"):
            log.append(payload(**{reserved: 1}))


@pytest.mark.parametrize("missing", ["kind", "minute", "runner_now_ns"])
def test_every_core_field_is_required_to_be_present(tmp_path: Path, missing: str) -> None:
    record = payload()
    del record[missing]
    with DecisionLog.open(tmp_path) as log:
        with pytest.raises(DecisionLogError, match="missing required field"):
            log.append(record)


def test_minute_may_be_null_but_never_absent(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        assert log.append(payload(minute=None)).record["minute"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_the_infinities_are_refused_at_write_time(
    tmp_path: Path, value: float
) -> None:
    with DecisionLog.open(tmp_path) as log:
        with pytest.raises(DecisionLogError, match="not JSON"):
            log.append(payload(ledger_effect={"equity": value}))
        # A finite float in the same place is fine.
        assert log.append(payload(ledger_effect={"equity": 1.5})).seq == 1


def test_a_raw_decimal_is_refused_and_the_message_names_the_way_out(
    tmp_path: Path,
) -> None:
    with DecisionLog.open(tmp_path) as log:
        with pytest.raises(DecisionLogError, match="decimal_str"):
            log.append(payload(signal={"target_qty": Decimal("0.5")}))
        assert log.append(payload(signal={"target_qty": decimal_str(Decimal("0.5"), scale=3)}))


@pytest.mark.parametrize("value", [Path("/tmp/state"), {1, 2}, b"bytes", 1 + 2j])
def test_a_value_that_is_not_json_is_refused_by_name(tmp_path: Path, value: object) -> None:
    with DecisionLog.open(tmp_path) as log:
        with pytest.raises(DecisionLogError, match="not a JSON value"):
            log.append(payload(detail=value))


def test_a_non_string_object_key_is_refused(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        with pytest.raises(DecisionLogError, match="keys are strings"):
            log.append(payload(inputs={1: "one"}))


def test_a_hash_field_at_any_depth_must_carry_the_prefix(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        with pytest.raises(DecisionLogError, match="record.rule.rule_hash"):
            log.append(payload(rule={"id": "R1_carry", "rule_hash": "a" * 64}))
        with pytest.raises(DecisionLogError, match="record.inputs.contract_hash"):
            log.append(payload(inputs={"contract_hash": None}))
        # Prefixed, it passes; and a digest deliberately written bare is named
        # `*_digest` and is not caught by the rule.
        assert log.append(
            payload(
                rule={"rule_hash": A_HASH},
                inputs={"um_minute_digest": "b" * 64, "contract_hash": A_HASH},
            )
        )


def test_an_unknown_kind_is_refused_and_every_named_kind_is_accepted(
    tmp_path: Path,
) -> None:
    with DecisionLog.open(tmp_path) as log:
        with pytest.raises(DecisionLogError, match="vocabulary is closed"):
            log.append(payload(kind="PROFIT"))
    for index, kind in enumerate(sorted(k.value for k in RecordKind)):
        with DecisionLog.open(tmp_path / str(index)) as log:
            assert log.append(payload(kind=kind)).kind is RecordKind(kind)


def test_the_kinds_section_9_1_names_are_exactly_the_ones_the_log_admits() -> None:
    assert {k.value for k in RecordKind} == {
        "DECISION",
        "FUNDING",
        "RECONCILIATION",
        "OPERATOR",
        "HALT",
        "RESUME",
        "STARTUP",
        "SHUTDOWN",
        "INCOMPLETE_STATE",
        "SKIPPED_STALE",
        "LIQUIDATION_TOUCH",
        # Section 9.3's recovery record. The kind is part of the schema; when one
        # is written is the runner's decision and not this module's.
        "RECOVERY",
    }


def test_section_9_4s_two_lists_are_restated_and_not_derived() -> None:
    """Both halves are literal, because both halves are a scientific boundary."""
    assert EVIDENCE_KINDS == {
        RecordKind.DECISION,
        RecordKind.FUNDING,
        RecordKind.RECONCILIATION,
        RecordKind.OPERATOR,
        RecordKind.LIQUIDATION_TOUCH,
    }
    assert OPERATIONAL_KINDS == {
        RecordKind.STARTUP,
        RecordKind.SHUTDOWN,
        RecordKind.INCOMPLETE_STATE,
        RecordKind.SKIPPED_STALE,
    }
    assert is_evidence("DECISION") and is_evidence(RecordKind.OPERATOR)
    assert not is_evidence("STARTUP")


def test_the_kinds_9_4_does_not_classify_are_left_unclassified() -> None:
    """HALT, RESUME and RECOVERY are in neither of section 9.4's lists.

    Deriving the operational set as the complement of the evidence set would put
    them there silently, which decides — without the adopted plan saying so —
    that a campaign's halts are not scored. PR-12 and PR-14 own that; this test
    pins the refusal to answer it here, and pins that the three sets still cover
    every kind so nothing can go missing from all of them.
    """
    assert UNCLASSIFIED_KINDS == {
        RecordKind.HALT,
        RecordKind.RESUME,
        RecordKind.RECOVERY,
    }
    assert EVIDENCE_KINDS | OPERATIONAL_KINDS | UNCLASSIFIED_KINDS == set(RecordKind)
    assert not EVIDENCE_KINDS & OPERATIONAL_KINDS
    assert not EVIDENCE_KINDS & UNCLASSIFIED_KINDS
    assert not OPERATIONAL_KINDS & UNCLASSIFIED_KINDS
    # An unclassified kind is not evidence, which is what `is_evidence` answers;
    # whether it is *scored* is the question deliberately left open.
    assert not is_evidence(RecordKind.HALT)
    assert not is_evidence(RecordKind.RECOVERY)


# --- timestamps and numbers -------------------------------------------------
@pytest.mark.parametrize(
    "good",
    [
        "2026-11-23T00:01:00+00:00",
        "2026-11-23T00:01:00.123456+00:00",
        "1970-01-01T00:00:00+00:00",
    ],
)
def test_a_canonical_utc_timestamp_is_accepted(good: str) -> None:
    assert require_iso_utc(good, field_name="minute") == good


@pytest.mark.parametrize(
    "bad",
    [
        "2026-11-23T00:01:00Z",
        "2026-11-23T00:01:00",
        "2026-11-23T00:01:00+01:00",
        "2026-11-23T00:01:00-00:00",
        "2026-11-23 00:01:00+00:00",
        "2026-11-23T00:01:00.000000+00:00",
        "2026-11-23",
        1795305720,
    ],
)
def test_a_timestamp_that_is_not_canonical_utc_is_refused(bad: object) -> None:
    with pytest.raises(DecisionLogError):
        require_iso_utc(bad, field_name="minute")


def test_the_record_minute_is_held_to_the_same_rule(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        with pytest.raises(DecisionLogError, match="ends in '\\+00:00'"):
            log.append(payload(kind="DECISION", minute="2026-11-23T00:01:00Z"))
        assert log.append(payload(kind="DECISION", minute="2026-11-23T00:01:00+00:00"))


def test_iso_minute_renders_the_form_the_record_requires() -> None:
    rendered = iso_minute(1_795_392_060_000_000_000)
    assert rendered == "2026-11-23T00:01:00+00:00"
    assert require_iso_utc(rendered, field_name="minute") == rendered


@pytest.mark.parametrize(
    ("value", "scale", "expected"),
    [
        (Decimal("0.5"), 3, "0.500"),
        (Decimal("0.500"), 3, "0.500"),
        (Decimal("0"), 3, "0.000"),
        (Decimal("-0.000"), 3, "0.000"),
        (Decimal("-1.5"), 2, "-1.50"),
        (Decimal("61234.5"), 2, "61234.50"),
        (Decimal("100"), 0, "100"),
        (Decimal("1E+2"), 2, "100.00"),
        # str() would render this one as "1E-8"; a fee in scientific notation is
        # a number two readers can disagree about.
        (Decimal("0.00000001"), 8, "0.00000001"),
    ],
)
def test_a_decimal_is_rendered_at_the_scale_of_its_constraint(
    value: Decimal, scale: int, expected: str
) -> None:
    assert decimal_str(value, scale=scale) == expected


@pytest.mark.parametrize(
    ("value", "scale", "expected"),
    [
        # 2**53 + 1, the first integer a binary float cannot hold exactly.
        (Decimal("9007199254740993"), 0, "9007199254740993"),
        # A price carrying more significant digits than a float has.
        (Decimal("61234.567890123456"), 12, "61234.567890123456"),
        # The canonical one: a tenth is not a tenth in binary.
        (Decimal("0.1"), 20, "0.10000000000000000000"),
    ],
)
def test_a_decimal_is_never_routed_through_binary_floating_point(
    value: Decimal, scale: int, expected: str
) -> None:
    """Each of these renders differently if the value passes through a float.

    The second assertion is what makes this a test of the *route* rather than of
    the digits: it pins that the float rendering of the same quantity is a
    different string, so an implementation that promoted the Decimal on the way
    out could not agree with the first assertion by accident.
    """
    assert decimal_str(value, scale=scale) == expected
    assert format(float(value), f".{scale}f") != expected


def test_a_decimal_that_does_not_fit_the_scale_is_refused_not_rounded() -> None:
    # Exactly at the scale: accepted.
    assert decimal_str(Decimal("0.001"), scale=3) == "0.001"
    # One digit past it: refused rather than rounded to 0.001 or 0.000.
    with pytest.raises(DecisionLogError, match="does not fit scale"):
        decimal_str(Decimal("0.0015"), scale=3)


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_a_non_finite_decimal_is_refused(bad: Decimal) -> None:
    with pytest.raises(DecisionLogError, match="not a finite quantity"):
        decimal_str(bad, scale=2)


def test_decimal_str_does_not_promote_a_float() -> None:
    """The promotion is exactly where binary rounding would enter the evidence."""
    with pytest.raises(DecisionLogError, match="renders a Decimal"):
        decimal_str(0.1, scale=2)  # type: ignore[arg-type]
    with pytest.raises(DecisionLogError, match="non-negative integer"):
        decimal_str(Decimal("1"), scale=-1)


def test_the_hash_form_is_the_only_one_accepted() -> None:
    assert is_hash(ZERO_PREV_HASH)
    assert is_hash("sha256:" + "0123456789abcdef" * 4)
    assert not is_hash("0" * 64)
    assert not is_hash("sha256:" + "0" * 63)
    assert not is_hash("sha256:" + "A" * 64)
    assert not is_hash("SHA256:" + "0" * 64)
    assert not is_hash(None)


def test_the_schema_string_is_the_one_the_plan_names() -> None:
    assert DECISION_RECORD_SCHEMA == "chimera.decision-record/1"
    assert ZERO_PREV_HASH == "sha256:" + "0" * 64


# --- the campaign's first record --------------------------------------------
def test_deleting_the_campaigns_first_record_is_found(tmp_path: Path) -> None:
    """The head of the chain is the record a per-file walk has nothing to check.

    Every surviving record still hashes correctly and still links to the one
    before it, and section 9.3's persisted ``last_record_hash`` names a tail that
    was never touched. Only the anchor — section 9.2's rule that the first record
    of a campaign chains to :data:`ZERO_PREV_HASH` — says the opening is missing.
    """
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))
        last = log.append(payload(kind="STARTUP", runner_now_ns=NEXT_DAY_NS))
    root = tmp_path / "decision_log"
    sound = verify_log(root, expected_last_hash=last.record_hash)
    assert sound.ok and sound.records == 3 and sound.first_seq == 1

    lines = first.path.read_bytes().splitlines(keepends=True)
    first.path.write_bytes(lines[1])
    beheaded = verify_log(root, expected_last_hash=last.record_hash)
    assert not beheaded.ok
    assert beheaded.faults == (ChainFault.PREV_HASH_MISMATCH,)
    assert beheaded.first_seq == 2
    assert beheaded.is_forged


def test_deleting_the_campaigns_first_day_is_found(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_DAY_NS))
        log.append(payload(kind="STARTUP", runner_now_ns=THIRD_DAY_NS))
    root = tmp_path / "decision_log"
    assert verify_log(root).ok

    first.path.unlink()
    gone = verify_log(root)
    assert gone.faults == (ChainFault.PREV_HASH_MISMATCH,)
    # Both surviving files verify on their own, which is exactly why the anchor
    # rather than the per-file walk is what catches this.
    assert verify_chain(root / "2026-11-24.ndjson").ok
    assert verify_chain(root / "2026-11-25.ndjson", expected_prev_hash=None).ok


def test_the_first_records_seq_is_anchored_too(tmp_path: Path) -> None:
    """A campaign starts at seq 1, and a head that claims otherwise is found."""
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    root = tmp_path / "decision_log"
    assert verify_log(root).ok

    body = {
        k: v for k, v in json.loads(first.line.decode("ascii")).items() if k != "record_hash"
    }
    body["seq"] = 0
    body["record_hash"] = compute_record_hash(body)
    first.path.write_bytes(canonical_line(body))
    forged = verify_log(root)
    # The record hashes correctly and chains to the zero hash; only its seq is wrong.
    assert forged.faults == (ChainFault.DUPLICATE_SEQ,)


def test_open_refuses_a_log_whose_first_record_was_removed(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))
    lines = first.path.read_bytes().splitlines(keepends=True)
    first.path.write_bytes(lines[1])
    with pytest.raises(DecisionLogTailError, match="PREV_HASH_MISMATCH") as raised:
        DecisionLog.open(tmp_path)
    assert raised.value.verification is not None
    assert raised.value.verification.is_forged


def test_an_empty_day_file_does_not_move_the_anchor_off_the_first_record(
    tmp_path: Path,
) -> None:
    """The anchor belongs to the first file that holds a record, not to the first
    file that exists.

    An empty day file is reachable — a crash between opening a day and committing
    its first record leaves one — and it is also the cheapest thing to leave in
    place of a day whose contents were removed. Counting it as the campaign's
    first file would put :data:`ZERO_PREV_HASH` on a file that carries nothing
    and leave the real opening record the one thing :meth:`DecisionLog.open`
    never checks, which is exactly the hole the anchor exists to close.
    """
    root = tmp_path / "decision_log"
    root.mkdir(parents=True)
    (root / "2026-11-22.ndjson").write_bytes(b"")
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload(runner_now_ns=NEXT_DAY_NS))
        log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_DAY_NS + 60_000_000_000))

    # Two-sided: the untouched campaign still reopens with the empty file present.
    with DecisionLog.open(tmp_path) as log:
        assert log.next_seq == 3

    lines = first.path.read_bytes().splitlines(keepends=True)
    first.path.write_bytes(lines[1])
    assert verify_log(root).faults == (ChainFault.PREV_HASH_MISMATCH,)
    with pytest.raises(DecisionLogTailError, match="PREV_HASH_MISMATCH"):
        DecisionLog.open(tmp_path)


# --- torn, forged, and a tail the state file disagrees with ------------------
def test_a_tail_the_state_file_disagrees_with_is_not_a_forgery(tmp_path: Path) -> None:
    """Section 9.3's write order puts the log ahead of the state on any crash.

    State files, then the log record, then ``last_record_hash``: a crash in the
    last window leaves every record complete, canonical and correctly chained,
    and only the runner's persisted head behind. A recovery routine that branched
    on forgery would declare an ordinary power cut to be tampering.
    """
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
        second = log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))

    behind = verify_chain(first.path, expected_last_hash=first.record_hash)
    assert behind.faults == (ChainFault.TAIL_HASH_MISMATCH,)
    assert not behind.ok
    assert behind.is_tail_disagreement
    assert not behind.is_forged and not behind.is_torn

    # Two-sided: a record that is actually wrong is a forgery and says so, and a
    # log that agrees with the persisted head has none of the three.
    agreed = verify_chain(first.path, expected_last_hash=second.record_hash)
    assert agreed.ok and not agreed.is_tail_disagreement
    lines = first.path.read_bytes().splitlines(keepends=True)
    first.path.write_bytes(
        lines[0].replace(b'"kind":"STARTUP"', b'"kind":"HALT"   ') + lines[1]
    )
    forged = verify_chain(first.path, expected_last_hash=second.record_hash)
    assert forged.faults == (ChainFault.RECORD_HASH_MISMATCH,)
    assert forged.is_forged and not forged.is_tail_disagreement


def test_a_number_that_overflowed_to_infinity_is_reported_not_raised(
    tmp_path: Path,
) -> None:
    """``1e999`` is not a token ``parse_constant`` ever sees; it is a float.

    ``float("1e999")`` is ``inf`` and raises nothing, so the line parses and
    reaches the hash — where the canonical serializer refuses it. The verifier's
    contract is to report what it found on corruption-controlled input, so this
    must be a defect and never an exception escaping the walk.
    """

    def line(value: bytes) -> bytes:
        return (
            b'{"kind":"STARTUP","minute":null,"prev_hash":"' + ZERO_PREV_HASH.encode() + b'",'
            b'"record_hash":"' + A_HASH.encode() + b'","runner_now_ns":1,"schema":'
            b'"chimera.decision-record/1","seq":1,"x":' + value + b"}\n"
        )

    assert json.loads('{"x":1e999}')["x"] == float("inf")
    path = tmp_path / "2026-11-23.ndjson"
    for overflowed in (b"1e999", b"-1e999"):
        path.write_bytes(line(overflowed))
        verification = verify_chain(path)
        assert verification.faults == (ChainFault.MALFORMED_RECORD,)
        assert "non-finite" in verification.defects[0].detail
    # A finite number in the same place is read normally, and the record is then
    # judged on its hash like any other.
    path.write_bytes(line(b"1e9"))
    assert verify_chain(path).faults == (ChainFault.RECORD_HASH_MISMATCH,)


def test_appending_after_a_torn_tail_is_refused_rather_than_glued(tmp_path: Path) -> None:
    """Gluing turns a recoverable crash into an unreadable line that reads forged."""
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    torn = b'{"kind":"SHUT'
    first.path.write_bytes(first.line + torn)

    resumed = DecisionLog(
        tmp_path / "decision_log", seq=first.seq, prev_hash=first.record_hash
    )
    with pytest.raises(DecisionLogTailError, match="no terminating newline"):
        resumed.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))
    # Nothing was written, so the file is still exactly what the crash left and
    # still reads as torn rather than as a forgery.
    assert first.path.read_bytes() == first.line + torn
    verification = verify_chain(first.path)
    assert verification.is_torn and not verification.is_forged


def test_recover_tail_removes_the_unfinished_bytes_and_preserves_them(
    tmp_path: Path,
) -> None:
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    torn = b'{"kind":"SHUT'
    first.path.write_bytes(first.line + torn)

    repair = recover_tail(tmp_path)
    assert repair.repaired
    assert repair.path == first.path
    assert repair.truncated_bytes == len(torn)
    assert repair.truncated_path is not None
    assert repair.truncated_path.read_bytes() == torn
    assert first.path.read_bytes() == first.line
    assert verify_chain(first.path).ok

    # And the campaign continues: reopening succeeds and the chain carries on.
    with DecisionLog.open(tmp_path) as log:
        second = log.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))
    assert second.prev_hash == first.record_hash
    assert verify_log(tmp_path / "decision_log").ok


def test_recover_tail_never_removes_a_complete_record_that_is_wrong(
    tmp_path: Path,
) -> None:
    """A forgery is evidence of a problem, not a mess for the writer to tidy."""
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    tampered = first.line.replace(b'"kind":"STARTUP"', b'"kind":"HALT"   ')
    first.path.write_bytes(tampered)

    repair = recover_tail(tmp_path)
    assert not repair.repaired and repair.truncated_bytes == 0
    assert first.path.read_bytes() == tampered
    assert verify_chain(first.path).is_forged
    assert not (first.path.parent / (first.path.name + ".truncated")).exists()


def test_recover_tail_on_an_empty_or_intact_log_changes_nothing(tmp_path: Path) -> None:
    empty = recover_tail(tmp_path)
    assert empty.path is None and not empty.repaired
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    intact = recover_tail(tmp_path)
    assert intact.path == first.path and not intact.repaired
    assert first.path.read_bytes() == first.line


def test_the_tail_error_carries_the_verification_and_not_only_its_wording(
    tmp_path: Path,
) -> None:
    """PR-10 branches on torn against forged; it must not parse a message to."""
    with DecisionLog.open(tmp_path) as log:
        first = log.append(payload())
    first.path.write_bytes(first.line + b"a torn half-record")
    with pytest.raises(DecisionLogTailError) as torn:
        DecisionLog.open(tmp_path)
    assert torn.value.verification is not None
    assert torn.value.verification.is_torn and not torn.value.verification.is_forged

    first.path.write_bytes(first.line.replace(b'"kind":"STARTUP"', b'"kind":"HALT"   '))
    with pytest.raises(DecisionLogTailError) as forged:
        DecisionLog.open(tmp_path)
    assert forged.value.verification is not None
    assert forged.value.verification.is_forged and not forged.value.verification.is_torn


# --- where the day files go --------------------------------------------------
def test_open_takes_the_state_directory_and_the_constructor_takes_the_log_directory(
    tmp_path: Path,
) -> None:
    """The two arguments are one level apart, and getting it wrong is silent."""
    opened = DecisionLog.open(tmp_path)
    assert opened.log_dir == tmp_path / "decision_log"
    first = opened.append(payload())
    opened.close()
    assert first.path == tmp_path / "decision_log" / "2026-11-23.ndjson"

    # Handed the resolved log directory, the constructor writes into it and does
    # not nest a second `decision_log` under it.
    direct = DecisionLog(tmp_path / "decision_log", seq=first.seq, prev_hash=first.record_hash)
    assert direct.log_dir == opened.log_dir
    second = direct.append(payload(kind="SHUTDOWN", runner_now_ns=NEXT_NS))
    direct.close()
    assert second.path == first.path
    assert verify_log(tmp_path / "decision_log", expected_last_hash=second.record_hash).ok


# --- what every record must carry, and what PR-10 stamps ---------------------
def test_config_hash_and_software_are_pr_10s_to_require_and_this_pins_that(
    tmp_path: Path,
) -> None:
    """Section 9.1 puts both on every record; PR-09 owns their form, not presence."""
    assert CORE_FIELDS == ("kind", "minute", "runner_now_ns")
    with DecisionLog.open(tmp_path) as log:
        assert log.append(payload()).seq == 1
        # Present, `config_hash` is still held to the one hash form.
        with pytest.raises(DecisionLogError, match="record.config_hash"):
            log.append(payload(runner_now_ns=NEXT_NS, config_hash="a" * 64))
        assert (
            log.append(
                payload(
                    runner_now_ns=NEXT_NS,
                    config_hash=A_HASH,
                    software={"revision": "abc", "dirty": False},
                )
            ).seq
            == 2
        )


# --- a minute is a minute ----------------------------------------------------
def test_iso_minute_refuses_an_instant_that_is_not_a_minute() -> None:
    """Refused rather than floored: the field is compared byte for byte."""
    assert iso_minute(1_795_392_060_000_000_000) == "2026-11-23T00:01:00+00:00"
    # Just above the boundary, and a whole recorded receipt: both refused.
    with pytest.raises(DecisionLogError, match="not a minute boundary"):
        iso_minute(1_795_392_060_000_000_001)
    with pytest.raises(DecisionLogError, match="not a minute boundary"):
        iso_minute(NOW_NS)
    # Just below the next boundary, and exactly at it.
    with pytest.raises(DecisionLogError, match="not a minute boundary"):
        iso_minute(1_795_392_119_999_999_999)
    assert iso_minute(1_795_392_120_000_000_000) == "2026-11-23T00:02:00+00:00"


def test_require_iso_minute_refuses_an_instant_inside_a_minute() -> None:
    assert require_iso_minute("2026-11-23T00:01:00+00:00") == "2026-11-23T00:01:00+00:00"
    for inside in ("2026-11-23T00:01:30+00:00", "2026-11-23T00:01:00.500000+00:00"):
        with pytest.raises(DecisionLogError, match="inside a minute"):
            require_iso_minute(inside)
    # The offset rule still applies first.
    with pytest.raises(DecisionLogError, match="ends in '\\+00:00'"):
        require_iso_minute("2026-11-23T00:01:00Z")


def test_the_record_minute_must_be_a_minute(tmp_path: Path) -> None:
    with DecisionLog.open(tmp_path) as log:
        with pytest.raises(DecisionLogError, match="inside a minute"):
            log.append(payload(kind="DECISION", minute="2026-11-23T00:01:30+00:00"))
        assert log.append(payload(kind="DECISION", minute="2026-11-23T00:01:00+00:00"))
