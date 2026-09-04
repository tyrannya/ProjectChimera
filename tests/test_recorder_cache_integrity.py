"""The normalize cache cannot vouch for itself, so nothing reads it until it can.

The incremental normalizer exists because re-reading a busy day costs minutes,
and the price of not re-reading it is that a file on disk gets to say what the
day's minutes were. That file is engineering state — rebuildable, deletable, in
no contract and no manifest — but "rebuildable" is only half of "not evidence".
The other half is that a cache whose numbers were changed must never be the
thing a normalized day is rendered from.

That is what this file tests, and it tests it the only way that means anything:
by editing a cache in ways that leave its schema, its market, its day, its
contract hash and its cursor offsets all still looking right, and then asking
whether the day that comes back is the day the raw files say. Every assertion
compares against :meth:`MinuteNormalizer.build_day` — the full re-read — because
a test that compared the incremental path against itself would agree with a
corrupted cache just as readily as with a good one.

**The two-sided control.** A refusal test proves nothing unless the material it
refuses would otherwise have changed the answer.
:func:`test_a_trusted_mutation_would_have_changed_the_day` renders the mutated
states directly, bypassing the check, and shows each one producing a day the raw
files do not support. Only then is refusing them a result.

**Two ways to edit, and both are covered.** An editor who does not know about the
seal leaves it alone, and the seal catches them. An editor who does know
recomputes it, and then the only thing standing between a malformed document and
``RecorderService`` is the structural validation — so the shape cases below all
re-seal after mutating, or they would be testing the seal a second time instead
of the validation.
"""

from __future__ import annotations

import json

import pytest

from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.events import UM_BOOK_TICKER, UM_KLINE_1M, UM_MARK_PRICE
from chimera.recorder.incremental import (
    CACHE_DIGEST_DOMAIN,
    CACHE_DIGEST_FIELD,
    CACHE_SCHEMA,
    IncrementalNormalizer,
    NormalizeCacheError,
    cache_digest,
    state_from_document,
)
from chimera.recorder.normalize import MinuteNormalizer
from chimera.recorder.sink import RawSink
from tests.recorder_synthetic import DAY, book_event, kline_event, mark_event, minute_ms

CONTRACT = load_recorder_contract()

#: Four minutes, and every mutation reaches into the third of them: a rebuild
#: that quietly dropped a prefix or a tail would still look right if the material
#: only ever exercised its first minute.
MINUTES = 4
TARGET = 2


def material() -> dict[str, list]:
    """One short day of all three perpetual streams, with distinguishable values.

    Three marks a minute rather than one, so a mutated open, close, high, low or
    event count is a different day rather than the same day written twice; and a
    different bid every minute, so a moved book price cannot coincide with the
    one it replaced.
    """
    return {
        UM_KLINE_1M: [
            kline_event(minute_ms(index), close=f"{60_000 + index}.50")
            for index in range(MINUTES)
        ],
        UM_MARK_PRICE: [
            mark_event(minute_ms(index) + offset, mark=f"{60_050 + index * 3 + step}.00")
            for index in range(MINUTES)
            for step, offset in enumerate((0, 20_000, 40_000))
        ],
        UM_BOOK_TICKER: [
            book_event(1_000 + index, event_ms=minute_ms(index), bid=f"{59_999 + index}.90")
            for index in range(MINUTES)
        ],
    }


def write(root, events_by_stream) -> None:
    for stream, events in events_by_stream.items():
        with RawSink(root, stream, contract=CONTRACT) as sink:
            for event in events:
                sink.append(event)
            sink.sync()


def seeded(tmp_path):
    """A root with a folded day and a written cache, plus the oracle for it.

    Two roots: the oracle re-reads its own copy of the same raw files, so the
    comparison is between two independent trees rather than between one tree and
    a memory of itself.
    """
    cursor_root = tmp_path / "cursor"
    oracle_root = tmp_path / "oracle"
    for root in (cursor_root, oracle_root):
        root.mkdir(parents=True, exist_ok=True)
        write(root, material())
    oracle = MinuteNormalizer(oracle_root, CONTRACT).build_day("um", DAY)
    incremental = IncrementalNormalizer(cursor_root, CONTRACT)
    incremental.build_day("um", DAY)
    return incremental, oracle, cursor_root, oracle_root


def read_cache(incremental) -> dict:
    return json.loads(incremental.cache_path("um", DAY).read_text(encoding="utf-8"))


def write_cache(incremental, document, *, reseal: bool) -> None:
    """Put a document back, with or without recomputing the seal over it."""
    if reseal:
        document.pop(CACHE_DIGEST_FIELD, None)
        document[CACHE_DIGEST_FIELD] = cache_digest(document)
    incremental.cache_path("um", DAY).write_text(json.dumps(document), encoding="utf-8")


def clear_output(root) -> None:
    """Delete the derived day, so it has to be produced again to be compared."""
    normalizer = MinuteNormalizer(root, CONTRACT)
    normalizer.parquet_path("um", DAY).unlink(missing_ok=True)
    normalizer.meta_path("um", DAY).unlink(missing_ok=True)


def assert_rebuilt_to_the_oracle(incremental, oracle, cursor_root, oracle_root) -> None:
    """The cache was not trusted, the raw was re-read, and the day is the day."""
    clear_output(cursor_root)
    rendered = incremental.build_day("um", DAY)
    status = incremental.status[("um", DAY)]

    assert status.rebuilt is True, "the cache was trusted"
    assert status.resumed is False
    assert rendered.digest == oracle.digest, "the rebuilt day is not the day raw describes"
    assert rendered.rows == oracle.rows == MINUTES
    assert rendered.missing == oracle.missing
    assert rendered.conflicts == oracle.conflicts

    # The digest is a function of the minutes, and a doctored tally or resume
    # instant moves neither, so the metadata is compared too.
    left = json.loads(
        MinuteNormalizer(oracle_root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
    )
    right = json.loads(
        MinuteNormalizer(cursor_root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
    )
    assert right["streams"] == left["streams"], "a doctored tally survived into the metadata"
    assert right["digest"] == left["digest"]


# --- the mutations ------------------------------------------------------------
def minute_key(offset: int = 0) -> str:
    return str(minute_ms(TARGET + offset))


def mutate_book_price(document: dict) -> None:
    entry = document["books"][minute_key()]
    entry["values"] = [1.0, *entry["values"][1:]]


def mutate_book_update_id(document: dict) -> None:
    document["books"][minute_key()]["update_id"] = 999_999_999


def mutate_mark_close(document: dict) -> None:
    entry = document["marks"][minute_key()]
    entry["close"] = [1.0, *entry["close"][1:]]


def mutate_mark_open(document: dict) -> None:
    entry = document["marks"][minute_key()]
    entry["open"] = [2.0, *entry["open"][1:]]


def mutate_kline_value(document: dict) -> None:
    entry = document["klines"][minute_key()]
    entry["values"] = [*entry["values"][:3], 3.0, *entry["values"][4:]]


def mutate_kline_conflict(document: dict) -> None:
    document["klines"][minute_key()]["conflict"] = True


def mutate_tallies(document: dict) -> None:
    document["tallies"][UM_MARK_PRICE]["records"] = 10_000


def mutate_last_canonical(document: dict) -> None:
    document["last_canonical"][UM_MARK_PRICE] = 1


def mutate_time_basis(document: dict) -> None:
    # A label that is a *valid* one, so only the seal can tell it is wrong: the
    # perpetual's book is exchange-stamped and this claims it was receipt-stamped.
    document["books"][minute_key()]["time_basis"] = "RECEIPT"


#: Every edit that leaves a syntactically valid cache whose schema, market, day,
#: contract hash and cursor offsets are all still exactly right.
SEMANTIC = {
    "book price": mutate_book_price,
    "book update id": mutate_book_update_id,
    "mark close": mutate_mark_close,
    "mark open": mutate_mark_open,
    "kline value": mutate_kline_value,
    "kline conflict flag": mutate_kline_conflict,
    "tallies": mutate_tallies,
    "last canonical": mutate_last_canonical,
    "time basis": mutate_time_basis,
}


# --- A. the material can tell a trusted cache from a refused one ---------------
def read_state(document):
    """The state a loader gets from a document, re-sealed so the seal is not the test."""
    document = json.loads(json.dumps(document))
    document.pop(CACHE_DIGEST_FIELD, None)
    document[CACHE_DIGEST_FIELD] = cache_digest(document)
    return state_from_document(document, market="um", day=DAY, hash_=CONTRACT.contract_hash)


def test_a_trusted_mutation_would_have_changed_the_day(tmp_path):
    """Two-sided control: rendering these states *does* produce a different day.

    Without this, every refusal below could be refusing something harmless. Each
    state is built from the pristine document and rendered directly, around the
    check — which is exactly what the loader did before it had one.
    """
    incremental, _, _, _ = seeded(tmp_path)
    pristine = read_cache(incremental)
    baseline = incremental.render(read_state(pristine))

    #: All but one of them move a value the normalized day is made of.
    rendered_differently = sorted(set(SEMANTIC) - {"last canonical"})
    changed = []
    for name in rendered_differently:
        edited = json.loads(json.dumps(pristine))
        SEMANTIC[name](edited)
        if incremental.render(read_state(edited)) != baseline:
            changed.append(name)
    assert changed == rendered_differently, (
        "a mutation the loader would have rendered identically proves nothing when it is "
        f"refused; these changed nothing: {sorted(set(rendered_differently) - set(changed))}"
    )

    # The resume instant is not rendered into the day — it is what a restart
    # gap-fills from — so its control is that a trusted cache would hand the
    # recorder a different one.
    doctored = json.loads(json.dumps(pristine))
    mutate_last_canonical(doctored)
    assert read_state(doctored).last_canonical != read_state(pristine).last_canonical


# --- B. the seal: an edited cache is refused and the raw is re-read ------------
@pytest.mark.parametrize("name", sorted(SEMANTIC))
def test_a_semantically_edited_cache_is_refused_and_the_day_is_rebuilt(tmp_path, name):
    """Schema, market, day, contract hash and cursor all valid — and still refused."""
    incremental, oracle, cursor_root, oracle_root = seeded(tmp_path)
    document = read_cache(incremental)
    SEMANTIC[name](document)
    write_cache(incremental, document, reseal=False)

    # Everything the old loader checked still passes, which is the point.
    on_disk = read_cache(incremental)
    assert on_disk["cache_schema"] == CACHE_SCHEMA
    assert on_disk["contract_hash"] == CONTRACT.contract_hash
    assert on_disk["market"] == "um" and on_disk["day"] == DAY
    events_path = RawSink(cursor_root, UM_KLINE_1M, contract=CONTRACT).events_path(DAY)
    cursor = on_disk["cursors"][f"{UM_KLINE_1M}:main"]
    assert 0 <= cursor["offset"] <= events_path.stat().st_size

    assert_rebuilt_to_the_oracle(incremental, oracle, cursor_root, oracle_root)


def test_a_wrong_but_well_formed_seal_is_refused(tmp_path):
    incremental, oracle, cursor_root, oracle_root = seeded(tmp_path)
    document = read_cache(incremental)
    document[CACHE_DIGEST_FIELD] = "a" * 64
    write_cache(incremental, document, reseal=False)
    assert_rebuilt_to_the_oracle(incremental, oracle, cursor_root, oracle_root)


def test_a_missing_seal_is_refused(tmp_path):
    """Absent is not a pass. A cache with no seal is a cache nothing vouches for."""
    incremental, oracle, cursor_root, oracle_root = seeded(tmp_path)
    document = read_cache(incremental)
    del document[CACHE_DIGEST_FIELD]
    write_cache(incremental, document, reseal=False)
    assert_rebuilt_to_the_oracle(incremental, oracle, cursor_root, oracle_root)


@pytest.mark.parametrize(
    "seal",
    ["", "z" * 64, "A" * 64, "abc", "0" * 63, "0" * 65, 12345, None, ["0" * 64]],
    ids=["empty", "not hex", "uppercase", "short", "63", "65", "number", "null", "list"],
)
def test_a_malformed_seal_is_refused(tmp_path, seal):
    incremental, oracle, cursor_root, oracle_root = seeded(tmp_path)
    document = read_cache(incremental)
    document[CACHE_DIGEST_FIELD] = seal
    write_cache(incremental, document, reseal=False)
    assert_rebuilt_to_the_oracle(incremental, oracle, cursor_root, oracle_root)


def test_the_seal_covers_every_part_of_the_document(tmp_path):
    """Every field named in the design, one at a time, moves the digest."""
    incremental, _, _, _ = seeded(tmp_path)
    document = read_cache(incremental)
    sealed = cache_digest(document)
    assert document[CACHE_DIGEST_FIELD] == sealed

    edits = {
        "cache_schema": lambda d: d.__setitem__("cache_schema", "other/1"),
        "market": lambda d: d.__setitem__("market", "spot"),
        "day": lambda d: d.__setitem__("day", "2026-01-01"),
        "contract_hash": lambda d: d.__setitem__("contract_hash", "0" * 64),
        "cursors": lambda d: d["cursors"][f"{UM_KLINE_1M}:main"].__setitem__("offset", 0),
        "tallies": mutate_tallies,
        "last_canonical": mutate_last_canonical,
        "klines": mutate_kline_value,
        "marks": mutate_mark_close,
        "books": mutate_book_price,
        "note": lambda d: d.__setitem__("note", "harmless"),
    }
    for field, edit in edits.items():
        candidate = read_cache(incremental)
        edit(candidate)
        assert cache_digest(candidate) != sealed, f"the seal does not cover {field!r}"


def test_the_seal_is_domain_separated(tmp_path):
    """Not a bare SHA-256 of the payload, so it cannot collide with another one."""
    import hashlib

    from chimera.recorder.events import canonical_json

    incremental, _, _, _ = seeded(tmp_path)
    document = read_cache(incremental)
    payload = {k: v for k, v in document.items() if k != CACHE_DIGEST_FIELD}
    bare = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    assert cache_digest(document) != bare
    assert (
        cache_digest(document)
        == hashlib.sha256(
            CACHE_DIGEST_DOMAIN + canonical_json(payload).encode("utf-8")
        ).hexdigest()
    )


# --- C. the structural validation, reached by re-sealing after the edit --------
def malformed_key(document: dict) -> None:
    document["books"][minute_key()]["key"] = [1, 2]


def malformed_book_tuple(document: dict) -> None:
    document["books"][minute_key()]["values"] = [1.0, 2.0, 3.0]


def malformed_mark_tuple(document: dict) -> None:
    document["marks"][minute_key()]["close"] = [1.0, 2.0]


def malformed_kline_tuple(document: dict) -> None:
    document["klines"][minute_key()]["values"] = [1.0, 2.0, 3.0, 4.0]


def malformed_published_tuple(document: dict) -> None:
    document["klines"][minute_key()]["published"] = ["1", "2", "3"]


def negative_cursor_offset(document: dict) -> None:
    document["cursors"][f"{UM_KLINE_1M}:main"]["offset"] = -5


def negative_cursor_lines(document: dict) -> None:
    document["cursors"][f"{UM_KLINE_1M}:main"]["lines"] = -3


def unexpected_cursor_variant(document: dict) -> None:
    document["cursors"][f"{UM_KLINE_1M}:main"]["variant"] = "../../etc/passwd"


def invalid_time_basis(document: dict) -> None:
    document["books"][minute_key()]["time_basis"] = "sundial"


def invalid_source(document: dict) -> None:
    document["klines"][minute_key()]["source"] = "TELEPATHY"


def non_integer_minute(document: dict) -> None:
    document["books"]["not-a-minute"] = document["books"][minute_key()]


def negative_minute(document: dict) -> None:
    document["books"]["-60000"] = document["books"][minute_key()]


def negative_tally(document: dict) -> None:
    document["tallies"][UM_KLINE_1M]["records"] = -1


def negative_last_canonical(document: dict) -> None:
    document["last_canonical"][UM_KLINE_1M] = -1


def zero_mark_events(document: dict) -> None:
    document["marks"][minute_key()]["events"] = 0


def fractional_trade_count(document: dict) -> None:
    entry = document["klines"][minute_key()]
    entry["values"] = [*entry["values"][:5], 1.5, *entry["values"][6:]]


def string_where_a_number_belongs(document: dict) -> None:
    entry = document["books"][minute_key()]
    entry["values"] = ["1.0", *entry["values"][1:]]


def boolean_where_a_count_belongs(document: dict) -> None:
    document["marks"][minute_key()]["events"] = True


def conflict_is_not_a_boolean(document: dict) -> None:
    document["klines"][minute_key()]["conflict"] = 1


def cursors_is_not_a_mapping(document: dict) -> None:
    document["cursors"] = [1, 2, 3]


def an_aggregate_is_not_a_mapping(document: dict) -> None:
    document["books"][minute_key()] = ["bid", "ask"]


def a_required_mapping_is_missing(document: dict) -> None:
    del document["klines"]


def last_canonical_is_missing(document: dict) -> None:
    del document["last_canonical"]


def an_unknown_field_appears(document: dict) -> None:
    document["adjusted_by"] = "somebody"


def an_unknown_entry_field_appears(document: dict) -> None:
    document["cursors"][f"{UM_KLINE_1M}:main"]["skip_to"] = 10


def a_required_entry_field_is_missing(document: dict) -> None:
    del document["books"][minute_key()]["update_id"]


STRUCTURAL = {
    "ordering key of the wrong length": malformed_key,
    "book aggregate of the wrong arity": malformed_book_tuple,
    "mark aggregate of the wrong arity": malformed_mark_tuple,
    "kline aggregate of the wrong arity": malformed_kline_tuple,
    "published tuple of the wrong arity": malformed_published_tuple,
    "negative cursor offset": negative_cursor_offset,
    "negative cursor line count": negative_cursor_lines,
    "unexpected cursor variant": unexpected_cursor_variant,
    "invalid time basis": invalid_time_basis,
    "invalid event source": invalid_source,
    "minute key that is not an integer": non_integer_minute,
    "minute key before the epoch": negative_minute,
    "negative tally": negative_tally,
    "negative last canonical": negative_last_canonical,
    "mark minute with no events": zero_mark_events,
    "fractional trade count": fractional_trade_count,
    "string where a price belongs": string_where_a_number_belongs,
    "boolean where a count belongs": boolean_where_a_count_belongs,
    "conflict flag that is not a boolean": conflict_is_not_a_boolean,
    "cursors that are not a mapping": cursors_is_not_a_mapping,
    "aggregate that is not a mapping": an_aggregate_is_not_a_mapping,
    "required mapping missing": a_required_mapping_is_missing,
    "last canonical missing": last_canonical_is_missing,
    "unknown top-level field": an_unknown_field_appears,
    "unknown field inside an entry": an_unknown_entry_field_appears,
    "required field inside an entry missing": a_required_entry_field_is_missing,
}


@pytest.mark.parametrize("name", sorted(STRUCTURAL))
def test_a_malformed_but_correctly_sealed_cache_is_refused(tmp_path, name):
    """Re-sealed after the edit, so the seal passes and only the shapes can refuse it.

    No ``IndexError``, ``KeyError``, ``TypeError``, ``ValueError`` or ``OSError``
    may leave the module: ``build_day`` catches ``NormalizeCacheError`` and
    nothing else, so anything that escapes here would escape into
    ``RecorderService`` in production.
    """
    incremental, oracle, cursor_root, oracle_root = seeded(tmp_path)
    document = read_cache(incremental)
    STRUCTURAL[name](document)
    write_cache(incremental, document, reseal=True)

    on_disk = read_cache(incremental)
    assert on_disk.get(CACHE_DIGEST_FIELD) == cache_digest(on_disk), "the seal must pass"

    assert_rebuilt_to_the_oracle(incremental, oracle, cursor_root, oracle_root)


@pytest.mark.parametrize("name", sorted(STRUCTURAL))
def test_the_reader_refuses_a_malformed_cache_with_its_own_error(tmp_path, name):
    """And it is one error type, not whatever Python happened to raise first."""
    incremental, _, _, _ = seeded(tmp_path)
    document = read_cache(incremental)
    STRUCTURAL[name](document)
    document.pop(CACHE_DIGEST_FIELD, None)
    document[CACHE_DIGEST_FIELD] = cache_digest(document)
    with pytest.raises(NormalizeCacheError):
        state_from_document(document, market="um", day=DAY, hash_=CONTRACT.contract_hash)


def test_a_cache_that_is_not_an_object_at_all_is_refused(tmp_path):
    incremental, oracle, cursor_root, oracle_root = seeded(tmp_path)
    for body in ("[]", '"a string"', "null", "17"):
        incremental.cache_path("um", DAY).write_text(body, encoding="utf-8")
        clear_output(cursor_root)
        rendered = incremental.build_day("um", DAY)
        assert incremental.status[("um", DAY)].rebuilt is True, body
        assert rendered.digest == oracle.digest, body


# --- D. the schema bump: a version 1 cache is stale, not migrated -------------
def test_a_cache_from_the_unsealed_schema_is_not_migrated(tmp_path):
    """Version 1 had no seal, so its aggregates are exactly what must not be read.

    Reading them "because the rest of the document looks fine" is the defect the
    bump exists to close, so a ``/1`` file is stale engineering state and its day
    is folded again from the raw.
    """
    incremental, oracle, cursor_root, oracle_root = seeded(tmp_path)
    document = read_cache(incremental)
    mutate_book_price(document)
    del document[CACHE_DIGEST_FIELD]
    document["cache_schema"] = "chimera.recorder-normalize-cache/1"
    incremental.cache_path("um", DAY).write_text(json.dumps(document), encoding="utf-8")

    assert_rebuilt_to_the_oracle(incremental, oracle, cursor_root, oracle_root)
    assert CACHE_SCHEMA == "chimera.recorder-normalize-cache/2"


def test_the_cache_this_build_writes_carries_a_seal_over_its_own_contents(tmp_path):
    incremental, _, _, _ = seeded(tmp_path)
    document = read_cache(incremental)
    assert document["cache_schema"] == CACHE_SCHEMA
    seal = document[CACHE_DIGEST_FIELD]
    assert isinstance(seal, str) and len(seal) == 64
    assert seal == cache_digest(document)
    # And it reads back as the state it was written from.
    state = state_from_document(document, market="um", day=DAY, hash_=CONTRACT.contract_hash)
    assert state.records == sum(len(events) for events in material().values())


# --- E. the seal stays engineering-only ---------------------------------------
def test_the_seal_is_in_nothing_that_identifies_a_recording(tmp_path):
    """Not in the day's metadata, not in its digest, not in the contract hash."""
    incremental, oracle, cursor_root, _ = seeded(tmp_path)
    seal = read_cache(incremental)[CACHE_DIGEST_FIELD]

    document = json.loads(
        MinuteNormalizer(cursor_root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
    )
    text = json.dumps(document)
    assert CACHE_DIGEST_FIELD not in text
    assert seal not in text
    assert "cache" not in text

    # The contract hash is a property of the contract and moves for nothing here.
    assert CONTRACT.contract_hash == load_recorder_contract().contract_hash
    assert seal != CONTRACT.contract_hash
    assert seal != oracle.digest

    # And the day is still the day when the cache is deleted outright.
    incremental.drop("um", DAY)
    clear_output(cursor_root)
    assert IncrementalNormalizer(cursor_root, CONTRACT).build_day("um", DAY).digest == (
        oracle.digest
    )


# --- F. the durability order --------------------------------------------------
class Crash(RuntimeError):
    """An injected failure, standing in for the process dying at that instant."""


def test_a_crash_after_durable_raw_and_before_the_cache_replays_the_tail_once(
    tmp_path, monkeypatch
):
    """Window one. The cursor never claimed the tail, so the tail is folded once.

    The counts are what catch a double fold: a replayed event inflates a stream's
    record tally and a minute's mark count while leaving the digest alone,
    because an extremum is idempotent and a count is not.
    """
    incremental, oracle, cursor_root, oracle_root = seeded(tmp_path)
    before = read_cache(incremental)

    # More raw arrives and is fsynced. The process dies before the checkpoint.
    with RawSink(cursor_root, UM_MARK_PRICE, contract=CONTRACT) as sink:
        for offset in (10_000, 30_000, 50_000):
            sink.append(mark_event(minute_ms(TARGET) + offset, mark="60123.00"))
        sink.sync()
    with RawSink(oracle_root, UM_MARK_PRICE, contract=CONTRACT) as sink:
        for offset in (10_000, 30_000, 50_000):
            sink.append(mark_event(minute_ms(TARGET) + offset, mark="60123.00"))
        sink.sync()
    fuller = MinuteNormalizer(oracle_root, CONTRACT)
    fuller.parquet_path("um", DAY).unlink()
    fuller.meta_path("um", DAY).unlink()
    oracle = fuller.build_day("um", DAY)

    monkeypatch.setattr(
        IncrementalNormalizer, "save", lambda self, state: (_ for _ in ()).throw(Crash())
    )
    clear_output(cursor_root)
    with pytest.raises(Crash):
        incremental.build_day("um", DAY)
    monkeypatch.undo()

    assert read_cache(incremental) == before, "the checkpoint moved before the crash"
    assert not MinuteNormalizer(cursor_root, CONTRACT).parquet_path("um", DAY).exists()

    restarted = IncrementalNormalizer(cursor_root, CONTRACT)
    rendered = restarted.build_day("um", DAY)
    assert restarted.status[("um", DAY)].resumed is True
    assert restarted.status[("um", DAY)].replayed_records == 3, "the tail was not read once"
    assert rendered.digest == oracle.digest
    left = json.loads(fuller.meta_path("um", DAY).read_text("utf-8"))
    right = json.loads(
        MinuteNormalizer(cursor_root, CONTRACT).meta_path("um", DAY).read_text("utf-8")
    )
    assert right["streams"] == left["streams"], "a record was folded twice, or not at all"


def test_a_crash_after_the_cache_and_before_the_output_rereads_no_raw(tmp_path, monkeypatch):
    """Window two, tested at the boundary rather than by deleting the day after.

    ``write_day`` is made to fail at the instant the checkpoint has just been
    written, which is the only way to observe that the checkpoint really is
    durable *first*. What must follow is a render with no raw re-read at all.
    """
    incremental, oracle, cursor_root, oracle_root = seeded(tmp_path)
    with RawSink(cursor_root, UM_BOOK_TICKER, contract=CONTRACT) as sink:
        sink.append(book_event(9_000, event_ms=minute_ms(TARGET) + 30_000, bid="60111.10"))
        sink.sync()
    with RawSink(oracle_root, UM_BOOK_TICKER, contract=CONTRACT) as sink:
        sink.append(book_event(9_000, event_ms=minute_ms(TARGET) + 30_000, bid="60111.10"))
        sink.sync()
    fuller = MinuteNormalizer(oracle_root, CONTRACT)
    fuller.parquet_path("um", DAY).unlink()
    fuller.meta_path("um", DAY).unlink()
    oracle = fuller.build_day("um", DAY)

    monkeypatch.setattr(
        MinuteNormalizer,
        "write_day",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(Crash()),
    )
    clear_output(cursor_root)
    with pytest.raises(Crash):
        incremental.build_day("um", DAY)
    monkeypatch.undo()

    # The checkpoint is on disk, it verifies, and the derived output is not.
    document = read_cache(incremental)
    assert document[CACHE_DIGEST_FIELD] == cache_digest(document)
    state = state_from_document(document, market="um", day=DAY, hash_=CONTRACT.contract_hash)
    assert state.cursor(f"{UM_BOOK_TICKER}:main").lines == MINUTES + 1
    assert not MinuteNormalizer(cursor_root, CONTRACT).parquet_path("um", DAY).exists()

    restarted = IncrementalNormalizer(cursor_root, CONTRACT)
    rendered = restarted.build_day("um", DAY)
    status = restarted.status[("um", DAY)]
    assert status.resumed is True and status.rebuilt is False
    assert status.replayed_records == 0, "the prefix was read again"
    assert status.replayed_bytes == 0
    assert rendered.digest == oracle.digest
    assert MinuteNormalizer(cursor_root, CONTRACT).parquet_path("um", DAY).exists()


def test_an_ordinary_build_writes_raw_then_cache_then_the_normalized_day(tmp_path):
    """Window three: the order itself, observed rather than asserted from prose."""
    cursor_root = tmp_path / "cursor"
    cursor_root.mkdir(parents=True)
    write(cursor_root, material())
    incremental = IncrementalNormalizer(cursor_root, CONTRACT)
    normalizer = MinuteNormalizer(cursor_root, CONTRACT)
    raw = RawSink(cursor_root, UM_KLINE_1M, contract=CONTRACT).events_path(DAY)
    parquet = normalizer.parquet_path("um", DAY)
    cache = incremental.cache_path("um", DAY)

    order: list[str] = []
    real_save = IncrementalNormalizer.save
    real_write = MinuteNormalizer.write_day

    def watched_save(self, state):
        # The raw this checkpoint describes is already durable, and the derived
        # output it will be rendered into does not exist yet.
        assert raw.exists() and raw.stat().st_size > 0
        assert not parquet.exists()
        order.append("cache")
        return real_save(self, state)

    def watched_write(self, *args, **kwargs):
        # And by now the checkpoint is on disk and verifies.
        assert cache.exists()
        document = json.loads(cache.read_text(encoding="utf-8"))
        assert document[CACHE_DIGEST_FIELD] == cache_digest(document)
        order.append("normalized")
        return real_write(self, *args, **kwargs)

    IncrementalNormalizer.save = watched_save
    MinuteNormalizer.write_day = watched_write
    try:
        incremental.build_day("um", DAY)
    finally:
        IncrementalNormalizer.save = real_save
        MinuteNormalizer.write_day = real_write

    assert order == ["cache", "normalized"]
    assert parquet.exists()


def test_a_checkpoint_that_will_not_read_back_is_removed_and_the_day_still_renders(
    tmp_path, monkeypatch
):
    """The cache is a memo. Failing to write one costs a fold, never a day.

    A torn or unverifiable checkpoint is found by ``save`` itself, while the raw
    behind it can still simply be folded again — and it is removed rather than
    left for the next start to trip over.
    """
    incremental, oracle, cursor_root, _ = seeded(tmp_path)
    incremental.drop("um", DAY)
    clear_output(cursor_root)

    from chimera.recorder import incremental as module

    real_write_json = module.write_json_atomic

    def torn_write(path, payload):
        """Lands a document whose seal does not match what was written."""
        real_write_json(path, {**dict(payload), CACHE_DIGEST_FIELD: "b" * 64})

    monkeypatch.setattr(module, "write_json_atomic", torn_write)
    rendered = incremental.build_day("um", DAY)
    monkeypatch.undo()

    assert rendered.digest == oracle.digest, "the day was not rendered"
    assert not incremental.cache_path("um", DAY).exists(), "a bad checkpoint was left behind"
    reason = incremental.status[("um", DAY)].reason
    assert "cache not saved" in reason

    # And the next pass simply folds the day again.
    clear_output(cursor_root)
    again = IncrementalNormalizer(cursor_root, CONTRACT)
    assert again.build_day("um", DAY).digest == oracle.digest
    assert again.status[("um", DAY)].replayed_records == sum(
        len(events) for events in material().values()
    )


# --- G. what the recorder reads out of the cache on a restart -----------------
def test_a_doctored_resume_instant_is_not_handed_to_recovery(tmp_path):
    """``peek_last_canonical`` reads the cache, so it refuses one too.

    A resume instant nobody folded would send a gap-fill to the wrong place, so
    an unusable cache must produce no answer rather than a confident wrong one.
    """
    incremental, _, cursor_root, _ = seeded(tmp_path)
    honest = incremental.peek_last_canonical("um", DAY)
    assert honest[UM_MARK_PRICE] > 0
    pristine = read_cache(incremental)

    # Edited and left unsealed: the seal refuses it.
    doctored = json.loads(json.dumps(pristine))
    mutate_last_canonical(doctored)
    write_cache(incremental, doctored, reseal=False)
    assert incremental.peek_last_canonical("um", DAY) == {}, "a doctored instant was believed"

    # Edited and re-sealed, from the pristine document: the shape refuses it.
    malformed = json.loads(json.dumps(pristine))
    negative_last_canonical(malformed)
    write_cache(incremental, malformed, reseal=True)
    assert incremental.peek_last_canonical("um", DAY) == {}

    # And an untouched cache is still read, so the refusals above are not vacuous.
    write_cache(incremental, json.loads(json.dumps(pristine)), reseal=False)
    assert incremental.peek_last_canonical("um", DAY) == honest
