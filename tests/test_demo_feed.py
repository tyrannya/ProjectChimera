"""`FeedCursor`, `MarketState` and the per-minute digest.

Synthetic files only; `chimera.demo.fixtures` invents every number.
"""

from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from chimera.demo.feed import FeedCursor, MarketState, plain_json
from chimera.demo.fixtures import MinuteShape, SyntheticFeed
from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.normalize import digest
from tests.demo_harness import DAY, NEXT_DAY


@pytest.fixture
def contract():
    return load_recorder_contract("btcusdt-prospective-gen3")


@pytest.fixture
def written(tmp_path, contract):
    feed = SyntheticFeed(tmp_path, contract)
    feed.write_days([DAY])
    feed.write_settlements([DAY])
    return feed


def canonical(state: MarketState) -> str:
    return json.dumps(
        state.canonical(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


# --- reading ---------------------------------------------------------------
def test_the_first_minute_is_where_the_evidence_starts(tmp_path, contract, written):
    cursor = FeedCursor(tmp_path, contract)
    first = cursor.next_minute_ms()
    assert first is not None
    assert cursor.day_of(first) == DAY


def test_a_root_with_no_days_offers_no_minute(tmp_path, contract):
    """A campaign does not invent a start instant."""
    assert FeedCursor(tmp_path, contract).next_minute_ms() is None


def test_a_complete_minute_carries_both_legs_and_a_basis(tmp_path, contract, written):
    cursor = FeedCursor(tmp_path, contract)
    state = cursor.state_for(cursor.next_minute_ms())
    assert state.complete
    assert state.missing == ()
    assert state.spot_close is not None and state.perp_close is not None
    assert state.basis == state.perp_close - state.spot_close
    assert state.book_perp["bid"] < state.book_perp["ask"]


@pytest.mark.parametrize(
    "shape,expected",
    [
        (MinuteShape(present=False), "spot_minute"),
        (MinuteShape(book=False), "spot_book"),
    ],
)
def test_a_minute_the_feed_could_not_supply_is_incomplete(tmp_path, contract, shape, expected):
    feed = SyntheticFeed(tmp_path, contract)
    feed.write_days([DAY], shapes={f"spot:{DAY}": {3: shape}})
    cursor = FeedCursor(tmp_path, contract)
    state = cursor.state_for(cursor.next_minute_ms() + 3 * 60_000)
    assert not state.complete
    assert expected in state.missing


def test_a_perpetual_without_a_mark_is_incomplete(tmp_path, contract):
    feed = SyntheticFeed(tmp_path, contract)
    feed.write_days([DAY], shapes={f"um:{DAY}": {2: MinuteShape(mark=False)}})
    cursor = FeedCursor(tmp_path, contract)
    state = cursor.state_for(cursor.next_minute_ms() + 2 * 60_000)
    assert "um_mark" in state.missing


# --- the digest ------------------------------------------------------------
def test_the_minute_digest_is_the_recorders_own_function(tmp_path, contract, written):
    """Not a convention invented for the demo: the frozen function, one row wide."""
    cursor = FeedCursor(tmp_path, contract)
    minute = cursor.next_minute_ms()
    record = cursor.record("um", minute)
    frame = written.frame("um", DAY).iloc[0:1].reset_index(drop=True)
    assert record.row_digest == digest(frame, market="um")


def test_the_digest_is_bare_hex_because_the_log_exempts_digest_fields(
    tmp_path, contract, written
):
    cursor = FeedCursor(tmp_path, contract)
    value = cursor.state_for(cursor.next_minute_ms()).perp_digest
    assert len(value) == 64
    assert not value.startswith("sha256:")
    assert set(value) <= set("0123456789abcdef")


def test_two_different_minutes_have_two_different_digests(tmp_path, contract, written):
    cursor = FeedCursor(tmp_path, contract)
    first = cursor.next_minute_ms()
    assert cursor.state_for(first).perp_digest != cursor.state_for(first + 60_000).perp_digest


def test_the_same_minute_read_twice_has_the_same_digest(tmp_path, contract, written):
    """The negative control for the test above: identity, not just difference."""
    first = FeedCursor(tmp_path, contract).next_minute_ms()
    a = FeedCursor(tmp_path, contract).state_for(first)
    b = FeedCursor(tmp_path, contract).state_for(first)
    assert a.perp_digest == b.perp_digest
    assert a.spot_digest == b.spot_digest


# --- determinism -----------------------------------------------------------
def test_the_canonical_state_is_byte_identical_across_readers(tmp_path, contract, written):
    """What section 10's parity comparison rests on."""
    first = FeedCursor(tmp_path, contract).next_minute_ms()
    assert canonical(FeedCursor(tmp_path, contract).state_for(first)) == canonical(
        FeedCursor(tmp_path, contract).state_for(first)
    )


def test_the_canonical_state_holds_only_json_types(tmp_path, contract, written):
    """A parquet frame yields numpy scalars, which the decision log refuses."""
    cursor = FeedCursor(tmp_path, contract)
    payload = cursor.state_for(cursor.next_minute_ms()).canonical()
    json.dumps(payload, allow_nan=False)  # raises if a numpy scalar leaked

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert value is None or isinstance(value, (str, int, float, bool)), type(value)

    walk(payload)


def test_plain_json_converts_a_numpy_scalar_and_leaves_plain_values_alone():
    numpy_like = __import__("numpy")
    assert plain_json(numpy_like.int64(7)) == 7
    assert isinstance(plain_json(numpy_like.int64(7)), int)
    assert plain_json(D("1.50")) == "1.50"
    assert plain_json({"b": 1, "a": 2}) == {"a": 2, "b": 1}
    assert plain_json(None) is None
    assert plain_json("x") == "x"


# --- the cursor ------------------------------------------------------------
def test_the_cursor_is_monotone(tmp_path, contract, written):
    cursor = FeedCursor(tmp_path, contract)
    first = cursor.next_minute_ms()
    cursor.mark_processed(first + 5 * 60_000)
    cursor.mark_processed(first)  # an earlier minute must not move it back
    assert cursor.last_minute_processed == first + 5 * 60_000


def test_the_cursor_resumes_from_persisted_state(tmp_path, contract, written):
    first = FeedCursor(tmp_path, contract).next_minute_ms()
    resumed = FeedCursor(tmp_path, contract, {"last_minute_processed": first})
    assert resumed.next_minute_ms() == first + 60_000


def test_the_cursor_does_not_run_past_now(tmp_path, contract, written):
    cursor = FeedCursor(tmp_path, contract)
    first = cursor.next_minute_ms()
    assert cursor.next_minute_ms(now_ms=first - 1) is None
    assert cursor.next_minute_ms(now_ms=first) == first


def test_the_cursor_crosses_a_day_boundary(tmp_path, contract):
    feed = SyntheticFeed(tmp_path, contract)
    feed.write_days([DAY, NEXT_DAY])
    cursor = FeedCursor(tmp_path, contract)
    last_of_day = cursor.next_minute_ms() + 1439 * 60_000
    cursor.mark_processed(last_of_day)
    following = cursor.next_minute_ms()
    assert cursor.day_of(following) == NEXT_DAY
    assert cursor.state_for(following).complete


# --- funding ---------------------------------------------------------------
def test_the_last_settlement_at_or_before_a_minute_is_found(tmp_path, contract, written):
    cursor = FeedCursor(tmp_path, contract)
    first = cursor.next_minute_ms()
    assert cursor.last_settlement_at_or_before(first) is not None
    assert cursor.last_settlement_at_or_before(first - 1) is None


def test_a_state_carries_the_next_funding_time(tmp_path, contract, written):
    cursor = FeedCursor(tmp_path, contract)
    state = cursor.state_for(cursor.next_minute_ms())
    assert state.next_funding_time_ms is not None
    assert state.next_funding_time_ms > cursor.next_minute_ms()


# --- refusal ---------------------------------------------------------------
def test_a_day_whose_columns_are_not_the_recorders_is_refused(tmp_path, contract, written):
    import pandas as pd

    from chimera.demo.feed import FeedError
    from chimera.recorder.normalize import MinuteNormalizer

    minute = FeedCursor(tmp_path, contract).next_minute_ms()
    path = MinuteNormalizer(tmp_path, contract).parquet_path("um", DAY)
    pd.DataFrame({"nonsense": [1]}).to_parquet(path, index=False)
    with pytest.raises(FeedError, match="expected"):
        # A fresh cursor, so the day is read from disk rather than from cache.
        FeedCursor(tmp_path, contract).record("um", minute)
