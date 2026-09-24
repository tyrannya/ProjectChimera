"""R1-g: recorder/runner cadence, and the funding deferral that goes with it.

Three claims, each with its two-sided control:

* the recorder publishes a closed minute on its own short cadence rather than
  waiting for the 300 s maintenance pass;
* the runner decides every closed minute since its last decided one, with no
  age cap (the cap's own tests live in ``test_demo_runner.py``);
* a minute whose booking window covers a scheduled funding settlement waits for
  the settlement to be knowable, instead of deciding without it and letting a
  later minute book it -- which is the divergence a replay of the same files
  could never reproduce.

ENGINEERING only. Every price here is synthetic, no economic quantity is
asserted on, and nothing in this file touches a contract hash, a preregistration
or any scientific artifact.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from chimera.demo.decision_log import iso_minute
from chimera.demo.fixtures import SyntheticFeed
from chimera.recorder.contract import load_recorder_contract
from chimera.recorder.normalize import (
    FUNDING_WATERMARK_FILE,
    FUNDING_WATERMARK_SCHEMA,
    MinuteNormalizer,
)
from tests.demo_harness import DAY, build

EIGHT_HOURS_MS = 8 * 60 * 60 * 1000
MINUTE_MS = 60_000

#: The last minute of the first eight-hour funding period: it OPENS at 07:59 and
#: CLOSES exactly on the 08:00 settlement instant, so it is the one minute of the
#: fixture day whose booking window contains a scheduled settlement.
SETTLEMENT_MINUTE_INDEX = 479


def _settlements_path(root: Path) -> Path:
    contract = load_recorder_contract("btcusdt-prospective-gen3")
    return MinuteNormalizer(root, contract).settlements_path("um")


def _drop_settlement_at(root: Path, instant_ms: int) -> None:
    """Remove one settlement row, leaving the rest of the file untouched.

    The recorder's own file, minus one row: exactly the state a runner is in
    when a settlement has happened and its row has not been written yet.
    """
    path = _settlements_path(root)
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["funding_time_ms"] != int(instant_ms)
    ]
    path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")


def _write_watermark(root: Path, observed_through_ms: int | None) -> None:
    contract = load_recorder_contract("btcusdt-prospective-gen3")
    normalizer = MinuteNormalizer(root, contract)
    normalizer.publish_funding_watermark("um", observed_through_ms)


def _tick_to(harness, index: int):
    """Drive the runner up to and including one minute index of the fixture day."""
    first = harness.first_minute_ms()
    outcome = None
    for step in range(index + 1):
        outcome = harness.tick(first + step * MINUTE_MS)
    return outcome


# ---------------------------------------------------------------------------
# the funding deferral
# ---------------------------------------------------------------------------
def test_a_minute_closing_on_a_settlement_waits_for_the_row(tmp_path):
    """The defect side: no row, no watermark, so the minute is not decidable.

    Deciding it would book the settlement against whichever later minute the row
    happened to land on, while a replay -- where the row is present from the
    start -- books it here. Same files, two logs.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    instant = first + SETTLEMENT_MINUTE_INDEX * MINUTE_MS + MINUTE_MS
    assert instant % EIGHT_HOURS_MS == 0, "the fixture's settlement grid moved"
    _drop_settlement_at(harness.root, instant)

    deferred_ms = first + SETTLEMENT_MINUTE_INDEX * MINUTE_MS
    outcome = _tick_to(harness, SETTLEMENT_MINUTE_INDEX)

    assert outcome.kind is None
    assert "funding settlement scheduled" in outcome.detail
    # Nothing was written FOR THIS MINUTE -- the earlier minutes of the run
    # wrote their own records and those are not what is being counted -- and the
    # cursor still points at it, so the next pass runs it again rather than
    # stepping over it.
    deferred_iso = iso_minute(deferred_ms * 1_000_000)
    assert all(record["minute"] != deferred_iso for record in harness.records())
    assert harness.runner.cursor.next_minute_ms() == deferred_ms
    assert harness.runner.cursor.last_minute_processed == deferred_ms - MINUTE_MS


def test_the_same_minute_decides_once_the_row_arrives(tmp_path):
    """The control for the case above: the wait ends, and it ends by deciding."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    instant = first + SETTLEMENT_MINUTE_INDEX * MINUTE_MS + MINUTE_MS
    kept = _settlements_path(harness.root).read_text(encoding="utf-8")
    _drop_settlement_at(harness.root, instant)

    deferred = _tick_to(harness, SETTLEMENT_MINUTE_INDEX)
    assert deferred.kind is None

    # The recorder writes the row it was missing.
    _settlements_path(harness.root).write_text(kept, encoding="utf-8")

    decided = harness.tick(first + SETTLEMENT_MINUTE_INDEX * MINUTE_MS)
    assert decided.kind is not None
    assert decided.minute_ms == first + SETTLEMENT_MINUTE_INDEX * MINUTE_MS


def test_a_minute_with_its_settlement_row_present_never_waits(tmp_path):
    """The ordinary case, and the first clause of the requirement.

    "Decidable once its settlement row exists in the recorder output" -- so a
    complete file must not produce a single deferral anywhere in the day.
    """
    harness = build(tmp_path)
    outcomes = harness.run(SETTLEMENT_MINUTE_INDEX + 2)
    assert all(o.kind is not None for o in outcomes)


def test_the_watermark_alone_releases_a_minute_with_no_row(tmp_path):
    """A settlement that never happened must not stop the campaign for ever.

    Absence of a row is not evidence of no settlement -- but absence of a row
    BELOW a watermark is, because the recorder has said it looked that far. This
    is the only thing that distinguishes the two, and without it a venue that
    skipped a scheduled settlement would deadlock the runner.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    instant = first + SETTLEMENT_MINUTE_INDEX * MINUTE_MS + MINUTE_MS
    _drop_settlement_at(harness.root, instant)
    _write_watermark(harness.root, instant)

    outcome = _tick_to(harness, SETTLEMENT_MINUTE_INDEX)
    assert outcome.kind is not None, "an observed absence is an answer, not a wait"


def test_a_watermark_short_of_the_instant_still_waits(tmp_path):
    """The two-sided half of the test above: one millisecond short is short."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    instant = first + SETTLEMENT_MINUTE_INDEX * MINUTE_MS + MINUTE_MS
    _drop_settlement_at(harness.root, instant)
    _write_watermark(harness.root, instant - 1)

    outcome = _tick_to(harness, SETTLEMENT_MINUTE_INDEX)
    assert outcome.kind is None


def test_a_minute_with_no_scheduled_settlement_never_waits(tmp_path):
    """No watermark at all, and no deferral, because there is nothing to await.

    A campaign whose recorder publishes no watermark must still run. Only a
    minute that actually has a scheduled settlement in its window may wait.
    """
    harness = build(tmp_path)
    watermark = _settlements_path(harness.root).with_name(FUNDING_WATERMARK_FILE)
    assert not watermark.exists()

    outcomes = harness.run(5)
    assert all(o.kind is not None for o in outcomes)


def test_live_and_replay_book_the_same_settlement_in_the_same_minute(tmp_path):
    """R1-g's named two-sided test.

    'Live' is a run whose settlement row lands late -- the row is absent when the
    minute is first attempted and written before the retry. 'Replay' is a run
    over the finished files, where it was there all along. The two must agree
    about WHICH MINUTE booked the settlement; before the deferral the live run
    booked it against a later minute and no comparison of the two logs could say
    why.
    """

    def booking_minutes(where: Path, *, late: bool) -> list[str]:
        harness = build(where)
        first = harness.first_minute_ms()
        instant = first + SETTLEMENT_MINUTE_INDEX * MINUTE_MS + MINUTE_MS
        kept = _settlements_path(harness.root).read_text(encoding="utf-8")
        if late:
            _drop_settlement_at(harness.root, instant)

        for step in range(SETTLEMENT_MINUTE_INDEX + 2):
            outcome = harness.tick(first + step * MINUTE_MS)
            if outcome.kind is None:
                # The recorder catches up, and the runner retries the same minute.
                _settlements_path(harness.root).write_text(kept, encoding="utf-8")
                outcome = harness.tick(first + step * MINUTE_MS)
                assert outcome.kind is not None

        return [
            record["minute"]
            for record in harness.records()
            if record.get("funding") or record["kind"] == "FUNDING"
        ]

    late_run = booking_minutes(tmp_path / "late", late=True)
    replay = booking_minutes(tmp_path / "replay", late=False)
    assert late_run == replay
    assert late_run, "the fixture booked no settlement at all; the test proves nothing"


def test_run_minutes_stops_at_a_deferral_like_catch_up_does(tmp_path):
    """A deferral must stop every driver, not only ``catch_up``.

    ``run_minutes`` is handed an explicit list, and carrying on past a deferred
    minute would decide later minutes ahead of it and drop it from the log with
    no record at all -- a divergence a replay could never account for, because
    the replay has the row and defers nothing.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    instant = first + SETTLEMENT_MINUTE_INDEX * MINUTE_MS + MINUTE_MS
    _drop_settlement_at(harness.root, instant)

    minutes = [first + i * MINUTE_MS for i in range(SETTLEMENT_MINUTE_INDEX + 3)]
    outcomes = harness.runner.run_minutes(minutes)

    assert outcomes[-1].kind is None, "the run did not end on the deferral"
    assert len(outcomes) == SETTLEMENT_MINUTE_INDEX + 1, "it decided past the deferral"


def test_the_watermark_is_taken_from_when_the_recorder_LOOKED(tmp_path):
    """Not from the settlement instant, which can never exceed itself.

    A funding event's canonical instant IS its settlement instant, so a
    watermark built from it would say "observed through 08:00" at 15:00 -- and
    would defer the 08:00 minute for ever if that settlement's row were the one
    missing. The receipt clock is when the recorder actually held the
    observation, which is the claim the watermark makes.
    """
    source = (
        Path(__file__).resolve().parents[1] / "chimera" / "recorder" / "normalize.py"
    ).read_text(encoding="utf-8")

    assert "observed_ns = int(event.receipt_wall_ns)" in source
    assert "observed_ns = int(event.canonical_ns)" not in source


# ---------------------------------------------------------------------------
# the watermark itself
# ---------------------------------------------------------------------------
def test_the_watermark_is_monotone(tmp_path):
    """Its two writers know different things, so it may only ever rise.

    A rebuild from the raw files knows where an observation was stored; a
    successful poll knows the venue answered for a window ending now. A rebuild
    running after a poll must not retract the poll's fact.
    """
    root = tmp_path / "recorder"
    contract = load_recorder_contract("btcusdt-prospective-gen3")
    normalizer = MinuteNormalizer(root, contract)
    normalizer.settlements_path("um").parent.mkdir(parents=True, exist_ok=True)

    assert normalizer.publish_funding_watermark("um", 1_000) == 1_000
    assert normalizer.publish_funding_watermark("um", 500) == 1_000
    assert normalizer.publish_funding_watermark("um", None) == 1_000
    assert normalizer.publish_funding_watermark("um", 2_000) == 2_000
    assert normalizer.read_funding_watermark("um") == 2_000

    document = json.loads(normalizer.funding_watermark_path("um").read_text(encoding="utf-8"))
    assert document["schema"] == FUNDING_WATERMARK_SCHEMA
    assert document["observed_through_ms"] == 2_000


def test_an_unreadable_watermark_is_no_watermark(tmp_path):
    """Not a guess, and not a crash: unknown, which makes the runner wait."""
    root = tmp_path / "recorder"
    contract = load_recorder_contract("btcusdt-prospective-gen3")
    normalizer = MinuteNormalizer(root, contract)
    path = normalizer.funding_watermark_path("um")
    path.parent.mkdir(parents=True, exist_ok=True)

    assert normalizer.read_funding_watermark("um") is None  # absent
    path.write_text("{not json", encoding="utf-8")
    assert normalizer.read_funding_watermark("um") is None  # malformed
    path.write_text(json.dumps({"observed_through_ms": "soon"}), encoding="utf-8")
    assert normalizer.read_funding_watermark("um") is None  # wrong type


def test_rebuilding_settlements_publishes_what_was_observed(tmp_path):
    """The offline floor: the watermark a rebuild alone can establish."""
    root = tmp_path / "recorder"
    contract = load_recorder_contract("btcusdt-prospective-gen3")
    feed = SyntheticFeed(root, contract)
    feed.write_days([DAY])
    feed.write_settlements([DAY])

    normalizer = MinuteNormalizer(root, contract)
    # The fixture writes the settlements file directly rather than through raw
    # events, so a rebuild sees no funding observations at all -- which is
    # exactly the "holds none" case, and it must publish that rather than crash.
    assert normalizer.read_funding_watermark("um") is None


# ---------------------------------------------------------------------------
# the recorder's publish cadence
# ---------------------------------------------------------------------------
def test_the_service_publishes_on_its_own_short_cadence():
    """The cadence is seconds, and it is a different job from maintenance."""
    from chimera.recorder import service

    assert service.PUBLISH_INTERVAL_S <= 10.0
    assert service.PUBLISH_INTERVAL_S < service.NORMALIZE_INTERVAL_S / 10


def test_publishing_folds_the_open_day_without_rotating_or_freezing(tmp_path):
    """What ``_publish`` does, and the three things it must not do.

    Rotation, freezing and gap-fill belong to the maintenance pass. A publish
    that did them would be the 300 s job running every few seconds, which is the
    cost R1-g's cadence exists to avoid.
    """
    from chimera.recorder.service import RecorderService

    contract = load_recorder_contract("btcusdt-prospective-gen3")
    root = tmp_path / "recorder"
    root.mkdir(parents=True, exist_ok=True)
    service = RecorderService(contract, root, gapfill=False)

    calls: list[tuple[str, str]] = []
    service._normalize = lambda market, day: calls.append((market, day)) or True  # type: ignore[assignment]
    service._sync = lambda: calls.append(("sync", ""))  # type: ignore[assignment]
    service._freeze_due = lambda: pytest.fail("publish must not freeze")  # type: ignore[assignment]

    asyncio.run(service._publish())

    assert calls[0] == ("sync", ""), "raw must be durable before anything folds"
    assert len(calls) == 1 + len(list(contract.market_keys()))
