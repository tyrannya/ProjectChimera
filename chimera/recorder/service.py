"""The recorder as a running process: streams, pollers, sink, normalizer, health.

Everything below is orchestration. Not one byte of market data is parsed here,
no file format is defined here, and no minute is decided here: this module owns
*when* things happen and in *what order*, and delegates every question about
what a record means to the layers that already answer it —
:mod:`chimera.recorder.events` for parsing, :mod:`chimera.recorder.sink` for
append-only storage, :mod:`chimera.recorder.normalize` for the minute grid,
:mod:`chimera.recorder.streams` and :mod:`chimera.recorder.rest` for the two
ways an observation arrives.

**Startup is a recovery, every time.** A process that has just started and one
that has just crashed are indistinguishable from the disk's point of view, so
there is one path and it always runs (section 4.3):

1. validate the tail of every open raw file, preserving anything torn;
2. read back the last canonical instant per stream;
3. REST gap-fill the closed klines and the funding settlements since then, and
   read the current mark and index state once;
4. re-normalize the current UTC day from raw;
5. connect the streams.

**Nothing is repaired, only recorded.** Gap-fill writes REST rows into the same
append-only raw files, through the same parsers, labelled
:attr:`EventSource.REST_GAPFILL`. A minute that neither the websocket nor the
REST endpoint produced has no row and stays that way. There is no interpolation
here, no carry-forward, and no cross-stream substitution — and when both sources
produce the same minute, both observations survive on disk, because the sink's
key is a function of the payload and disagreement is evidence rather than noise.

**What this module deliberately does not do.** It does not reconcile against the
venue's archives, does not compute coverage, does not write ``GATE.json``, and
does not freeze a normalized day: a frozen day is one the reconciliation has
finished with, and section 4.2 gives the reconciliation the right to re-normalize
a day that late events changed. Those are PR-06's, and their absence here is the
boundary rather than an omission.

**Shutdown is complete or it is a bug.** Every task this service creates is
owned by it, cancelled by it and awaited by it. A task that fails does not leave
the recorder reporting itself up: the failure is recorded, the stop event is set,
the remaining tasks are wound down and :meth:`RecorderService.run` re-raises.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from chimera.recorder.contract import RecorderContract
from chimera.recorder.events import (
    MS_PER_MINUTE,
    NS_PER_MILLISECOND,
    NS_PER_SECOND,
    UM_FUNDING,
    UM_MARK_PRICE,
    EventSource,
    FundingSettlement,
    KlineEvent,
    MarkPriceEvent,
    RawEvent,
    RecorderEventError,
    day_start_ns,
    utc_day,
)
from chimera.recorder.health import (
    HEARTBEAT_INTERVAL_S,
    HeartbeatWriter,
    RecorderHealth,
    initial_health,
)
from chimera.recorder.incremental import IncrementalNormalizer
from chimera.recorder.normalize import MinuteNormalizer, RecorderNormalizeError
from chimera.recorder.rest import (
    FUNDING_CATCHUP_INTERVAL_S,
    FUNDING_POLL_DELAY_S,
    RecorderRestError,
    RestPoller,
    expected_funding_instants_ms,
    premium_index_payload,
)
from chimera.recorder.sink import (
    AppendOutcome,
    RawSink,
    RecorderSinkError,
    available_days,
    read_raw_events,
    write_json_atomic,
)
from chimera.recorder.streams import StreamClient, clients_for

logger = logging.getLogger(__name__)

#: How often the open raw files are fsynced. Section 4.3's "at most once per
#: second"; the sink deliberately owns no timer, so the cadence lives here.
SYNC_INTERVAL_S = 1.0

#: How often the current UTC day is re-normalized while it is still open, at
#: least. Often enough that ``missing_minutes`` and the normalized files are
#: meaningful on a running recorder; rarely enough that a 1440-row Parquet is not
#: rewritten every minute.
NORMALIZE_INTERVAL_S = 300.0

#: The largest share of wall time the service will spend re-normalizing the open
#: day. Re-normalizing re-reads every raw event of the day, so its cost grows
#: with the day: measured on a real BTCUSDT recording, ``build_day`` for the
#: perpetual took 66 s once the book stream held 1.5 M rows, and a full day holds
#: far more than that. A fixed interval would therefore have the recorder
#: spending most of its time normalizing by evening. The interval below is a
#: floor and this is the ceiling: whichever is longer wins, so the work stays a
#: bounded fraction of the machine no matter how large the day gets.
NORMALIZE_DUTY_CYCLE = 0.1

#: How long after a UTC midnight the previous day's raw files are frozen. A
#: frame stamped 23:59:59 can be delivered after 00:00:00, and freezing at the
#: stroke of midnight would route genuinely in-time observations to the late
#: file. Two minutes is comfortably longer than any delivery delay that is not
#: already a disconnect.
ROTATION_GRACE_S = 120.0

#: How far back a cold start looks for klines it may have missed. A recorder
#: down for longer than this is not gap-filling, it is backfilling, and the
#: archive reconciliation is the right tool for that.
MAX_GAPFILL_MINUTES = 1440

#: How far back a cold start looks for funding settlements. Three days covers a
#: weekend outage at the eight-hour cadence in force, with room to spare.
FUNDING_LOOKBACK_MS = 3 * 24 * 60 * 60 * 1000

#: How often the premium index is polled. Section 4.1: "every minute".
PREMIUM_INDEX_INTERVAL_S = 60.0

#: Where the storage root keeps a copy of the contract it was recorded under.
#: Section 4.3's layout, and one of the three things .gitignore re-includes, so
#: a directory of recordings says for itself what it was recorded under.
CONTRACT_DIRECTORY = "contract"


class RecorderServiceError(RuntimeError):
    """The service cannot start, or cannot keep a promise it has made."""


@dataclass
class ServiceResult:
    """What one run of the service did. Returned rather than logged and lost."""

    started_ns: int
    stopped_ns: int
    events: int = 0
    duplicates: int = 0
    late: int = 0
    write_errors: int = 0
    gapfill_rows: int = 0
    reconnects: int = 0
    heartbeats: int = 0
    normalized_days: tuple[str, ...] = ()
    halted_streams: tuple[str, ...] = ()
    dropped_after_halt: int = 0
    errors: tuple[str, ...] = ()

    @property
    def seconds(self) -> float:
        return (self.stopped_ns - self.started_ns) / NS_PER_SECOND

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_ns": self.started_ns,
            "stopped_ns": self.stopped_ns,
            "seconds": self.seconds,
            "events": self.events,
            "duplicates": self.duplicates,
            "late": self.late,
            "write_errors": self.write_errors,
            "gapfill_rows": self.gapfill_rows,
            "reconnects": self.reconnects,
            "heartbeats": self.heartbeats,
            "normalized_days": list(self.normalized_days),
            "halted_streams": list(self.halted_streams),
            "dropped_after_halt": self.dropped_after_halt,
            "errors": list(self.errors),
        }


@dataclass
class _Recovery:
    """What the startup pass found and did. Reported, and asserted by tests."""

    tails: dict[str, int] = field(default_factory=dict)
    last_canonical_ns: dict[str, int | None] = field(default_factory=dict)
    gapfill_rows: dict[str, int] = field(default_factory=dict)
    settlements: int = 0
    normalized: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tails": dict(self.tails),
            "last_canonical_ns": dict(self.last_canonical_ns),
            "gapfill_rows": dict(self.gapfill_rows),
            "settlements": self.settlements,
            "normalized": list(self.normalized),
            "notes": list(self.notes),
        }


class RecorderService:
    """Runs the prospective recorder for one contract into one storage root."""

    def __init__(
        self,
        contract: RecorderContract,
        root: str | Path,
        *,
        poller: RestPoller | None = None,
        clients: Sequence[StreamClient] | None = None,
        client_options: Mapping[str, Any] | None = None,
        wall_ns: Callable[[], int] = time.time_ns,
        source_revision: str | None = None,
        sync_interval_s: float = SYNC_INTERVAL_S,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
        normalize_interval_s: float = NORMALIZE_INTERVAL_S,
        premium_index_interval_s: float = PREMIUM_INDEX_INTERVAL_S,
        funding_catchup_interval_s: float = FUNDING_CATCHUP_INTERVAL_S,
        rotation_grace_s: float = ROTATION_GRACE_S,
        gapfill: bool = True,
    ) -> None:
        self.contract = contract
        self.root = Path(root)
        self._wall_ns = wall_ns
        self.poller = poller if poller is not None else RestPoller()
        self.sync_interval_s = sync_interval_s
        self.heartbeat_interval_s = heartbeat_interval_s
        self.normalize_interval_s = normalize_interval_s
        self.premium_index_interval_s = premium_index_interval_s
        self.funding_catchup_interval_s = funding_catchup_interval_s
        self.rotation_grace_s = rotation_grace_s
        self.gapfill = gapfill

        self.sinks: dict[str, RawSink] = {
            stream: RawSink(self.root, stream, contract=contract)
            for stream in contract.streams
        }
        self.normalizer = MinuteNormalizer(self.root, contract)
        #: The path the service actually renders through. It folds the open day
        #: from a cursor instead of re-reading it, and falls back to
        #: ``self.normalizer`` — the authoritative full rebuild — whenever its
        #: cache cannot be vouched for. Both write the day through the same
        #: writer, so a fallback costs time and changes no value.
        self.incremental = IncrementalNormalizer(
            self.root, contract, normalizer=self.normalizer
        )
        self.health: RecorderHealth = initial_health(contract, source_revision=source_revision)
        self.heartbeat = HeartbeatWriter(self.root, wall_ns=wall_ns)
        self.clients: tuple[StreamClient, ...] = (
            tuple(clients)
            if clients is not None
            else clients_for(contract, self._record, **dict(client_options or {}))
        )
        self.recovery = _Recovery()
        self._stop: asyncio.Event | None = None
        self._errors: list[str] = []
        self._current_day: str | None = None
        self._normalized_days: list[str] = []
        self._frozen_after: dict[str, float] = {}
        self._last_maintenance_s = 0.0

    # --- writing ----------------------------------------------------------
    def _record(self, event: RawEvent) -> None:
        """Append one observation and account for what happened to it.

        Called from the websocket read loop and from every poller, so it is the
        single place a write error, a duplicate or a late arrival is counted.

        **A write failure halts the stream** (section 2.2, the ``A -> B`` arrow:
        "write error -> recorder halts that stream, health metric flips, alert").
        Counting the failure and carrying on would leave the recorder presenting
        a stream as operational while dropping every observation on it — the
        socket keeps delivering, the metric keeps saying up, and the silence
        looks like a quiet market rather than a broken disk. So the failure is
        latched, further observations for that stream are refused and counted,
        and only a restart, through the whole recovery path, can lift it.

        Exactly one stream is halted, not the run: the other five keep recording
        if they can, and a client that multiplexes several streams over one
        socket keeps serving its healthy ones, because the halt is applied here,
        per stream id, rather than by dropping a connection.
        """
        stream = self.health.stream(event.stream)
        if stream.halted:
            stream.dropped_after_halt += 1
            return
        sink = self.sinks.get(event.stream)
        if sink is None:
            self._halt(
                event.stream, "the contract declares no such stream, so there is no sink"
            )
            return
        try:
            result = sink.append(event)
        except (RecorderSinkError, OSError) as exc:
            self._halt(event.stream, f"append failed: {exc}")
            return
        if result.outcome is AppendOutcome.ACCEPTED:
            stream.events += 1
        elif result.outcome is AppendOutcome.DUPLICATE:
            stream.duplicates += 1
        elif result.outcome is AppendOutcome.LATE:
            stream.late += 1
        else:
            stream.late += 1
            stream.duplicates += 1
        if result.accepted or result.outcome is AppendOutcome.LATE:
            last = stream.last_event_ns
            if last is None or event.canonical_ns > last:
                stream.last_event_ns = event.canonical_ns
        if event.source is EventSource.REST_GAPFILL and result.accepted:
            stream.gapfill_rows += 1

    def _note(self, message: str) -> None:
        """Record something an operator has to see, without stopping."""
        logger.warning("%s", message)
        self._errors.append(message)
        self.health.errors = tuple(self._errors[-32:])

    def _halt(self, name: str, reason: str) -> None:
        """Latch a storage failure for one stream and stop recording it.

        The sink is closed on the way out so that nothing can append to it by
        another path, and because closing is the last chance to flush what was
        already accepted. Closing may itself fail — the storage is, after all,
        what just failed — and that is noted rather than raised: the stream is
        already halted and there is nothing further to protect.
        """
        stream = self.health.stream(name)
        first = stream.halt(reason, now_ns=self._wall_ns())
        if not first:
            return
        stream.connected = False
        logger.error("%s halted: %s", name, reason)
        self._note(f"{name} halted and is recording nothing: {reason}")
        sink = self.sinks.get(name)
        if sink is not None:
            try:
                sink.close()
            except (RecorderSinkError, OSError) as exc:
                self._note(f"{name} could not be closed after its halt: {exc}")

    # --- startup ----------------------------------------------------------
    def recover(self) -> _Recovery:
        """Steps 1, 2 and 4 of section 4.3's recovery. Synchronous and offline.

        Separated from the gap-fill so that a recorder can be brought up, and
        tested, without a network: everything here reads and rewrites this
        host's own files.
        """
        recovery = _Recovery()
        self.bind_contract()
        self._sync()
        now_ns = self._wall_ns()
        today = utc_day(now_ns)
        self._current_day = today
        for stream, sink in self.sinks.items():
            repaired = 0
            for day in available_days(self.root, stream):
                if sink.is_frozen(day):
                    continue
                for late in (False, True):
                    outcome = sink.recover_tail(day, late=late)
                    repaired += outcome.truncated_records
            recovery.tails[stream] = repaired
            if repaired:
                recovery.notes += (f"{stream}: repaired {repaired} torn record(s)",)
        # The latest instant per stream, from the normalize cache where there is
        # one. Reading it by parsing every record is what made a restart cost
        # minutes on a busy day, and the cache already holds the maximum over
        # everything it has folded, so only the unfolded tail is looked at.
        cached: dict[str, int] = {}
        for market in self.contract.market_keys():
            cached.update(self.incremental.peek_last_canonical(market, today))
        for stream in self.sinks:
            last = cached.get(stream)
            if last is None:
                last = self._last_canonical_ns(stream)
            recovery.last_canonical_ns[stream] = last
            if last is not None:
                self.health.stream(stream).last_event_ns = last
        for market in self.contract.market_keys():
            # The same renderer the running service uses: incremental where the
            # cache allows it, the authoritative rebuild where it does not. A
            # restart that rendered through a different path than the run would
            # be a restart that could disagree with itself.
            if self._normalize(market, today):
                recovery.normalized += (f"{market}/{today}",)
                status = self.incremental.status.get((market, today))
                if status is not None and status.rebuilt:
                    recovery.notes += (f"{market} {today}: {status.reason}",)
        self.health.open_day = today
        self.health.normalized_day = today
        self.recovery = recovery
        return recovery

    def bind_contract(self) -> Path:
        """Write the contract copy into the storage root, or refuse to mix contracts.

        Section 4.3 puts ``contract/<id>.json`` beside the data, and the
        contract's own version policy says one campaign never mixes contracts.
        Both are enforced here, at the one moment a process takes ownership of a
        root: if the root already carries a contract, its identity must be the
        identity of the contract this service was given, and a mismatch stops the
        service rather than appending days of one generation to another's
        directory.
        """
        path = self.root / CONTRACT_DIRECTORY / f"{self.contract.contract_id}.json"
        document = {
            "contract": self.contract.to_dict(),
            "contract_hash": self.contract.contract_hash,
            "storage_layout_version": self.contract.storage_layout_version,
        }
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RecorderServiceError(
                    f"{path} is not readable as a contract copy: {exc}. A storage root whose "
                    "contract cannot be read is one nothing can say the provenance of"
                ) from exc
            recorded = existing.get("contract_hash")
            if recorded != self.contract.contract_hash:
                raise RecorderServiceError(
                    f"{self.root} holds data recorded under contract hash {recorded}, and "
                    f"this recorder has {self.contract.contract_hash}. One campaign never "
                    "mixes contracts: record the new generation under its own root"
                )
            return path
        write_json_atomic(path, document)
        logger.info("bound %s to %s", self.root, self.contract.label)
        return path

    def _last_canonical_ns(self, stream: str) -> int | None:
        """The last canonical instant this host holds for one stream.

        Read from the sink's own manifests and open files rather than kept in
        memory, because the point of it is to survive a process that is no
        longer running.
        """
        sink = self.sinks[stream]
        latest: int | None = None
        for day in available_days(self.root, stream):
            manifest = sink.manifest_path(day)
            if manifest.exists():
                try:
                    document = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    self._note(f"unreadable manifest {manifest}: {exc}")
                    continue
                value = document.get("last_canonical_ns")
                if isinstance(value, int) and (latest is None or value > latest):
                    latest = value
                continue
            for event in self._read_day(stream, day):
                if latest is None or event.canonical_ns > latest:
                    latest = event.canonical_ns
        return latest

    def _read_day(self, stream: str, day: str) -> list[RawEvent]:
        """Every observation on disk for one stream and day, late file included."""
        try:
            return read_raw_events(self.root, stream, day)
        except (RecorderSinkError, RecorderEventError, OSError) as exc:
            self._note(f"could not read {stream} {day}: {exc}")
            return []

    # --- gap fill ---------------------------------------------------------
    async def fill_kline_gap(self, market: str, *, since_ns: int | None = None) -> int:
        """Fetch the closed klines this host may have missed, and record them.

        The window starts at the minute after the last one recorded — or at the
        gap-fill horizon, whichever is later — and ends at the last *closed*
        minute. The currently forming minute is never requested: a REST row for
        it would be a partial candle wearing a closed candle's shape.

        The horizon is ``MAX_GAPFILL_MINUTES`` *candles*, not that many minutes
        of offset. Binance's range is inclusive at both ends, so the oldest open
        included is ``last_closed - (N - 1) * 60_000``: subtracting the whole
        ``N`` would ask for ``N + 1`` candles, and a cold start under a 1440
        cap fetched 1441.
        """
        stream = f"{market}.kline_1m"
        if stream not in self.sinks:
            return 0
        now_ns = self._wall_ns()
        last_closed_open_ms = (now_ns // NS_PER_MILLISECOND) // MS_PER_MINUTE * MS_PER_MINUTE
        last_closed_open_ms -= MS_PER_MINUTE
        horizon_ms = last_closed_open_ms - (MAX_GAPFILL_MINUTES - 1) * MS_PER_MINUTE
        if since_ns is None:
            since_ns = self.recovery.last_canonical_ns.get(stream)
        start_ms = (
            horizon_ms if since_ns is None else since_ns // NS_PER_MILLISECOND + MS_PER_MINUTE
        )
        start_ms = max(start_ms, horizon_ms)
        if start_ms > last_closed_open_ms:
            return 0
        symbol = self.contract.market(market).symbol
        try:
            rows = await asyncio.to_thread(
                self.poller.klines, market, start_ms, last_closed_open_ms, symbol=symbol
            )
        except RecorderRestError as exc:
            self._note(f"{stream} gap-fill failed: {exc}")
            return 0
        before = self.health.stream(stream).gapfill_rows
        now = self._wall_ns()
        mono = time.monotonic_ns()
        for row in rows:
            # The stored payload is the adapted REST row, not the raw array, so
            # a gap-filled minute reads back through exactly the parser a pushed
            # frame does and nothing published is dropped on the way.
            try:
                payload = KlineEvent.rest_payload(row)
                parsed = KlineEvent.from_payload(payload, stream=stream)
            except RecorderEventError as exc:
                self._note(f"{stream} gap-fill row is not a kline: {exc}")
                continue
            self._record(
                parsed.to_raw_event(
                    payload,
                    receipt_wall_ns=now,
                    receipt_mono_ns=mono,
                    source=EventSource.REST_GAPFILL,
                )
            )
        filled = self.health.stream(stream).gapfill_rows - before
        self.recovery.gapfill_rows[stream] = self.recovery.gapfill_rows.get(stream, 0) + filled
        if filled:
            logger.info("%s gap-filled %d closed minute(s)", stream, filled)
        return filled

    async def poll_funding(self, *, since_ms: int | None = None) -> int:
        """Fetch realised settlements and record every one the exchange returns.

        ``fundingRate`` is authoritative for a settlement that *happened*. How
        many the day was scheduled to have is a different question this service
        does not ask and does not assume: the rows are recorded exactly as they
        come back, and PR-06's reconciliation establishes the schedule from the
        archive.
        """
        if UM_FUNDING not in self.sinks:
            return 0
        now_ms = self._wall_ns() // NS_PER_MILLISECOND
        start_ms = now_ms - FUNDING_LOOKBACK_MS if since_ms is None else since_ms
        symbol = self.contract.market("um").symbol
        try:
            rows = await asyncio.to_thread(
                self.poller.funding_rate, max(0, start_ms), now_ms, symbol=symbol
            )
        except RecorderRestError as exc:
            self._note(f"funding poll failed: {exc}")
            return 0
        recorded = 0
        wall = self._wall_ns()
        mono = time.monotonic_ns()
        for row in rows:
            try:
                settlement = FundingSettlement.from_rest(row, stream=UM_FUNDING)
            except RecorderEventError as exc:
                self._note(f"a fundingRate row is not a settlement: {exc}")
                continue
            self._record(
                settlement.to_raw_event(
                    row,
                    receipt_wall_ns=wall,
                    receipt_mono_ns=mono,
                    source=EventSource.REST_POLL,
                )
            )
            recorded += 1
        if recorded:
            self._rebuild_settlements()
        self.recovery.settlements += recorded
        return recorded

    async def poll_premium_index(self) -> bool:
        """Record the exchange's current mark, index and funding state.

        Section 4.1 polls ``premiumIndex`` every minute for current state. It is
        stored on ``um.markPrice`` — the stream that already carries exactly
        these five published values — with :attr:`EventSource.REST_POLL`, so
        that a websocket outage does not leave the mark and index with no
        observation at all for the minutes it covers. Nothing is invented: the
        adapter renames the endpoint's own fields onto the payload shape the
        parser reads, and ``lastFundingRate`` is current state and is never
        recorded as a settlement.
        """
        if UM_MARK_PRICE not in self.sinks:
            return False
        symbol = self.contract.market("um").symbol
        try:
            row = await asyncio.to_thread(self.poller.premium_index, symbol=symbol)
        except RecorderRestError as exc:
            self._note(f"premiumIndex poll failed: {exc}")
            return False
        try:
            payload = premium_index_payload(row)
            mark = MarkPriceEvent.from_payload(payload, stream=UM_MARK_PRICE)
        except (RecorderRestError, RecorderEventError) as exc:
            self._note(f"premiumIndex answered something unusable: {exc}")
            return False
        self._record(
            mark.to_raw_event(
                payload,
                receipt_wall_ns=self._wall_ns(),
                receipt_mono_ns=time.monotonic_ns(),
                source=EventSource.REST_POLL,
            )
        )
        return True

    def _rebuild_settlements(self) -> None:
        try:
            self.normalizer.build_settlements("um")
        except RecorderNormalizeError as exc:
            self._note(f"settlements rebuild refused: {exc}")

    # --- the run ----------------------------------------------------------
    async def run(self, stop: asyncio.Event | None = None) -> ServiceResult:
        """Recover, connect, and run until ``stop`` is set.

        Returns a :class:`ServiceResult` on a clean stop. Re-raises on a task
        that failed, after winding the rest of the service down: a recorder that
        kept reporting itself up with a dead stream would be worse than one that
        exited.
        """
        stop = stop or asyncio.Event()
        self._stop = stop
        started_ns = self._wall_ns()
        self.health.started_ns = started_ns
        self.recover()
        if self.gapfill:
            for market in self.contract.market_keys():
                await self.fill_kline_gap(market)
            await self.poll_funding()
            # Current state, once, before the periodic loops start. They sleep
            # before they act, which is right for a steady state and wrong for a
            # start: without this the mark and index have no observation at all
            # until a minute has passed, and on a start that follows an outage
            # that is a minute nothing holds any reading for.
            await self.poll_premium_index()
        self.heartbeat.write(self.health)

        tasks: list[asyncio.Task[Any]] = []
        for client in self.clients:
            tasks.append(asyncio.create_task(client.run(stop), name=client.name))
        tasks.append(self._loop(stop, self.sync_interval_s, self._sync, "sync"))
        tasks.append(self._loop(stop, self.heartbeat_interval_s, self._beat, "heartbeat"))
        tasks.append(self._maintenance_task(stop))
        tasks.append(
            self._loop(
                stop, self.premium_index_interval_s, self.poll_premium_index, "premium-index"
            )
        )
        tasks.append(self._funding_task(stop))

        failure: BaseException | None = None
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    failure = exc
                    self._note(f"task {task.get_name()} failed: {exc!r}")
                    break
        finally:
            stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._shutdown()

        result = self._result(started_ns)
        if failure is not None:
            raise RecorderServiceError(
                f"the recorder stopped because a task failed: {failure!r}"
            ) from failure
        return result

    def _loop(
        self, stop: asyncio.Event, interval: float, step: Callable[[], Any], name: str
    ) -> asyncio.Task[Any]:
        """A task that runs ``step`` every ``interval`` seconds until ``stop``."""

        async def body() -> None:
            while not stop.is_set():
                if await _sleep_or_stop(stop, interval):
                    return
                outcome = step()
                if inspect.isawaitable(outcome):
                    await outcome

        return asyncio.create_task(body(), name=name)

    def _maintenance_task(self, stop: asyncio.Event) -> asyncio.Task[Any]:
        """Roll the day over and re-normalize the open one, paced by its own cost.

        Separate from :meth:`_loop` because this is the one periodic job whose
        cost is not constant: it re-reads the day's raw events, so it gets slower
        as the day fills. The wait before each pass is the configured interval or
        the previous pass's duration divided by :data:`NORMALIZE_DUTY_CYCLE`,
        whichever is longer.
        """

        async def body() -> None:
            while not stop.is_set():
                delay = max(
                    self.normalize_interval_s,
                    self._last_maintenance_s / NORMALIZE_DUTY_CYCLE,
                )
                if await _sleep_or_stop(stop, delay):
                    return
                began = time.monotonic()
                await self._maintain()
                self._last_maintenance_s = time.monotonic() - began
                if self._last_maintenance_s > self.normalize_interval_s:
                    logger.warning(
                        "re-normalizing the open day took %.0fs, longer than the %.0fs "
                        "interval; the next pass waits %.0fs to stay under a %.0f%% duty "
                        "cycle",
                        self._last_maintenance_s,
                        self.normalize_interval_s,
                        self._last_maintenance_s / NORMALIZE_DUTY_CYCLE,
                        NORMALIZE_DUTY_CYCLE * 100,
                    )

        return asyncio.create_task(body(), name="normalize")

    def _funding_task(self, stop: asyncio.Event) -> asyncio.Task[Any]:
        """Poll shortly after each expected settlement, and hourly as a catch-up.

        The schedule says when to *ask*. It never says what the answer has to be:
        a day the venue settled a different number of times is recorded as it
        happened, and the poller does not check the count against a constant.
        """

        async def body() -> None:
            while not stop.is_set():
                delay = min(
                    self.funding_catchup_interval_s,
                    self._seconds_to_next_funding_poll(),
                )
                if await _sleep_or_stop(stop, max(1.0, delay)):
                    return
                await self.poll_funding()

        return asyncio.create_task(body(), name="funding")

    def _seconds_to_next_funding_poll(self) -> float:
        """Seconds until 60 s past the next expected settlement instant."""
        now_ms = self._wall_ns() // NS_PER_MILLISECOND
        today_ms = day_start_ns(utc_day(self._wall_ns())) // NS_PER_MILLISECOND
        candidates = [
            instant + int(FUNDING_POLL_DELAY_S * 1000)
            for instant in (
                *expected_funding_instants_ms(today_ms),
                *expected_funding_instants_ms(today_ms + 24 * 60 * 60 * 1000),
            )
        ]
        upcoming = [instant for instant in candidates if instant > now_ms]
        if not upcoming:
            return self.funding_catchup_interval_s
        return (min(upcoming) - now_ms) / 1000.0

    # --- periodic work ----------------------------------------------------
    def _sync(self) -> None:
        """Flush every open stream, and halt any whose durability has gone.

        An fsync that fails means the bytes already accepted are not on the disk,
        which is the same fact as a failed append arriving a moment later. It is
        treated the same way rather than counted and forgotten.
        """
        for stream, sink in self.sinks.items():
            if self.health.stream(stream).halted:
                continue
            try:
                sink.sync()
            except (RecorderSinkError, OSError) as exc:
                self._halt(stream, f"fsync failed: {exc}")

    def _beat(self) -> None:
        self._refresh_connection_state()
        self.heartbeat.write(self.health)

    def _refresh_connection_state(self) -> None:
        """Copy what the clients know into the health snapshot.

        A halted stream keeps ``connected`` false whatever its socket is doing.
        The socket is very likely fine — a disk failing does not close a
        connection — and letting this refresh copy that fact over the latch is
        exactly how a halted stream would come back up in the report a moment
        after it stopped recording. ``halted`` is never written here.
        """
        skews = []
        for client in self.clients:
            for stream_id in client.stream_ids:
                stream = self.health.stream(stream_id)
                stream.connected = client.connected and not stream.halted
                stream.reconnects = client.counters.reconnects
                stream.out_of_order = client.counters.out_of_order
                stream.decode_errors = client.counters.decode_errors
            median = client.skew.median_ms()
            if median is not None:
                skews.append(median)
        self.health.clock_skew_ms = None if not skews else sum(skews) / len(skews)
        self.health.open_day = self._current_day

    async def _maintain(self) -> None:
        """Roll the day over when it turns, then re-normalize what is open.

        Every step here reads or rewrites whole files, and on a real recording
        those files are large: re-normalizing one day of the perpetual re-reads
        the entire book stream, and freezing a day gzips it. Doing that on the
        event loop stops the websocket read loops for as long as it takes, the
        exchange's ping goes unanswered, and the recorder disconnects itself —
        which is exactly what a 44-minute live run did, once every interval, on
        both connections at the same instant. It runs in a worker thread.
        """
        today = utc_day(self._wall_ns())
        if self._current_day is not None and today != self._current_day:
            closed = self._current_day
            self._current_day = today
            self._frozen_after[closed] = time.monotonic() + self.rotation_grace_s
            for market in self.contract.market_keys():
                await asyncio.to_thread(self._normalize, market, closed)
            if self.gapfill:
                for market in self.contract.market_keys():
                    await self.fill_kline_gap(market)
        await asyncio.to_thread(self._freeze_due)
        # Raw first, always. A cursor may only ever claim material the raw files
        # already hold durably, so the fsync happens before anything folds.
        self._sync()
        for market in self.contract.market_keys():
            await asyncio.to_thread(self._normalize, market, today)
        self.health.normalized_day = today

    def _normalize(self, market: str, day: str) -> bool:
        """Render one day, incrementally where the cache allows it.

        Reads the raw files and writes the normalized ones; it touches no
        stream's health beyond the missing-minute count, so a stream halted by a
        storage failure stays halted and is not revived by having its day
        rendered. Returns whether the day was written.
        """
        try:
            report = self.incremental.build_day(market, day, provenance=self._provenance())
        except RecorderNormalizeError as exc:
            self._note(f"{market} {day} not normalized: {exc}")
            return False
        self._remember_normalized(market, day, len(report.missing))
        return True

    def _remember_normalized(self, market: str, day: str, missing: int) -> None:
        label = f"{market}/{day}"
        if label not in self._normalized_days:
            self._normalized_days.append(label)
        stream = f"{market}.kline_1m"
        if stream in self.sinks:
            self.health.stream(stream).missing_minutes = missing

    def _freeze_due(self) -> None:
        """Freeze the raw files of a day whose grace period has passed."""
        now = time.monotonic()
        for day, due in sorted(self._frozen_after.items()):
            if now < due:
                continue
            for stream, sink in self.sinks.items():
                if sink.is_frozen(day):
                    continue
                try:
                    sink.freeze_day(day, provenance=self._provenance())
                except RecorderSinkError as exc:
                    self._note(f"{stream} {day} not frozen: {exc}")
            del self._frozen_after[day]

    def _provenance(self) -> dict[str, Any]:
        return {
            "recorder": "chimera.recorder.service",
            "source_revision": self.health.source_revision,
            "evidence_class": self.health.evidence_class,
        }

    # --- shutdown ---------------------------------------------------------
    def _shutdown(self) -> None:
        """Sync, normalize what is open, write a last heartbeat, close the files."""
        self._sync()
        today = utc_day(self._wall_ns())
        for market in self.contract.market_keys():
            self._normalize(market, today)
        self._refresh_connection_state()
        for client in self.clients:
            for stream_id in client.stream_ids:
                self.health.stream(stream_id).connected = False
        self.heartbeat.write(self.health)
        for stream, sink in self.sinks.items():
            if self.health.stream(stream).halted:
                continue  # closed when it was halted; reopening it writes nothing
            try:
                sink.close()
            except (RecorderSinkError, OSError) as exc:
                self._note(f"could not close {stream}: {exc}")

    def _result(self, started_ns: int) -> ServiceResult:
        streams = self.health.streams.values()
        return ServiceResult(
            started_ns=started_ns,
            stopped_ns=self._wall_ns(),
            events=sum(stream.events for stream in streams),
            duplicates=sum(stream.duplicates for stream in streams),
            late=sum(stream.late for stream in streams),
            write_errors=sum(stream.write_errors for stream in streams),
            gapfill_rows=sum(stream.gapfill_rows for stream in streams),
            reconnects=sum(client.counters.reconnects for client in self.clients),
            heartbeats=self.heartbeat.writes,
            normalized_days=tuple(self._normalized_days),
            halted_streams=self.health.halted_streams,
            dropped_after_halt=sum(stream.dropped_after_halt for stream in streams),
            errors=tuple(self._errors),
        )


async def _sleep_or_stop(stop: asyncio.Event, delay: float) -> bool:
    """Wait ``delay`` seconds or until ``stop``. ``True`` when ``stop`` fired."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
    except asyncio.TimeoutError:
        return False
    return True


def build_service(
    contract: RecorderContract, base_dir: str | Path, **options: Any
) -> RecorderService:
    """A service writing under the contract's own storage root beneath ``base_dir``."""
    root = contract.storage_root(base_dir)
    return RecorderService(contract, root, **options)
