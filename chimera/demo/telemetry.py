"""The demo runner's Prometheus surface, and the only place on the path that has one.

Section 11.1 names eighteen runner series plus three the Freqtrade path already
publishes. They are emitted from here rather than from the runner because a
single emission surface is a claim a test can check: after this module exists,
the set of files under ``chimera/demo/`` and ``chimera/carry/`` that import
:mod:`chimera.metrics` is exactly ``{chimera/demo/telemetry.py}``, so no rule, no
hedge, no ledger, no fill model and no log writer can name a metric at all.

**Observation may not change what it observes.** Everything here writes; nothing
here reads a metric back, returns a value, or touches a store, a ledger, a risk
engine or a file. Each method is a statement the runner executes and discards.
That is asserted three ways in ``tests/test_demo_observability.py``: structurally
(no metric value is read anywhere in the runtime packages, every method returns
``None``, and every ``self.telemetry.…`` call in the runner is a bare statement),
positionally (no telemetry call sits inside rule evaluation, the risk check or
execution, and the decision-derived series move only after the DECISION record is
on disk), and behaviourally (three campaigns over the same synthetic day -- one
with this emitter, one with :class:`NullTelemetry`, one with a wall clock frozen
three hundred million seconds away -- write byte-identical decision logs).

**The wall clock is read here and nowhere else on the demo path.** The runner's
own instants come from :class:`~chimera.demo.clock.RunnerClock`, which is advanced
only by observed record instants, because section 10's replay parity is a byte
comparison. An age in seconds is nevertheless the question an operator asks --
"is this process alive, and is its feed current?" -- and that question is about
wall time. So the two ages and the heartbeat are stamped from ``time.time_ns``
*here*, where the value cannot reach a record: the frozen-clock run in the tests
is what proves it, not this paragraph.

**What is deliberately not emitted.** No ``chimera.modes`` series: this module
does not import that family and the demo deployment publishes none of it, because
a mode metric on a single-mode campaign would suggest a mode selection that is
not happening. No per-symbol, per-order, per-price or per-reason-text label: every
label value below comes from a committed ``Enum`` or the campaign configuration's
own rule ids, so the number of time series is fixed before the campaign starts.
No latency histogram: nothing on this path is latency-bounded, and a histogram
nothing reads is a panel that reads as healthy because it is empty. And no series
derived from a metric: every number below is read from state that already exists
and is already evidence.

**Aegis's own three series.** ``chimera_risk_halted``, ``chimera_drawdown`` and
``chimera_rejected_entries_total`` are section 11.1's for the demo deployment and
already exist for the Freqtrade path. They are written here, from
:meth:`~chimera.risk.RiskEngine.snapshot` and
:meth:`~chimera.risk.RiskEngine.current_drawdown`, both of which are documented
read-only views. Neither is redeclared in :mod:`chimera.metrics`; the demo
deployment simply becomes a second writer of the same three.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from chimera.carry.hedge import HedgeState
from chimera.demo.decision_log import RecordKind
from chimera.metrics import (
    DEMO_BASIS,
    DEMO_DECISIONS,
    DEMO_DISK_FREE,
    DEMO_EQUITY,
    DEMO_FEED_AGE,
    DEMO_FUNDING,
    DEMO_FUNDING_ADVERSE_STREAK,
    DEMO_HEARTBEAT,
    DEMO_HEDGE_IMBALANCE,
    DEMO_LAST_MINUTE_AGE,
    DEMO_LIQUIDATION_DISTANCE,
    DEMO_LOG_RECORDS,
    DEMO_LOG_WRITE_ERRORS,
    DEMO_NET_PNL,
    DEMO_TICKS,
    DEMO_UP,
    DRAWDOWN,
    REJECTED_ENTRIES,
    RISK_HALTED,
    set_demo_state,
    set_hedge_state,
)
from chimera.recorder.events import NS_PER_SECOND
from chimera.recorder.health import disk_free_bytes

logger = logging.getLogger(__name__)

__all__ = [
    "LEGS",
    "MARKETS",
    "SIGNAL_KINDS",
    "VETO_LABELS",
    "NullTelemetry",
    "RunnerTelemetry",
]

MINUTE_NS = 60 * NS_PER_SECOND

#: The two markets ``MarketState.feed_age_ns`` is keyed by, and the two prefixes
#: its ``missing`` tuple uses (``um_minute``, ``spot_minute``).
MARKETS: tuple[str, ...] = ("um", "spot")
#: The two legs of a carry position, named as ``chimera.carry.hedge`` names them.
LEGS: tuple[str, ...] = ("spot", "perp")
#: The two words ``RuleDecision.to_record()`` puts in ``signal.kind``.
SIGNAL_KINDS: tuple[str, ...] = ("target", "signal_only")
#: The runner's own veto labels. ``no_intent`` is the only one ``_decide`` can
#: produce; anything else collapses to ``other``, so the label set cannot grow
#: with traffic even if a later veto arrives with a new name. Aegis's own reasons
#: already have a bounded home in ``chimera_futures_risk_vetoes_total``.
VETO_LABELS: frozenset[str] = frozenset({"no_intent"})
#: The collapse target for a veto label this module does not know.
OTHER_VETO = "other"


class NullTelemetry:
    """Every method of :class:`RunnerTelemetry`, doing nothing.

    Not a convenience for tests that dislike side effects: it is the control arm.
    A campaign driven by this must write a decision log byte-identical to one
    driven by the real emitter, and that comparison is what says observability
    cannot reach a decision. Any method added to :class:`RunnerTelemetry` and not
    to this class breaks the control rather than the campaign, which is the right
    way round.
    """

    def on_state(self, state: str) -> None: ...

    def on_minute(self, *, minute_ns: int, missing: Sequence[str]) -> None: ...

    def on_record(self, kind: str) -> None: ...

    def on_log_write_error(self) -> None: ...

    def on_position(self, position: Any) -> None: ...

    def on_reporting(
        self,
        *,
        position: Any,
        risk: Any,
        mark: Any,
        market: Any,
        decisions: Sequence[Any],
        veto: Mapping[str, Any] | None,
    ) -> None: ...

    def on_shutdown(self) -> None: ...


class RunnerTelemetry:
    """Section 11.1's series, written from values the runner has already decided.

    ``states`` and ``rules`` are passed in rather than imported: importing
    ``RunnerState`` here would make the module the runner imports import the
    runner back, and the rule ids are the campaign configuration's, which this
    module has no business parsing. Both are used only to pre-create the label
    children at zero, so ``rate()`` and ``increase()`` are defined from the
    campaign's first minute instead of from its first event -- an empty child is
    the difference between "no halts" and "no data" on a dashboard.
    """

    def __init__(
        self,
        *,
        state_dir: str | Path,
        states: Sequence[str],
        rules: Sequence[str],
        wall_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._state_dir = Path(state_dir)
        self._states = tuple(states)
        self._hedge_states = tuple(state.value for state in HedgeState)
        self._wall_ns = wall_ns
        # The ledger reports funding as two absolute, PERSISTED magnitudes; a
        # Prometheus counter is monotone, per-process, and takes increments. The
        # totals last seen are held here so each report publishes the difference,
        # the same way `chimera.recorder.health.publish` turns absolute stream
        # counts into increments. `None` is "not yet observed": the first report
        # of a process adopts the ledger's totals as its starting point rather
        # than incrementing by them, because a restart would otherwise re-inject
        # the whole campaign's funding history into a counter Prometheus has just
        # watched reset to zero. The ledger file stays the authority on the
        # campaign total; this counter is the authority on what THIS process saw.
        # Neither total can decrease -- `CarryLedger.book_funding` only ever adds
        # a magnitude to one of them -- so the difference is never negative and
        # `.inc()` can never be handed a negative number.
        self._funding_seen: dict[str, Any] = {"paid": None, "received": None}
        self._market_close_ns: dict[str, int] = {}

        DEMO_UP.set(1.0)
        for kind in RecordKind:
            DEMO_LOG_RECORDS.labels(kind=kind.value).inc(0)
        for rule_id in rules:
            for signal_kind in SIGNAL_KINDS:
                DEMO_DECISIONS.labels(rule=rule_id, kind=signal_kind).inc(0)
        for direction in ("paid", "received"):
            DEMO_FUNDING.labels(direction=direction).inc(0)
        for label in (*sorted(VETO_LABELS), OTHER_VETO):
            REJECTED_ENTRIES.labels(reason=label).inc(0)
        for leg in LEGS:
            DEMO_LIQUIDATION_DISTANCE.labels(leg=leg).set(math.nan)
        set_hedge_state(HedgeState.FLAT.value, states=self._hedge_states)

    # ------------------------------------------------------------------
    # the runner's state machine
    # ------------------------------------------------------------------
    def on_state(self, state: str) -> None:
        """One section 8.1 state change.

        The heartbeat is wall time, not the runner's clock. It answers "is the
        process alive", which a data-derived instant cannot: replaying a
        historical day, a heartbeat carrying the replayed minute would read as
        years stale and the liveness alert would fire for ever. How far behind
        the *feed* is has its own series, ``chimera_demo_last_minute_age_seconds``.
        """
        set_demo_state(state, states=self._states)
        DEMO_HEARTBEAT.set(self._wall_ns() / NS_PER_SECOND)

    def on_minute(self, *, minute_ns: int, missing: Sequence[str]) -> None:
        """One attempted minute, counted before anything is decided about it.

        Deliberately upstream of rule evaluation, and deliberately carrying no
        decision-derived value: a minute that halts inside a rule is still a
        minute the runner attempted, and a tick counter that only counted the
        minutes that finished would read healthiest exactly when the campaign was
        failing. Nothing written here is read by anything.

        A market's age is measured from the close of the newest minute that
        market actually produced, so an incomplete minute leaves the missing
        market's age rising while the other's resets -- which is the fact
        distinguishing "the whole feed stopped" from "one stream stopped".
        """
        close_ns = int(minute_ns) + MINUTE_NS
        absent = set(missing)
        now_ns = self._wall_ns()
        for market in MARKETS:
            if f"{market}_minute" not in absent:
                self._market_close_ns[market] = close_ns
            seen = self._market_close_ns.get(market)
            if seen is not None:
                DEMO_FEED_AGE.labels(market=market).set(_age_seconds(now_ns, seen))
        DEMO_LAST_MINUTE_AGE.set(_age_seconds(now_ns, close_ns))
        DEMO_TICKS.inc()

    def on_shutdown(self) -> None:
        """The process is going away on purpose, so ``up`` says so rather than
        being inferred from a scrape that stopped answering."""
        DEMO_UP.set(0.0)

    # ------------------------------------------------------------------
    # the log
    # ------------------------------------------------------------------
    def on_record(self, kind: str) -> None:
        """One record, counted only once it is on disk. See `DemoRunner._append`."""
        DEMO_LOG_RECORDS.labels(kind=kind).inc()

    def on_log_write_error(self) -> None:
        """An append that raised. The runner re-raises; this only counts it."""
        DEMO_LOG_WRITE_ERRORS.inc()

    # ------------------------------------------------------------------
    # the position
    # ------------------------------------------------------------------
    def on_position(self, position: Any) -> None:
        """The hedge's state and imbalance, after whatever moved them."""
        set_hedge_state(position.state.value, states=self._hedge_states)
        DEMO_HEDGE_IMBALANCE.set(float(position.imbalance()))

    def on_reporting(
        self,
        *,
        position: Any,
        risk: Any,
        mark: Any,
        market: Any,
        decisions: Sequence[Any],
        veto: Mapping[str, Any] | None,
    ) -> None:
        """Section 8.1's REPORTING state, which is the whole of what it does.

        Reached only after the DECISION record is on disk and the runner state
        file is written, so every decision-derived number below describes a minute
        that is already evidence. Everything is read by value from objects the
        runner hands over; nothing is stored, saved, marked or recomputed.
        """
        self.on_position(position)
        for decision in decisions:
            DEMO_DECISIONS.labels(
                rule=decision.rule_id,
                kind=SIGNAL_KINDS[0] if decision.is_actionable else SIGNAL_KINDS[1],
            ).inc()

        ledger = position.ledger.state
        DEMO_EQUITY.set(float(mark.equity))
        DEMO_NET_PNL.set(float(mark.equity - ledger.capital))
        DEMO_BASIS.set(float(mark.basis))
        self._publish_funding(ledger)
        self._publish_liquidation(position, mark, market)

        snapshot = risk.snapshot()
        RISK_HALTED.set(1.0 if snapshot["halted"] else 0.0)
        # `snapshot` deliberately omits the drawdown, because it is a function of
        # two fields that are in it; the engine's own accessor is therefore the
        # single definition rather than a second one computed here.
        DRAWDOWN.set(risk.current_drawdown())
        DEMO_FUNDING_ADVERSE_STREAK.set(float(snapshot["funding_adverse_streak"]))

        if veto is not None:
            label = str(veto.get("label", ""))
            REJECTED_ENTRIES.labels(reason=label if label in VETO_LABELS else OTHER_VETO).inc()

        free = disk_free_bytes(self._state_dir)
        if free is not None:
            DEMO_DISK_FREE.set(free)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _publish_funding(self, ledger: Any) -> None:
        """The two funding directions as increments, and never as a net figure.

        Amendment A10 keeps paid and received apart everywhere, because a carry
        campaign that received 100 and paid 100 is not a campaign that did
        nothing: netting them at the metric would destroy the distinction before
        any dashboard could show it.
        """
        for direction, total in (
            ("paid", ledger.funding_paid),
            ("received", ledger.funding_received),
        ):
            seen = self._funding_seen[direction]
            self._funding_seen[direction] = total
            if seen is None:
                continue
            delta = total - seen
            if delta > 0:
                DEMO_FUNDING.labels(direction=direction).inc(float(delta))

    def _publish_liquidation(self, position: Any, mark: Any, market: Any) -> None:
        """Each leg's distance to liquidation, or NaN where there is no such thing.

        ``FuturesExecutor.margin`` returns ``None`` for a flat leg rather than a
        zeroed record, and NaN is the only honest gauge value for that: every
        PromQL comparison against NaN is false, so no alert can read a flat leg as
        being one tick from liquidation.

        The margin read is guarded because it is the one call here that can raise
        -- a non-positive mark, or a symbol the venue's constraint table does not
        carry. A campaign that halted because its monitoring could not compute a
        gauge would be observability changing the run, which is the single thing
        this module may not do; the leg reports NaN and the reason is logged.
        """
        prices = {"spot": mark.spot_close, "perp": market.mark}
        legs = {
            "spot": (position.spot, position.config.spot_symbol),
            "perp": (position.perp, position.config.perp_symbol),
        }
        for leg in LEGS:
            executor, symbol = legs[leg]
            price = prices[leg]
            distance = math.nan
            if price is not None:
                try:
                    state = executor.margin(symbol, price)
                except Exception as exc:  # pragma: no cover - defensive only
                    logger.warning("liquidation distance for %s unavailable: %s", leg, exc)
                    state = None
                if state is not None:
                    distance = float(state.liquidation_distance)
            DEMO_LIQUIDATION_DISTANCE.labels(leg=leg).set(distance)


def _age_seconds(now_ns: int, close_ns: int) -> float:
    """Wall seconds from a minute's close until now, floored at zero.

    A minute is processed as soon as its close has passed, but the two clocks are
    different clocks: a host whose wall time sits marginally behind the exchange's
    would otherwise publish a negative age, and a negative age on a staleness
    gauge reads as "fresher than possible" rather than as the clock skew it is.
    """
    return max(0.0, (now_ns - close_ns) / NS_PER_SECOND)
