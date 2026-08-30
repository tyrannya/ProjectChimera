"""Trading modes: temporal operating states, and the rules for entering them.

A **mode** here is a trading *style* — how fast a decision is taken and which
clocks it reads. It is not an instrument, a venue, a margin type or a strategy
category: `SCALPING` is not "futures", `SWING` is not "HODL", and none of these
is spot-versus-margin. Instrument selection is outside this generation entirely;
execution is Binance USD-M perpetual futures, isolated margin, exactly 1x,
dry-run, and this module does not touch any of that.

**This scaffold claims no alpha, and is built so that it cannot.**

* A mode is `ELIGIBLE` only when every specialist it names has been *screened*
  and found viable. Eligibility is read from committed research evidence, not
  from anything observed at run time.
* There is **no profit-based selection anywhere**. Nothing in this module reads a
  realised return, a PnL, an equity curve or a backtest rank, and
  :func:`assert_no_profit_input` exists so a future edit that added one fails a
  test rather than a review.
* Choosing automatically *between* eligible modes is a research question — `P8`,
  preregistered and **not opened**. Until it is answered, the mode is declared by
  an operator and this module's only job is to refuse a declaration the evidence
  does not support.
* `FLAT` is a first-class successful outcome, not a failure path. A system with
  no eligible mode is working correctly when it holds no position.

The decision rule inside an eligible mode is
:func:`chimera.consensus.decide` — the same function `P7` measured — so the
scaffold expresses the rule that was tested rather than a second one that agrees
with it today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

from chimera.consensus import ConsensusRule
from chimera.consensus import decide as consensus_decide
from chimera.contracts import Signal

__all__ = [
    "TradingMode",
    "ReasonCode",
    "SpecialistStatus",
    "ModeSpec",
    "MODE_SPECS",
    "Eligibility",
    "ModeDecision",
    "TransitionPlan",
    "ModeError",
    "evaluate_eligibility",
    "decide_mode",
    "plan_mode_transition",
    "assert_no_profit_input",
    "selection_source",
    "PROFIT_TOKENS",
]


class ModeError(ValueError):
    """A mode cannot mean what it has been asked to mean."""


class TradingMode(str, Enum):
    """The four operating states. Exactly one is active at a time."""

    SCALPING = "SCALPING"
    DAY_TRADING = "DAY_TRADING"
    SWING = "SWING"
    FLAT = "FLAT"


class ReasonCode(str, Enum):
    """Why the controller decided what it decided.

    A bounded enum, because these values become metric labels. A free-text
    reason as a label is a new time series per event, and Prometheus keeps them
    forever — the rule `chimera/metrics.py` states and this obeys.
    """

    #: The mode is eligible and its consensus reached a direction.
    CONSENSUS_LONG = "consensus_long"
    CONSENSUS_SHORT = "consensus_short"
    #: The mode is eligible and its specialists did not agree.
    NO_CONSENSUS = "no_consensus"
    #: The mode is eligible but a specialist has not produced a prediction yet.
    SPECIALIST_UNAVAILABLE = "specialist_unavailable"
    #: The declared mode names a specialist that has never been screened.
    SPECIALIST_UNSCREENED = "specialist_unscreened"
    #: The declared mode names a specialist screened and found not viable.
    SPECIALIST_NOT_VIABLE = "specialist_not_viable"
    #: No mode was declared, or FLAT was.
    NO_MODE_DECLARED = "no_mode_declared"
    #: Automatic selection was asked for. That is P8 and it is not opened.
    AUTO_ROUTING_NOT_OPENED = "auto_routing_not_opened"


@dataclass(frozen=True)
class SpecialistStatus:
    """What the research programme concluded about one clock's specialist.

    ``viable`` is the verdict of a *committed, preregistered* screen — P6 for the
    five short clocks, P6-EXT for `4h` and `1d`. It is not recomputed here and it
    is not observable at run time: a mode's eligibility is a fact about the
    evidence tree, which is what stops it becoming a fact about recent returns.
    """

    clock: str
    screened: bool
    viable: bool
    checkpoint: str = ""

    def __post_init__(self) -> None:
        if self.viable and not self.screened:
            raise ModeError(
                f"{self.clock}: a specialist cannot be viable without having been "
                "screened; viability is a verdict, not a default"
            )


@dataclass(frozen=True)
class ModeSpec:
    """One mode's clocks, cadence and decision rule.

    ``primary_clocks`` are the ones whose horizon defines the style;
    ``context_clocks`` are slower confirmation. Both vote — the split is
    descriptive, and what actually decides is ``rule``.
    """

    mode: TradingMode
    decision_clock: str
    primary_clocks: tuple[str, ...]
    context_clocks: tuple[str, ...]
    rule: ConsensusRule
    purpose: str

    @property
    def specialists(self) -> tuple[str, ...]:
        return tuple(self.primary_clocks) + tuple(self.context_clocks)

    @property
    def decision_cadence(self) -> str:
        """How often this mode takes a decision: once per decision-clock bar."""
        return f"once per closed {self.decision_clock} bar"

    def __post_init__(self) -> None:
        if self.decision_clock not in self.primary_clocks:
            raise ModeError(
                f"{self.mode.value}: the decision clock {self.decision_clock!r} is not "
                "among the primary clocks"
            )
        if set(self.primary_clocks) & set(self.context_clocks):
            raise ModeError(f"{self.mode.value}: a clock is both primary and context")
        if tuple(self.rule.specialists) != self.specialists:
            raise ModeError(
                f"{self.mode.value}: the consensus rule votes on "
                f"{list(self.rule.specialists)} and the mode names {list(self.specialists)}"
            )
        if self.rule.decision_clock != self.decision_clock:
            raise ModeError(f"{self.mode.value}: the rule and the mode disagree on the clock")


def _spec(
    mode: TradingMode,
    decision_clock: str,
    primary: tuple[str, ...],
    context: tuple[str, ...],
    agreement: int,
    purpose: str,
) -> ModeSpec:
    specialists = primary + context
    return ModeSpec(
        mode=mode,
        decision_clock=decision_clock,
        primary_clocks=primary,
        context_clocks=context,
        rule=ConsensusRule(
            mode=mode.value,
            decision_clock=decision_clock,
            specialists=specialists,
            veto_specialist=specialists[-1],
            agreement_required=agreement,
        ),
        purpose=purpose,
    )


#: The three directional modes. `FLAT` has no spec because it names no
#: specialist and takes no decision — it is the absence of a position, not a
#: fourth strategy.
#:
#: The scalping and day-trading specialist sets and agreement counts are the ones
#: `P7` measured, not new ones: the scaffold expresses the rule that was tested.
#: `SWING`'s rule has never been measured by any checkpoint, and it is marked so
#: in `docs/trading_modes_v1.md` rather than presented as though it had been.
MODE_SPECS: dict[TradingMode, ModeSpec] = {
    TradingMode.SCALPING: _spec(
        TradingMode.SCALPING,
        "1m",
        ("1m", "5m"),
        ("15m",),
        2,
        "seconds-to-minutes entries; a decision every closed minute",
    ),
    TradingMode.DAY_TRADING: _spec(
        TradingMode.DAY_TRADING,
        "5m",
        ("5m", "15m"),
        ("30m", "1h"),
        3,
        "intraday positions lasting minutes to hours",
    ),
    TradingMode.SWING: _spec(
        TradingMode.SWING,
        "30m",
        ("30m", "1h", "4h"),
        ("1d",),
        3,
        "multi-hour to multi-day directional positions",
    ),
}


@dataclass(frozen=True)
class Eligibility:
    """Whether a mode may be entered at all, and why not when it may not."""

    mode: TradingMode
    eligible: bool
    reason: ReasonCode | None
    unscreened: tuple[str, ...] = ()
    not_viable: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "eligible": self.eligible,
            "reason": None if self.reason is None else self.reason.value,
            "unscreened": list(self.unscreened),
            "not_viable": list(self.not_viable),
        }


def evaluate_eligibility(
    status: Mapping[str, SpecialistStatus], modes: Iterable[TradingMode] | None = None
) -> dict[TradingMode, Eligibility]:
    """Which modes the committed evidence supports entering.

    A mode is eligible only when **every** specialist it names has been screened
    and found viable. A clock absent from ``status`` counts as unscreened, which
    is the conservative reading: a specialist nobody measured is not a
    specialist that passed.

    Ordering matters in the reason: an unscreened specialist is a different
    problem from one that was measured and failed, and collapsing them would hide
    which of the two a mode is waiting on.
    """
    wanted = list(modes) if modes is not None else list(MODE_SPECS)
    result: dict[TradingMode, Eligibility] = {}
    for mode in wanted:
        if mode is TradingMode.FLAT:
            result[mode] = Eligibility(mode, True, None)
            continue
        spec = MODE_SPECS[mode]
        unscreened = tuple(
            clock
            for clock in spec.specialists
            if not status.get(clock, _absent(clock)).screened
        )
        not_viable = tuple(
            clock
            for clock in spec.specialists
            if status.get(clock, _absent(clock)).screened and not status[clock].viable
        )
        if unscreened:
            reason = ReasonCode.SPECIALIST_UNSCREENED
        elif not_viable:
            reason = ReasonCode.SPECIALIST_NOT_VIABLE
        else:
            reason = None
        result[mode] = Eligibility(
            mode, reason is None, reason, unscreened=unscreened, not_viable=not_viable
        )
    return result


def _absent(clock: str) -> SpecialistStatus:
    return SpecialistStatus(clock=clock, screened=False, viable=False)


@dataclass(frozen=True)
class ModeDecision:
    """What the controller decided, and everything a reader needs to check it.

    Every field is bounded: a mode, a signal, a reason code, a confidence bucket,
    a sorted list of mode names. None of them is free text and none grows with
    traffic, so all of them are safe as metric labels.
    """

    mode: TradingMode
    signal: Signal
    reason: ReasonCode
    eligible_modes: tuple[TradingMode, ...]
    consensus_state: dict[str, object] = field(default_factory=dict)

    @property
    def is_flat(self) -> bool:
        return self.mode is TradingMode.FLAT or self.signal is Signal.HOLD

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "signal": self.signal.value,
            "reason": self.reason.value,
            "eligible_modes": [mode.value for mode in self.eligible_modes],
            "consensus_state": dict(self.consensus_state),
        }


def decide_mode(
    declared: TradingMode | None,
    actions: Mapping[str, Signal | None],
    eligibility: Mapping[TradingMode, Eligibility],
) -> ModeDecision:
    """The operator's declared mode, honoured only if the evidence supports it.

    ``declared`` is an operator's standing choice, not a run-time selection.
    There is deliberately no ``AUTO``: choosing automatically between eligible
    modes is `P8`, which is preregistered and **not opened**, and a scaffold that
    quietly did it anyway would be running an unopened checkpoint in production.

    Returns `FLAT` — with a reason — whenever the declared mode is absent,
    ineligible, or eligible but without agreement. `FLAT` is a successful
    outcome; there is no error path here for "did not trade".
    """
    eligible = tuple(
        sorted(
            (
                mode
                for mode, row in eligibility.items()
                if row.eligible and mode is not TradingMode.FLAT
            ),
            key=lambda mode: mode.value,
        )
    )
    if declared is None or declared is TradingMode.FLAT:
        return ModeDecision(
            TradingMode.FLAT, Signal.HOLD, ReasonCode.NO_MODE_DECLARED, eligible
        )

    row = eligibility.get(declared)
    if row is None or not row.eligible:
        reason = (
            ReasonCode.SPECIALIST_UNSCREENED
            if row is None or row.reason is None
            else row.reason
        )
        return ModeDecision(TradingMode.FLAT, Signal.HOLD, reason, eligible)

    spec = MODE_SPECS[declared]
    missing = [clock for clock in spec.specialists if actions.get(clock) is None]
    if missing:
        return ModeDecision(
            TradingMode.FLAT,
            Signal.HOLD,
            ReasonCode.SPECIALIST_UNAVAILABLE,
            eligible,
            {"unavailable": sorted(missing)},
        )

    signal = consensus_decide(actions, spec.rule)
    reason = {
        Signal.LONG: ReasonCode.CONSENSUS_LONG,
        Signal.SHORT: ReasonCode.CONSENSUS_SHORT,
        Signal.HOLD: ReasonCode.NO_CONSENSUS,
    }[signal]
    state = {
        "long_votes": sum(1 for clock in spec.specialists if actions[clock] is Signal.LONG),
        "short_votes": sum(1 for clock in spec.specialists if actions[clock] is Signal.SHORT),
        "agreement_required": spec.rule.agreement_required,
        "veto": spec.rule.veto_specialist,
    }
    return ModeDecision(declared, signal, reason, eligible, state)


@dataclass(frozen=True)
class TransitionPlan:
    """How to get from one mode to another without inheriting a position.

    v1 runs **one** directional mode at a time. A mode change with an open
    position is therefore not a re-target — the new mode's horizon and cadence
    are different, so the position it would inherit is one it never chose. The
    plan flattens first and requires reconciliation before the new mode may act.
    """

    from_mode: TradingMode
    to_mode: TradingMode
    must_flatten: bool
    must_reconcile: bool
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "from_mode": self.from_mode.value,
            "to_mode": self.to_mode.value,
            "must_flatten": self.must_flatten,
            "must_reconcile": self.must_reconcile,
            "note": self.note,
        }


def plan_mode_transition(
    from_mode: TradingMode, to_mode: TradingMode, *, position_is_flat: bool
) -> TransitionPlan:
    """The deterministic transition. Same inputs, same plan, always."""
    if from_mode is to_mode:
        return TransitionPlan(from_mode, to_mode, False, False, "no mode change")
    if position_is_flat:
        return TransitionPlan(
            from_mode, to_mode, False, False, "mode changed while flat; nothing to unwind"
        )
    return TransitionPlan(
        from_mode,
        to_mode,
        True,
        True,
        "a position opened under one mode's horizon and cadence is not a position the "
        "next mode chose; flatten and reconcile before it may act",
    )


#: Words that would mean this module had started selecting on outcomes. The list
#: is short and specific on purpose: it is a tripwire for the one property the
#: whole scaffold rests on, not a general style check.
PROFIT_TOKENS: tuple[str, ...] = (
    "pnl",
    "profit",
    "realised_return",
    "realized_return",
    "net_return",
    "equity_curve",
    "backtest",
    "sharpe",
    "win_rate",
    "drawdown",
)


def selection_source() -> str:
    """The text of every function that can influence which mode is entered.

    Deliberately these three and not the whole module: the tripwire below names
    the very words it is looking for, and a scan of the file that contained the
    list would only ever find the list. Read through `inspect` so that a new
    branch added to any of them is covered without anyone updating a path.
    """
    import inspect

    return "\n".join(
        inspect.getsource(function)
        for function in (evaluate_eligibility, decide_mode, plan_mode_transition)
    )


def assert_no_profit_input(source: str) -> None:
    """Refuse selection code that would choose a mode by how it has been doing.

    Applied by the test suite to :func:`selection_source`. A mode selected on
    recent profit is the failure this scaffold exists to make impossible, and the
    cheapest place to catch it is before it is written.
    """
    lowered = source.lower()
    found = sorted({token for token in PROFIT_TOKENS if token in lowered})
    if found:
        raise ModeError(
            f"mode selection may not read {found}: choosing a mode by how it has recently "
            "performed is exactly the selection P8 exists to test and has not tested"
        )
