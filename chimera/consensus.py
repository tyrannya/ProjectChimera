"""Pythia's cross-timeframe consensus: agreement between temporal specialists.

One pure function and one frozen rule. No I/O, no clock, no data access, no
randomness — given the same specialist actions it returns the same signal, which
is what lets the research replay in :mod:`nn.p7` and the live trading-mode
controller share a decision rather than each implement one.

**What a "specialist" is here.** A model fitted on one timeframe's own bars,
emitting one of :class:`chimera.contracts.Signal` per decision. This module does
not know how it was fitted, what it read, or whether it is any good; it knows
what it said. ``None`` means the specialist has nothing to say yet — its bar has
not closed — and is not the same as ``HOLD``.

**What this module is not.** It is not a strategy, it does not size a position,
it does not decide whether to trade, and it never speaks to a venue. Aegis still
holds every veto that matters and Hermes still routes every order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from chimera.contracts import Signal

__all__ = ["ConsensusRule", "ConsensusError", "decide", "explain"]


class ConsensusError(ValueError):
    """A consensus rule cannot mean what it has been asked to mean."""


@dataclass(frozen=True)
class ConsensusRule:
    """A counted vote among named specialists, with one of them holding a veto.

    ``agreement_required`` must be a strict majority. That is not a stylistic
    preference: at or below half, LONG and SHORT could both reach the threshold
    on the same row and the rule would need a tie-break, which is exactly the
    kind of discretionary knob a preregistered consensus may not carry.
    """

    #: What this rule is for. Free text; carried into telemetry as a bounded
    #: enum by the caller, never as a label built from user input.
    mode: str
    #: The timeframe a decision is taken on.
    decision_clock: str
    #: Every specialist that votes, in the order the design names them.
    specialists: tuple[str, ...]
    #: The slow member. It cannot force a decision, only block the opposite one.
    veto_specialist: str
    #: How many specialists must agree. A strict majority of ``specialists``.
    agreement_required: int

    def __post_init__(self) -> None:
        if not self.specialists:
            raise ConsensusError(f"{self.mode}: a consensus needs at least one specialist")
        if len(set(self.specialists)) != len(self.specialists):
            raise ConsensusError(f"{self.mode}: a specialist may not vote twice")
        if self.veto_specialist not in self.specialists:
            raise ConsensusError(
                f"{self.mode}: the veto specialist {self.veto_specialist!r} does not vote; "
                f"the voters are {list(self.specialists)}"
            )
        if self.decision_clock not in self.specialists:
            raise ConsensusError(
                f"{self.mode}: the decision clock {self.decision_clock!r} is not among the "
                "specialists, so the mode would trade a timeframe nothing in it reads"
            )
        total = len(self.specialists)
        if not (total // 2 < self.agreement_required <= total):
            raise ConsensusError(
                f"{self.mode}: agreement_required={self.agreement_required} of {total} is "
                "not a strict majority. At or below half, LONG and SHORT could both reach "
                "the threshold on one row and the rule would need a tie-break."
            )


def decide(actions: Mapping[str, Signal | None], rule: ConsensusRule) -> Signal:
    """The consensus signal for one decision row.

    * every specialist the rule names must be present, and none may be ``None``
      — a missing specialist yields ``HOLD`` rather than a vote among whoever
      happens to have spoken. A partial vote would let the rule reach its
      threshold on the members present, which makes the outcome a function of
      where a block starts;
    * ``LONG`` when at least ``agreement_required`` specialists are actively
      LONG **and** the veto specialist is not actively SHORT;
    * ``SHORT`` symmetrically;
    * ``HOLD`` otherwise.

    ``HOLD`` is neither a vote nor a veto: a specialist saying HOLD does not
    count towards agreement and does not block anything.
    """
    votes: list[Signal] = []
    for name in rule.specialists:
        action = actions.get(name)
        if action is None:
            return Signal.HOLD
        votes.append(action)

    veto = actions[rule.veto_specialist]
    longs = sum(1 for vote in votes if vote is Signal.LONG)
    shorts = sum(1 for vote in votes if vote is Signal.SHORT)

    # The two branches are mutually exclusive because `agreement_required` is a
    # strict majority, checked when the rule was built. The order below is
    # therefore not a tie-break and cannot become one.
    if longs >= rule.agreement_required and veto is not Signal.SHORT:
        return Signal.LONG
    if shorts >= rule.agreement_required and veto is not Signal.LONG:
        return Signal.SHORT
    return Signal.HOLD


def explain(actions: Mapping[str, Signal | None], rule: ConsensusRule) -> dict[str, object]:
    """The same decision plus why, for telemetry and for a reader.

    Every field is a bounded value — a signal, a count, a boolean — so a caller
    can label a metric with it without turning traffic into time series.
    """
    missing = [name for name in rule.specialists if actions.get(name) is None]
    votes = [actions.get(name) for name in rule.specialists]
    longs = sum(1 for vote in votes if vote is Signal.LONG)
    shorts = sum(1 for vote in votes if vote is Signal.SHORT)
    veto = actions.get(rule.veto_specialist)
    signal = decide(actions, rule)
    return {
        "mode": rule.mode,
        "signal": signal.value,
        "long_votes": longs,
        "short_votes": shorts,
        "hold_votes": len(rule.specialists) - longs - shorts - len(missing),
        "unavailable": len(missing),
        "veto_signal": None if veto is None else veto.value,
        "veto_blocked": bool(
            not missing
            and (
                (longs >= rule.agreement_required and veto is Signal.SHORT)
                or (shorts >= rule.agreement_required and veto is Signal.LONG)
            )
        ),
        "agreement_reached": bool(
            not missing
            and (longs >= rule.agreement_required or shorts >= rule.agreement_required)
        ),
    }
