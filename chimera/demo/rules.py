"""The rule contract: what a rule may see, and what it may say.

Section 12.3 line 1437 fixes the five public names -- `Rule`, `RuleDecision`,
`HedgeTarget`, `SignalOnly`, `RuleRegistry` -- and section 2.2 line 121 fixes
the decision shape and two semantics that this module exists to enforce:

    "rule raises -> runner enters HALT with reason `rule_exception`; never a
    partial decision"

    "rules hold no hidden state beyond what the log records"

Both are structural here rather than advisory. A `RuleDecision` is frozen and
is only ever constructed complete, so there is no way to emit half of one; and
a rule is handed a `MarketState` and a portfolio view and nothing else, so the
only way for state to survive a minute is through the parameters the log
already records.

`HedgeTarget` is **defined in `chimera.carry.hedge`** and re-exported here.
Section 12.3 lists the name under both modules; section 13 makes PR-10 depend
on PR-08, so the dependency may only run this way, and carry is where the type
is consumed. Re-exporting keeps section 12.3's promise that
``from chimera.demo.rules import HedgeTarget`` works.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Protocol, runtime_checkable

from chimera.carry.hedge import HedgeTarget

__all__ = [
    "HedgeTarget",
    "Rule",
    "RuleDecision",
    "RuleError",
    "RuleRegistry",
    "SignalOnly",
    "canonical_hash",
]


class RuleError(RuntimeError):
    """A rule could not decide. The runner turns this into HALT, never a guess."""


def canonical_hash(payload: Mapping[str, Any]) -> str:
    """``sha256:`` over section 9.2's canonical bytes of ``payload``."""
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return "sha256:" + hashlib.sha256(body.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SignalOnly:
    """A rule's opinion that must never reach an executor.

    Section 8.1's RULE_EVALUATION row: "shadow rules produce `SignalOnly`". The
    type is the enforcement: nothing in the runner can hand one to
    `HedgedPosition.plan`, which accepts a `HedgeTarget` and nothing else, so a
    shadow rule cannot size a position by accident or by a later edit.
    """

    quantity: Decimal
    note: str = ""

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise RuleError("a signal quantity is a magnitude and cannot be negative")


@dataclass(frozen=True)
class RuleDecision:
    """One rule's complete answer for one minute.

    Section 2.2 line 121's shape exactly: ``rule_id``, ``rule_hash``, a target
    that is either a `HedgeTarget` or a `SignalOnly`, a ``reason``, and the
    ``inputs_hash`` of the state it was computed from.
    """

    rule_id: str
    rule_hash: str
    target: HedgeTarget | SignalOnly
    reason: str
    inputs_hash: str
    params_hash: str = ""
    version: str = "1.0.0"
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Whether this decision may reach an executor at all."""
        return isinstance(self.target, HedgeTarget)

    def to_record(self) -> dict[str, Any]:
        """The ``rule`` and ``signal`` blocks of a section 9.1 record."""
        quantity = (
            self.target.quantity
            if isinstance(self.target, HedgeTarget)
            else self.target.quantity
        )
        return {
            "rule": {
                "id": self.rule_id,
                "version": self.version,
                "rule_hash": self.rule_hash,
                "params_hash": self.params_hash,
            },
            "signal": {
                "target_qty": str(quantity),
                "reason": self.reason,
                "kind": "target" if self.is_actionable else "signal_only",
            },
        }


@runtime_checkable
class Rule(Protocol):
    """What the runner requires of a rule.

    ``evaluate`` is given the minute's `MarketState` and a read-only portfolio
    view. It returns a complete `RuleDecision` or raises; there is no third
    outcome, which is what "never a partial decision" means.
    """

    rule_id: str
    version: str

    def rule_hash(self) -> str:
        """Identity of the decision procedure itself."""

    def params_hash(self) -> str:
        """Identity of the parameters it was given."""

    def evaluate(self, state: Any, portfolio: Mapping[str, Any]) -> RuleDecision:
        """Decide for one minute, or raise `RuleError`."""


class RuleRegistry:
    """The rules a campaign runs, in a fixed order.

    Order is part of the evidence: the decision log records rules in the order
    they were evaluated, so a registry that iterated a set would make two runs
    over the same files produce different bytes.
    """

    def __init__(self, rules: "list[Rule] | None" = None) -> None:
        self._rules: list[Rule] = []
        for rule in rules or []:
            self.register(rule)

    def register(self, rule: Rule) -> None:
        if any(existing.rule_id == rule.rule_id for existing in self._rules):
            raise RuleError(
                f"rule id {rule.rule_id!r} is already registered. Two rules with one id "
                "would make the decision log ambiguous about which one decided"
            )
        self._rules.append(rule)

    def __iter__(self):
        return iter(tuple(self._rules))

    def __len__(self) -> int:
        return len(self._rules)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(rule.rule_id for rule in self._rules)

    def actionable(self) -> tuple[Rule, ...]:
        """Rules whose decisions may reach an executor.

        A campaign runs exactly one of these -- section 8.1 evaluates every rule
        but only R1 sizes a position -- and the runner asserts that rather than
        assuming it.
        """
        return tuple(r for r in self._rules if getattr(r, "actionable", False))
