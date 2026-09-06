"""R1, the carry rule. Parameterised; no parameter is chosen in this file.

Section 12.3 line 1438: "CarryRule (R1) parameters and evaluate; parameters
frozen by the S2 protocol". Section 14's EXACT SCOPE for PR-10 says the same in
fewer words: "`rules.py` with `CarryRule` parameterised (values from config)".

**Why there are no defaults here.** The S2 protocol that freezes R1's
parameters is PR-14, which lands after this PR, and section 17's S3 STOP list
forbids "any rule parameter chosen from recorded data". A default written into
this module would be a parameter chosen by its author -- before the protocol
that is supposed to freeze it, and with the recorded data already on disk. So
`CarryRule` refuses to be constructed without explicit values, and
`chimera/demo/config.py` is the only place they can come from. A campaign whose
config omits them does not run.

The rule is deliberately dull. It says: hold a hedged position of the sized
quantity while the basis is at least ``min_basis``, and hold nothing otherwise.
It is the smallest rule that exercises every path in section 8.1's tick loop,
which is what PR-10 needs from it. No claim is made that it earns anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from chimera.carry.accounting import Allocation, Costs, Quote, Venue, hedge_quantity
from chimera.demo.rules import (
    HedgeTarget,
    RuleDecision,
    RuleError,
    canonical_hash,
)

__all__ = ["CarryRule", "CarryParams", "RULE_ID"]

RULE_ID = "R1_carry"
_VERSION = "1.0.0"


@dataclass(frozen=True)
class CarryParams:
    """R1's parameters. Every one is required; none has a default here.

    ``spot_leg_fraction`` splits the deployed capital between the two legs.
    `Allocation` refuses ``spot + perp > total_capital``, calling that "leverage
    introduced by arithmetic", so the split is a real parameter rather than an
    implementation detail and is named in the config like the rest.

    The four cost rates are NOT rule parameters in the same sense -- they are
    the venue's published values from section 5.3 -- but the sizing arithmetic
    needs them, and reading them from config rather than hard-coding them here
    keeps this module free of any number an author chose.
    """

    min_basis: Decimal
    max_notional_fraction: Decimal
    step_size: Decimal
    min_notional: Decimal
    spot_leg_fraction: Decimal
    spot_fee_rate: Decimal
    perp_fee_rate: Decimal
    slippage_rate: Decimal

    @classmethod
    def from_config(cls, params: Mapping[str, Any]) -> "CarryParams":
        """Read the four values from a config block, refusing anything missing.

        The error names the missing key rather than substituting a value,
        because a silently defaulted parameter is exactly the failure mode
        section 17's STOP list is written against.
        """
        required = (
            "min_basis",
            "max_notional_fraction",
            "step_size",
            "min_notional",
            "spot_leg_fraction",
            "spot_fee_rate",
            "perp_fee_rate",
            "slippage_rate",
        )
        absent = [key for key in required if params.get(key) is None]
        if absent:
            raise RuleError(
                f"{RULE_ID} needs {', '.join(absent)} from the campaign config. This rule "
                "has no defaults: its parameters are frozen by the S2 protocol, and a "
                "value chosen here would be a parameter chosen after the data was recorded"
            )
        try:
            return cls(**{key: Decimal(str(params[key])) for key in required})
        except (ArithmeticError, ValueError) as exc:
            raise RuleError(f"{RULE_ID} parameters are not all numbers: {exc}") from exc

    def to_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in sorted(self.__dataclass_fields__)}


class CarryRule:
    """R1: hold the hedge while the basis is wide enough, otherwise hold nothing."""

    actionable = True

    def __init__(self, params: CarryParams) -> None:
        self.params = params
        self.rule_id = RULE_ID
        self.version = _VERSION

    # --- identity ---------------------------------------------------------
    def rule_hash(self) -> str:
        """Identity of the procedure, not of its parameters.

        Deliberately a declaration of what the rule does rather than a hash of
        this file's bytes: reformatting the module must not look like a changed
        decision rule, and a changed rule must not hide behind unchanged bytes.
        """
        return canonical_hash(
            {
                "id": RULE_ID,
                "version": _VERSION,
                "decides": "hedged carry target from basis",
                "inputs": ["perp_close", "spot_close", "book_perp", "book_spot", "equity"],
                "sizing": "chimera.carry.accounting.hedge_quantity",
            }
        )

    def params_hash(self) -> str:
        return canonical_hash(self.params.to_dict())

    # --- the decision -----------------------------------------------------
    def evaluate(self, state: Any, portfolio: Mapping[str, Any]) -> RuleDecision:
        inputs_hash = canonical_hash(state.canonical())

        if not state.complete:
            raise RuleError(
                f"{RULE_ID} was asked to decide from an incomplete minute "
                f"({', '.join(state.missing)}). Section 2.2 stops before this: an "
                "incomplete state is logged as INCOMPLETE_STATE and no rule evaluates"
            )

        basis = state.basis
        if basis is None:
            raise RuleError(f"{RULE_ID}: a complete minute with no basis is a contradiction")

        equity = Decimal(str(portfolio.get("equity", "0")))
        if basis < self.params.min_basis or equity <= 0:
            return self._decide(
                HedgeTarget(Decimal("0")),
                reason=(
                    "basis below min_basis"
                    if basis < self.params.min_basis
                    else "no equity to allocate"
                ),
                inputs_hash=inputs_hash,
                detail={"basis": str(basis), "min_basis": str(self.params.min_basis)},
            )

        deployed = equity * self.params.max_notional_fraction
        spot_share = deployed * self.params.spot_leg_fraction
        allocation = Allocation(
            total_capital=equity,
            spot=spot_share,
            perp=deployed - spot_share,
        )
        # `Quote` separates the MARK from the FILL, and the two are different
        # prices here. `spot`/`perp` are the minute's closes and drive `basis`;
        # `spot_open`/`perp_open` are what `spot_fill`/`perp_fill` return, and
        # `hedge_quantity` sizes against those.
        #
        # The ported arithmetic calls those fields "opens" because P13 fills at
        # the next candle's OPEN. The demo does not: it fills against the
        # recorded book through `RecordedQuoteFillModel`, crossing the spread.
        # So the CROSSED BOOK is what goes in the fill fields -- the spot leg
        # buys at the ask, the perpetual leg sells at the bid. Sizing against
        # the mid instead would ask the venue for a quantity the allocation
        # cannot actually fund once the spread is paid.
        entry = Quote(
            instant_ns=state.minute_ns,
            spot=state.spot_close,
            perp=state.perp_close,
            mark=state.mark,
            spot_open=state.book_spot.get("ask") or state.spot_close,
            perp_open=state.book_perp.get("bid") or state.perp_close,
        )
        costs = Costs(
            spot_fee=self.params.spot_fee_rate,
            spot_slippage=self.params.slippage_rate,
            perp_fee=self.params.perp_fee_rate,
            perp_slippage=self.params.slippage_rate,
        )
        venue = Venue(
            step_size=self.params.step_size,
            min_notional=self.params.min_notional,
            maintenance_margin_rate=Decimal("0.004"),
        )
        quantity = hedge_quantity(entry, allocation, costs, venue)
        if quantity * entry.spot_fill < self.params.min_notional:
            return self._decide(
                HedgeTarget(Decimal("0")),
                reason="sized below the venue minimum notional",
                inputs_hash=inputs_hash,
                detail={"basis": str(basis), "sized": str(quantity)},
            )
        return self._decide(
            HedgeTarget(quantity),
            reason="basis at or above min_basis",
            inputs_hash=inputs_hash,
            detail={"basis": str(basis), "min_basis": str(self.params.min_basis)},
        )

    def _decide(
        self,
        target: HedgeTarget,
        *,
        reason: str,
        inputs_hash: str,
        detail: Mapping[str, Any],
    ) -> RuleDecision:
        return RuleDecision(
            rule_id=RULE_ID,
            rule_hash=self.rule_hash(),
            target=target,
            reason=reason,
            inputs_hash=inputs_hash,
            params_hash=self.params_hash(),
            version=_VERSION,
            detail=dict(detail),
        )
