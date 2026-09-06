"""R2 and R3: shadow rules. They observe, and they never size a position.

Section 12.3 line 1439: "`FrozenLogisticRule`, `DailyMomentumRule` (signal-only;
parameters frozen by S2)". Section 8.1's RULE_EVALUATION row: "shadow rules
produce `SignalOnly`".

The signal-only guarantee is carried by the TYPE, not by a convention. Both
rules return a `SignalOnly`, `HedgedPosition.plan` accepts a `HedgeTarget` and
nothing else, and `RuleDecision.is_actionable` is false for them. A later edit
that tried to let a shadow rule trade would have to change a type signature,
which a reviewer sees, rather than a comparison, which a reviewer might not.

**No coefficient is chosen in this file.** Section 17's S2 says R2 and R3 are
"fitted once on research-visible gen2 data before the campaign and never
refitted", and that fit belongs to PR-14. Section 17's S3 STOP list forbids any
rule parameter chosen from recorded data. So both rules refuse to be
constructed without explicit coefficients, exactly as `CarryRule` does, and
a campaign config that omits them does not run.

Section 17's S3 STOP also permits dropping the shadow rules entirely if they
would delay the campaign; they are implemented here because the runner has to
prove it can evaluate a rule whose output must never reach an executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

from chimera.demo.rules import RuleDecision, RuleError, SignalOnly, canonical_hash

__all__ = ["FrozenLogisticRule", "DailyMomentumRule", "ShadowParams"]

_ZERO = Decimal("0")


@dataclass(frozen=True)
class ShadowParams:
    """Coefficients for a shadow rule. Required; never defaulted here."""

    coefficients: tuple[Decimal, ...]
    intercept: Decimal
    threshold: Decimal

    @classmethod
    def from_config(cls, rule_id: str, params: Mapping[str, Any]) -> "ShadowParams":
        missing = [
            k for k in ("coefficients", "intercept", "threshold") if params.get(k) is None
        ]
        if missing:
            raise RuleError(
                f"{rule_id} needs {', '.join(missing)} from the campaign config. Shadow "
                "rule coefficients are fitted once by the S2 protocol and never chosen "
                "here; a default in this file would be a parameter picked after the fact"
            )
        raw: Sequence[Any] = params["coefficients"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            raise RuleError(f"{rule_id}: coefficients must be a non-empty list")
        try:
            return cls(
                coefficients=tuple(Decimal(str(v)) for v in raw),
                intercept=Decimal(str(params["intercept"])),
                threshold=Decimal(str(params["threshold"])),
            )
        except (ArithmeticError, ValueError) as exc:
            raise RuleError(f"{rule_id} parameters are not all numbers: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": [str(c) for c in self.coefficients],
            "intercept": str(self.intercept),
            "threshold": str(self.threshold),
        }


class _ShadowRule:
    """Shared machinery. Subclasses supply the features and the identity."""

    actionable = False
    version = "1.0.0"
    rule_id = "shadow"
    _decides = ""
    _features: tuple[str, ...] = ()

    def __init__(self, params: ShadowParams) -> None:
        if len(params.coefficients) != len(self._features):
            raise RuleError(
                f"{self.rule_id} has {len(self._features)} features "
                f"({', '.join(self._features)}) but was given "
                f"{len(params.coefficients)} coefficients"
            )
        self.params = params

    def rule_hash(self) -> str:
        return canonical_hash(
            {
                "id": self.rule_id,
                "version": self.version,
                "decides": self._decides,
                "features": list(self._features),
                "actionable": False,
            }
        )

    def params_hash(self) -> str:
        return canonical_hash(self.params.to_dict())

    def features(self, state: Any, portfolio: Mapping[str, Any]) -> tuple[Decimal, ...]:
        raise NotImplementedError

    def evaluate(self, state: Any, portfolio: Mapping[str, Any]) -> RuleDecision:
        inputs_hash = canonical_hash(state.canonical())
        if not state.complete:
            raise RuleError(
                f"{self.rule_id} was asked to decide from an incomplete minute "
                f"({', '.join(state.missing)}); no rule evaluates on one"
            )
        values = self.features(state, portfolio)
        score = self.params.intercept + sum(
            (c * v for c, v in zip(self.params.coefficients, values)), _ZERO
        )
        fires = score >= self.params.threshold
        return RuleDecision(
            rule_id=self.rule_id,
            rule_hash=self.rule_hash(),
            target=SignalOnly(Decimal("1") if fires else _ZERO, note=self._decides),
            reason="score at or above threshold" if fires else "score below threshold",
            inputs_hash=inputs_hash,
            params_hash=self.params_hash(),
            version=self.version,
            detail={"score": str(score), "threshold": str(self.params.threshold)},
        )


class FrozenLogisticRule(_ShadowRule):
    """R2: a fixed linear score over the minute's own observable quantities.

    "Frozen" is the whole point: the coefficients arrive from config and this
    class cannot fit, refit, or adapt them. There is no training path in this
    module and no state that survives a call.
    """

    rule_id = "R2_frozen_logistic"
    _decides = "signal-only score over basis and spread"
    _features = ("basis", "spot_spread", "perp_spread")

    def features(self, state: Any, portfolio: Mapping[str, Any]) -> tuple[Decimal, ...]:
        def spread(book: Mapping[str, Any]) -> Decimal:
            ask, bid = book.get("ask"), book.get("bid")
            return (ask - bid) if (ask is not None and bid is not None) else _ZERO

        return (state.basis or _ZERO, spread(state.book_spot), spread(state.book_perp))


class DailyMomentumRule(_ShadowRule):
    """R3: a signal-only read of the minute's own candle.

    Section 8.1 evaluates rules on the minute the feed just delivered, and a
    rule "holds no hidden state beyond what the log records" (section 2.2), so
    this reads the candle in front of it rather than accumulating a window
    across ticks. A windowed version would need persisted state, which is PR-14's
    to specify rather than this PR's to invent.
    """

    rule_id = "R3_daily_momentum"
    _decides = "signal-only read of the minute's own candle"
    _features = ("perp_return", "spot_return")

    def features(self, state: Any, portfolio: Mapping[str, Any]) -> tuple[Decimal, ...]:
        def move(ohlcv: Mapping[str, Decimal | None]) -> Decimal:
            open_, close = ohlcv.get("open"), ohlcv.get("close")
            if open_ is None or close is None or open_ == _ZERO:
                return _ZERO
            return (close - open_) / open_

        return (move(state.perp_ohlcv), move(state.spot_ohlcv))
