"""Shared semantics between the model, the inference service and the strategy.

The single most important rule in this file: **the model's output and the
trading decision use the same units.** The model emits class probabilities over
{SHORT, HOLD, LONG}, where the classes are defined by a *cost-aware* rule on
future return. The strategy therefore compares a probability against a
probability threshold — never a price against a probability, which is what the
original code did.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from chimera.features import FeatureSpec


class Signal(str, Enum):
    """Trading signal. The string values are the wire format."""

    SHORT = "SHORT"
    HOLD = "HOLD"
    LONG = "LONG"


#: Class index order used by the model head and by every probability vector on
#: the wire. Fixed forever: metadata records it, tests assert it.
CLASS_ORDER: tuple[Signal, ...] = (Signal.SHORT, Signal.HOLD, Signal.LONG)

SHORT_IDX, HOLD_IDX, LONG_IDX = 0, 1, 2


@dataclass(frozen=True)
class TargetSpec:
    """Definition of what the model predicts.

    The label for candle ``t`` is derived from the return realised over the
    next ``horizon`` candles::

        future_return(t) = close[t + horizon] / close[t] - 1

        LONG   if future_return >  cost_threshold
        SHORT  if future_return < -cost_threshold
        HOLD   otherwise

    ``cost_threshold`` is a round-trip cost estimate, not a tuning knob. A move
    smaller than the cost of capturing it is not a trading opportunity, so
    labelling it HOLD is the honest choice: it stops the model from learning to
    chase noise that would lose money after fees.
    """

    #: Candles ahead the label looks. At 1h candles the default is 6 hours.
    horizon: int = 6
    #: Taker fee per side, as a fraction. 5 bps is a common retail spot taker fee.
    fee_rate: float = 0.0005
    #: Conservative per-side slippage allowance, as a fraction.
    slippage_rate: float = 0.0005

    @property
    def cost_threshold(self) -> float:
        """Round-trip cost: entry and exit, each paying fee plus slippage."""
        return 2.0 * (self.fee_rate + self.slippage_rate)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cost_threshold"] = self.cost_threshold
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetSpec":
        return cls(
            horizon=int(data["horizon"]),
            fee_rate=float(data["fee_rate"]),
            slippage_rate=float(data["slippage_rate"]),
        )


@dataclass
class ModelMetadata:
    """Everything needed to reproduce the preprocessing an artifact expects.

    Written next to the weights. The inference service refuses to serve a model
    whose metadata it cannot read, and refuses a request whose feature matrix
    disagrees with ``feature_names``/``sequence_length``.
    """

    model_version: str
    #: Ordered feature names. Request width and order must match exactly.
    feature_names: list[str]
    sequence_length: int
    feature_spec: FeatureSpec
    target_spec: TargetSpec
    #: Per-feature standardisation parameters, fitted on the training split only.
    scaler_mean: list[float]
    scaler_std: list[float]
    #: Minimum probability of the winning non-HOLD class before the strategy
    #: acts. Selected on the validation split, never on test.
    decision_threshold: float
    class_order: list[str] = field(default_factory=lambda: [c.value for c in CLASS_ORDER])
    trained_at: str = ""
    dataset_start: str = ""
    dataset_end: str = ""
    #: Last candle in the training split — the newest data the *weights* saw.
    train_end: str = ""
    #: Last candle in the validation split. Early stopping and the decision
    #: threshold were fitted on validation, so this — not ``train_end`` — is the
    #: point after which data is genuinely unseen. See :attr:`training_cutoff`.
    validation_end: str = ""
    exchange: str = ""
    pair: str = ""
    timeframe: str = ""
    #: Human-readable id of the ``nn.research_contract`` generation this artifact
    #: was trained under. Empty for artifacts that predate research contracts,
    #: which callers must read as "unrecorded", never as "the current one".
    research_contract_id: str = ""
    #: SHA-256 of that contract's research-defining content. The id can be
    #: reused while the contract behind it changes, so this — not the id — is
    #: what identifies the exact generation.
    research_contract_hash: str = ""
    #: Metrics recorded at promotion time, for the record.
    validation_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def training_cutoff(self) -> str:
        """Newest timestamp any fitted quantity in this artifact has seen.

        Everything at or before it is in-sample: the weights were fitted on the
        training split, and the early-stopping epoch and decision threshold were
        chosen on validation. Only data strictly after this point is an honest
        out-of-sample test. Empty when the artifact predates this field, which
        callers must treat as "unknown", never as "safe".
        """
        return self.validation_end or self.train_end

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "feature_names": list(self.feature_names),
            "sequence_length": self.sequence_length,
            "feature_spec": self.feature_spec.to_dict(),
            "target_spec": self.target_spec.to_dict(),
            "scaler_mean": list(self.scaler_mean),
            "scaler_std": list(self.scaler_std),
            "decision_threshold": self.decision_threshold,
            "class_order": list(self.class_order),
            "trained_at": self.trained_at,
            "dataset_start": self.dataset_start,
            "dataset_end": self.dataset_end,
            "train_end": self.train_end,
            "validation_end": self.validation_end,
            "exchange": self.exchange,
            "pair": self.pair,
            "timeframe": self.timeframe,
            "research_contract_id": self.research_contract_id,
            "research_contract_hash": self.research_contract_hash,
            "validation_metrics": self.validation_metrics,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelMetadata":
        return cls(
            model_version=str(data["model_version"]),
            feature_names=list(data["feature_names"]),
            sequence_length=int(data["sequence_length"]),
            feature_spec=FeatureSpec.from_dict(data["feature_spec"]),
            target_spec=TargetSpec.from_dict(data["target_spec"]),
            scaler_mean=[float(x) for x in data["scaler_mean"]],
            scaler_std=[float(x) for x in data["scaler_std"]],
            decision_threshold=float(data["decision_threshold"]),
            class_order=list(data.get("class_order", [c.value for c in CLASS_ORDER])),
            trained_at=str(data.get("trained_at", "")),
            dataset_start=str(data.get("dataset_start", "")),
            dataset_end=str(data.get("dataset_end", "")),
            train_end=str(data.get("train_end", "")),
            validation_end=str(data.get("validation_end", "")),
            exchange=str(data.get("exchange", "")),
            pair=str(data.get("pair", "")),
            timeframe=str(data.get("timeframe", "")),
            research_contract_id=str(data.get("research_contract_id", "")),
            research_contract_hash=str(data.get("research_contract_hash", "")),
            validation_metrics=dict(data.get("validation_metrics", {})),
        )


def decide(probabilities: Mapping[str, float], threshold: float) -> Signal:
    """Map a probability vector to a signal.

    Fails closed: anything that is not a confident SHORT or LONG is HOLD. In
    particular a model that is merely *less* unsure about LONG than about HOLD
    does not produce a trade — the winning class must clear ``threshold`` in
    absolute terms.
    """
    p_short = float(probabilities.get(Signal.SHORT.value, 0.0))
    p_long = float(probabilities.get(Signal.LONG.value, 0.0))
    if p_long >= threshold and p_long > p_short:
        return Signal.LONG
    if p_short >= threshold and p_short > p_long:
        return Signal.SHORT
    return Signal.HOLD
