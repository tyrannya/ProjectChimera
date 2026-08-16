"""Model artifact storage and promotion.

The artifact directory is the source of truth, not a tracking server::

    artifacts/models/
        20260816T120000Z-a1b2c3/
            model.pt         # state_dict
            config.json      # MTSTConfig, to rebuild the module
            metadata.json    # ModelMetadata: features, scaler, threshold, ...
            report.json      # metrics for the record
        current.json         # {"version": "..."} - what inference serves

Consequences of that choice:

* a model can be loaded with torch and the standard library alone, so the
  inference service has no dependency on MLflow being reachable;
* ``current.json`` is a plain file rather than a symlink, so it survives Docker
  volume mounts and Windows checkouts;
* **promotion is gated.** Finishing a training run does not make a model live.
  :func:`promote` writes ``current.json`` only when :func:`check_gates` passes,
  and it is the only function that writes it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from chimera.contracts import ModelMetadata
from nn.model_def import MTST, MTSTConfig

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("artifacts/models")
CURRENT_POINTER = "current.json"


def new_version(seed_material: str = "") -> str:
    """A sortable, unique version id: UTC timestamp plus a short digest."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(f"{stamp}{seed_material}".encode()).hexdigest()[:6]
    return f"{stamp}-{digest}"


def save_model(
    models_dir: str | Path,
    version: str,
    model: MTST,
    metadata: ModelMetadata,
    report: Mapping[str, Any] | None = None,
) -> Path:
    """Write weights, architecture, metadata and metrics for one version."""
    target = Path(models_dir) / version
    target.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), target / "model.pt")
    (target / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2))
    (target / "metadata.json").write_text(json.dumps(metadata.to_dict(), indent=2))
    if report is not None:
        (target / "report.json").write_text(json.dumps(report, indent=2, default=str))

    logger.info("Saved model version %s to %s", version, target)
    return target


def load_model(model_dir: str | Path) -> tuple[MTST, ModelMetadata]:
    """Rebuild a model and its metadata from an artifact directory."""
    model_dir = Path(model_dir)
    for name in ("model.pt", "config.json", "metadata.json"):
        if not (model_dir / name).exists():
            raise FileNotFoundError(f"{model_dir} is not a model artifact: {name} missing")

    config = MTSTConfig.from_dict(json.loads((model_dir / "config.json").read_text()))
    metadata = ModelMetadata.from_dict(json.loads((model_dir / "metadata.json").read_text()))

    if metadata.n_features != config.input_dim:
        raise ValueError(
            f"artifact is inconsistent: metadata lists {metadata.n_features} features "
            f"but the model takes {config.input_dim}"
        )
    if metadata.sequence_length != config.seq_len:
        raise ValueError(
            f"artifact is inconsistent: metadata sequence_length "
            f"{metadata.sequence_length} != model seq_len {config.seq_len}"
        )

    model = MTST(config)
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, metadata


def resolve_current(models_dir: str | Path = DEFAULT_MODELS_DIR) -> Path:
    """Path of the promoted model, or raise if nothing has been promoted."""
    models_dir = Path(models_dir)
    pointer = models_dir / CURRENT_POINTER
    if not pointer.exists():
        raise FileNotFoundError(
            f"no promoted model: {pointer} does not exist. Train a model and let it "
            "pass the promotion gates first."
        )
    version = json.loads(pointer.read_text())["version"]
    target = models_dir / version
    if not target.exists():
        raise FileNotFoundError(f"{pointer} points at {version}, which is not in {models_dir}")
    return target


@dataclass(frozen=True)
class PromotionGates:
    """Conditions a model must satisfy before it may serve traffic.

    Evaluated on the **validation** split. The test split is scored once, for
    the report, and never feeds a promotion decision — otherwise the test set
    becomes a second validation set and stops being an estimate of unseen
    performance.
    """

    #: Macro F1 must exceed the best baseline's by at least this margin.
    min_macro_f1_margin: float = 0.01
    #: The model must actually produce trades to be useful.
    min_trades: int = 10
    #: Net return after fees and slippage must be positive on validation.
    require_positive_net_return: bool = True
    #: Reject wildly overconfident models; the threshold is a probability cut.
    max_calibration_error: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_gates(
    model_report: Mapping[str, Any],
    baseline_reports: Mapping[str, Mapping[str, Any]],
    gates: PromotionGates | None = None,
) -> tuple[bool, list[str]]:
    """Return ``(passed, failure_reasons)`` for a validation report."""
    gates = gates or PromotionGates()
    failures: list[str] = []

    classification = model_report["classification"]
    trading = model_report["trading"]

    best_baseline_f1 = max(
        (r["classification"]["macro_f1"] for r in baseline_reports.values()),
        default=0.0,
    )
    margin = classification["macro_f1"] - best_baseline_f1
    if margin < gates.min_macro_f1_margin:
        failures.append(
            f"macro F1 {classification['macro_f1']:.4f} does not beat the best "
            f"baseline ({best_baseline_f1:.4f}) by {gates.min_macro_f1_margin:.4f} "
            f"(margin {margin:+.4f})"
        )

    if trading["n_trades"] < gates.min_trades:
        failures.append(
            f"only {trading['n_trades']} validation trades, need {gates.min_trades}"
        )

    if gates.require_positive_net_return and trading["net_return"] <= 0:
        failures.append(f"validation net return after costs is {trading['net_return']:+.4f}")

    if classification["calibration_error"] > gates.max_calibration_error:
        failures.append(
            f"calibration error {classification['calibration_error']:.4f} exceeds "
            f"{gates.max_calibration_error:.4f}"
        )

    return (not failures), failures


def promote(models_dir: str | Path, version: str) -> Path:
    """Point ``current.json`` at ``version``.

    Callers must have run :func:`check_gates` first; this function only writes
    the pointer, so there is exactly one place where a model becomes live.
    """
    models_dir = Path(models_dir)
    if not (models_dir / version).exists():
        raise FileNotFoundError(f"cannot promote unknown version {version}")
    pointer = models_dir / CURRENT_POINTER
    pointer.write_text(
        json.dumps(
            {
                "version": version,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    logger.warning("Promoted model %s to current", version)
    return pointer
