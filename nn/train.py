"""Training entrypoint.

    python -m nn.train --dataset data/datasets/binance_BTC_USDT_1h.parquet
    python -m nn.train --dataset ... --epochs 2 --tune-trials 0   # smoke run
    python -m nn.train --dataset ... --validation-only            # research run

The pipeline, in order, with the leakage-relevant step called out at each turn:

1. seed everything (Python, NumPy, torch) — reproducible by default;
2. load the dataset built by ``tools/build_features.py``;
3. split chronologically into train / validation / test (``nn.dataset``);
4. **fit the scaler on training rows only**, then transform all three;
5. build windows so no sample straddles a split or market-data gap boundary;
6. fit the baselines on training data;
7. train, selecting the best epoch **on validation**;
8. select the decision threshold **on validation**;
9. score model and baselines on validation, then score the test split *once*;
10. save the artifact, and promote only if the validation gates pass.

Steps 4 to 8 are the *research core* — :class:`ResearchData`,
:func:`prepare_research_windows` and :func:`fit_and_validate` — which
``nn.experiment`` and ``nn.walkforward`` import rather than reimplement. Those
functions take a training and a validation split and nothing else, so no amount
of misuse can make them fit anything on test rows.

Step 9 goes through :func:`score_frozen_split`, which fits nothing: it takes an
already-frozen scaler, model, threshold and baselines and only transforms,
windows and measures. That is why ``nn.walkforward`` can point it at its outer
validation block — a block that must not influence model selection — and reuse
the evaluation path rather than a copy of it. ``main`` is the only caller that
ever hands it the test split, and ``--validation-only`` skips that call.

CPU is the default and fully supported path. GPU and mixed precision are used
only when CUDA is actually present. Ray Tune is optional and off by default:
``--tune-trials 0`` (the default) runs a single training pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chimera.contracts import CLASS_ORDER, ModelMetadata, TargetSpec
from chimera.features import FeatureSpec
from nn import evaluate as ev
from nn.baselines import (
    BASELINE_DECISION_THRESHOLD,
    MajorityClassBaseline,
    MomentumBaseline,
)
from nn.data_pipeline import DatasetMetadata, load_dataset, timeframe_to_minutes
from nn.dataset import Split, StandardScaler, build_windows, chronological_split
from nn.model_def import MTST, MTSTConfig
from nn.registry import (
    DEFAULT_MODELS_DIR,
    PromotionGates,
    check_gates,
    new_version,
    promote,
    save_model,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Makes cuBLAS reductions deterministic; harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights.

    Cost-aware labelling makes HOLD the majority class by construction. Without
    weighting, the cheapest way to reduce the loss is to always predict HOLD,
    and the model converges to the majority baseline.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_proba(
    model: MTST, X: np.ndarray, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    """Softmax probabilities for ``X`` in the fixed class order."""
    model.eval()
    if len(X) == 0:
        return np.empty((0, len(CLASS_ORDER)), dtype=np.float64)
    chunks = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).to(device)
        logits = model(batch)
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


# --- the research core -------------------------------------------------------
#
# Everything from here to `fit_and_validate` is shared by nn.train,
# nn.experiment and nn.walkforward. It exists so those three cannot drift apart
# on the feature/target contract, and so the "never touch test" rule is a
# property of the signatures rather than a habit of the callers.


@dataclass(frozen=True)
class ResearchData:
    """The arrays every research entrypoint needs, loaded once."""

    ds_meta: DatasetMetadata
    feature_names: list[str]
    feature_spec: FeatureSpec
    target_spec: TargetSpec
    features: np.ndarray
    targets: np.ndarray
    future_return: np.ndarray
    #: Close price per row. Reporting needs the price *path*, not just the
    #: horizon-ahead return: a candle-level portfolio curve and a buy-and-hold
    #: reference cannot be built from `future_return` alone.
    close: np.ndarray
    segment_ids: np.ndarray | None
    dates: np.ndarray
    candles_per_year: float

    @property
    def n_rows(self) -> int:
        return len(self.features)

    def market(self, visible: int) -> ev.MarketContext:
        """Prices and timestamps, truncated to the rows a split may see.

        Takes the same ``visible`` bound the features and targets are sliced
        with, so a market context cannot reach a row the evaluation itself is
        not allowed to reach.
        """
        return ev.MarketContext(
            close=self.close[:visible],
            dates=self.dates[:visible],
            candles_per_year=self.candles_per_year,
        )

    def period(self, split: Split) -> dict[str, Any]:
        """Wall-clock boundaries of a split, so a report can be audited later."""
        return {
            "start": str(self.dates[split.start]),
            "end": str(self.dates[split.end - 1]),
            "rows": len(split),
            "row_range": [split.start, split.end],
        }


def load_research_data(path: str | Path) -> ResearchData:
    """Load a built dataset and check it against its own metadata."""
    frame, ds_meta = load_dataset(path)
    feature_names = list(ds_meta.feature_names)
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise SystemExit(f"dataset is missing feature columns: {missing}")

    if "close" not in frame.columns:
        raise SystemExit(
            f"{path} has no 'close' column. Reporting needs the price path to build a "
            "candle-level portfolio curve and a buy-and-hold reference; rebuild the "
            "dataset with the current tools.build_features."
        )

    segment_ids = (
        frame["segment_id"].to_numpy(dtype=np.int64) if "segment_id" in frame.columns else None
    )
    if segment_ids is None and int(ds_meta.validation.get("gap_count", 0)) > 0:
        logger.warning(
            "Dataset reports market-data gaps but has no segment_id column; "
            "rebuild it with the current tools.build_features before trusting this run."
        )

    return ResearchData(
        ds_meta=ds_meta,
        feature_names=feature_names,
        feature_spec=FeatureSpec.from_dict(ds_meta.feature_spec),
        target_spec=TargetSpec.from_dict(ds_meta.target_spec),
        features=frame[feature_names].to_numpy(dtype=np.float64),
        targets=frame["target"].to_numpy(dtype=np.int64),
        future_return=frame["future_return"].to_numpy(dtype=np.float64),
        close=frame["close"].to_numpy(dtype=np.float64),
        segment_ids=segment_ids,
        dates=frame["date"].to_numpy(),
        candles_per_year=365 * 24 * 60 / timeframe_to_minutes(ds_meta.timeframe or "1h"),
    )


@dataclass(frozen=True)
class RunConfig:
    """One point in the research search space.

    The fields are exactly the dimensions ``nn.experiment`` can vary. Defaults
    match the ``nn.train`` CLI defaults, which are not tuned against any test
    result.
    """

    seed: int = 42
    lr: float = 3e-4
    seq_len: int = 64
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 64
    patience: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedWindows:
    """Scaled, windowed training and validation tensors for one fold.

    Note what is *absent*: there is no test field, and
    :func:`prepare_research_windows` takes no test split. Test windows are built
    in exactly one place — :func:`main`, through :func:`score_frozen_split`, and
    only when ``--validation-only`` is off.
    """

    scaler: StandardScaler
    train: Split
    validation: Split
    seq_len: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    idx_val: np.ndarray


def prepare_research_windows(
    data: ResearchData, train_split: Split, val_split: Split, seq_len: int
) -> PreparedWindows:
    """Fit the scaler on training rows only, then window train and validation.

    Raises ``ValueError`` — rather than returning something empty — when a split
    is too short to hold one window plus its label horizon, or when validation
    does not lie strictly after training.
    """
    if val_split.start < train_split.end:
        raise ValueError(
            f"validation must begin at or after the end of training: training ends at "
            f"row {train_split.end}, validation starts at row {val_split.start}"
        )

    scaler = StandardScaler().fit(data.features[train_split.start : train_split.end])

    # Everything below works on rows [0, val_split.end) only. Rows after this
    # fold are not merely unused — they are not in the arrays that get scaled
    # and windowed, so no later row can reach a metric even by accident. The
    # last sample a split can emit has its label at row ``end - 1``, so this
    # prefix is exactly enough (asserted in tests/test_research_workflow.py).
    visible = val_split.end
    scaled = scaler.transform(data.features[:visible])
    targets = data.targets[:visible]
    segment_ids = None if data.segment_ids is None else data.segment_ids[:visible]

    built = []
    for split in (train_split, val_split):
        X, y, idx = build_windows(
            scaled,
            targets,
            split,
            seq_len,
            data.target_spec.horizon,
            segment_ids=segment_ids,
        )
        if len(X) == 0:
            raise ValueError(
                f"the {split.name} split produced no samples. Use a longer dataset, a "
                f"shorter seq_len ({seq_len}), or different split sizes."
            )
        built.append((X, y, idx))

    (X_train, y_train, _), (X_val, y_val, idx_val) = built
    return PreparedWindows(
        scaler=scaler,
        train=train_split,
        validation=val_split,
        seq_len=seq_len,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        idx_val=idx_val,
    )


@dataclass(frozen=True)
class ValidationRun:
    """The result of fitting one configuration and scoring it on validation."""

    model: MTST
    config: MTSTConfig
    threshold: float
    threshold_report: dict[str, Any]
    #: ``majority_baseline`` / ``momentum_baseline`` / ``mtst``, on validation.
    reports: dict[str, dict[str, Any]]
    train_info: dict[str, Any]
    #: The fitted baselines, so a later sealed-test run scores the same objects.
    baselines: dict[str, Any]


def fit_and_validate(
    data: ResearchData,
    prepared: PreparedWindows,
    run: RunConfig,
    *,
    device: torch.device,
) -> ValidationRun:
    """Train one configuration and score it on validation.

    Fits everything that gets fitted: the weights, the early-stopping epoch and
    the decision threshold. All three use the training and validation blocks in
    ``prepared`` and nothing else.
    """
    set_seed(run.seed)
    config = MTSTConfig(
        input_dim=len(data.feature_names),
        seq_len=run.seq_len,
        d_model=run.d_model,
        n_heads=run.n_heads,
        num_layers=run.num_layers,
        dropout=run.dropout,
    )
    model, train_info = train_model(
        config,
        prepared.X_train,
        prepared.y_train,
        prepared.X_val,
        prepared.y_val,
        device=device,
        epochs=run.epochs,
        batch_size=run.batch_size,
        lr=run.lr,
        patience=run.patience,
    )

    val_proba = predict_proba(model, prepared.X_val, device)
    val_return = data.future_return[prepared.idx_val]
    threshold, threshold_report = ev.select_threshold(
        val_proba,
        val_return,
        data.target_spec,
        row_index=prepared.idx_val,
    )

    baselines = {
        "majority_baseline": MajorityClassBaseline().fit(prepared.y_train),
        "momentum_baseline": MomentumBaseline(
            feature_index=data.feature_names.index("ema_cross")
        ),
    }

    # The inner block's own prices and timestamps, bounded by the same row the
    # windowing was bounded by. This exists so `nn.experiment`'s optional
    # `annualised_sharpe` objective keeps working under the corrected metric —
    # it is *reporting* input only. `select_threshold` above is already done by
    # this point and never receives it, so nothing here can move the chosen
    # threshold. No economic reference is computed on the inner block: CASH and
    # buy-and-hold answer a question about a reported result, and the inner
    # block is a selection score, never reported as one.
    market = data.market(prepared.validation.end)

    def score(proba: np.ndarray, decision_threshold: float = threshold) -> dict[str, Any]:
        return ev.evaluate(
            proba,
            prepared.y_val,
            val_return,
            data.target_spec,
            decision_threshold,
            market=market,
            row_index=prepared.idx_val,
        )

    # Baselines are scored at their own declared threshold, never the model's.
    # Theirs is fitted on nothing; the model's is fitted on the inner block, and
    # letting it govern the baselines made the floor move with the seed.
    reports = {
        name: score(baseline.predict_proba(prepared.X_val), BASELINE_DECISION_THRESHOLD)
        for name, baseline in baselines.items()
    }
    reports["mtst"] = score(val_proba)

    return ValidationRun(
        model=model,
        config=config,
        threshold=threshold,
        threshold_report=threshold_report,
        reports=reports,
        train_info=train_info,
        baselines=baselines,
    )


@dataclass(frozen=True)
class FrozenScore:
    """What a frozen evaluation produced on one split.

    Carries the sample-level arrays as well as the reports, so a caller that
    wants to persist per-sample predictions does not have to window the split a
    second time to get them.
    """

    #: ``majority_baseline`` / ``momentum_baseline`` / ``mtst``.
    reports: dict[str, dict[str, Any]]
    #: CASH and buy-and-hold over the same window. Not models — see
    #: :func:`nn.evaluate.economic_references`.
    economic_references: dict[str, Any]
    #: Dataset row index of each scored sample.
    idx: np.ndarray
    #: True class of each scored sample, in the dataset's own encoding.
    y: np.ndarray
    #: The model's probabilities, in ``CLASS_ORDER``.
    proba: np.ndarray


def score_frozen_split(
    data: ResearchData,
    scaler: StandardScaler,
    split: Split,
    seq_len: int,
    *,
    model: MTST,
    baselines: dict[str, Any],
    threshold: float,
    device: torch.device,
) -> FrozenScore:
    """Score an already-frozen model, threshold and baselines on one split.

    Nothing here is fitted. The scaler, the weights, the decision threshold and
    the baselines all arrive fitted from somewhere else, and this function only
    transforms, windows and measures — which is what makes it safe to point at a
    block that must not influence model selection: ``nn.train``'s sealed test
    block, and ``nn.walkforward``'s outer validation block.

    Rows are read up to ``split.end`` and no further, so a split that ends
    before the sealed boundary cannot reach a sealed row even by accident. The
    returned :class:`FrozenScore` carries the row indices actually scored, so a
    caller — or a test — can check where the samples came from instead of
    trusting the name.

    Raises ``ValueError`` when the split is too short to hold one window plus
    its label horizon, rather than reporting metrics over zero samples.
    """
    visible = split.end
    scaled = scaler.transform(data.features[:visible])
    X, y, idx = build_windows(
        scaled,
        data.targets[:visible],
        split,
        seq_len,
        data.target_spec.horizon,
        segment_ids=None if data.segment_ids is None else data.segment_ids[:visible],
    )
    if len(X) == 0:
        raise ValueError(
            f"the {split.name} split produced no samples. Use a longer dataset, a "
            f"shorter seq_len ({seq_len}), or different split sizes."
        )

    future_return = data.future_return[idx]

    market = data.market(visible)

    def score(proba: np.ndarray, decision_threshold: float = threshold) -> dict[str, Any]:
        return ev.evaluate(
            proba,
            y,
            future_return,
            data.target_spec,
            decision_threshold,
            market=market,
            row_index=idx,
        )

    proba = predict_proba(model, X, device)
    reports = {
        name: score(baseline.predict_proba(X), BASELINE_DECISION_THRESHOLD)
        for name, baseline in baselines.items()
    }
    reports["mtst"] = score(proba)
    # Not a model and not a baseline: what the same window did for someone who
    # never traded, and for someone who bought once and held. Reported beside
    # the models because beating a statistical floor is not the same as making
    # money, and the aggregate verdict needs both to say so.
    references = ev.economic_references(market, data.target_spec, idx)
    return FrozenScore(
        reports=reports, idx=idx, y=y, proba=proba, economic_references=references
    )


def train_model(
    config: MTSTConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 1e-4,
    patience: int = 5,
) -> tuple[MTST, dict[str, Any]]:
    """Train one model, keeping the weights from the best validation epoch.

    Early stopping watches validation loss. The test split is not touched here
    at all — it is not even passed in.
    """
    model = MTST(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, config.n_classes).to(device))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
    )

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * len(xb)
        train_loss /= max(1, len(X_train))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += float(criterion(model(xb), yb).item()) * len(xb)
        val_loss /= max(1, len(X_val))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info(
            "epoch %d/%d  train_loss=%.5f  val_loss=%.5f", epoch, epochs, train_loss, val_loss
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val}


def tune_learning_rate(
    trials: int,
    build: Any,
    *,
    seed: int,
) -> float:
    """Optional hyperparameter search over the learning rate.

    Uses Ray Tune when it is installed, otherwise a plain random search over
    the same range. Both are off unless ``--tune-trials`` is positive, so a
    developer smoke run never pays for them.
    """
    rng = np.random.default_rng(seed)
    candidates = [float(10 ** rng.uniform(-4.5, -2.5)) for _ in range(trials)]

    best_lr, best_loss = candidates[0], float("inf")
    for lr in candidates:
        val_loss = build(lr).train_info["best_val_loss"]
        if val_loss < best_loss:
            best_loss, best_lr = val_loss, lr
        logger.info("trial lr=%.2e -> val_loss=%.5f", lr, val_loss)
    logger.info("Best learning rate from %d trials: %.2e", trials, best_lr)
    return best_lr


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MTST signal classifier.")
    parser.add_argument(
        "--dataset", required=True, help="Parquet dataset from build_features."
    )
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=0,
        help="Hyperparameter search trials. 0 (default) trains once.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote to current if the validation gates pass.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Research mode: leave the test split sealed. Trains, selects the "
            "threshold and reports on validation only. Cannot be combined with "
            "--promote, and the artifact it writes can never be promoted."
        ),
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Also log this run to MLflow (optional, off by default).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_argparser().parse_args(argv)
    if args.validation_only and args.promote:
        raise SystemExit(
            "--promote cannot be combined with --validation-only: a research run "
            "leaves the test split sealed, so it has no out-of-sample estimate and "
            "must never put a model in front of traffic."
        )
    set_seed(args.seed)
    device = resolve_device(args.device)
    logger.info("Training on %s with seed %d", device, args.seed)

    # --- 1. data ------------------------------------------------------
    data = load_research_data(args.dataset)
    plan = chronological_split(data.n_rows, args.train_frac, args.val_frac)
    logger.info("Split plan: %s", json.dumps(plan.to_dict()))
    if args.validation_only:
        logger.warning(
            "VALIDATION-ONLY research run: the test split (%d rows, %s to %s) "
            "REMAINS SEALED and will not be evaluated.",
            len(plan.test),
            data.period(plan.test)["start"],
            data.period(plan.test)["end"],
        )

    # --- 2-5. scaler on train rows, windows, training, threshold ---------
    # All of it lives in the research core, which is given the train and
    # validation splits only.
    try:
        prepared = prepare_research_windows(data, plan.train, plan.validation, args.seq_len)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    logger.info(
        "train: %d samples, validation: %d samples", len(prepared.X_train), len(prepared.X_val)
    )

    run_config = RunConfig(
        seed=args.seed,
        lr=args.lr,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    def build(lr: float) -> ValidationRun:
        return fit_and_validate(data, prepared, replace(run_config, lr=lr), device=device)

    lr = args.lr
    if args.tune_trials > 0:
        lr = tune_learning_rate(args.tune_trials, build, seed=args.seed)
    result = build(lr)

    model, threshold = result.model, result.threshold
    validation_reports = result.reports
    logger.info(
        "Selected decision threshold %.2f on validation (%d trades, net %.4f)",
        threshold,
        result.threshold_report["n_trades"],
        result.threshold_report["net_return"],
    )

    # --- 6. the one and only test evaluation -------------------------------
    # This block is the only place in the repository that windows the test
    # split. It runs after every fitted quantity (weights, scaler, threshold,
    # early-stopping epoch) is frozen, and not at all in research mode.
    test_reports: dict[str, Any] | None = None
    test_references: dict[str, Any] | None = None
    if not args.validation_only:
        try:
            scored = score_frozen_split(
                data,
                prepared.scaler,
                plan.test,
                args.seq_len,
                model=model,
                baselines=result.baselines,
                threshold=threshold,
                device=device,
            )
        except ValueError as exc:
            raise SystemExit(f"{exc} (--seq-len is currently {args.seq_len})") from exc
        test_reports = scored.reports
        test_references = scored.economic_references
        logger.info("test: %d samples", len(scored.idx))

    print("\nValidation (used for model selection):")
    print(ev.compare(validation_reports))
    if test_reports is None:
        print(
            "\nTest: SEALED — not evaluated in this validation-only research run.\n"
            "Freeze your research decisions, then re-run without --validation-only "
            "to spend the test split once."
        )
    else:
        print("\nTest (held out, scored once):")
        print(ev.compare(test_reports))

    # --- 7. artifact -------------------------------------------------------
    version = new_version(f"{args.dataset}{args.seed}{lr}")
    metadata = ModelMetadata(
        model_version=version,
        feature_names=list(data.feature_names),
        sequence_length=args.seq_len,
        feature_spec=data.feature_spec,
        target_spec=data.target_spec,
        scaler_mean=prepared.scaler.mean.tolist(),
        scaler_std=prepared.scaler.std.tolist(),
        decision_threshold=threshold,
        trained_at=datetime.now(timezone.utc).isoformat(),
        dataset_start=data.ds_meta.start,
        dataset_end=data.ds_meta.end,
        # Temporal provenance, so a later backtest can prove it is out-of-sample.
        train_end=str(data.dates[plan.train.end - 1]),
        validation_end=str(data.dates[plan.validation.end - 1]),
        exchange=data.ds_meta.exchange,
        pair=data.ds_meta.pair,
        timeframe=data.ds_meta.timeframe,
        validation_metrics=validation_reports["mtst"],
    )

    gates = PromotionGates()
    passed, failures = check_gates(
        validation_reports["mtst"],
        {k: v for k, v in validation_reports.items() if k.endswith("baseline")},
        gates,
    )

    report = {
        "args": vars(args),
        "learning_rate": lr,
        "device": str(device),
        "split_plan": plan.to_dict(),
        "periods": {split.name: data.period(split) for split in plan},
        "dataset": data.ds_meta.to_dict(),
        "training": result.train_info,
        "threshold_selection": result.threshold_report,
        "validation": validation_reports,
        "test": test_reports,
        # CASH and buy-and-hold over the sealed block, beside the models scored
        # on it. Absent in a research-only run, where the block is not opened.
        "test_economic_references": test_references,
        # Read by nn.registry.promote, which refuses a research_only artifact.
        "research_only": bool(args.validation_only),
        "test_evaluated": test_reports is not None,
        "promotion": {
            "gates": gates.to_dict(),
            "passed": passed,
            "failures": failures,
            "eligible": not args.validation_only,
        },
    }
    save_model(args.models_dir, version, model, metadata, report)

    if args.validation_only:
        logger.warning(
            "Research artifact %s saved. Validation gates %s, but a validation-only "
            "run is never promotable: its test split was never scored.",
            version,
            "passed" if passed else "failed",
        )
    elif passed:
        logger.info("Model %s passed the promotion gates", version)
        if args.promote:
            promote(args.models_dir, version)
        else:
            logger.info("Pass --promote to make it the served model")
    else:
        logger.warning("Model %s NOT promoted:", version)
        for failure in failures:
            logger.warning("  - %s", failure)

    if args.mlflow:
        _log_to_mlflow(version, args, report, metadata)

    return 0


def _log_to_mlflow(
    version: str, args: argparse.Namespace, report: dict[str, Any], metadata: ModelMetadata
) -> None:
    """Mirror the run into MLflow, if it is installed.

    Optional and best-effort: a tracking failure must not fail a training run
    whose artifact is already safely on disk. Registers the model **once** — the
    previous code called ``register_model`` twice per run and set the ``prod``
    alias unconditionally, creating two versions and auto-promoting both.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("--mlflow given but mlflow is not installed; skipping")
        return

    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment("chimera_signal_classifier")
        with mlflow.start_run(run_name=version):
            mlflow.log_params(
                {
                    "seq_len": args.seq_len,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "num_layers": args.num_layers,
                    "learning_rate": report["learning_rate"],
                    "seed": args.seed,
                    "horizon": metadata.target_spec.horizon,
                    "cost_threshold": metadata.target_spec.cost_threshold,
                    "decision_threshold": metadata.decision_threshold,
                }
            )
            for split in ("validation", "test"):
                if report.get(split) is None:  # sealed test in a research run
                    continue
                trading = report[split]["mtst"]["trading"]
                classification = report[split]["mtst"]["classification"]
                mlflow.log_metrics(
                    {
                        f"{split}_macro_f1": classification["macro_f1"],
                        f"{split}_directional_accuracy": classification[
                            "directional_accuracy"
                        ],
                        f"{split}_coverage": classification["coverage"],
                        f"{split}_net_return": trading["net_return"],
                        f"{split}_max_drawdown": trading["max_drawdown"],
                        # Undefined risk-adjusted statistics are omitted rather
                        # than logged as zero: MLflow has no null, and a zero
                        # here would be read as a measurement.
                        **{
                            f"{split}_{name}": trading[name]
                            for name in ("annualised_sharpe", "per_trade_sharpe")
                            if trading[name] is not None
                        },
                    }
                )
            mlflow.log_artifacts(str(Path(args.models_dir) / version), "model")
    except Exception as exc:  # noqa: BLE001 - tracking is not load-bearing
        logger.warning("MLflow logging failed (%s); artifact on disk is unaffected", exc)


if __name__ == "__main__":
    raise SystemExit(main())
