"""Training entrypoint.

    python -m nn.train --dataset data/datasets/binance_BTC_USDT_1h.parquet
    python -m nn.train --dataset ... --epochs 2 --tune-trials 0   # smoke run

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
from nn.baselines import MajorityClassBaseline, MomentumBaseline
from nn.data_pipeline import load_dataset, timeframe_to_minutes
from nn.dataset import StandardScaler, build_windows, chronological_split
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
        _, info = build(lr)
        if info["best_val_loss"] < best_loss:
            best_loss, best_lr = info["best_val_loss"], lr
        logger.info("trial lr=%.2e -> val_loss=%.5f", lr, info["best_val_loss"])
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
    set_seed(args.seed)
    device = resolve_device(args.device)
    logger.info("Training on %s with seed %d", device, args.seed)

    # --- 1. data ------------------------------------------------------
    frame, ds_meta = load_dataset(args.dataset)
    feature_names = ds_meta.feature_names
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise SystemExit(f"dataset is missing feature columns: {missing}")

    target_spec = TargetSpec.from_dict(ds_meta.target_spec)
    feature_spec = FeatureSpec.from_dict(ds_meta.feature_spec)

    features = frame[feature_names].to_numpy(dtype=np.float64)
    targets = frame["target"].to_numpy(dtype=np.int64)
    future_return = frame["future_return"].to_numpy(dtype=np.float64)
    segment_ids = (
        frame["segment_id"].to_numpy(dtype=np.int64)
        if "segment_id" in frame.columns
        else None
    )
    if segment_ids is None and int(ds_meta.validation.get("gap_count", 0)) > 0:
        logger.warning(
            "Dataset reports market-data gaps but has no segment_id column; "
            "rebuild it with the current tools.build_features before trusting this run."
        )

    plan = chronological_split(len(frame), args.train_frac, args.val_frac)
    logger.info("Split plan: %s", json.dumps(plan.to_dict()))

    # --- 2. scaler: train rows only ------------------------------------
    scaler = StandardScaler().fit(features[plan.train.start : plan.train.end])
    scaled = scaler.transform(features)

    windows = {
        split.name: build_windows(
            scaled,
            targets,
            split,
            args.seq_len,
            target_spec.horizon,
            segment_ids=segment_ids,
        )
        for split in plan
    }
    for name, (X, _, _) in windows.items():
        logger.info("%s: %d samples", name, len(X))
        if len(X) == 0:
            raise SystemExit(
                f"the {name} split produced no samples. Use a longer dataset, a "
                f"shorter --seq-len ({args.seq_len}), or different split fractions."
            )

    X_train, y_train, _ = windows["train"]
    X_val, y_val, idx_val = windows["validation"]
    X_test, y_test, idx_test = windows["test"]

    candles_per_year = 365 * 24 * 60 / timeframe_to_minutes(ds_meta.timeframe or "1h")

    # --- 3. baselines ---------------------------------------------------
    majority = MajorityClassBaseline().fit(y_train)
    momentum = MomentumBaseline(feature_index=feature_names.index("ema_cross"))

    # --- 4. train --------------------------------------------------------
    config = MTSTConfig(
        input_dim=len(feature_names),
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    def build(lr: float) -> tuple[MTST, dict[str, Any]]:
        set_seed(args.seed)
        return train_model(
            config,
            X_train,
            y_train,
            X_val,
            y_val,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=lr,
            patience=args.patience,
        )

    lr = args.lr
    if args.tune_trials > 0:
        lr = tune_learning_rate(args.tune_trials, build, seed=args.seed)
    model, train_info = build(lr)

    # --- 5. threshold on validation --------------------------------------
    val_proba = predict_proba(model, X_val, device)
    threshold, threshold_report = ev.select_threshold(
        val_proba, future_return[idx_val], target_spec
    )
    logger.info(
        "Selected decision threshold %.2f on validation (%d trades, net %.4f)",
        threshold,
        threshold_report["n_trades"],
        threshold_report["net_return"],
    )

    # --- 6. reports -------------------------------------------------------
    def report_for(proba: np.ndarray, split_name: str) -> dict[str, Any]:
        _, y, idx = windows[split_name]
        return ev.evaluate(
            proba,
            y,
            future_return[idx],
            target_spec,
            threshold,
            candles_per_year=candles_per_year,
        )

    validation_reports = {
        "majority_baseline": report_for(majority.predict_proba(X_val), "validation"),
        "momentum_baseline": report_for(momentum.predict_proba(X_val), "validation"),
        "mtst": report_for(val_proba, "validation"),
    }

    # The test split is scored exactly once, here, after every fitted quantity
    # (weights, scaler, threshold, early-stopping epoch) is frozen.
    test_reports = {
        "majority_baseline": report_for(majority.predict_proba(X_test), "test"),
        "momentum_baseline": report_for(momentum.predict_proba(X_test), "test"),
        "mtst": report_for(predict_proba(model, X_test, device), "test"),
    }

    print("\nValidation (used for model selection):")
    print(ev.compare(validation_reports))
    print("\nTest (held out, scored once):")
    print(ev.compare(test_reports))

    # --- 7. artifact -------------------------------------------------------
    version = new_version(f"{args.dataset}{args.seed}{lr}")
    metadata = ModelMetadata(
        model_version=version,
        feature_names=list(feature_names),
        sequence_length=args.seq_len,
        feature_spec=feature_spec,
        target_spec=target_spec,
        scaler_mean=scaler.mean.tolist(),
        scaler_std=scaler.std.tolist(),
        decision_threshold=threshold,
        trained_at=datetime.now(timezone.utc).isoformat(),
        dataset_start=ds_meta.start,
        dataset_end=ds_meta.end,
        # Temporal provenance, so a later backtest can prove it is out-of-sample.
        train_end=str(frame["date"].iloc[plan.train.end - 1]),
        validation_end=str(frame["date"].iloc[plan.validation.end - 1]),
        exchange=ds_meta.exchange,
        pair=ds_meta.pair,
        timeframe=ds_meta.timeframe,
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
        "dataset": ds_meta.to_dict(),
        "training": train_info,
        "threshold_selection": threshold_report,
        "validation": validation_reports,
        "test": test_reports,
        "promotion": {"gates": gates.to_dict(), "passed": passed, "failures": failures},
    }
    save_model(args.models_dir, version, model, metadata, report)

    if passed:
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
                        f"{split}_sharpe": trading["sharpe"],
                        f"{split}_max_drawdown": trading["max_drawdown"],
                    }
                )
            mlflow.log_artifacts(str(Path(args.models_dir) / version), "model")
    except Exception as exc:  # noqa: BLE001 - tracking is not load-bearing
        logger.warning("MLflow logging failed (%s); artifact on disk is unaffected", exc)


if __name__ == "__main__":
    raise SystemExit(main())
