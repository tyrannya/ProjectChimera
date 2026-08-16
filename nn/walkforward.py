"""Walk-forward evaluation.

    python -m nn.walkforward --dataset data/datasets/binance_BTC_USDT_1h.parquet \
        --folds 4 --epochs 5 --out artifacts/walkforward

A single train/validation/test split tells you how one model did on one future
period. Walk-forward asks the harder question: does the *procedure* keep
working as the market moves? Each fold repeats the whole pipeline — fit scaler,
train, pick a threshold — on its own training window, freezes it, and scores
the next unseen block::

    fold 0: [--- train ---][ val ][ test ]
    fold 1:        [--- train ---][ val ][ test ]
    fold 2:               [--- train ---][ val ][ test ]

Nothing is carried between folds, so a fold's test block is never seen by any
fitted quantity that scores it. Results are written as JSON (machine-readable)
and a short Markdown summary, with the baselines alongside every fold — a model
that beats its baselines in one fold out of four has not been shown to work.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from chimera.contracts import TargetSpec
from nn import evaluate as ev
from nn.baselines import MajorityClassBaseline, MomentumBaseline
from nn.data_pipeline import load_dataset, timeframe_to_minutes
from nn.dataset import Split, StandardScaler, build_windows
from nn.model_def import MTSTConfig
from nn.train import predict_proba, resolve_device, set_seed, train_model

logger = logging.getLogger(__name__)


def plan_folds(
    n_rows: int, folds: int, train_size: int, val_size: int, test_size: int
) -> list[tuple[Split, Split, Split]]:
    """Rolling windows of fixed size, stepped by ``test_size``.

    Raises when the dataset is too short for the requested plan rather than
    quietly producing fewer folds than asked for.
    """
    span = train_size + val_size + test_size
    step = test_size
    needed = span + step * (folds - 1)
    if needed > n_rows:
        raise ValueError(
            f"need {needed} rows for {folds} folds of "
            f"{train_size}/{val_size}/{test_size}, dataset has {n_rows}"
        )

    plans = []
    for k in range(folds):
        start = k * step
        train_end = start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        plans.append(
            (
                Split("train", start, train_end),
                Split("validation", train_end, val_end),
                Split("test", val_end, test_end),
            )
        )
    return plans


def run_fold(
    fold: int,
    splits: tuple[Split, Split, Split],
    features: np.ndarray,
    targets: np.ndarray,
    future_return: np.ndarray,
    dates: Any,
    feature_names: list[str],
    target_spec: TargetSpec,
    args: argparse.Namespace,
    candles_per_year: float,
) -> dict[str, Any]:
    """Train and score one fold. Every fitted quantity is local to this call."""
    train_split, val_split, test_split = splits
    device = resolve_device(args.device)
    set_seed(args.seed + fold)

    scaler = StandardScaler().fit(features[train_split.start : train_split.end])
    scaled = scaler.transform(features)

    windows = {
        split.name: build_windows(scaled, targets, split, args.seq_len, target_spec.horizon)
        for split in (train_split, val_split, test_split)
    }
    for name, (X, _, _) in windows.items():
        if len(X) == 0:
            raise ValueError(f"fold {fold}: split {name} produced no samples")

    X_train, y_train, _ = windows["train"]
    X_val, y_val, idx_val = windows["validation"]
    X_test, y_test, idx_test = windows["test"]

    config = MTSTConfig(
        input_dim=len(feature_names),
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model, train_info = train_model(
        config,
        X_train,
        y_train,
        X_val,
        y_val,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    val_proba = predict_proba(model, X_val, device)
    threshold, _ = ev.select_threshold(val_proba, future_return[idx_val], target_spec)

    majority = MajorityClassBaseline().fit(y_train)
    momentum = MomentumBaseline(feature_index=feature_names.index("ema_cross"))

    def score(proba: np.ndarray) -> dict[str, Any]:
        return ev.evaluate(
            proba,
            y_test,
            future_return[idx_test],
            target_spec,
            threshold,
            candles_per_year=candles_per_year,
        )

    return {
        "fold": fold,
        "threshold": threshold,
        "best_epoch": train_info["best_epoch"],
        "periods": {
            name: {
                "start": str(dates[split.start]),
                "end": str(dates[split.end - 1]),
                "rows": len(split),
            }
            for name, split in (
                ("train", train_split),
                ("validation", val_split),
                ("test", test_split),
            )
        },
        "test": {
            "majority_baseline": score(majority.predict_proba(X_test)),
            "momentum_baseline": score(momentum.predict_proba(X_test)),
            "mtst": score(predict_proba(model, X_test, device)),
        },
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-fold test metrics, including how often the model won."""
    models = ["majority_baseline", "momentum_baseline", "mtst"]
    summary: dict[str, Any] = {"folds": len(results), "per_model": {}}

    for name in models:
        net = [r["test"][name]["trading"]["net_return"] for r in results]
        sharpe = [r["test"][name]["trading"]["sharpe"] for r in results]
        f1 = [r["test"][name]["classification"]["macro_f1"] for r in results]
        summary["per_model"][name] = {
            "mean_net_return": round(statistics.fmean(net), 6),
            "median_net_return": round(statistics.median(net), 6),
            "positive_folds": sum(1 for v in net if v > 0),
            "mean_sharpe": round(statistics.fmean(sharpe), 4),
            "mean_macro_f1": round(statistics.fmean(f1), 4),
        }

    beat = sum(
        1
        for r in results
        if r["test"]["mtst"]["trading"]["net_return"]
        > max(
            r["test"]["majority_baseline"]["trading"]["net_return"],
            r["test"]["momentum_baseline"]["trading"]["net_return"],
        )
    )
    summary["mtst_beat_baselines_in_folds"] = beat
    summary["verdict"] = (
        "model beat both baselines in a majority of folds"
        if beat * 2 > len(results)
        else "model did NOT consistently beat the baselines"
    )
    return summary


def to_markdown(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Walk-forward evaluation",
        "",
        "Out-of-sample results per fold. Every fold retrains from scratch and",
        "picks its own threshold on its own validation block.",
        "",
        "| fold | test period | model | trades | net return | Sharpe | max DD | macro F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        period = result["periods"]["test"]
        window = f"{period['start'][:10]} to {period['end'][:10]}"
        for name in ("majority_baseline", "momentum_baseline", "mtst"):
            trading = result["test"][name]["trading"]
            classification = result["test"][name]["classification"]
            lines.append(
                f"| {result['fold']} | {window} | {name} | {trading['n_trades']} | "
                f"{trading['net_return']:+.4f} | {trading['sharpe']:.2f} | "
                f"{trading['max_drawdown']:.4f} | {classification['macro_f1']:.4f} |"
            )

    lines += [
        "",
        "## Summary",
        "",
        "| model | mean net return | positive folds | mean Sharpe | mean macro F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, stats in summary["per_model"].items():
        lines.append(
            f"| {name} | {stats['mean_net_return']:+.4f} | "
            f"{stats['positive_folds']}/{summary['folds']} | "
            f"{stats['mean_sharpe']:.2f} | {stats['mean_macro_f1']:.4f} |"
        )
    lines += [
        "",
        f"**Verdict:** {summary['verdict']} "
        f"({summary['mtst_beat_baselines_in_folds']}/{summary['folds']} folds).",
        "",
        "Historical backtest performance does not guarantee future profitability.",
        "",
    ]
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="artifacts/walkforward")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    frame, ds_meta = load_dataset(args.dataset)
    target_spec = TargetSpec.from_dict(ds_meta.target_spec)
    feature_names = ds_meta.feature_names

    features = frame[feature_names].to_numpy(dtype=np.float64)
    targets = frame["target"].to_numpy(dtype=np.int64)
    future_return = frame["future_return"].to_numpy(dtype=np.float64)
    dates = frame["date"].to_numpy()

    n_rows = len(frame)
    # Default plan: fit as many equal folds as the dataset allows, splitting
    # each window 70/15/15 to match the single-split defaults in nn.train.
    if args.test_size is None or args.val_size is None or args.train_size is None:
        window = n_rows // (1 + (args.folds - 1) * 0.15)
        test_size = args.test_size or max(
            args.seq_len + target_spec.horizon + 10, int(window * 0.15)
        )
        val_size = args.val_size or test_size
        train_size = args.train_size or int(window - test_size - val_size)
    else:
        train_size, val_size, test_size = args.train_size, args.val_size, args.test_size

    folds = plan_folds(n_rows, args.folds, train_size, val_size, test_size)
    logger.info(
        "%d folds of train=%d val=%d test=%d over %d rows",
        len(folds),
        train_size,
        val_size,
        test_size,
        n_rows,
    )

    candles_per_year = 365 * 24 * 60 / timeframe_to_minutes(ds_meta.timeframe or "1h")

    results = []
    for i, splits in enumerate(folds):
        logger.info("--- fold %d/%d ---", i + 1, len(folds))
        results.append(
            run_fold(
                i,
                splits,
                features,
                targets,
                future_return,
                dates,
                feature_names,
                target_spec,
                args,
                candles_per_year,
            )
        )

    summary = summarise(results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "walkforward.json").write_text(
        json.dumps(
            {
                "dataset": ds_meta.to_dict(),
                "config": vars(args),
                "folds": results,
                "summary": summary,
            },
            indent=2,
            default=str,
        )
    )
    markdown = to_markdown(results, summary)
    (out_dir / "walkforward.md").write_text(markdown)
    print(markdown)
    logger.info("Wrote results to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
