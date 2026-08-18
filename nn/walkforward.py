"""Expanding-window walk-forward validation.

    python -m nn.walkforward --dataset data/datasets/binance_BTC_USDT_1h.parquet \
        --folds 4 --epochs 5 --out artifacts/walkforward

One train/validation split tells you how one model did on one stretch of
market. Walk-forward asks the harder question: does the *procedure* keep working
as the market moves? The training window expands, and each fold is validated on
the block that comes immediately after it::

    fold 0: [--- train ---][ val ]
    fold 1: [------ train ------][ val ]
    fold 2: [--------- train --------][ val ]

Every fold refits its own scaler on its own training rows, retrains from
scratch, early-stops on its own validation block and selects its own decision
threshold there. Nothing carries between folds, and validation always begins at
or after the row where training ends — asserted in
``tests/test_research_workflow.py``, not merely intended.

**This is a validation tool, and it is bounded.** Folds are planned over
``[0, sealed_test_start)`` — the rows before the sealed test block under the
same 70/15/15 contract ``nn.train`` uses — and never over the dataset length.
That distinction is the whole safeguard, and it was learned the hard way twice:

* an early version scored an explicit per-fold *test* block, spending the sealed
  estimate on every research iteration;
* its replacement stopped naming any split "test" but still planned folds over
  all rows, so with the default geometry the last two validation windows landed
  inside the sealed block. The output said ``test_evaluated: false`` and was
  wrong — the rows were sealed rows wearing the label "validation".

Renaming a split does not unseal its rows. So the boundary is a row index that
both this module and ``nn.train`` compute through
:func:`nn.dataset.sealed_test_start`, the planner takes that index rather than
``n_rows``, and ``tests/test_research_workflow.py`` asserts on row indices —
never on split names, which is exactly the check that missed it.

Results are written as JSON and a short Markdown summary, with the baselines
alongside every fold — a model that beats its baselines in one fold out of four
has not been shown to work.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from nn.dataset import Split, sealed_test_start
from nn.train import (
    ResearchData,
    RunConfig,
    fit_and_validate,
    load_research_data,
    prepare_research_windows,
    resolve_device,
)

logger = logging.getLogger(__name__)

MODELS = ("majority_baseline", "momentum_baseline", "mtst")

#: Metrics aggregated across folds: where to find each in an evaluation report.
SUMMARY_METRICS = {
    "macro_f1": ("classification", "macro_f1"),
    "directional_accuracy": ("classification", "directional_accuracy"),
    "coverage": ("classification", "coverage"),
    "n_trades": ("trading", "n_trades"),
    "net_return": ("trading", "net_return"),
    "sharpe": ("trading", "sharpe"),
    "max_drawdown": ("trading", "max_drawdown"),
}


def plan_expanding_folds(
    boundary: int, folds: int, min_train: int, val_size: int, step: int
) -> list[tuple[Split, Split]]:
    """Expanding training windows, each validated on the block right after it.

    ``boundary`` is the first sealed row — normally
    :func:`nn.dataset.sealed_test_start`. **Every row this function hands out,
    training and validation alike, is strictly below it.** The planner takes the
    boundary rather than the dataset length precisely because the two are not
    the same number: planning over the dataset length is what silently walked
    the last folds into the sealed block.

    Fold ``k`` trains on rows ``[0, min_train + k*step)`` and validates on the
    ``val_size`` rows that follow. Training therefore only ever grows forward in
    time, and no fold's validation rows are available to an earlier fold.

    Raises when the research region is too short for the requested plan, rather
    than quietly returning fewer folds than asked for — or, worse, borrowing the
    rows it is short by from the sealed block.
    """
    if folds < 1:
        raise ValueError("folds must be at least 1")
    if min_train < 1 or val_size < 1 or step < 1:
        raise ValueError(
            f"min_train ({min_train}), val_size ({val_size}) and step ({step}) "
            "must all be positive"
        )

    needed = min_train + step * (folds - 1) + val_size
    if needed > boundary:
        raise ValueError(
            f"need {needed} rows for {folds} expanding folds with min_train={min_train}, "
            f"val_size={val_size}, step={step}, but only {boundary} rows lie before the "
            "sealed test block. Reduce --folds, --min-train-frac or --fold-val-frac; "
            "the sealed rows are not available to make up the difference."
        )

    plans = []
    for k in range(folds):
        train_end = min_train + k * step
        plans.append(
            (
                Split("train", 0, train_end),
                Split("validation", train_end, train_end + val_size),
            )
        )

    # The arithmetic above already guarantees this; assert it anyway, because
    # the failure it guards against is invisible in the output — sealed rows
    # scored under the name "validation" look exactly like validation.
    for train_split, val_split in plans:
        if train_split.end > boundary or val_split.end > boundary:
            raise AssertionError(
                f"fold plan crosses the sealed boundary {boundary}: "
                f"train ends {train_split.end}, validation ends {val_split.end}"
            )
    return plans


def run_fold(
    fold: int,
    data: ResearchData,
    splits: tuple[Split, Split],
    run: RunConfig,
    *,
    device: Any,
) -> dict[str, Any]:
    """Train and validate one fold. Every fitted quantity is local to this call."""
    train_split, val_split = splits
    prepared = prepare_research_windows(data, train_split, val_split, run.seq_len)
    # Vary the seed per fold so a lucky initialisation cannot flatter the whole
    # run, while staying reproducible for a given --seed.
    result = fit_and_validate(
        data, prepared, replace(run, seed=run.seed + fold), device=device
    )

    return {
        "fold": fold,
        "seed": run.seed + fold,
        "threshold": result.threshold,
        "best_epoch": result.train_info["best_epoch"],
        "best_val_loss": result.train_info["best_val_loss"],
        "samples": {"train": len(prepared.X_train), "validation": len(prepared.X_val)},
        "periods": {
            "train": data.period(train_split),
            "validation": data.period(val_split),
        },
        "validation": result.reports,
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation of each metric across folds, per model."""
    summary: dict[str, Any] = {"folds": len(results), "per_model": {}}

    for name in MODELS:
        stats: dict[str, Any] = {}
        for metric, (section, key) in SUMMARY_METRICS.items():
            values = [float(r["validation"][name][section][key]) for r in results]
            stats[metric] = {
                "mean": round(statistics.fmean(values), 6),
                # Sample standard deviation needs two folds; one fold has none.
                "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
                "values": [round(v, 6) for v in values],
            }
        stats["positive_net_return_folds"] = sum(
            1 for v in stats["net_return"]["values"] if v > 0
        )
        summary["per_model"][name] = stats

    beat = sum(
        1
        for r in results
        if r["validation"]["mtst"]["trading"]["net_return"]
        > max(
            r["validation"]["majority_baseline"]["trading"]["net_return"],
            r["validation"]["momentum_baseline"]["trading"]["net_return"],
        )
    )
    summary["mtst_beat_baselines_in_folds"] = beat
    summary["verdict"] = (
        "model beat both baselines in a majority of folds"
        if beat * 2 > len(results)
        else "model did NOT consistently beat the baselines"
    )
    return summary


def to_markdown(
    results: list[dict[str, Any]], summary: dict[str, Any], sealed: dict[str, Any]
) -> str:
    lines = [
        "# Walk-forward validation",
        "",
        "Expanding training window; each fold is validated on the block that follows",
        "it. Every fold refits its scaler, retrains, and selects its own threshold.",
        "",
        f"**Sealed test block:** rows {sealed['row_range'][0]}-{sealed['row_range'][1]}, "
        f"{sealed['start'][:10]} to {sealed['end'][:10]}. No fold below plans, trains",
        "on, or scores a row at or after that boundary.",
        "",
        "| fold | train period | validation period | model | trades | net return | "
        "Sharpe | max DD | macro F1 | dir acc | coverage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        train_period = result["periods"]["train"]
        val_period = result["periods"]["validation"]
        train_window = f"{train_period['start'][:10]} to {train_period['end'][:10]}"
        val_window = f"{val_period['start'][:10]} to {val_period['end'][:10]}"
        for name in MODELS:
            report = result["validation"][name]
            trading, classification = report["trading"], report["classification"]
            lines.append(
                f"| {result['fold']} | {train_window} | {val_window} | {name} | "
                f"{trading['n_trades']} | {trading['net_return']:+.4f} | "
                f"{trading['sharpe']:.2f} | {trading['max_drawdown']:.4f} | "
                f"{classification['macro_f1']:.4f} | "
                f"{classification['directional_accuracy']:.4f} | "
                f"{classification['coverage']:.4f} |"
            )

    lines += [
        "",
        "## Across folds (mean ± std)",
        "",
        "| model | net return | Sharpe | max DD | macro F1 | dir acc | coverage | "
        "trades | positive folds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, stats in summary["per_model"].items():

        def cell(metric: str, places: int = 4) -> str:
            value = stats[metric]
            return f"{value['mean']:.{places}f} ± {value['std']:.{places}f}"

        lines.append(
            f"| {name} | {cell('net_return')} | {cell('sharpe', 2)} | "
            f"{cell('max_drawdown')} | {cell('macro_f1')} | "
            f"{cell('directional_accuracy')} | {cell('coverage')} | "
            f"{cell('n_trades', 1)} | "
            f"{stats['positive_net_return_folds']}/{summary['folds']} |"
        )

    lines += [
        "",
        f"**Verdict:** {summary['verdict']} "
        f"({summary['mtst_beat_baselines_in_folds']}/{summary['folds']} folds).",
        "",
        "These are validation numbers used to choose a candidate. They are not an",
        "out-of-sample result, and historical performance does not guarantee future",
        "profitability.",
        "",
    ]
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expanding-window walk-forward validation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="artifacts/walkforward")
    parser.add_argument("--folds", type=int, default=4)
    # The sealed-test contract, with the same flags and defaults as nn.train.
    # Walk-forward uses them for one purpose: to locate the first sealed row.
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="nn.train's train fraction. Used only to locate the sealed test block.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="nn.train's validation fraction. Used only to locate the sealed test block.",
    )

    # Fold geometry. These fractions are of the *research* region — the rows
    # before the sealed boundary — not of the whole dataset.
    parser.add_argument(
        "--min-train-frac",
        type=float,
        default=0.5,
        help="Fraction of the research region in the first fold's training window.",
    )
    parser.add_argument(
        "--min-train-size", type=int, default=None, help="Rows; overrides --min-train-frac."
    )
    parser.add_argument(
        "--fold-val-frac",
        type=float,
        default=0.15,
        help="Fraction of the research region in each fold's validation window.",
    )
    parser.add_argument(
        "--fold-val-size", type=int, default=None, help="Rows; overrides --fold-val-frac."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "Rows the training window grows by between folds. Defaults to spreading "
            "the folds evenly over the research region after the first training window."
        ),
    )
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


def resolve_sizes(args: argparse.Namespace, research_rows: int) -> tuple[int, int, int]:
    """Turn the CLI's fractions into row counts.

    ``research_rows`` is the number of rows before the sealed boundary, never
    the dataset length: every fraction here is a fraction of what research is
    allowed to use.
    """
    min_train = args.min_train_size or int(research_rows * args.min_train_frac)
    val_size = args.fold_val_size or int(research_rows * args.fold_val_frac)
    if args.step is not None:
        step = args.step
    elif args.folds > 1:
        # Spread the remaining folds evenly over whatever is left of the
        # research region after the first training and validation windows.
        step = max(1, (research_rows - min_train - val_size) // (args.folds - 1))
    else:
        step = 1
    return min_train, val_size, step


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    data = load_research_data(args.dataset)
    boundary = sealed_test_start(data.n_rows, args.train_frac, args.val_frac)
    sealed_split = Split("sealed_test", boundary, data.n_rows)
    min_train, val_size, step = resolve_sizes(args, boundary)
    folds = plan_expanding_folds(boundary, args.folds, min_train, val_size, step)
    logger.warning(
        "%d expanding folds over rows [0, %d) of %d (min_train=%d, val=%d, step=%d). "
        "The sealed test block starts at row %d (%s) and is NOT planned over, "
        "trained on, or evaluated.",
        len(folds),
        boundary,
        data.n_rows,
        min_train,
        val_size,
        step,
        boundary,
        data.period(sealed_split)["start"],
    )

    run = RunConfig(
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
    device = resolve_device(args.device)

    results = []
    for i, splits in enumerate(folds):
        logger.info("--- fold %d/%d ---", i + 1, len(folds))
        results.append(run_fold(i, data, splits, run, device=device))

    summary = summarise(results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "walkforward.json").write_text(
        json.dumps(
            {
                "dataset": data.ds_meta.to_dict(),
                "config": vars(args),
                "sizes": {"min_train": min_train, "val_size": val_size, "step": step},
                "sealed_test": {
                    "start_row": boundary,
                    "period": data.period(sealed_split),
                    "evaluated": False,
                },
                "research_rows": boundary,
                "test_evaluated": False,
                "folds": results,
                "summary": summary,
            },
            indent=2,
            default=str,
        )
    )
    markdown = to_markdown(results, summary, sealed=data.period(sealed_split))
    (out_dir / "walkforward.md").write_text(markdown)
    print(markdown)
    logger.info("Wrote results to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
