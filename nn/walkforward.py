"""Nested walk-forward validation.

    python -m nn.walkforward --dataset data/datasets/binance_BTC_USDT_1h.parquet \
        --folds 4 --epochs 5 --out artifacts/walkforward

One train/validation split tells you how one model did on one stretch of
market. Walk-forward asks the harder question: does the *procedure* keep working
as the market moves? The training window expands, and each fold is scored on the
block that comes after it.

**Each fold has three chronological regions, not two.** ::

    fold 0: [--- train ---][ inner ][ outer ]
    fold 1: [------ train ------][ inner ][ outer ]
    fold 2: [--------- train --------][ inner ][ outer ]

* TRAIN — the scaler is fitted here, the weights are fitted here.
* INNER VALIDATION — early stopping, decision-threshold selection, and any
  other model-selection quantity. It is never reported as fold performance.
* OUTER VALIDATION — the frozen model, at the frozen threshold, measured once.
  This block influences nothing. It is the only block whose numbers become the
  fold's result, the across-fold mean ± std, and the verdict.

The previous version had two regions and used the second one twice: it chose the
early-stopping epoch and the decision threshold on the validation block, then
reported that same block as the fold's performance. Both quantities are fitted
on the data they are then scored on, so the reported numbers were optimistic by
construction — a selection score reported as a result. Splitting the block in
two is what makes the reported number an evaluation instead of a selection.

**Outer blocks never overlap.** ``--step`` defaults to the outer block size, so
consecutive outer blocks are back to back and each row is scored as outer
validation in at most one fold; a step smaller than the outer block is refused
rather than allowed to double-count rows. A fold's inner block may be a
*previous* fold's outer block — by then it is history, and using history to
choose a threshold is the point of the exercise.

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
has not been shown to work. Outer-validation folds are research evidence used
during model development. They are not an out-of-sample result: the sealed test
block remains unopened.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass, replace
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
    score_frozen_split,
)

logger = logging.getLogger(__name__)

MODELS = ("majority_baseline", "momentum_baseline", "mtst")

#: Metrics aggregated across folds: where to find each in an evaluation report.
#: Every one of them is read from the fold's **outer** validation report.
SUMMARY_METRICS = {
    "macro_f1": ("classification", "macro_f1"),
    "directional_accuracy": ("classification", "directional_accuracy"),
    "coverage": ("classification", "coverage"),
    "calibration_error": ("classification", "calibration_error"),
    "n_trades": ("trading", "n_trades"),
    "net_return": ("trading", "net_return"),
    "sharpe": ("trading", "sharpe"),
    "max_drawdown": ("trading", "max_drawdown"),
}


@dataclass(frozen=True)
class FoldPlan:
    """The three chronological regions of one fold, in time order."""

    train: Split
    inner: Split
    outer: Split


def plan_nested_folds(
    boundary: int,
    folds: int,
    min_train: int,
    inner_size: int,
    outer_size: int,
    step: int,
) -> list[FoldPlan]:
    """Expanding training windows, each followed by an inner and an outer block.

    ``boundary`` is the first sealed row — normally
    :func:`nn.dataset.sealed_test_start`. **Every row this function hands out —
    training, inner validation and outer validation alike — is strictly below
    it.** The planner takes the boundary rather than the dataset length
    precisely because the two are not the same number: planning over the dataset
    length is what silently walked the last folds into the sealed block.

    Fold ``k`` trains on rows ``[0, min_train + k*step)``, selects on the
    ``inner_size`` rows that follow, and is *reported* on the ``outer_size`` rows
    after those. Training therefore only ever grows forward in time.

    ``step`` must be at least ``outer_size``: outer blocks advance by ``step``,
    so a smaller step would make consecutive outer blocks overlap and score the
    same rows twice. At exactly ``outer_size`` — the default — the outer blocks
    are contiguous and partition one stretch of the research region.

    Raises when the research region is too short for the requested plan, rather
    than quietly returning fewer folds than asked for — or, worse, borrowing the
    rows it is short by from the sealed block.
    """
    if folds < 1:
        raise ValueError("folds must be at least 1")
    if min_train < 1 or inner_size < 1 or outer_size < 1 or step < 1:
        raise ValueError(
            f"min_train ({min_train}), inner_size ({inner_size}), outer_size "
            f"({outer_size}) and step ({step}) must all be positive"
        )
    if step < outer_size:
        raise ValueError(
            f"step ({step}) is smaller than the outer validation block ({outer_size}), "
            "so consecutive outer blocks would overlap and the same rows would be "
            "reported as the result of two folds. Use --step >= --outer-val-size."
        )

    needed = min_train + step * (folds - 1) + inner_size + outer_size
    if needed > boundary:
        raise ValueError(
            f"need {needed} rows for {folds} nested folds with min_train={min_train}, "
            f"inner_size={inner_size}, outer_size={outer_size}, step={step}, but only "
            f"{boundary} rows lie before the sealed test block. Reduce --folds, "
            "--min-train-frac, --inner-val-frac or --outer-val-frac; the sealed rows "
            "are not available to make up the difference."
        )

    plans = []
    for k in range(folds):
        train_end = min_train + k * step
        inner_end = train_end + inner_size
        plans.append(
            FoldPlan(
                train=Split("train", 0, train_end),
                inner=Split("inner_validation", train_end, inner_end),
                outer=Split("outer_validation", inner_end, inner_end + outer_size),
            )
        )

    # The arithmetic above already guarantees both properties below; assert them
    # anyway, because the failures they guard against are invisible in the
    # output. Sealed rows scored under the name "validation" look exactly like
    # validation, and a row reported twice looks exactly like two folds agreeing.
    for k, plan in enumerate(plans):
        if plan.train.end > boundary or plan.inner.end > boundary:
            raise AssertionError(
                f"fold {k} crosses the sealed boundary {boundary}: train ends "
                f"{plan.train.end}, inner validation ends {plan.inner.end}"
            )
        if plan.outer.end > boundary:
            raise AssertionError(
                f"fold {k} outer validation ends at {plan.outer.end}, at or beyond "
                f"the sealed boundary {boundary}"
            )
        if k and plan.outer.start < plans[k - 1].outer.end:
            raise AssertionError(
                f"fold {k} outer validation starts at {plan.outer.start}, inside "
                f"fold {k - 1}'s outer block which ends at {plans[k - 1].outer.end}"
            )
    return plans


def run_fold(
    fold: int,
    data: ResearchData,
    plan: FoldPlan,
    run: RunConfig,
    *,
    device: Any,
) -> dict[str, Any]:
    """Train, select on inner validation, then measure once on outer validation.

    Every fitted quantity is local to this call and comes from ``plan.train`` and
    ``plan.inner`` only. ``plan.outer`` reaches exactly one function —
    :func:`nn.train.score_frozen_split`, which fits nothing.
    """
    prepared = prepare_research_windows(data, plan.train, plan.inner, run.seq_len)
    # Vary the seed per fold so a lucky initialisation cannot flatter the whole
    # run, while staying reproducible for a given --seed.
    selection = fit_and_validate(
        data, prepared, replace(run, seed=run.seed + fold), device=device
    )

    # The model, the threshold and the baselines are now frozen. `selection`
    # also carries reports on the inner block; they are deliberately not
    # reported as this fold's performance — the threshold was chosen there.
    outer_reports, idx_outer = score_frozen_split(
        data,
        prepared.scaler,
        plan.outer,
        run.seq_len,
        model=selection.model,
        baselines=selection.baselines,
        threshold=selection.threshold,
        device=device,
    )

    return {
        "fold": fold,
        "seed": run.seed + fold,
        "samples": {
            "train": len(prepared.X_train),
            "inner_validation": len(prepared.X_val),
            "outer_validation": len(idx_outer),
        },
        "periods": {
            "train": data.period(plan.train),
            "inner_validation": data.period(plan.inner),
            "outer_validation": data.period(plan.outer),
        },
        "selection": {
            "best_epoch": selection.train_info["best_epoch"],
            "threshold": selection.threshold,
            "inner_validation_loss": selection.train_info["best_val_loss"],
        },
        "outer_validation": outer_reports,
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation of each metric across folds, per model.

    Reads ``outer_validation`` and nothing else. The inner block chose the
    threshold and the epoch; aggregating it here would put the selection score
    back into the result it was supposed to be separated from.
    """
    summary: dict[str, Any] = {
        "folds": len(results),
        "aggregated_from": "outer_validation",
        "per_model": {},
    }

    for name in MODELS:
        stats: dict[str, Any] = {}
        for metric, (section, key) in SUMMARY_METRICS.items():
            values = [float(r["outer_validation"][name][section][key]) for r in results]
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
        if r["outer_validation"]["mtst"]["trading"]["net_return"]
        > max(
            r["outer_validation"]["majority_baseline"]["trading"]["net_return"],
            r["outer_validation"]["momentum_baseline"]["trading"]["net_return"],
        )
    )
    summary["mtst_beat_baselines_in_folds"] = beat
    summary["verdict"] = (
        "model beat both baselines in a majority of outer-validation folds"
        if beat * 2 > len(results)
        else "model did NOT consistently beat the baselines on outer validation"
    )
    return summary


def to_markdown(
    results: list[dict[str, Any]], summary: dict[str, Any], sealed: dict[str, Any]
) -> str:
    lines = [
        "# Nested walk-forward validation",
        "",
        "Expanding training window. Each fold has three chronological regions:",
        "train -> inner validation -> outer validation. The scaler and the weights",
        "are fitted on train; early stopping and the decision threshold are chosen on",
        "inner validation; the frozen model is measured once on outer validation.",
        "**Only the outer block is reported below.** Outer blocks do not overlap, so",
        "no row is reported as a result twice.",
        "",
        f"**Sealed test block:** rows {sealed['row_range'][0]}-{sealed['row_range'][1]}, "
        f"{sealed['start'][:10]} to {sealed['end'][:10]}. No fold below plans, trains",
        "on, selects on, or scores a row at or after that boundary.",
        "",
        "## Fold geometry",
        "",
        "| fold | train rows | inner rows | outer rows | outer period | threshold | "
        "best epoch |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        periods = result["periods"]
        outer = periods["outer_validation"]
        lines.append(
            f"| {result['fold']} | {periods['train']['row_range'][0]}-"
            f"{periods['train']['row_range'][1]} | "
            f"{periods['inner_validation']['row_range'][0]}-"
            f"{periods['inner_validation']['row_range'][1]} | "
            f"{outer['row_range'][0]}-{outer['row_range'][1]} | "
            f"{outer['start'][:10]} to {outer['end'][:10]} | "
            f"{result['selection']['threshold']:.2f} | "
            f"{result['selection']['best_epoch']} |"
        )

    lines += [
        "",
        "## Outer validation (the reported result)",
        "",
        "| fold | outer period | model | trades | net return | Sharpe | max DD | "
        "macro F1 | dir acc | coverage | calib err |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        outer = result["periods"]["outer_validation"]
        window = f"{outer['start'][:10]} to {outer['end'][:10]}"
        for name in MODELS:
            report = result["outer_validation"][name]
            trading, classification = report["trading"], report["classification"]
            lines.append(
                f"| {result['fold']} | {window} | {name} | "
                f"{trading['n_trades']} | {trading['net_return']:+.4f} | "
                f"{trading['sharpe']:.2f} | {trading['max_drawdown']:.4f} | "
                f"{classification['macro_f1']:.4f} | "
                f"{classification['directional_accuracy']:.4f} | "
                f"{classification['coverage']:.4f} | "
                f"{classification['calibration_error']:.4f} |"
            )

    lines += [
        "",
        "## Across folds, outer validation only (mean ± std)",
        "",
        "| model | net return | Sharpe | max DD | macro F1 | dir acc | coverage | "
        "calib err | trades | positive folds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, stats in summary["per_model"].items():

        def cell(metric: str, places: int = 4) -> str:
            value = stats[metric]
            return f"{value['mean']:.{places}f} ± {value['std']:.{places}f}"

        lines.append(
            f"| {name} | {cell('net_return')} | {cell('sharpe', 2)} | "
            f"{cell('max_drawdown')} | {cell('macro_f1')} | "
            f"{cell('directional_accuracy')} | {cell('coverage')} | "
            f"{cell('calibration_error')} | {cell('n_trades', 1)} | "
            f"{stats['positive_net_return_folds']}/{summary['folds']} |"
        )

    lines += [
        "",
        f"**Verdict:** {summary['verdict']} "
        f"({summary['mtst_beat_baselines_in_folds']}/{summary['folds']} folds).",
        "",
        "These are outer-validation numbers from model development. Nothing was",
        "fitted on them, which is what makes them worth reading — but the folds were",
        "run repeatedly while the method was being built, so they are research",
        "evidence, not an out-of-sample result and not a claim of profitability. The",
        "sealed test block remains unopened.",
        "",
    ]
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested walk-forward validation.")
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
        default=0.45,
        help="Fraction of the research region in the first fold's training window.",
    )
    parser.add_argument(
        "--min-train-size", type=int, default=None, help="Rows; overrides --min-train-frac."
    )
    parser.add_argument(
        "--inner-val-frac",
        type=float,
        default=0.10,
        help=(
            "Fraction of the research region in each fold's INNER validation block: "
            "early stopping and threshold selection. Never reported as fold performance."
        ),
    )
    parser.add_argument(
        "--inner-val-size", type=int, default=None, help="Rows; overrides --inner-val-frac."
    )
    parser.add_argument(
        "--outer-val-frac",
        type=float,
        default=0.10,
        help=(
            "Fraction of the research region in each fold's OUTER validation block: "
            "the frozen evaluation that becomes the fold's reported result."
        ),
    )
    parser.add_argument(
        "--outer-val-size", type=int, default=None, help="Rows; overrides --outer-val-frac."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "Rows the training window grows by between folds. Defaults to the outer "
            "validation size, which makes consecutive outer blocks contiguous. Must "
            "be at least that size, or outer blocks would overlap."
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


def resolve_sizes(args: argparse.Namespace, research_rows: int) -> tuple[int, int, int, int]:
    """Turn the CLI's fractions into row counts.

    ``research_rows`` is the number of rows before the sealed boundary, never
    the dataset length: every fraction here is a fraction of what research is
    allowed to use.

    ``step`` defaults to the outer block size so that outer blocks tile the
    research region without overlapping. A larger step spreads the folds further
    apart; a smaller one is rejected by :func:`plan_nested_folds`.
    """
    min_train = args.min_train_size or int(research_rows * args.min_train_frac)
    inner_size = args.inner_val_size or int(research_rows * args.inner_val_frac)
    outer_size = args.outer_val_size or int(research_rows * args.outer_val_frac)
    step = args.step if args.step is not None else outer_size
    return min_train, inner_size, outer_size, step


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    data = load_research_data(args.dataset)
    boundary = sealed_test_start(data.n_rows, args.train_frac, args.val_frac)
    sealed_split = Split("sealed_test", boundary, data.n_rows)
    min_train, inner_size, outer_size, step = resolve_sizes(args, boundary)
    folds = plan_nested_folds(boundary, args.folds, min_train, inner_size, outer_size, step)
    logger.warning(
        "%d nested folds over rows [0, %d) of %d (min_train=%d, inner=%d, outer=%d, "
        "step=%d). Outer blocks: %s — disjoint, and only these are reported. The "
        "sealed test block starts at row %d (%s) and is NOT planned over, trained on, "
        "selected on, or evaluated.",
        len(folds),
        boundary,
        data.n_rows,
        min_train,
        inner_size,
        outer_size,
        step,
        ", ".join(f"[{p.outer.start}, {p.outer.end})" for p in folds),
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
    for i, plan in enumerate(folds):
        logger.info("--- fold %d/%d ---", i + 1, len(folds))
        results.append(run_fold(i, data, plan, run, device=device))

    summary = summarise(results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "walkforward.json").write_text(
        json.dumps(
            {
                "dataset": data.ds_meta.to_dict(),
                "config": vars(args),
                "sizes": {
                    "min_train": min_train,
                    "inner_val_size": inner_size,
                    "outer_val_size": outer_size,
                    "step": step,
                },
                "sealed_test": {
                    "start_row": boundary,
                    "period": data.period(sealed_split),
                    "evaluated": False,
                },
                "research_rows": boundary,
                "test_evaluated": False,
                "reported_block": "outer_validation",
                "folds": results,
                "summary": summary,
            },
            indent=2,
            default=str,
        )
        # Trailing newline: these artifacts get committed, and a file without
        # one fails the repository's end-of-file hook.
        + "\n"
    )
    markdown = to_markdown(results, summary, sealed=data.period(sealed_split))
    (out_dir / "walkforward.md").write_text(markdown)
    print(markdown)
    logger.info("Wrote results to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
