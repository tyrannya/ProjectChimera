"""Validation-only experiment runner.

    python -m nn.experiment --dataset data/datasets/binance_BTC_USDT_1h.parquet \
        --seed 1 2 --lr 1e-4 3e-4 --seq-len 32 64 --epochs 10

Runs a **predeclared** grid — the full cartesian product of the values given on
the command line, enumerated and written to the output file *before* any
training starts — and scores every configuration on validation.

The test split is never touched. Not "not by default": this module has no code
path that windows it. It calls :func:`nn.train.prepare_research_windows`, whose
signature takes a training and a validation split and nothing else, so the model
selection done here cannot consume the sealed estimate that a later
``nn.train`` run will spend once.

Three things it deliberately does *not* do, because a research tool that hides
its failures is worse than none:

* **Failed runs are kept.** A configuration that raises is recorded with its
  error and appears in the output and the console summary. It is not skipped.
* **Baselines travel with every run.** The majority-class and momentum
  baselines are refitted per configuration and reported next to it, because a
  ranking of models against each other says nothing about whether any of them
  beat a rule.
* **It does not claim profitability.** It ranks configurations by a stated
  validation objective. That is a selection signal, not a result.

Output: ``experiments.json`` (full reports) and ``experiments.csv`` (one row per
run, for a spreadsheet or a plot).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from nn.dataset import chronological_split
from nn.train import (
    RunConfig,
    fit_and_validate,
    load_research_data,
    prepare_research_windows,
    resolve_device,
)

logger = logging.getLogger(__name__)

#: The dimensions the grid can vary, and the CLI flag for each.
GRID_DIMENSIONS = (
    "seed",
    "lr",
    "seq_len",
    "d_model",
    "n_heads",
    "num_layers",
    "dropout",
)

#: Validation objectives, all "higher is better".
#:
#: ``net_return`` — net return after the configured round-trip costs, on the
#: validation block, at the threshold selected on that same block. The default,
#: because it is the only one denominated in the thing the system exists to
#: produce.
#: ``sharpe`` — the same trades, risk-adjusted and annualised.
#: ``macro_f1`` — classification quality, unweighted across the three classes.
OBJECTIVES: dict[str, Callable[[dict[str, Any]], float]] = {
    "net_return": lambda report: report["trading"]["net_return"],
    "sharpe": lambda report: report["trading"]["sharpe"],
    "macro_f1": lambda report: report["classification"]["macro_f1"],
}

#: Columns of experiments.csv, in order.
CSV_COLUMNS = (
    ["run_id", "status", "objective_name", "objective"]
    + list(GRID_DIMENSIONS)
    + [
        "epochs",
        "batch_size",
        "patience",
        "best_epoch",
        "best_val_loss",
        "threshold",
        "macro_f1",
        "directional_accuracy",
        "coverage",
        "n_trades",
        "net_return",
        "sharpe",
        "max_drawdown",
        "calibration_error",
        "majority_macro_f1",
        "majority_net_return",
        "momentum_macro_f1",
        "momentum_net_return",
        "error",
    ]
)


def build_grid(values: dict[str, list[Any]], base: RunConfig) -> list[RunConfig]:
    """Expand the cartesian product of ``values`` over ``base``.

    Order is fixed by :data:`GRID_DIMENSIONS`, so the same flags always produce
    the same grid in the same order.
    """
    axes = [values[name] for name in GRID_DIMENSIONS]
    return [
        replace(base, **dict(zip(GRID_DIMENSIONS, combination)))
        for combination in itertools.product(*axes)
    ]


def flatten_run(record: dict[str, Any], objective_name: str) -> dict[str, Any]:
    """One experiment record as a flat CSV row."""
    row: dict[str, Any] = {name: "" for name in CSV_COLUMNS}
    row.update(
        run_id=record["run_id"],
        status=record["status"],
        objective_name=objective_name,
        objective=record.get("objective", ""),
        error=record.get("error") or "",
        **{name: record["config"][name] for name in GRID_DIMENSIONS},
        **{name: record["config"][name] for name in ("epochs", "batch_size", "patience")},
    )
    if record["status"] != "ok":
        return row

    mtst = record["validation"]["mtst"]
    row.update(
        best_epoch=record["best_epoch"],
        best_val_loss=record["best_val_loss"],
        threshold=record["threshold"],
        macro_f1=mtst["classification"]["macro_f1"],
        directional_accuracy=mtst["classification"]["directional_accuracy"],
        coverage=mtst["classification"]["coverage"],
        calibration_error=mtst["classification"]["calibration_error"],
        n_trades=mtst["trading"]["n_trades"],
        net_return=mtst["trading"]["net_return"],
        sharpe=mtst["trading"]["sharpe"],
        max_drawdown=mtst["trading"]["max_drawdown"],
        majority_macro_f1=record["validation"]["majority_baseline"]["classification"][
            "macro_f1"
        ],
        majority_net_return=record["validation"]["majority_baseline"]["trading"]["net_return"],
        momentum_macro_f1=record["validation"]["momentum_baseline"]["classification"][
            "macro_f1"
        ],
        momentum_net_return=record["validation"]["momentum_baseline"]["trading"]["net_return"],
    )
    return row


def to_markdown(records: list[dict[str, Any]], objective_name: str) -> str:
    """A ranked table for the console, best first."""
    ok = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]

    lines = [
        f"# Experiment ranking by validation {objective_name}",
        "",
        "Validation metrics only — the test split was not evaluated.",
        "",
        "| rank | run | seed | lr | seq_len | d_model | heads | layers | dropout | "
        "macro F1 | dir acc | coverage | trades | net return | Sharpe | max DD | "
        "best baseline net |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- | --- |",
    ]
    for rank, record in enumerate(ok, start=1):
        config = record["config"]
        mtst = record["validation"]["mtst"]
        cls, trade = mtst["classification"], mtst["trading"]
        best_baseline = max(
            record["validation"][name]["trading"]["net_return"]
            for name in ("majority_baseline", "momentum_baseline")
        )
        lines.append(
            f"| {rank} | {record['run_id']} | {config['seed']} | {config['lr']:.1e} | "
            f"{config['seq_len']} | {config['d_model']} | {config['n_heads']} | "
            f"{config['num_layers']} | {config['dropout']} | {cls['macro_f1']:.4f} | "
            f"{cls['directional_accuracy']:.4f} | {cls['coverage']:.4f} | "
            f"{trade['n_trades']} | {trade['net_return']:+.4f} | {trade['sharpe']:.3f} | "
            f"{trade['max_drawdown']:.4f} | {best_baseline:+.4f} |"
        )

    if failed:
        lines += ["", f"## Failed runs ({len(failed)})", ""]
        for record in failed:
            lines.append(f"- run {record['run_id']} {record['config']}: {record['error']}")

    lines += [
        "",
        "Ranking is a model-selection signal on one validation block, not a result. "
        "Nothing here says the strategy is profitable.",
        "",
    ]
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a predeclared grid of configurations against validation only."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="artifacts/experiments")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--objective",
        default="net_return",
        choices=sorted(OBJECTIVES),
        help="Validation metric to rank by, highest first. Default: net_return.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # Fixed across the grid.
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)

    # The grid itself. Defaults are single-valued, so omitting a flag pins that
    # dimension to the nn.train default rather than searching it.
    defaults = RunConfig()
    grid = parser.add_argument_group("grid dimensions (each takes one or more values)")
    grid.add_argument("--seed", type=int, nargs="+", default=[defaults.seed])
    grid.add_argument("--lr", type=float, nargs="+", default=[defaults.lr])
    grid.add_argument("--seq-len", type=int, nargs="+", default=[defaults.seq_len])
    grid.add_argument("--d-model", type=int, nargs="+", default=[defaults.d_model])
    grid.add_argument("--n-heads", type=int, nargs="+", default=[defaults.n_heads])
    grid.add_argument("--num-layers", type=int, nargs="+", default=[defaults.num_layers])
    grid.add_argument("--dropout", type=float, nargs="+", default=[defaults.dropout])
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    data = load_research_data(args.dataset)
    plan = chronological_split(data.n_rows, args.train_frac, args.val_frac)
    device = resolve_device(args.device)
    objective = OBJECTIVES[args.objective]

    base = RunConfig(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
    configs = build_grid({name: getattr(args, name) for name in GRID_DIMENSIONS}, base)

    logger.warning(
        "Validation-only experiment: %d configurations, validation rows %s to %s. "
        "The test split (%d rows) REMAINS SEALED.",
        len(configs),
        data.period(plan.validation)["start"],
        data.period(plan.validation)["end"],
        len(plan.test),
    )

    records: list[dict[str, Any]] = []
    # Windowing is the expensive part and depends only on seq_len, so runs that
    # share one reuse it. Sorting by seq_len keeps that to a single rebuild per
    # distinct value; run_id still refers to the declared grid order.
    prepared = None
    prepared_seq_len = None
    for run_id, config in sorted(enumerate(configs), key=lambda item: item[1].seq_len):
        record: dict[str, Any] = {
            "run_id": run_id,
            "config": config.to_dict(),
            "status": "ok",
            "error": None,
        }
        try:
            if prepared_seq_len != config.seq_len:
                prepared = prepare_research_windows(
                    data, plan.train, plan.validation, config.seq_len
                )
                prepared_seq_len = config.seq_len
            result = fit_and_validate(data, prepared, config, device=device)
        except Exception as exc:  # noqa: BLE001 - a failed run is a result, not a crash
            logger.error("run %d failed: %s: %s", run_id, type(exc).__name__, exc)
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        else:
            record.update(
                threshold=result.threshold,
                best_epoch=result.train_info["best_epoch"],
                best_val_loss=result.train_info["best_val_loss"],
                validation=result.reports,
                objective=objective(result.reports["mtst"]),
            )
            logger.info(
                "run %d %s -> %s=%.6f (macro F1 %.4f, %d trades)",
                run_id,
                config.to_dict(),
                args.objective,
                record["objective"],
                result.reports["mtst"]["classification"]["macro_f1"],
                result.reports["mtst"]["trading"]["n_trades"],
            )
        records.append(record)

    # Rank: objective first, macro F1 as the tie-break, declared grid order last.
    ok = sorted(
        (r for r in records if r["status"] == "ok"),
        key=lambda r: (r["objective"], r["validation"]["mtst"]["classification"]["macro_f1"]),
        reverse=True,
    )
    failed = [r for r in records if r["status"] != "ok"]
    ranked = ok + sorted(failed, key=lambda r: r["run_id"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiments.json").write_text(
        json.dumps(
            {
                "argv": list(argv) if argv is not None else sys.argv[1:],
                "dataset": data.ds_meta.to_dict(),
                "split_plan": plan.to_dict(),
                "periods": {
                    "train": data.period(plan.train),
                    "validation": data.period(plan.validation),
                },
                "test_evaluated": False,
                "objective": args.objective,
                "grid": {name: getattr(args, name) for name in GRID_DIMENSIONS},
                "fixed": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "train_frac": args.train_frac,
                    "val_frac": args.val_frac,
                },
                "n_runs": len(records),
                "n_failed": len(failed),
                "runs": ranked,
                "best": ok[0] if ok else None,
            },
            indent=2,
            default=str,
        )
    )

    with (out_dir / "experiments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in ranked:
            writer.writerow(flatten_run(record, args.objective))

    print(to_markdown(ranked, args.objective))
    logger.info("Wrote %s and %s", out_dir / "experiments.json", out_dir / "experiments.csv")
    if failed:
        logger.error(
            "%d of %d runs failed; see the output for each error", len(failed), len(records)
        )
    if not ok:
        logger.error("every configuration failed — nothing was selected")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
