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
has not been shown to work. Per-sample outer predictions go beside them in
``outer_predictions.parquet``, because the aggregates cannot answer "did the
longs or the shorts lose the money?" and nothing can answer it after the fact
without them. Outer rows only: writing a sealed-test prediction is refused.

Outer-validation folds are research evidence used during model development. They
are not an out-of-sample result: the sealed test block remains unopened.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chimera.contracts import CLASS_ORDER
from nn import evaluate as ev
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

#: Per-sample outer-validation predictions, written beside walkforward.json.
#:
#: The reports in walkforward.json are aggregates; they cannot answer "did the
#: longs or the shorts lose the money?". These rows can, and they are the only
#: way to answer it after the fact without retraining. Outer rows only — a
#: sealed-test prediction is never written, asserted before the file is created.
PREDICTIONS_NAME = "outer_predictions.parquet"

#: Where the fold-level economic references are stored, beside the models.
#:
#: Not a model and not a baseline, so not in :data:`MODELS`: nothing predicts
#: anything here. It sits inside ``outer_validation`` because it describes that
#: block and must travel with it.
REFERENCES_KEY = "economic_references"

#: Metrics aggregated across folds: where to find each in an evaluation report.
#: Every one of them is read from the fold's **outer** validation report.
#:
#: ``exposure`` is aggregated because "how often was capital actually deployed"
#: is not a footnote to the risk-adjusted numbers, it is how to read them.
SUMMARY_METRICS = {
    "macro_f1": ("classification", "macro_f1"),
    "directional_accuracy": ("classification", "directional_accuracy"),
    "coverage": ("classification", "coverage"),
    "calibration_error": ("classification", "calibration_error"),
    "n_trades": ("trading", "n_trades"),
    "net_return": ("trading", "net_return"),
    "annualised_sharpe": ("trading", "annualised_sharpe"),
    "per_trade_sharpe": ("trading", "per_trade_sharpe"),
    "exposure": ("trading", "exposure"),
    "max_drawdown": ("trading", "max_drawdown"),
}

#: Metrics that may legitimately be ``None`` in a fold report.
#:
#: A risk-adjusted statistic is undefined when the portfolio never moved — no
#: trades, or no variance. That is a real state, reported as ``None`` rather
#: than ``0.0`` so it cannot be read as a measurement, and the aggregation below
#: has to carry it through instead of averaging a fabricated zero.
NULLABLE_METRICS = ("annualised_sharpe", "per_trade_sharpe")


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
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train, select on inner validation, then measure once on outer validation.

    Returns the fold's record for the JSON report and its per-sample outer
    predictions, which the caller concatenates and persists.

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
    scored = score_frozen_split(
        data,
        prepared.scaler,
        plan.outer,
        run.seq_len,
        model=selection.model,
        baselines=selection.baselines,
        threshold=selection.threshold,
        device=device,
    )
    outer_reports, idx_outer = dict(scored.reports), scored.idx
    # Beside the models, not among them: these make no predictions. Kept inside
    # the outer block's reports because they describe that block's window and
    # must not drift away from the numbers they qualify.
    outer_reports[REFERENCES_KEY] = scored.economic_references
    predictions = outer_predictions(fold, run.seed + fold, data, scored, selection.threshold)

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
    }, predictions


def outer_predictions(
    fold: int,
    seed: int,
    data: ResearchData,
    scored: Any,
    threshold: float,
) -> pd.DataFrame:
    """One row per outer-validation sample: what was predicted, and what happened.

    ``selected_action`` comes from :func:`nn.evaluate.signals_from_proba` — the
    same function the reports use — so the persisted action is the action that
    was scored, not a second interpretation of the threshold. Classes are the
    dataset's own integer encoding (``CLASS_ORDER``), so nothing has to agree
    about a second set of names.
    """
    actions = ev.signals_from_proba(scored.proba, threshold)
    short_idx, hold_idx, long_idx = range(len(CLASS_ORDER))
    return pd.DataFrame(
        {
            "fold": np.full(len(scored.idx), fold, dtype=np.int64),
            "seed": np.full(len(scored.idx), seed, dtype=np.int64),
            "row_index": scored.idx.astype(np.int64),
            "timestamp": data.dates[scored.idx],
            "true_target": scored.y.astype(np.int64),
            "future_return": data.future_return[scored.idx],
            "p_short": scored.proba[:, short_idx],
            "p_hold": scored.proba[:, hold_idx],
            "p_long": scored.proba[:, long_idx],
            "selected_action": actions.astype(np.int64),
            "threshold": np.full(len(scored.idx), float(threshold)),
        }
    )


def write_predictions(out_dir: Path, frames: list[pd.DataFrame], boundary: int) -> Path:
    """Write the per-sample outer predictions, refusing to persist a sealed row.

    The outer blocks are below the boundary by construction, so this assertion
    should never fire. It is here because the file it guards is the one thing
    walk-forward writes that contains per-row market outcomes, and a silent
    off-by-one would put sealed rows on disk under a research filename.
    """
    predictions = pd.concat(frames, ignore_index=True)
    highest = int(predictions["row_index"].max())
    if highest >= boundary:
        raise AssertionError(
            f"refusing to write outer predictions: row {highest} is at or beyond the "
            f"sealed test block starting at {boundary}"
        )
    path = out_dir / PREDICTIONS_NAME
    predictions.to_parquet(path, index=False)
    return path


def spread(values: list[float | None]) -> dict[str, Any]:
    """Mean and standard deviation across folds, carrying undefined entries.

    A ``None`` is a metric that does not exist for that fold — a portfolio that
    never moved has no Sharpe — not a zero. Averaging it as zero would invent a
    measurement, so the mean is taken over the folds where it is defined and
    ``defined_folds`` says how many those were. All-undefined stays undefined.
    """
    defined = [v for v in values if v is not None]
    return {
        "mean": round(statistics.fmean(defined), 6) if defined else None,
        # Sample standard deviation needs two folds; one fold has none.
        "std": round(statistics.stdev(defined), 6) if len(defined) > 1 else 0.0,
        "values": [None if v is None else round(v, 6) for v in values],
        "defined_folds": len(defined),
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
            raw = [r["outer_validation"][name][section][key] for r in results]
            stats[metric] = spread([None if v is None else float(v) for v in raw])
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
    summary["economic_comparison"] = economic_comparison(results)
    summary["verdict"] = verdict(summary, len(results))
    return summary


def economic_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Did the model make money, beat doing nothing, and beat owning the asset?

    Three questions the baseline comparison cannot answer, computed from the
    economic references each fold recorded. CASH is the zero line, so "beat
    CASH" is exactly "net return above zero" — stated as a policy comparison
    rather than a sign check, because that is what a reader is asking.
    """
    net_returns = [r["outer_validation"]["mtst"]["trading"]["net_return"] for r in results]
    holds = [
        (r["outer_validation"].get(REFERENCES_KEY) or {}).get("buy_and_hold") or {}
        for r in results
    ]
    available = [h for h in holds if h.get("available")]
    hold_returns = [h["net_return"] for h in available]

    return {
        "folds": len(results),
        "references_available": len(available),
        "mean_net_return": round(statistics.fmean(net_returns), 6) if net_returns else None,
        "mean_buy_and_hold_net_return": (
            round(statistics.fmean(hold_returns), 6) if hold_returns else None
        ),
        # CASH earns exactly zero and pays no costs, so this is the count of
        # folds in which trading was better than not trading at all.
        "beat_cash_folds": sum(1 for v in net_returns if v > 0),
        "beat_buy_and_hold_folds": sum(
            1
            for r, h in zip(results, holds)
            if h.get("available")
            and r["outer_validation"]["mtst"]["trading"]["net_return"] > h["net_return"]
        ),
    }


def verdict(summary: dict[str, Any], folds: int) -> str:
    """One sentence that cannot be quoted into meaning something it does not.

    The statistical clause and the economic clause are always both present. The
    previous version emitted "model beat both baselines in a majority of
    outer-validation folds" on its own — a true sentence that was read as
    evidence of profitability while the model was losing money in every fold it
    won. Beating majority-class and momentum is a floor test; whether anything
    was earned is a separate question with a separate answer, and neither
    sentence is allowed to stand without the other.
    """
    beat = summary["mtst_beat_baselines_in_folds"]
    economic = summary["economic_comparison"]
    mean_net = economic["mean_net_return"]

    statistical = (
        f"beat both statistical/rule baselines in {beat}/{folds} outer folds"
        if beat * 2 > folds
        else f"did NOT consistently beat the statistical/rule baselines "
        f"({beat}/{folds} outer folds)"
    )

    cash = economic["beat_cash_folds"]
    if mean_net is None:
        money = "no net return was recorded"
    elif mean_net > 0:
        money = (
            f"made money on outer validation (mean net return {mean_net:+.4f}), "
            f"beating CASH in {cash}/{folds} folds"
        )
    else:
        money = (
            f"LOST money on outer validation (mean net return {mean_net:+.4f}), "
            f"beating CASH — doing nothing — in only {cash}/{folds} folds"
        )

    hold = economic["mean_buy_and_hold_net_return"]
    if hold is None:
        holding = "buy-and-hold was not recorded for these folds"
    else:
        holding = (
            f"buy-and-hold returned {hold:+.4f} on the same windows and was beaten in "
            f"{economic['beat_buy_and_hold_folds']}/{folds} folds"
        )

    return (
        f"Model {statistical}, and {money}; {holding}. Beating majority-class and "
        "momentum is a floor test, not evidence of profitability."
    )


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
        "### Statistical / rule baselines and the model",
        "",
        "What these answer: **did the model learn more than a trivial rule?** They are a",
        "floor, not a business case. Beating them says nothing about whether money was",
        "made — that is the next section.",
        "",
        f"`ann. Sharpe` is {ev.SHARPE_BASIS}. `n/a` means undefined, not zero: a",
        "portfolio that never moved has no Sharpe. Annualising magnifies whatever",
        "happened in a short block, so compare folds against each other and against",
        "the references below — a single fold's figure is not an expected annual result.",
        "",
        "| fold | outer period | model | trades | net return | ann. Sharpe | "
        "per-trade Sharpe | exposure | max DD | macro F1 | dir acc | coverage | "
        "calib err |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
                f"{ev.number(trading['annualised_sharpe'], '.2f')} | "
                f"{ev.number(trading['per_trade_sharpe'], '.2f')} | "
                f"{trading['exposure']:.4f} | {trading['max_drawdown']:.4f} | "
                f"{classification['macro_f1']:.4f} | "
                f"{classification['directional_accuracy']:.4f} | "
                f"{classification['coverage']:.4f} | "
                f"{classification['calibration_error']:.4f} |"
            )

    lines += _references_markdown(results)

    lines += [
        "",
        "### Across folds, outer validation only (mean ± std)",
        "",
        "| model | net return | ann. Sharpe | per-trade Sharpe | exposure | max DD | "
        "macro F1 | dir acc | coverage | calib err | trades | positive folds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, stats in summary["per_model"].items():

        def cell(metric: str, places: int = 4) -> str:
            value = stats[metric]
            if value["mean"] is None:
                return f"n/a (0/{summary['folds']} folds)"
            body = f"{value['mean']:.{places}f} ± {value['std']:.{places}f}"
            missing = summary["folds"] - value["defined_folds"]
            return (
                body
                if not missing
                else f"{body} ({value['defined_folds']}/{summary['folds']})"
            )

        lines.append(
            f"| {name} | {cell('net_return')} | {cell('annualised_sharpe', 2)} | "
            f"{cell('per_trade_sharpe', 2)} | {cell('exposure')} | "
            f"{cell('max_drawdown')} | {cell('macro_f1')} | "
            f"{cell('directional_accuracy')} | {cell('coverage')} | "
            f"{cell('calibration_error')} | {cell('n_trades', 1)} | "
            f"{stats['positive_net_return_folds']}/{summary['folds']} |"
        )

    lines += [
        "",
        f"**Verdict:** {summary['verdict']}",
        "",
        "These are outer-validation numbers from model development. Nothing was",
        "fitted on them, which is what makes them worth reading — but the folds were",
        "run repeatedly while the method was being built, so they are research",
        "evidence, not an out-of-sample result and not a claim of profitability. The",
        "sealed test block remains unopened.",
        "",
    ]
    return "\n".join(lines)


def _references_markdown(results: list[dict[str, Any]]) -> list[str]:
    """The economic references table: CASH and buy-and-hold, per fold.

    Deliberately its own section with its own heading. Putting a no-trade policy
    in the same table as the models invites reading it as one more contender
    that happened to score badly, when it is the line every one of them has to
    clear before any of the other columns mean anything.
    """
    lines = [
        "",
        "### Economic references (not models, not baselines)",
        "",
        "What these answer: **did the strategy make money, did it beat doing nothing,",
        "and did it beat simply owning the asset?** CASH never trades and returns",
        "exactly zero. Buy-and-hold buys at the first scored candle and sells at the",
        "candle closing the last scored sample's horizon — the same window the model",
        "traded in, as a *continuous market span* rather than the scored-sample set —",
        "and pays **one** round trip for the whole hold, not one per horizon.",
        "",
        "Their `ann. Sharpe` is built the same way as the models' above, from the same",
        "cost model, so the columns are comparable.",
        "",
        "| fold | policy | trades | gross return | net return | ann. Sharpe | "
        "candle max DD | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        references = result["outer_validation"].get(REFERENCES_KEY) or {}
        cash = references.get("cash") or {}
        if cash:
            lines.append(
                f"| {result['fold']} | CASH | 0 | +0.0000 | +0.0000 | n/a | 0.0000 | "
                "no position, no fees |"
            )
        hold = references.get("buy_and_hold") or {}
        if not hold:
            continue
        if not hold.get("available"):
            lines.append(
                f"| {result['fold']} | buy-and-hold | — | — | — | n/a | — | "
                f"{hold.get('reason', 'unavailable')} |"
            )
            continue
        note = (
            f"spans {hold['missing_candles']} missing candle(s); Sharpe withheld, "
            "drawdown is a lower bound"
            if hold["spans_market_data_gap"]
            else f"{hold['candles_held']} candles held"
        )
        lines.append(
            f"| {result['fold']} | buy-and-hold | 1 | {hold['gross_return']:+.4f} | "
            f"{hold['net_return']:+.4f} | "
            f"{ev.number(hold['annualised_sharpe'], '.2f')} | "
            f"{hold['candle_max_drawdown']:.4f} | {note} |"
        )
    return lines


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
    predictions = []
    for i, plan in enumerate(folds):
        logger.info("--- fold %d/%d ---", i + 1, len(folds))
        record, fold_predictions = run_fold(i, data, plan, run, device=device)
        results.append(record)
        predictions.append(fold_predictions)

    summary = summarise(results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = write_predictions(out_dir, predictions, boundary)
    logger.info(
        "Wrote %d outer predictions to %s",
        sum(len(frame) for frame in predictions),
        predictions_path,
    )
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
                "outer_predictions": PREDICTIONS_NAME,
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
