"""What the four P2b outer periods actually looked like, and where SMC helped.

    python -m nn.p2b_regimes --runs artifacts/benchmark/btc_p2b_* \
        --out artifacts/benchmark/btc_p2b_regimes

**Descriptive. Nothing here is fitted, and nothing here becomes a filter.**

P2b reports four numbers per arm because there are four temporal outer blocks.
Four is few enough that "the combined arm won two of them" is a statement about
two specific stretches of 2023-2025 as much as about market structure, and a
reader who cannot see what those stretches were has no way to judge which. So
this describes them — trend, volatility, range, directionality, chop,
autocorrelation, volume, label balance — and lays the per-fold deltas beside the
description.

**It is deliberately not a regime filter.** Fitting a rule that trades market
structure only in the periods where it happened to win, on the same four periods
that revealed the win, is the purest form of the error this repository exists to
avoid: the rule would fit four observations and report the fit as a result. The
output here is a hypothesis for a later research generation to test on periods
it has not seen — nothing more.

Every statistic is computed from the *research-visible* rows of the outer block
itself: the close path, the candle ranges, the volumes and the labels the
benchmark already scored. No sealed data, and no data from outside the block.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chimera.contracts import CLASS_ORDER
from nn.data_pipeline import load_dataset
from nn.information_sets import COMBINED, OHLCV14, SMC_V1
from nn.p2b import DEFAULT_MANIFEST, plan_from_manifest
from nn.p2b_compare import ComparisonError, fold_row, load_cell
from tools.freeze_evidence import DERIVED

logger = logging.getLogger(__name__)

REGIMES_JSON = "p2b_regimes.json"
REGIMES_MD = "p2b_regimes.md"


def describe_block(
    raw: pd.DataFrame, spine: pd.DataFrame, start: int, end: int, candles_per_year: float
) -> dict[str, Any]:
    """Market statistics over one outer block, from that block's rows only.

    ``spine`` supplies the labels and the close path the benchmark scored;
    ``raw`` supplies the highs, lows and volumes the processed dataset does not
    carry. Both are sliced to ``[start, end)`` before anything is measured, so a
    statistic cannot describe a candle the fold could not see.
    """
    rows = spine.iloc[start:end]
    dates = pd.to_datetime(rows["date"], utc=True)
    candles = raw[raw["date"].isin(dates.to_numpy())].reset_index(drop=True)
    if len(candles) != len(rows):
        raise ComparisonError(
            f"outer block [{start}, {end}) has {len(rows)} research rows but "
            f"{len(candles)} raw candles; the two files disagree about this period"
        )

    close = rows["close"].to_numpy(dtype=np.float64)
    high = candles["high"].to_numpy(dtype=np.float64)
    low = candles["low"].to_numpy(dtype=np.float64)
    volume = candles["volume"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(close))

    total_return = float(close[-1] / close[0] - 1.0)
    # Sum of absolute candle moves against the net move: 1.0 is a straight line,
    # near 0 is a round trip. The standard efficiency-ratio construction, and the
    # least loaded word for it here is "directionality".
    path_length = float(np.abs(np.diff(close)).sum())
    directionality = abs(close[-1] - close[0]) / path_length if path_length > 0 else 0.0

    counts = rows["target"].value_counts().to_dict()
    labels = {sig.value: int(counts.get(i, 0)) for i, sig in enumerate(CLASS_ORDER)}
    long_short = labels["LONG"] / labels["SHORT"] if labels["SHORT"] else float("inf")

    autocorr = float(np.corrcoef(log_ret[:-1], log_ret[1:])[0, 1]) if len(log_ret) > 2 else 0.0
    true_range = np.maximum(high - low, 0.0)[1:]
    prev_close = close[:-1]
    true_range = np.maximum(
        true_range, np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close))
    )

    return {
        "rows": len(rows),
        "start": str(dates.iloc[0]),
        "end": str(dates.iloc[-1]),
        "total_return": round(total_return, 6),
        "annualised_return": (
            round(float((1.0 + total_return) ** (candles_per_year / len(rows)) - 1.0), 6)
            if total_return > -1.0
            else None
        ),
        "realised_vol_annualised": round(
            float(np.std(log_ret, ddof=1) * np.sqrt(candles_per_year)), 6
        ),
        "high_low_range": round(float(high.max() / low.min() - 1.0), 6),
        "max_drawdown": round(float(1.0 - (close / np.maximum.accumulate(close)).min()), 6),
        "directionality": round(float(directionality), 6),
        "chop": round(float(1.0 - directionality), 6),
        "atr_pct_mean": round(float(np.mean(true_range / close[1:])), 6),
        "return_autocorr_lag1": round(autocorr, 6),
        "volume_mean": round(float(volume.mean()), 4),
        "volume_trend": round(
            float(volume[len(volume) // 2 :].mean() / volume[: len(volume) // 2].mean() - 1.0),
            6,
        ),
        "label_balance": labels,
        "long_over_short": round(long_short, 4) if np.isfinite(long_short) else None,
        "hold_share": round(labels["HOLD"] / len(rows), 6),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def to_markdown(payload: dict[str, Any]) -> str:
    periods = payload["periods"]
    lines = [
        "# P2b — what the four outer periods were, and where market structure helped",
        "",
        "**Descriptive. Nothing here is fitted, and nothing here becomes a filter.**",
        "",
        "Four temporal blocks is few enough that a per-fold result is a statement about four",
        "specific stretches of market as much as about the information set. This describes",
        "them from research-visible rows only and lays the deltas beside the description, so",
        "a reader can judge which. Fitting a rule that trades market structure only in the",
        "periods where it won — on the same four periods that revealed the win — would fit",
        "four observations and report the fit as a result. That is not done here, and the",
        "output is a hypothesis for a later generation to test on periods it has not seen.",
        "",
        "## The four outer periods",
        "",
        "| fold | period | rows | total return | ann. vol | high/low range | max DD "
        "| directionality | ATR% | autocorr | volume trend | LONG/SHORT | HOLD share |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in periods:
        d = entry["market"]
        lines.append(
            f"| {entry['fold']} | {d['start'][:10]} → {d['end'][:10]} | {d['rows']} "
            f"| {d['total_return']:+.4f} | {d['realised_vol_annualised']:.4f} "
            f"| {d['high_low_range']:.4f} | {d['max_drawdown']:.4f} "
            f"| {d['directionality']:.4f} | {d['atr_pct_mean']:.5f} "
            f"| {d['return_autocorr_lag1']:+.4f} | {d['volume_trend']:+.4f} "
            f"| {d['long_over_short']} | {d['hold_share']:.4f} |"
        )

    lines += [
        "",
        "`directionality` is the efficiency ratio: the net move divided by the summed",
        "absolute candle moves. 1.0 is a straight line; near 0 is a round trip. `chop` is",
        "its complement.",
        "",
        "## Where market structure helped, and where it hurt",
        "",
    ]
    for model, deltas in payload["deltas_by_model"].items():
        lines += [
            f"### {model}",
            "",
            "| fold | period | directionality | ann. vol | Δ net (`smc_v1` − `ohlcv14`) "
            "| Δ net (`combined` − `ohlcv14`) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for entry in periods:
            fold = entry["fold"]
            row = deltas.get(str(fold), deltas.get(fold, {}))
            smc = row.get(SMC_V1)
            comb = row.get(COMBINED)
            lines.append(
                f"| {fold} | {entry['market']['start'][:10]} → {entry['market']['end'][:10]} "
                f"| {entry['market']['directionality']:.4f} "
                f"| {entry['market']['realised_vol_annualised']:.4f} "
                f"| {'n/a' if smc is None else format(smc, '+.6f')} "
                f"| {'n/a' if comb is None else format(comb, '+.6f')} |"
            )
        lines.append("")

    lines += ["## Observations", ""]
    for note in payload["observations"]:
        lines.append(f"- {note}")
    lines += [
        "",
        "These are readings of the table above, not causal claims. With four periods, any",
        "association between a market statistic and a delta is compatible with coincidence,",
        "and the honest use of it is to state a hypothesis a later generation can test.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent.parent.parent
    spine, ds_meta = load_dataset(root / manifest["processed_outer_coverage"]["path"])
    raw = pd.read_parquet(root / manifest["raw_pre_styx"]["path"])
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    folds, _ = plan_from_manifest(manifest, len(spine))
    candles_per_year = 365 * 24 * 60 / 60

    cells = [load_cell(Path(d)) for d in sorted(args.runs)]
    by_model: dict[str, dict[str, dict[int, float]]] = {}
    for cell in cells:
        rows = {r["fold"]: fold_row(r, cell["model"]) for r in cell["payload"]["folds"]}
        by_model.setdefault(cell["model"], {})[cell["information_set"]] = {
            fold: row["net_return"] for fold, row in rows.items()
        }

    deltas_by_model: dict[str, Any] = {}
    for model, by_set in by_model.items():
        control = by_set.get(OHLCV14)
        if control is None:
            continue
        deltas_by_model[model] = {
            str(fold): {
                name: round(by_set[name][fold] - control[fold], 6)
                for name in (SMC_V1, COMBINED)
                if name in by_set
            }
            for fold in sorted(control)
        }

    periods = [
        {
            "fold": i,
            "row_range": [plan.outer.start, plan.outer.end],
            "market": describe_block(
                raw, spine, plan.outer.start, plan.outer.end, candles_per_year
            ),
        }
        for i, plan in enumerate(folds)
    ]

    observations = build_observations(periods, deltas_by_model)
    payload = {
        "checkpoint": "P2b-regimes",
        "evidence_class": DERIVED,
        "status": "descriptive; nothing fitted, no regime filter derived",
        "sealed_test": False,
        "computed_from": "research-visible rows of each outer block only",
        "periods": periods,
        "deltas_by_model": deltas_by_model,
        "observations": observations,
        "caveats": [
            "four temporal periods; any association here is compatible with coincidence",
            "no rule was fitted to these statistics and none may be, on these periods",
            "the statistics describe the outer block, not the training data behind it",
        ],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REGIMES_JSON).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (out_dir / REGIMES_MD).write_text(to_markdown(payload))
    logger.info("wrote %s and %s", out_dir / REGIMES_JSON, out_dir / REGIMES_MD)
    return 0


def build_observations(
    periods: list[dict[str, Any]], deltas_by_model: dict[str, Any]
) -> list[str]:
    """Readings of the table, phrased as readings rather than as findings."""
    notes: list[str] = []
    directional = sorted(periods, key=lambda p: -p["market"]["directionality"])
    notes.append(
        f"fold {directional[0]['fold']} was the most directional period "
        f"(efficiency ratio {directional[0]['market']['directionality']:.4f}) and fold "
        f"{directional[-1]['fold']} the choppiest "
        f"({directional[-1]['market']['directionality']:.4f})"
    )
    volatile = sorted(periods, key=lambda p: -p["market"]["realised_vol_annualised"])
    notes.append(
        f"fold {volatile[0]['fold']} had the highest annualised realised volatility "
        f"({volatile[0]['market']['realised_vol_annualised']:.4f}), fold "
        f"{volatile[-1]['fold']} the lowest "
        f"({volatile[-1]['market']['realised_vol_annualised']:.4f})"
    )
    for model, deltas in deltas_by_model.items():
        for arm in (SMC_V1, COMBINED):
            values = [(int(fold), row[arm]) for fold, row in deltas.items() if arm in row]
            if not values:
                continue
            helped = [f for f, v in values if v > 0]
            mean = statistics.fmean(v for _, v in values)
            notes.append(
                f"{model}: `{arm}` beat the control in "
                f"{len(helped)} of {len(values)} folds"
                + (f" (folds {helped})" if helped else "")
                + f", mean Δ net {mean:+.6f}"
            )
    return notes


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
