"""P2b robustness: which market-structure family is carrying the combined arm?

    python -m nn.p2b_ablation --full artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_xgboost \
        --ablations artifacts/benchmark/btc_p2b_ablation_* \
        --control artifacts/benchmark/btc_p2b_ohlcv14_xgboost \
        --out artifacts/benchmark/btc_p2b_ablation

**This is descriptive, and it is post-hoc.** The canonical P2b result is one
predeclared comparison between three information sets. This is seven more
comparisons chosen *after* that result was produced, on the same four outer
blocks, which is exactly the situation where a chart of deltas starts to look
like a finding. It is reported as a source of hypotheses for a later research
generation and never as confirmation of anything.

**What a leave-one-family-out delta measures.** Each ablation arm is the
combined arm minus one declared family, on the same rows, with the same model
configuration and the same inner-only threshold rule. ``full - ablated`` is
therefore what that family contributed *given everything else was present* — a
marginal contribution, not a standalone value. A family can measure near zero
here because it is genuinely uninformative or because another family already
carries the same information, and this analysis cannot tell those apart.

**Sign consistency over four folds is the only aggregate worth reading.** With
four temporal periods, a mean delta is one number over a sample of four and its
standard deviation is barely a measurement. How many of the four folds agree on
the sign is the honest summary, and it is the one reported first.

The families are the six declared in ``docs/smc_v1.md`` §4 and fixed in
:data:`nn.information_sets.ABLATABLE_FAMILIES`. They were named before any P2b
result existed, which is what keeps this an ablation rather than a search.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Sequence

from nn.information_sets import ABLATION_PREFIX, COMBINED, SMC_FEATURE_FAMILIES
from nn.p2b_compare import (
    ComparisonError,
    check_cells_agree,
    checkpoint_of,
    fold_row,
    load_cell,
    recompute_cell,
    spread_of,
)
from tools.freeze_evidence import DERIVED

logger = logging.getLogger(__name__)

ABLATION_JSON = "p2b_ablation.json"
ABLATION_MD = "p2b_ablation.md"

#: Deltas reported per family, per fold.
DELTA_METRICS = (
    "net_return",
    "annualised_sharpe",
    "max_drawdown",
    "exposure",
    "n_trades",
    "macro_f1",
    "directional_accuracy",
)


def family_of(set_name: str) -> str:
    if not set_name.startswith(ABLATION_PREFIX):
        raise ComparisonError(f"{set_name!r} is not a leave-one-family-out arm")
    return set_name[len(ABLATION_PREFIX) :]


def sign_consistency(values: Sequence[float | None]) -> dict[str, Any]:
    """How many of the four folds agree on the sign, and which way.

    Reported instead of a significance claim. Four observations do not support
    one, and quoting a p-value over four temporal blocks whose returns are
    serially dependent would be worse than quoting nothing.
    """
    defined = [v for v in values if v is not None]
    positive = sum(1 for v in defined if v > 0)
    negative = sum(1 for v in defined if v < 0)
    return {
        "folds": len(defined),
        "positive": positive,
        "negative": negative,
        "zero": len(defined) - positive - negative,
        "agreeing_folds": max(positive, negative),
        "direction": (
            "positive"
            if positive > negative
            else ("negative" if negative > positive else "split")
        ),
    }


def importance_by_family(cell: dict[str, Any]) -> dict[str, Any]:
    """Per-family and per-feature importance share, per fold and averaged.

    Averaged across folds *and* reported per fold, because the interesting
    question is not which feature ranked highest overall but whether the ranking
    held still: a column that dominates one temporal period and vanishes in the
    next three is describing that period, not the market.
    """
    per_fold: list[dict[str, Any]] = []
    for record in cell["payload"]["folds"]:
        importance = record["model"].get("importance") or {}
        share = importance.get("per_feature_share")
        if not share:
            continue
        families = {
            family: round(sum(share.get(c, 0.0) for c in columns), 8)
            for family, columns in SMC_FEATURE_FAMILIES.items()
        }
        families["ohlcv14"] = round(1.0 - sum(families.values()), 8)
        per_fold.append(
            {
                "fold": record["fold"],
                "importance_type": importance.get("importance_type"),
                "by_family": families,
                "top_features": sorted(share.items(), key=lambda kv: -kv[1])[:15],
            }
        )
    if not per_fold:
        return {"available": False, "reason": "the fitted estimator exposed no importance"}

    names = sorted({n for f in per_fold for n, _ in f["top_features"]})
    stability = {}
    for name in names:
        ranks = []
        for entry in per_fold:
            ordered = [n for n, _ in entry["top_features"]]
            ranks.append(ordered.index(name) + 1 if name in ordered else None)
        stability[name] = {
            "top15_in_folds": sum(1 for r in ranks if r is not None),
            "ranks": ranks,
        }
    return {
        "available": True,
        "per_fold": per_fold,
        "family_share_mean": {
            family: round(statistics.fmean([f["by_family"][family] for f in per_fold]), 8)
            for family in per_fold[0]["by_family"]
        },
        "top_feature_stability": dict(
            sorted(stability.items(), key=lambda kv: -kv[1]["top15_in_folds"])
        ),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--full", type=Path, required=True, help="the combined-arm cell")
    parser.add_argument("--ablations", nargs="+", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=None, help="the ohlcv14 cell")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P2b robustness — leave-one-family-out ablation",
        "",
        f"Model: **{payload['model']}**. Reference arm: `{COMBINED}`.",
        "",
        "**Post-hoc and descriptive.** These comparisons were chosen after the canonical",
        "P2b result was produced, on the same four outer blocks. They are a source of",
        "hypotheses for a later research generation, not confirmation of anything, and no",
        "model or threshold anywhere in P2b was selected using them.",
        "",
        "**What a delta means.** `full − ablated` is what a family contributed *given the",
        "other five and OHLCV14 were present*. A near-zero delta means the family added",
        "nothing on top of the rest — which is not the same as the family being",
        "uninformative, because another family may already carry the same information.",
        "",
        "**Why sign consistency comes first.** Four temporal periods do not support a",
        "significance claim, and their returns are serially dependent. How many of the four",
        "agree on the sign is the honest summary.",
        "",
        "## Marginal contribution by family",
        "",
        "| family | columns | Δ net mean | Δ net min | Δ net max | folds agreeing "
        "| direction |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for family, entry in payload["families"].items():
        net = entry["aggregate"]["net_return"]
        sign = entry["sign_consistency"]["net_return"]
        lines.append(
            f"| `{family}` | {entry['columns_removed']} | {net['mean']:+.6f} "
            f"| {net['min']:+.6f} | {net['max']:+.6f} "
            f"| **{sign['agreeing_folds']} of {sign['folds']}** | {sign['direction']} |"
        )

    lines += ["", "## Per fold", ""]
    for family, entry in payload["families"].items():
        lines += [
            f"### `{family}` — {entry['columns_removed']} columns removed",
            "",
            "| fold | outer period | full net | ablated net | **Δ net** | Δ Sharpe | "
            "Δ max DD | Δ trades | Δ macro F1 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for d in entry["per_fold"]:
            sharpe = (
                "n/a" if d["annualised_sharpe"] is None else f"{d['annualised_sharpe']:+.4f}"
            )
            lines.append(
                f"| {d['fold']} | {d['period_start'][:10]} → {d['period_end'][:10]} "
                f"| {d['full_net_return']:+.6f} | {d['ablated_net_return']:+.6f} "
                f"| **{d['net_return']:+.6f}** | {sharpe} | {d['max_drawdown']:+.4f} "
                f"| {d['n_trades']:+.0f} | {d['macro_f1']:+.4f} |"
            )
        lines.append("")

    importance = payload["importance"]
    if importance.get("available"):
        lines += [
            "## Importance share by family",
            "",
            f"Native `{importance['per_fold'][0]['importance_type']}` importance from "
            "the fitted trees, summed",
            "across each feature's 64 timesteps and then across its family. Tree",
            "importance describes what the fitted model used, not what the market rewards —",
            "a column can dominate because it is finely resolved rather than informative.",
            "",
            "| family | "
            + " | ".join(f"fold {f['fold']}" for f in importance["per_fold"])
            + " | mean |",
            "| --- | " + " | ".join("---" for _ in importance["per_fold"]) + " | --- |",
        ]
        for family, mean in sorted(
            importance["family_share_mean"].items(), key=lambda kv: -kv[1]
        ):
            per = " | ".join(f"{f['by_family'][family]:.4f}" for f in importance["per_fold"])
            lines.append(f"| `{family}` | {per} | **{mean:.4f}** |")

        lines += [
            "",
            "### Feature stability across the four temporal folds",
            "",
            "How often each column appeared in the fold's top 15, and at what rank. A column",
            "in the top 15 of one fold and absent from the other three is describing that",
            "period.",
            "",
            "| feature | folds in top 15 | ranks |",
            "| --- | --- | --- |",
        ]
        for name, entry in list(importance["top_feature_stability"].items())[:25]:
            ranks = ", ".join("—" if r is None else str(r) for r in entry["ranks"])
            lines.append(f"| `{name}` | {entry['top15_in_folds']} of 4 | {ranks} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    full = load_cell(args.full)
    if full["information_set"] != COMBINED:
        raise ComparisonError(
            f"--full must be the {COMBINED} arm, got {full['information_set']}"
        )
    ablations = [load_cell(p) for p in sorted(args.ablations)]
    if not ablations:
        raise ComparisonError("no ablation arms given")

    cells = [full, *ablations]
    if args.control is not None:
        cells.append(load_cell(args.control))
    models = {c["model"] for c in cells}
    if len(models) != 1:
        raise ComparisonError(
            f"the ablation compares one model at a time, got {sorted(models)}"
        )

    # Same guarantees the canonical comparison rests on. `checkpoint_of` first:
    # an ablation of one checkpoint's combined arm has nothing to say about
    # another's, and the six diagnostic arms belong to P2b by declaration rather
    # than by their names happening to start with the right prefix.
    checkpoint = checkpoint_of(cells)
    # If the arms did not score the same rows, every delta below measures the
    # sample universe instead.
    parity = check_cells_agree(cells)
    mismatches = [
        {"cell": f"{c['information_set']}::{c['model']}", **finding}
        for c in cells
        for finding in recompute_cell(c)
        if finding["mismatches"]
    ]
    if mismatches:
        raise ComparisonError(
            "a cell's report disagrees with its own persisted predictions:\n"
            + json.dumps(mismatches, indent=2)
        )

    full_rows = {r["fold"]: fold_row(r, full["model"]) for r in full["payload"]["folds"]}
    n_full = len(full["payload"]["information_parity"]["feature_names"])

    families: dict[str, Any] = {}
    for cell in ablations:
        family = family_of(cell["information_set"])
        rows = [fold_row(r, cell["model"]) for r in cell["payload"]["folds"]]
        per_fold = []
        for row in rows:
            base = full_rows[row["fold"]]
            entry = {
                "fold": row["fold"],
                "period_start": row["period_start"],
                "period_end": row["period_end"],
                "full_net_return": base["net_return"],
                "ablated_net_return": row["net_return"],
            }
            for metric in DELTA_METRICS:
                a, b = base[metric], row[metric]
                entry[metric] = (
                    None if a is None or b is None else round(float(a) - float(b), 6)
                )
            per_fold.append(entry)
        families[family] = {
            "removed_family": family,
            "columns_removed": n_full
            - len(cell["payload"]["information_parity"]["feature_names"]),
            "arm": cell["information_set"],
            "per_fold": per_fold,
            "aggregate": {m: spread_of(d[m] for d in per_fold) for m in DELTA_METRICS},
            "sign_consistency": {
                m: sign_consistency([d[m] for d in per_fold]) for m in DELTA_METRICS
            },
        }

    ordered = {f: families[f] for f in SMC_FEATURE_FAMILIES if f in families}
    ordered.update({f: v for f, v in families.items() if f not in ordered})

    payload = {
        "checkpoint": f"{checkpoint.name}-robustness",
        "of_checkpoint": checkpoint.name,
        "evidence_class": DERIVED,
        "status": "post-hoc descriptive diagnostic; not canonical P2b evidence",
        "question": (
            f"which {checkpoint.family} family carries the combined arm, given the others?"
        ),
        "model": full["model"],
        "reference_arm": COMBINED,
        "delta_definition": "full minus ablated, per fold, on identical rows",
        "caveats": [
            f"chosen after the canonical {checkpoint.name} result was produced, on the "
            "same four outer blocks — adaptive, and a source of hypotheses rather than "
            "confirmation",
            "a marginal contribution given the other families, not a standalone value; a "
            "near-zero delta cannot distinguish an uninformative family from a redundant one",
            "four temporal periods do not support a significance claim, so sign consistency "
            "is reported instead of one",
            f"no model, threshold or feature constant anywhere in {checkpoint.name} was "
            "selected using this",
        ],
        "contract": full["payload"]["contract"],
        "feature_spec": full["payload"]["feature_spec"],
        "sealed_test": False,
        "parity": parity,
        "families": ordered,
        "importance": importance_by_family(full),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ABLATION_JSON).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (out_dir / ABLATION_MD).write_text(to_markdown(payload))
    logger.info("wrote %s and %s", out_dir / ABLATION_JSON, out_dir / ABLATION_MD)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
