"""P6's runner: one specialist per clock, fitted on that clock's own bars.

    python -m nn.p6 --clock 5m --model xgboost --out artifacts/benchmark/btc_p6_5m_xgboost
    python -m nn.p6 --all --out-root artifacts/benchmark

**Almost nothing here is new machinery, and that is deliberate.** The fit, the
threshold selection, the frozen scoring, the baselines, the cost model and the
report construction are :func:`nn.benchmark.run_fold` — the same function P2a
went through — so a difference between a 1m cell and a 1h cell cannot be a
difference in how they were measured. What this module adds is exactly the two
things a change of clock needs and the existing runners cannot supply:

1. **A dataset per clock.** The 1m source is resampled by :mod:`nn.multiclock`,
   passed through the same :func:`nn.data_pipeline.build_dataset` every previous
   checkpoint used, and vouched for by
   :func:`nn.data_pipeline.check_label_consistency` before a model sees it.
2. **Folds mapped by timestamp.** `nn.p2b.plan_from_manifest` plans in row
   indices, and a 1m row 21,697 is not the fortnight a 1h row 21,697 is. P6's
   four periods are frozen as *instants* in :mod:`nn.p6_preregistration`, and
   :func:`plan_folds` resolves them into each clock's own rows.

The three model families are fitted together per fold, in one
:func:`nn.benchmark.run_fold` call, because they must see the same arrays: the
shared baselines and economic references are checked to be one value rather than
three, which is the data-level proof that the families scored the same rows. The
record is then split into one cell per (clock, model), which is the layout
§12 of the preregistration froze.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from chimera.contracts import TargetSpec
from nn.benchmark import BASELINE_NAMES, THRESHOLD_OBJECTIVE, threshold_grid
from nn.benchmark import run_fold as benchmark_run_fold
from nn.data_pipeline import build_dataset, check_label_consistency
from nn.dataset import Split
from nn.multiclock import (
    RESEARCH_VISIBLE_END,
    STYX_START,
    resample_from_minutes,
)
from nn.data_fingerprint import fingerprint_research_input
from nn.p2b import (
    FIT_THREADS,
    MIN_TRADES,
    code_revision,
    numerical_environment,
    threadpool_record,
)
from nn.p6_preregistration import (
    CHECKPOINT,
    CLOCKS,
    COSTS,
    EVIDENCE_CEILING,
    FOLD_PERIODS,
    HORIZON_BARS,
    HORIZONS,
    MODELS,
    QUESTION,
    REGION,
    SEED,
    SEQ_LEN,
    preregistration_hash,
)
from nn.research_contract import load_contract
from nn.simple_models import SIMPLE_MODELS
from nn.train import ResearchData, RunConfig, research_data_from_frame
from nn.walkforward import REFERENCES_KEY, FoldPlan
from tools.acquire_multiclock_source import MANIFEST_NAME
from tools.verify_multiclock_snapshot import SnapshotError, verify

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "p6.json"
MARKDOWN_NAME = "p6.md"
PREDICTIONS_NAME = "outer_predictions.parquet"
STATUS_NAME = "STATUS.md"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "research" / MANIFEST_NAME
DEFAULT_OUT_ROOT = REPO_ROOT / "artifacts" / "benchmark"

#: What a P6 cell is evidence about. Deliberately not the literal "derived", so
#: `tools.freeze_evidence` hashes it: a cell is primary evidence that cannot be
#: rebuilt without re-fitting.
EVIDENCE_CLASS = "one native-timeframe specialist, fitted once; P6 primary evidence"


@dataclass(frozen=True)
class Registration:
    """Which preregistration a run is executing, and where its cells go.

    P6-EXT is the same runner on two further clocks under its own frozen design.
    Selecting the registration rather than copying the runner is what keeps the
    two checkpoints one code path: a change to how a cell is fitted cannot reach
    one and miss the other.
    """

    name: str
    checkpoint: str
    question: str
    evidence_ceiling: str
    clocks: tuple[str, ...]
    horizons: dict[str, str]
    prefix: str
    preregistration_hash: str
    manifest: str


def registration(name: str) -> Registration:
    """The frozen design a run executes, by name."""
    if name == "p6":
        return Registration(
            name="p6",
            checkpoint=CHECKPOINT,
            question=QUESTION,
            evidence_ceiling=EVIDENCE_CEILING,
            clocks=tuple(CLOCKS),
            horizons=dict(HORIZONS),
            prefix="btc_p6",
            preregistration_hash=preregistration_hash(),
            manifest="artifacts/btc_p6_SHA256SUMS.txt",
        )
    if name == "p6ext":
        from nn import p6_extension_preregistration as ext

        return Registration(
            name="p6ext",
            checkpoint=ext.CHECKPOINT,
            question=ext.QUESTION,
            evidence_ceiling=ext.EVIDENCE_CEILING,
            clocks=tuple(ext.CLOCKS),
            horizons=dict(ext.HORIZONS),
            prefix="btc_p6ext",
            preregistration_hash=ext.preregistration_hash(),
            manifest="artifacts/btc_p6ext_SHA256SUMS.txt",
        )
    raise SystemExit(f"unknown registration {name!r}; this runner executes 'p6' or 'p6ext'")


def contract_id(clock: str) -> str:
    """The generation-2 contract describing this clock.

    One per clock, because a contract declares the timeframe it speaks about and
    a specialist fitted on 5m samples may not be scored under a contract that
    speaks about hours. All of them carry the same sealed instant.
    """
    return f"btc-usdt-{clock}-gen2"


def load_minutes(manifest_path: Path = DEFAULT_MANIFEST) -> pd.DataFrame:
    """Verify the committed multi-clock source end to end, then read it.

    The verification is :func:`tools.verify_multiclock_snapshot.verify` itself —
    the same checks ``make check`` runs, not a second reading of the manifest. It
    runs *here*, in the only function that reads the source, rather than in a
    ``make`` target: a target is a convention and ``python -m nn.p6`` bypasses
    it, so a corrupt source must fail the load rather than reach a model fit.
    """
    try:
        report = verify(manifest_path)
    except SnapshotError as exc:
        raise SystemExit(
            f"the multi-clock source at {manifest_path} failed verification ({exc}). "
            "Refusing to fit a specialist on data whose own manifest does not describe it."
        ) from exc
    logger.info("multi-clock source verified: %s", json.dumps(report["clocks"]))
    manifest = json.loads(manifest_path.read_text())
    return pd.read_parquet(REPO_ROOT / manifest["minutes"]["path"])


def clock_dataset(minutes: pd.DataFrame, clock: str) -> tuple[pd.DataFrame, Any]:
    """One clock's bars, features and cost-aware labels.

    The label is six of *this clock's* bars ahead — the scale-consistent rule
    §4 of the preregistration froze — and the cost threshold is the same 20 bps
    on every clock. Both are `TargetSpec` defaults; they are passed explicitly
    anyway, so that a future change to the defaults cannot silently move P6.
    """
    candles = resample_from_minutes(minutes, clock)
    frame, meta = build_dataset(
        candles,
        target_spec=TargetSpec(
            horizon=HORIZON_BARS,
            fee_rate=COSTS["fee_rate"],
            slippage_rate=COSTS["slippage_rate"],
        ),
        exchange="binance",
        pair="BTC/USDT",
        timeframe=clock,
    )
    # The same vouching `nn.train.load_research_data` performs, applied to a
    # frame built in memory: a table whose labels were produced at a horizon its
    # sidecar does not declare puts its own label back into every training row.
    check_label_consistency(frame, meta)
    return frame, meta


def clock_research_data(minutes: pd.DataFrame, clock: str) -> ResearchData:
    """The arrays a specialist is fitted on, with the contract's scope checked.

    **The sealed boundary is not resolved here, and its absence is the check.**
    `nn.dataset.resolve_sealed_boundary` refuses a dataset in which no row
    reaches the sealed instant, because for a 1h generation that meant a
    walk-forward run with no test block to withhold. For this generation it is
    the *expected* state: the source stops at the retired P4-HOLD boundary, three
    months before Styx, so a resolvable seal inside these rows would mean the
    acquisition had overrun. What is asserted instead is the stronger property —
    that every row is strictly before the sealed instant — plus the contract's
    own scope check, which fails closed on a clock/contract mismatch.
    """
    frame, meta = clock_dataset(minutes, clock)
    data = research_data_from_frame(frame, meta)
    contract = load_contract(contract_id(clock))
    contract.require_scope(
        exchange=meta.exchange,
        pair=meta.pair,
        timeframe=meta.timeframe,
        source=f"the {clock} specialist's dataset",
    )
    dates = pd.DatetimeIndex(pd.to_datetime(data.dates, utc=True))
    sealed = int((dates >= pd.Timestamp(contract.sealed_test_start)).sum())
    beyond = int((dates >= RESEARCH_VISIBLE_END).sum())
    if sealed or beyond:
        raise SystemExit(
            f"{clock}: {beyond} row(s) reach the research-visible boundary and {sealed} "
            "reach the sealed instant; the source should stop before both"
        )
    return data


def plan_folds(data: ResearchData, clock: str) -> list[FoldPlan]:
    """The four frozen periods, resolved into this clock's row indices.

    Every boundary is a `searchsorted` on the clock's own timestamps, so the four
    blocks cover the same real-world windows on all five clocks and the fold
    geometry is a consequence of the calendar rather than of the row count.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(data.dates, utc=True))
    if not dates.is_monotonic_increasing:
        raise SystemExit(f"{clock}: dataset timestamps are not increasing")

    def row(stamp: str) -> int:
        return int(dates.searchsorted(pd.Timestamp(stamp)))

    plans: list[FoldPlan] = []
    for frozen in FOLD_PERIODS:
        train = Split("train", row(frozen["train_start"]), row(frozen["inner_start"]))
        inner = Split(
            "inner_validation", row(frozen["inner_start"]), row(frozen["outer_start"])
        )
        outer = Split("outer_validation", row(frozen["outer_start"]), row(frozen["outer_end"]))
        for block in (train, inner, outer):
            if block.end <= block.start:
                raise SystemExit(
                    f"{clock}: fold {frozen['fold']}'s {block.name} block is empty; the "
                    "frozen periods and this clock's coverage disagree"
                )
        plans.append(FoldPlan(train=train, inner=inner, outer=outer))

    # Outer blocks must tile forward without overlapping, exactly as
    # nn.walkforward.plan_nested_folds guarantees for the row-indexed plan.
    for earlier, later in zip(plans, plans[1:]):
        if later.outer.start < earlier.outer.end:
            raise SystemExit(f"{clock}: outer blocks overlap")
    last = dates[plans[-1].outer.end - 1]
    if last >= RESEARCH_VISIBLE_END:
        raise SystemExit(f"{clock}: the last outer row {last} reaches the research boundary")
    return plans


def split_record(record: dict[str, Any], model: str) -> dict[str, Any]:
    """One model's view of a fold that was measured for all three.

    `nn.benchmark.run_fold` fits every family against the same arrays and checks
    that the baselines and economic references came out as one value rather than
    three. This keeps that guarantee and files the result the way P2b and P5 file
    theirs — one cell per (clock, model), the model's own block beside the shared
    floors — so a P6 cell reads exactly like a P5 cell.
    """
    outer = {model: record["outer_validation"][model]}
    for name in (*BASELINE_NAMES, REFERENCES_KEY):
        outer[name] = record["outer_validation"][name]
    return {
        "fold": record["fold"],
        "run_seed": record["run_seed"],
        "fold_seed": record["fold_seed"],
        "samples": record["samples"],
        "periods": record["periods"],
        "model": record["models"][model],
        "outer_validation": outer,
    }


def manifest_label(path: Path) -> str:
    """How a cell names the manifest it was actually produced from."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def cell_payload(
    clock: str,
    model: str,
    data: ResearchData,
    records: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    registered: Registration,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Everything a P6 cell records about how it was produced."""
    contract = load_contract(contract_id(clock))
    dates = pd.DatetimeIndex(pd.to_datetime(data.dates, utc=True))
    returns = [
        record["outer_validation"][model]["trading"]["net_return"] for record in records
    ]
    trades = [record["outer_validation"][model]["trading"]["n_trades"] for record in records]
    return {
        "checkpoint": registered.checkpoint,
        "question": registered.question,
        "evidence_class": EVIDENCE_CLASS,
        "preregistration_hash": registered.preregistration_hash,
        "evidence_ceiling": registered.evidence_ceiling,
        "clock": clock,
        "model": model,
        "horizon_bars": HORIZON_BARS,
        "horizon": registered.horizons[clock],
        "source": {
            "manifest": manifest_label(manifest_path),
            "minutes_digest": manifest["minutes"]["digest"],
            "clock_digest": manifest["clocks"][clock]["digest"],
            "clock_rows": manifest["clocks"][clock]["rows"],
        },
        "contract": {
            "contract_id": contract.contract_id,
            "contract_hash": contract.contract_hash,
            "research_generation": contract.research_generation,
            "sealed_test_start": contract.sealed_test_start.isoformat(),
        },
        "dataset": {
            "rows": int(len(data.dates)),
            "start": dates[0].isoformat(),
            "end": dates[-1].isoformat(),
            "segments": int(np.unique(data.segment_ids).size),
            "feature_names": list(data.feature_names),
            "feature_spec": data.feature_spec.to_dict(),
            "class_balance": data.ds_meta.class_balance,
            "candles_per_year": data.candles_per_year,
        },
        "region": dict(REGION),
        "target": {
            **data.target_spec.to_dict(),
            "horizon_unit": f"{clock} bars",
        },
        "sample_universe": {
            # Every row is research-visible on this clock, so the fingerprint's
            # research region is the whole table. Taken directly rather than
            # through `ResearchData.input_fingerprint`, which needs a resolved
            # sealed boundary this generation deliberately does not have.
            "digest": fingerprint_research_input(
                data.research_columns(),
                feature_names=data.feature_names,
                exchange=data.ds_meta.exchange,
                pair=data.ds_meta.pair,
                timeframe=data.ds_meta.timeframe,
                feature_spec=data.ds_meta.feature_spec,
                target_spec=data.ds_meta.target_spec,
                research_rows=data.n_rows,
                total_rows=data.n_rows,
            ).research_input_hash,
            "note": "the identity of every value research read on this clock",
        },
        "config": {"seed": SEED, "seq_len": SEQ_LEN, "min_trades": MIN_TRADES},
        "threshold_selection": {
            "block": "inner_validation",
            "objective": THRESHOLD_OBJECTIVE,
            "grid": [float(value) for value in threshold_grid()],
            "min_trades": MIN_TRADES,
        },
        "tuning": "none: predeclared fixed configurations, no hyperparameter search",
        "adaptive_status": EVIDENCE_CEILING,
        "sealed_test": {
            "styx_start": STYX_START.isoformat(),
            "research_visible_end": RESEARCH_VISIBLE_END.isoformat(),
            "rows_at_or_after_research_visible_end": int(
                (dates >= RESEARCH_VISIBLE_END).sum()
            ),
        },
        "code": code_revision(),
        "numerics": numerical_environment(),
        "outer_predictions": PREDICTIONS_NAME,
        "folds": records,
        "summary": {
            "outer_net_returns": returns,
            "mean_outer_net_return": round(float(np.mean(returns)), 12),
            "positive_folds": int(sum(1 for value in returns if value > 0.0)),
            "outer_trades": trades,
            "total_outer_trades": int(sum(trades)),
        },
    }


def to_markdown(payload: dict[str, Any]) -> str:
    """The cell, readable without a JSON parser."""
    lines = [
        f"# {payload['checkpoint']} — {payload['clock']} specialist, {payload['model']}",
        "",
        f"**{payload['evidence_class']}**",
        "",
        f"- preregistration: `{payload['preregistration_hash']}`",
        f"- horizon: {payload['horizon_bars']} native bars = {payload['horizon']}",
        f"- clock rows: {payload['dataset']['rows']:,}",
        f"- class balance: {payload['dataset']['class_balance']}",
        f"- evidence ceiling: {payload['evidence_ceiling']}",
        "",
        "| fold | outer period | threshold | net return | trades | turnover |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    model = payload["model"]
    for record in payload["folds"]:
        period = record["periods"]["outer_validation"]
        trading = record["outer_validation"][model]["trading"]
        lines.append(
            f"| {record['fold']} | {period['start']} → {period['end']} | "
            f"{record['model']['selection']['threshold']} | "
            f"{trading['net_return']} | {trading['n_trades']} | {trading['turnover']} |"
        )
    summary = payload["summary"]
    lines += [
        "",
        f"Mean outer net return **{summary['mean_outer_net_return']}**, "
        f"positive in **{summary['positive_folds']} of {len(payload['folds'])}** folds.",
        "",
        "This cell is one specialist on one clock. P6's verdict for this clock is "
        "XGBoost's, and it is the decision artifact that applies the gate.",
    ]
    return "\n".join(lines) + "\n"


def status_markdown(payload: dict[str, Any], registered: Registration) -> str:
    return (
        f"# CURRENT — {payload['checkpoint']} {payload['clock']} x {payload['model']}\n\n"
        f"{payload['evidence_class']}.\n\n"
        f"Preregistration `{payload['preregistration_hash']}`.\n"
        f"Frozen under `{registered.manifest}`.\n"
    )


def run_clock(
    minutes: pd.DataFrame,
    clock: str,
    *,
    manifest: dict[str, Any],
    out_root: Path,
    models: tuple[str, ...] = MODELS,
    registered: Registration | None = None,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Path]:
    """Every model on one clock, fitted once against the same arrays."""
    registered = registered or registration("p6")
    logger.info("=== %s %s specialist ===", registered.checkpoint, clock)
    data = clock_research_data(minutes, clock)
    plans = plan_folds(data, clock)
    run = RunConfig(seed=SEED, seq_len=SEQ_LEN)
    specs = tuple(spec for spec in SIMPLE_MODELS if spec.name in models)

    records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    # Built before the limit is entered, for the reason nn.p2b records: the
    # libraries import lazily, and a pool that is not loaded yet cannot be
    # limited, so fold 0 would otherwise run under a thread configuration no
    # other fold used.
    for spec in specs:
        spec.build(SEED)

    with threadpool_limits(limits=FIT_THREADS):
        threads = threadpool_record()
        for fold, plan in enumerate(plans):
            logger.info(
                "%s fold %d: train=%d inner=%d outer=%d rows",
                clock,
                fold,
                plan.train.end - plan.train.start,
                plan.inner.end - plan.inner.start,
                plan.outer.end - plan.outer.start,
            )
            record, predictions = benchmark_run_fold(
                fold, data, plan, run, specs, min_trades=MIN_TRADES
            )
            records.append(record)
            frames.append(predictions)

    written: dict[str, Path] = {}
    predictions = pd.concat(frames, ignore_index=True)
    for spec in specs:
        out_dir = out_root / f"{registered.prefix}_{clock}_{spec.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = cell_payload(
            clock,
            spec.name,
            data,
            [split_record(record, spec.name) for record in records],
            manifest=manifest,
            registered=registered,
            manifest_path=manifest_path,
        )
        payload["numerics"]["threadpools"] = threads
        (out_dir / ARTIFACT_NAME).write_text(json.dumps(payload, indent=2) + "\n")
        (out_dir / MARKDOWN_NAME).write_text(to_markdown(payload))
        (out_dir / STATUS_NAME).write_text(status_markdown(payload, registered))
        predictions.loc[predictions["model"] == spec.name].reset_index(drop=True).to_parquet(
            out_dir / PREDICTIONS_NAME, index=False
        )
        written[spec.name] = out_dir
        logger.info(
            "%s x %s: mean outer net return %s, positive in %d of 4",
            clock,
            spec.name,
            payload["summary"]["mean_outer_net_return"],
            payload["summary"]["positive_folds"],
        )
    return written


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration",
        choices=("p6", "p6ext"),
        default="p6",
        help="which frozen design to execute (default: p6)",
    )
    parser.add_argument("--clock")
    parser.add_argument("--all", action="store_true", help="every clock in the registration")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_argparser().parse_args(argv)
    if not args.all and not args.clock:
        raise SystemExit("pass --clock CLOCK or --all")

    registered = registration(args.registration)
    if args.clock and args.clock not in registered.clocks:
        raise SystemExit(
            f"{registered.checkpoint} registered {list(registered.clocks)}, not {args.clock!r}"
        )
    clocks = list(registered.clocks) if args.all else [args.clock]
    minutes = load_minutes(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    for clock in clocks:
        run_clock(
            minutes,
            clock,
            manifest=manifest,
            manifest_path=args.manifest,
            out_root=args.out_root,
            models=tuple(args.models),
            registered=registered,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
