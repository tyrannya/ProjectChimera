"""P2a aggregation: the simple models against the frozen MTST v4 evidence.

    python -m nn.benchmark_compare \
        --benchmark artifacts/benchmark/btc_p2a_seed_{42,142,242,342,442} \
        --mtst artifacts/walkforward/btc_nested_v4_seed_{42,142,242,342,442} \
        --dataset data/datasets/binance_BTC_USDT_1h_refresh_2026-08-20.parquet \
        --out artifacts/benchmark/btc_p2a_comparison

**MTST is not retrained.** Its numbers are read from the committed v4 artifacts,
which this module opens read-only and never rewrites. The whole comparison rests
on those artifacts and the P2a runs describing the same experiment, so that is
checked before anything is averaged — and when it does not hold, nothing is
averaged at all. :func:`comparability_problems` refuses on a mismatch of the
research contract, the contract hash, the research-input fingerprint, the sealed
anchor or its resolved row, the fold count or geometry, the dataset's feature
contract, the target/horizon/cost semantics, the sequence length, or the
sealed-test flag. It further requires, where both sides persisted per-sample
predictions, that the two families scored the **same outer rows** and that their
shared statistical baselines produced **identical** outer numbers: the baselines
are fitted on the fold's training labels and on nothing, so on the same rows they
are a constant, and a discrepancy in them is a discrepancy in the samples.

**What the report answers.** For each of logistic regression, LightGBM and
XGBoost: does it beat MTST, majority, momentum, CASH and buy-and-hold — with the
predictive question kept apart from the economic one throughout. Beating a
statistical floor is a *predictive-baseline improvement*; making money after
costs is *economic alpha*, and only the second is denominated in the thing the
system exists to produce. No aggregate here calls a model profitable for
clearing majority or momentum.

**What it cannot answer.** P2a is an adaptive follow-up, designed after prior
outer-validation results had been seen, and it reuses those outer folds. They
are research evidence, not a pristine out-of-sample test, and no number in this
report is a claim about live profitability. The sealed test block is unopened.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from chimera.contracts import TargetSpec
from nn import evaluate as ev
from nn import regime
from nn.benchmark import ARTIFACT_NAME, BASELINE_NAMES, PREDICTIONS_NAME
from nn.simple_models import SIMPLE_MODEL_NAMES
from nn.walkforward import PREDICTIONS_NAME as MTST_PREDICTIONS_NAME
from nn.walkforward import REFERENCES_KEY, SUMMARY_METRICS
from nn.wf_diagnostics import ARTIFACT_NAME as MTST_ARTIFACT_NAME
from nn.wf_diagnostics import IDENTITY_FIELDS, REGIONS

logger = logging.getLogger(__name__)

#: What this tool writes into ``--out``.
REPORT_JSON = "p2a_comparison.json"
REPORT_MD = "p2a_comparison.md"

#: The frozen MTST model key inside a walk-forward artifact.
MTST = "mtst"

#: Every predictive contender, in report order. The baselines are contenders in
#: the *predictive* sense only — they are floors, and the economic references
#: below are what says whether anything was earned.
CONTENDERS = SIMPLE_MODEL_NAMES + (MTST,) + BASELINE_NAMES

#: Metrics aggregated across every (seed, fold) outer observation.
AGGREGATED = tuple(SUMMARY_METRICS)

#: Which family of runs each contender's numbers are read from.
#:
#: The two statistical baselines appear in *both* families' artifacts, and
#: comparability has already required those copies to be identical fold for
#: fold. Reading them from one side is therefore exact, and reading them from
#: both would count every baseline observation twice — inflating its fold count
#: and its dispersion against contenders that appear once.
MODEL_KIND = {
    **{name: "p2a" for name in SIMPLE_MODEL_NAMES},
    **{name: "p2a" for name in BASELINE_NAMES},
    MTST: "mtst",
}


class ComparabilityError(SystemExit):
    """Raised instead of producing an aggregate over runs that do not compare."""


@dataclass(frozen=True)
class Run:
    """One loaded run — a P2a benchmark or a frozen MTST walk-forward.

    Both are read through one shape so the comparability checks are written
    once and cannot drift into being stricter about one family than the other.
    """

    kind: str
    name: str
    path: Path
    payload: dict[str, Any]

    @property
    def folds(self) -> list[dict[str, Any]]:
        return self.payload["folds"]

    @property
    def seed(self) -> Any:
        return (self.payload.get("config") or {}).get("seed")

    @property
    def seq_len(self) -> Any:
        return (self.payload.get("config") or {}).get("seq_len")

    @property
    def sealed(self) -> dict[str, Any]:
        return self.payload.get("sealed_test") or {}

    @property
    def contract(self) -> dict[str, Any]:
        return self.sealed.get("research_contract") or {}

    @property
    def contract_hash(self) -> Any:
        return self.contract.get("contract_hash")

    @property
    def contract_id(self) -> Any:
        return self.contract.get("contract_id")

    @property
    def sealed_anchor(self) -> Any:
        return self.sealed.get("anchor_timestamp")

    @property
    def sealed_start_row(self) -> Any:
        return self.sealed.get("start_row")

    @property
    def research_input_hash(self) -> Any:
        return (self.payload.get("research_input") or {}).get("research_input_hash")

    @property
    def dataset(self) -> dict[str, Any]:
        return self.payload.get("dataset") or {}

    @property
    def geometry(self) -> list[tuple[int, int]]:
        """Every region's row range, flattened: the run's comparable shape."""
        return [
            tuple(int(v) for v in fold["periods"][region]["row_range"])  # type: ignore[misc]
            for fold in self.folds
            for region in REGIONS
        ]

    @property
    def predictions_path(self) -> Path:
        name = PREDICTIONS_NAME if self.kind == "p2a" else MTST_PREDICTIONS_NAME
        return self.path.parent / name

    def models(self) -> tuple[str, ...]:
        """The model keys this run's outer reports actually carry."""
        return tuple(key for key in self.folds[0]["outer_validation"] if key != REFERENCES_KEY)

    def metric(self, fold: dict[str, Any], model: str, metric: str) -> float | None:
        section, key = SUMMARY_METRICS[metric]
        value = fold["outer_validation"][model][section][key]
        return None if value is None else float(value)

    def references(self, fold: dict[str, Any]) -> dict[str, Any]:
        return fold["outer_validation"].get(REFERENCES_KEY) or {}


def load_run(path: str | Path, *, kind: str) -> Run:
    """Load one artifact, failing with the file it wanted rather than a partial run."""
    given = Path(path)
    expected = ARTIFACT_NAME if kind == "p2a" else MTST_ARTIFACT_NAME
    artifact = given if given.suffix == ".json" else given / expected
    if not artifact.is_file():
        raise ComparabilityError(
            f"no {kind} artifact at {artifact}. Give the directory the run wrote "
            f"(containing {expected}) or the JSON file itself."
        )
    try:
        payload = json.loads(artifact.read_text())
    except json.JSONDecodeError as exc:
        raise ComparabilityError(f"{artifact} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparabilityError(f"{artifact} is not an object")

    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ComparabilityError(f"{artifact} has no folds to report on")
    for position, fold in enumerate(folds):
        if not isinstance(fold, dict) or "outer_validation" not in fold:
            raise ComparabilityError(
                f"{artifact} fold at position {position} has no outer_validation block"
            )
        missing = [r for r in REGIONS if r not in (fold.get("periods") or {})]
        if missing:
            raise ComparabilityError(
                f"{artifact} fold {fold.get('fold', position)} is missing the "
                f"{', '.join(missing)} period(s)"
            )
    if kind == "p2a" and payload.get("checkpoint") != "P2a":
        raise ComparabilityError(
            f"{artifact} does not declare checkpoint P2a, so it is not a benchmark run"
        )
    return Run(
        kind=kind,
        name=(given.parent if given.suffix == ".json" else given).name or str(given),
        path=artifact,
        payload=payload,
    )


def _identity(run: Run) -> dict[str, Any]:
    """The fields two runs must agree on to describe the same experiment."""
    return {
        "contract_id": run.contract_id,
        "contract_hash": run.contract_hash,
        "research_input_hash": run.research_input_hash,
        "sealed_anchor": run.sealed_anchor,
        "sealed_start_row": run.sealed_start_row,
        "sealed_test_evaluated": bool(run.sealed.get("evaluated"))
        or bool(run.payload.get("test_evaluated")),
        "seq_len": run.seq_len,
        "n_folds": len(run.folds),
        "geometry": run.geometry,
        **{f"dataset.{field}": run.dataset.get(field) for field in IDENTITY_FIELDS},
    }


def comparability_problems(runs: list[Run]) -> list[str]:
    """Every reason these runs are not measurements of one experiment.

    Empty means comparable. Anything else means the aggregate is withheld:
    averaging a P2a fold against an MTST fold that scored different rows, under a
    different contract, or over a dataset whose history has since been corrected
    is not a comparison, and producing it anyway is the failure this function
    exists to prevent.
    """
    problems: list[str] = []
    reference = runs[0]
    expected = _identity(reference)

    for run in runs[1:]:
        actual = _identity(run)
        for field, value in expected.items():
            if actual[field] == value:
                continue
            if field == "geometry":
                differing = [
                    f"fold {fold} {region}"
                    for fold in range(min(len(run.folds), len(reference.folds)))
                    for position, region in enumerate(REGIONS)
                    if actual["geometry"][fold * len(REGIONS) + position]
                    != value[fold * len(REGIONS) + position]
                ]
                problems.append(
                    f"{run.name} does not share {reference.name}'s fold geometry "
                    f"({', '.join(differing) or 'fold counts differ'})"
                )
                continue
            problems.append(
                f"{run.name} has {field} {actual[field]!r} but {reference.name} has "
                f"{value!r}; the two are not the same experiment"
            )

    for run in runs:
        if run.sealed.get("evaluated") or run.payload.get("test_evaluated"):
            problems.append(
                f"{run.name} reports the sealed test block as evaluated; P2a compares "
                "research folds only and will not aggregate a run that opened the seal"
            )
        if run.kind == "p2a" and MTST in run.models():
            problems.append(
                f"{run.name} carries an 'mtst' report of its own; the frozen v4 evidence "
                "is the only MTST in this comparison and must not be regenerated"
            )
        if run.kind == "mtst" and MTST not in run.models():
            problems.append(f"{run.name} has no 'mtst' outer report to compare against")

    problems += _baseline_parity_problems(runs)
    problems += _scored_row_problems(runs)
    return problems


def _baseline_parity_problems(runs: list[Run]) -> list[str]:
    """The shared floors must be *identical*, fold for fold, across every run.

    The majority rule is fitted on the fold's training labels and the momentum
    rule is fitted on nothing, so on the same rows with the same data they are a
    constant — the same constant the frozen MTST artifacts already recorded.
    Requiring bit-equality here turns them into a direct check that both families
    saw the same samples, which no amount of agreement between metadata blocks
    can establish on its own.
    """
    problems: list[str] = []
    reference = runs[0]
    for run in runs[1:]:
        for position, (fold, expected_fold) in enumerate(zip(run.folds, reference.folds)):
            for name in BASELINE_NAMES:
                actual = fold["outer_validation"].get(name)
                expected = expected_fold["outer_validation"].get(name)
                if actual is None or expected is None:
                    problems.append(
                        f"{run.name} fold {position} has no {name} report to check "
                        f"against {reference.name}"
                    )
                elif actual != expected:
                    problems.append(
                        f"{run.name} fold {position} {name} differs from "
                        f"{reference.name}'s. That rule is fitted on the fold's training "
                        "labels and on nothing else, so on identical samples it is a "
                        "constant: the two runs did not score the same rows"
                    )
    return problems


def _scored_row_problems(runs: list[Run]) -> list[str]:
    """Where both families persisted predictions, the scored rows must match exactly.

    The strongest available statement of sample identity on real data, and the
    reason it is worth reading the parquet files here: metadata can agree while
    the arrays differ, but the row indices actually scored cannot.
    """
    problems: list[str] = []
    rows: dict[str, dict[int, np.ndarray]] = {}
    for run in runs:
        path = run.predictions_path
        if not path.is_file():
            continue
        columns = ["fold", "row_index"]
        frame = pd.read_parquet(path)
        has_model = "model" in frame.columns
        frame = frame[columns + (["model"] if has_model else [])]

        # A P2a file holds one block of rows per model. Each model's rows are
        # checked separately and required to agree with each other first, so a
        # row missing from one model cannot hide behind the two that still have
        # it — which is exactly what comparing a de-duplicated union would do.
        per_fold: dict[int, np.ndarray] = {}
        for fold, group in frame.groupby("fold", sort=True):
            models = sorted(set(group["model"].astype(str))) if has_model else [""]
            sets = {
                model: np.sort(
                    (group[group["model"] == model] if has_model else group)[
                        "row_index"
                    ].to_numpy(dtype=np.int64)
                )
                for model in models
            }
            reference_model = models[0]
            for model in models[1:]:
                if not np.array_equal(sets[model], sets[reference_model]):
                    problems.append(
                        f"{run.name} fold {int(fold)}: {model} was scored on different "
                        f"rows than {reference_model} within the same run, so its models "
                        "were not given the same samples"
                    )
            per_fold[int(fold)] = sets[reference_model]
        rows[run.name] = per_fold

    if len(rows) < 2:
        return problems
    reference_name = next(iter(rows))
    reference = rows[reference_name]
    for name, folds in list(rows.items())[1:]:
        if set(folds) != set(reference):
            problems.append(
                f"{name} persisted predictions for folds {sorted(folds)} but "
                f"{reference_name} for {sorted(reference)}"
            )
            continue
        for fold in sorted(folds):
            if not np.array_equal(folds[fold], reference[fold]):
                problems.append(
                    f"{name} fold {fold} scored {len(folds[fold])} outer rows and "
                    f"{reference_name} scored {len(reference[fold])}, and the row sets "
                    "are not identical, so the two families were not given the same "
                    "samples"
                )
    return problems


def _spread(values: list[float | None]) -> dict[str, Any]:
    """Mean and sample standard deviation over the observations where defined.

    ``None`` is a metric that does not exist for that observation — a portfolio
    that never moved has no Sharpe — never a zero, so it is carried rather than
    averaged in.
    """
    defined = [v for v in values if v is not None]
    return {
        "mean": round(statistics.fmean(defined), 6) if defined else None,
        "std": round(statistics.stdev(defined), 6) if len(defined) > 1 else 0.0,
        "min": round(min(defined), 6) if defined else None,
        "max": round(max(defined), 6) if defined else None,
        "observations": len(values),
        "defined": len(defined),
    }


def _model_run(runs: list[Run], model: str) -> list[Run]:
    """The runs ``model``'s numbers are read from, in the order given."""
    kind = MODEL_KIND[model]
    return [run for run in runs if run.kind == kind and model in run.models()]


def aggregate_model(runs: list[Run], model: str) -> dict[str, Any]:
    """Every outer observation of one model, aggregated across seeds and folds.

    Three dispersions are reported, not one, because they answer different
    questions: ``per_fold`` is how much the market window mattered, ``per_seed``
    is how much the fit's randomness mattered, and ``seed_dispersion`` is the
    spread of the *per-seed means* — the number that says whether a headline
    mean survives a different seed.
    """
    carriers = _model_run(runs, model)
    observations: dict[str, list[float | None]] = {metric: [] for metric in AGGREGATED}
    per_seed: dict[str, dict[str, Any]] = {}
    per_fold: dict[int, list[float | None]] = {}

    for run in carriers:
        net: list[float | None] = []
        for fold in run.folds:
            index = int(fold["fold"])
            per_fold.setdefault(index, []).append(run.metric(fold, model, "net_return"))
            net.append(run.metric(fold, model, "net_return"))
            for metric in AGGREGATED:
                observations[metric].append(run.metric(fold, model, metric))
        per_seed[str(run.seed)] = {
            "net_return": _spread(net),
            "positive_folds": sum(1 for v in net if v is not None and v > 0),
            "folds": len(net),
        }

    seed_means = [
        stats["net_return"]["mean"]
        for stats in per_seed.values()
        if stats["net_return"]["mean"] is not None
    ]
    net_values = [v for v in observations["net_return"] if v is not None]
    return {
        "runs": [run.name for run in carriers],
        "seeds": [run.seed for run in carriers],
        "observations": len(observations["net_return"]),
        "metrics": {metric: _spread(observations[metric]) for metric in AGGREGATED},
        "positive_outer_folds": sum(1 for v in net_values if v > 0),
        "per_seed_net_return": per_seed,
        "seed_dispersion_of_mean_net_return": (
            round(statistics.stdev(seed_means), 6) if len(seed_means) > 1 else 0.0
        ),
        "fold_dispersion_of_mean_net_return": (
            round(
                statistics.stdev(
                    [
                        statistics.fmean([v for v in values if v is not None])
                        for values in per_fold.values()
                        if any(v is not None for v in values)
                    ]
                ),
                6,
            )
            if len(per_fold) > 1
            else 0.0
        ),
        "per_fold_mean_net_return": {
            str(fold): (
                round(statistics.fmean([v for v in values if v is not None]), 6)
                if any(v is not None for v in values)
                else None
            )
            for fold, values in sorted(per_fold.items())
        },
    }


def buy_and_hold_by_fold(runs: list[Run]) -> dict[int, float]:
    """Buy-and-hold net return per fold, from whichever runs recorded it.

    The reference describes the *window*, not the model, so every comparable run
    must report the same number for a given fold; a disagreement means the
    windows differ and :func:`comparability_problems` has already refused.
    """
    holds: dict[int, float] = {}
    for run in runs:
        for fold in run.folds:
            hold = run.references(fold).get("buy_and_hold") or {}
            if hold.get("available"):
                holds[int(fold["fold"])] = float(hold["net_return"])
    return holds


def head_to_head(runs: list[Run], challenger: str, incumbent: str) -> dict[str, Any]:
    """How often, and by how much, ``challenger`` beat ``incumbent`` on net return.

    Compared fold by fold within the same seed, because that is the only pairing
    in which the two saw the same window under the same fit budget. The mean
    difference is reported beside the win count so a model that wins often by a
    hair and loses rarely by a mile cannot read as the better one.
    """
    differences: list[float] = []
    wins = 0
    for run in _model_run(runs, challenger):
        partners = [other for other in _model_run(runs, incumbent) if other.seed == run.seed]
        for other in partners:
            for fold, other_fold in zip(run.folds, other.folds):
                mine = run.metric(fold, challenger, "net_return")
                theirs = other.metric(other_fold, incumbent, "net_return")
                if mine is None or theirs is None:
                    continue
                differences.append(mine - theirs)
                wins += int(mine > theirs)
    return {
        "comparisons": len(differences),
        "challenger_wins": wins,
        "mean_net_return_difference": (
            round(statistics.fmean(differences), 6) if differences else None
        ),
    }


def economic_verdict(runs: list[Run], model: str, holds: dict[int, float]) -> dict[str, Any]:
    """Did this model make money, beat doing nothing, and beat owning the asset?

    CASH earns exactly zero and pays no costs, so "beat CASH" is "net return
    above zero" — stated as a policy comparison because that is the question a
    reader is asking. Kept apart from the baseline comparison on purpose: a model
    can clear majority and momentum in every fold while losing money in every
    fold, and this repository has already published exactly that.
    """
    net: list[float] = []
    beat_hold = 0
    hold_comparisons = 0
    for run in _model_run(runs, model):
        for fold in run.folds:
            value = run.metric(fold, model, "net_return")
            if value is None:
                continue
            net.append(value)
            hold = holds.get(int(fold["fold"]))
            if hold is not None:
                hold_comparisons += 1
                beat_hold += int(value > hold)
    return {
        "observations": len(net),
        "mean_net_return": round(statistics.fmean(net), 6) if net else None,
        "beat_cash_observations": sum(1 for v in net if v > 0),
        "buy_and_hold_comparisons": hold_comparisons,
        "beat_buy_and_hold_observations": beat_hold,
        "made_money": bool(net) and statistics.fmean(net) > 0,
    }


def attribution(runs: list[Run], target_spec: TargetSpec) -> dict[str, Any]:
    """Per-direction trade attribution for every model, from persisted predictions.

    Exact, never inferred from aggregates: :func:`nn.regime.direction_attribution`
    replays the persisted actions through :func:`nn.evaluate.realised_trades`, so
    the LONG/SHORT split uses the one cost model the reports themselves used.
    """
    frames: dict[str, list[pd.DataFrame]] = {}
    missing: list[str] = []
    for run in runs:
        path = run.predictions_path
        if not path.is_file():
            missing.append(run.name)
            continue
        frame = regime.load_predictions(path, sealed_test_start=int(run.sealed_start_row))
        if "model" in frame.columns:
            for model, group in frame.groupby("model", sort=True):
                frames.setdefault(str(model), []).append(group)
        else:
            frames.setdefault(MTST, []).append(frame)

    if not frames:
        return {
            "available": False,
            "reason": "no run persisted per-sample outer predictions",
            "runs_without_predictions": missing,
        }
    return {
        "available": True,
        "runs_without_predictions": missing,
        "per_model": {
            model: regime.direction_attribution(
                pd.concat(parts, ignore_index=True), target_spec
            )
            for model, parts in sorted(frames.items())
        },
    }


def reproduce_reports(runs: list[Run]) -> dict[str, Any]:
    """Recompute each reported outer report from the persisted rows, and compare.

    A report nobody can rebuild from the predictions beside it is a number on
    trust. The candle-level statistics are excluded by name
    (:data:`nn.regime.NON_REPRODUCIBLE_TRADING_METRICS`) rather than approximated:
    they need the price path through the last trade's exit candle, which lies
    past the last scored row and is not in a per-sample file.
    """
    checked = 0
    problems: list[str] = []
    for run in runs:
        path = run.predictions_path
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        target_spec = TargetSpec.from_dict(run.dataset["target_spec"])
        # A predictions file covers the models it names. `nn.walkforward` writes
        # one file of MTST rows and no `model` column; `nn.benchmark` writes one
        # file per model. The baselines are never persisted per sample by either,
        # so they are checked by the parity requirement above, not here.
        models = (
            sorted(set(frame["model"].astype(str))) if "model" in frame.columns else [MTST]
        )
        for fold in run.folds:
            index = int(fold["fold"])
            for model in models:
                if model not in run.models():
                    continue
                part = frame[frame["fold"] == index]
                if "model" in frame.columns:
                    part = part[part["model"] == model]
                if part.empty:
                    continue
                threshold = float(part["threshold"].iloc[0])
                recomputed = _comparable(
                    _report_from_predictions(part, target_spec, threshold)
                )
                recorded = _comparable(fold["outer_validation"][model])
                checked += 1
                if recomputed != recorded:
                    problems.append(
                        f"{run.name} fold {index} {model}: the persisted predictions do "
                        "not reproduce the reported outer metrics"
                    )
    return {"reports_checked": checked, "problems": problems}


def _report_from_predictions(
    part: pd.DataFrame, target_spec: TargetSpec, threshold: float
) -> dict[str, Any]:
    """Rebuild one outer report from persisted rows alone."""
    ordered = part.sort_values("row_index")
    proba = ordered[["p_short", "p_hold", "p_long"]].to_numpy(dtype=np.float64)
    return ev.evaluate(
        proba,
        ordered["true_target"].to_numpy(dtype=np.int64),
        ordered["future_return"].to_numpy(dtype=np.float64),
        target_spec,
        threshold,
        market=None,
        row_index=ordered["row_index"].to_numpy(dtype=np.int64),
    )


def _comparable(report: dict[str, Any]) -> dict[str, Any]:
    """A report reduced to the fields a per-sample predictions file can rebuild."""
    out = {key: value for key, value in report.items() if key != "trading"}
    out["trading"] = {
        key: value
        for key, value in (report.get("trading") or {}).items()
        if key not in regime.NON_REPRODUCIBLE_TRADING_METRICS
    }
    return out


def build_report(
    benchmark_runs: list[Run], mtst_runs: list[Run], target_spec: TargetSpec | None
) -> dict[str, Any]:
    """The whole comparison, once comparability has been established."""
    runs = benchmark_runs + mtst_runs
    holds = buy_and_hold_by_fold(runs)
    reference = benchmark_runs[0]

    per_model = {model: aggregate_model(runs, model) for model in CONTENDERS}
    economics = {model: economic_verdict(runs, model, holds) for model in CONTENDERS}
    versus = {
        model: {
            opponent: head_to_head(runs, model, opponent)
            for opponent in (MTST,) + BASELINE_NAMES
            if opponent != model
        }
        for model in SIMPLE_MODEL_NAMES
    }

    return {
        "checkpoint": "P2a",
        "question": (
            "Does model complexity add predictive or economic value when simple models "
            "receive exactly the same information as MTST?"
        ),
        "research_evidence_status": (
            "P2a is an adaptive follow-up designed after prior outer-validation results "
            "were observed, and it reuses those outer folds. They are research evidence, "
            "not a pristine final out-of-sample test. The sealed test block is unopened "
            "and no number here is a claim about live profitability."
        ),
        "sealed_test": {
            "evaluated": False,
            "anchor_timestamp": reference.sealed_anchor,
            "start_row": reference.sealed_start_row,
        },
        "research_contract": reference.contract,
        "research_input_hash": reference.research_input_hash,
        "information_parity": reference.payload.get("information_parity"),
        "runs": {
            "benchmark": [
                {"name": r.name, "path": str(r.path), "seed": r.seed} for r in benchmark_runs
            ],
            "frozen_mtst": [
                {"name": r.name, "path": str(r.path), "seed": r.seed} for r in mtst_runs
            ],
        },
        "mtst_retrained": False,
        "per_model": per_model,
        "economic_references": {
            "cash_net_return": 0.0,
            "buy_and_hold_net_return_per_fold": {
                str(fold): round(value, 6) for fold, value in sorted(holds.items())
            },
            "mean_buy_and_hold_net_return": (
                round(statistics.fmean(holds.values()), 6) if holds else None
            ),
        },
        "economics": economics,
        "head_to_head": versus,
        "thresholds": {
            model: sorted(
                {
                    round(float(fold["models"][model]["selection"]["threshold"]), 4)
                    for run in benchmark_runs
                    for fold in run.folds
                }
            )
            for model in SIMPLE_MODEL_NAMES
        },
        "mtst_thresholds": sorted(
            {
                round(float(fold["selection"]["threshold"]), 4)
                for run in mtst_runs
                for fold in run.folds
            }
        ),
        "attribution": (
            attribution(runs, target_spec)
            if target_spec is not None
            else {
                "available": False,
                "reason": (
                    "directional attribution needs the dataset's target spec for its "
                    "cost model: re-run with --dataset"
                ),
            }
        ),
        "recomputation": reproduce_reports(runs),
        "answers": answers(per_model, economics, versus, holds),
    }


def answers(
    per_model: dict[str, Any],
    economics: dict[str, Any],
    versus: dict[str, Any],
    holds: dict[int, float],
) -> dict[str, Any]:
    """The checkpoint's questions, answered one line each.

    Two verdict fields per model, never one. ``predictive_baseline_improvement``
    is "did it learn more than a trivial rule"; ``economic_alpha_after_costs`` is
    "did acting on it make money after fees and slippage". They are different
    questions with different answers, and collapsing them into a single
    "beats the baselines" sentence is how a losing model gets read as a working
    one.
    """
    out: dict[str, Any] = {}
    for model in SIMPLE_MODEL_NAMES:
        economic = economics[model]
        mean_net = per_model[model]["metrics"]["net_return"]["mean"]
        beats_floor = all(
            versus[model][name]["mean_net_return_difference"] is not None
            and versus[model][name]["mean_net_return_difference"] > 0
            for name in BASELINE_NAMES
        )
        out[model] = {
            "beats_mtst_on_mean_net_return": (
                versus[model][MTST]["mean_net_return_difference"] is not None
                and versus[model][MTST]["mean_net_return_difference"] > 0
            ),
            "beats_mtst_in_folds": (
                f"{versus[model][MTST]['challenger_wins']}/"
                f"{versus[model][MTST]['comparisons']}"
            ),
            "beats_majority_on_mean_net_return": (
                versus[model]["majority_baseline"]["mean_net_return_difference"] is not None
                and versus[model]["majority_baseline"]["mean_net_return_difference"] > 0
            ),
            "beats_momentum_on_mean_net_return": (
                versus[model]["momentum_baseline"]["mean_net_return_difference"] is not None
                and versus[model]["momentum_baseline"]["mean_net_return_difference"] > 0
            ),
            "beats_cash": bool(mean_net is not None and mean_net > 0),
            "beats_buy_and_hold_in_observations": (
                f"{economic['beat_buy_and_hold_observations']}/"
                f"{economic['buy_and_hold_comparisons']}"
            ),
            "mean_outer_net_return": mean_net,
            "positive_outer_folds": (
                f"{per_model[model]['positive_outer_folds']}/"
                f"{per_model[model]['observations']}"
            ),
            "predictive_baseline_improvement": beats_floor,
            "economic_alpha_after_costs": bool(economic["made_money"]),
            "note": (
                "predictive-baseline improvement is not economic alpha: clearing "
                "majority-class and momentum is a floor test, and only a positive mean "
                "net return after costs — above CASH — is money"
            ),
        }
    out["mtst"] = {
        "mean_outer_net_return": per_model[MTST]["metrics"]["net_return"]["mean"],
        "positive_outer_folds": (
            f"{per_model[MTST]['positive_outer_folds']}/{per_model[MTST]['observations']}"
        ),
        "economic_alpha_after_costs": bool(economics[MTST]["made_money"]),
        "source": "frozen v4 artifacts, read only; not retrained for P2a",
    }
    out["buy_and_hold_mean_net_return"] = (
        round(statistics.fmean(holds.values()), 6) if holds else None
    )
    return out


def to_markdown(report: dict[str, Any]) -> str:
    parity = report.get("information_parity") or {}
    contract = report.get("research_contract") or {}
    lines = [
        "# P2a — do simple models beat MTST on MTST's own information?",
        "",
        report["question"],
        "",
        "Three untuned, predeclared models — logistic regression, LightGBM, XGBoost —",
        "were fitted on the same samples MTST was fitted on: each row is MTST's",
        f"`{parity.get('seq_len')} x {parity.get('n_features')}` window flattened to "
        f"`{parity.get('flattened_input_dim')}` values",
        f"({parity.get('flatten_order')}). MTST was **not retrained**: its numbers are the",
        "frozen v4 artifacts, read only.",
        "",
        f"**Research contract:** `{contract.get('contract_id')}`, "
        f"semantic identity `sha256:{contract.get('contract_hash')}`.",
        "",
        f"**Research input:** `sha256:{report['research_input_hash']}`.",
        "",
        f"**Sealed test:** anchor `{report['sealed_test']['anchor_timestamp']}`, first "
        f"sealed row {report['sealed_test']['start_row']}, `evaluated: false`.",
        "",
        "## Outer validation across every seed and fold",
        "",
        "`ann. Sharpe` and `per-trade Sharpe` are `n/a` where undefined, never zero.",
        "",
        "| model | mean net return | ± std | positive folds | trades | exposure | "
        "max DD | ann. Sharpe | per-trade Sharpe | macro F1 | dir acc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def cell(stats: dict[str, Any], key: str, places: int = 4) -> str:
        value = stats["metrics"][key]["mean"]
        return "n/a" if value is None else format(value, f".{places}f")

    for model in CONTENDERS:
        stats = report["per_model"][model]
        net = stats["metrics"]["net_return"]
        lines.append(
            f"| {model} | {net['mean']:+.4f} | {net['std']:.4f} | "
            f"{stats['positive_outer_folds']}/{stats['observations']} | "
            f"{cell(stats, 'n_trades', 1)} | {cell(stats, 'exposure')} | "
            f"{cell(stats, 'max_drawdown')} | {cell(stats, 'annualised_sharpe', 2)} | "
            f"{cell(stats, 'per_trade_sharpe', 2)} | {cell(stats, 'macro_f1')} | "
            f"{cell(stats, 'directional_accuracy')} |"
        )

    hold = report["economic_references"]["mean_buy_and_hold_net_return"]
    lines += [
        "",
        "### Economic references (not models, not baselines)",
        "",
        "| policy | mean net return over the same windows |",
        "| --- | --- |",
        "| CASH (never trade) | +0.0000 |",
        f"| buy-and-hold | {'n/a' if hold is None else format(hold, '+.4f')} |",
        "",
        "### Dispersion",
        "",
        "| model | seed dispersion of mean net return | fold dispersion | "
        "min fold | max fold |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model in CONTENDERS:
        stats = report["per_model"][model]
        net = stats["metrics"]["net_return"]
        lines.append(
            f"| {model} | {stats['seed_dispersion_of_mean_net_return']:.4f} | "
            f"{stats['fold_dispersion_of_mean_net_return']:.4f} | "
            f"{net['min']:+.4f} | {net['max']:+.4f} |"
        )

    lines += [
        "",
        "### The checkpoint's questions",
        "",
        "| model | beats MTST | beats majority | beats momentum | beats CASH | "
        "beats buy-and-hold | predictive-baseline improvement | economic alpha after costs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in SIMPLE_MODEL_NAMES:
        answer = report["answers"][model]
        yes = lambda flag: "yes" if flag else "no"  # noqa: E731
        lines.append(
            f"| {model} | {yes(answer['beats_mtst_on_mean_net_return'])} "
            f"({answer['beats_mtst_in_folds']} folds) | "
            f"{yes(answer['beats_majority_on_mean_net_return'])} | "
            f"{yes(answer['beats_momentum_on_mean_net_return'])} | "
            f"{yes(answer['beats_cash'])} | "
            f"{answer['beats_buy_and_hold_in_observations']} | "
            f"{yes(answer['predictive_baseline_improvement'])} | "
            f"{yes(answer['economic_alpha_after_costs'])} |"
        )

    lines += [
        "",
        "**Read the last two columns apart.** *Predictive-baseline improvement* means the",
        "model learned more than a trivial rule — a floor test. *Economic alpha after",
        "costs* means acting on it earned money after fees and slippage, above CASH. A",
        "model can pass the first and fail the second, and this repository has already",
        "published exactly that; no model is called profitable here for clearing majority",
        "or momentum.",
        "",
        "### Selected thresholds",
        "",
        "| model | thresholds chosen across folds and seeds |",
        "| --- | --- |",
    ]
    for model in SIMPLE_MODEL_NAMES:
        lines.append(
            f"| {model} | {', '.join(f'{v:.2f}' for v in report['thresholds'][model])} |"
        )
    lines.append(
        f"| mtst (frozen, not retrained) | "
        f"{', '.join(f'{v:.2f}' for v in report['mtst_thresholds'])} |"
    )

    lines += _attribution_markdown(report)

    checked = report["recomputation"]
    lines += [
        "",
        "### Integrity",
        "",
        f"* {checked['reports_checked']} outer report(s) recomputed from the persisted "
        "per-sample predictions and matched"
        + (
            "."
            if not checked["problems"]
            else f"; {len(checked['problems'])} did NOT match: "
            + "; ".join(checked["problems"])
        ),
        "* MTST was not retrained; the frozen v4 artifacts were read only.",
        "* Every run shares one research contract, one research-input fingerprint, one",
        "  sealed anchor and one fold geometry, and the two families scored the same",
        "  outer rows with identical statistical baselines — checked before aggregation.",
        "",
        report["research_evidence_status"],
        "",
    ]
    return "\n".join(lines)


def _attribution_markdown(report: dict[str, Any]) -> list[str]:
    attributions = report["attribution"]
    if not attributions.get("available"):
        return [
            "",
            "### Directional attribution",
            "",
            attributions.get("reason", "unavailable"),
            "",
        ]
    lines = [
        "",
        "### Directional attribution (exact, from persisted predictions)",
        "",
        "| model | LONG trades | LONG hit rate | LONG sum | SHORT trades | "
        "SHORT hit rate | SHORT sum |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model, split in attributions["per_model"].items():
        long_side, short_side = split["long"], split["short"]
        lines.append(
            f"| {model} | {long_side['trades']} | {long_side['hit_rate']:.4f} | "
            f"{long_side['additive_trade_return_sum']:+.4f} | {short_side['trades']} | "
            f"{short_side['hit_rate']:.4f} | "
            f"{short_side['additive_trade_return_sum']:+.4f} |"
        )
    return lines


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare P2a simple-model runs against the frozen MTST v4 evidence."
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help=f"P2a run directories (each containing {ARTIFACT_NAME}).",
    )
    parser.add_argument(
        "--mtst",
        nargs="+",
        required=True,
        help=(
            "Frozen MTST walk-forward directories (each containing "
            f"{MTST_ARTIFACT_NAME}). Read only; never regenerated."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "The built dataset, for the target spec the directional attribution's cost "
            "model needs. Without it, attribution is reported as unavailable rather "
            "than approximated."
        ),
    )
    parser.add_argument("--out", default="artifacts/benchmark/p2a_comparison")
    return parser


def _target_spec(runs: Iterable[Run], dataset: str | None) -> TargetSpec | None:
    """The target spec, from the dataset when given, else from the artifacts.

    The artifacts all record it and comparability has already required them to
    agree, so reading it from disk is a convenience rather than a second source
    of truth — and its absence costs only the attribution section.
    """
    if dataset is not None:
        from nn.train import load_research_data

        return load_research_data(dataset).target_spec
    for run in runs:
        raw = run.dataset.get("target_spec")
        if isinstance(raw, dict):
            return TargetSpec.from_dict(raw)
    return None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    benchmark_runs = [load_run(path, kind="p2a") for path in args.benchmark]
    mtst_runs = [load_run(path, kind="mtst") for path in args.mtst]
    runs = benchmark_runs + mtst_runs

    problems = comparability_problems(runs)
    if problems:
        for problem in problems:
            logger.error("%s", problem)
        raise ComparabilityError(
            f"refusing to aggregate {len(runs)} run(s): {len(problems)} comparability "
            "problem(s) above. These are not measurements of one experiment, and an "
            "average over them would not mean anything."
        )

    report = build_report(benchmark_runs, mtst_runs, _target_spec(runs, args.dataset))
    if report["recomputation"]["problems"]:
        for problem in report["recomputation"]["problems"]:
            logger.error("%s", problem)
        raise ComparabilityError(
            "refusing to publish a comparison whose reported metrics cannot be "
            "recomputed from the persisted predictions"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_JSON).write_text(json.dumps(report, indent=2, default=str) + "\n")
    markdown = to_markdown(report)
    (out_dir / REPORT_MD).write_text(markdown)
    print(markdown)
    logger.info("Wrote the P2a comparison to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
