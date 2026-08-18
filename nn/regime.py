"""Market-regime statistics for a walk-forward outer block.

``nn.wf_diagnostics`` can tell you that fold 2 earned more than fold 1. It
cannot tell you *what was different about the market* in those two stretches,
because it never opens a dataset. This module does, under three rules.

**The sealed boundary is enforced by slicing, not by discipline.**
:func:`load_research_frame` truncates the processed dataset at
``sealed_test_start`` the moment it is read. Every statistic below is computed
from that object, so a sealed row is not merely unused — it is not in the frame
at all, and no off-by-one in a block range can reach one. The row range is
checked against the boundary again on the way in, because a cheap assertion is
worth more than a comment.

**Model-facing statistics describe the rows the model was scored on, not the
block they sit in.** An outer block of 4,821 rows does not yield 4,821 samples:
the first ``seq_len - 1`` rows cannot open a window, the last ``horizon`` rows
cannot close a label, and any candidate straddling a market-data gap is dropped.
Summarising the block instead of the samples would describe a slightly different
stretch of market than the number it is being used to explain, so
:func:`block_statistics` selects rows through :func:`nn.dataset.sample_indices`
— the same function ``build_windows`` uses — and reports both counts so the
difference is visible rather than assumed away.

**Row indices are only meaningful against the dataset they came from.** The
indices in ``walkforward.json`` address one specific processed dataset. Point
this at a rebuilt one and every index silently means a different candle, so the
loader refuses a dataset that disagrees with the artifact on any recorded
identity field — exchange, pair, timeframe, row count, first and last timestamp,
feature contract or target spec — and cross-checks the metadata against the
frame's own first and last timestamps.

**Raw candles are joined on timestamps, never on position.** The processed
dataset has lost warm-up and label rows, so processed row *i* is not raw row
*i*. :func:`align_raw` matches exact timestamps and fails closed on a missing
one, a duplicate, or a timeframe that does not agree — rather than shifting
rows and reporting a confident number about the wrong hours.

Nothing here explains anything. It reports what differed; whether a difference
caused anything is not a question this data can answer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chimera.contracts import CLASS_ORDER, TargetSpec
from nn import evaluate as ev
from nn.data_pipeline import OHLCV_COLUMNS, load_dataset, timeframe_to_minutes
from nn.dataset import Split, sample_indices

logger = logging.getLogger(__name__)

FEATURE_STATS: dict[str, tuple[str, ...]] = {
    "ema_cross": ("mean", "median", "std", "fraction_positive", "fraction_negative"),
    "ema_fast_ratio": ("mean", "median"),
    "ema_slow_ratio": ("mean", "median"),
    "atr_norm": ("mean", "median", "p90"),
    "realized_vol": ("mean", "median", "p90"),
    "hl_range": ("mean", "median"),
    "volume_change": ("mean", "median"),
    "volume_z": ("mean", "median", "p90_abs"),
}

_STATISTICS = {
    "mean": lambda x: float(np.mean(x)),
    "median": lambda x: float(np.median(x)),
    "std": lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    "p90": lambda x: float(np.percentile(x, 90)),
    "p90_abs": lambda x: float(np.percentile(np.abs(x), 90)),
    "fraction_positive": lambda x: float(np.mean(x > 0)),
    "fraction_negative": lambda x: float(np.mean(x < 0)),
}

_CLASS_NAMES = [c.value for c in CLASS_ORDER]


class RegimeDataError(Exception):
    """The dataset cannot be trusted to mean what the artifact's rows say."""


@dataclass(frozen=True)
class ResearchFrame:
    """The processed dataset, already truncated at the sealed boundary."""

    frame: pd.DataFrame
    sealed_test_start: int
    feature_names: list[str]
    target_spec: TargetSpec
    timeframe: str
    segment_ids: np.ndarray | None = None

    def block(self, start: int, end: int) -> pd.DataFrame:
        """Rows ``[start, end)``, refusing anything at or past the boundary."""
        if not 0 <= start < end:
            raise RegimeDataError(f"invalid block range [{start}, {end})")
        if end > self.sealed_test_start:
            raise RegimeDataError(
                f"block [{start}, {end}) reaches row {end - 1}, at or beyond the sealed "
                f"test block starting at {self.sealed_test_start}"
            )
        return self.frame.iloc[start:end]

    def scored_rows(self, start: int, end: int, seq_len: int) -> np.ndarray:
        """The rows in ``[start, end)`` the model was actually evaluated on."""
        self.block(start, end)
        idx = sample_indices(
            Split("outer_validation", start, end),
            seq_len,
            self.target_spec.horizon,
            segment_ids=self.segment_ids,
        )
        if idx.size == 0:
            raise RegimeDataError(
                f"block [{start}, {end}) yields no scored samples at seq_len={seq_len} "
                f"and horizon={self.target_spec.horizon}"
            )
        return idx


IDENTITY_FIELDS = (
    "rows",
    "exchange",
    "pair",
    "timeframe",
    "start",
    "end",
    "feature_names",
    "feature_spec",
    "target_spec",
)


def _identity_mismatch(name: str, actual: Any, expected: Any) -> str | None:
    """Compare one identity field, normalising containers that JSON round-trips."""
    if expected is None:
        return None
    if isinstance(expected, Mapping):
        actual, expected = dict(actual or {}), dict(expected)
    elif isinstance(expected, (list, tuple)):
        actual, expected = list(actual or []), list(expected)
    if actual == expected:
        return None
    return f"{name} is {actual!r} but the artifact recorded {expected!r}"


def load_research_frame(
    path: str | Path,
    *,
    sealed_test_start: int,
    identity: Mapping[str, Any] | None = None,
) -> ResearchFrame:
    """Read the processed dataset and cut it at the sealed boundary immediately."""
    frame, meta = load_dataset(path)
    n_rows = len(frame)
    identity = dict(identity or {})

    for column in ("date", "target", "future_return"):
        if column not in frame.columns:
            raise RegimeDataError(f"{path} has no {column!r} column")

    actual = {**meta.to_dict(), "rows": n_rows}
    problems = [
        message
        for name in IDENTITY_FIELDS
        if (message := _identity_mismatch(name, actual.get(name), identity.get(name)))
    ]
    if problems:
        raise RegimeDataError(
            f"{path} is not the dataset this artifact was produced against — row "
            "indices would address different candles, so no statistics are computed. "
            + "; ".join(problems)
        )

    dates = pd.to_datetime(frame["date"], utc=True)
    for label, recorded, observed in (
        ("start", meta.start, dates.iloc[0]),
        ("end", meta.end, dates.iloc[-1]),
    ):
        if recorded and pd.Timestamp(recorded, tz="UTC") != observed:
            raise RegimeDataError(
                f"{path} metadata says the data {label}s at {recorded} but the frame "
                f"{label}s at {observed}"
            )

    if sealed_test_start > n_rows:
        raise RegimeDataError(
            f"the artifact's sealed boundary is row {sealed_test_start} but {path} has "
            f"only {n_rows} rows"
        )

    research = frame.iloc[:sealed_test_start].copy()
    research["date"] = dates.iloc[:sealed_test_start].to_numpy()
    segment_ids = (
        research["segment_id"].to_numpy(dtype=np.int64)
        if "segment_id" in research.columns
        else None
    )
    return ResearchFrame(
        frame=research,
        sealed_test_start=sealed_test_start,
        feature_names=list(meta.feature_names),
        target_spec=TargetSpec.from_dict(meta.target_spec),
        timeframe=meta.timeframe or "1h",
        segment_ids=segment_ids,
    )


def _summarise(values: np.ndarray, statistics: Sequence[str]) -> dict[str, float]:
    return {name: round(_STATISTICS[name](values), 8) for name in statistics}


def block_statistics(
    research: ResearchFrame, start: int, end: int, seq_len: int
) -> dict[str, Any]:
    """What the processed dataset says about the rows the model was scored on."""
    scored = research.scored_rows(start, end, seq_len)
    block = research.frame.iloc[scored]
    future_return = block["future_return"].to_numpy(dtype=np.float64)
    targets = block["target"].to_numpy(dtype=np.int64)

    missing = [name for name in FEATURE_STATS if name not in block.columns]
    if missing:
        raise RegimeDataError(
            f"the dataset is missing feature column(s) {missing}; regime statistics "
            "are defined against the feature names the artifact recorded"
        )

    features = {
        name: _summarise(block[name].to_numpy(dtype=np.float64), statistics)
        for name, statistics in FEATURE_STATS.items()
    }

    counts = np.bincount(targets, minlength=len(_CLASS_NAMES))
    target_distribution: dict[str, Any] = {}
    for index, name in enumerate(_CLASS_NAMES):
        selected = future_return[targets == index]
        target_distribution[name] = {
            "count": int(counts[index]),
            "fraction": round(float(counts[index] / len(block)), 8),
            "mean_future_return": round(float(selected.mean()), 8) if selected.size else 0.0,
            "median_future_return": (
                round(float(np.median(selected)), 8) if selected.size else 0.0
            ),
        }

    return {
        "row_range": [start, end],
        "block_rows": end - start,
        "scored_rows": len(scored),
        "scored_row_range": [int(scored[0]), int(scored[-1]) + 1],
        "seq_len": seq_len,
        "horizon": research.target_spec.horizon,
        "period": {
            "start": str(block["date"].iloc[0]),
            "end": str(block["date"].iloc[-1]),
        },
        "future_return": {
            "mean": round(float(future_return.mean()), 8),
            "median": round(float(np.median(future_return)), 8),
            "std": round(float(future_return.std(ddof=1)), 8),
            "mean_abs": round(float(np.abs(future_return).mean()), 8),
            "fraction_positive": round(float((future_return > 0).mean()), 8),
            "fraction_negative": round(float((future_return < 0).mean()), 8),
        },
        "features": features,
        "target_distribution": target_distribution,
    }


def load_raw_ohlcv(path: str | Path) -> pd.DataFrame:
    """Read raw candles as a timestamp-indexed lookup table."""
    frame = pd.read_parquet(path)
    missing = [c for c in ("date", *OHLCV_COLUMNS) if c not in frame.columns]
    if missing:
        raise RegimeDataError(f"{path} is missing OHLCV column(s): {missing}")

    dates = pd.to_datetime(frame["date"], utc=True)
    if dates.isna().any():
        raise RegimeDataError(f"{path} has {int(dates.isna().sum())} unparseable timestamps")
    duplicated = dates.duplicated()
    if duplicated.any():
        examples = dates[duplicated].head(3).tolist()
        raise RegimeDataError(
            f"{path} has {int(duplicated.sum())} duplicate timestamps (e.g. {examples}); "
            "a timestamp join would be ambiguous"
        )

    out = frame.assign(date=dates).sort_values("date").set_index("date")
    return out[list(OHLCV_COLUMNS)]


def align_raw(raw: pd.DataFrame, timestamps: pd.Series, timeframe: str) -> pd.DataFrame:
    """Match processed timestamps to raw candles exactly. Never positionally."""
    wanted = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True), utc=True)
    if wanted.duplicated().any():
        raise RegimeDataError("the processed block contains duplicate timestamps")

    known = raw.index
    present = wanted.isin(known)
    if not present.all():
        absent = wanted[~present]
        raise RegimeDataError(
            f"{int((~present).sum())} of {len(wanted)} processed timestamps have no raw "
            f"candle (e.g. {absent.head(3).tolist()}); refusing to align by position"
        )

    expected = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
    deltas = known.to_series().diff().dropna()
    if not deltas.empty and deltas.median() != expected:
        raise RegimeDataError(
            f"raw candles are spaced {deltas.median()} apart but the dataset's timeframe "
            f"{timeframe!r} implies {expected}"
        )
    return raw.loc[wanted.to_numpy()]


def raw_block_statistics(
    raw: pd.DataFrame, timestamps: pd.Series, timeframe: str
) -> dict[str, Any]:
    """Market behaviour over one outer block, from the raw candles it covers."""
    candles = align_raw(raw, timestamps, timeframe)
    close = candles["close"].to_numpy(dtype=np.float64)
    open_ = candles["open"].to_numpy(dtype=np.float64)

    step = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
    gaps = candles.index.to_series().diff().to_numpy()[1:]
    adjacent = gaps == step.to_timedelta64()
    returns = (close[1:] / close[:-1] - 1.0)[adjacent]

    peak = np.maximum.accumulate(close)
    candles_per_year = 365 * 24 * 60 / timeframe_to_minutes(timeframe)

    return {
        "candles": len(candles),
        "start_close": round(float(close[0]), 8),
        "end_close": round(float(close[-1]), 8),
        "market_return": round(float(close[-1] / close[0] - 1.0), 8),
        "mean_candle_return": round(float(returns.mean()), 8) if returns.size else 0.0,
        "median_candle_return": (round(float(np.median(returns)), 8) if returns.size else 0.0),
        "candle_return_std": (
            round(float(returns.std(ddof=1)), 8) if returns.size > 1 else 0.0
        ),
        "annualised_volatility": (
            round(float(returns.std(ddof=1) * np.sqrt(candles_per_year)), 8)
            if returns.size > 1
            else 0.0
        ),
        "mean_abs_candle_return": (
            round(float(np.abs(returns).mean()), 8) if returns.size else 0.0
        ),
        "positive_candle_fraction": round(float((close > open_).mean()), 8),
        "negative_candle_fraction": round(float((close < open_).mean()), 8),
        "max_drawdown": round(float((1.0 - close / peak).max()), 8),
        "gap_pairs_skipped": int((~adjacent).sum()),
    }


PREDICTION_COLUMNS = (
    "fold",
    "seed",
    "row_index",
    "timestamp",
    "true_target",
    "future_return",
    "p_short",
    "p_hold",
    "p_long",
    "selected_action",
    "threshold",
)


def _prediction_artifact(path: Path, sealed_test_start: int) -> dict[str, Any]:
    """Load the adjacent walk-forward artifact that explicitly declares ``path``."""
    artifact_path = path.parent / "walkforward.json"
    if not artifact_path.is_file():
        raise RegimeDataError(
            f"{path} has no adjacent walkforward.json; prediction identity cannot be verified"
        )
    try:
        artifact = json.loads(artifact_path.read_text())
    except json.JSONDecodeError as exc:
        raise RegimeDataError(f"{artifact_path} is not readable JSON: {exc}") from exc
    if not isinstance(artifact, dict):
        raise RegimeDataError(f"{artifact_path} is not a JSON object")

    declared = artifact.get("outer_predictions")
    if declared != path.name:
        raise RegimeDataError(
            f"{path} exists but {artifact_path} declares outer_predictions={declared!r}; "
            "refusing a stale or unrelated prediction file"
        )
    recorded_boundary = (artifact.get("sealed_test") or {}).get("start_row")
    if recorded_boundary is None or int(recorded_boundary) != sealed_test_start:
        raise RegimeDataError(
            f"{artifact_path} sealed boundary is {recorded_boundary!r}, but diagnostics "
            f"expects {sealed_test_start}"
        )
    return artifact


def _integral_column(frame: pd.DataFrame, column: str, path: Path) -> np.ndarray:
    """Return an integer column without silently truncating malformed floats."""
    try:
        numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RegimeDataError(f"{path} has non-numeric {column} values") from exc
    if not np.isfinite(numeric).all():
        raise RegimeDataError(f"{path} has non-finite {column} values")
    integer = numeric.astype(np.int64)
    if not np.array_equal(numeric, integer.astype(np.float64)):
        raise RegimeDataError(f"{path} has non-integral {column} values")
    return integer


def _prediction_report(
    part: pd.DataFrame, artifact: Mapping[str, Any], threshold: float
) -> dict:
    """Recompute the MTST report from persisted rows to bind parquet to JSON content."""
    target_raw = (artifact.get("dataset") or {}).get("target_spec")
    timeframe = (artifact.get("dataset") or {}).get("timeframe") or "1h"
    if not isinstance(target_raw, Mapping):
        raise RegimeDataError("walkforward artifact has no dataset.target_spec")
    target_spec = TargetSpec.from_dict(target_raw)
    candles_per_year = 365 * 24 * 60 / timeframe_to_minutes(str(timeframe))

    ordered = part.sort_values("row_index")
    proba = ordered[["p_short", "p_hold", "p_long"]].to_numpy(dtype=np.float64)
    if not np.isfinite(proba).all():
        raise RegimeDataError("prediction probabilities contain non-finite values")
    actions = _integral_column(ordered, "selected_action", Path("outer_predictions.parquet"))
    recomputed_actions = ev.signals_from_proba(proba, threshold)
    if not np.array_equal(actions, recomputed_actions):
        raise RegimeDataError(
            "persisted selected_action does not match probabilities and selected threshold"
        )

    targets = _integral_column(ordered, "true_target", Path("outer_predictions.parquet"))
    return ev.evaluate(
        proba,
        targets,
        ordered["future_return"].to_numpy(dtype=np.float64),
        target_spec,
        threshold,
        candles_per_year=candles_per_year,
        row_index=ordered["row_index"].to_numpy(dtype=np.int64),
    )


def _validate_predictions_against_artifact(
    frame: pd.DataFrame, artifact: Mapping[str, Any], path: Path
) -> None:
    """Prove the parquet is the sample-level companion of its JSON artifact."""
    folds = artifact.get("folds")
    base_seed = (artifact.get("config") or {}).get("seed")
    if not isinstance(folds, list) or not folds:
        raise RegimeDataError(f"{path.parent / 'walkforward.json'} has no folds to validate")
    if not isinstance(base_seed, int):
        raise RegimeDataError(
            f"{path.parent / 'walkforward.json'} has no integer config.seed to validate"
        )

    fold_values = _integral_column(frame, "fold", path)
    seed_values = _integral_column(frame, "seed", path)
    row_values = _integral_column(frame, "row_index", path)
    frame = frame.copy()
    frame["fold"] = fold_values
    frame["seed"] = seed_values
    frame["row_index"] = row_values

    actual_folds = set(fold_values.tolist())
    expected_folds = {int(fold.get("fold", position)) for position, fold in enumerate(folds)}
    if actual_folds != expected_folds:
        raise RegimeDataError(
            f"{path} fold ids are {sorted(actual_folds)}, artifact records "
            f"{sorted(expected_folds)}"
        )

    for position, fold in enumerate(folds):
        fold_id = int(fold.get("fold", position))
        part = frame[frame["fold"] == fold_id]
        samples = (fold.get("samples") or {}).get("outer_validation")
        if samples is None:
            raise RegimeDataError(
                f"artifact fold {fold_id} has no samples.outer_validation to validate {path}"
            )
        if len(part) != int(samples):
            raise RegimeDataError(
                f"{path} fold {fold_id} has {len(part)} rows, artifact scored {samples}"
            )

        outer = ((fold.get("periods") or {}).get("outer_validation") or {}).get("row_range")
        if not isinstance(outer, list) or len(outer) != 2:
            raise RegimeDataError(
                f"artifact fold {fold_id} has no outer_validation row_range to validate {path}"
            )
        start, end = int(outer[0]), int(outer[1])
        rows = part["row_index"].to_numpy(dtype=np.int64)
        if len(np.unique(rows)) != len(rows):
            raise RegimeDataError(f"{path} fold {fold_id} contains duplicate row_index values")
        if np.any(rows < start) or np.any(rows >= end):
            bad = rows[(rows < start) | (rows >= end)][0]
            raise RegimeDataError(
                f"{path} fold {fold_id} contains row {int(bad)} outside artifact outer "
                f"range [{start}, {end})"
            )

        seeds = set(part["seed"].tolist())
        expected_seed = base_seed + fold_id
        if seeds != {expected_seed}:
            raise RegimeDataError(
                f"{path} fold {fold_id} seed values are {sorted(seeds)}, expected "
                f"only {expected_seed} from artifact config.seed={base_seed}"
            )

        threshold_raw = (fold.get("selection") or {}).get("threshold")
        if threshold_raw is None:
            raise RegimeDataError(f"artifact fold {fold_id} has no selection.threshold")
        expected_threshold = float(threshold_raw)
        thresholds = part["threshold"].to_numpy(dtype=np.float64)
        if not np.isfinite(thresholds).all() or not np.allclose(
            thresholds, expected_threshold, rtol=0.0, atol=1e-12
        ):
            raise RegimeDataError(
                f"{path} fold {fold_id} threshold does not match artifact selection "
                f"threshold {expected_threshold}"
            )

        recorded_report = (fold.get("outer_validation") or {}).get("mtst")
        if not isinstance(recorded_report, Mapping):
            raise RegimeDataError(
                f"artifact fold {fold_id} has no outer_validation.mtst report"
            )
        try:
            recomputed_report = _prediction_report(part, artifact, expected_threshold)
        except (TypeError, ValueError) as exc:
            raise RegimeDataError(
                f"{path} fold {fold_id} cannot be recomputed against its artifact: {exc}"
            ) from exc
        if recomputed_report != dict(recorded_report):
            raise RegimeDataError(
                f"{path} fold {fold_id} does not reproduce the artifact's MTST report; "
                "the parquet is stale, unrelated, or was produced under different "
                "evaluation semantics"
            )


def load_predictions(path: str | Path, *, sealed_test_start: int) -> pd.DataFrame:
    """Read outer predictions only after enforcing sealing and artifact identity."""
    path = Path(path)
    frame = pd.read_parquet(path)
    missing = [c for c in PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise RegimeDataError(f"{path} is missing prediction column(s): {missing}")
    if frame.empty:
        raise RegimeDataError(f"{path} contains no predictions")

    row_values = _integral_column(frame, "row_index", path)
    highest = int(row_values.max())
    if highest >= sealed_test_start:
        raise RegimeDataError(
            f"{path} contains row {highest}, at or beyond the sealed test block starting "
            f"at {sealed_test_start}. This artifact is not research output; refusing it."
        )

    artifact = _prediction_artifact(path, sealed_test_start)
    _validate_predictions_against_artifact(frame, artifact, path)
    frame = frame.copy()
    frame["fold"] = _integral_column(frame, "fold", path)
    frame["seed"] = _integral_column(frame, "seed", path)
    frame["row_index"] = row_values
    frame["_run_name"] = path.parent.name
    return frame


def direction_attribution(
    predictions: pd.DataFrame, target_spec: TargetSpec
) -> dict[str, Any]:
    """Split realised trades into LONG and SHORT with exact run and row identity."""
    short_idx, hold_idx, long_idx = range(len(CLASS_ORDER))
    directions: list[np.ndarray] = []
    returns: list[np.ndarray] = []

    group_keys = ["fold", "seed"]
    if "_run_name" in predictions.columns:
        group_keys.insert(0, "_run_name")
    ordered = predictions.sort_values([*group_keys, "row_index"])
    for _, group in ordered.groupby(group_keys, sort=True):
        _, group_directions, group_returns = ev.realised_trades(
            group["selected_action"].to_numpy(dtype=np.int64),
            group["future_return"].to_numpy(dtype=np.float64),
            target_spec,
            row_index=group["row_index"].to_numpy(dtype=np.int64),
        )
        directions.append(group_directions)
        returns.append(group_returns)

    all_directions = np.concatenate(directions) if directions else np.empty(0)
    all_returns = np.concatenate(returns) if returns else np.empty(0)

    def side(mask: np.ndarray) -> dict[str, Any]:
        selected = all_returns[mask]
        return {
            "trades": int(mask.sum()),
            "hit_rate": round(float((selected > 0).mean()), 6) if selected.size else 0.0,
            "mean_net_return": round(float(selected.mean()), 8) if selected.size else 0.0,
            "median_net_return": (
                round(float(np.median(selected)), 8) if selected.size else 0.0
            ),
            "additive_trade_return_sum": round(float(selected.sum()), 8),
        }

    actions = predictions["selected_action"].to_numpy(dtype=np.int64)
    n_samples = len(actions)
    return {
        "samples": n_samples,
        "long": side(all_directions > 0),
        "short": side(all_directions < 0),
        "hold_samples": int((actions == hold_idx).sum()),
        "long_coverage": round(float((actions == long_idx).mean()), 6),
        "short_coverage": round(float((actions == short_idx).mean()), 6),
    }
