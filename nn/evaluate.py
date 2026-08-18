"""Classification and trading metrics.

Two separate questions, deliberately kept apart:

*Is the classifier any good?* — class distribution, per-class precision/recall/
F1, confusion matrix, calibration error, directional accuracy on the trades it
actually calls, and coverage (how often it calls one at all). Accuracy alone is
excluded from the headline on purpose: with cost-aware labels the HOLD class
dominates, so "accuracy" mostly measures the class prior.

*Would acting on it have made money?* — :func:`trading_metrics` turns a signal
series into non-overlapping trades, charges fees and slippage on both sides,
and reports net return, Sharpe, max drawdown, win rate, profit factor, trade
count, exposure, average trade and turnover.

:func:`trading_metrics` is a *signal evaluation*, not a backtest. It assumes
entry and exit at the close of their respective candles and applies a flat cost.
Freqtrade's backtester remains the authority on execution; this exists so that
model selection can be made against a cost-aware objective instead of accuracy.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from chimera.contracts import CLASS_ORDER, HOLD_IDX, LONG_IDX, SHORT_IDX, TargetSpec

N_CLASSES = len(CLASS_ORDER)
_CLASS_NAMES = [c.value for c in CLASS_ORDER]

#: Position implied by each class index: SHORT -> -1, HOLD -> 0, LONG -> +1.
_DIRECTION = np.array([-1.0, 0.0, 1.0])


def signals_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to class indices under a confidence threshold.

    Mirrors :func:`chimera.contracts.decide` exactly, in vectorised form: a
    non-HOLD class is only emitted when it both wins and clears ``threshold``.
    If these two ever disagree, the strategy and the offline evaluation are
    measuring different systems.
    """
    proba = np.asarray(proba, dtype=np.float64)
    out = np.full(len(proba), HOLD_IDX, dtype=np.int64)
    p_short = proba[:, SHORT_IDX]
    p_long = proba[:, LONG_IDX]
    out[(p_long >= threshold) & (p_long > p_short)] = LONG_IDX
    out[(p_short >= threshold) & (p_short > p_long)] = SHORT_IDX
    return out


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Rows are true classes, columns predicted."""
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        matrix[t, p] += 1
    return matrix


def expected_calibration_error(
    proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> float:
    """Gap between confidence and accuracy, averaged over confidence bins.

    Zero means the model's stated 70% confidence is right 70% of the time.
    Matters here because the trading threshold *is* a probability cut: an
    overconfident model trades far more than its threshold implies.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if len(proba) == 0:
        return 0.0
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    correct = (predicted == y_true.astype(int)).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count:
            error += (count / len(proba)) * abs(
                correct[in_bin].mean() - confidence[in_bin].mean()
            )
    return float(error)


def classification_metrics(
    proba: np.ndarray, y_true: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Per-class and aggregate classification quality."""
    proba = np.asarray(proba, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = signals_from_proba(proba, threshold)
    matrix = confusion_matrix(y_true, y_pred)

    per_class: dict[str, dict[str, float]] = {}
    f1_scores = []
    for idx, name in enumerate(_CLASS_NAMES):
        true_positive = int(matrix[idx, idx])
        predicted = int(matrix[:, idx].sum())
        actual = int(matrix[idx, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": actual,
            "predicted": predicted,
        }

    # Directional accuracy: of the trades actually called, how many had the
    # right sign? HOLD predictions are excluded — they are not directional
    # calls, and counting them would inflate the number.
    traded = y_pred != HOLD_IDX
    n_traded = int(traded.sum())
    if n_traded:
        directional = float((_DIRECTION[y_pred[traded]] == _DIRECTION[y_true[traded]]).mean())
    else:
        directional = 0.0

    counts = np.bincount(y_true, minlength=N_CLASSES)
    return {
        "n_samples": int(len(y_true)),
        "threshold": round(float(threshold), 4),
        "class_distribution": {name: int(counts[i]) for i, name in enumerate(_CLASS_NAMES)},
        "predicted_distribution": {
            name: int((y_pred == i).sum()) for i, name in enumerate(_CLASS_NAMES)
        },
        "per_class": per_class,
        "macro_f1": round(float(np.mean(f1_scores)), 4),
        "accuracy": round(float((y_pred == y_true).mean()), 4),
        "directional_accuracy": round(directional, 4),
        "coverage": round(n_traded / len(y_true), 4) if len(y_true) else 0.0,
        "calibration_error": round(expected_calibration_error(proba, y_true), 4),
        "confusion_matrix": {
            "order": _CLASS_NAMES,
            "rows_are_true": True,
            "matrix": matrix.tolist(),
        },
    }


def _trade_rows(n: int, row_index: np.ndarray | None) -> np.ndarray:
    """Return the candle-row coordinate for each scored sample.

    A scored array can be compressed around market-data gaps because samples
    whose window/label crosses a gap are removed. In that case array position is
    not candle time, so non-overlap must be enforced against the persisted
    dataset row indices instead. With no explicit indices, preserve the historic
    contiguous-array behaviour exactly.
    """
    if row_index is None:
        return np.arange(n, dtype=np.int64)
    rows = np.asarray(row_index, dtype=np.int64)
    if rows.ndim != 1 or len(rows) != n:
        raise ValueError("row_index must be a 1-D array with one entry per signal")
    if len(rows) > 1 and np.any(np.diff(rows) <= 0):
        raise ValueError("row_index must be strictly increasing")
    return rows


def realised_trades(
    signals: np.ndarray,
    future_return: np.ndarray,
    target_spec: TargetSpec,
    *,
    row_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The non-overlapping trades a signal series actually takes.

    Returns ``(positions, directions, net_returns)``: the index into ``signals``
    each trade was entered at, its direction (``-1.0`` short, ``+1.0`` long), and
    its return after the round-trip cost.

    Trades are taken greedily in time order and held for ``horizon`` candles;
    signals that fire while a position is open are ignored. When ``row_index``
    is supplied, candle distance is measured in those dataset rows rather than
    compressed-array positions, so a market-data gap cannot make a pre-gap trade
    suppress valid post-gap signals.

    This is the single definition of "what trades did this signal series take,
    and what did each one net". :func:`trading_metrics` aggregates it, and
    per-direction attribution splits it — neither reimplements the cost model,
    so a change to costs cannot land in one and not the other.
    """
    signals = np.asarray(signals, dtype=np.int64)
    future_return = np.asarray(future_return, dtype=np.float64)
    if len(signals) != len(future_return):
        raise ValueError("signals and future_return must have the same length")

    horizon = target_spec.horizon
    cost = target_spec.cost_threshold  # round-trip fees + slippage
    rows = _trade_rows(len(signals), row_index)

    positions: list[int] = []
    directions: list[float] = []
    returns: list[float] = []
    i = 0
    n = len(signals)
    while i < n:
        direction = _DIRECTION[signals[i]]
        if direction == 0.0:
            i += 1
            continue
        entry_row = int(rows[i])
        positions.append(i)
        directions.append(float(direction))
        returns.append(direction * future_return[i] - cost)
        i += 1
        next_row = entry_row + horizon
        while i < n and int(rows[i]) < next_row:
            i += 1

    return (
        np.asarray(positions, dtype=np.int64),
        np.asarray(directions, dtype=np.float64),
        np.asarray(returns, dtype=np.float64),
    )


def trading_metrics(
    signals: np.ndarray,
    future_return: np.ndarray,
    target_spec: TargetSpec,
    *,
    candles_per_year: float = 24 * 365,
    row_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score a signal series as a sequence of non-overlapping trades.

    The trades themselves come from :func:`realised_trades`; everything here is
    aggregation over them. ``row_index`` keeps both trade spacing and exposure
    in candle-row coordinates when the scored array is discontinuous around
    gaps. Without explicit indices, the historic contiguous-array semantics are
    preserved exactly.
    """
    horizon = target_spec.horizon
    cost = target_spec.cost_threshold
    n = len(signals)
    rows = _trade_rows(n, row_index)
    _, _, returns = realised_trades(signals, future_return, target_spec, row_index=rows)

    n_trades = len(returns)
    if n_trades == 0:
        return {
            "n_trades": 0,
            "net_return": 0.0,
            "gross_return": 0.0,
            "total_costs": 0.0,
            "avg_trade": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "exposure": 0.0,
            "turnover": 0.0,
            "cost_per_trade": round(cost, 6),
        }

    trade_returns = returns
    equity = np.cumprod(1.0 + trade_returns)
    peak = np.maximum.accumulate(equity)
    max_drawdown = float((1.0 - equity / peak).max())

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    std = float(trade_returns.std(ddof=1)) if n_trades > 1 else 0.0
    trades_per_year = candles_per_year / horizon
    sharpe = float(trade_returns.mean() / std * np.sqrt(trades_per_year)) if std > 0 else 0.0
    covered_rows = int(rows[-1] - rows[0] + 1)

    return {
        "n_trades": n_trades,
        "net_return": round(float(equity[-1] - 1.0), 6),
        "gross_return": round(float(trade_returns.sum() + n_trades * cost), 6),
        "total_costs": round(n_trades * cost, 6),
        "avg_trade": round(float(trade_returns.mean()), 6),
        "win_rate": round(len(wins) / n_trades, 4),
        "profit_factor": (
            round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf")
        ),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 4),
        "exposure": round(min(1.0, n_trades * horizon / covered_rows), 4),
        "turnover": round(2.0 * n_trades, 1),
        "cost_per_trade": round(cost, 6),
    }


def evaluate(
    proba: np.ndarray,
    y_true: np.ndarray,
    future_return: np.ndarray,
    target_spec: TargetSpec,
    threshold: float,
    *,
    candles_per_year: float = 24 * 365,
    row_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full report for one model on one split."""
    signals = signals_from_proba(proba, threshold)
    return {
        "classification": classification_metrics(proba, y_true, threshold),
        "trading": trading_metrics(
            signals,
            future_return,
            target_spec,
            candles_per_year=candles_per_year,
            row_index=row_index,
        ),
    }


def select_threshold(
    proba: np.ndarray,
    future_return: np.ndarray,
    target_spec: TargetSpec,
    *,
    grid: np.ndarray | None = None,
    min_trades: int = 10,
    row_index: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Pick the decision threshold on a validation split.

    Chosen to maximise net return after costs, subject to producing at least
    ``min_trades`` trades — a threshold that fires three times and gets lucky is
    not a threshold. Must never be called with test data: the returned value is
    a fitted parameter.

    ``row_index`` is the validation sample's dataset-row coordinate. It keeps
    threshold selection on the same gap-aware trade semantics used by frozen
    evaluation and attribution.

    Falls back to the most permissive grid point when no threshold clears the
    trade floor, and the caller sees ``n_trades`` in the returned report and can
    judge for itself.
    """
    if grid is None:
        grid = np.round(np.arange(0.34, 0.91, 0.02), 4)

    best_threshold = float(grid[0])
    best_report: dict[str, Any] = {}
    best_score = -np.inf

    for threshold in grid:
        signals = signals_from_proba(proba, float(threshold))
        report = trading_metrics(signals, future_return, target_spec, row_index=row_index)
        if report["n_trades"] < min_trades:
            continue
        score = report["net_return"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_report = report

    if not best_report:
        best_threshold = float(grid[0])
        best_report = trading_metrics(
            signals_from_proba(proba, best_threshold),
            future_return,
            target_spec,
            row_index=row_index,
        )
    return best_threshold, best_report


def compare(reports: Mapping[str, Mapping[str, Any]]) -> str:
    """Render a Markdown table of model-vs-baseline results."""
    header = (
        "| model | macro F1 | directional acc | coverage | trades | "
        "net return | Sharpe | max DD |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for name, report in reports.items():
        cls = report["classification"]
        trade = report["trading"]
        rows.append(
            f"| {name} | {cls['macro_f1']:.4f} | {cls['directional_accuracy']:.4f} | "
            f"{cls['coverage']:.4f} | {trade['n_trades']} | {trade['net_return']:+.4f} | "
            f"{trade['sharpe']:.3f} | {trade['max_drawdown']:.4f} |"
        )
    return header + "\n".join(rows)
