"""Classification and trading metrics.

Two separate questions, deliberately kept apart:

*Is the classifier any good?* — class distribution, per-class precision/recall/
F1, confusion matrix, calibration error, directional accuracy on the trades it
actually calls, and coverage (how often it calls one at all). Accuracy alone is
excluded from the headline on purpose: with cost-aware labels the HOLD class
dominates, so "accuracy" mostly measures the class prior.

*Would acting on it have made money?* — :func:`trading_metrics` turns a signal
series into non-overlapping trades, charges fees and slippage on both sides,
and reports net return, risk-adjusted statistics, max drawdown, win rate, profit
factor, trade count, exposure, average trade and turnover.

*Would it have beaten doing nothing?* — that is a different question again, and
neither of the above answers it. :func:`economic_references` answers it, with
two policies that are not models and make no predictions: :func:`cash_reference`
(never trade) and :func:`buy_and_hold_reference` (buy once, hold the whole
evaluation window, sell once). Beating a statistical floor is not the same as
making money, and the report keeps the two groups apart so a reader cannot
mistake one for the other.

**Two risk-adjusted statistics, named for what they are.** Neither is called
plain "Sharpe", because that name was doing two jobs and neither honestly:

* ``per_trade_sharpe`` — mean net trade return over its standard deviation
  *across trades*. A per-trade quality ratio. **Not annualised**, and not
  comparable to a published Sharpe.
* ``annualised_sharpe`` — computed from an actual candle-by-candle portfolio
  return series (:func:`portfolio_equity`): equity is unchanged while flat and
  marked to market while a position is open. A strategy that holds cash 96% of
  the time gets the Sharpe of a portfolio that holds cash 96% of the time.

The version this replaced annualised per-trade statistics by
``candles_per_year / horizon`` — the number of trades the strategy *could* have
taken back to back, regardless of how many it actually took. On a signal with 3%
coverage that overstated the figure several-fold, and it grew as the target
horizon shrank, so shortening the horizon "improved" it without trading any more
often. Reports produced before that change are flagged rather than compared.

:func:`trading_metrics` is a *signal evaluation*, not a backtest. It assumes
entry and exit at the close of their respective candles and applies a flat cost.
Freqtrade's backtester remains the authority on execution; this exists so that
model selection can be made against a cost-aware objective instead of accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


#: What ``annualised_sharpe`` means, wherever it appears.
#:
#: Recorded in every report that carries the number, so two rows in one table
#: can be *checked* to be comparable rather than assumed to be. A policy that
#: cannot produce this exact series does not report the metric at all — it
#: reports ``None`` and a reason. Publishing a differently-defined number under
#: a shared name is the failure this string exists to make impossible.
SHARPE_BASIS = (
    "candle-level portfolio returns (equity unchanged while flat, marked to market "
    "while a position is open, both cost sides charged when they are paid), "
    "annualised by sqrt(candles_per_year) over elapsed wall-clock time, with a "
    "zero return for each calendar interval absent from the processed dataset"
)

#: Why a risk-adjusted statistic is absent, when it is.
SHARPE_UNDEFINED_NO_VARIANCE = "undefined: the return series has no variance"
SHARPE_UNDEFINED_NO_TRADES = "undefined: no trades, so the portfolio never moved"
SHARPE_UNDEFINED_NO_MARKET = "unavailable: no market context (close/date series) supplied"
SHARPE_UNDEFINED_ONE_TRADE = "undefined: a single trade has no dispersion to measure"


@dataclass(frozen=True)
class MarketContext:
    """Prices and timestamps for the candles an evaluation spans.

    ``close`` and ``dates`` are indexed by *dataset row*, matching the
    ``row_index`` arrays the evaluators pass around, so nothing has to agree
    about a second coordinate system. They come from the processed dataset the
    run was scored on and are never rebuilt or resampled here.

    ``candles_per_year`` is a property of the timeframe, not of the strategy.
    It is the only annualisation input, which is the point: no term in it
    depends on how often anything trades.
    """

    close: np.ndarray
    dates: np.ndarray
    candles_per_year: float

    @property
    def timeframe_minutes(self) -> float:
        """Minutes per candle — the exact inverse of ``candles_per_year``."""
        return 365 * 24 * 60 / self.candles_per_year

    def elapsed_intervals(self, start_row: int, end_row: int) -> int:
        """Candle intervals of wall-clock time between the two rows' timestamps.

        **Intervals, not timestamps.** Rows ``[0, 150]`` are 151 candles but 150
        elapsed intervals, and 150 is the number of return observations a series
        over that span can have: a return needs a start and an end, so N+1
        observation times yield N returns. Counting timestamps here would hand
        the annualisation one observation more than time actually passed.

        Not ``end_row - start_row`` either, because the processed research
        dataset does not contain every calendar candle: consecutive rows can be
        hours apart. Each absent interval is counted here as time that passed —
        because it did — and the return series is padded with a zero for it.
        Counting dataset rows instead would treat a six-hour discontinuity as a
        single hour and inflate any statistic annualised from the count. With no
        absent intervals the two numbers agree, and a test asserts that.

        An absent interval is **not** necessarily a candle the exchange failed
        to publish. :mod:`nn.data_pipeline` also drops per-segment warm-up rows
        and rows whose features are undefined, then re-segments what is left by
        timestamp. So a discontinuity here means "absent from the processed
        research dataset", which may be a source-data gap or a deliberate
        omission — and for the strategies this measures the distinction does not
        change the arithmetic, because ``build_windows`` refuses any window or
        label that crosses a segment boundary, so no scored position can be held
        across one either way.

        Raises rather than rounding away a disagreement: an elapsed span that is
        not a near-integer multiple of the timeframe means the dates and the
        declared timeframe describe different things.
        """
        minutes = float((self.dates[end_row] - self.dates[start_row]) / np.timedelta64(1, "m"))
        exact = minutes / self.timeframe_minutes
        intervals = round(exact)
        if abs(exact - intervals) > 1e-6:
            raise ValueError(
                f"rows {start_row}-{end_row} span {minutes:g} minutes, which is not a "
                f"whole number of {self.timeframe_minutes:g}-minute candles "
                f"({exact:.6f}). The dates and candles_per_year disagree."
            )
        if intervals < end_row - start_row:
            raise ValueError(
                f"rows {start_row}-{end_row} are {end_row - start_row} apart but span "
                f"only {intervals} candle intervals: the dates are not increasing at "
                "least one timeframe per row."
            )
        return intervals

    def missing_candles(self, start_row: int, end_row: int) -> int:
        """Calendar candle intervals in the span that the dataset does not carry.

        Kept under this name because that is what a reader of a report wants to
        know, but it means *absent from the processed research dataset* rather
        than *never published by the exchange* — the processed dataset also
        drops warm-up and undefined-feature rows, so the two are not the same
        claim. See :meth:`elapsed_intervals`.
        """
        return self.elapsed_intervals(start_row, end_row) - (end_row - start_row)


def portfolio_equity(
    market: MarketContext,
    positions: Sequence[tuple[int, int, float]],
    cost_per_side: float,
    start_row: int,
    end_row: int,
) -> np.ndarray:
    """Equity after every candle in ``[start_row, end_row]``, starting from 1.0.

    ``positions`` is ``(entry_row, exit_row, direction)`` per trade, in time
    order and non-overlapping — exactly what :func:`realised_trades` produces.

    The returned array has one entry per candle in ``[start_row, end_row]``, so
    ``end_row - start_row + 1`` equity *levels* and one fewer interval between
    them. ``equity[0]`` is starting capital at the opening candle: a position
    opening on that very candle has been established but has neither earned nor
    cost anything yet.

    The construction is the existing trade semantics, refined to candle
    resolution rather than replaced::

        E(t) = E0                     at and before the entry candle
        E(t) = E0 · (1 + direction·(close[t] − close[entry])/close[entry] − paid)
        paid = cost_per_side          while the position is open
             = 2·cost_per_side        on the exit candle
        E(t) = E(t−1)                 while flat

    where ``E0`` is equity immediately before the trade opened. The whole
    portfolio backs each trade and is not re-levered inside it. Both cost sides
    fall inside the position's own holding period, so every cost is a return
    over an interval that actually elapsed.

    **Two invariants follow, and both are asserted by tests.** Telescoping the
    marked-to-market term gives ``close[exit]/close[entry] − 1 =
    future_return(entry)``, so a trade multiplies equity by exactly
    ``1 + direction·future_return − cost_threshold`` — its net return from
    :func:`realised_trades`. Flat candles multiply by one. So at every completed
    trade boundary this curve equals ``cumprod(1 + trade_returns)``, the equity
    curve :func:`trading_metrics` already reports ``net_return`` and
    ``max_drawdown`` from. This is one curve at candle resolution, not a second
    return model — which is what makes a Sharpe taken from it comparable to the
    numbers beside it.

    Marking against the position's own fixed entry price is what keeps a SHORT
    exact. Compounding candle returns at ``-1`` would silently re-lever it and
    stop reconciling with the trade return being reported.
    """
    if end_row < start_row:
        raise ValueError(f"empty span [{start_row}, {end_row}]")
    if end_row >= len(market.close):
        raise ValueError(
            f"the close series has {len(market.close)} rows but the evaluation span "
            f"ends at row {end_row}. It must extend to the last trade's exit candle; "
            "truncating it would price an exit at the wrong candle."
        )

    opens = {
        int(entry): (int(exit_row), float(direction))
        for entry, exit_row, direction in positions
    }
    equity = np.empty(end_row - start_row + 1, dtype=np.float64)
    current = 1.0
    live: tuple[int, int, float, float] | None = None

    for offset in range(len(equity)):
        row = start_row + offset
        # Mark and close first, so a trade may open on the candle another exits.
        if live is not None:
            entry, exit_row, direction, opening = live
            current = _marked(
                market.close, entry, exit_row, direction, opening, row, cost_per_side
            )
            if row == exit_row:
                live = None
        if live is None and row in opens:
            exit_row, direction = opens[row]
            live = (row, exit_row, direction, current)
            current = _marked(
                market.close, row, exit_row, direction, current, row, cost_per_side
            )
        equity[offset] = current

    return equity


def _marked(
    close: np.ndarray,
    entry: int,
    exit_row: int,
    direction: float,
    opening: float,
    row: int,
    cost_per_side: float,
) -> float:
    """Equity while one position is open, at ``row``. See :func:`portfolio_equity`.

    On the entry candle itself equity is unchanged. The position is established
    at that close, so nothing has been earned yet — and its entry cost belongs
    to the first interval the position is actually held, not to the interval
    that ended before it existed.
    """
    if row <= entry:
        return opening
    move = direction * (close[row] - close[entry]) / close[entry]
    paid = cost_per_side * (2.0 if row == exit_row else 1.0)
    return opening * (1.0 + move - paid)


def portfolio_returns(equity: np.ndarray) -> np.ndarray:
    """Returns between consecutive points of an equity curve.

    ``N + 1`` equity observations give ``N`` returns, one per elapsed interval,
    matching :meth:`MarketContext.elapsed_intervals`. The first point is
    starting capital at the span's opening candle, and it is a *level*, not a
    return — an earlier version divided it by 1.0 to manufacture an extra
    observation, which handed the annualisation one more interval than had
    elapsed and, for buy-and-hold, booked an instantaneous entry cost as a
    full candle's return.
    """
    if (equity <= 0).any():
        raise ValueError("portfolio equity reached zero or below; returns are undefined")
    return equity[1:] / equity[:-1] - 1.0


def annualised_sharpe(
    returns: np.ndarray,
    elapsed_intervals: int,
    candles_per_year: float,
) -> tuple[float | None, str]:
    """Annualised Sharpe of a candle-level return series, and its basis.

    ``returns`` holds one observation per candle *present in the dataset*;
    ``elapsed_intervals`` is how many candle intervals of wall-clock time the
    span actually covered. The series is zero-padded to that length before the
    mean and variance are taken: a calendar interval absent from the processed
    research dataset is still elapsed time, and the portfolios this is called
    for hold nothing across one — ``build_windows`` refuses any window or label
    crossing a segment boundary, so no scored position can span a discontinuity
    — which makes zero the return they genuinely earned rather than a stand-in
    for an unknown one. (Buy-and-hold is *not* flat across one: it is holding
    the asset through it, and what the price did in between is not recorded. It
    does not call this; see :func:`buy_and_hold_reference`.)

    Padding is applied through the sums rather than by materialising the zeros;
    a test asserts the two agree. Returns ``(None, reason)`` when there is no
    dispersion to divide by, never ``0.0`` — a flat portfolio has an undefined
    Sharpe, and reporting zero would read as a measured result. The same rule
    applies to the model, to the baselines and to CASH alike.
    """
    if elapsed_intervals < len(returns):
        raise ValueError(
            f"cannot pad {len(returns)} observations into {elapsed_intervals} intervals"
        )
    if elapsed_intervals < 2:
        return None, SHARPE_UNDEFINED_NO_VARIANCE

    total = float(returns.sum())
    total_squares = float((returns**2).sum())
    mean = total / elapsed_intervals
    # Sample variance over the padded series: the zeros contribute nothing to
    # either sum, only to the count.
    variance = (total_squares - elapsed_intervals * mean**2) / (elapsed_intervals - 1)
    if variance <= 0:
        return None, SHARPE_UNDEFINED_NO_VARIANCE
    return float(mean / np.sqrt(variance) * np.sqrt(candles_per_year)), SHARPE_BASIS


def max_drawdown_of(equity: np.ndarray) -> float:
    """Deepest peak-to-trough fall of an equity curve, measured from 1.0.

    **Starting capital is part of the curve.** ``equity`` holds the value *after*
    each step, so its first entry is already the result of the first one; taking
    the running peak from there measures the fall from wherever the strategy had
    got to, not from what it started with. A single trade losing 10% then
    reported a drawdown of zero — the curve had one point, so it was its own
    peak — and two consecutive 10% losses reported 10% rather than 19%.

    Prepending the opening 1.0 fixes it: the peak can never begin below starting
    capital, so a strategy that is under water from its first step reports the
    loss it actually took.
    """
    if len(equity) == 0:
        return 0.0
    curve = np.concatenate(([1.0], np.asarray(equity, dtype=np.float64)))
    peak = np.maximum.accumulate(curve)
    return float((1.0 - curve / peak).max())


def _risk_adjusted(
    trade_returns: np.ndarray,
    market: MarketContext | None,
    positions: Sequence[tuple[int, int, float]],
    cost_per_side: float,
    start_row: int,
    end_row: int,
) -> dict[str, Any]:
    """The risk-adjusted half of a trading report: both Sharpes and the curve.

    One implementation, used by the model, by both baselines and by
    buy-and-hold, so a change to the definition cannot land in one and not the
    others.
    """
    n_trades = len(trade_returns)
    if n_trades < 2:
        per_trade: float | None = None
        per_trade_reason = (
            SHARPE_UNDEFINED_NO_TRADES if not n_trades else SHARPE_UNDEFINED_ONE_TRADE
        )
    else:
        spread = float(trade_returns.std(ddof=1))
        per_trade = float(trade_returns.mean() / spread) if spread > 0 else None
        per_trade_reason = "" if per_trade is not None else SHARPE_UNDEFINED_NO_VARIANCE

    report: dict[str, Any] = {
        "per_trade_sharpe": None if per_trade is None else round(per_trade, 4),
        "per_trade_sharpe_reason": per_trade_reason,
    }

    if market is None:
        report.update(
            annualised_sharpe=None,
            annualised_sharpe_reason=SHARPE_UNDEFINED_NO_MARKET,
            sharpe_basis=SHARPE_UNDEFINED_NO_MARKET,
            candle_max_drawdown=None,
            elapsed_intervals=None,
        )
        return report

    equity = portfolio_equity(market, positions, cost_per_side, start_row, end_row)
    intervals = market.elapsed_intervals(start_row, end_row)
    try:
        returns = portfolio_returns(equity)
    except ValueError as exc:
        report.update(
            annualised_sharpe=None,
            annualised_sharpe_reason=f"undefined: {exc}",
            sharpe_basis=SHARPE_BASIS,
            candle_max_drawdown=round(max_drawdown_of(equity), 6),
            elapsed_intervals=intervals,
        )
        return report

    value, basis = annualised_sharpe(returns, intervals, market.candles_per_year)
    report.update(
        annualised_sharpe=None if value is None else round(value, 4),
        annualised_sharpe_reason="" if value is not None else basis,
        sharpe_basis=SHARPE_BASIS if value is not None else basis,
        candle_max_drawdown=round(max_drawdown_of(equity), 6),
        elapsed_intervals=intervals,
    )
    return report


def trading_metrics(
    signals: np.ndarray,
    future_return: np.ndarray,
    target_spec: TargetSpec,
    *,
    market: MarketContext | None = None,
    row_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score a signal series as a sequence of non-overlapping trades.

    The trades themselves come from :func:`realised_trades`; everything here is
    aggregation over them. ``row_index`` keeps trade spacing, exposure and the
    portfolio span in candle-row coordinates when the scored array is
    discontinuous around gaps. Without explicit indices, the historic
    contiguous-array semantics are preserved exactly.

    ``market`` unlocks the candle-level statistics — ``annualised_sharpe`` and
    ``candle_max_drawdown`` — which need prices and timestamps rather than
    trade returns alone. Without it those fields are ``None`` with a stated
    reason; every other number is unaffected, which is why
    :func:`select_threshold` can go on scoring ``net_return`` without one and
    cannot have its choice moved by anything added here.
    """
    horizon = target_spec.horizon
    cost = target_spec.cost_threshold
    n = len(signals)
    rows = _trade_rows(n, row_index)
    entries, directions, returns = realised_trades(
        signals, future_return, target_spec, row_index=rows
    )

    n_trades = len(returns)
    # (entry_row, exit_row, direction) in dataset-row coordinates: `entries`
    # indexes the signal array, `rows` maps that to the candle it happened on.
    positions = [
        (int(rows[position]), int(rows[position]) + horizon, float(direction))
        for position, direction in zip(entries, directions)
    ]
    span_start = int(rows[0]) if n else 0
    span_end = (int(rows[-1]) + horizon) if n else 0
    risk = _risk_adjusted(returns, market, positions, cost / 2.0, span_start, span_end)

    if n_trades == 0:
        return {
            "n_trades": 0,
            "net_return": 0.0,
            "gross_return": 0.0,
            "total_costs": 0.0,
            "avg_trade": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "exposure": 0.0,
            "turnover": 0.0,
            "cost_per_trade": round(cost, 6),
            **risk,
        }

    trade_returns = returns
    equity = np.cumprod(1.0 + trade_returns)
    # Same curve, same starting point: `max_drawdown_of` prepends the opening
    # 1.0, so a strategy that loses from its first trade reports that loss
    # rather than zero. One definition, used by both resolutions.
    max_drawdown = max_drawdown_of(equity)

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
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
        # Drawdown of the same portfolio equity curve as `candle_max_drawdown`,
        # sampled at trade boundaries instead of every candle. The candle-level
        # figure is therefore always the larger of the two: it also sees the
        # drawdown suffered *inside* an open position.
        "max_drawdown": round(max_drawdown, 4),
        "exposure": round(min(1.0, n_trades * horizon / covered_rows), 4),
        "turnover": round(2.0 * n_trades, 1),
        "cost_per_trade": round(cost, 6),
        **risk,
    }


def evaluate(
    proba: np.ndarray,
    y_true: np.ndarray,
    future_return: np.ndarray,
    target_spec: TargetSpec,
    threshold: float,
    *,
    market: MarketContext | None = None,
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
            market=market,
            row_index=row_index,
        ),
    }


# --- economic references ------------------------------------------------------
#
# Not baselines, and deliberately not in `nn.baselines`. That module's contract
# is "emit a probability matrix in the model's shape", which these cannot
# honour: buy-and-hold is not a per-sample decision rule, and encoding it as one
# would make it re-enter and pay the round-trip cost every `horizon` candles —
# turning "hold the asset" into "churn the asset", which is a different policy
# with a different answer.
#
# Beating majority-class and momentum says the model learned more than a
# statistical floor. It says nothing about whether acting on it made money.
# These two answer that, and the reports keep the groups apart.


def cash_reference(target_spec: TargetSpec) -> dict[str, Any]:
    """Never trade. The zero every strategy has to clear before anything else.

    A constant, stated rather than computed: no position at any time means no
    trades, no fees, no slippage, no drawdown and no return. ``target_spec`` is
    taken only to record that its cost model was *not* applied — CASH does not
    pay the strategy's costs, and a test pins that at a non-zero cost threshold.

    Both risk-adjusted statistics are ``None``, not ``0.0``: a flat portfolio
    has no variance, so its Sharpe is undefined, and a zero would read as a
    measured result. Any strategy with no trades reports the same thing, for the
    same reason — the rule does not change with the name on the row.
    """
    return {
        "policy": "cash",
        "description": "no position at any time; no trades, no fees, no slippage",
        "n_trades": 0,
        "gross_return": 0.0,
        "total_costs": 0.0,
        "net_return": 0.0,
        "max_drawdown": 0.0,
        "candle_max_drawdown": 0.0,
        "exposure": 0.0,
        "per_trade_sharpe": None,
        "per_trade_sharpe_reason": SHARPE_UNDEFINED_NO_TRADES,
        "annualised_sharpe": None,
        "annualised_sharpe_reason": SHARPE_UNDEFINED_NO_VARIANCE,
        "sharpe_basis": SHARPE_UNDEFINED_NO_VARIANCE,
        "cost_model_applied": False,
        "cost_per_trade": round(target_spec.cost_threshold, 6),
    }


def buy_and_hold_reference(
    market: MarketContext | None,
    target_spec: TargetSpec,
    row_index: np.ndarray,
) -> dict[str, Any]:
    """Buy at the start of the evaluation window, hold, sell at the end.

    **Alignment.** The window is ``[row_index[0], row_index[-1] + horizon]``:
    the candle the first scored sample could have entered on, through the candle
    that closes the last scored sample's label. That is exactly the stretch in
    which the model was able to open and close trades, so the two are answering
    the same question about the same hours. It is a *continuous market span*
    while the model-facing metrics are over *scored samples* — the two counts
    differ, and both are reported rather than quietly equated.

    **Cost.** One round trip, charged once: one entry, one exit, for the whole
    hold. Not once per horizon — that would be a different policy. ``gross_return``
    is reported beside it so the charge is visible rather than baked in.

    **Risk-adjusted metrics.** Same construction as the strategy's, through the
    same :func:`portfolio_equity`, so ``annualised_sharpe`` is net of the same
    cost model on the same candle grid and carries the same ``sharpe_basis``.

    **Except across a discontinuity in the processed dataset.** No model trade
    ever spans one — ``build_windows`` refuses any window or label that crosses a
    segment boundary — so the strategy's portfolio is provably flat through it
    and padding with zeros invents nothing. A *hold* is not flat through it: it
    owns the asset the whole time, across intervals the dataset does not carry.
    Rather than treat a six-hour discontinuity as one ordinary candle and publish
    the result under a name that promises comparability, the Sharpe is withheld
    with a reason. Return is still exact — a hold's start-to-end return is well
    defined even when its path is not — and the drawdown is flagged as a lower
    bound, since an unobserved low in between can only deepen it.

    Whether such an interval is a genuine exchange gap or a candle deliberately
    absent from the feature dataset is not something this can tell, and it does
    not need to: either way the price path over it was not recorded.
    """
    if market is None:
        return {
            "policy": "buy_and_hold",
            "available": False,
            "reason": SHARPE_UNDEFINED_NO_MARKET,
        }

    horizon = target_spec.horizon
    cost = target_spec.cost_threshold
    entry_row = int(row_index[0])
    exit_row = int(row_index[-1]) + horizon
    gross = float(market.close[exit_row] / market.close[entry_row] - 1.0)

    equity = portfolio_equity(
        market, [(entry_row, exit_row, 1.0)], cost / 2.0, entry_row, exit_row
    )
    missing = market.missing_candles(entry_row, exit_row)
    report: dict[str, Any] = {
        "policy": "buy_and_hold",
        "available": True,
        "description": (
            "buy at the close of the first scored candle, hold, sell at the close of "
            "the candle that closes the last scored sample's label horizon"
        ),
        "window": "continuous market span (not the scored-sample set)",
        "entry_row": entry_row,
        "exit_row": exit_row,
        "entry_timestamp": str(market.dates[entry_row]),
        "exit_timestamp": str(market.dates[exit_row]),
        "entry_price": round(float(market.close[entry_row]), 8),
        "exit_price": round(float(market.close[exit_row]), 8),
        "candles_held": exit_row - entry_row,
        "elapsed_intervals": market.elapsed_intervals(entry_row, exit_row),
        "missing_candles": missing,
        "spans_market_data_gap": bool(missing),
        "n_trades": 1,
        "gross_return": round(gross, 6),
        "total_costs": round(cost, 6),
        "net_return": round(gross - cost, 6),
        "cost_model": "one round trip charged once over the whole hold",
        "candle_max_drawdown": round(max_drawdown_of(equity), 6),
        "drawdown_is_lower_bound": bool(missing),
        "exposure": 1.0,
        "per_trade_sharpe": None,
        "per_trade_sharpe_reason": SHARPE_UNDEFINED_ONE_TRADE,
    }

    if missing:
        reason = (
            f"unavailable: the hold spans {missing} calendar interval(s) absent from "
            "the processed dataset. What price did in between is not recorded there, "
            "so a candle-level series comparable to the strategy's cannot be built "
            "without inventing a path"
        )
        report.update(
            annualised_sharpe=None, annualised_sharpe_reason=reason, sharpe_basis=reason
        )
        return report

    try:
        returns = portfolio_returns(equity)
    except ValueError as exc:
        report.update(
            annualised_sharpe=None,
            annualised_sharpe_reason=f"undefined: {exc}",
            sharpe_basis=SHARPE_BASIS,
        )
        return report

    value, basis = annualised_sharpe(
        returns,
        market.elapsed_intervals(entry_row, exit_row),
        market.candles_per_year,
    )
    report.update(
        annualised_sharpe=None if value is None else round(value, 4),
        annualised_sharpe_reason="" if value is not None else basis,
        sharpe_basis=SHARPE_BASIS if value is not None else basis,
    )
    return report


def economic_references(
    market: MarketContext | None,
    target_spec: TargetSpec,
    row_index: np.ndarray,
) -> dict[str, Any]:
    """CASH and buy-and-hold over one evaluation window.

    The economic question — did this make money, and did it beat doing nothing
    or simply owning the asset — kept in one place so no report module answers
    it a second way.
    """
    return {
        "note": (
            "Economic reference policies, not models and not statistical baselines. "
            "A strategy that beats majority-class and momentum but loses to CASH lost "
            "money."
        ),
        "cash": cash_reference(target_spec),
        "buy_and_hold": buy_and_hold_reference(market, target_spec, row_index),
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


def number(value: Any, spec: str = ".4f") -> str:
    """Format a metric, or ``n/a`` when it is undefined.

    Undefined is a real state here — a portfolio that never moved has no Sharpe
    — and it prints as ``n/a`` rather than ``0.000`` so a reader is not shown a
    measurement that was never made.
    """
    return "n/a" if value is None else format(value, spec)


def compare(reports: Mapping[str, Mapping[str, Any]]) -> str:
    """Render a Markdown table of model-vs-baseline results.

    Statistical/rule floors only. What these say is whether the model learned
    more than a trivial rule — never whether it made money; see
    :func:`economic_references` for that.
    """
    header = (
        "| model | macro F1 | directional acc | coverage | trades | "
        "net return | ann. Sharpe | per-trade Sharpe | exposure | max DD |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for name, report in reports.items():
        cls = report["classification"]
        trade = report["trading"]
        rows.append(
            f"| {name} | {cls['macro_f1']:.4f} | {cls['directional_accuracy']:.4f} | "
            f"{cls['coverage']:.4f} | {trade['n_trades']} | {trade['net_return']:+.4f} | "
            f"{number(trade['annualised_sharpe'], '.3f')} | "
            f"{number(trade['per_trade_sharpe'], '.3f')} | "
            f"{trade['exposure']:.4f} | {trade['max_drawdown']:.4f} |"
        )
    return header + "\n".join(rows)
