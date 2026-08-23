"""Generate synthetic OHLCV candles for tests, smoke runs and CI.

    python -m tools.make_sample_data --rows 3000 --out data/raw/sample/SYNTH_USDT_1h.parquet

The series is a random walk with a mild, mean-reverting drift component, so the
labels are not pure noise and a short training run produces something other
than a degenerate all-HOLD model. It is **not** market data and nothing
measured on it says anything about real performance — it exists so the pipeline
can be exercised end to end without a network or an exchange account.

By default the series is positioned so it straddles the immutable sealed-test
anchor of ``synth-usdt-1h-gen1`` — the committed research contract this series
belongs to — because "end to end" includes the train/validation/sealed
partition.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nn.research_contract import SYNTHETIC_CONTRACT_ID, load_contract

logger = logging.getLogger(__name__)

#: The instant the synthetic research contract seals at.
#:
#: Read from the committed contract rather than restated here, so a synthetic
#: series straddles exactly the boundary the entrypoints resolve when they are
#: pointed at it.
SEALED_TEST_START_UTC = load_contract(SYNTHETIC_CONTRACT_ID).sealed_test_start

#: Share of a default synthetic series that falls before the sealed anchor.
#:
#: 0.85 mirrors the 70/15 research weights, so a default series divides into a
#: research region and a sealed block in roughly the proportions a real dataset
#: does.
_RESEARCH_SHARE = 0.85


def generate_candles(
    rows: int = 3000,
    *,
    seed: int = 7,
    start: str | pd.Timestamp | None = None,
    timeframe_minutes: int = 60,
    start_price: float = 30000.0,
    volatility: float = 0.004,
) -> pd.DataFrame:
    """Deterministic synthetic candles that satisfy the OHLC invariants.

    ``start`` defaults to a position that puts the synthetic research contract's
    sealed instant exactly on row ``int(rows * 0.85)``, so the series
    straddles the immutable sealed-test anchor. This generator exists to exercise
    the research pipeline end to end, and a series that lies entirely on one side
    of the seal cannot be partitioned into a research region and a sealed block
    at all — every entrypoint would (correctly) refuse it. Pass an explicit
    ``start`` when the dates themselves are what a caller is testing.
    """
    rng = np.random.default_rng(seed)

    if start is None:
        offset = pd.Timedelta(minutes=timeframe_minutes * int(rows * _RESEARCH_SHARE))
        start = SEALED_TEST_START_UTC - offset

    # AR(1) drift gives the series short-lived trends; without it the labels
    # are unpredictable by construction and a smoke test proves nothing.
    drift = np.zeros(rows)
    for i in range(1, rows):
        drift[i] = 0.92 * drift[i - 1] + rng.normal(0.0, volatility * 0.35)

    returns = drift + rng.normal(0.0, volatility, rows)
    close = start_price * np.exp(np.cumsum(returns))

    open_ = np.empty(rows)
    open_[0] = start_price
    open_[1:] = close[:-1]

    span = np.abs(rng.normal(0.0, volatility, rows)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    low = np.clip(low, 1e-8, None)

    volume = rng.lognormal(mean=6.0, sigma=0.4, size=rows) * (1.0 + 8.0 * np.abs(returns))

    dates = pd.date_range(
        start=(
            pd.Timestamp(start).tz_localize("UTC")
            if pd.Timestamp(start).tzinfo is None
            else pd.Timestamp(start).tz_convert("UTC")
        ),
        periods=rows,
        freq=f"{timeframe_minutes}min",
    )
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def trades_reproducing(
    candles: pd.DataFrame,
    *,
    seed: int = 11,
    trades_per_hour: int = 24,
    skip_hours: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic trades that aggregate back to ``candles``, hour for hour.

    :func:`generate_trades` draws its own price path, so a fixture built from
    both generators holds two recordings of two different markets. That was
    invisible while nothing compared them; it stopped being invisible when
    ``tools.verify_trade_snapshot`` gained ``cross_source_values``, and a
    fixture that cannot pass a check cannot exercise it.

    Each hour emits at least four trades: the first at the candle's ``open``,
    the second at its ``high``, the third at its ``low``, the last at its
    ``close``, and any remainder drawn inside ``[low, high]``. Quantities are
    log-uniform weights rescaled to sum to the candle's ``volume``. The
    aggregator's ``first_price``, ``high_price``, ``low_price``, ``last_price``
    and ``qty`` therefore reproduce the candle exactly, up to the float error of
    one rescaled sum.

    ``skip_hours`` omits whole hours by position, as in :func:`generate_trades`:
    an hour with no trade has no aggregate row, which is what starts a segment.
    """
    if trades_per_hour < 4:
        raise ValueError("an hour needs four trades to place its open, high, low and close")
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pd.to_datetime(candles["date"], utc=True))
    if any(int(stamp.value) % 3_600_000_000_000 for stamp in dates):
        raise ValueError("synthetic trades are placed inside whole UTC hours")

    skipped = set(skip_hours)
    stamps: list[np.ndarray] = []
    prices: list[np.ndarray] = []
    quantities: list[np.ndarray] = []
    for position in range(len(dates)):
        if position in skipped:
            continue
        row = candles.iloc[position]
        count = int(rng.integers(4, trades_per_hour * 2 + 1))
        hour = int(dates[position].value)
        stamps.append(hour + np.linspace(0, 3_500_000_000_000, count, dtype=np.int64))

        price = np.empty(count, dtype=np.float64)
        price[0] = float(row["open"])
        price[1] = float(row["high"])
        price[2] = float(row["low"])
        price[-1] = float(row["close"])
        if count > 4:
            price[3:-1] = rng.uniform(float(row["low"]), float(row["high"]), size=count - 4)
        prices.append(price)

        weights = 10.0 ** rng.uniform(-3.0, 1.0, size=count)
        quantities.append(float(row["volume"]) * weights / weights.sum())

    ts = np.concatenate(stamps).astype(np.int64)
    price = np.concatenate(prices)
    qty = np.concatenate(quantities)
    buyer_maker = rng.random(len(ts)) < 0.5
    agg_id = np.arange(1, len(ts) + 1, dtype=np.int64)
    return ts, price, qty, buyer_maker, agg_id


def generate_trades(
    hours: int = 240,
    *,
    seed: int = 11,
    start: str | pd.Timestamp = "2020-01-01T00:00:00+00:00",
    trades_per_hour: int = 24,
    start_price: float = 30000.0,
    volatility: float = 0.0008,
    skip_hours: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic trades: ``(ts_ns, price, qty, buyer_maker, agg_id)``.

    The counterpart of :func:`generate_candles`, for the trade-level source P3
    reads. It exists for exactly one reason: **the aggregator, the verifier and
    every leakage guard must be exercised against inputs that can be corrupted to
    order**, and the public archive cannot be made to serve a duplicated trade id
    or a sealed timestamp on request.

    It is not market data, nothing measured on it is a research result, and no
    P3 evidence in this repository was produced from it. The arrays satisfy the
    invariants :func:`nn.trade_aggregates.check_table` enforces — strictly
    increasing ids, non-decreasing timestamps, positive prices and quantities,
    at least one trade per emitted hour — because a fixture that could not pass
    the checks would test the checks and nothing else.

    ``skip_hours`` omits whole hours, which is how a gap is built: an hour with
    no trade has no aggregate row, and that is what starts a new segment.
    """
    if hours < 1 or trades_per_hour < 1:
        raise ValueError("need at least one hour and one trade per hour")
    rng = np.random.default_rng(seed)
    origin = int(pd.Timestamp(start).tz_convert("UTC").value)
    if origin % 3_600_000_000_000:
        raise ValueError("synthetic trades start on an hour boundary")

    # Hourly activity *varies*, and deliberately. A generator that put the same
    # number of trades in every hour would make every trailing ratio in
    # `microstructure_v1` identically 1.0, and a test asserting that a full
    # window differs from its default would pass without the window ever having
    # been computed.
    counts = rng.integers(max(trades_per_hour // 2, 1), trades_per_hour * 2 + 1, size=hours)
    # Offsets spread across the hour but never at its very end, so a boundary
    # test can place its own trade at the last nanosecond and mean it.
    ts = np.concatenate(
        [
            origin
            + h * 3_600_000_000_000
            + np.linspace(0, 3_500_000_000_000, int(c), dtype=np.int64)
            for h, c in enumerate(counts)
        ]
    ).astype(np.int64)
    n = int(len(ts))
    steps = rng.normal(0.0, volatility, size=n)
    price = start_price * np.exp(np.cumsum(steps))
    # Log-uniform sizes over four decades, which is the shape the half-decade
    # histogram of nn.trade_aggregates is built for.
    qty = 10.0 ** rng.uniform(-3.0, 1.0, size=n)
    buyer_maker = rng.random(n) < 0.5
    agg_id = np.arange(1, n + 1, dtype=np.int64)

    # `skip_hours` deletes rows from a series that was drawn for the whole span,
    # so removing an hour leaves every other hour byte-identical. A generator
    # that redrew after a skip would make the gap tests compare two unrelated
    # series and pass or fail for the wrong reason.
    if skip_hours:
        keep = ~np.isin((ts - origin) // 3_600_000_000_000, np.asarray(skip_hours))
        ts, price, qty, buyer_maker, agg_id = (
            ts[keep],
            price[keep],
            qty[keep],
            buyer_maker[keep],
            agg_id[keep],
        )
    return ts, price, qty, buyer_maker, agg_id


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic OHLCV candles.")
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeframe-minutes", type=int, default=60)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    candles = generate_candles(
        args.rows, seed=args.seed, timeframe_minutes=args.timeframe_minutes
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    candles.to_parquet(out, index=False)
    logger.info("Wrote %d synthetic candles to %s", len(candles), out)
    logger.warning("Synthetic data: results from it mean nothing about real markets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
