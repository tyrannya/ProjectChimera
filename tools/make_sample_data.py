"""Generate synthetic OHLCV candles for tests, smoke runs and CI.

    python -m tools.make_sample_data --rows 3000 --out data/raw/sample/SYNTH_USDT_1h.parquet

The series is a random walk with a mild, mean-reverting drift component, so the
labels are not pure noise and a short training run produces something other
than a degenerate all-HOLD model. It is **not** market data and nothing
measured on it says anything about real performance — it exists so the pipeline
can be exercised end to end without a network or an exchange account.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_candles(
    rows: int = 3000,
    *,
    seed: int = 7,
    start: str = "2023-01-01",
    timeframe_minutes: int = 60,
    start_price: float = 30000.0,
    volatility: float = 0.004,
) -> pd.DataFrame:
    """Deterministic synthetic candles that satisfy the OHLC invariants."""
    rng = np.random.default_rng(seed)

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
        start=pd.Timestamp(start, tz="UTC"), periods=rows, freq=f"{timeframe_minutes}min"
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
