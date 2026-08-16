"""Download and validate historical OHLCV candles.

    python -m tools.backfill --exchange binance --pair BTC/USDT \
        --timeframe 1h --start 2023-01-01

Writes validated candles to ``data/raw/<exchange>/<pair>_<timeframe>.parquet``
plus a ``.meta.json`` sidecar with the validation report. Feature engineering is
a separate step (``tools/build_features.py``) so that a re-run of the feature
code does not require re-downloading the market data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from nn.data_pipeline import DataValidationError, download_ohlcv, validate_ohlcv

logger = logging.getLogger(__name__)


def safe_name(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download historical OHLCV candles.")
    parser.add_argument("--exchange", required=True, help="ccxt exchange id, e.g. binance")
    parser.add_argument("--pair", required=True, help="Trading pair, e.g. BTC/USDT")
    parser.add_argument("--timeframe", required=True, help="Candle timeframe, e.g. 1h")
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD (default: now)")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use the exchange's sandbox/testnet endpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    logger.info(
        "Downloading %s %s %s from %s", args.exchange, args.pair, args.timeframe, args.start
    )
    raw = download_ohlcv(
        args.exchange,
        args.pair,
        args.timeframe,
        args.start,
        end_date=args.end,
        sandbox=args.sandbox,
    )
    if raw.empty:
        logger.error("Exchange returned no candles; nothing written.")
        return 1

    try:
        candles, report = validate_ohlcv(raw, args.timeframe)
    except DataValidationError as exc:
        logger.error("Downloaded data failed validation: %s", exc)
        return 1

    logger.info("Validation report: %s", json.dumps(report.to_dict()))

    out_dir = Path(args.out_dir) / args.exchange
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name(args.pair)}_{args.timeframe}.parquet"
    candles.to_parquet(out_path, index=False)
    out_path.with_suffix(".parquet.meta.json").write_text(
        json.dumps(
            {
                "exchange": args.exchange,
                "pair": args.pair,
                "timeframe": args.timeframe,
                "validation": report.to_dict(),
            },
            indent=2,
        )
    )
    logger.info("Wrote %d candles to %s", len(candles), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
