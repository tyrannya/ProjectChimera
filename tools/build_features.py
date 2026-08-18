"""Turn validated candles into a training dataset.

    python -m tools.build_features \
        --candles data/raw/binance/BTC_USDT_1h.parquet \
        --out data/datasets/binance_BTC_USDT_1h.parquet

Features are causal and the label is cost-aware; both are described in
:mod:`chimera.features` and :class:`chimera.contracts.TargetSpec`. The horizon
and the cost assumptions are CLI flags because they are modelling decisions,
and they are recorded in the dataset's metadata so a model can never be trained
against one definition and served against another.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec
from nn.data_pipeline import DataValidationError, build_dataset, save_dataset, validate_ohlcv

logger = logging.getLogger(__name__)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a training dataset from candles.")
    parser.add_argument("--candles", required=True, help="Parquet file from tools.backfill")
    parser.add_argument("--out", required=True, help="Output Parquet path")
    parser.add_argument("--exchange", default="")
    parser.add_argument("--pair", default="")
    parser.add_argument("--timeframe", default="")
    parser.add_argument(
        "--horizon",
        type=int,
        default=TargetSpec.horizon,
        help="Candles ahead the label looks.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=TargetSpec.fee_rate,
        help="Taker fee per side, as a fraction.",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=TargetSpec.slippage_rate,
        help="Conservative slippage allowance per side, as a fraction.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)

    candles = pd.read_parquet(args.candles)

    # Read provenance from the sidecar when the caller did not pass it.
    sidecar = Path(str(args.candles) + ".meta.json")
    if not sidecar.exists():
        sidecar = Path(args.candles).with_suffix(".parquet.meta.json")
    provenance = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    exchange = args.exchange or provenance.get("exchange", "")
    pair = args.pair or provenance.get("pair", "")
    timeframe = args.timeframe or provenance.get("timeframe", "")

    try:
        candles, report = validate_ohlcv(candles, timeframe or None)
    except DataValidationError as exc:
        logger.error("Candles failed validation: %s", exc)
        return 1

    target_spec = TargetSpec(
        horizon=args.horizon, fee_rate=args.fee_rate, slippage_rate=args.slippage_rate
    )
    logger.info(
        "Labelling with horizon=%d and round-trip cost threshold %.5f",
        target_spec.horizon,
        target_spec.cost_threshold,
    )

    try:
        frame, metadata = build_dataset(
            candles,
            FeatureSpec(),
            target_spec,
            exchange=exchange,
            pair=pair,
            timeframe=timeframe,
            validation=report.to_dict(),
        )
    except DataValidationError as exc:
        logger.error("Could not build a dataset: %s", exc)
        return 1

    save_dataset(args.out, frame, metadata)
    logger.info("Class balance: %s", json.dumps(metadata.class_balance))
    total = sum(metadata.class_balance.values())
    if total:
        tradeable = metadata.class_balance["LONG"] + metadata.class_balance["SHORT"]
        logger.info(
            "%.1f%% of rows are tradeable (non-HOLD) at this cost threshold",
            100.0 * tradeable / total,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
