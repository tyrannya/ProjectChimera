import os
# import sys # sys is no longer used directly
import argparse
import logging
from typing import Optional # Added import
import pandas as pd
import ccxt
from nn import data_pipeline


def main() -> None: # Added return type hint
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)  # Added logger instance

    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data, enrich it, and create features."
    )
    parser.add_argument("exchange", type=str, help="Name of the exchange (e.g., 'binance').")
    parser.add_argument("symbol", type=str, help="Trading symbol (e.g., 'BTC/USDT').")
    parser.add_argument("start_date", type=str, help="Start date for data download (YYYY-MM-DD).")
    parser.add_argument("tf", type=str, help="Timeframe for data (e.g., '1h', '4h', '1d').")

    args = parser.parse_args()

    exchange_name: str = args.exchange
    symbol_str: str = args.symbol
    start_date_str: str = args.start_date
    timeframe: str = args.tf

    safe_symbol: str = symbol_str.replace("/", "_")
    raw_dir: str = f"data/raw/{exchange_name}/{safe_symbol}_{timeframe}"
    os.makedirs(raw_dir, exist_ok=True)

    # Initialize CCXT exchange instance
    # TODO: Consider adding specific type hint for exchange instance if possible, e.g., ccxt.binance
    # For now, ccxt.Exchange provides a general type.
    try:
        exchange_instance: ccxt.Exchange = getattr(ccxt, exchange_name)()
    except AttributeError:
        logger.error(f"Exchange '{exchange_name}' not found in ccxt library.")
        return # Or sys.exit(1)

    logger.info(
        f"Downloading {exchange_name} {symbol_str} {timeframe} from {start_date_str}"
    )
    df: pd.DataFrame = data_pipeline.download_ohlcv(exchange_name, symbol_str, timeframe, start_date_str)

    if df.empty:
        logger.warning("Downloaded DataFrame is empty. Skipping further processing.")
        return

    df = data_pipeline.add_funding_rate(df, exchange_instance, symbol_str)

    glassnode_api_key: Optional[str] = os.environ.get("GLASSNODE_API_KEY")
    if glassnode_api_key:
        logger.info("GLASSNODE_API_KEY found, enriching with on-chain data.")
        df = data_pipeline.enrich_onchain(df, glassnode_api_key)
    else:
        logger.info("GLASSNODE_API_KEY not found, skipping on-chain data enrichment.")

    data_pipeline.save_delta(raw_dir, df)
    logger.info(f"Raw data saved to {raw_dir}")

    feat_dir: str = f"data/features/{exchange_name}/{safe_symbol}_{timeframe}"
    os.makedirs(feat_dir, exist_ok=True)

    features_df: pd.DataFrame = data_pipeline.make_features(df)
    data_pipeline.save_delta(feat_dir, features_df)
    logger.info(f"Features saved to {feat_dir}")

    # Логирование пропусков
    nan_cols: pd.Series = df.isna().sum() # Original df, not features_df, for missing source data
    missing: pd.Series = nan_cols[nan_cols > 0]
    if not missing.empty:
        logger.warning(
            "Columns with missing values after processing:"
        )  # Replaced print with logger.warning
        for col, count in missing.items():
            logger.warning(f"  {col}: {count} missing values")
    else:
        logger.info("No missing values found in the processed data.")


if __name__ == "__main__":
    main()
