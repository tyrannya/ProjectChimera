import os
import sys
import pandas as pd
import ccxt
from nn import data_pipeline

def main():
    if len(sys.argv) != 5:
        print("Usage: python backfill.py <exchange> <symbol> <start_date> <tf>")
        sys.exit(1)
    exchange, symbol, start, tf = sys.argv[1:]
    safe_symbol = symbol.replace("/", "_")
    raw_dir = f"data/raw/{exchange}/{safe_symbol}_{tf}"
    os.makedirs(raw_dir, exist_ok=True)
    ex = getattr(ccxt, exchange)()
    print(f"Downloading {exchange} {symbol} {tf} from {start}")
    df = data_pipeline.download_ohlcv(exchange, symbol, tf, start)
    df = data_pipeline.add_funding_rate(df, ex, symbol)
    glassnode_key = os.environ.get("GLASSNODE_API_KEY")
    if glassnode_key:
        df = data_pipeline.enrich_onchain(df, glassnode_key)
    data_pipeline.save_delta(raw_dir, df)
    feat_dir = f"data/features/{exchange}/{safe_symbol}_{tf}"
    os.makedirs(feat_dir, exist_ok=True)
    feats = data_pipeline.make_features(df)
    data_pipeline.save_delta(feat_dir, feats)
    # Логирование пропусков
    nan_cols = df.isna().sum()
    missing = nan_cols[nan_cols > 0]
    if not missing.empty:
        print("Columns with missing values:")
        print(missing)

if __name__ == "__main__":
    main()
