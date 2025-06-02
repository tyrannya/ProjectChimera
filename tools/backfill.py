import os
import sys
import pandas as pd
from nn import data_pipeline

def main():
    if len(sys.argv) != 5:
        print("Usage: python backfill.py <exchange> <symbol> <start_date> <tf>")
        sys.exit(1)
    exchange, symbol, start, tf = sys.argv[1:]
    safe_symbol = symbol.replace("/", "_")
    raw_dir = f"data/raw/{exchange}"
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = f"{raw_dir}/{safe_symbol}_{tf}_{start}.parquet"
    if os.path.exists(raw_path):
        print(f"Raw file exists: {raw_path}")
        df = pd.read_parquet(raw_path)
    else:
        print(f"Downloading {exchange} {symbol} {tf} from {start}")
        df = data_pipeline.download_ohlcv(exchange, symbol, tf, start)
        if df.empty:
            print("No data fetched.")
            return
        df.to_parquet(raw_path, index=False)
    glassnode_key = os.environ.get("GLASSNODE_API_KEY")
    if glassnode_key:
        df = data_pipeline.enrich_onchain(df, glassnode_key)
    feat_dir = f"data/features/{exchange}"
    os.makedirs(feat_dir, exist_ok=True)
    feat_path = f"{feat_dir}/{safe_symbol}_{tf}_{start}_features.parquet"
    if os.path.exists(feat_path):
        print(f"Features file exists: {feat_path}")
    else:
        print(f"Creating features at {feat_path}")
        data_pipeline.make_features(feat_path, df)
    # Логирование пропусков
    nan_cols = df.isna().sum()
    missing = nan_cols[nan_cols > 0]
    if not missing.empty:
        print("Columns with missing values:")
        print(missing)

if __name__ == "__main__":
    main()
