import os
import pandas as pd
import ccxt
import requests

def download_ohlcv(exchange: str, symbol: str, tf: str, start: str) -> pd.DataFrame:
    ex = getattr(ccxt, exchange)()
    since = int(pd.Timestamp(start).timestamp() * 1000)
    all_ohlcv = []
    limit = 1000
    while True:
        batch = ex.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        if not batch:
            break
        all_ohlcv.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < limit:
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def enrich_onchain(df: pd.DataFrame, glassnode_api_key: str, asset='BTC') -> pd.DataFrame:
    url = f"https://api.glassnode.com/v1/metrics/addresses/active_count"
    params = {
        'a': asset,
        'api_key': glassnode_api_key,
        's': int(df['timestamp'].min() // 1000),
        'u': int(df['timestamp'].max() // 1000),
        'i': '24h'
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        gdf = pd.DataFrame(data)
        gdf['timestamp'] = gdf['t'] * 1000
        gdf = gdf[['timestamp', 'v']]
        gdf.rename(columns={'v': 'active_addresses'}, inplace=True)
        df = df.merge(gdf, on='timestamp', how='left')
    else:
        df['active_addresses'] = None
    return df

def make_features(parquet_path_out: str, df: pd.DataFrame) -> None:
    features = pd.DataFrame()
    features['close'] = df['close']
    features['return_1'] = df['close'].pct_change(1)
    features['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    features['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    features['volume'] = df['volume']
    if 'active_addresses' in df.columns:
        features['active_addresses'] = df['active_addresses']
    features = features.dropna()
    features.to_parquet(parquet_path_out, index=False)
