import os
import pandas as pd
import ccxt
import requests
from deltalake import write_deltalake
import ta


def download_ohlcv(exchange: str, symbol: str, tf: str, start: str, sandbox: bool = False) -> pd.DataFrame:
    ex = getattr(ccxt, exchange)()
    if sandbox and hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(True)
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
    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    tf_ms = ex.parse_timeframe(tf) * 1000
    gaps = df["timestamp"].diff() > tf_ms
    if gaps.any():
        print("Warning: gaps found in OHLCV data")
    if (df["volume"] == 0).any():
        print("Warning: zero-volume bars detected")
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

def add_funding_rate(df: pd.DataFrame, ex, symbol: str) -> pd.DataFrame:
    fetch = getattr(ex, 'fetch_funding_rate_history', None)
    if not fetch:
        df['funding_rate'] = None
        return df
    try:
        fr = fetch(symbol, since=df['timestamp'].min(), limit=len(df))
        fr_df = pd.DataFrame(fr)
        if 'fundingRate' in fr_df.columns:
            fr_df = fr_df[['timestamp', 'fundingRate']]
            fr_df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
            df = df.merge(fr_df, on='timestamp', how='left')
        else:
            df['funding_rate'] = None
    except Exception:
        df['funding_rate'] = None
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features['close'] = df['close']
    features['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    features['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    features['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.macd_diff(df['close'])
    features['macd'] = macd
    features['return_1'] = df['close'].pct_change(1)
    features['volume'] = df['volume']
    if 'active_addresses' in df.columns:
        features['active_addresses'] = df['active_addresses']
    if 'funding_rate' in df.columns:
        features['funding_rate'] = df['funding_rate']
    features = features.dropna()
    return features

def save_delta(path: str, df: pd.DataFrame) -> None:
    write_deltalake(path, df, mode="append", overwrite_schema=True)
