import os # Keep os, it might be used implicitly or in other parts not shown
from typing import List, Dict, Any, Optional # Added imports
import pandas as pd
import ccxt
import requests
from deltalake import write_deltalake
import ta


def download_ohlcv(
    exchange_name: str, symbol: str, tf: str, start_date: str, sandbox: bool = False
) -> pd.DataFrame:
    ex: ccxt.Exchange = getattr(ccxt, exchange_name)()
    if sandbox and hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(True) # type: ignore # set_sandbox_mode might not be on all exchanges

    since: int = int(pd.Timestamp(start_date).timestamp() * 1000)
    all_ohlcv: List[List[Any]] = [] # To store batches of OHLCV data
    limit: int = 1000

    while True:
        # Type of batch can be List[List[Union[int, float]]]
        # but List[Any] is simpler for now as ccxt types can be complex
        batch: List[Any] = ex.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        if not batch:
            break
        all_ohlcv.extend(batch)
        since = batch[-1][0] + 1 # Assuming timestamp is the first element
        if len(batch) < limit:
            break

    if not all_ohlcv:
        # Return empty DataFrame with expected columns if no data
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"])

    df: pd.DataFrame = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Check for gaps, ensure ex.parse_timeframe returns an int
    tf_ms_obj: Optional[int] = ex.parse_timeframe(tf)
    if tf_ms_obj is None:
        # Handle case where timeframe parsing fails, though unlikely for valid tfs
        # Or raise an error: raise ValueError(f"Invalid timeframe: {tf}")
        print(f"Warning: Could not parse timeframe {tf}, gap check skipped.")
    else:
        tf_ms: int = tf_ms_obj * 1000
        gaps: pd.Series = df["timestamp"].diff() > tf_ms
        if gaps.any():
            print("Warning: gaps found in OHLCV data") # Consider logging

    if (df["volume"] == 0).any():
        print("Warning: zero-volume bars detected") # Consider logging
    return df


def enrich_onchain(
    df: pd.DataFrame, glassnode_api_key: str, asset: str = "BTC"
) -> pd.DataFrame:
    if df.empty: # Handle empty input DataFrame
        df["active_addresses"] = None
        return df

    url: str = f"https://api.glassnode.com/v1/metrics/addresses/active_count"
    params: Dict[str, Any] = {
        "a": asset,
        "api_key": glassnode_api_key,
        "s": int(df["timestamp"].min() // 1000), # Ensure df['timestamp'] is not empty
        "u": int(df["timestamp"].max() // 1000), # Ensure df['timestamp'] is not empty
        "i": "24h",
    }

    try:
        resp: requests.Response = requests.get(url, params=params, timeout=10) # Added timeout
        if resp.status_code == 200:
            data: List[Dict[str, Any]] = resp.json()
            if not data: # Handle empty response from Glassnode
                df["active_addresses"] = None
                return df
            gdf: pd.DataFrame = pd.DataFrame(data)
            gdf["timestamp"] = gdf["t"] * 1000
            gdf = gdf[["timestamp", "v"]]
            gdf.rename(columns={"v": "active_addresses"}, inplace=True)
            df = pd.merge(df, gdf, on="timestamp", how="left") # Use pd.merge for clarity
        else:
            print(f"Warning: Glassnode API request failed with status {resp.status_code}: {resp.text}") # Consider logging
            df["active_addresses"] = None
    except requests.exceptions.RequestException as e:
        print(f"Warning: Glassnode API request failed: {e}") # Consider logging
        df["active_addresses"] = None
    return df


def add_funding_rate(df: pd.DataFrame, ex: ccxt.Exchange, symbol: str) -> pd.DataFrame:
    if df.empty: # Handle empty input DataFrame
        df["funding_rate"] = None
        return df

    fetch_funding_history_method: Optional[callable] = getattr(ex, "fetch_funding_rate_history", None)

    if not fetch_funding_history_method:
        df["funding_rate"] = None
        return df

    try:
        # Assuming timestamp is present and not empty
        funding_rates_data: List[Dict[str, Any]] = fetch_funding_history_method(
            symbol, since=int(df["timestamp"].min()), limit=len(df)
        )
        if not funding_rates_data: # Handle empty response
            df["funding_rate"] = None
            return df

        fr_df: pd.DataFrame = pd.DataFrame(funding_rates_data)
        if "fundingRate" in fr_df.columns:
            fr_df = fr_df[["timestamp", "fundingRate"]]
            fr_df.rename(columns={"fundingRate": "funding_rate"}, inplace=True)
            df = pd.merge(df, fr_df, on="timestamp", how="left") # Use pd.merge
        else:
            df["funding_rate"] = None
    except Exception as e: # Catch specific exceptions if possible
        print(f"Warning: Could not fetch or process funding rates: {e}") # Consider logging
        df["funding_rate"] = None
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns or df['close'].isnull().all():
        # Return empty DataFrame with expected feature columns if input is unsuitable
        # Or define expected columns explicitly
        return pd.DataFrame()

    features: pd.DataFrame = pd.DataFrame(index=df.index)
    features["close"] = df["close"]

    # Ensure input series for TA functions are not all NaN, or handle potential errors
    if not df["close"].isnull().all():
        features["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
        features["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
        features["rsi"] = ta.momentum.rsi(df["close"], window=14)
        macd_series: Optional[pd.Series] = ta.trend.macd_diff(df["close"])
        features["macd"] = macd_series if macd_series is not None else pd.NA
    else:
        features["sma_20"] = pd.NA
        features["ema_50"] = pd.NA
        features["rsi"] = pd.NA
        features["macd"] = pd.NA

    features["return_1"] = df["close"].pct_change(1)
    features["volume"] = df["volume"]

    if "active_addresses" in df.columns:
        features["active_addresses"] = df["active_addresses"]
    if "funding_rate" in df.columns:
        features["funding_rate"] = df["funding_rate"]

    features = features.dropna() # Consider how NaNs from pct_change(1) or TA on short series are handled
    return features


def save_delta(path: str, df: pd.DataFrame) -> None:
    write_deltalake(path, df, mode="append", overwrite_schema=True)
