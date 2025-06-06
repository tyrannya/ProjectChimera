import pytest
import pandas as pd
import numpy as np
from nn.data_pipeline import make_features

@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """
    Generates a sample OHLCV DataFrame for testing.
    Includes enough data to generate meaningful indicator values.
    """
    np.random.seed(42) # for reproducibility
    num_rows = 100
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=num_rows, freq='1h').astype(np.int64) // 10**9,
        'open': np.random.uniform(100, 200, num_rows),
        'high': np.random.uniform(200, 300, num_rows),
        'low': np.random.uniform(50, 100, num_rows),
        'close': np.random.uniform(100, 200, num_rows),
        'volume': np.random.uniform(1000, 5000, num_rows),
        'active_addresses': np.random.randint(1000, 2000, num_rows), # Optional column
        'funding_rate': np.random.uniform(-0.001, 0.001, num_rows) # Optional column
    }
    df = pd.DataFrame(data)
    # Ensure high is always >= open and close, and low is always <= open and close
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    return df

def test_make_features_creates_columns(sample_ohlcv_df: pd.DataFrame):
    """
    Tests that make_features creates the expected columns.
    """
    input_df = sample_ohlcv_df.copy()
    output_df = make_features(input_df)

    assert isinstance(output_df, pd.DataFrame)

    expected_columns = [
        'close',
        'sma_20',
        'ema_50',
        'rsi',
        'macd',
        'return_1',
        'volume',
        'active_addresses', # Present in fixture
        'funding_rate'    # Present in fixture
    ]

    for col in expected_columns:
        assert col in output_df.columns, f"Expected column '{col}' not found in output."

    # Check number of rows: make_features drops rows with NaNs
    # The number of NaNs depends on the largest lookback period of indicators
    # e.g., ema_50 needs 49 prior data points, return_1 needs 1.
    # So, after dropna(), we expect num_rows - (max_lookback - 1) if all indicators start at same time.
    # MACD uses EMA(12) and EMA(26), so effective lookback is around 25 for MACD.
    # EMA_50 has the largest explicit lookback.
    # The dropna() will remove rows where any of these are NaN.
    # For EMA_50, the first 49 values will be NaN.
    # For return_1, the first value is NaN.
    # Thus, at least 49 rows will be dropped if input is >= 50.
    if len(input_df) >= 50:
         assert len(output_df) <= len(input_df) - 49, \
             "Number of rows in output_df is not consistent with dropna behavior for EMA_50."
    elif not input_df.empty:
         assert len(output_df) < len(input_df) # Should drop at least one row for return_1 if not empty
    else: # if input_df is empty
        assert len(output_df) == 0


def test_make_features_handles_empty_input():
    """
    Tests make_features with an empty DataFrame.
    """
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']) # Ensure correct columns for initial check
    output_df = make_features(empty_df)
    assert isinstance(output_df, pd.DataFrame)
    assert output_df.empty, "Expected an empty DataFrame for empty input."

def test_make_features_handles_empty_input_no_columns():
    """
    Tests make_features with an empty DataFrame that has no columns.
    """
    empty_df_no_cols = pd.DataFrame()
    output_df = make_features(empty_df_no_cols)
    assert isinstance(output_df, pd.DataFrame)
    assert output_df.empty, "Expected an empty DataFrame for empty input with no columns."


def test_make_features_handles_insufficient_data(sample_ohlcv_df: pd.DataFrame):
    """
    Tests make_features with insufficient data for indicators.
    """
    insufficient_df = sample_ohlcv_df.head(5).copy() # Only 5 rows
    output_df = make_features(insufficient_df)
    assert isinstance(output_df, pd.DataFrame)
    # make_features calls dropna(). With only 5 rows, most indicators will be NaN,
    # and 'return_1' will be NaN for the first row.
    # Thus, all rows should be dropped.
    assert output_df.empty, "Expected an empty DataFrame due to dropna() with insufficient data."

def test_make_features_without_optional_columns(sample_ohlcv_df: pd.DataFrame):
    """
    Tests that make_features works correctly when optional columns ('active_addresses', 'funding_rate') are not present.
    """
    input_df = sample_ohlcv_df.drop(columns=['active_addresses', 'funding_rate']).copy()
    output_df = make_features(input_df)

    assert isinstance(output_df, pd.DataFrame)
    assert 'active_addresses' not in output_df.columns
    assert 'funding_rate' not in output_df.columns

    expected_base_columns = [
        'close', 'sma_20', 'ema_50', 'rsi', 'macd', 'return_1', 'volume'
    ]
    for col in expected_base_columns:
        assert col in output_df.columns, f"Expected base column '{col}' not found."

    if len(input_df) >= 50:
         assert len(output_df) <= len(input_df) - 49
    elif not input_df.empty:
         assert len(output_df) < len(input_df)
    else:
        assert len(output_df) == 0
