"""
Data Loader for Section 4: Advanced Time Series Models
Focus on NASDAQ Index and TQQQ ETF
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def load_nasdaq_tqqq_data(start_date='2020-01-01', end_date=None, interval='1d'):
    """
    Load NASDAQ Index (^IXIC) and TQQQ ETF data
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str or None
        End date in 'YYYY-MM-DD' format. If None, uses today
    interval : str
        Data interval ('1d', '1h', '5m', etc.)
    
    Returns:
    --------
    dict
        Dictionary with 'nasdaq' and 'tqqq' DataFrames
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Loading data from {start_date} to {end_date}...")
    
    # Load NASDAQ Index
    nasdaq = yf.download('^IXIC', start=start_date, end=end_date, interval=interval, progress=False)
    nasdaq.columns = [col[0] if isinstance(col, tuple) else col for col in nasdaq.columns]
    
    # Load TQQQ ETF
    tqqq = yf.download('TQQQ', start=start_date, end=end_date, interval=interval, progress=False)
    tqqq.columns = [col[0] if isinstance(col, tuple) else col for col in tqqq.columns]
    
    # Calculate returns
    nasdaq['Returns'] = nasdaq['Close'].pct_change()
    tqqq['Returns'] = tqqq['Close'].pct_change()
    
    # Calculate log returns
    nasdaq['Log_Returns'] = np.log(nasdaq['Close'] / nasdaq['Close'].shift(1))
    tqqq['Log_Returns'] = np.log(tqqq['Close'] / tqqq['Close'].shift(1))
    
    # Remove NaN
    nasdaq = nasdaq.dropna()
    tqqq = tqqq.dropna()
    
    print(f"NASDAQ data: {len(nasdaq)} observations")
    print(f"TQQQ data: {len(tqqq)} observations")
    
    return {
        'nasdaq': nasdaq,
        'tqqq': tqqq
    }


def calculate_realized_volatility(prices, window='5min'):
    """
    Calculate realized volatility from high-frequency data
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    window : str
        Resampling window
    
    Returns:
    --------
    pd.Series
        Realized volatility series
    """
    returns = prices.pct_change().dropna()
    rv = returns.resample(window).apply(lambda x: np.sum(x**2))
    return np.sqrt(rv * 252)  # Annualized


def align_data(data1, data2):
    """
    Align two time series by common dates
    
    Parameters:
    -----------
    data1 : pd.Series or pd.DataFrame
        First time series
    data2 : pd.Series or pd.DataFrame
        Second time series
    
    Returns:
    --------
    tuple
        Aligned (data1, data2)
    """
    common_dates = data1.index.intersection(data2.index)
    return data1.loc[common_dates], data2.loc[common_dates]


def prepare_sequences(data, lookback=60, forecast_horizon=1):
    """
    Prepare sequences for LSTM/ML models
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    lookback : int
        Number of time steps to look back
    forecast_horizon : int
        Number of time steps to forecast
    
    Returns:
    --------
    tuple
        (X, y) arrays for training
    """
    X, y = [], []
    for i in range(lookback, len(data) - forecast_horizon + 1):
        X.append(data[i-lookback:i].values)
        y.append(data[i:i+forecast_horizon].values)
    return np.array(X), np.array(y)

