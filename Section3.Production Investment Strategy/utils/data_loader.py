"""
Data loading and preprocessing utilities
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def load_data(tickers: List[str],
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              interval: str = '1d') -> pd.DataFrame:
    """
    Load market data from Yahoo Finance.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : Optional[str]
        Start date (YYYY-MM-DD format). None = 10 years ago
    end_date : Optional[str]
        End date (YYYY-MM-DD format). None = today
    interval : str
        Data interval (default: '1d' for daily)
        
    Returns:
    --------
    pd.DataFrame
        Market data with Date index
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    
    # Download data
    df = yf.download(tickers, start=start_date, end=end_date,
                     interval=interval, auto_adjust=False, progress=False)
    df.reset_index(inplace=True)
    
    # Set index to Date
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')  # Daily frequency
    
    # Remove rows with negative values
    df = df[(df >= 0).all(axis=1)]
    
    # Remove NaN values
    df = df.dropna()
    
    return df


def split_data(df: pd.DataFrame, 
               target_index: str,
               train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    target_index : str
        Target variable name
    train_ratio : float
        Training data ratio (default: 0.7)
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_data, test_data, ohlc_data)
        train_data and test_data use 'Open' prices
        ohlc_data contains full OHLC data for test period
    """
    split_index = int(len(df) * train_ratio)
    train_data = df['Open'].iloc[:split_index]
    test_data = df['Open'].iloc[split_index:]
    ohlc_data = df.iloc[split_index:]
    
    return train_data, test_data, ohlc_data

