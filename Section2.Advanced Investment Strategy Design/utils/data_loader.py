"""
Standardized data loading and preprocessing utilities for investment strategies
"""

import pandas as pd
import yfinance as yf
import warnings
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


def load_financial_data(
    tickers: List[str] or str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d',
    auto_adjust: bool = False,
    progress: bool = False
) -> pd.DataFrame:
    """
    Load financial data from Yahoo Finance with standardized preprocessing.
    
    Parameters:
    -----------
    tickers : List[str] or str
        Ticker symbol(s) to download
    start_date : Optional[str]
        Start date in 'YYYY-MM-DD' format. If None, uses 10 years ago
    end_date : Optional[str]
        End date in 'YYYY-MM-DD' format. If None, uses today
    interval : str
        Data interval ('1d', '1h', etc.)
    auto_adjust : bool
        Whether to use auto-adjusted prices
    progress : bool
        Whether to show download progress
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed financial data with Date index
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    
    try:
        # Download data
        df = yf.download(tickers, start=start_date, end=end_date, 
                        interval=interval, auto_adjust=auto_adjust, progress=progress)
        
        # Reset index if Date is in columns
        if 'Date' in df.columns or isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df.reset_index(inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
        
        # Set frequency
        df = df.asfreq('D')
        
        # Remove rows with negative values
        df = df[(df >= 0).all(axis=1)]
        
        # Remove NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading financial data: {e}")


def preprocess_data(
    df: pd.DataFrame,
    frequency: str = 'D',
    use_adjusted: bool = False
) -> pd.DataFrame:
    """
    Preprocess financial data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw financial data
    frequency : str
        Frequency to set ('D', 'B', etc.)
    use_adjusted : bool
        Whether to use adjusted prices
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    df = df.copy()
    
    # Set frequency
    if frequency:
        df = df.asfreq(frequency)
    
    # Remove negative values
    df = df[(df >= 0).all(axis=1)]
    
    # Remove NaN
    df = df.dropna()
    
    return df


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    price_column: str = 'Open'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    train_ratio : float
        Ratio of data to use for training (default: 0.7)
    price_column : str
        Column to extract ('Open', 'Close', etc.)
        
    Returns:
    --------
    tuple: (train_data, test_data, ohlc_data)
        Training data, test data, and OHLC data for test period
    """
    split_index = int(len(df) * train_ratio)
    
    # Extract price column
    if isinstance(df.columns, pd.MultiIndex):
        if price_column in df.columns.levels[0]:
            price_data = df[price_column]
        else:
            price_data = df.iloc[:, 0]  # Use first column
    elif price_column in df.columns:
        price_data = df[price_column]
    else:
        price_data = df.iloc[:, 0]
    
    train_data = price_data.iloc[:split_index]
    test_data = price_data.iloc[split_index:]
    ohlc_data = df.iloc[split_index:]
    
    return train_data, test_data, ohlc_data

