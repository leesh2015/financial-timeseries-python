"""
Standardized data loading and preprocessing utilities
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
    progress: bool = False
) -> pd.DataFrame:
    """
    Load financial data from Yahoo Finance with standardized preprocessing.
    
    Parameters:
    -----------
    tickers : List[str] or str
        Ticker symbol(s) to download
    start_date : Optional[str]
        Start date in 'YYYY-MM-DD' format. If None, uses 3 years ago
    end_date : Optional[str]
        End date in 'YYYY-MM-DD' format. If None, uses today
    interval : str
        Data interval ('1d', '1h', etc.)
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
        start_date = (datetime.today() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    try:
        # Download data
        df = yf.download(tickers, start=start_date, end=end_date, 
                        interval=interval, progress=progress, auto_adjust=False)
        
        # Reset index if Date is in columns
        if 'Date' in df.columns or isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df.reset_index(inplace=True)
        
        # Set Date index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading data for {tickers}: {str(e)}")


def preprocess_data(
    df: pd.DataFrame,
    frequency: str = 'D',
    remove_negative: bool = True,
    remove_na: bool = True,
    use_adjusted: bool = False
) -> pd.DataFrame:
    """
    Standardized data preprocessing pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw financial data
    frequency : str
        Frequency for asfreq ('D', 'B', etc.)
    remove_negative : bool
        Whether to remove rows with negative values
    remove_na : bool
        Whether to remove NaN values
    use_adjusted : bool
        If True, use 'Adj Close' column, else use all columns
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    df = df.copy()
    
    # Select columns
    if use_adjusted and 'Adj Close' in df.columns:
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Adj Close']
        else:
            df = df[['Adj Close']]
    
    # Set frequency
    if frequency:
        df = df.asfreq(frequency)
    
    # Remove negative values
    if remove_negative:
        df = df[(df >= 0).all(axis=1)]
    
    # Remove NaN values
    if remove_na:
        df = df.dropna()
    
    return df


def split_train_test(
    data: pd.DataFrame,
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full dataset
    train_ratio : float
        Ratio of data to use for training (default: 0.7)
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_data, test_data)
    """
    split_index = int(len(data) * train_ratio)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    return train_data, test_data

