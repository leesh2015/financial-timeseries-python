"""
Data loading utilities for Fama-French factor models
"""

import pandas as pd
import yfinance as yf
import warnings
from typing import Optional, Tuple

warnings.filterwarnings("ignore")




def load_ff_factors(model_type: str = '3-factor', frequency: str = 'daily') -> pd.DataFrame:
    """
    Load Fama-French factor data from Kenneth French's website
    
    Parameters:
    -----------
    model_type : str
        Type of model: '3-factor', '5-factor', or '6-factor'
    frequency : str
        Data frequency: 'daily' or 'monthly'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with factor data (Date index, factor columns)
    """
    base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    
    # Map model types to file names
    file_map = {
        ('3-factor', 'daily'): 'F-F_Research_Data_Factors_daily_CSV.zip',
        ('3-factor', 'monthly'): 'F-F_Research_Data_Factors_CSV.zip',
        ('5-factor', 'daily'): 'F-F_Research_Data_5_Factors_2x3_daily_CSV.zip',
        ('5-factor', 'monthly'): 'F-F_Research_Data_5_Factors_2x3_CSV.zip',
    }
    
    # For 6-factor, we need to merge 5-factor and momentum
    if model_type == '6-factor':
        if frequency == 'daily':
            ff5_file = 'F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
            mom_file = 'F-F_Momentum_Factor_daily_CSV.zip'
            ff5_skiprows = 3
            mom_skiprows = 13
        else:
            ff5_file = 'F-F_Research_Data_5_Factors_2x3_CSV.zip'
            mom_file = 'F-F_Momentum_Factor_CSV.zip'
            ff5_skiprows = 3
            mom_skiprows = 13
        
        # Load 5-factor data
        url_ff5 = base_url + ff5_file
        ff5_data = pd.read_csv(url_ff5, skiprows=ff5_skiprows, skipfooter=1, engine='python')
        
        # Load momentum data
        url_mom = base_url + mom_file
        mom_data = pd.read_csv(url_mom, skiprows=mom_skiprows, skipfooter=1, engine='python')
        
        # Process 5-factor data
        if frequency == 'daily':
            ff5_data.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        else:
            ff5_data.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        
        ff5_data['Date'] = pd.to_datetime(ff5_data['Date'], format='%Y%m%d', errors='coerce')
        ff5_data = ff5_data.dropna(subset=['Date'])
        ff5_data.set_index('Date', inplace=True)
        
        # Process momentum data
        mom_data.columns = ['Date', 'MOM']
        mom_data['Date'] = pd.to_datetime(mom_data['Date'], format='%Y%m%d', errors='coerce')
        mom_data = mom_data.dropna(subset=['Date'])
        mom_data.set_index('Date', inplace=True)
        
        # Merge
        result = pd.merge(ff5_data, mom_data, left_index=True, right_index=True, how='inner')
        return result
    
    # For 3-factor and 5-factor
    file_name = file_map.get((model_type, frequency))
    if file_name is None:
        raise ValueError(f"Unsupported model_type '{model_type}' and frequency '{frequency}' combination")
    
    url = base_url + file_name
    skiprows = 3 if frequency == 'daily' else 3
    
    # Use pd.read_csv directly with URL (same as original working code)
    data = pd.read_csv(url, skiprows=skiprows, skipfooter=1, engine='python')
    
    # Set column names based on model type
    if model_type == '3-factor':
        data.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    elif model_type == '5-factor':
        data.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    
    # Convert date column
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d', errors='coerce')
    data = data.dropna(subset=['Date'])
    data.set_index('Date', inplace=True)
    
    return data


def load_stock_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Load stock price data using yfinance
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start : str
        Start date (YYYY-MM-DD)
    end : str
        End date (YYYY-MM-DD)
    interval : str
        Data interval (default: '1d' for daily)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with stock price data
    """
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Remove timezone information to match Fama-French data (timezone-naive)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    return data


def align_data(
    stock_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    price_column: str = 'Close'
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Align stock returns with factor data by date
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Stock price data
    factor_data : pd.DataFrame
        Fama-French factor data
    price_column : str
        Column name for stock prices (default: 'Close')
    
    Returns:
    --------
    Tuple[pd.Series, pd.DataFrame]
        Aligned stock returns and factor data
    """
    # Calculate stock returns
    stock_returns = stock_data[price_column].pct_change().dropna()
    
    # Align by date index
    common_dates = stock_returns.index.intersection(factor_data.index)
    stock_returns_aligned = stock_returns.loc[common_dates]
    factor_data_aligned = factor_data.loc[common_dates]
    
    return stock_returns_aligned, factor_data_aligned

