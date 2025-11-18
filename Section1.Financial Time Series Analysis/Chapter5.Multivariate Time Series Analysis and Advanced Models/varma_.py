"""
VARMA(p,q) Model Implementation

This script demonstrates how to fit a Vector Autoregressive Moving Average
(VARMA) model to multivariate time series data.

VARMA(p,q) Model:
Combines VAR and VMA components:

Y_t = c + Σᵢ₌₁ᵖ AᵢY_{t-i} + Σⱼ₌₁ᵠ Bⱼε_{t-j} + ε_t

where:
- Y_t: k×1 vector of endogenous variables at time t
- c: k×1 vector of constants
- Aᵢ: k×k AR coefficient matrices for lag i
- Bⱼ: k×k MA coefficient matrices for lag j
- ε_t: k×1 vector of error terms (white noise)
- p: AR order
- q: MA order

VARMA models are more flexible than VAR but computationally more complex.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VAR, VARMAX
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from utils.data_loader import load_financial_data, preprocess_data
    from utils.config import config
    USE_UTILS = True
except ImportError:
    USE_UTILS = False

# Configuration
INTERVAL = '1d'
YEARS_BACK = 10
TICKERS = ['CL=F', 'RB=F', 'ZB=F', 'YM=F', 'NQ=F', 'DX-Y.NYB', '^TNX', 'BZ=F', '^VIX', 'SPY']
TARGET_VAR = 'SPY'
VARMA_P = 1  # VAR order
VARMA_Q = 1  # VMA order
MAX_ITER = 10


def make_stationary(data):
    """
    Make time series stationary by differencing if necessary.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
        
    Returns:
    --------
    pd.Series
        Stationary time series
    """
    result = adfuller(data)
    if result[1] > 0.05:
        return data.diff().dropna()
    return data


def main():
    """Main function"""
    # Set dates
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*YEARS_BACK)).strftime('%Y-%m-%d')
    
    # Download data
    if USE_UTILS:
        df = load_financial_data(TICKERS, start_date, end_date, interval=INTERVAL, progress=False)
        df = preprocess_data(df, frequency='D', use_adjusted=False)
        df = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
    else:
        df = yf.download(TICKERS, start=start_date, end=end_date, interval=INTERVAL, progress=False)['Close']
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.asfreq('D')
        df = df.dropna()
    
    # Stationarity test and differencing
    df_diff = df.apply(make_stationary).dropna()
    
    # Re-test stationarity
    result_diff = adfuller(df_diff[TARGET_VAR])
    print('ADF Statistic after differencing:', result_diff[0])
    print('p-value after differencing:', result_diff[1])
    
    # Fit VARMA model
    print(f"\n{'='*60}")
    print(f"Fitting VARMA({VARMA_P},{VARMA_Q}) Model")
    print("="*60)
    print("Note: VARMA models are computationally intensive and may take time...")
    
    model = VARMAX(df_diff, order=(VARMA_P, VARMA_Q))
    fitted_model = model.fit(disp=True, maxiter=MAX_ITER, method='nm', start_params=None)
    
    print(f"\n{'='*60}")
    print("VARMA Model Summary")
    print("="*60)
    print(fitted_model.summary())


if __name__ == "__main__":
    main()
