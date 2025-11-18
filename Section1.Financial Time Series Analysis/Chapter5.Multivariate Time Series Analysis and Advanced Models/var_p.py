"""
VAR(p) Model Implementation

This script demonstrates how to fit a Vector Autoregression (VAR) model
to multivariate time series data.

VAR(p) Model:
A VAR model of order p for k variables can be written as:

Y_t = c + A₁Y_{t-1} + A₂Y_{t-2} + ... + AₚY_{t-p} + ε_t

where:
- Y_t: k×1 vector of endogenous variables at time t
- c: k×1 vector of constants
- Aᵢ: k×k coefficient matrices for lag i
- ε_t: k×1 vector of error terms (white noise)
- p: lag order

The model captures dynamic interactions between multiple time series.
"""

import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from datetime import datetime, timedelta
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
MAX_LAGS = 5


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
    
    # Re-test for stationarity
    result_diff = adfuller(df_diff[TARGET_VAR])
    print('ADF Statistic after differencing:', result_diff[0])
    print('p-value after differencing:', result_diff[1])
    
    # Build and evaluate the VAR model (AIC criteria)
    print(f"\n{'='*60}")
    print("Fitting VAR model using AIC criterion")
    print("="*60)
    model_aic = VAR(df_diff)
    fitted_model_aic = model_aic.fit(maxlags=MAX_LAGS, ic='aic')
    
    # Evaluate model fit (AIC criteria)
    lag_order_aic = fitted_model_aic.k_ar
    aic_value = fitted_model_aic.aic
    print(f"AIC-based Model - Lag Order: {lag_order_aic}")
    print(f"AIC Value: {aic_value:.4f}")
    print(f"\nAIC-based Model Summary:\n{fitted_model_aic.summary()}")
    
    # Build and evaluate the VAR model (BIC criteria)
    print(f"\n{'='*60}")
    print("Fitting VAR model using BIC criterion")
    print("="*60)
    model_bic = VAR(df_diff)
    fitted_model_bic = model_bic.fit(maxlags=MAX_LAGS, ic='bic')
    
    # Evaluate model fit (BIC criteria)
    lag_order_bic = fitted_model_bic.k_ar
    bic_value = fitted_model_bic.bic
    print(f"BIC-based Model - Lag Order: {lag_order_bic}")
    print(f"BIC Value: {bic_value:.4f}")
    print(f"\nBIC-based Model Summary:\n{fitted_model_bic.summary()}")
    
    # Comparison
    print(f"\n{'='*60}")
    print("Model Comparison")
    print("="*60)
    print(f"AIC Model: Lag {lag_order_aic}, AIC = {aic_value:.4f}")
    print(f"BIC Model: Lag {lag_order_bic}, BIC = {bic_value:.4f}")
    print(f"\nNote: BIC typically selects simpler models (lower lag order) than AIC")


if __name__ == "__main__":
    main()
