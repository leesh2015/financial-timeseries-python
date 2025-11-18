"""
Partial Autocorrelation Function (PACF) Analysis

This script demonstrates how to calculate and visualize the PACF
for financial time series data.

PACF (Partial Autocorrelation Function):
Measures the correlation between Y_t and Y_{t-k} after removing
the effects of intermediate lags (Y_{t-1}, ..., Y_{t-k+1}).

Mathematical definition:
PACF(k) = Corr(Y_t, Y_{t-k} | Y_{t-1}, ..., Y_{t-k+1})

where:
- k: lag (time difference)
- Conditional correlation: correlation after controlling for intermediate lags

Properties:
- PACF(1) = ACF(1) (no intermediate lags)
- For AR(p) process: PACF cuts off after lag p
- For MA(q) process: PACF decays slowly
- Significant PACF values help identify AR order

This example uses log-differenced AAPL prices to identify AR structure.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from utils.data_loader import load_financial_data, preprocess_data
    USE_UTILS = True
except ImportError:
    USE_UTILS = False

# Configuration
TICKER = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2021-01-01'
MAX_LAGS = 30


def main():
    """Main function"""
    # Download Apple close price data
    print(f"{'='*60}")
    print("Step 1: Loading Data")
    print("="*60)
    if USE_UTILS:
        data_raw = load_financial_data(TICKER, START_DATE, END_DATE, progress=False)
        data = preprocess_data(data_raw, frequency='D', use_adjusted=False)
        
        # Extract Close price - handle different data structures
        if isinstance(data, pd.Series):
            close_prices = data.copy()
        elif isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close'].iloc[:, 0] if data['Close'].shape[1] > 0 else data['Close']
        elif 'Close' in data.columns:
            close_prices = data['Close'].copy()
        else:
            close_prices = data.iloc[:, 0].copy()
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        
        # Handle yfinance data structure
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close'].iloc[:, 0]
        else:
            close_prices = data['Close'].copy()
    
    # Ensure it's a Series
    if not isinstance(close_prices, pd.Series):
        if hasattr(close_prices, 'index'):
            close_prices = pd.Series(close_prices.values, index=close_prices.index, name=TICKER)
        else:
            raise ValueError("close_prices is not a Series and has no index")
    
    print(f"Loaded {len(close_prices)} observations")
    print(f"Date range: {close_prices.index.min()} to {close_prices.index.max()}")
    
    # Calculate log difference (log returns)
    print(f"\n{'='*60}")
    print("Step 2: Calculating Log Returns")
    print("="*60)
    print("Log difference: Δlog(Y_t) = log(Y_t) - log(Y_{t-1})")
    log_diff = np.log(close_prices).diff().dropna()
    
    # Ensure scalar values for mean and std
    log_diff_mean = float(log_diff.mean())
    log_diff_std = float(log_diff.std())
    
    print(f"Log returns shape: {log_diff.shape}")
    print(f"  Mean: {log_diff_mean:.6f} (≈ 0)")
    print(f"  Std: {log_diff_std:.6f}")
    
    # Plot PACF for differenced data
    print(f"\n{'='*60}")
    print(f"Step 3: PACF Analysis (max_lags={MAX_LAGS})")
    print("="*60)
    print("PACF measures partial correlation after removing intermediate lags")
    print("  - Significant spikes indicate AR terms")
    print("  - PACF cuts off after lag p for AR(p) process")
    
    plt.figure(figsize=(12, 6))
    plot_pacf(log_diff, lags=MAX_LAGS, ax=plt.gca(), method='ywm', alpha=0.05)
    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Partial Autocorrelation', fontsize=12)
    plt.title(f'Partial Autocorrelation Function (PACF) for {TICKER} Log Returns', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print("Interpretation")
    print("="*60)
    print("  - PACF helps identify AR order in ARIMA models")
    print("  - Values within confidence bands are not significant")
    print("  - Significant PACF at lag k suggests AR(k) term needed")
    print("  - Compare with ACF to distinguish AR vs MA processes")


if __name__ == "__main__":
    main()
