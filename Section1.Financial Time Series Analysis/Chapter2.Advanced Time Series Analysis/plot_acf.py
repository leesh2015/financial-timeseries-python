"""
Autocorrelation Function (ACF) Analysis

This script demonstrates how to calculate and visualize the ACF
for financial time series data.

ACF (Autocorrelation Function):
Measures the correlation between a time series and its lagged values.

Mathematical definition:
ACF(k) = Corr(Y_t, Y_{t-k}) = E[(Y_t - μ)(Y_{t-k} - μ)] / σ²

where:
- k: lag (time difference)
- μ: mean of the series
- σ²: variance of the series

Properties:
- ACF(0) = 1 (perfect correlation with itself)
- ACF(k) = ACF(-k) (symmetric)
- For stationary series: ACF decays to 0 as k increases
- Significant ACF values indicate serial correlation

This example uses log-differenced AAPL prices (returns) which are
typically stationary and suitable for ACF analysis.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
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
    
    # Calculate autocorrelation coefficients
    print(f"\n{'='*60}")
    print(f"Step 3: Calculating ACF (max_lags={MAX_LAGS})")
    print("="*60)
    autocorrelations_acf = acf(log_diff, nlags=MAX_LAGS, fft=False)
    
    # Print results
    print("\nLag | Autocorrelation")
    print("-" * 30)
    for lag, value in enumerate(autocorrelations_acf[:11]):  # Show first 11 lags
        print(f"{lag:3} | {value:8.5f}")
    print("...")
    
    # Calculate significant lags (outside f-string to avoid syntax issues)
    significant_lags = np.where(np.abs(autocorrelations_acf[1:]) > 0.1)[0] + 1
    
    # Ensure scalar value for ACF(0)
    acf_0 = float(autocorrelations_acf[0])
    
    print(f"\n  ACF(0) = {acf_0:.5f} (always 1.0)")
    print(f"  Significant lags: {significant_lags.tolist()}")
    
    # Plot ACF
    print(f"\n{'='*60}")
    print("Step 4: ACF Visualization")
    print("="*60)
    plt.figure(figsize=(12, 6))
    plot_acf(log_diff, lags=MAX_LAGS, ax=plt.gca(), alpha=0.05)
    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    plt.title(f'Autocorrelation Function (ACF) for {TICKER} Log Returns', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print("Interpretation")
    print("="*60)
    print("  - ACF measures serial correlation in the time series")
    print("  - Values within confidence bands are not significant")
    print("  - Significant ACF values indicate predictable patterns")
    print("  - For log returns: ACF should be close to 0 (weak autocorrelation)")


if __name__ == "__main__":
    main()
