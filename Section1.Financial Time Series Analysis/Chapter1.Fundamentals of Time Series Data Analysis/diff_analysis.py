"""
Differencing Analysis

This script demonstrates differencing and log-differencing techniques
to transform non-stationary time series into stationary series.

Differencing:
First difference: ΔY_t = Y_t - Y_{t-1}

Log Differencing (Log Returns):
Log difference: Δlog(Y_t) = log(Y_t) - log(Y_{t-1}) = log(Y_t / Y_{t-1})

Properties:
- Differencing removes trends and makes series stationary
- Log differencing provides percentage changes (returns)
- Log differencing is preferred for financial data (multiplicative effects)
- After differencing: E[ΔY_t] ≈ 0, Var[ΔY_t] ≈ constant (stationary)

Mean and variance analysis:
- Original series: Non-stationary (mean/variance change over time)
- Differenced series: Stationary (mean ≈ 0, constant variance)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from utils.data_loader import load_financial_data, preprocess_data
    USE_UTILS = True
except ImportError:
    USE_UTILS = False

# Configuration
TICKERS = ['AAPL', 'TSLA']
START_DATE = '2020-01-01'
END_DATE = '2021-01-01'


def main():
    """Main function"""
    # Load stock data
    if USE_UTILS:
        aapl_raw = load_financial_data('AAPL', START_DATE, END_DATE, progress=False)
        tsla_raw = load_financial_data('TSLA', START_DATE, END_DATE, progress=False)
        aapl = preprocess_data(aapl_raw, frequency='D', use_adjusted=False)
        tsla = preprocess_data(tsla_raw, frequency='D', use_adjusted=False)
        
        # Extract Close prices - handle different data structures
        if isinstance(aapl, pd.Series):
            aapl_close = aapl.copy()
        elif isinstance(aapl.columns, pd.MultiIndex):
            aapl_close = aapl['Close'].iloc[:, 0] if aapl['Close'].shape[1] > 0 else aapl['Close']
        elif 'Close' in aapl.columns:
            aapl_close = aapl['Close'].copy()
        else:
            aapl_close = aapl.iloc[:, 0].copy()
        
        if isinstance(tsla, pd.Series):
            tsla_close = tsla.copy()
        elif isinstance(tsla.columns, pd.MultiIndex):
            tsla_close = tsla['Close'].iloc[:, 0] if tsla['Close'].shape[1] > 0 else tsla['Close']
        elif 'Close' in tsla.columns:
            tsla_close = tsla['Close'].copy()
        else:
            tsla_close = tsla.iloc[:, 0].copy()
    else:
        aapl = yf.download('AAPL', start=START_DATE, end=END_DATE, progress=False)
        tsla = yf.download('TSLA', start=START_DATE, end=END_DATE, progress=False)
        
        # Handle yfinance data structure
        if isinstance(aapl.columns, pd.MultiIndex):
            aapl_close = aapl['Close'].iloc[:, 0]
        else:
            aapl_close = aapl['Close'].copy()
        
        if isinstance(tsla.columns, pd.MultiIndex):
            tsla_close = tsla['Close'].iloc[:, 0]
        else:
            tsla_close = tsla['Close'].copy()
    
    # Ensure both are Series
    if not isinstance(aapl_close, pd.Series):
        if hasattr(aapl_close, 'index'):
            aapl_close = pd.Series(aapl_close.values, index=aapl_close.index, name='AAPL')
        else:
            raise ValueError("aapl_close is not a Series and has no index")
    
    if not isinstance(tsla_close, pd.Series):
        if hasattr(tsla_close, 'index'):
            tsla_close = pd.Series(tsla_close.values, index=tsla_close.index, name='TSLA')
        else:
            raise ValueError("tsla_close is not a Series and has no index")

    # Plot original data
    print(f"{'='*60}")
    print("Step 1: Original Stock Prices")
    print("="*60)
    plt.figure(figsize=(12, 6))
    plt.plot(aapl_close.index, aapl_close.values, label='AAPL', linewidth=2)
    plt.plot(tsla_close.index, tsla_close.values, label='TSLA', linewidth=2)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.title('AAPL vs TSLA Stock Prices', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate and print mean and variance
    # Ensure we get scalar values, not Series
    aapl_mean = float(aapl_close.mean())
    aapl_var = float(aapl_close.var())
    tsla_mean = float(tsla_close.mean())
    tsla_var = float(tsla_close.var())
    
    print(f"\nOriginal Series Statistics:")
    print(f"  AAPL Mean: ${aapl_mean:.2f}")
    print(f"  AAPL Variance: {aapl_var:.2f}")
    print(f"  TSLA Mean: ${tsla_mean:.2f}")
    print(f"  TSLA Variance: {tsla_var:.2f}")
    print(f"\n  Note: Non-stationary series (mean/variance may change over time)")
    
    # Differencing: ΔY_t = Y_t - Y_{t-1}
    price_diff_aapl = aapl_close.diff().dropna()
    price_diff_tsla = tsla_close.diff().dropna()
    
    # Log Differencing: Δlog(Y_t) = log(Y_t) - log(Y_{t-1})
    log_diff_aapl = np.log(aapl_close).diff().dropna()
    log_diff_tsla = np.log(tsla_close).diff().dropna()
    
    # Plot differenced data
    print(f"\n{'='*60}")
    print("Step 2: Differenced Prices (ΔY_t = Y_t - Y_{t-1})")
    print("="*60)
    plt.figure(figsize=(12, 6))
    plt.plot(price_diff_aapl.index, price_diff_aapl.values, label='AAPL Differenced', linewidth=2)
    plt.plot(price_diff_tsla.index, price_diff_tsla.values, label='TSLA Differenced', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price Difference (USD)', fontsize=12)
    plt.title('AAPL vs TSLA Differenced Prices', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot log differenced data
    print(f"\n{'='*60}")
    print("Step 3: Log Differenced Prices (Log Returns)")
    print("="*60)
    plt.figure(figsize=(12, 6))
    plt.plot(log_diff_aapl.index, log_diff_aapl.values, label='AAPL Log Differenced', linewidth=2)
    plt.plot(log_diff_tsla.index, log_diff_tsla.values, label='TSLA Log Differenced', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Log Price Difference (Log Returns)', fontsize=12)
    plt.title('AAPL vs TSLA Log Differenced Prices', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print mean and variance of differenced and log differenced data
    print(f"\n{'='*60}")
    print("Step 4: Stationarity Analysis")
    print("="*60)
    print(f"\nDifferenced Series Statistics:")
    # Ensure scalar values
    price_diff_aapl_mean = float(price_diff_aapl.mean())
    price_diff_aapl_var = float(price_diff_aapl.var())
    price_diff_tsla_mean = float(price_diff_tsla.mean())
    price_diff_tsla_var = float(price_diff_tsla.var())
    
    print(f"  AAPL Differenced Mean: {price_diff_aapl_mean:.6f} (≈ 0)")
    print(f"  AAPL Differenced Variance: {price_diff_aapl_var:.6f}")
    print(f"  TSLA Differenced Mean: {price_diff_tsla_mean:.6f} (≈ 0)")
    print(f"  TSLA Differenced Variance: {price_diff_tsla_var:.6f}")
    
    print(f"\nLog Differenced Series Statistics:")
    # Ensure scalar values
    log_diff_aapl_mean = float(log_diff_aapl.mean())
    log_diff_aapl_var = float(log_diff_aapl.var())
    log_diff_tsla_mean = float(log_diff_tsla.mean())
    log_diff_tsla_var = float(log_diff_tsla.var())
    
    print(f"  AAPL Log Differenced Mean: {log_diff_aapl_mean:.6f} (≈ 0)")
    print(f"  AAPL Log Differenced Variance: {log_diff_aapl_var:.6f}")
    print(f"  TSLA Log Differenced Mean: {log_diff_tsla_mean:.6f} (≈ 0)")
    print(f"  TSLA Log Differenced Variance: {log_diff_tsla_var:.6f}")
    
    print(f"\n  Differenced series are approximately stationary:")
    print(f"    - Mean ≈ 0 (no trend)")
    print(f"    - Constant variance (homoscedastic)")
    print(f"    - Suitable for time series modeling")


if __name__ == "__main__":
    main()
