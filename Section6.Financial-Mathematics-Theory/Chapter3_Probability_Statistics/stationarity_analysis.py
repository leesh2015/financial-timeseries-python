"""
Chapter 3: Probability & Time Series Statistics - Stationarity Analysis

Core Analogy: "Spring"
- Stationarity: Property of returning to original position (mean) when stretched
- Non-stationarity: Time series where mean or variance changes over time

This example demonstrates:
1. Understanding the concept of stationarity
2. ADF Test (Augmented Dickey-Fuller Test)
3. Stationarization methods (differencing, log transformation)
4. Stationarity analysis of financial time series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def generate_stationary_series():
    """
    1. Generate Stationary Time Series (Example)
    """
    print("=" * 60)
    print("1. Stationary vs Non-Stationary Time Series")
    print("=" * 60)
    
    np.random.seed(42)
    n = 200
    
    # Stationary time series: AR(1) model (mean reversion)
    # y_t = 0.7 * y_{t-1} + ε_t, ε_t ~ N(0, 1)
    stationary = np.zeros(n)
    for i in range(1, n):
        stationary[i] = 0.7 * stationary[i-1] + np.random.randn()
    
    # Non-stationary time series: Random walk (trend)
    # y_t = y_{t-1} + ε_t
    non_stationary = np.zeros(n)
    for i in range(1, n):
        non_stationary[i] = non_stationary[i-1] + np.random.randn()
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Stationary time series
    axes[0, 0].plot(stationary, 'b-', linewidth=1.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Stationary Time Series (Mean Reversion)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time', fontsize=10)
    axes[0, 0].set_ylabel('Value', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Moving average of stationary series
    window = 20
    ma_stationary = pd.Series(stationary).rolling(window=window).mean()
    axes[0, 1].plot(stationary, 'b-', linewidth=1, alpha=0.5, label='Original')
    axes[0, 1].plot(ma_stationary, 'r-', linewidth=2, label=f'{window}-period Moving Average')
    axes[0, 1].axhline(y=0, color='g', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Stationary Time Series: Constant Mean', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time', fontsize=10)
    axes[0, 1].set_ylabel('Value', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Non-stationary time series
    axes[1, 0].plot(non_stationary, 'r-', linewidth=1.5)
    axes[1, 0].set_title('Non-Stationary Time Series (Random Walk)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time', fontsize=10)
    axes[1, 0].set_ylabel('Value', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Moving average of non-stationary series
    ma_non_stationary = pd.Series(non_stationary).rolling(window=window).mean()
    axes[1, 1].plot(non_stationary, 'r-', linewidth=1, alpha=0.5, label='Original')
    axes[1, 1].plot(ma_non_stationary, 'b-', linewidth=2, label=f'{window}-period Moving Average')
    axes[1, 1].set_title('Non-Stationary Time Series: Changing Mean', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time', fontsize=10)
    axes[1, 1].set_ylabel('Value', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'stationary_vs_nonstationary.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return stationary, non_stationary


def adf_test(series, title="Time Series"):
    """
    2. Perform ADF Test
    """
    print(f"\n[{title} ADF Test]")
    result = adfuller(series.dropna())
    
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    print(f"  ADF Statistic: {adf_statistic:.6f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Critical Values:")
    for key, value in critical_values.items():
        print(f"    {key}: {value:.6f}")
    
    if p_value <= 0.05:
        print(f"  → Stationary (p-value < 0.05)")
        is_stationary = True
    else:
        print(f"  → Non-stationary (p-value >= 0.05)")
        is_stationary = False
    
    return adf_statistic, p_value, is_stationary


def analyze_financial_series(ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    """
    3. Real Financial Time Series Analysis
    """
    print("\n" + "=" * 60)
    print("3. Real Financial Time Series Stationarity Analysis")
    print("=" * 60)
    
    # Download data
    print(f"\nDownloading {ticker} data...")
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    
    # Prices and returns
    prices = data
    returns = data.pct_change().dropna()
    log_returns = np.log(data / data.shift(1)).dropna()
    
    print(f"Data period: {prices.index[0].date()} ~ {prices.index[-1].date()}")
    print(f"Number of observations: {len(prices)}")
    
    # ADF Test
    print("\n[Price Time Series]")
    adf_prices, pval_prices, is_stat_prices = adf_test(prices, "Price")
    
    print("\n[Returns Time Series]")
    adf_returns, pval_returns, is_stat_returns = adf_test(returns, "Returns")
    
    print("\n[Log Returns Time Series]")
    adf_log_returns, pval_log_returns, is_stat_log_returns = adf_test(log_returns, "Log Returns")
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Prices
    axes[0, 0].plot(prices.index, prices.values, 'b-', linewidth=1.5)
    axes[0, 0].set_title(f'{ticker} Price (Non-Stationary)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date', fontsize=10)
    axes[0, 0].set_ylabel('Price', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Moving average of prices
    ma_prices = prices.rolling(window=20).mean()
    axes[0, 1].plot(prices.index, prices.values, 'b-', linewidth=1, alpha=0.5, label='Price')
    axes[0, 1].plot(ma_prices.index, ma_prices.values, 'r-', linewidth=2, label='20-day Moving Average')
    axes[0, 1].set_title('Price: Has Trend (Non-Stationary)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Date', fontsize=10)
    axes[0, 1].set_ylabel('Price', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Returns
    axes[1, 0].plot(returns.index, returns.values, 'g-', linewidth=1, alpha=0.7)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_title(f'{ticker} Returns (Stationary)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Date', fontsize=10)
    axes[1, 0].set_ylabel('Returns', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Moving average of returns
    ma_returns = returns.rolling(window=20).mean()
    axes[1, 1].plot(returns.index, returns.values, 'g-', linewidth=1, alpha=0.3, label='Returns')
    axes[1, 1].plot(ma_returns.index, ma_returns.values, 'r-', linewidth=2, label='20-day Moving Average')
    axes[1, 1].axhline(y=0, color='b', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Returns: Constant Mean (Stationary)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Date', fontsize=10)
    axes[1, 1].set_ylabel('Returns', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Log returns
    axes[2, 0].plot(log_returns.index, log_returns.values, 'purple', linewidth=1, alpha=0.7)
    axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[2, 0].set_title(f'{ticker} Log Returns (Stationary)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Date', fontsize=10)
    axes[2, 0].set_ylabel('Log Returns', fontsize=10)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Returns distribution
    axes[2, 1].hist(returns.values, bins=50, density=True, alpha=0.7, color='green', label='Returns')
    returns_mean = np.mean(returns.values)  # Use numpy to ensure scalar value
    axes[2, 1].axvline(x=returns_mean, color='r', linestyle='--', linewidth=2, label='Mean')
    axes[2, 1].set_title('Returns Distribution', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Returns', fontsize=10)
    axes[2, 1].set_ylabel('Density', fontsize=10)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'financial_stationarity_{ticker}.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return prices, returns, log_returns


def make_stationary(series, method='diff'):
    """
    4. Methods to Make Non-Stationary Time Series Stationary
    """
    print("\n" + "=" * 60)
    print("4. Methods to Stationarize Non-Stationary Time Series")
    print("=" * 60)
    
    if method == 'diff':
        # Differencing: y_t - y_{t-1}
        stationary_series = series.diff().dropna()
        print("\n[Differencing Method]")
        print("  Formula: Δy_t = y_t - y_{t-1}")
        print("  Effect: Removes trend")
    elif method == 'log_diff':
        # Log differencing: log(y_t) - log(y_{t-1}) = log(y_t / y_{t-1})
        log_series = np.log(series)
        stationary_series = log_series.diff().dropna()
        print("\n[Log Differencing Method]")
        print("  Formula: Δlog(y_t) = log(y_t) - log(y_{t-1})")
        print("  Effect: Removes trend + Converts to proportional change")
    
    # ADF Test
    adf_stat, p_value, is_stationary = adf_test(stationary_series, "Stationarized Time Series")
    
    return stationary_series


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 3: Probability & Time Series Statistics - Stationarity Analysis")
    print("=" * 60)
    
    # 1. Stationary vs Non-stationary time series
    stationary, non_stationary = generate_stationary_series()
    
    # ADF Test
    print("\n" + "=" * 60)
    print("2. ADF Test Results")
    print("=" * 60)
    
    stationary_series = pd.Series(stationary)
    non_stationary_series = pd.Series(non_stationary)
    
    adf_test(stationary_series, "Stationary Time Series")
    adf_test(non_stationary_series, "Non-Stationary Time Series")
    
    # 3. Real financial data analysis
    prices, returns, log_returns = analyze_financial_series('AAPL')
    
    # 4. Stationarization methods
    stationary_prices = make_stationary(prices, method='diff')
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Stationarity: Mean and variance are constant over time")
    print("2. Financial prices are usually non-stationary (trend exists)")
    print("3. Financial returns are usually stationary (stationarized by differencing)")
    print("4. ADF Test: p-value < 0.05 indicates stationarity")
    print("5. ARIMA models assume stationary time series, so differencing is needed")


if __name__ == "__main__":
    main()

