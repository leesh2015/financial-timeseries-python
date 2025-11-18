"""
Stock Price Pattern - Moving Average Analysis

This script visualizes how stock prices fluctuate around a moving average,
demonstrating mean-reverting behavior patterns.

The visualization shows:
- Stock price (SPY) over time
- 50-day moving average (trend/mean)
- Green shading: Price above MA
- Red shading: Price below MA

This pattern analysis helps identify:
- Mean-reverting behavior
- Trend-following opportunities
- Potential cointegration with moving average

Mathematical concept:
Moving Average: MA_t = (1/n) × Σᵢ₌₀ⁿ⁻¹ P_{t-i}

where:
- P_t: Price at time t
- n: Window size (e.g., 50 days)
- MA_t: Moving average at time t

Mean-reverting behavior:
- When P_t > MA_t: Price tends to decrease (overvalued)
- When P_t < MA_t: Price tends to increase (undervalued)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
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
TICKER = 'SPY'
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
MA_WINDOW = 50


def main():
    """Main function"""
    # Download data
    print(f"{'='*60}")
    print("Step 1: Loading Data")
    print("="*60)
    if USE_UTILS:
        spy_raw = load_financial_data(TICKER, START_DATE, END_DATE, progress=False)
        spy = preprocess_data(spy_raw, frequency='D', use_adjusted=False)
        
        # Extract Close price
        if isinstance(spy, pd.Series):
            spy_close = spy.copy()
        elif isinstance(spy.columns, pd.MultiIndex):
            spy_close = spy['Close'].iloc[:, 0] if spy['Close'].shape[1] > 0 else spy['Close']
        elif 'Close' in spy.columns:
            spy_close = spy['Close'].copy()
        else:
            spy_close = spy.iloc[:, 0].copy()
    else:
        spy = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        
        # Handle yfinance data structure
        if isinstance(spy.columns, pd.MultiIndex):
            spy_close = spy['Close'].iloc[:, 0]
        else:
            spy_close = spy['Close'].copy()
    
    # Ensure it's a Series
    if not isinstance(spy_close, pd.Series):
        if hasattr(spy_close, 'index'):
            spy_close = pd.Series(spy_close.values, index=spy_close.index, name='SPY')
        else:
            raise ValueError("spy_close is not a Series and has no index")
    
    print(f"Loaded {len(spy_close)} observations")
    print(f"Date range: {spy_close.index.min()} to {spy_close.index.max()}")
    
    # Remove missing values
    spy_close = spy_close.dropna()
    
    # Calculate moving average
    print(f"\n{'='*60}")
    print(f"Step 2: Calculating {MA_WINDOW}-Day Moving Average")
    print("="*60)
    print(f"Formula: MA_t = (1/{MA_WINDOW}) × Σᵢ₌₀^{MA_WINDOW-1} P_{{t-i}}")
    spy_ma = spy_close.rolling(window=MA_WINDOW).mean()
    
    # Create DataFrame for easier plotting
    data = pd.DataFrame({
        'Close': spy_close,
        'Mean': spy_ma
    })
    
    # Remove rows where MA is NaN (first MA_WINDOW-1 rows)
    data = data.dropna()
    
    print(f"Data points after MA calculation: {len(data)}")
    print(f"  Mean price: ${data['Close'].mean():.2f}")
    print(f"  Mean MA: ${data['Mean'].mean():.2f}")
    
    # Plotting
    print(f"\n{'='*60}")
    print("Step 3: Visualization")
    print("="*60)
    plt.figure(figsize=(14, 7))
    
    # Plot price and moving average
    plt.plot(data.index, data['Close'], label=f'{TICKER} (S&P 500 ETF)', 
             color='green', linewidth=2)
    plt.plot(data.index, data['Mean'], label=f'{MA_WINDOW}-Day Moving Average', 
             color='red', linestyle='--', linewidth=2)
    
    # Fill areas: green when price > MA, red when price < MA
    plt.fill_between(data.index, data['Close'], data['Mean'], 
                     where=(data['Close'] > data['Mean']), 
                     color='green', alpha=0.3, interpolate=True, 
                     label='Price > MA (Overvalued)')
    plt.fill_between(data.index, data['Close'], data['Mean'], 
                     where=(data['Close'] <= data['Mean']), 
                     color='red', alpha=0.3, interpolate=True, 
                     label='Price < MA (Undervalued)')
    
    plt.title(f'{TICKER} (S&P 500 ETF) Fluctuating Around a Moving Average', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print(f"\n{'='*60}")
    print("Step 4: Pattern Analysis")
    print("="*60)
    above_ma = (data['Close'] > data['Mean']).sum()
    below_ma = (data['Close'] <= data['Mean']).sum()
    total = len(data)
    
    print(f"Days above MA: {above_ma} ({above_ma/total*100:.1f}%)")
    print(f"Days below MA: {below_ma} ({below_ma/total*100:.1f}%)")
    print(f"\n  Mean-reverting behavior: Price oscillates around MA")
    print(f"  Green areas: Potential sell signals (overvalued)")
    print(f"  Red areas: Potential buy signals (undervalued)")


if __name__ == "__main__":
    main()
