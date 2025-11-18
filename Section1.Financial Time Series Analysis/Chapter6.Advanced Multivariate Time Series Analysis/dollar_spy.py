"""
DXY vs SPY Co-movement Analysis

This script visualizes the relationship between the US Dollar Index (DXY)
and the S&P 500 ETF (SPY) to identify co-moving and counter-moving periods.

The script:
- Downloads DXY and SPY data
- Calculates 50-day moving averages
- Highlights periods where both move in the same direction (co-moving)
- Highlights periods where they move in opposite directions (counter-moving)

This analysis helps identify:
- Correlation patterns between dollar strength and equity markets
- Potential cointegration relationships
- Trading opportunities based on relative movements
"""

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
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
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
MA_WINDOW = 50


def main():
    """Main function"""
    # Download data
    if USE_UTILS:
        dxy_raw = load_financial_data('DX-Y.NYB', START_DATE, END_DATE, progress=False)
        spy_raw = load_financial_data('SPY', START_DATE, END_DATE, progress=False)
        dxy = preprocess_data(dxy_raw, frequency='D', use_adjusted=False)
        spy = preprocess_data(spy_raw, frequency='D', use_adjusted=False)
        
        # Extract Close prices - handle different data structures
        # yfinance returns DataFrame with Date index and 'Close' column for single ticker
        if isinstance(dxy, pd.Series):
            dxy_close = dxy.copy()
        elif isinstance(dxy.columns, pd.MultiIndex):
            # MultiIndex: select 'Close' column
            if 'Close' in dxy.columns.levels[0]:
                dxy_close = dxy['Close'].iloc[:, 0] if dxy['Close'].shape[1] > 0 else dxy['Close']
            else:
                dxy_close = dxy.iloc[:, 0]
        elif 'Close' in dxy.columns:
            dxy_close = dxy['Close'].copy()
        else:
            dxy_close = dxy.iloc[:, 0].copy()
        
        if isinstance(spy, pd.Series):
            spy_close = spy.copy()
        elif isinstance(spy.columns, pd.MultiIndex):
            if 'Close' in spy.columns.levels[0]:
                spy_close = spy['Close'].iloc[:, 0] if spy['Close'].shape[1] > 0 else spy['Close']
            else:
                spy_close = spy.iloc[:, 0]
        elif 'Close' in spy.columns:
            spy_close = spy['Close'].copy()
        else:
            spy_close = spy.iloc[:, 0].copy()
    else:
        # Direct yfinance download
        dxy = yf.download('DX-Y.NYB', start=START_DATE, end=END_DATE, progress=False)
        spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
        
        # yfinance returns DataFrame with Date index
        # For single ticker: columns are ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if isinstance(dxy.columns, pd.MultiIndex):
            # Multiple tickers case (shouldn't happen for single ticker)
            dxy_close = dxy['Close'].iloc[:, 0]
        else:
            dxy_close = dxy['Close'].copy()
        
        if isinstance(spy.columns, pd.MultiIndex):
            spy_close = spy['Close'].iloc[:, 0]
        else:
            spy_close = spy['Close'].copy()
    
    # Ensure both are Series with DatetimeIndex
    if not isinstance(dxy_close, pd.Series):
        if hasattr(dxy_close, 'index'):
            dxy_close = pd.Series(dxy_close.values, index=dxy_close.index, name='DXY')
        else:
            raise ValueError("dxy_close is not a Series and has no index")
    
    if not isinstance(spy_close, pd.Series):
        if hasattr(spy_close, 'index'):
            spy_close = pd.Series(spy_close.values, index=spy_close.index, name='SPY')
        else:
            raise ValueError("spy_close is not a Series and has no index")
    
    # Align indices (inner join to keep only common dates)
    dxy_close, spy_close = dxy_close.align(spy_close, join='inner')
    
    # Create DataFrame with aligned Series
    data = pd.DataFrame({
        'DXY': dxy_close,
        'SPY': spy_close
    })
    
    # Remove any remaining missing values
    data = data.dropna()

    # Calculate moving averages
    data['DXY_MA'] = data['DXY'].rolling(window=MA_WINDOW).mean()
    data['SPY_MA'] = data['SPY'].rolling(window=MA_WINDOW).mean()

    # Create chart
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # First y-axis (DXY and DXY_MA)
    ax1.plot(data.index, data['DXY'], color='blue', label=f'DXY (Dollar Index)', linewidth=2)
    ax1.plot(data.index, data['DXY_MA'], color='darkblue', linestyle='--', label=f'DXY {MA_WINDOW}-Day MA', linewidth=1.5)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('DXY (Dollar Index)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Second y-axis (SPY and SPY_MA)
    ax2 = ax1.twinx()
    ax2.plot(data.index, data['SPY'], color='green', label='SPY (S&P 500 ETF)', linewidth=2)
    ax2.plot(data.index, data['SPY_MA'], color='darkgreen', linestyle='--', label=f'SPY {MA_WINDOW}-Day MA', linewidth=1.5)
    ax2.set_ylabel('SPY (S&P 500 ETF)', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right', fontsize=10)
    
    # Color background for co-moving and counter-moving sections
    for i in range(MA_WINDOW, len(data)):  # Start from MA_WINDOW to account for the moving average window
        dxy_above_ma = data['DXY'].iloc[i] > data['DXY_MA'].iloc[i]
        spy_above_ma = data['SPY'].iloc[i] > data['SPY_MA'].iloc[i]
        
        if (dxy_above_ma and spy_above_ma) or (not dxy_above_ma and not spy_above_ma):
            # Co-moving: both above or both below their MAs
            ax1.axvspan(data.index[i-1], data.index[i], color='lightgreen', alpha=0.3)
        else:
            # Counter-moving: one above, one below
            ax1.axvspan(data.index[i-1], data.index[i], color='lightcoral', alpha=0.3)
    
    plt.title('DXY (Dollar Index) vs SPY (S&P 500 ETF) - Co-movement Analysis', 
              fontsize=14, fontweight='bold', pad=20)
    plt.figtext(0.5, 0.02, 'Green: Co-moving periods | Red: Counter-moving periods', 
                ha='center', fontsize=10, style='italic')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

