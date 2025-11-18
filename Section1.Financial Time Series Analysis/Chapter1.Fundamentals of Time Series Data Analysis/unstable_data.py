"""
Unstable Time Series Data Analysis

This script demonstrates seasonal decomposition of financial time series data.
It loads AAPL stock prices and decomposes them into trend, seasonality, and residuals.

Note: period=252 represents annual trading days. For daily stock data, 
you may want to use weekly (7) or monthly (30) periods depending on your analysis.
"""

import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import os

# Add utils to path for standardized data loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from utils.data_loader import load_financial_data, preprocess_data
    from utils.config import config
    USE_UTILS = True
except ImportError:
    USE_UTILS = False
    print("Note: Using direct yfinance download. Install utils for standardized loading.")

# Configuration
TICKER = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
PERIOD = 252  # Annual trading days (adjust based on your analysis needs)

def main():
    """Main function to perform seasonal decomposition"""
    # Load financial time series data
    if USE_UTILS:
        finance_data = load_financial_data(TICKER, START_DATE, END_DATE)
        finance_data = preprocess_data(finance_data, frequency='D')
    else:
        finance_data = yf.download(TICKER, start=START_DATE, end=END_DATE)
    
    # Decompose the time series data
    # Note: period=252 is for annual seasonality. Adjust based on your needs:
    # - Weekly: period=7
    # - Monthly: period=30
    # - Quarterly: period=90
    # - Annual: period=252
    result = seasonal_decompose(finance_data['Close'], model='additive', period=PERIOD)

    # Plot the trend, seasonality, and residuals
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(finance_data['Close'], label='Original')
    plt.legend(loc='upper left')
    plt.title(f'{TICKER} Stock Price - Original Data')
    plt.subplot(412)
    plt.plot(result.trend, label='Trend')
    plt.legend(loc='upper left')
    plt.title('Trend Component')
    plt.subplot(413)
    plt.plot(result.seasonal, label='Seasonality')
    plt.legend(loc='upper left')
    plt.title(f'Seasonal Component (period={PERIOD})')
    plt.subplot(414)
    plt.plot(result.resid, label='Residuals')
    plt.legend(loc='upper left')
    plt.title('Residual Component')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
