"""
MA (Moving Average) Model Implementation

This script demonstrates how to fit an MA model using ACF (Autocorrelation Function)
to determine the optimal MA order, then fits an ARIMA(0,0,q) model.

ACF helps identify the MA order: significant spikes indicate MA terms.
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
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
    import yfinance as yf

# Configuration
TICKER = 'SPY'
START_DATE = '2020-01-01'
END_DATE = '2023-01-01'
MA_ORDER = 1  # MA order (can be adjusted based on ACF plot)


def main():
    """Main function"""
    # Load data - following original code order exactly
    if USE_UTILS:
        data = load_financial_data(TICKER, START_DATE, END_DATE)
        close_prices = data['Close']
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        close_prices = data['Close']
    
    # Follow original code order exactly:
    # 1. Add frequency information to date index
    close_prices = close_prices.asfreq('B')  # 'B' stands for business day frequency
    
    # 2. Check and remove NaN values
    close_prices = close_prices.dropna()
    
    # 3. Create log difference data of closing prices
    log_diff = np.log(close_prices).diff().dropna()
    
    # 4. Plot ACF (Autocorrelation Function) - BEFORE asfreq('B')
    # ACF helps identify MA order: significant spikes beyond confidence interval
    # indicate MA terms
    plt.figure(figsize=(10, 6))
    plot_acf(log_diff, lags=30, ax=plt.gca())
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation Function (ACF) for Log-Differenced {TICKER} Prices')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 5. Add frequency information AFTER plotting (as in original code)
    log_diff = log_diff.asfreq('B')
    
    # 6. Fit ARIMA model based on MA order
    # ARIMA(0, 0, q) is equivalent to MA(q) model
    print(f"\nFitting MA({MA_ORDER}) model (ARIMA(0, 0, {MA_ORDER}))...")
    model = sm.tsa.ARIMA(log_diff, order=(0, 0, MA_ORDER))
    results = model.fit()
    
    # Print model summary
    print("\n" + "="*60)
    print("MA Model Summary")
    print("="*60)
    print(results.summary())


if __name__ == "__main__":
    main()
