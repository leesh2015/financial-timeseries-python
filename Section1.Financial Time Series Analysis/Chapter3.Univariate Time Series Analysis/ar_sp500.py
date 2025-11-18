"""
AR (Autoregressive) Model Implementation

This script demonstrates how to fit an AR model using PACF (Partial Autocorrelation Function)
to determine the optimal AR order, then fits an ARIMA(p,0,0) model.

PACF helps identify the AR order: significant spikes indicate AR terms.
"""

import numpy as np
import statsmodels.api as sm
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
    from utils.config import config
    USE_UTILS = True
except ImportError:
    USE_UTILS = False
    import yfinance as yf

# Configuration
TICKER = 'SPY'
START_DATE = '2020-01-01'
END_DATE = '2023-01-01'
AR_ORDER = 1  # AR order (can be adjusted based on PACF plot)


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
    
    # 4. Plot PACF (Partial Autocorrelation Function) - BEFORE asfreq('B')
    # PACF helps identify AR order: significant spikes beyond confidence interval
    # indicate AR terms
    plt.figure(figsize=(10, 6))
    plot_pacf(log_diff, lags=30, ax=plt.gca())
    plt.xlabel('Lags')
    plt.ylabel('Partial Autocorrelation')
    plt.title(f'Partial Autocorrelation Function (PACF) for Log-Differenced {TICKER} Prices')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 5. Add frequency information AFTER plotting (as in original code)
    log_diff = log_diff.asfreq('B')
    
    # 6. Fit ARIMA model based on AR order
    # ARIMA(p, 0, 0) is equivalent to AR(p) model
    print(f"\nFitting AR({AR_ORDER}) model (ARIMA({AR_ORDER}, 0, 0))...")
    model = sm.tsa.ARIMA(log_diff, order=(AR_ORDER, 0, 0))
    results = model.fit()
    
    # Print model summary
    print("\n" + "="*60)
    print("AR Model Summary")
    print("="*60)
    print(results.summary())


if __name__ == "__main__":
    main()

