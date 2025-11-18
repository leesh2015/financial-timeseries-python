"""
Ljung-Box Test for Residual Autocorrelation

This script demonstrates the Ljung-Box test, which tests whether the residuals
from a time series model exhibit autocorrelation.

Null hypothesis: No autocorrelation in residuals (residuals are white noise)
Alternative hypothesis: Autocorrelation exists in residuals
"""

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
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
TICKER = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2021-01-01'
TEST_LAGS = [10]  # Lags to test
SIGNIFICANCE_LEVEL = 0.05


def perform_ljung_box_test(data, lags=[10]):
    """
    Perform Ljung-Box test for autocorrelation.
    
    Parameters:
    -----------
    data : pd.Series or np.ndarray
        Time series data (typically residuals)
    lags : list
        List of lag values to test
        
    Returns:
    --------
    dict
        Test results dictionary
    """
    # Perform Ljung-Box test (following original code exactly)
    lb_test = acorr_ljungbox(data, lags=lags, return_df=True)
    print(lb_test)
    
    # Interpret results (following original code exactly)
    p_value = lb_test['lb_pvalue'].iloc[-1]
    print(f"P-value: {p_value}")
    
    if p_value < 0.05:
        print("Autocorrelation exists. Use an ARIMA model.")
    else:
        print("No autocorrelation exists. Use a Moving Average model.")
    
    return {
        'test_statistic': lb_test['lb_stat'].iloc[-1],
        'p_value': p_value,
        'has_autocorrelation': p_value < 0.05
    }


def main():
    """Main function"""
    # Load data
    if USE_UTILS:
        data = load_financial_data(TICKER, START_DATE, END_DATE)
        data = preprocess_data(data, frequency='D', use_adjusted=False)
        close_prices = data['Close']
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        close_prices = data['Close']
    
    # Calculate log difference (stationary transformation)
    log_diff = np.log(close_prices).diff().dropna()
    
    # Perform Ljung-Box test (following original code exactly)
    results = perform_ljung_box_test(log_diff, lags=TEST_LAGS)


if __name__ == "__main__":
    main()
