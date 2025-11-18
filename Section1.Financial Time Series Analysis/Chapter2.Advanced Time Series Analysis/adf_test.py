"""
Augmented Dickey-Fuller (ADF) Test for Stationarity

This script demonstrates how to test for stationarity using the ADF test.
It downloads AAPL stock prices, calculates log differences, and performs
the ADF test to determine if the differenced series is stationary.

The null hypothesis: The time series has a unit root (non-stationary)
The alternative hypothesis: The time series is stationary
"""

import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from utils.data_loader import load_financial_data, preprocess_data
    from utils.config import config
    USE_UTILS = True
except ImportError:
    USE_UTILS = False

# Configuration
TICKER = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2021-01-01'
SIGNIFICANCE_LEVEL = 0.05


def perform_adf_test(data, title="Data"):
    """
    Perform ADF test and print results.
    
    Parameters:
    -----------
    data : pd.Series or np.ndarray
        Time series data to test
    title : str
        Title for the test results
    """
    # Perform ADF test
    adf_result = adfuller(data)
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    # Print results
    print(f'\n{"="*60}')
    print(f'ADF Test Results: {title}')
    print(f'{"="*60}')
    print(f'ADF Statistic: {adf_stat:.6f}')
    print(f'p-value: {p_value:.6f}')
    print(f'\nCritical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value:.6f}')
    
    # Interpret ADF test result
    print(f'\nInterpretation:')
    if adf_stat < critical_values['1%']:
        print('The data is stationary at the 1% significance level.')
        is_stationary = True
    elif adf_stat < critical_values['5%']:
        print('The data is stationary at the 5% significance level.')
        is_stationary = True
    elif adf_stat < critical_values['10%']:
        print('The data is stationary at the 10% significance level.')
        is_stationary = True
    else:
        print('The data is NOT stationary.')
        is_stationary = False
    
    # P-value interpretation
    if p_value < SIGNIFICANCE_LEVEL:
        print(f'p-value ({p_value:.6f}) < {SIGNIFICANCE_LEVEL}. Reject null hypothesis. Data is stationary.')
    else:
        print(f'p-value ({p_value:.6f}) >= {SIGNIFICANCE_LEVEL}. Cannot reject null hypothesis. Data is NOT stationary.')
    
    return adf_stat, p_value, critical_values, is_stationary


def main():
    """Main function"""
    # Load data
    if USE_UTILS:
        data = load_financial_data(TICKER, START_DATE, END_DATE)
        data = preprocess_data(data, frequency='D')
        close_prices = data['Close']
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE)
        close_prices = data['Close']
    
    # Calculate log difference (first difference of log prices)
    log_diff = np.log(close_prices).diff().dropna()
    
    # Perform ADF test on log-differenced data
    perform_adf_test(log_diff, "Log-Differenced AAPL Prices")


if __name__ == "__main__":
    main()
