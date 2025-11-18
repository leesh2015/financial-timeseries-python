"""
Granger Causality Analysis

This script performs Granger causality tests to identify causal relationships
between multiple financial time series variables.

Granger Causality Test:
Tests whether past values of variable X help predict variable Y beyond what
past values of Y alone can predict.

Mathematical formulation:
- Null hypothesis H₀: X does NOT Granger-cause Y
- Alternative hypothesis H₁: X Granger-causes Y

The test uses F-statistic to compare:
- Restricted model: Y_t = α + ΣβᵢY_{t-i} + ε_t
- Unrestricted model: Y_t = α + ΣβᵢY_{t-i} + ΣγⱼX_{t-j} + ε_t

If F-statistic is significant (p < 0.05), we reject H₀ and conclude that
X Granger-causes Y.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from datetime import datetime, timedelta
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

# Configuration
INTERVAL = '1d'
YEARS_BACK = 5
TICKERS = ["CL=F", "HO=F", "RB=F", "ZB=F", "YM=F", "ZO=F", "SPY", "NQ=F", 
           "DX-Y.NYB", "^TNX", "GC=F", "BZ=F", "SI=F", "NG=F", "^VIX"]
TARGET_VAR = "SPY"
MAX_LAG = 10
SIGNIFICANCE_LEVEL = 0.05


def make_stationary(data):
    """
    Make time series stationary by differencing if necessary.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
        
    Returns:
    --------
    pd.Series
        Stationary time series
    """
    result = adfuller(data)
    if result[1] > 0.05:
        return data.diff().dropna()  # Remove NaN values after differencing
    return data


def main():
    """Main function"""
    # Set dates
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*YEARS_BACK)).strftime('%Y-%m-%d')
    
    # Download data
    if USE_UTILS:
        df = load_financial_data(TICKERS, start_date, end_date, interval=INTERVAL, progress=False)
        df = preprocess_data(df, frequency='D', use_adjusted=False)
        df = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
    else:
        df = yf.download(TICKERS, start=start_date, end=end_date, interval=INTERVAL, progress=False)['Close']
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.asfreq('D')
        df = df.dropna()

    # Stationarity test and differencing
    df_diff = df.apply(make_stationary).dropna()
    
    # Perform the ADF test again
    result_diff = adfuller(df_diff[TARGET_VAR])
    print('ADF Statistic after differencing:', result_diff[0])
    print('p-value after differencing:', result_diff[1])
    
    # Perform Granger causality tests among all variables
    print(f"\nPerforming Granger causality tests (max_lag={MAX_LAG})...")
    causality_results = {}
    
    for var in TICKERS:
        causality_results[var] = {}
        for target in TICKERS:
            if var != target:
                try:
                    granger_test_result = grangercausalitytests(df_diff[[target, var]], MAX_LAG, verbose=False)
                    p_values = [granger_test_result[i+1][0]['ssr_ftest'][1] for i in range(MAX_LAG)]
                    min_p_value = np.min(p_values)
                    causality_results[var][target] = min_p_value
                except Exception as e:
                    causality_results[var][target] = np.nan
    
    # Separate significant and insignificant relationships
    significant_causality = {(var, target): p_value for var, targets in causality_results.items() 
                             for target, p_value in targets.items() if p_value < SIGNIFICANCE_LEVEL}
    insignificant_causality = {(var, target): p_value for var, targets in causality_results.items() 
                               for target, p_value in targets.items() if p_value >= SIGNIFICANCE_LEVEL or np.isnan(p_value)}
    
    # Print significant relationships
    print(f"\n{'='*60}")
    print("Significant causality pairs (p < 0.05):")
    print("="*60)
    for (var, target), p_value in significant_causality.items():
        print(f"  {var} -> {target}: p-value = {p_value:.6f}")
    
    # Print insignificant relationships
    print(f"\n{'='*60}")
    print("Insignificant causality pairs (p >= 0.05):")
    print("="*60)
    print(f"  Total: {len(insignificant_causality)} pairs")
    
    # Select variables with significant influence on target variable
    significant_vars_for_target = {var: causality_results[var][TARGET_VAR] 
                                    for var in TICKERS 
                                    if var != TARGET_VAR and TARGET_VAR in causality_results.get(var, {}) 
                                    and not np.isnan(causality_results[var].get(TARGET_VAR, np.nan))
                                    and causality_results[var][TARGET_VAR] < SIGNIFICANCE_LEVEL}
    
    # Print the results
    print(f"\n{'='*60}")
    print(f"Significant variables for {TARGET_VAR} (variables that Granger-cause {TARGET_VAR}):")
    print("="*60)
    if significant_vars_for_target:
        for var, p_value in significant_vars_for_target.items():
            print(f"  {var} -> {TARGET_VAR}: p-value = {p_value:.6f}")
    else:
        print(f"  No significant Granger causality found at {SIGNIFICANCE_LEVEL*100}% significance level.")


if __name__ == "__main__":
    main()
