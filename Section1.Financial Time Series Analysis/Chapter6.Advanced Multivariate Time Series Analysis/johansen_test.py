"""
Johansen Cointegration Test

This script performs the Johansen cointegration test to identify long-term
equilibrium relationships between multiple non-stationary time series.

Johansen Test:
Tests for cointegration rank (number of cointegrating relationships) in
a system of non-stationary variables.

Mathematical formulation:
For a VAR(p) model: Y_t = A₁Y_{t-1} + ... + AₚY_{t-p} + ε_t

The VECM representation is:
ΔY_t = ΠY_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + ε_t

where:
- Π = αβ' (reduced rank matrix)
- α: adjustment speed matrix (error correction coefficients)
- β: cointegrating vectors (long-run relationships)
- rank(Π) = r: number of cointegrating relationships

The Johansen test uses:
- Trace test: H₀: rank ≤ r vs H₁: rank > r
- Max eigenvalue test: H₀: rank = r vs H₁: rank = r+1

If test statistic > critical value, reject H₀ (cointegration exists).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VECM
from statsmodels.tsa.stattools import grangercausalitytests
from arch import arch_model
import warnings
from datetime import datetime, timedelta
import sys
import os

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
YEARS_BACK = 10
TICKERS = ['CL=F', 'YM=F', 'NQ=F', 'DX-Y.NYB', 'GC=F', 'NG=F', '^VIX', 'SPY']
TARGET_INDEX = "SPY"
MAX_LAGS = 15
DETERMINISTIC = "colo"  # Constant term and linear trend
DET_ORDER = 1
METHOD = 'maxeig'  # 'maxeig' or 'trace'


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
    
    # Find the optimal lag order
    from statsmodels.tsa.vector_ar.vecm import select_order
    print(f"{'='*60}")
    print("Step 1: Selecting Optimal Lag Order")
    print("="*60)
    lag_order = select_order(df, maxlags=MAX_LAGS, deterministic=DETERMINISTIC)
    time_lag = lag_order.aic
    print(f"Optimal lag order (AIC): {time_lag}")
    print(f"  AIC: {lag_order.aic}")
    print(f"  BIC: {lag_order.bic}")
    
    # Find the cointegration rank using the Johansen test
    from statsmodels.tsa.vector_ar.vecm import select_coint_rank
    print(f"\n{'='*60}")
    print(f"Step 2: Johansen Cointegration Test ({METHOD.upper()} Method)")
    print("="*60)
    coint_rank_test = select_coint_rank(df, det_order=DET_ORDER, k_ar_diff=time_lag, method=METHOD)
    
    # Print the results of the analysis
    print(f"\nOptimal Cointegration Rank: {coint_rank_test.rank}")
    print(f"\nTest Statistics:")
    
    # Access test statistics if available
    if hasattr(coint_rank_test, 'test_statistic'):
        print(f"Test Statistic: {coint_rank_test.test_statistic}")
    if hasattr(coint_rank_test, 'critical_values'):
        print(f"Critical Values:\n{coint_rank_test.critical_values}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("Interpretation:")
    print("="*60)
    print(f"  - Cointegration rank: {coint_rank_test.rank}")
    print(f"  - This means there are {coint_rank_test.rank} long-term equilibrium relationships")
    print(f"    among the {len(TICKERS)} variables.")
    if coint_rank_test.rank > 0:
        print(f"  - The variables are cointegrated and can be modeled using VECM.")
    else:
        print(f"  - No cointegration found. Consider using VAR in differences instead.")


if __name__ == "__main__":
    main()
