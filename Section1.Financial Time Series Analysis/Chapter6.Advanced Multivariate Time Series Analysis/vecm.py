"""
VECM (Vector Error Correction Model) Implementation

This script fits a VECM model to multivariate time series data with cointegration.

VECM Model:
For cointegrated variables, VECM captures both short-term dynamics and
long-term equilibrium relationships.

Mathematical formulation:
ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + ε_t

where:
- ΔY_t: first differences of variables (stationary)
- α: adjustment speed matrix (error correction coefficients)
- β: cointegrating vectors (long-run equilibrium relationships)
- β'Y_{t-1}: error correction term (ECT) - deviation from equilibrium
- Γᵢ: short-term dynamics coefficients
- ε_t: error terms

Key features:
- α < 0: mean-reverting (converges to equilibrium)
- α > 0: divergent (moves away from equilibrium)
- |α|: speed of adjustment to equilibrium
"""

import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VECM
from datetime import datetime, timedelta
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
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
TICKERS = ['CL=F', 'YM=F', 'NQ=F', 'DX-Y.NYB', 'GC=F', 'NG=F', '^VIX', 'SPY']
YEARS_BACK = 10
INTERVAL = '1d'
MAX_LAGS = 15
DETERMINISTIC = "colo"  # Constant term and linear trend
DET_ORDER = 1
METHOD = "maxeig"
OUTPUT_FILE = "vecm_summary.txt"


def main():
    """Main function"""
    # Set dates
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*YEARS_BACK)).strftime('%Y-%m-%d')
    
    # Download historical data
    if USE_UTILS:
        df = load_financial_data(TICKERS, start_date, end_date, interval=INTERVAL, progress=False)
        df = preprocess_data(df, frequency='D', use_adjusted=True)
        endog = df['Adj Close'] if isinstance(df.columns, pd.MultiIndex) else df
    else:
        df = yf.download(TICKERS, start=start_date, end=end_date, interval=INTERVAL, progress=False)
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.asfreq('D')
        df = df[(df >= 0).all(axis=1)]
        df = df.dropna()
        endog = df['Adj Close']  # Use adjusted close prices
    
    # Set up VECM model
    print(f"{'='*60}")
    print("Step 1: Selecting Optimal Lag Order")
    print("="*60)
    vecm_order = select_order(endog, maxlags=MAX_LAGS, deterministic=DETERMINISTIC)
    time_lag = vecm_order.aic
    print(f"Optimal lag order (AIC): {time_lag}")
    
    print(f"\n{'='*60}")
    print("Step 2: Determining Cointegration Rank")
    print("="*60)
    coint_rank = select_coint_rank(endog, det_order=DET_ORDER, k_ar_diff=time_lag, method=METHOD)
    print(f"Optimal cointegration rank: {coint_rank.rank}")
    
    # Fit VECM model
    print(f"\n{'='*60}")
    print("Step 3: Fitting VECM Model")
    print("="*60)
    vecm = VECM(endog, k_ar_diff=time_lag, coint_rank=coint_rank.rank, deterministic=DETERMINISTIC)
    vecm_fit = vecm.fit()
    
    # Summarize VECM model
    vecm_summary = vecm_fit.summary()
    summary_text = vecm_summary.as_text()
    
    # Save summary results to text file
    with open(OUTPUT_FILE, "w", encoding='utf-8') as file:
        file.write(summary_text)
    
    print(f"\n{'='*60}")
    print("VECM Model Summary")
    print("="*60)
    print(vecm_summary)
    print(f"\nVECM summary results have been saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
