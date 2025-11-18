"""
Cholesky Decomposition for VAR Models

This script demonstrates Cholesky decomposition of the error covariance matrix
from a VAR model, which is used for orthogonalizing shocks in IRF and FEVD analysis.

Cholesky Decomposition:
Decomposes a positive definite covariance matrix Σ into:
Σ = LL'

where:
- L: lower triangular matrix
- L': transpose of L (upper triangular)

This decomposition is used to:
1. Orthogonalize shocks (eliminate contemporaneous correlation)
2. Identify structural shocks in VAR models
3. Calculate orthogonalized IRF and FEVD

The decomposition is order-dependent: different variable orderings
produce different results.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
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
TICKERS = ['CL=F', 'RB=F', 'ZB=F', 'YM=F', 'NQ=F', 'DX-Y.NYB', '^TNX', 'BZ=F', '^VIX', 'SPY']
YEARS_BACK = 10
INTERVAL = '1d'
MAX_LAGS = 5
IC_CRITERION = 'aic'


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
        return data.diff().dropna()
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
        df = df.dropna()
    else:
        df = yf.download(TICKERS, start=start_date, end=end_date, interval=INTERVAL, progress=False)['Close']
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.asfreq('D')
        df = df.dropna()
    
    # Stationarity test and differencing
    print(f"{'='*60}")
    print("Step 1: Preparing Data (Differencing for Stationarity)")
    print("="*60)
    df_diff = df.apply(make_stationary).dropna()
    print(f"Data shape after differencing: {df_diff.shape}")
    
    # Fit VAR model
    print(f"\n{'='*60}")
    print(f"Step 2: Fitting VAR Model (max_lags={MAX_LAGS}, ic={IC_CRITERION})")
    print("="*60)
    model = VAR(df_diff)
    fitted_model = model.fit(maxlags=MAX_LAGS, ic=IC_CRITERION)
    print(f"Selected lag order: {fitted_model.k_ar}")
    
    # Cholesky decomposition of the covariance matrix of errors
    print(f"\n{'='*60}")
    print("Step 3: Cholesky Decomposition")
    print("="*60)
    cov_matrix = fitted_model.sigma_u
    L = np.linalg.cholesky(cov_matrix)
    
    print("\nCovariance matrix of errors (Σ):")
    print(f"Shape: {cov_matrix.shape}")
    print(cov_matrix)
    
    print("\nCholesky decomposition matrix (L):")
    print(f"Shape: {L.shape}")
    print(L)
    
    # Verify decomposition: L @ L' should equal Σ
    print("\nVerification: L @ L' should equal Σ")
    reconstructed = L @ L.T
    print(f"Max difference: {np.max(np.abs(cov_matrix - reconstructed)):.2e}")
    print("Decomposition verified" if np.allclose(cov_matrix, reconstructed) else "Decomposition error")
    
    # Display decomposed errors (orthogonalized)
    print(f"\n{'='*60}")
    print("Step 4: Orthogonalized Errors")
    print("="*60)
    print("Decomposed errors (orthogonalized): ε_orth = ε @ L⁻¹")
    errors_orth = fitted_model.resid @ np.linalg.inv(L)
    print(f"Shape: {errors_orth.shape}")
    print(f"Sample (first 5 rows):\n{errors_orth[:5]}")
    print(f"\nCovariance of orthogonalized errors (should be identity):")
    cov_orth = np.cov(errors_orth.T)
    print(cov_orth)
    
    print(f"\n{'='*60}")
    print("Interpretation:")
    print("="*60)
    print("  - Cholesky decomposition orthogonalizes the error terms")
    print("  - This is used in orthogonalized IRF and FEVD analysis")
    print("  - The decomposition is order-dependent (depends on variable ordering)")


if __name__ == "__main__":
    main()


