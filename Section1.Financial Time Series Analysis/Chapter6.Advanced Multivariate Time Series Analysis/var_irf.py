"""
VAR Impulse Response Function (IRF)

This script calculates and visualizes impulse response functions from a VAR model
to analyze how shocks to one variable affect other variables over time.

Impulse Response Function:
IRF measures the response of each variable to a one-time shock (impulse) in
another variable, holding all other shocks constant.

Mathematical formulation:
For VAR(p): Y_t = A₁Y_{t-1} + ... + AₚY_{t-p} + ε_t

The IRF at horizon h is:
IRF(h) = ∂Y_{t+h} / ∂ε_t

Orthogonalized IRF (Cholesky decomposition):
- Uses Cholesky decomposition: Σ = LL'
- L: lower triangular matrix
- Orthogonalizes shocks to eliminate contemporaneous correlation
- Order-dependent (depends on variable ordering)

Non-orthogonalized IRF:
- Direct response to structural shocks
- Order-independent
"""

import yfinance as yf
from statsmodels.tsa.api import VAR
from datetime import datetime, timedelta
import sys
import os
import matplotlib.pyplot as plt

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
MAX_LAGS = 5
IRF_HORIZON = 10
IC_CRITERION = 'aic'


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
        df = df.dropna()  # Remove NaN values
    
    # Difference the data (ensure stationarity)
    print(f"{'='*60}")
    print("Step 1: Preparing Data (Differencing for Stationarity)")
    print("="*60)
    df_diff = df.diff().dropna()
    print(f"Data shape after differencing: {df_diff.shape}")
    
    # Fit VAR model
    print(f"\n{'='*60}")
    print(f"Step 2: Fitting VAR Model (max_lags={MAX_LAGS}, ic={IC_CRITERION})")
    print("="*60)
    model = VAR(df_diff)
    results = model.fit(maxlags=MAX_LAGS, ic=IC_CRITERION)
    print(f"Selected lag order: {results.k_ar}")
    
    # Calculate impulse response function
    print(f"\n{'='*60}")
    print(f"Step 3: Calculating IRF (horizon={IRF_HORIZON})")
    print("="*60)
    irf = results.irf(IRF_HORIZON)
    
    # Plot and save orthogonalized impulse response function
    print("\nGenerating orthogonalized IRF plot...")
    fig_orth = irf.plot(orth=True, figsize=(19.2, 10.8))
    # Save to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_orth = os.path.join(script_dir, 'orthogonalized_irf.png')
    fig_orth.savefig(output_file_orth, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file_orth}")
    
    # Plot and save non-orthogonalized impulse response function
    print("\nGenerating non-orthogonalized IRF plot...")
    fig_non_orth = irf.plot(orth=False, figsize=(19.2, 10.8))
    output_file_non_orth = os.path.join(script_dir, 'non_orthogonalized_irf.png')
    fig_non_orth.savefig(output_file_non_orth, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file_non_orth}")
    
    print(f"\n{'='*60}")
    print("IRF Analysis Complete")
    print("="*60)
    print("Note: Orthogonalized IRF uses Cholesky decomposition and is order-dependent.")
    print("      Non-orthogonalized IRF shows direct responses and is order-independent.")
    
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    main()
