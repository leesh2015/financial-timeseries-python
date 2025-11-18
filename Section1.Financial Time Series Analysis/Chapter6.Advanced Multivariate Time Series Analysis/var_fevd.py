"""
VAR Forecast Error Variance Decomposition (FEVD)

This script calculates and visualizes FEVD from a VAR model to analyze
the proportion of forecast error variance in each variable explained by
shocks to other variables.

Forecast Error Variance Decomposition:
FEVD decomposes the forecast error variance of each variable into
contributions from shocks to all variables in the system.

Mathematical formulation:
For VAR(p): Y_t = A₁Y_{t-1} + ... + AₚY_{t-p} + ε_t

The h-step ahead forecast error is:
e_{t+h} = Y_{t+h} - E[Y_{t+h} | I_t]

FEVD measures the proportion of variance in e_{t+h} explained by
shocks to each variable:

FEVD_{ij}(h) = Proportion of variance in variable i's forecast error
               at horizon h explained by shocks to variable j

FEVD is order-dependent when using Cholesky decomposition (orthogonalized).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import sys
import os
import warnings
import matplotlib.pyplot as plt

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
TARGET_VAR = 'SPY'
YEARS_BACK = 10
INTERVAL = '1d'
MAX_LAGS = 10
FEVD_HORIZON = 10
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
    results = model.fit(maxlags=MAX_LAGS, ic=IC_CRITERION)
    print(f"Selected lag order: {results.k_ar}")
    
    # Calculate FEVD
    print(f"\n{'='*60}")
    print(f"Step 3: Calculating FEVD (horizon={FEVD_HORIZON})")
    print("="*60)
    fevd = results.fevd(FEVD_HORIZON)
    
    # Plot 1: FEVD for target variable (SPY)
    if TARGET_VAR in df.columns:
        print(f"\nGenerating FEVD plot for {TARGET_VAR}...")
        target_index = df.columns.get_loc(TARGET_VAR)
        fevd_values = fevd.decomp[:, target_index, :]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        days = np.arange(1, fevd_values.shape[0] + 1)
        for i, var in enumerate(df.columns):
            ax.plot(days, fevd_values[:, i], label=var, linewidth=2)
        ax.set_title(f'Forecast Error Variance Decomposition for {TARGET_VAR}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Days Ahead', fontsize=12)
        ax.set_ylabel('Variance Proportion', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        # Save to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, f'fevd_{TARGET_VAR.lower()}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.show()
    
    # Plot 2: FEVD for all variables (stacked bar chart)
    print(f"\nGenerating FEVD stacked bar charts for all variables...")
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    
    fig, axes = plt.subplots(nrows=len(fevd.names), ncols=1, figsize=(10, 2 * len(fevd.names)), dpi=80)
    fig.subplots_adjust(hspace=0.5)
    
    for i, (name, ax) in enumerate(zip(fevd.names, axes)):
        data = fevd.decomp[:, i, :].T
        
        bottom = np.zeros(data.shape[1])
        for j, color in enumerate(colors[:len(fevd.names)]):
            if j < data.shape[0]:
                ax.bar(range(data.shape[1]), data[j], bottom=bottom, color=color, label=fevd.names[j])
                bottom += data[j]
        
        ax.set_title(f'FEVD for {name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Days Ahead', fontsize=10)
        ax.set_ylabel('Variance Proportion', fontsize=10)
        ax.set_xlim(-0.5, data.shape[1] - 0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    handles, labels = ax.get_legend_handles_labels()
    patches = [plt.plot([],[], marker="s", ls="", color=colors[i], markersize=10)[0] 
               for i in range(len(fevd.names))]
    fig.legend(patches, labels, loc='upper right', ncol=2, fontsize=9)
    # Save to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'fevd_all_variables.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()
    
    print(f"\n{'='*60}")
    print("FEVD Analysis Complete")
    print("="*60)
    print("Note: FEVD shows the proportion of forecast error variance explained by each variable.")
    print("      Higher values indicate stronger influence on the target variable.")


if __name__ == "__main__":
    main()
