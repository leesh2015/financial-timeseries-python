"""
Chapter 1: State-Space Models

This script demonstrates state-space modeling for financial time series,
focusing on index and leveraged ETF relationship tracking.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_nasdaq_tqqq_data, align_data
import warnings

warnings.filterwarnings("ignore")


def state_space_tracking_error(nasdaq_returns, tqqq_returns):
    """
    Model TQQQ returns as a state-space model tracking NASDAQ
    
    State equation: beta_t = beta_{t-1} + w_t  (time-varying beta)
    Observation equation: r_tqqq_t = alpha + beta_t * r_nasdaq_t + v_t
    
    Parameters:
    -----------
    nasdaq_returns : pd.Series
        NASDAQ returns
    tqqq_returns : pd.Series
        TQQQ returns
    
    Returns:
    --------
    dict
        Estimated parameters and state estimates
    """
    # Align data
    nasdaq_ret, tqqq_ret = align_data(nasdaq_returns, tqqq_returns)
    
    # Simple OLS to get initial estimates
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    X = nasdaq_ret.values.reshape(-1, 1)
    y = tqqq_ret.values
    lr.fit(X, y)
    
    alpha_init = lr.intercept_
    beta_init = lr.coef_[0]
    
    print(f"Initial OLS estimates:")
    print(f"  Alpha: {alpha_init:.6f}")
    print(f"  Beta: {beta_init:.4f}")
    print(f"  R-squared: {lr.score(X, y):.4f}")
    
    # State-space model: Simple rolling window approach
    # Beta is estimated using rolling window
    window = 60  # 60-day rolling window
    rolling_beta = []
    rolling_alpha = []
    
    for i in range(window, len(nasdaq_ret)):
        X_window = nasdaq_ret.iloc[i-window:i].values.reshape(-1, 1)
        y_window = tqqq_ret.iloc[i-window:i].values
        
        lr_window = LinearRegression()
        lr_window.fit(X_window, y_window)
        
        rolling_alpha.append(lr_window.intercept_)
        rolling_beta.append(lr_window.coef_[0])
    
    rolling_beta = pd.Series(rolling_beta, index=nasdaq_ret.index[window:])
    rolling_alpha = pd.Series(rolling_alpha, index=nasdaq_ret.index[window:])
    
    # Calculate tracking error
    predicted_returns = rolling_alpha + rolling_beta * nasdaq_ret.iloc[window:]
    tracking_error = tqqq_ret.iloc[window:] - predicted_returns
    
    return {
        'rolling_beta': rolling_beta,
        'rolling_alpha': rolling_alpha,
        'tracking_error': tracking_error,
        'predicted_returns': predicted_returns,
        'actual_returns': tqqq_ret.iloc[window:],
        'nasdaq_returns': nasdaq_ret.iloc[window:]
    }


def visualize_state_space_results(results):
    """Visualize state-space model results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Time-varying beta
    axes[0].plot(results['rolling_beta'].index, results['rolling_beta'].values, 
                 label='Time-varying Beta', linewidth=2, color='blue')
    axes[0].axhline(y=3.0, color='r', linestyle='--', label='Theoretical Beta (3x)')
    axes[0].set_title('Time-Varying Beta: TQQQ vs NASDAQ', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Beta', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted Returns
    axes[1].plot(results['actual_returns'].index, results['actual_returns'].values, 
                 label='Actual TQQQ Returns', alpha=0.7, linewidth=1.5)
    axes[1].plot(results['predicted_returns'].index, results['predicted_returns'].values, 
                 label='Predicted Returns (State-Space)', alpha=0.7, linewidth=1.5, linestyle='--')
    axes[1].set_title('Actual vs Predicted TQQQ Returns', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Returns', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Tracking Error
    axes[2].plot(results['tracking_error'].index, results['tracking_error'].values, 
                 label='Tracking Error', linewidth=1.5, color='red', alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].fill_between(results['tracking_error'].index, 
                        results['tracking_error'].values, 0, 
                        alpha=0.3, color='red')
    axes[2].set_title('Tracking Error (Actual - Predicted)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylabel('Tracking Error', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("State-Space Model Statistics")
    print(f"{'='*60}")
    print(f"Beta Statistics:")
    print(f"  Mean: {results['rolling_beta'].mean():.4f}")
    print(f"  Std: {results['rolling_beta'].std():.4f}")
    print(f"  Min: {results['rolling_beta'].min():.4f}")
    print(f"  Max: {results['rolling_beta'].max():.4f}")
    print(f"\nTracking Error Statistics:")
    print(f"  Mean: {results['tracking_error'].mean():.6f}")
    print(f"  Std: {results['tracking_error'].std():.6f}")
    print(f"  RMSE: {np.sqrt((results['tracking_error']**2).mean()):.6f}")


def main():
    """Main function"""
    print("="*60)
    print("Chapter 1: State-Space Models")
    print("Index and Leveraged ETF Analysis")
    print("="*60)
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    nasdaq = data['nasdaq']
    tqqq = data['tqqq']
    
    # Use returns
    nasdaq_returns = nasdaq['Returns']
    tqqq_returns = tqqq['Returns']
    
    # State-space model
    results = state_space_tracking_error(nasdaq_returns, tqqq_returns)
    
    # Visualize
    visualize_state_space_results(results)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

