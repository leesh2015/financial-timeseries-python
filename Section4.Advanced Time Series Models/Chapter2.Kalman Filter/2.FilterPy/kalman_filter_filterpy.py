"""
Chapter 2: Kalman Filter - FilterPy Implementation

This script demonstrates Kalman filtering using filterpy library for:
1. Price trend estimation and noise removal
2. Time-varying beta estimation
3. Dynamic tracking of index-ETF relationship
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_loader import load_nasdaq_tqqq_data, align_data

warnings.filterwarnings("ignore")

try:
    from filterpy.kalman import KalmanFilter as FilterPyKalman
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    print("Warning: filterpy not available. Install with: pip install filterpy")


def kalman_filter_price_filterpy(prices, use_adaptive=True):
    """
    Apply Kalman filter to price series for trend estimation using filterpy
    WITH FilterPy's advanced features: adaptive noise estimation and numerical stability
    
    State: [price, velocity]
    Observation: price
    
    Parameters:
    -----------
    use_adaptive : bool
        If True, use adaptive noise estimation (FilterPy's advantage)
        Adapts Q and R based on prediction errors
    """
    if not HAS_FILTERPY:
        raise ImportError("filterpy is required for this function")
    
    kf = FilterPyKalman(dim_x=2, dim_z=1)
    
    # Initial state: [price, 0]
    kf.x = np.array([[prices.iloc[0]], [0.0]])
    
    # State transition: constant velocity model
    kf.F = np.array([[1., 1.],
                     [0., 1.]])
    
    # Observation matrix: observe price only
    kf.H = np.array([[1., 0.]])
    
    # Process noise (small for smooth trend)
    kf.Q = np.array([[0.1, 0.],
                     [0., 0.1]])
    
    # Observation noise (price volatility)
    kf.R = np.array([[prices.std()**2]])
    
    # Initial covariance (FilterPy's optimized initialization)
    kf.P = np.eye(2) * 1000.
    
    # FilterPy's unique features: adaptive noise estimation
    if use_adaptive:
        print("  Using adaptive noise estimation (FilterPy feature)...")
        # Track prediction errors for adaptive Q and R
        prediction_errors = []
        innovation_errors = []
        
        # Initial adaptive parameters
        alpha_q = 0.95  # Smoothing factor for Q adaptation
        alpha_r = 0.95  # Smoothing factor for R adaptation
        base_q = kf.Q.copy()
        base_r = kf.R.copy()
    
    filtered_prices = []
    filtered_velocities = []
    
    for i, price in enumerate(prices):
        # Predict step
        kf.predict()
        
        # Store prediction for adaptive estimation
        if use_adaptive and i > 0:
            pred_error = abs(price - kf.x[0, 0])
            prediction_errors.append(pred_error)
            
            # Adapt Q based on prediction errors (FilterPy's adaptive feature)
            if len(prediction_errors) > 10:
                recent_errors = np.array(prediction_errors[-10:])
                adaptive_q_scale = np.mean(recent_errors) / prices.std()
                kf.Q = base_q * (1 + alpha_q * adaptive_q_scale)
        
        # Update step
        # FilterPy's update doesn't return innovation, so we calculate it manually
        if use_adaptive and i > 0:
            # Calculate innovation: z - H * x (predicted observation - actual observation)
            predicted_observation = kf.H @ kf.x
            innovation_value = price - predicted_observation[0, 0]
            innovation_errors.append(abs(innovation_value))
        
        kf.update(np.array([[price]]))
        
        if use_adaptive and i > 0:
            # Adapt R based on innovation errors
            if len(innovation_errors) > 10:
                recent_innovations = np.array(innovation_errors[-10:])
                adaptive_r_scale = np.mean(recent_innovations) / prices.std()
                kf.R = base_r * (1 + alpha_r * adaptive_r_scale)
        
        filtered_prices.append(kf.x[0, 0])
        filtered_velocities.append(kf.x[1, 0])
    
    if use_adaptive:
        print(f"  Final adaptive Q:\n{kf.Q}")
        print(f"  Final adaptive R:\n{kf.R}")
    
    return pd.Series(filtered_prices, index=prices.index), \
           pd.Series(filtered_velocities, index=prices.index)


def kalman_filter_beta_filterpy(nasdaq_returns, tqqq_returns):
    """
    Estimate time-varying beta using Kalman filter (filterpy)
    
    State: [alpha, beta]
    Observation: TQQQ return = alpha + beta * NASDAQ return + noise
    """
    if not HAS_FILTERPY:
        raise ImportError("filterpy is required for this function")
    
    # Align data
    nasdaq_ret, tqqq_ret = align_data(nasdaq_returns, tqqq_returns)
    
    kf = FilterPyKalman(dim_x=2, dim_z=1)
    
    # Initial state: [alpha, beta]
    # Beta should be around 3 for TQQQ (3x leveraged)
    kf.x = np.array([[0.0], [3.0]])
    
    # State transition: random walk
    kf.F = np.eye(2)
    
    # Process noise
    kf.Q = np.array([[0.0001, 0.],
                     [0., 0.01]])
    
    # Observation noise
    kf.R = np.array([[tqqq_ret.std()**2]])
    
    # Initial covariance
    kf.P *= 1000.
    
    filtered_alpha = []
    filtered_beta = []
    
    for i in range(len(nasdaq_ret)):
        # Update observation matrix with current NASDAQ return
        kf.H = np.array([[1., nasdaq_ret.iloc[i]]])
        
        kf.predict()
        kf.update(np.array([[tqqq_ret.iloc[i]]]))
        
        filtered_alpha.append(kf.x[0, 0])
        filtered_beta.append(kf.x[1, 0])
    
    return pd.Series(filtered_alpha, index=nasdaq_ret.index), \
           pd.Series(filtered_beta, index=nasdaq_ret.index)


def visualize_kalman_results(data, filtered_prices_nasdaq, filtered_prices_tqqq,
                           filtered_beta, filtered_alpha):
    """Visualize Kalman filter results"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Plot 1: NASDAQ price and filtered trend
    axes[0].plot(data['nasdaq'].index, data['nasdaq']['Close'].values, 
                 label='NASDAQ Actual', alpha=0.5, linewidth=1)
    axes[0].plot(filtered_prices_nasdaq.index, filtered_prices_nasdaq.values, 
                 label='NASDAQ Filtered (FilterPy)', linewidth=2, color='blue')
    axes[0].set_title('NASDAQ Index: Actual vs Kalman Filtered (FilterPy)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: TQQQ price and filtered trend
    axes[1].plot(data['tqqq'].index, data['tqqq']['Close'].values, 
                 label='TQQQ Actual', alpha=0.5, linewidth=1, color='orange')
    axes[1].plot(filtered_prices_tqqq.index, filtered_prices_tqqq.values, 
                 label='TQQQ Filtered (FilterPy)', linewidth=2, color='red')
    axes[1].set_title('TQQQ ETF: Actual vs Kalman Filtered (FilterPy)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Price', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Time-varying beta
    axes[2].plot(filtered_beta.index, filtered_beta.values, 
                 label='Time-varying Beta (FilterPy)', linewidth=2, color='green')
    axes[2].axhline(y=3.0, color='r', linestyle='--', label='Theoretical Beta (3x)')
    axes[2].set_title('Time-Varying Beta: FilterPy Kalman Filter Estimate', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Beta', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Alpha (tracking error component)
    axes[3].plot(filtered_alpha.index, filtered_alpha.values, 
                 label='Alpha (FilterPy)', linewidth=2, color='purple')
    axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[3].set_title('Alpha (Excess Return Component)', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Date', fontsize=12)
    axes[3].set_ylabel('Alpha', fontsize=12)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("FilterPy Kalman Filter Statistics")
    print(f"{'='*60}")
    print(f"Beta Statistics:")
    print(f"  Mean: {filtered_beta.mean():.4f}")
    print(f"  Std: {filtered_beta.std():.4f}")
    print(f"  Min: {filtered_beta.min():.4f}")
    print(f"  Max: {filtered_beta.max():.4f}")
    print(f"\nAlpha Statistics:")
    print(f"  Mean: {filtered_alpha.mean():.6f}")
    print(f"  Std: {filtered_alpha.std():.6f}")


def main():
    """Main function"""
    if not HAS_FILTERPY:
        print("Error: filterpy is required. Install with: pip install filterpy")
        return
    
    print("="*60)
    print("Chapter 2: Kalman Filter - FilterPy Implementation")
    print("Index and Leveraged ETF Analysis")
    print("="*60)
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    nasdaq = data['nasdaq']
    tqqq = data['tqqq']
    
    # 1. Kalman filter for price trend estimation (with adaptive noise)
    print("\n1. Applying FilterPy Kalman filter to price series (with adaptive noise)...")
    filtered_prices_nasdaq, _ = kalman_filter_price_filterpy(nasdaq['Close'], use_adaptive=True)
    filtered_prices_tqqq, _ = kalman_filter_price_filterpy(tqqq['Close'], use_adaptive=True)
    
    # 2. Kalman filter for time-varying beta
    print("2. Estimating time-varying beta using FilterPy Kalman filter...")
    filtered_alpha, filtered_beta = kalman_filter_beta_filterpy(
        nasdaq['Returns'], tqqq['Returns']
    )
    
    # Visualize
    visualize_kalman_results(data, filtered_prices_nasdaq, filtered_prices_tqqq,
                            filtered_beta, filtered_alpha)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

