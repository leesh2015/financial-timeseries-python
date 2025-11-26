"""
Backtest for Chapter 2: Kalman Filter - FilterPy Implementation
Filtered trend-based trading strategy
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

from utils.data_loader import load_nasdaq_tqqq_data
from utils.backtest import BacktestEngine, walk_forward_backtest

warnings.filterwarnings("ignore")

try:
    from filterpy.kalman import KalmanFilter as FilterPyKalman
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    print("Warning: filterpy not available. Install with: pip install filterpy")


def generate_trading_signals_filterpy(prices, lookback=60, trend_threshold=0.001, use_adaptive=True):
    """
    Generate trading signals based on FilterPy Kalman filtered trend
    WITH adaptive noise estimation (FilterPy's unique feature)
    
    Strategy:
    - Buy when filtered trend turns positive and price is below filtered price
    - Sell when filtered trend turns negative or price is above filtered price
    - Hold otherwise
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    lookback : int
        Minimum data points needed
    trend_threshold : float
        Minimum trend strength for signal
    use_adaptive : bool
        If True, use adaptive noise estimation (FilterPy's advantage)
        Adapts Q and R based on prediction/innovation errors
    
    Returns:
    --------
    pd.Series
        Trading signals
    """
    if not HAS_FILTERPY:
        raise ImportError("filterpy is required for this function")
    
    signals = pd.Series('hold', index=prices.index)
    
    for i in range(lookback, len(prices)):
        current_date = prices.index[i]
        
        # IMPORTANT: Use only historical data up to current date (no look-ahead)
        history = prices.iloc[:i+1]
        
        if len(history) < lookback:
            continue
        
        # Apply FilterPy Kalman filter with adaptive noise
        kf = FilterPyKalman(dim_x=2, dim_z=1)
        kf.x = np.array([[history.iloc[0]], [0.0]])
        kf.F = np.array([[1., 1.], [0., 1.]])
        kf.H = np.array([[1., 0.]])
        
        # Initial Q and R
        base_q = np.array([[0.1, 0.], [0., 0.1]])
        base_r = history.std()**2
        
        # FilterPy's adaptive noise estimation - Sage-Husa style (conservative)
        if use_adaptive:
            # Use forget_factor for smooth adaptation (Sage-Husa algorithm style)
            forget_factor = 0.98  # Conservative: 98% weight on previous, 2% on new
            kf.Q = base_q
            kf.R = np.array([[base_r]])
            
            # Track for dynamic adaptation during filtering
            prediction_errors = []
            innovation_errors = []
        else:
            kf.Q = base_q
            kf.R = np.array([[base_r]])
        
        kf.P = np.eye(2) * 1000.
        
        # Filter all historical data
        filtered = []
        velocities = []
        for j, price in enumerate(history):
            # Predict step
            kf.predict()
            
            # Update step
            kf.update(np.array([[price]]))
            
            # Sage-Husa style adaptive noise estimation (conservative)
            if use_adaptive and j > 0:
                # Calculate innovation
                predicted_observation = kf.H @ kf.x
                innovation = price - predicted_observation[0, 0]
                innovation_errors.append(innovation)
                
                # Update R using forget_factor (Sage-Husa algorithm)
                # R = forget_factor * R_old + (1 - forget_factor) * innovation_variance
                if len(innovation_errors) > 5:  # Need some history
                    # Estimate innovation variance
                    recent_innovations = np.array(innovation_errors[-10:])
                    innovation_variance = np.var(recent_innovations)
                    
                    # Smooth update: 98% old, 2% new (conservative)
                    new_r = forget_factor * kf.R[0, 0] + (1 - forget_factor) * max(innovation_variance, base_r * 0.5)
                    kf.R = np.array([[np.clip(new_r, base_r * 0.8, base_r * 1.5)]])  # Conservative range
                
                # Update Q based on prediction errors (more conservative)
                pred_error = abs(innovation)
                prediction_errors.append(pred_error)
                
                if len(prediction_errors) > 5:
                    # Estimate process noise from prediction errors
                    recent_errors = np.array(prediction_errors[-10:])
                    error_variance = np.var(recent_errors)
                    
                    # Smooth update for Q (only diagonal elements)
                    new_q_val = forget_factor * kf.Q[0, 0] + (1 - forget_factor) * max(error_variance * 0.1, base_q[0, 0] * 0.8)
                    new_q_val = np.clip(new_q_val, base_q[0, 0] * 0.8, base_q[0, 0] * 1.5)  # Conservative range
                    kf.Q = np.array([[new_q_val, 0.], [0., new_q_val]])
            
            filtered.append(kf.x[0, 0])
            velocities.append(kf.x[1, 0])
        
        filtered_price = filtered[-1]
        filtered_velocity = velocities[-1]
        current_price = history.iloc[-1]
        
        # Generate signal
        price_diff = (filtered_price - current_price) / current_price
        
        if filtered_velocity > trend_threshold and price_diff > 0.01:  # Uptrend and undervalued
            signals.loc[current_date] = 'buy'
        elif filtered_velocity < -trend_threshold or price_diff < -0.01:  # Downtrend or overvalued
            signals.loc[current_date] = 'sell'
        else:
            signals.loc[current_date] = 'hold'
    
    return signals


def run_backtest_filterpy():
    """Run backtest for FilterPy Kalman filter"""
    if not HAS_FILTERPY:
        print("Error: filterpy is required. Install with: pip install filterpy")
        return
    
    print("="*60)
    print("Chapter 2: Kalman Filter - FilterPy Backtest (Adaptive Noise)")
    print("="*60)
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    tqqq = data['tqqq']
    
    # Generate signals with adaptive noise estimation (FilterPy's unique feature)
    print("\n1. Generating trading signals using FilterPy Kalman filter (with adaptive noise)...")
    signals = generate_trading_signals_filterpy(tqqq['Close'], lookback=60, use_adaptive=True)
    
    # Run backtest
    print("2. Running backtest...")
    results = walk_forward_backtest(tqqq['Close'], signals, train_ratio=0.7)
    
    # Print results
    print(f"\n{'='*60}")
    print("Backtest Results (FilterPy)")
    print(f"{'='*60}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Benchmark Return: {results['benchmark_return']:.2f}%")
    print(f"Annualized Return: {results['annualized_return']:.2f}%")
    print(f"Volatility: {results['volatility']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    
    # Visualize
    visualize_backtest_results(results, tqqq['Close'])


def visualize_backtest_results(results, prices):
    """Visualize backtest results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Equity curve
    equity_curve = results['equity_curve']
    test_prices = prices.loc[equity_curve.index]
    
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(test_prices.index, test_prices.values, 
            label='TQQQ Price', linewidth=2, color='blue', alpha=0.7)
    ax1_twin.plot(equity_curve.index, equity_curve['total_value'].values,
                  label='Portfolio Value', linewidth=2, color='red')
    
    ax1.set_ylabel('Price ($)', fontsize=12, color='blue')
    ax1_twin.set_ylabel('Portfolio Value ($)', fontsize=12, color='red')
    ax1.set_title('Equity Curve vs Price (FilterPy)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    portfolio_values = equity_curve['total_value']
    portfolio_returns = portfolio_values.pct_change().dropna()
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max - 1) * 100
    
    axes[1].fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.3, color='red', label='Drawdown')
    axes[1].plot(drawdown.index, drawdown.values, linewidth=2, color='red')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Trade history
    trades_df = pd.DataFrame(results['trade_history'])
    if len(trades_df) > 0:
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        axes[2].scatter(buy_trades['date'], buy_trades['price'], 
                       marker='^', color='green', s=100, label='Buy', zorder=5)
        axes[2].scatter(sell_trades['date'], sell_trades['price'], 
                       marker='v', color='red', s=100, label='Sell', zorder=5)
        axes[2].plot(test_prices.index, test_prices.values, 
                    linewidth=1, color='gray', alpha=0.5, label='Price')
        axes[2].set_ylabel('Price ($)', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].set_title('Trade History', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    run_backtest_filterpy()

