"""
Backtest for Chapter 2: Kalman Filter
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
from kalman_filter import KalmanFilter

warnings.filterwarnings("ignore")


def generate_trading_signals_kalman(prices, lookback=60, trend_threshold=0.001):
    """
    Generate trading signals based on a single pass Kalman filter.

    We run the filter once across the full series (O(N)) and then use the
    *previous day's* filtered state to decide what to do today, eliminating
    the expensive O(N^2) re-fitting loop and the look-ahead bias.
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[prices.iloc[0]], [0.0]])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.Q = np.array([[0.1, 0.], [0., 0.1]])
    kf.P *= 1000.

    # Adaptive measurement noise based on rolling std to keep the filter stable
    rolling_std = prices.pct_change().rolling(window=lookback, min_periods=10).std().fillna(method='bfill')

    filtered_price = []
    filtered_velocity = []
    for idx, price in enumerate(prices):
        if idx > 0:
            kf.predict()
        measurement = np.array([[price]])
        noise = rolling_std.iloc[idx] if idx < len(rolling_std) else rolling_std.iloc[-1]
        noise = max(noise, 1e-4)
        kf.R = np.array([[noise**2]])
        kf.update(measurement)
        filtered_price.append(kf.x[0, 0])
        filtered_velocity.append(kf.x[1, 0])

    filtered_price = pd.Series(filtered_price, index=prices.index)
    filtered_velocity = pd.Series(filtered_velocity, index=prices.index)

    # Use yesterday's filtered estimates to trade today
    price_gap = ((filtered_price.shift(1) - prices) / prices)
    trend = filtered_velocity.shift(1)

    signals = pd.Series('hold', index=prices.index, dtype=object)
    signals.loc[(trend > trend_threshold) & (price_gap > 0.01)] = 'buy'
    signals.loc[(trend < -trend_threshold) | (price_gap < -0.01)] = 'sell'

    # Warm-up period: force hold until enough history is accumulated
    signals.iloc[:lookback] = 'hold'
    return signals


def run_backtest_kalman():
    """Run backtest for Kalman filter"""
    print("="*60)
    print("Chapter 2: Kalman Filter - Backtest")
    print("="*60)
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    tqqq = data['tqqq']
    
    # Generate signals
    print("\n1. Generating trading signals using Kalman filter...")
    signals = generate_trading_signals_kalman(tqqq['Close'], lookback=60)
    
    # Run backtest
    print("2. Running backtest...")
    results = walk_forward_backtest(tqqq['Close'], signals, train_ratio=0.7)
    
    # Print results
    print(f"\n{'='*60}")
    print("Backtest Results")
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
    ax1.set_title('Equity Curve vs Price', fontsize=14, fontweight='bold')
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
    run_backtest_kalman()

