"""
Backtest for Chapter 1: State-Space Models
Time-varying beta based trading strategy
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
from utils.backtest import BacktestEngine, walk_forward_backtest
# Note: Import from state_space_model if needed, but we'll implement inline to avoid circular imports

warnings.filterwarnings("ignore")


def generate_trading_signals_state_space(nasdaq_returns, tqqq_returns, tqqq_prices,
                                        lookback=60, beta_threshold_low=2.5, beta_threshold_high=3.5):
    """
    Generate trading signals from rolling beta without look-ahead bias.

    We pre-compute rolling betas via vectorized covariance/variance to avoid
    re-fitting OLS models inside a Python loop (huge speed-up) and shift the
    resulting series by one day so that the signal at time t only uses data
    through t-1.
    """
    nasdaq_ret, tqqq_ret = align_data(nasdaq_returns, tqqq_returns)
    common_idx = nasdaq_ret.index.intersection(tqqq_prices.index)
    nasdaq_ret = nasdaq_ret.loc[common_idx]
    tqqq_ret = tqqq_ret.loc[common_idx]
    tqqq_prices_aligned = tqqq_prices.loc[common_idx]

    # Vectorized rolling beta/alpha
    rolling_cov = tqqq_ret.rolling(window=lookback, min_periods=lookback).cov(nasdaq_ret)
    rolling_var = nasdaq_ret.rolling(window=lookback, min_periods=lookback).var()
    rolling_beta = (rolling_cov / rolling_var).dropna()

    nasdaq_mean = nasdaq_ret.rolling(window=lookback, min_periods=lookback).mean()
    tqqq_mean = tqqq_ret.rolling(window=lookback, min_periods=lookback).mean()
    rolling_alpha = (tqqq_mean - rolling_beta * nasdaq_mean).dropna()

    betas = rolling_beta.shift(1).reindex(tqqq_prices_aligned.index)  # ensure t uses data <= t-1
    signals = pd.Series('hold', index=tqqq_prices_aligned.index, dtype=object)

    signals.loc[betas < beta_threshold_low] = 'buy'
    signals.loc[betas > beta_threshold_high] = 'sell'

    valid_betas = betas.dropna()
    if len(valid_betas) > 0:
        print(f"\n   Beta Statistics (shifted, no look-ahead):")
        print(f"   - Min beta: {valid_betas.min():.2f}")
        print(f"   - Max beta: {valid_betas.max():.2f}")
        print(f"   - Mean beta: {valid_betas.mean():.2f}")
        print(f"   - Median beta: {valid_betas.median():.2f}")
        below = (valid_betas < beta_threshold_low).sum()
        above = (valid_betas > beta_threshold_high).sum()
        total = len(valid_betas)
        print(f"   - Beta < {beta_threshold_low}: {below} ({below/total*100:.1f}%)")
        print(f"   - Beta > {beta_threshold_high}: {above} ({above/total*100:.1f}%)")
        print(f"   - Beta in range: {total - below - above} ({(total - below - above)/total*100:.1f}%)")

    return signals


def run_backtest_state_space():
    """Run backtest for state-space model"""
    print("="*60)
    print("Chapter 1: State-Space Models - Backtest")
    print("="*60)
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    nasdaq = data['nasdaq']
    tqqq = data['tqqq']
    
    # Generate signals
    print("\n1. Generating trading signals...")
    
    # First, calculate beta distribution to set adaptive thresholds without looping
    lookback = 60
    nasdaq_ret, tqqq_ret = align_data(nasdaq['Returns'], tqqq['Returns'])
    common_idx = nasdaq_ret.index.intersection(tqqq['Close'].index)
    nasdaq_ret = nasdaq_ret.loc[common_idx]
    tqqq_ret = tqqq_ret.loc[common_idx]
    rolling_cov = tqqq_ret.rolling(window=lookback, min_periods=lookback).cov(nasdaq_ret)
    rolling_var = nasdaq_ret.rolling(window=lookback, min_periods=lookback).var()
    beta_series = (rolling_cov / rolling_var).dropna()
    
    if len(beta_series) > 0:
        beta_threshold_low = np.percentile(beta_series.values, 25)
        beta_threshold_high = np.percentile(beta_series.values, 75)
        print(f"   Strategy: Buy when beta < {beta_threshold_low:.2f} (25th percentile)")
        print(f"            Sell when beta > {beta_threshold_high:.2f} (75th percentile)")
        print(f"   Beta range: [{beta_series.min():.2f}, {beta_series.max():.2f}]")
        print(f"   Mean beta: {beta_series.mean():.2f}, Std: {beta_series.std():.2f}")
    else:
        beta_threshold_low = 2.7
        beta_threshold_high = 3.2
        print(f"   Strategy: Buy when beta < {beta_threshold_low:.2f}, Sell when beta > {beta_threshold_high:.2f}")
        print("   (Using fixed thresholds as fallback)")
    
    signals = generate_trading_signals_state_space(
        nasdaq['Returns'], tqqq['Returns'], tqqq['Close'],
        lookback=60, beta_threshold_low=beta_threshold_low, beta_threshold_high=beta_threshold_high
    )
    
    # Debug: Check signal generation
    total_signals = len(signals[signals != 'hold'])
    buy_signals = len(signals[signals == 'buy'])
    sell_signals = len(signals[signals == 'sell'])
    print(f"\n   Signal Statistics:")
    print(f"   - Total signals (non-hold): {total_signals}")
    print(f"   - Buy signals: {buy_signals}")
    print(f"   - Sell signals: {sell_signals}")
    print(f"   - Hold signals: {len(signals) - total_signals}")
    
    # Check signals in test period
    split_idx = int(len(tqqq['Close']) * 0.7)
    # Align signals with prices
    common_dates = signals.index.intersection(tqqq['Close'].index)
    signals_aligned = signals.loc[common_dates]
    test_signals = signals_aligned.iloc[split_idx:] if len(signals_aligned) > split_idx else pd.Series(dtype=str)
    test_total = len(test_signals[test_signals != 'hold']) if len(test_signals) > 0 else 0
    test_buy = len(test_signals[test_signals == 'buy']) if len(test_signals) > 0 else 0
    test_sell = len(test_signals[test_signals == 'sell']) if len(test_signals) > 0 else 0
    print(f"\n   Test Period Signals (after {split_idx}):")
    print(f"   - Total signals: {test_total}")
    print(f"   - Buy signals: {test_buy}")
    print(f"   - Sell signals: {test_sell}")
    
    # Run backtest
    print("\n2. Running backtest...")
    # Use aligned signals
    results = walk_forward_backtest(tqqq['Close'], signals_aligned, train_ratio=0.7)
    
    # Print results
    print(f"\n{'='*60}")
    print("Backtest Results")
    print(f"{'='*60}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Benchmark Return: {results['benchmark_return']:.2f}%")
    print(f"Annualized Return: {results['annualized_return']:.2f}%")
    print(f"Benchmark Annualized: {results['benchmark_annualized']:.2f}%")
    print(f"Volatility: {results['volatility']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    
    # Visualize
    visualize_backtest_results(results, tqqq['Close'])
    
    return results


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


if __name__ == "__main__":
    run_backtest_state_space()

