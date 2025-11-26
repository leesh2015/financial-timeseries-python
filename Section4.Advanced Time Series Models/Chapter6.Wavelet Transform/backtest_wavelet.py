"""
Backtest for Chapter 6: Wavelet Transform
Denoised signal-based trading strategy
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

from utils.data_loader import load_nasdaq_tqqq_data
from utils.backtest import BacktestEngine, walk_forward_backtest

warnings.filterwarnings("ignore")

try:
    import pywt
    HAS_PYWAVELETS = True
except ImportError:
    HAS_PYWAVELETS = False


def generate_trading_signals_wavelet(returns, prices, lookback=60, wavelet='db4', level=5):
    """
    Generate trading signals based on wavelet denoised trend
    
    Strategy:
    - Buy when denoised trend is positive and price is below denoised price
    - Sell when denoised trend is negative or price is above denoised price
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    prices : pd.Series
        Price series
    lookback : int
        Minimum data points needed
    wavelet : str
        Wavelet type
    level : int
        Decomposition level
    
    Returns:
    --------
    pd.Series
        Trading signals
    """
    if not HAS_PYWAVELETS:
        raise ImportError("PyWavelets is not installed. Install with: pip install pywavelets")
    
    signals = pd.Series('hold', index=prices.index)
    
    # Debug statistics
    trend_values = []
    price_diff_values = []
    buy_count = 0
    sell_count = 0
    hold_count = 0
    
    for i in range(lookback, len(returns)):
        current_date = returns.index[i]
        
        if current_date not in prices.index:
            continue
        
        # IMPORTANT: Use only historical data up to current date (no look-ahead)
        # Data up to index i (inclusive) is available at time i
        history_returns = returns.iloc[:i+1]
        
        if len(history_returns) < lookback:
            continue
        
        try:
            # Wavelet decomposition
            coeffs = pywt.wavedec(history_returns.values, wavelet, level=level)
            
            # Reconstruct approximation (trend) only using upcoef (more reliable)
            cA = coeffs[0]  # Approximation coefficients
            denoised_returns = pywt.upcoef('a', cA, wavelet, level=level)
            denoised_returns = denoised_returns[:len(history_returns)]
            denoised_returns = pd.Series(denoised_returns, index=history_returns.index)
            
            # Get initial price for scaling
            initial_price = prices.loc[history_returns.index[0]]
            
            # Calculate denoised price (cumulative returns applied to initial price)
            denoised_prices = initial_price * (1 + denoised_returns).cumprod()
            
            # Get current values
            current_price = prices.loc[current_date]
            denoised_price = denoised_prices.iloc[-1] if len(denoised_prices) > 0 else current_price
            
            # Calculate trend (recent change in denoised returns)
            if len(denoised_returns) >= 5:
                recent_trend = np.mean(denoised_returns[-5:])
            else:
                recent_trend = 0
            
            # Generate signal
            price_diff = (denoised_price - current_price) / current_price
            
            # Store for statistics
            trend_values.append(recent_trend)
            price_diff_values.append(price_diff)
            
            if recent_trend > 0.001 and price_diff > 0.01:  # Positive trend and undervalued
                signals.loc[current_date] = 'buy'
                buy_count += 1
            elif recent_trend < -0.001 or price_diff < -0.01:  # Negative trend or overvalued
                signals.loc[current_date] = 'sell'
                sell_count += 1
            else:
                signals.loc[current_date] = 'hold'
                hold_count += 1
        
        except Exception as e:
            signals.loc[current_date] = 'hold'
            hold_count += 1
            continue
    
    # Print debug statistics
    if len(trend_values) > 0:
        trend_array = np.array(trend_values)
        price_diff_array = np.array(price_diff_values)
        
        print(f"\n   Signal Generation Statistics:")
        print(f"   - Total periods analyzed: {len(trend_values)}")
        print(f"   - Buy signals: {buy_count}")
        print(f"   - Sell signals: {sell_count}")
        print(f"   - Hold signals: {hold_count}")
        print(f"\n   Trend Statistics:")
        print(f"   - Min trend: {trend_array.min():.6f}")
        print(f"   - Max trend: {trend_array.max():.6f}")
        print(f"   - Mean trend: {trend_array.mean():.6f}")
        print(f"   - Trend > 0.001: {(trend_array > 0.001).sum()} ({(trend_array > 0.001).sum()/len(trend_array)*100:.1f}%)")
        print(f"   - Trend < -0.001: {(trend_array < -0.001).sum()} ({(trend_array < -0.001).sum()/len(trend_array)*100:.1f}%)")
        print(f"\n   Price Diff Statistics:")
        print(f"   - Min price_diff: {price_diff_array.min():.4f} ({price_diff_array.min()*100:.2f}%)")
        print(f"   - Max price_diff: {price_diff_array.max():.4f} ({price_diff_array.max()*100:.2f}%)")
        print(f"   - Mean price_diff: {price_diff_array.mean():.4f} ({price_diff_array.mean()*100:.2f}%)")
        print(f"   - Price_diff > 0.01: {(price_diff_array > 0.01).sum()} ({(price_diff_array > 0.01).sum()/len(price_diff_array)*100:.1f}%)")
        print(f"   - Price_diff < -0.01: {(price_diff_array < -0.01).sum()} ({(price_diff_array < -0.01).sum()/len(price_diff_array)*100:.1f}%)")
        print(f"\n   Combined Conditions:")
        buy_condition = (trend_array > 0.001) & (price_diff_array > 0.01)
        sell_condition = (trend_array < -0.001) | (price_diff_array < -0.01)
        print(f"   - Buy condition met: {buy_condition.sum()} ({buy_condition.sum()/len(trend_array)*100:.1f}%)")
        print(f"   - Sell condition met: {sell_condition.sum()} ({sell_condition.sum()/len(trend_array)*100:.1f}%)")
    
    return signals


def run_backtest_wavelet():
    """Run backtest for wavelet transform"""
    print("="*60)
    print("Chapter 6: Wavelet Transform - Backtest")
    print("="*60)
    
    if not HAS_PYWAVELETS:
        print("\nError: PyWavelets is not installed.")
        print("Install with: pip install pywavelets")
        return
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    tqqq = data['tqqq']
    
    # Generate signals
    print("\n1. Generating trading signals using wavelet denoised trend...")
    signals = generate_trading_signals_wavelet(
        tqqq['Returns'].dropna(), tqqq['Close'], lookback=60
    )
    
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
    
    return results


def visualize_backtest_results(results, prices):
    """Visualize backtest results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
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
    run_backtest_wavelet()

