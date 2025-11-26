"""
Backtest for Chapter 7: Copula Models
Portfolio risk-based trading strategy
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_nasdaq_tqqq_data, align_data
from utils.backtest import BacktestEngine, walk_forward_backtest

warnings.filterwarnings("ignore")

try:
    from copulae import GaussianCopula, StudentCopula
    HAS_COPULAE = True
except ImportError:
    HAS_COPULAE = False


def generate_trading_signals_copula(nasdaq_returns, tqqq_returns, tqqq_prices, 
                                    lookback=60, var_threshold=0.05, 
                                    n_simulations=500, recalc_interval=1,
                                    strategy_type='risk_averse'):
    """
    Generate trading signals based on copula-based portfolio risk
    
    Two Strategy Types:
    
    1. 'risk_averse' (default): Conservative approach
       - Buy when VaR decreases (risk environment improving)
         * VaR becomes less negative (e.g., -0.05 → -0.03)
         * Lower expected loss → safer to invest
       - Sell when VaR increases (risk environment deteriorating)
         * VaR becomes more negative (e.g., -0.03 → -0.05)
         * Higher expected loss → reduce exposure
       - Problem: May buy late (after price already rose) and sell late (after price already fell)
    
    2. 'risk_seeking' (contrarian): Aggressive approach
       - Buy when VaR increases (risk environment deteriorating)
         * High risk = high potential reward (volatility premium)
         * Mean reversion: extreme risk often followed by recovery
         * Fear buying: market overreaction creates opportunity
       - Sell when VaR decreases (risk environment improving)
         * Lock in profits during stable periods
         * Low volatility = limited upside potential
       - Theory: Risk premium, volatility trading, contrarian investing
    
    Parameters:
    -----------
    nasdaq_returns : pd.Series
        NASDAQ returns
    tqqq_returns : pd.Series
        TQQQ returns
    tqqq_prices : pd.Series
        TQQQ prices
    lookback : int
        Minimum data points needed
    var_threshold : float
        VaR threshold for risk assessment
    n_simulations : int
        Number of simulations for VaR calculation (default: 500, reduced for speed)
    recalc_interval : int
        Recalculate VaR every N days (default: 1 = every day, 5 = every 5 days)
    strategy_type : str
        'risk_averse' or 'risk_seeking' (default: 'risk_averse')
    
    Returns:
    --------
    pd.Series
        Trading signals
    """
    # Align data
    nasdaq_ret, tqqq_ret = align_data(nasdaq_returns, tqqq_returns)
    
    signals = pd.Series('hold', index=tqqq_prices.index)
    previous_var = None
    
    total_days = len(nasdaq_ret) - lookback
    processed_days = 0
    
    for i in range(lookback, len(nasdaq_ret)):
        current_date = nasdaq_ret.index[i]
        
        if current_date not in tqqq_prices.index:
            continue
        
        # Skip if not time to recalculate (for performance optimization)
        # Only recalculate VaR every N days, use previous signal for skipped days
        if recalc_interval > 1 and (i - lookback) % recalc_interval != 0:
            # Use previous signal for skipped days
            if previous_var is not None and i > lookback:
                # Copy signal from last calculated day
                prev_idx = i - ((i - lookback) % recalc_interval)
                if prev_idx >= lookback:
                    prev_date = nasdaq_ret.index[prev_idx]
                    if prev_date in signals.index:
                        signals.loc[current_date] = signals.loc[prev_date]
            continue
        
        # IMPORTANT: Use only historical data up to current date (no look-ahead)
        # Data up to index i (inclusive) is available at time i
        history_nasdaq = nasdaq_ret.iloc[:i+1]
        history_tqqq = tqqq_ret.iloc[:i+1]
        
        if len(history_nasdaq) < lookback:
            continue
        
        try:
            # Use recent window
            window_start = max(0, i - lookback + 1)
            window_nasdaq = history_nasdaq.iloc[window_start:i+1]
            window_tqqq = history_tqqq.iloc[window_start:i+1]
            
            if len(window_nasdaq) < 30:  # Need minimum data
                continue
            
            # Estimate marginal distributions (t-distribution)
            params_nasdaq = stats.t.fit(window_nasdaq.dropna())
            params_tqqq = stats.t.fit(window_tqqq.dropna())
            
            # Transform to uniform marginals
            u_nasdaq = stats.t.cdf(window_nasdaq, *params_nasdaq)
            u_tqqq = stats.t.cdf(window_tqqq, *params_tqqq)
            
            u_data = np.column_stack([u_nasdaq, u_tqqq])
            
            # Fit copula or use correlation
            if HAS_COPULAE:
                try:
                    copula = GaussianCopula(dim=2)
                    copula.fit(u_data)
                    simulated_u = copula.random(n_simulations)
                except:
                    # Fallback to correlation
                    correlation = np.corrcoef(window_nasdaq, window_tqqq)[0, 1]
                    mean = [0, 0]
                    cov = [[1, correlation], [correlation, 1]]
                    simulated_norm = np.random.multivariate_normal(mean, cov, n_simulations)
                    simulated_u = stats.norm.cdf(simulated_norm)
            else:
                # Use correlation
                correlation = np.corrcoef(window_nasdaq, window_tqqq)[0, 1]
                mean = [0, 0]
                cov = [[1, correlation], [correlation, 1]]
                simulated_norm = np.random.multivariate_normal(mean, cov, n_simulations)
                simulated_u = stats.norm.cdf(simulated_norm)
            
            # Transform back to returns
            simulated_ret_nasdaq = stats.t.ppf(simulated_u[:, 0], *params_nasdaq)
            simulated_ret_tqqq = stats.t.ppf(simulated_u[:, 1], *params_tqqq)
            
            # Calculate portfolio returns (50/50 portfolio)
            portfolio_returns = 0.5 * simulated_ret_nasdaq + 0.5 * simulated_ret_tqqq
            
            # Calculate VaR (negative value: e.g., -0.05 means 5% loss)
            current_var = np.percentile(portfolio_returns, var_threshold * 100)
            
            # Generate signal based on VaR change
            # VaR is negative: more negative = higher risk, less negative = lower risk
            if previous_var is not None:
                var_change = current_var - previous_var
                
                if strategy_type == 'risk_averse':
                    # Conservative: Buy when risk decreases, Sell when risk increases
                    if var_change > 0.01:  # Risk decreasing: VaR less negative (e.g., -0.05 → -0.03)
                        signals.loc[current_date] = 'buy'
                    elif var_change < -0.01:  # Risk increasing: VaR more negative (e.g., -0.03 → -0.05)
                        signals.loc[current_date] = 'sell'
                    else:
                        signals.loc[current_date] = 'hold'
                elif strategy_type == 'risk_seeking':
                    # Contrarian: Buy when risk increases, Sell when risk decreases
                    # Theory: High risk = high reward (volatility premium, mean reversion)
                    if var_change < -0.01:  # Risk increasing: VaR more negative (e.g., -0.03 → -0.05)
                        signals.loc[current_date] = 'buy'  # Buy the fear
                    elif var_change > 0.01:  # Risk decreasing: VaR less negative (e.g., -0.05 → -0.03)
                        signals.loc[current_date] = 'sell'  # Sell the calm
                    else:
                        signals.loc[current_date] = 'hold'
                else:
                    raise ValueError(f"Unknown strategy_type: {strategy_type}. Use 'risk_averse' or 'risk_seeking'")
            
            previous_var = current_var
            processed_days += 1
            
            # Progress update every 100 days
            if processed_days % 100 == 0:
                progress = (processed_days / total_days) * 100
                print(f"  Progress: {processed_days}/{total_days} days ({progress:.1f}%)")
        
        except Exception as e:
            signals.loc[current_date] = 'hold'
            continue
    
    print(f"  Signal generation complete: {processed_days} days processed")
    return signals


def run_backtest_copula(strategy_type='risk_averse'):
    """
    Run backtest for copula model
    
    Parameters:
    -----------
    strategy_type : str
        'risk_averse' or 'risk_seeking' (default: 'risk_averse')
    """
    print("="*60)
    print("Chapter 7: Copula Models - Backtest")
    print(f"Strategy: {strategy_type.upper().replace('_', '-')}")
    print("="*60)
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    nasdaq = data['nasdaq']
    tqqq = data['tqqq']
    
    # Generate signals
    print("\n1. Generating trading signals using copula-based risk analysis...")
    print("   (This may take a while - calculating VaR for each day)")
    if strategy_type == 'risk_averse':
        print("   Strategy: Risk-Averse (Buy when risk decreases, Sell when risk increases)")
    else:
        print("   Strategy: Risk-Seeking (Buy when risk increases, Sell when risk decreases)")
    signals = generate_trading_signals_copula(
        nasdaq['Returns'].dropna(), tqqq['Returns'].dropna(), tqqq['Close'],
        lookback=60, var_threshold=0.05, n_simulations=500, recalc_interval=1,
        strategy_type=strategy_type
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
    import sys
    
    # Allow strategy selection from command line
    # Default: risk_seeking (if no argument provided)
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        if strategy == 'compare':
            # Run both strategies for comparison
            print("\n" + "="*60)
            print("COMPARING BOTH STRATEGIES")
            print("="*60)
            
            print("\n" + "-"*60)
            print("STRATEGY 1: Risk-Averse (Conservative)")
            print("-"*60)
            results_averse = run_backtest_copula(strategy_type='risk_averse')
            
            print("\n" + "-"*60)
            print("STRATEGY 2: Risk-Seeking (Contrarian)")
            print("-"*60)
            results_seeking = run_backtest_copula(strategy_type='risk_seeking')
            
            # Compare results
            print("\n" + "="*60)
            print("STRATEGY COMPARISON")
            print("="*60)
            print(f"{'Metric':<25} {'Risk-Averse':<15} {'Risk-Seeking':<15}")
            print("-"*60)
            print(f"{'Total Return':<25} {results_averse['total_return']:>14.2f}% {results_seeking['total_return']:>14.2f}%")
            print(f"{'Annualized Return':<25} {results_averse['annualized_return']:>14.2f}% {results_seeking['annualized_return']:>14.2f}%")
            print(f"{'Sharpe Ratio':<25} {results_averse['sharpe_ratio']:>14.4f} {results_seeking['sharpe_ratio']:>14.4f}")
            print(f"{'Max Drawdown':<25} {results_averse['max_drawdown']:>14.2f}% {results_seeking['max_drawdown']:>14.2f}%")
            print(f"{'Total Trades':<25} {results_averse['total_trades']:>14} {results_seeking['total_trades']:>14}")
            print(f"{'Win Rate':<25} {results_averse['win_rate']:>14.2f}% {results_seeking['win_rate']:>14.2f}%")
            print("="*60)
        elif strategy not in ['risk_averse', 'risk_seeking']:
            print("Usage: python backtest_copula.py [risk_averse|risk_seeking|compare]")
            print("Default: risk_seeking (if no argument)")
            print("  risk_averse: Conservative strategy (buy when risk decreases)")
            print("  risk_seeking: Contrarian strategy (buy when risk increases) - DEFAULT")
            print("  compare: Run both strategies and compare results")
            strategy = 'risk_seeking'
        else:
            # Run single strategy
            run_backtest_copula(strategy_type=strategy)
    else:
        # Default: risk_seeking
        run_backtest_copula(strategy_type='risk_seeking')

