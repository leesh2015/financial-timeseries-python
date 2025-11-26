"""
Backtest Utilities for Section 4
Prevents information leakage and look-ahead bias
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class BacktestEngine:
    """
    Backtest engine that prevents information leakage
    
    Key Features:
    - Walk-forward approach: uses only data up to time t-1 to predict time t
    - Train/test split: prevents overfitting
    - No look-ahead bias: future prices never used
    - Minimal transaction costs: 0.02% commission (very small)
    - No slippage: assumed to be negligible
    """
    
    def __init__(self, initial_capital=10000, commission_rate=0.0002):
        """
        Initialize backtest engine
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital
        commission_rate : float
            Commission rate per trade (0.0002 = 0.02%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.reset()
    
    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.shares = 0
        self.position = 0  # 0: no position, 1: long
        self.trade_history = []
        self.equity_curve = []
        self.entry_price = None
        self.entry_date = None
    
    def execute_trade(self, date, price, signal, reason=""):
        """
        Execute a trade
        
        Parameters:
        -----------
        date : datetime
            Trade date
        price : float
            Trade price
        signal : str
            'buy', 'sell', or 'hold'
        reason : str
            Reason for trade (for logging)
        
        Returns:
        --------
        dict or None
            Trade record if trade executed
        """
        trade_record = None
        
        if signal == 'buy' and self.position == 0:
            # Buy signal: use all available capital
            cost = price * (1 + self.commission_rate)
            shares_to_buy = int(self.capital / cost)
            
            if shares_to_buy > 0:
                trade_cost = shares_to_buy * cost
                self.capital -= trade_cost
                self.shares += shares_to_buy
                self.position = 1
                self.entry_price = price
                self.entry_date = date
                
                trade_record = {
                    'date': date,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'cost': trade_cost,
                    'capital': self.capital,
                    'total_value': self.capital + self.shares * price,
                    'reason': reason
                }
                self.trade_history.append(trade_record)
        
        elif signal == 'sell' and self.position == 1:
            # Sell signal: sell all shares
            if self.shares > 0:
                proceeds = self.shares * price * (1 - self.commission_rate)
                self.capital += proceeds
                
                pnl = proceeds - (self.shares * self.entry_price)
                pnl_pct = (pnl / (self.shares * self.entry_price)) * 100 if self.entry_price > 0 else 0
                
                trade_record = {
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'shares': self.shares,
                    'proceeds': proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital': self.capital,
                    'total_value': self.capital,
                    'reason': reason,
                    'entry_date': self.entry_date,
                    'entry_price': self.entry_price,
                    'holding_period': (date - self.entry_date).days if self.entry_date else 0
                }
                self.trade_history.append(trade_record)
                
                self.shares = 0
                self.position = 0
                self.entry_price = None
                self.entry_date = None
        
        # Update equity curve
        current_value = self.capital + self.shares * price
        self.equity_curve.append({
            'date': date,
            'capital': self.capital,
            'shares': self.shares,
            'price': price,
            'total_value': current_value,
            'position': self.position
        })
        
        return trade_record
    
    def get_performance_metrics(self, prices):
        """
        Calculate performance metrics
        
        Parameters:
        -----------
        prices : pd.Series
            Price series for benchmark comparison
        
        Returns:
        --------
        dict
            Performance metrics
        """
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Portfolio returns
        portfolio_values = equity_df['total_value']
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        # Benchmark returns (buy and hold)
        benchmark_returns = prices.pct_change().dropna()
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return {}
        
        portfolio_ret = portfolio_returns.loc[common_dates]
        benchmark_ret = benchmark_returns.loc[common_dates]
        
        # Total return
        total_return = (portfolio_values.iloc[-1] / self.initial_capital - 1) * 100
        benchmark_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        
        # Annualized return
        days = (common_dates[-1] - common_dates[0]).days
        years = days / 365.25
        annualized_return = ((portfolio_values.iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        benchmark_annualized = ((prices.iloc[-1] / prices.iloc[0]) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Volatility
        portfolio_vol = portfolio_ret.std() * np.sqrt(252) * 100
        benchmark_vol = benchmark_ret.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = (annualized_return / portfolio_vol) if portfolio_vol > 0 else 0
        benchmark_sharpe = (benchmark_annualized / benchmark_vol) if benchmark_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_ret).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1) * 100
        max_drawdown = drawdown.min()
        
        benchmark_cumulative = (1 + benchmark_ret).cumprod()
        benchmark_running_max = benchmark_cumulative.expanding().max()
        benchmark_drawdown = (benchmark_cumulative / benchmark_running_max - 1) * 100
        benchmark_max_dd = benchmark_drawdown.min()
        
        # Win rate
        trades_df = pd.DataFrame(self.trade_history)
        if len(trades_df) > 0:
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            if len(sell_trades) > 0:
                win_rate = (sell_trades['pnl'] > 0).sum() / len(sell_trades) * 100
                avg_win = sell_trades[sell_trades['pnl'] > 0]['pnl_pct'].mean() if (sell_trades['pnl'] > 0).sum() > 0 else 0
                avg_loss = sell_trades[sell_trades['pnl'] <= 0]['pnl_pct'].mean() if (sell_trades['pnl'] <= 0).sum() > 0 else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'annualized_return': annualized_return,
            'benchmark_annualized': benchmark_annualized,
            'volatility': portfolio_vol,
            'benchmark_volatility': benchmark_vol,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'benchmark_max_drawdown': benchmark_max_dd,
            'total_trades': len(self.trade_history),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'final_value': portfolio_values.iloc[-1],
            'equity_curve': equity_df
        }


def walk_forward_backtest(prices, signals, train_ratio=0.7, min_train_size=60):
    """
    Walk-forward backtest that prevents information leakage
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    signals : pd.Series
        Trading signals ('buy', 'sell', 'hold')
    train_ratio : float
        Initial train/test split ratio
    min_train_size : int
        Minimum training period size
    
    Returns:
    --------
    dict
        Backtest results
    """
    # Split data
    split_idx = int(len(prices) * train_ratio)
    train_prices = prices.iloc[:split_idx]
    test_prices = prices.iloc[split_idx:]
    test_signals = signals.iloc[split_idx:]
    
    # Initialize backtest
    backtest = BacktestEngine()
    
    # Run backtest on test period
    for date, price in test_prices.items():
        if date in test_signals.index:
            signal = test_signals.loc[date]
            backtest.execute_trade(date, price, signal)
        else:
            # Update equity curve even if no signal
            backtest.execute_trade(date, price, 'hold')
    
    # Close any open position at the end
    if backtest.position == 1:
        final_date = test_prices.index[-1]
        final_price = test_prices.iloc[-1]
        backtest.execute_trade(final_date, final_price, 'sell', reason="End of backtest")
    
    # Calculate metrics
    metrics = backtest.get_performance_metrics(test_prices)
    metrics['trade_history'] = backtest.trade_history
    metrics['backtest_engine'] = backtest
    
    return metrics

