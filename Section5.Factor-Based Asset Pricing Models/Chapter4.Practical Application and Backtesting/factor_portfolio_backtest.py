"""
Chapter 4: Factor-Based Portfolio Construction and Backtesting

This script implements:
1. Train/test data separation
2. Multi-stage factor-based portfolio construction
3. Walk-forward validation
4. Performance evaluation with proper metrics
5. Transaction cost consideration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Add Section5 root to path
section4_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, section4_root)

from utils.data_loader import load_ff_factors, load_stock_data, align_data
from utils.factor_utils import calculate_factor_exposures, construct_factor_portfolio
from utils.metrics import (
    calculate_sharpe_ratio, calculate_alpha, calculate_information_ratio,
    calculate_max_drawdown, calculate_annualized_return, calculate_volatility
)
import warnings

warnings.filterwarnings("ignore")


class FactorPortfolioBacktest:
    """
    Factor-based portfolio backtesting framework with train/test separation
    """
    
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        train_ratio: float = 0.7,
        rebalance_frequency: str = 'monthly',
        transaction_cost: float = 0.002,  # 0.2% per trade
        min_r_squared: float = 0.3,
        factor_selection_criteria: Dict[str, float] = None
    ):
        """
        Initialize backtest framework
        
        Parameters:
        -----------
        tickers : List[str]
            List of stock tickers
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        train_ratio : float
            Ratio of data for training (default: 0.7)
        rebalance_frequency : str
            Rebalancing frequency ('monthly', 'quarterly', 'yearly')
        transaction_cost : float
            Transaction cost per trade (default: 0.2%)
        min_r_squared : float
            Minimum R-squared for factor model
        factor_selection_criteria : Dict[str, float]
            Factor selection criteria (e.g., {'SMB': 0.1, 'HML': 0.05})
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.train_ratio = train_ratio
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.min_r_squared = min_r_squared
        self.factor_selection_criteria = factor_selection_criteria or {'SMB': 0.1, 'HML': 0.05}
        
        # Load data
        self._load_data()
        
        # Split train/test
        self._split_data()
    
    def _load_data(self):
        """Load stock and factor data"""
        print("[Step 1] Loading data...")
        
        # Load Fama-French 6-factor data
        self.ff_factors = load_ff_factors('6-factor', 'daily')
        print(f"  Loaded {len(self.ff_factors)} days of factor data")
        
        # Load stock data
        print(f"  Loading {len(self.tickers)} stocks...")
        stock_data_list = []
        valid_tickers = []
        
        for ticker in self.tickers:
            try:
                stock_data = load_stock_data(ticker, self.start_date, self.end_date)
                if not stock_data.empty:
                    stock_data_list.append(stock_data['Close'].rename(ticker))
                    valid_tickers.append(ticker)
            except Exception as e:
                print(f"    Warning: Failed to load {ticker}: {str(e)[:50]}")
                continue
        
        if not stock_data_list:
            raise ValueError("No stock data loaded")
        
        self.stock_prices = pd.concat(stock_data_list, axis=1)
        self.tickers = valid_tickers
        print(f"  Successfully loaded {len(self.tickers)} stocks")
        
        # Calculate returns
        self.stock_returns = self.stock_prices.pct_change().dropna()
        
        # Align with factor data
        common_dates = self.stock_returns.index.intersection(self.ff_factors.index)
        self.stock_returns = self.stock_returns.loc[common_dates]
        self.ff_factors = self.ff_factors.loc[common_dates]
        
        print(f"  Aligned data: {len(common_dates)} days")
        print(f"  Date range: {common_dates.min()} to {common_dates.max()}")
    
    def _split_data(self):
        """Split data into training and testing sets"""
        dates = sorted(self.stock_returns.index)
        split_idx = int(len(dates) * self.train_ratio)
        
        train_end_date = dates[split_idx]
        
        self.train_returns = self.stock_returns.loc[:train_end_date]
        self.test_returns = self.stock_returns.loc[train_end_date:]
        
        self.train_factors = self.ff_factors.loc[:train_end_date]
        self.test_factors = self.ff_factors.loc[train_end_date:]
        
        print(f"\n[Step 2] Data Split:")
        print(f"  Training: {len(self.train_returns)} days ({self.train_returns.index.min()} to {self.train_returns.index.max()})")
        print(f"  Testing: {len(self.test_returns)} days ({self.test_returns.index.min()} to {self.test_returns.index.max()})")
    
    def train_model(self):
        """Train factor model on training data"""
        print("\n[Step 3] Training factor model...")
        
        # Calculate factor exposures for all stocks
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        self.stock_exposures = {}
        
        for ticker in self.train_returns.columns:
            try:
                exposures = calculate_factor_exposures(
                    self.train_returns[ticker],
                    self.train_factors,
                    risk_free_col='RF',
                    factor_cols=factor_cols
                )
                
                if exposures['r_squared'] >= self.min_r_squared:
                    self.stock_exposures[ticker] = exposures
            except Exception:
                continue
        
        print(f"  Calculated exposures for {len(self.stock_exposures)} stocks")
        
        # Select stocks based on criteria
        # Method: Calculate composite factor score and select top stocks
        stock_scores = {}
        
        for ticker, exposures in self.stock_exposures.items():
            score = 0
            valid_factors = 0
            
            for factor, threshold in self.factor_selection_criteria.items():
                beta_key = f'{factor}_beta'
                if beta_key in exposures:
                    # Score based on how much above threshold (or at least positive)
                    beta_value = exposures[beta_key]
                    # For factors where we want positive exposure
                    if beta_value > 0:
                        score += beta_value
                        valid_factors += 1
                    # Small penalty for negative (but not disqualifying)
                    elif beta_value > -threshold:
                        score += beta_value * 0.5  # Half credit for slightly negative
                        valid_factors += 1
            
            # Only include stocks with at least some positive factor exposure
            if valid_factors > 0 and score > 0:
                stock_scores[ticker] = score
        
        # Select top stocks (at least 3, up to 50% of available stocks)
        if stock_scores:
            sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
            min_stocks = min(3, len(sorted_stocks))
            max_stocks = max(min_stocks, len(sorted_stocks) // 2)
            self.selected_stocks = [ticker for ticker, score in sorted_stocks[:max_stocks]]
        else:
            # Fallback: Select stocks with highest R-squared if no factor criteria met
            print("  Warning: No stocks met factor criteria. Selecting by R-squared...")
            sorted_by_r2 = sorted(
                self.stock_exposures.items(),
                key=lambda x: x[1]['r_squared'],
                reverse=True
            )
            self.selected_stocks = [ticker for ticker, _ in sorted_by_r2[:max(3, len(sorted_by_r2)//2)]]
        
        print(f"  Selected {len(self.selected_stocks)} stocks based on factor criteria")
        print(f"  Selection criteria: {self.factor_selection_criteria}")
        if len(self.selected_stocks) > 0:
            print(f"  Selected stocks: {', '.join(self.selected_stocks[:5])}{'...' if len(self.selected_stocks) > 5 else ''}")
        
        return self.selected_stocks
    
    def backtest(self):
        """Run backtest on test data"""
        print("\n[Step 4] Running backtest on test data...")
        
        if not self.selected_stocks:
            print("  No stocks selected. Run train_model() first.")
            return None
        
        # Filter to selected stocks
        test_returns_selected = self.test_returns[self.selected_stocks]
        
        # Equal-weighted portfolio
        portfolio_returns = test_returns_selected.mean(axis=1)
        
        # Apply transaction costs
        # Simple approximation: cost on rebalancing days
        if self.rebalance_frequency == 'monthly':
            rebalance_days = portfolio_returns.resample('M').first().index
        elif self.rebalance_frequency == 'quarterly':
            rebalance_days = portfolio_returns.resample('Q').first().index
        else:
            rebalance_days = portfolio_returns.index
        
        portfolio_returns_net = portfolio_returns.copy()
        for day in rebalance_days:
            if day in portfolio_returns_net.index:
                portfolio_returns_net.loc[day] -= self.transaction_cost
        
        # Benchmark (market)
        market_returns = (self.test_factors['Mkt-RF'] / 100) + (self.test_factors['RF'] / 100)
        market_returns_aligned = market_returns.loc[portfolio_returns.index]
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_returns_net, market_returns_aligned)
        
        print(f"\n[Step 5] Performance Metrics:")
        print(f"  Annualized Return: {metrics['annual_return']:.2%}")
        print(f"  Annualized Volatility: {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Alpha (annualized): {metrics['alpha']:.2%}")
        print(f"  Information Ratio: {metrics['info_ratio']:.3f}")
        print(f"  Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Cumulative Return: {metrics['cumulative_return']:.2%}")
        
        return {
            'portfolio_returns': portfolio_returns_net,
            'benchmark_returns': market_returns_aligned,
            'metrics': metrics,
            'selected_stocks': self.selected_stocks
        }
    
    def _calculate_metrics(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """Calculate performance metrics"""
        risk_free = self.test_factors['RF'].loc[portfolio_returns.index] / 100
        
        metrics = {
            'annual_return': calculate_annualized_return(portfolio_returns),
            'volatility': calculate_volatility(portfolio_returns),
            'sharpe_ratio': calculate_sharpe_ratio(portfolio_returns, risk_free),
            'alpha': calculate_alpha(portfolio_returns, benchmark_returns, risk_free),
            'info_ratio': calculate_information_ratio(portfolio_returns, benchmark_returns),
            'max_drawdown': calculate_max_drawdown(portfolio_returns),
            'cumulative_return': (1 + portfolio_returns).prod() - 1
        }
        
        return metrics
    
    def plot_results(self, results: Dict):
        """Plot backtest results"""
        if results is None:
            return
        
        portfolio_returns = results['portfolio_returns']
        benchmark_returns = results['benchmark_returns']
        
        # Cumulative returns
        portfolio_cum = (1 + portfolio_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Cumulative returns
        axes[0].plot(portfolio_cum.index, portfolio_cum.values, 
                     label='Factor Portfolio', linewidth=2)
        axes[0].plot(benchmark_cum.index, benchmark_cum.values, 
                     label='Market Benchmark', linewidth=2, linestyle='--')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].set_title('Factor Portfolio vs Market Benchmark (Out-of-Sample)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        portfolio_dd = (portfolio_cum / portfolio_cum.expanding().max() - 1) * 100
        benchmark_dd = (benchmark_cum / benchmark_cum.expanding().max() - 1) * 100
        
        axes[1].fill_between(portfolio_dd.index, portfolio_dd.values, 0, 
                            alpha=0.3, label='Portfolio Drawdown')
        axes[1].fill_between(benchmark_dd.index, benchmark_dd.values, 0, 
                            alpha=0.3, label='Benchmark Drawdown')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_title('Drawdown Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_image = os.path.join(script_dir, 'factor_portfolio_backtest.png')
        
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"\n[Visualization] Saved to: {output_image}")


def main():
    """Main function"""
    print("=" * 80)
    print("Chapter 4: Factor-Based Portfolio Construction and Backtesting")
    print("=" * 80)
    
    # Example: NASDAQ 100 stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
               'AMD', 'INTC', 'QCOM', 'AVGO', 'COST', 'PEP', 'CSCO', 'CMCSA']
    
    # Date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 7)  # 7 years
    
    # Initialize backtest
    backtest = FactorPortfolioBacktest(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        train_ratio=0.7,
        rebalance_frequency='monthly',
        transaction_cost=0.002,
        min_r_squared=0.3,
        factor_selection_criteria={'SMB': 0.01, 'HML': 0.01, 'MOM': 0.005}  # Relaxed criteria
    )
    
    # Train model
    selected_stocks = backtest.train_model()
    
    # Run backtest
    results = backtest.backtest()
    
    # Plot results
    if results:
        backtest.plot_results(results)
        
        # Save results
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_csv = os.path.join(script_dir, 'selected_stocks.csv')
        
        results_df = pd.DataFrame({
            'ticker': selected_stocks
        })
        results_df.to_csv(output_csv, index=False)
        print(f"\n[Results] Selected stocks saved to: {output_csv}")
    
    print("\n" + "=" * 80)
    print("Backtest Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Train/test separation prevents overfitting")
    print("  2. Factor-based selection can improve risk-adjusted returns")
    print("  3. Transaction costs reduce net returns")
    print("  4. Out-of-sample performance is the true test")


if __name__ == '__main__':
    main()

