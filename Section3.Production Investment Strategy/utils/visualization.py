"""
Visualization utilities for trading simulation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime
import os


def create_trading_charts(test_data: pd.DataFrame,
                         target_index: str,
                         shares_history: List[int],
                         shares_dates: List,
                         results: List[float],
                         initial_capital: float,
                         commission_rate: float,
                         results_dir: str = 'results',
                         timestamp: str = None) -> str:
    """
    Create comprehensive trading simulation charts.
    
    Creates 3-panel chart:
    1. Price chart with holdings background (green gradient)
    2. Shares held (bar chart)
    3. Portfolio value vs Buy & Hold comparison
    
    Parameters:
    -----------
    test_data : pd.DataFrame
        Test period data
    target_index : str
        Target asset symbol
    shares_history : List[int]
        Shares held history
    shares_dates : List
        Dates for shares history
    results : List[float]
        Portfolio value history
    initial_capital : float
        Initial capital
    commission_rate : float
        Commission rate
    results_dir : str
        Results directory
    timestamp : str
        Timestamp for filename
        
    Returns:
    --------
    str
        Path to saved chart file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare chart data
    chart_df = pd.DataFrame({
        'date': [test_data.index[t] for t in range(len(test_data))],
        'price': [test_data[target_index].iloc[t] for t in range(len(test_data))],
    })
    
    # Match shares_history with dates
    shares_dict = dict(zip(shares_dates, shares_history))
    chart_df['shares'] = [shares_dict.get(test_data.index[t], 0) for t in range(len(test_data))]
    
    # Match portfolio_value with dates
    portfolio_value_list = []
    results_idx = 0
    for t in range(len(test_data)):
        current_date = test_data.index[t]
        if results_idx < len(results):
            portfolio_value_list.append(results[results_idx])
            if results_idx + 1 < len(results):
                results_idx += 1
        else:
            portfolio_value_list.append(results[-1] if results else initial_capital)
    chart_df['portfolio_value'] = portfolio_value_list
    chart_df.set_index('date', inplace=True)
    
    # Buy and Hold calculation
    initial_price = test_data[target_index].iloc[0]
    buy_hold_shares = initial_capital / initial_price
    buy_hold_commission_buy = buy_hold_shares * initial_price * commission_rate
    buy_hold_actual_shares = (initial_capital - buy_hold_commission_buy) / initial_price
    buy_hold_values = [buy_hold_actual_shares * price for price in chart_df['price']]
    chart_df['buy_hold_value'] = buy_hold_values
    
    # Normalize shares for background color intensity
    max_shares = chart_df['shares'].max() if chart_df['shares'].max() > 0 else 1
    chart_df['shares_normalized'] = chart_df['shares'] / max_shares  # 0~1 range
    
    # Create 3-panel vertical layout
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Panel 1: Price chart with holdings background
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.set_title(f'{target_index} Price with Holdings Background', fontsize=14, fontweight='bold')
    
    # Apply gradient background based on shares held
    for i in range(len(chart_df)):
        date = chart_df.index[i]
        shares_norm = chart_df['shares_normalized'].iloc[i]
        # Color intensity: max 0.35 alpha, min 0.15 alpha
        alpha_value = 0.15 + (shares_norm * 0.20)  # 0.15 ~ 0.35 range
        if i < len(chart_df) - 1:
            ax1.axvspan(date, chart_df.index[i+1], alpha=alpha_value, color='green', zorder=0)
    
    # Price line
    ax1.plot(chart_df.index, chart_df['price'], label=f'{target_index} Price', 
             linewidth=2, color='black', zorder=2)
    ax1.grid(True, alpha=0.3, zorder=1)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Panel 2: Shares held bar chart
    ax2.set_ylabel('Shares Held', fontsize=12)
    ax2.set_title('Holdings Quantity', fontsize=14, fontweight='bold')
    ax2.bar(chart_df.index, chart_df['shares'], width=1, alpha=0.6, color='orange', label='Shares Held')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper left', fontsize=10)
    
    # Panel 3: Portfolio value vs Buy & Hold
    ax3.set_ylabel('Portfolio Value (USD)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Portfolio Value vs Buy & Hold', fontsize=14, fontweight='bold')
    # Strategy portfolio value
    ax3.plot(chart_df.index, chart_df['portfolio_value'], label='Strategy Portfolio', 
             linewidth=2, color='blue')
    # Buy & Hold
    ax3.plot(chart_df.index, chart_df['buy_hold_value'], label='Buy & Hold', 
             linewidth=2, color='red', linestyle='--')
    # Initial capital baseline
    ax3.axhline(y=initial_capital, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Initial Capital')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=10)
    
    # Performance comparison text (bottom right)
    strategy_return = (chart_df['portfolio_value'].iloc[-1] - initial_capital) / initial_capital * 100
    buy_hold_return = (chart_df['buy_hold_value'].iloc[-1] - initial_capital) / initial_capital * 100
    outperformance = strategy_return - buy_hold_return
    
    ax3.text(0.98, 0.02, 
             f'Strategy: {strategy_return:.2f}%\nBuy & Hold: {buy_hold_return:.2f}%\nOutperformance: {outperformance:+.2f}%',
             transform=ax3.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save chart
    os.makedirs(results_dir, exist_ok=True)
    chart_path = os.path.join(results_dir, f'trading_simulation_chart_{timestamp}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    
    return chart_path

