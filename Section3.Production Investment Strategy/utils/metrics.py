"""
Performance metrics calculation utilities
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


def max_drawdown(results: List[float]) -> Tuple[float, float]:
    """
    Calculate maximum drawdown from portfolio value history.
    
    Maximum Drawdown Formula:
    MDD = max(Peak_t - Value_t) for all t
    
    where:
    - Peak_t: maximum value up to time t
    - Value_t: portfolio value at time t
    
    Parameters:
    -----------
    results : List[float]
        Portfolio value history
        
    Returns:
    --------
    Tuple[float, float]
        (peak_before_max_drawdown, max_drawdown)
    """
    drawdowns = []
    peak = results[0]
    peak_before_max_drawdown = peak
    max_drawdown = 0

    for value in results:
        if value > peak:
            peak = value
        drawdown = peak - value
        drawdowns.append(drawdown)
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            peak_before_max_drawdown = peak
    
    return peak_before_max_drawdown, max_drawdown


def max_loss(results: List[float]) -> int:
    """
    Calculate maximum consecutive loss days.
    
    Parameters:
    -----------
    results : List[float]
        Portfolio value history
        
    Returns:
    --------
    int
        Maximum consecutive loss days
    """
    loss_streaks = []
    current_streak = 0
    for i in range(1, len(results)):
        if results[i] < results[i-1]:
            current_streak += 1
        else:
            loss_streaks.append(current_streak)
            current_streak = 0
    loss_streaks.append(current_streak)
    max_loss_streak = max(loss_streaks)
    return max_loss_streak


def calculate_performance_metrics(results: List[float], 
                                  initial_capital: float,
                                  simulation_start_date: pd.Timestamp,
                                  simulation_end_date: pd.Timestamp,
                                  trading_days_per_year: int = 252) -> Dict:
    """
    Calculate comprehensive performance metrics.
    
    Metrics Formulas:
    - Total Return: R_total = (V_T - V_0) / V_0
    - Annualized Return: R_annual = (1 + R_total)^(1/T) - 1
    - Annualized Std: σ_annual = σ_daily * √252
    - Sharpe Ratio: SR = (R_annual - r_f) / σ_annual
    
    where:
    - V_T: final portfolio value
    - V_0: initial capital
    - T: simulation period in years
    - r_f: risk-free rate (assumed 0)
    
    Parameters:
    -----------
    results : List[float]
        Portfolio value history
    initial_capital : float
        Initial capital
    simulation_start_date : pd.Timestamp
        Simulation start date
    simulation_end_date : pd.Timestamp
        Simulation end date
    trading_days_per_year : int
        Trading days per year (default: 252)
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    if len(results) < 2:
        return {}
    
    # Calculate returns
    returns = np.diff(results) / results[:-1]
    
    # Annualized return
    annualized_return = np.mean(returns) * trading_days_per_year
    
    # Annualized standard deviation
    annualized_std_return = np.std(returns) * np.sqrt(trading_days_per_year)
    
    # Sharpe ratio
    sharpe_ratio = annualized_return / annualized_std_return if annualized_std_return > 0 else 0.0
    
    # Maximum drawdown
    peak, max_dd = max_drawdown(results)
    
    # Maximum consecutive loss days
    max_loss_streak = max_loss(results)
    
    # Total return
    total_return = (results[-1] - initial_capital) / initial_capital
    
    # Time-based annualized return
    simulation_days = (simulation_end_date - simulation_start_date).days
    simulation_years = simulation_days / 365.25
    annualized_return_time = (1 + total_return) ** (1 / simulation_years) - 1 if simulation_years > 0 else 0.0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_return_time': annualized_return_time,
        'annualized_std_return': annualized_std_return,
        'sharpe_ratio': sharpe_ratio,
        'peak': peak,
        'max_drawdown': max_dd,
        'max_consecutive_loss_days': max_loss_streak,
        'final_capital': results[-1]
    }


def calculate_buy_hold_metrics(test_data: pd.DataFrame,
                               target_index: str,
                               initial_capital: float,
                               commission_rate: float,
                               simulation_start_date: pd.Timestamp,
                               simulation_end_date: pd.Timestamp,
                               trading_days_per_year: int = 252) -> Dict:
    """
    Calculate buy-and-hold performance metrics.
    
    Buy & Hold Strategy:
    - Buy at initial price: Shares = (Capital - Commission) / Initial_Price
    - Hold until end: Final_Value = Shares * Final_Price
    
    Parameters:
    -----------
    test_data : pd.DataFrame
        Test period data
    target_index : str
        Target asset symbol
    initial_capital : float
        Initial capital
    commission_rate : float
        Commission rate
    simulation_start_date : pd.Timestamp
        Simulation start date
    simulation_end_date : pd.Timestamp
        Simulation end date
    trading_days_per_year : int
        Trading days per year
        
    Returns:
    --------
    dict
        Buy-and-hold performance metrics
    """
    initial_price = test_data[target_index].iloc[0]
    final_price = test_data[target_index].iloc[-1]
    
    # Buy and hold with commission
    buy_hold_shares = initial_capital / initial_price
    buy_hold_commission = buy_hold_shares * initial_price * commission_rate
    buy_hold_actual_shares = (initial_capital - buy_hold_commission) / initial_price
    buy_hold_final_value = buy_hold_actual_shares * final_price
    
    # Returns
    bnh_total_return = (final_price - initial_price) / initial_price
    
    # Annualized return
    simulation_days = (simulation_end_date - simulation_start_date).days
    simulation_years = simulation_days / 365.25
    bnh_annualized_return = (1 + bnh_total_return) ** (1 / simulation_years) - 1 if simulation_years > 0 else 0.0
    
    # Buy and hold returns for Sharpe
    bnh_returns = test_data[target_index].pct_change().dropna()
    bnh_annualized_std = bnh_returns.std() * np.sqrt(trading_days_per_year)
    bnh_sharpe_ratio = (bnh_annualized_return / bnh_annualized_std) if bnh_annualized_std > 0 else 0
    
    # Buy and hold drawdown
    bnh_values = test_data[target_index].values
    bnh_peak, bnh_max_drawdown = max_drawdown(bnh_values)
    
    return {
        'initial_price': initial_price,
        'final_price': final_price,
        'total_return': bnh_total_return,
        'annualized_return': bnh_annualized_return,
        'annualized_std': bnh_annualized_std,
        'sharpe_ratio': bnh_sharpe_ratio,
        'peak': bnh_peak,
        'max_drawdown': bnh_max_drawdown,
        'final_value': buy_hold_final_value
    }

