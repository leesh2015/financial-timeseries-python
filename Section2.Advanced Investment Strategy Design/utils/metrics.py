"""
Performance metrics for trading strategies
"""

import numpy as np
from typing import Tuple, Dict, Optional


def max_drawdown(results: list) -> Tuple[float, float]:
    """
    Calculate maximum drawdown from trading results.
    
    Maximum Drawdown: The maximum peak-to-trough decline in portfolio value.
    Formula: MD = max(Peak_t - Value_t) for all t
    
    Parameters:
    -----------
    results : list
        List of portfolio values over time
        
    Returns:
    --------
    tuple: (peak_before_max_drawdown, max_drawdown)
        Peak value before max drawdown and maximum drawdown amount
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


def max_loss(results: list) -> int:
    """
    Calculate maximum consecutive loss days.
    
    Parameters:
    -----------
    results : list
        List of portfolio values over time
        
    Returns:
    --------
    int
        Maximum number of consecutive days with losses
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
    
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    return max_loss_streak


def calculate_performance_metrics(
    results: list,
    initial_capital: float,
    trading_days_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Metrics:
    - Total return
    - Annualized return
    - Sharpe ratio
    - Maximum drawdown
    - Maximum consecutive loss days
    
    Parameters:
    -----------
    results : list
        List of portfolio values over time
    initial_capital : float
        Initial capital
    trading_days_per_year : int
        Number of trading days per year (default: 252)
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    if len(results) < 2:
        return {}
    
    # Calculate returns
    returns = np.diff(results) / results[:-1]
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        return {}
    
    # Basic metrics
    final_capital = results[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    
    # Annualized metrics
    annualized_return = mean_return * trading_days_per_year
    annualized_std = std_return * np.sqrt(trading_days_per_year)
    sharpe_ratio = annualized_return / annualized_std if annualized_std > 0 else 0.0
    
    # Drawdown metrics
    peak, max_dd = max_drawdown(results)
    max_loss_days = max_loss(results)
    
    metrics = {
        'final_capital': final_capital,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_std': annualized_std,
        'sharpe_ratio': sharpe_ratio,
        'peak': peak,
        'max_drawdown': max_dd,
        'max_consecutive_loss_days': max_loss_days
    }
    
    return metrics

