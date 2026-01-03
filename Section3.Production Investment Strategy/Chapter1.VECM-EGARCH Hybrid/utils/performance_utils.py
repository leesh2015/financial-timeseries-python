"""
Performance metrics utilities
"""

import numpy as np
from typing import Tuple, List

def max_drawdown(results: List[float]) -> Tuple[float, float]:
    """
    Calculate maximum drawdown from a series of portfolio values.
    
    Parameters:
    -----------
    results : List[float]
        List of portfolio values (or cumulative returns)
        
    Returns:
    --------
    Tuple[float, float]
        (peak_before_max_drawdown, max_drawdown_amount)
    """
    drawdowns = []
    if len(results) == 0:
        return 0.0, 0.0
        
    peak = results[0]
    peak_before_max_drawdown = peak
    max_drawdown_val = 0.0

    for value in results:
        if value > peak:
            peak = value
        drawdown = peak - value
        drawdowns.append(drawdown)
        if drawdown > max_drawdown_val:
            max_drawdown_val = drawdown
            peak_before_max_drawdown = peak
    
    return peak_before_max_drawdown, max_drawdown_val


def max_loss(results: List[float]) -> int:
    """
    Calculate maximum consecutive loss streak.
    
    Parameters:
    -----------
    results : List[float]
        List of portfolio values
        
    Returns:
    --------
    int
        Maximum number of consecutive days with losses
    """
    if len(results) < 2:
        return 0
        
    loss_streaks = []
    current_streak = 0
    for i in range(1, len(results)):
        if results[i] < results[i-1]:
            current_streak += 1
        else:
            loss_streaks.append(current_streak)
            current_streak = 0
    loss_streaks.append(current_streak)
    
    if not loss_streaks:
        return 0
        
    max_loss_streak = max(loss_streaks)
    return max_loss_streak
