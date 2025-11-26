"""
Performance metrics calculation utilities
"""

import numpy as np
import pandas as pd
from typing import Optional


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: Optional[pd.Series] = None,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    risk_free_rate : pd.Series, optional
        Risk-free rate (same frequency as returns)
    periods_per_year : int
        Number of periods per year (252 for daily, 12 for monthly)
    
    Returns:
    --------
    float
        Annualized Sharpe ratio
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    if risk_free_rate is not None:
        risk_free_aligned = risk_free_rate.loc[returns_clean.index]
        excess_returns = returns_clean - risk_free_aligned
    else:
        excess_returns = returns_clean
    
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    if std_excess == 0:
        return np.nan
    
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return sharpe


def calculate_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: Optional[pd.Series] = None,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Jensen's alpha (annualized)
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns (e.g., market returns)
    risk_free_rate : pd.Series, optional
        Risk-free rate
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float
        Annualized alpha
    """
    # Align data
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    port_ret = portfolio_returns.loc[common_dates]
    bench_ret = benchmark_returns.loc[common_dates]
    
    if risk_free_rate is not None:
        rf_aligned = risk_free_rate.loc[common_dates]
        port_excess = port_ret - rf_aligned
        bench_excess = bench_ret - rf_aligned
    else:
        port_excess = port_ret
        bench_excess = bench_ret
    
    # Calculate beta
    covariance = np.cov(port_excess, bench_excess)[0, 1]
    variance = np.var(bench_excess)
    
    if variance == 0:
        return np.nan
    
    beta = covariance / variance
    
    # Calculate alpha
    alpha = port_excess.mean() - beta * bench_excess.mean()
    
    # Annualize
    alpha_annualized = alpha * periods_per_year
    
    return alpha_annualized


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate information ratio (annualized)
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float
        Annualized information ratio
    """
    # Align data
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    port_ret = portfolio_returns.loc[common_dates]
    bench_ret = benchmark_returns.loc[common_dates]
    
    # Calculate tracking error
    active_returns = port_ret - bench_ret
    tracking_error = active_returns.std()
    
    if tracking_error == 0:
        return np.nan
    
    # Calculate information ratio
    mean_active_return = active_returns.mean()
    info_ratio = (mean_active_return / tracking_error) * np.sqrt(periods_per_year)
    
    return info_ratio


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    
    Returns:
    --------
    float
        Maximum drawdown (negative value)
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Calculate cumulative returns
    cumulative = (1 + returns_clean).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    
    return max_dd


def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float
        Annualized return
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Calculate cumulative return
    cumulative_return = (1 + returns_clean).prod()
    
    # Annualize
    n_periods = len(returns_clean)
    annualized = (cumulative_return ** (periods_per_year / n_periods)) - 1
    
    return annualized


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float
        Annualized volatility
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    volatility = returns_clean.std() * np.sqrt(periods_per_year)
    
    return volatility

