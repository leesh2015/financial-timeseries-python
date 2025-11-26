"""
Factor utility functions for calculating exposures and constructing portfolios
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
from scipy import stats

warnings.filterwarnings("ignore")


def calculate_factor_exposures(
    stock_returns: pd.Series,
    factor_data: pd.DataFrame,
    risk_free_col: str = 'RF',
    factor_cols: List[str] = None
) -> Dict[str, float]:
    """
    Calculate factor exposures (betas) using OLS regression
    
    Parameters:
    -----------
    stock_returns : pd.Series
        Stock excess returns (stock return - risk-free rate)
    factor_data : pd.DataFrame
        Fama-French factor data
    risk_free_col : str
        Column name for risk-free rate
    factor_cols : List[str]
        List of factor column names (e.g., ['Mkt-RF', 'SMB', 'HML'])
        If None, uses all columns except risk_free_col
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with factor exposures (betas) and statistics
    """
    # Align data
    common_dates = stock_returns.index.intersection(factor_data.index)
    
    if len(common_dates) < 30:
        raise ValueError("Insufficient data for regression (need at least 30 observations)")
    
    # Get aligned data - ensure both have same index
    stock_ret = stock_returns.loc[common_dates]
    factors = factor_data.loc[common_dates]
    
    # Ensure indices are exactly aligned
    stock_ret = stock_ret.loc[common_dates]
    factors = factors.loc[common_dates]
    
    # Calculate excess returns
    if risk_free_col in factors.columns:
        # Convert annual percentage to daily decimal
        # Fama-French RF is in annual percentage (e.g., 5.0 means 5%)
        risk_free = factors[risk_free_col] / 100
        
        # Ensure both have same index before subtraction
        common_idx = stock_ret.index.intersection(risk_free.index)
        
        if len(common_idx) < 30:
            raise ValueError("Insufficient aligned data after risk-free rate calculation")
        
        # Sort index to ensure alignment
        common_idx = sorted(common_idx)
        
        stock_ret_aligned = stock_ret.loc[common_idx]
        risk_free_aligned = risk_free.loc[common_idx]
        
        excess_returns = stock_ret_aligned - risk_free_aligned
        
        # Update factors to match excess_returns index
        factors = factors.loc[common_idx]
    else:
        excess_returns = stock_ret
    
    # Select factor columns
    if factor_cols is None:
        factor_cols = [col for col in factors.columns if col != risk_free_col]
    
    # Prepare regression data - ensure same length and sorted index
    final_idx = excess_returns.index.intersection(factors.index)
    final_idx = sorted(final_idx)
    
    X = factors[factor_cols].loc[final_idx].values / 100
    y = excess_returns.loc[final_idx].values
    
    # Remove NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) < 30:
        raise ValueError("Insufficient clean data for regression")
    
    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X_clean)), X_clean])
    
    # OLS regression
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y_clean, rcond=None)
    except np.linalg.LinAlgError:
        raise ValueError("Regression failed: singular matrix")
    
    # Calculate R-squared
    y_pred = X_with_const @ coeffs
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate standard errors and t-statistics
    n = len(y_clean)
    k = len(coeffs)
    mse = ss_res / (n - k) if (n - k) > 0 else 0
    
    if mse > 0 and rank == k:
        var_coeffs = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se_coeffs = np.sqrt(np.diag(var_coeffs))
        t_stats = coeffs / se_coeffs
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
    else:
        se_coeffs = np.full(len(coeffs), np.nan)
        t_stats = np.full(len(coeffs), np.nan)
        p_values = np.full(len(coeffs), np.nan)
    
    # Build result dictionary
    result = {
        'alpha': coeffs[0],
        'alpha_se': se_coeffs[0],
        'alpha_tstat': t_stats[0],
        'alpha_pvalue': p_values[0],
        'r_squared': r_squared,
        'n_observations': n,
    }
    
    # Add factor exposures
    for i, factor_name in enumerate(factor_cols):
        result[f'{factor_name}_beta'] = coeffs[i + 1]
        result[f'{factor_name}_se'] = se_coeffs[i + 1]
        result[f'{factor_name}_tstat'] = t_stats[i + 1]
        result[f'{factor_name}_pvalue'] = p_values[i + 1]
    
    return result


def construct_factor_portfolio(
    stock_returns: pd.DataFrame,
    factor_data: pd.DataFrame,
    factor_cols: List[str],
    selection_criteria: Dict[str, float],
    min_r_squared: float = 0.2
) -> List[str]:
    """
    Construct a factor-based portfolio by selecting stocks based on factor exposures
    
    Parameters:
    -----------
    stock_returns : pd.DataFrame
        DataFrame with stock returns (columns: tickers, index: dates)
    factor_data : pd.DataFrame
        Fama-French factor data
    factor_cols : List[str]
        List of factor column names
    selection_criteria : Dict[str, float]
        Dictionary with factor names as keys and threshold values
        Example: {'SMB': 0.1, 'HML': 0.05} means SMB beta > 0.1 and HML beta > 0.05
    min_r_squared : float
        Minimum R-squared for model fit
    
    Returns:
    --------
    List[str]
        List of selected ticker symbols
    """
    selected_tickers = []
    
    # Align data
    common_dates = stock_returns.index.intersection(factor_data.index)
    stock_ret_aligned = stock_returns.loc[common_dates]
    factor_data_aligned = factor_data.loc[common_dates]
    
    # Calculate factor exposures for each stock
    for ticker in stock_ret_aligned.columns:
        try:
            stock_ret = stock_ret_aligned[ticker].dropna()
            if len(stock_ret) < 30:
                continue
            
            exposures = calculate_factor_exposures(
                stock_ret,
                factor_data_aligned,
                factor_cols=factor_cols
            )
            
            # Check R-squared
            if exposures['r_squared'] < min_r_squared:
                continue
            
            # Check factor exposure criteria
            meets_criteria = True
            for factor_name, threshold in selection_criteria.items():
                beta_key = f'{factor_name}_beta'
                if beta_key not in exposures:
                    meets_criteria = False
                    break
                if exposures[beta_key] <= threshold:
                    meets_criteria = False
                    break
            
            if meets_criteria:
                selected_tickers.append(ticker)
        
        except Exception:
            continue
    
    return selected_tickers

