"""
Chapter 7: Copula Models

This script demonstrates copula modeling for:
1. Multivariate dependence modeling
2. Portfolio risk management
3. Tail dependence analysis
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

warnings.filterwarnings("ignore")

try:
    from copulae import GaussianCopula, StudentCopula
    HAS_COPULAE = True
except ImportError:
    HAS_COPULAE = False
    print("Warning: copulae not available. Using simplified copula approach.")


def estimate_marginal_distribution(returns):
    """
    Estimate marginal distribution (t-distribution)
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    
    Returns:
    --------
    dict
        Distribution parameters
    """
    # Fit t-distribution
    params = stats.t.fit(returns.dropna())
    df, loc, scale = params
    
    return {
        'dist': 't',
        'df': df,
        'loc': loc,
        'scale': scale
    }


def transform_to_uniform(returns, marginal_params):
    """
    Transform returns to uniform marginals using CDF
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    marginal_params : dict
        Marginal distribution parameters
    
    Returns:
    --------
    pd.Series
        Uniform marginals
    """
    if marginal_params['dist'] == 't':
        cdf_values = stats.t.cdf(returns, 
                                 df=marginal_params['df'],
                                 loc=marginal_params['loc'],
                                 scale=marginal_params['scale'])
    else:
        # Fallback to normal
        cdf_values = stats.norm.cdf(returns, 
                                    loc=marginal_params.get('loc', returns.mean()),
                                    scale=marginal_params.get('scale', returns.std()))
    
    return pd.Series(cdf_values, index=returns.index)


def copula_portfolio_risk(returns1, returns2, copula_type='gaussian'):
    """
    Analyze portfolio risk using copula
    
    Parameters:
    -----------
    returns1 : pd.Series
        First return series
    returns2 : pd.Series
        Second return series
    copula_type : str
        Type of copula ('gaussian' or 'student')
    
    Returns:
    --------
    dict
        Copula analysis results
    """
    # Align data
    ret1, ret2 = align_data(returns1, returns2)
    
    # Estimate marginal distributions
    marginal1 = estimate_marginal_distribution(ret1)
    marginal2 = estimate_marginal_distribution(ret2)
    
    # Transform to uniform marginals
    u1 = transform_to_uniform(ret1, marginal1)
    u2 = transform_to_uniform(ret2, marginal2)
    
    u_data = np.column_stack([u1.values, u2.values])
    
    if HAS_COPULAE:
        try:
            # Fit copula
            if copula_type == 'gaussian':
                copula = GaussianCopula(dim=2)
            elif copula_type == 'student':
                copula = StudentCopula(dim=2)
            else:
                copula = GaussianCopula(dim=2)
            
            copula.fit(u_data)
            
            # Simulate from copula
            n_sim = 10000
            simulated_u = copula.random(n_sim)
            
            # Transform back to original scale
            simulated_ret1 = stats.t.ppf(simulated_u[:, 0],
                                        df=marginal1['df'],
                                        loc=marginal1['loc'],
                                        scale=marginal1['scale'])
            simulated_ret2 = stats.t.ppf(simulated_u[:, 1],
                                        df=marginal2['df'],
                                        loc=marginal2['loc'],
                                        scale=marginal2['scale'])
            
            return {
                'copula': copula,
                'marginal1': marginal1,
                'marginal2': marginal2,
                'simulated_ret1': simulated_ret1,
                'simulated_ret2': simulated_ret2,
                'u_data': u_data,
                'method': 'copula'
            }
        except Exception as e:
            print(f"Error in copula fitting: {e}")
            # Fallback to correlation-based approach
            pass
    
    # Simplified approach: use correlation
    correlation = np.corrcoef(ret1, ret2)[0, 1]
    
    # Simulate using correlation
    n_sim = 10000
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    simulated_norm = np.random.multivariate_normal(mean, cov, n_sim)
    simulated_u = stats.norm.cdf(simulated_norm)
    
    # Transform back
    simulated_ret1 = stats.t.ppf(simulated_u[:, 0],
                                 df=marginal1['df'],
                                 loc=marginal1['loc'],
                                 scale=marginal1['scale'])
    simulated_ret2 = stats.t.ppf(simulated_u[:, 1],
                                 df=marginal2['df'],
                                 loc=marginal2['loc'],
                                 scale=marginal2['scale'])
    
    return {
        'correlation': correlation,
        'marginal1': marginal1,
        'marginal2': marginal2,
        'simulated_ret1': simulated_ret1,
        'simulated_ret2': simulated_ret2,
        'method': 'correlation'
    }


def calculate_portfolio_risk(simulated_ret1, simulated_ret2, weights=[0.5, 0.5]):
    """
    Calculate portfolio risk metrics
    
    Parameters:
    -----------
    simulated_ret1 : np.array
        Simulated returns for asset 1
    simulated_ret2 : np.array
        Simulated returns for asset 2
    weights : list
        Portfolio weights
    
    Returns:
    --------
    dict
        Risk metrics
    """
    portfolio_returns = weights[0] * simulated_ret1 + weights[1] * simulated_ret2
    
    # VaR and CVaR
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
    
    return {
        'portfolio_returns': portfolio_returns,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'mean': portfolio_returns.mean(),
        'std': portfolio_returns.std()
    }


def visualize_copula_results(results, nasdaq_returns, tqqq_returns):
    """Visualize copula analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter plot: actual returns
    axes[0, 0].scatter(nasdaq_returns.values, tqqq_returns.values, 
                      alpha=0.5, s=10, color='blue', label='Actual')
    axes[0, 0].set_xlabel('NASDAQ Returns', fontsize=12)
    axes[0, 0].set_ylabel('TQQQ Returns', fontsize=12)
    axes[0, 0].set_title('Actual Returns: NASDAQ vs TQQQ', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Scatter plot: simulated returns
    axes[0, 1].scatter(results['simulated_ret1'], results['simulated_ret2'], 
                      alpha=0.3, s=5, color='red', label='Simulated (Copula)')
    axes[0, 1].set_xlabel('NASDAQ Returns (Simulated)', fontsize=12)
    axes[0, 1].set_ylabel('TQQQ Returns (Simulated)', fontsize=12)
    axes[0, 1].set_title('Simulated Returns from Copula', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Portfolio risk distribution
    portfolio_risk = calculate_portfolio_risk(results['simulated_ret1'], 
                                             results['simulated_ret2'])
    
    axes[1, 0].hist(portfolio_risk['portfolio_returns'], bins=100, 
                   alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(portfolio_risk['var_95'], color='red', linestyle='--', 
                      linewidth=2, label=f"VaR 95%: {portfolio_risk['var_95']:.4f}")
    axes[1, 0].axvline(portfolio_risk['var_99'], color='darkred', linestyle='--', 
                      linewidth=2, label=f"VaR 99%: {portfolio_risk['var_99']:.4f}")
    axes[1, 0].set_xlabel('Portfolio Returns', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Portfolio Return Distribution (50/50 Portfolio)', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Tail dependence visualization
    # Lower tail: both returns below their 5th percentile
    lower_tail_nasdaq = np.percentile(results['simulated_ret1'], 5)
    lower_tail_tqqq = np.percentile(results['simulated_ret2'], 5)
    lower_tail_mask = (results['simulated_ret1'] < lower_tail_nasdaq) & \
                     (results['simulated_ret2'] < lower_tail_tqqq)
    lower_tail_prob = lower_tail_mask.sum() / len(results['simulated_ret1'])
    
    # Upper tail: both returns above their 95th percentile
    upper_tail_nasdaq = np.percentile(results['simulated_ret1'], 95)
    upper_tail_tqqq = np.percentile(results['simulated_ret2'], 95)
    upper_tail_mask = (results['simulated_ret1'] > upper_tail_nasdaq) & \
                     (results['simulated_ret2'] > upper_tail_tqqq)
    upper_tail_prob = upper_tail_mask.sum() / len(results['simulated_ret1'])
    
    axes[1, 1].bar(['Lower Tail', 'Upper Tail'], 
                  [lower_tail_prob, upper_tail_prob],
                  color=['red', 'green'], alpha=0.7)
    axes[1, 1].set_ylabel('Joint Probability', fontsize=12)
    axes[1, 1].set_title('Tail Dependence Analysis', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Copula Analysis Statistics")
    print(f"{'='*60}")
    if results['method'] == 'copula':
        print(f"Copula Type: {type(results['copula']).__name__}")
    else:
        print(f"Method: Correlation-based")
        print(f"Correlation: {results['correlation']:.4f}")
    
    print(f"\nPortfolio Risk Metrics (50/50 Portfolio):")
    print(f"  Mean Return: {portfolio_risk['mean']:.6f}")
    print(f"  Std Return: {portfolio_risk['std']:.6f}")
    print(f"  VaR (95%): {portfolio_risk['var_95']:.6f}")
    print(f"  CVaR (95%): {portfolio_risk['cvar_95']:.6f}")
    print(f"  VaR (99%): {portfolio_risk['var_99']:.6f}")
    print(f"  CVaR (99%): {portfolio_risk['cvar_99']:.6f}")
    
    print(f"\nTail Dependence:")
    print(f"  Lower Tail Probability: {lower_tail_prob:.4f}")
    print(f"  Upper Tail Probability: {upper_tail_prob:.4f}")


def main():
    """Main function"""
    print("="*60)
    print("Chapter 7: Copula Models")
    print("Multivariate Dependence Analysis")
    print("="*60)
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    nasdaq = data['nasdaq']
    tqqq = data['tqqq']
    
    nasdaq_returns = nasdaq['Returns'].dropna()
    tqqq_returns = tqqq['Returns'].dropna()
    
    # Apply copula analysis
    print("\n1. Applying copula model to NASDAQ-TQQQ relationship...")
    results = copula_portfolio_risk(nasdaq_returns, tqqq_returns, copula_type='gaussian')
    
    # Visualize
    visualize_copula_results(results, nasdaq_returns, tqqq_returns)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

