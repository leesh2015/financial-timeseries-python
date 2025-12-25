"""
Chapter 1: Linear Algebra - Portfolio Optimization Example

Core Analogy: "Cocktail Recipe"
- Vector: Amount of each ingredient [30ml gin, 90ml tonic, 10ml lime]
- Matrix: "Compatibility table" showing how ingredients work together (covariance matrix)
- Portfolio variance: σ² = wᵀΣw (How strong is this cocktail?)

This example demonstrates:
1. Understanding basic vector/matrix operations
2. Calculating covariance matrices
3. Computing portfolio variance
4. Implementing Mean-Variance Optimization (MVO)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import warnings
import os
warnings.filterwarnings('ignore')

# Korean font settings (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def demonstrate_basic_operations():
    """
    1. Understanding Basic Vector and Matrix Operations
    """
    print("=" * 60)
    print("1. Basic Vector and Matrix Operations")
    print("=" * 60)
    
    # Vector: weights for each asset
    # Example: [Apple 30%, Google 50%, Microsoft 20%]
    weights = np.array([0.3, 0.5, 0.2])
    print(f"\nPortfolio weights (vector): {weights}")
    print(f"Sum of weights: {weights.sum():.2f} (must be 1)")
    
    # Expected returns vector
    expected_returns = np.array([0.10, 0.12, 0.08])  # Annual returns
    print(f"\nExpected returns (vector): {expected_returns}")
    
    # Portfolio expected return: wᵀμ = Σ(w_i × μ_i)
    portfolio_return = np.dot(weights, expected_returns)
    print(f"Portfolio expected return: {portfolio_return:.4f} ({portfolio_return*100:.2f}%)")
    print("  → Dot product of vectors: sum of individual contributions")
    
    # Covariance matrix (correlation and volatility between assets)
    # Diagonal: variance of each asset (volatility²)
    # Off-diagonal: covariance between assets (correlation × volatility1 × volatility2)
    cov_matrix = np.array([
        [0.04, 0.02, 0.01],  # Apple's variance and covariance with other assets
        [0.02, 0.05, 0.02],  # Google
        [0.01, 0.02, 0.03]   # Microsoft
    ])
    print(f"\nCovariance matrix (Σ):")
    print(cov_matrix)
    print("  → Diagonal: variance of each asset (volatility²)")
    print("  → Off-diagonal: covariance between assets (correlation)")
    
    # Portfolio variance: σ² = wᵀΣw
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    print(f"\nPortfolio variance: {portfolio_variance:.6f}")
    print(f"Portfolio standard deviation (risk): {portfolio_std:.4f} ({portfolio_std*100:.2f}%)")
    print("  → Matrix multiplication: wᵀΣw = ΣΣ(w_i × w_j × σ_ij)")
    
    return weights, expected_returns, cov_matrix


def calculate_covariance_matrix_from_data(tickers, start_date='2020-01-01', end_date='2024-01-01'):
    """
    2. Calculate Covariance Matrix from Real Financial Data
    """
    print("\n" + "=" * 60)
    print("2. Calculate Covariance Matrix from Real Financial Data")
    print("=" * 60)
    
    # Download data
    print(f"\nDownloading {tickers} data...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Calculate returns
    returns = data.pct_change().dropna()
    print(f"Data period: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"Number of data points: {len(returns)}")
    
    # Calculate covariance matrix
    cov_matrix = returns.cov().values
    print(f"\nCovariance matrix (annualized):")
    print(pd.DataFrame(cov_matrix * 252, index=tickers, columns=tickers))
    
    # Correlation matrix (standardized covariance)
    corr_matrix = returns.corr()
    print(f"\nCorrelation matrix:")
    print(corr_matrix)
    
    return returns, cov_matrix, corr_matrix


def portfolio_variance(weights, cov_matrix):
    """
    Calculate portfolio variance: σ² = wᵀΣw
    """
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def portfolio_return(weights, expected_returns):
    """
    Calculate portfolio expected return: μ = wᵀμ
    """
    return np.dot(weights, expected_returns)


def mean_variance_optimization(returns, risk_free_rate=0.02):
    """
    3. Implement Mean-Variance Optimization (MVO)
    
    Objective: Find maximum return for a given risk level
    Or find minimum risk for a given return level
    """
    print("\n" + "=" * 60)
    print("3. Mean-Variance Optimization (MVO)")
    print("=" * 60)
    
    # Annualized expected returns and covariance matrix
    expected_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    
    n_assets = len(returns.columns)
    
    # Constraint: sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: each asset weight 0~1
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Minimum Variance Portfolio
    print("\n[Minimum Variance Portfolio]")
    result_min_var = minimize(
        portfolio_variance,
        x0=np.array([1/n_assets] * n_assets),
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    min_var_weights = result_min_var.x
    min_var_return = portfolio_return(min_var_weights, expected_returns)
    min_var_std = np.sqrt(portfolio_variance(min_var_weights, cov_matrix))
    
    print(f"Weights: {dict(zip(returns.columns, min_var_weights))}")
    print(f"Expected return: {min_var_return:.4f} ({min_var_return*100:.2f}%)")
    print(f"Standard deviation (risk): {min_var_std:.4f} ({min_var_std*100:.2f}%)")
    print(f"Sharpe ratio: {(min_var_return - risk_free_rate) / min_var_std:.4f}")
    
    # Maximum Sharpe Ratio Portfolio (Tangency Portfolio)
    print("\n[Maximum Sharpe Ratio Portfolio]")
    def negative_sharpe(weights):
        port_return = portfolio_return(weights, expected_returns)
        port_std = np.sqrt(portfolio_variance(weights, cov_matrix))
        return -(port_return - risk_free_rate) / port_std
    
    result_max_sharpe = minimize(
        negative_sharpe,
        x0=np.array([1/n_assets] * n_assets),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    max_sharpe_weights = result_max_sharpe.x
    max_sharpe_return = portfolio_return(max_sharpe_weights, expected_returns)
    max_sharpe_std = np.sqrt(portfolio_variance(max_sharpe_weights, cov_matrix))
    max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_std
    
    print(f"Weights: {dict(zip(returns.columns, max_sharpe_weights))}")
    print(f"Expected return: {max_sharpe_return:.4f} ({max_sharpe_return*100:.2f}%)")
    print(f"Standard deviation (risk): {max_sharpe_std:.4f} ({max_sharpe_std*100:.2f}%)")
    print(f"Sharpe ratio: {max_sharpe_ratio:.4f}")
    
    # Calculate efficient frontier
    print("\n[Calculating Efficient Frontier...]")
    # Limit target returns to reasonable range (not exceeding max individual asset return)
    max_individual_return = np.max(expected_returns)
    target_returns = np.linspace(min_var_return, min(max_sharpe_return * 1.2, max_individual_return), 50)
    efficient_portfolios = []
    
    for target_return in target_returns:
        constraints_with_return = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_return(w, expected_returns) - target_return}
        ]
        
        result = minimize(
            portfolio_variance,
            x0=np.array([1/n_assets] * n_assets),
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_with_return
        )
        
        if result.success:
            port_std = np.sqrt(result.fun)
            efficient_portfolios.append((port_std, target_return))
    
    efficient_portfolios = np.array(efficient_portfolios)
    
    return {
        'min_var': (min_var_weights, min_var_return, min_var_std),
        'max_sharpe': (max_sharpe_weights, max_sharpe_return, max_sharpe_std, max_sharpe_ratio),
        'efficient_frontier': efficient_portfolios,
        'expected_returns': expected_returns,
        'cov_matrix': cov_matrix
    }


def visualize_results(returns, results):
    """
    4. Visualize Results
    """
    print("\n" + "=" * 60)
    print("4. Visualize Results")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Efficient Frontier
    ax1 = axes[0, 0]
    efficient_frontier = results['efficient_frontier']
    ax1.plot(efficient_frontier[:, 0], efficient_frontier[:, 1], 
             'b-', linewidth=2, label='Efficient Frontier')
    
    min_var_std, min_var_return = results['min_var'][2], results['min_var'][1]
    max_sharpe_std, max_sharpe_return = results['max_sharpe'][2], results['max_sharpe'][1]
    
    ax1.scatter(min_var_std, min_var_return, color='green', s=200, 
                marker='*', label='Minimum Variance Portfolio', zorder=5)
    ax1.scatter(max_sharpe_std, max_sharpe_return, color='red', s=200, 
                marker='*', label='Maximum Sharpe Ratio Portfolio', zorder=5)
    
    # Individual assets
    individual_stds = np.sqrt(np.diag(results['cov_matrix']))
    individual_returns = results['expected_returns']
    ax1.scatter(individual_stds, individual_returns, color='gray', s=100, 
                alpha=0.6, label='Individual Assets')
    
    for i, ticker in enumerate(returns.columns):
        ax1.annotate(ticker, (individual_stds[i], individual_returns[i]), 
                    fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Risk (Standard Deviation)', fontsize=12)
    ax1.set_ylabel('Expected Return', fontsize=12)
    ax1.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Minimum Variance Portfolio Weights
    ax2 = axes[0, 1]
    min_var_weights = results['min_var'][0]
    ax2.bar(returns.columns, min_var_weights, color='green', alpha=0.7)
    ax2.set_ylabel('Weight', fontsize=12)
    ax2.set_title('Minimum Variance Portfolio Weights', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Maximum Sharpe Ratio Portfolio Weights
    ax3 = axes[1, 0]
    max_sharpe_weights = results['max_sharpe'][0]
    ax3.bar(returns.columns, max_sharpe_weights, color='red', alpha=0.7)
    ax3.set_ylabel('Weight', fontsize=12)
    ax3.set_title('Maximum Sharpe Ratio Portfolio Weights', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Correlation Matrix Heatmap
    ax4 = axes[1, 1]
    corr_matrix = returns.corr()
    im = ax4.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(returns.columns)))
    ax4.set_yticks(range(len(returns.columns)))
    ax4.set_xticklabels(returns.columns, rotation=45, ha='right')
    ax4.set_yticklabels(returns.columns)
    ax4.set_title('Asset Correlation', fontsize=14, fontweight='bold')
    
    # Display correlation values
    for i in range(len(returns.columns)):
        for j in range(len(returns.columns)):
            text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'portfolio_optimization_result.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved: {output_path}")
    plt.show()


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 1: Linear Algebra - Portfolio Optimization")
    print("=" * 60)
    
    # 1. Understand basic operations
    weights, expected_returns, cov_matrix = demonstrate_basic_operations()
    
    # 2. Calculate covariance matrix from real data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    returns, cov_matrix_real, corr_matrix = calculate_covariance_matrix_from_data(tickers)
    
    # 3. Mean-Variance Optimization
    results = mean_variance_optimization(returns)
    
    # 4. Visualization
    visualize_results(returns, results)
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Portfolio variance: σ² = wᵀΣw (matrix multiplication)")
    print("2. Covariance matrix Σ includes both correlation and volatility")
    print("3. Efficient frontier is the set of portfolios offering maximum return for given risk")
    print("4. Minimum variance portfolio minimizes risk")
    print("5. Maximum Sharpe ratio portfolio maximizes return per unit of risk")


if __name__ == "__main__":
    main()

