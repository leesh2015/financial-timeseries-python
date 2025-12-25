"""
Chapter 3: Probability & Time Series Statistics - Monte Carlo Simulation

Core Analogy: "Estimating probability through many trials"
- Probabilistic simulation: Solve complex problems through random sampling
- Law of Large Numbers: More trials lead to more accurate estimates
- Financial applications: Portfolio risk, option pricing, VaR calculation

This example demonstrates:
1. Principles of Monte Carlo simulation
2. Portfolio VaR calculation
3. Option pricing simulation
4. Risk scenario analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_monte_carlo():
    """
    1. Principles of Monte Carlo Simulation
    """
    print("=" * 60)
    print("1. Principles of Monte Carlo Simulation")
    print("=" * 60)
    
    print("\n[Monte Carlo Method]")
    print("  1. Generate random samples from probability distribution")
    print("  2. Calculate objective function for each sample")
    print("  3. Calculate statistics of results (mean, variance, quantiles)")
    print("  4. Law of Large Numbers: More samples lead to more accuracy")
    
    print("\n[Mathematical Foundation]")
    print("  E[f(X)] ≈ (1/N) Σ f(x_i)")
    print("  → Estimate expected value with sample mean")
    print("  → Error decreases as N increases (O(1/√N))")
    
    print("\n[Financial Applications]")
    print("  - Portfolio VaR: Simulate future returns")
    print("  - Option pricing: Simulate stock price paths")
    print("  - Risk scenarios: Analyze extreme situations")
    print("  - Price complex derivatives")


def portfolio_var_monte_carlo(returns, weights, n_simulations=10000, confidence_level=0.05):
    """
    2. Calculate Portfolio VaR using Monte Carlo
    """
    print("\n" + "=" * 60)
    print("2. Monte Carlo Portfolio VaR")
    print("=" * 60)
    
    # Covariance matrix and mean returns
    cov_matrix = returns.cov().values * 252  # Annualized
    mean_returns = returns.mean().values * 252
    
    # Portfolio statistics
    portfolio_mean = np.dot(weights, mean_returns)
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_var)
    
    print(f"\n[Portfolio Statistics]")
    print(f"  Expected return: {portfolio_mean*100:.2f}%")
    print(f"  Volatility: {portfolio_std*100:.2f}%")
    
    # Monte Carlo simulation
    print(f"\n[Monte Carlo Simulation]")
    print(f"  Number of simulations: {n_simulations:,}")
    
    # Sample from multivariate normal distribution
    np.random.seed(42)
    simulated_returns = np.random.multivariate_normal(
        mean_returns, cov_matrix, size=n_simulations
    )
    
    # Calculate portfolio returns
    portfolio_returns = np.dot(simulated_returns, weights)
    
    # Calculate VaR
    var_mc = -np.percentile(portfolio_returns, confidence_level * 100)
    cvar_mc = -portfolio_returns[portfolio_returns <= -var_mc].mean()
    
    # Theoretical VaR (assuming normal distribution)
    var_theoretical = -norm.ppf(confidence_level, portfolio_mean, portfolio_std)
    
    print(f"\n[VaR Estimation]")
    print(f"  Monte Carlo: {var_mc*100:.4f}%")
    print(f"  Theoretical (Normal): {var_theoretical*100:.4f}%")
    print(f"  CVaR (Monte Carlo): {cvar_mc*100:.4f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Portfolio return distribution
    axes[0].hist(portfolio_returns * 100, bins=100, density=True, alpha=0.7, color='blue')
    axes[0].axvline(x=-var_mc * 100, color='red', linestyle='--', 
                   linewidth=2, label=f'VaR ({var_mc*100:.2f}%)')
    axes[0].axvline(x=-cvar_mc * 100, color='orange', linestyle='--', 
                   linewidth=2, label=f'CVaR ({cvar_mc*100:.2f}%)')
    axes[0].set_xlabel('Portfolio Return (%)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Monte Carlo Portfolio Return Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Convergence analysis
    n_samples_list = [100, 500, 1000, 5000, 10000]
    var_convergence = []
    for n in n_samples_list:
        sample_returns = portfolio_returns[:n]
        var_n = -np.percentile(sample_returns, confidence_level * 100)
        var_convergence.append(var_n)
    
    axes[1].plot(n_samples_list, [v*100 for v in var_convergence], 
                'o-', linewidth=2, markersize=8, label='Monte Carlo VaR')
    axes[1].axhline(y=var_theoretical*100, color='r', linestyle='--', 
                   linewidth=2, label=f'Theoretical VaR ({var_theoretical*100:.2f}%)')
    axes[1].set_xlabel('Number of Simulations', fontsize=12)
    axes[1].set_ylabel('VaR (%)', fontsize=12)
    axes[1].set_title('Monte Carlo Convergence Analysis', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'monte_carlo_var.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return var_mc, cvar_mc, portfolio_returns


def option_pricing_monte_carlo(S0=100, K=100, T=1, r=0.05, sigma=0.2, 
                                n_simulations=100000, option_type='call'):
    """
    3. Calculate Option Price using Monte Carlo
    """
    print("\n" + "=" * 60)
    print("3. Monte Carlo Option Pricing")
    print("=" * 60)
    
    print(f"\n[Option Parameters]")
    print(f"  Current price S₀: {S0}")
    print(f"  Strike price K: {K}")
    print(f"  Maturity T: {T} year")
    print(f"  Risk-free rate r: {r*100}%")
    print(f"  Volatility σ: {sigma*100}%")
    print(f"  Option type: {option_type}")
    
    # GBM path simulation
    dt = T / 252  # Daily
    n_steps = int(T * 252)
    
    np.random.seed(42)
    # Generate multiple paths
    Z = np.random.randn(n_simulations, n_steps)
    
    # GBM paths
    log_S = np.log(S0) + np.cumsum(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1
    )
    S_T = np.exp(log_S[:, -1])  # Terminal stock price
    
    # Option payoff
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:  # put
        payoffs = np.maximum(K - S_T, 0)
    
    # Option price (present value)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    option_std = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
    
    # Black-Scholes price (for comparison)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        bs_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    print(f"\n[Option Price]")
    print(f"  Monte Carlo: {option_price:.4f} ± {option_std:.4f}")
    print(f"  Black-Scholes: {bs_price:.4f}")
    print(f"  Difference: {abs(option_price - bs_price):.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Terminal price distribution
    axes[0].hist(S_T, bins=100, density=True, alpha=0.7, color='blue', label='Terminal Price Distribution')
    axes[0].axvline(x=K, color='r', linestyle='--', linewidth=2, label=f'Strike Price ({K})')
    axes[0].set_xlabel('Terminal Stock Price', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Monte Carlo Terminal Price Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Option payoff distribution
    axes[1].hist(payoffs, bins=100, density=True, alpha=0.7, color='green', label='Option Payoff')
    axes[1].axvline(x=option_price * np.exp(r * T), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean Payoff ({np.mean(payoffs):.2f})')
    axes[1].set_xlabel('Option Payoff', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Option Payoff Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'monte_carlo_option.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return option_price, bs_price


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 3: Probability & Time Series Statistics - Monte Carlo Simulation")
    print("=" * 60)
    
    # 1. Explain Monte Carlo principles
    explain_monte_carlo()
    
    # 2. Portfolio VaR
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Close']
    returns = data.pct_change().dropna()
    weights = np.array([0.4, 0.3, 0.3])
    
    var_mc, cvar_mc, portfolio_returns = portfolio_var_monte_carlo(returns, weights)
    
    # 3. Option pricing
    option_price, bs_price = option_pricing_monte_carlo()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Monte Carlo: Solve complex problems through random sampling")
    print("2. Law of Large Numbers: More samples lead to more accuracy")
    print("3. Portfolio VaR: Simulate future returns")
    print("4. Option pricing: Simulate stock price paths")
    print("5. Useful for risk scenario analysis")


if __name__ == "__main__":
    main()

