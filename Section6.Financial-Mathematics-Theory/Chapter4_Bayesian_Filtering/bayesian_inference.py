"""
Chapter 4: Bayesian Statistics & Filtering - Bayesian Inference Example

Core Analogy: "Narrowing down suspects with new clues"
- Prior probability: Initially many suspects (prior knowledge)
- Observation: Evidence emerges (data)
- Posterior probability: Update probability of the culprit (Bayes' theorem)

This example demonstrates:
1. Understanding Bayes' theorem
2. Relationship between prior and posterior probabilities
3. Application to financial risk estimation
4. Bayesian regression analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from scipy.stats import beta, norm
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_bayes_theorem():
    """
    1. Basic Concepts of Bayes' Theorem
    """
    print("=" * 60)
    print("1. Bayes' Theorem")
    print("=" * 60)
    
    print("\n[Bayes' Theorem]")
    print("  P(A|B) = P(B|A) × P(A) / P(B)")
    print("\n  Where:")
    print("    P(A|B): Posterior probability - Probability of A after observing B")
    print("    P(B|A): Likelihood - Probability of B given A")
    print("    P(A): Prior probability - Prior knowledge about A")
    print("    P(B): Normalizing constant (Evidence)")
    
    print("\n[Financial Application]")
    print("  A: Probability that market will rise")
    print("  B: Past return data")
    print("  P(A|B): Probability of market rise after seeing data")
    
    print("\n[Bayesian Update Process]")
    print("  1. Set prior probability (past experience, expert opinion)")
    print("  2. Observe data")
    print("  3. Calculate likelihood")
    print("  4. Calculate posterior probability using Bayes' theorem")
    print("  5. Posterior becomes new prior (iterate)")


def coin_flip_example():
    """
    2. Coin Flip Example (Bayesian Update)
    """
    print("\n" + "=" * 60)
    print("2. Coin Flip Example")
    print("=" * 60)
    
    # Prior distribution: Bayesian estimation of probability p of heads
    # Use Beta distribution as prior (conjugate prior)
    
    # Prior distribution: Beta(α=2, β=2) - Belief in slightly fair coin
    alpha_prior = 2
    beta_prior = 2
    
    print(f"\n[Prior Distribution]")
    print(f"  Beta(α={alpha_prior}, β={beta_prior})")
    print(f"  → Belief in slightly fair coin")
    
    # Observation: 8 heads out of 10 tosses
    n_tosses = 10
    n_heads = 8
    
    print(f"\n[Observation Data]")
    print(f"  {n_heads} heads out of {n_tosses} tosses")
    
    # Posterior distribution: Beta(α + n_heads, β + n_tails)
    alpha_posterior = alpha_prior + n_heads
    beta_posterior = beta_prior + (n_tosses - n_heads)
    
    print(f"\n[Posterior Distribution]")
    print(f"  Beta(α={alpha_posterior}, β={beta_posterior})")
    print(f"  → Updated belief reflecting data")
    
    # Posterior distribution statistics
    posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
    posterior_mode = (alpha_posterior - 1) / (alpha_posterior + beta_posterior - 2)
    
    print(f"\n[Posterior Distribution Statistics]")
    print(f"  Mean: {posterior_mean:.4f}")
    print(f"  Mode: {posterior_mode:.4f}")
    
    # Visualization
    p_range = np.linspace(0, 1, 1000)
    prior_pdf = beta.pdf(p_range, alpha_prior, beta_prior)
    posterior_pdf = beta.pdf(p_range, alpha_posterior, beta_posterior)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prior vs Posterior
    axes[0].plot(p_range, prior_pdf, 'b-', linewidth=2, label='Prior Distribution', alpha=0.7)
    axes[0].plot(p_range, posterior_pdf, 'r-', linewidth=2, label='Posterior Distribution', alpha=0.7)
    axes[0].axvline(x=posterior_mean, color='g', linestyle='--', 
                   linewidth=2, label=f'Posterior Mean ({posterior_mean:.3f})')
    axes[0].set_xlabel('Probability of Heads p', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title('Bayesian Update: Prior → Posterior', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confidence interval
    ci_lower = beta.ppf(0.025, alpha_posterior, beta_posterior)
    ci_upper = beta.ppf(0.975, alpha_posterior, beta_posterior)
    
    axes[1].plot(p_range, posterior_pdf, 'r-', linewidth=2, label='Posterior Distribution')
    axes[1].fill_between(p_range[(p_range >= ci_lower) & (p_range <= ci_upper)],
                        posterior_pdf[(p_range >= ci_lower) & (p_range <= ci_upper)],
                        alpha=0.3, color='red', label='95% Confidence Interval')
    axes[1].axvline(x=ci_lower, color='g', linestyle='--', linewidth=1)
    axes[1].axvline(x=ci_upper, color='g', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Probability of Heads p', fontsize=12)
    axes[1].set_ylabel('Probability Density', fontsize=12)
    axes[1].set_title(f'95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]', 
                     fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'bayesian_update_coin.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n[95% Confidence Interval]")
    print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  → 95% probability that probability of heads is in this interval")


def bayesian_risk_estimation(ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    """
    3. Risk Estimation using Bayesian Method
    """
    print("\n" + "=" * 60)
    print("3. Bayesian Risk Estimation (Financial Application)")
    print("=" * 60)
    
    # Download data
    print(f"\nDownloading {ticker} data...")
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()
    
    print(f"Data period: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"Number of observations: {len(returns)}")
    
    # Estimate volatility using Bayesian method
    # Prior distribution: Prior belief about volatility
    # Use inverse-gamma prior for variance of normal distribution (simplified)
    
    # Prior parameters
    n_prior = 10  # Prior number of observations
    var_prior = np.var(returns.values) * 0.5  # Prior variance (conservative)
    
    # Posterior variance estimation
    n_data = len(returns)
    var_data = np.var(returns.values)
    
    # Bayesian update (weighted average)
    var_posterior = (n_prior * var_prior + n_data * var_data) / (n_prior + n_data)
    vol_posterior = np.sqrt(var_posterior)
    
    # Classical estimation (sample variance)
    vol_classical = np.std(returns.values)  # Use numpy to ensure scalar value
    
    print(f"\n[Volatility Estimation Comparison]")
    print(f"  Classical method (sample): {vol_classical*100:.4f}%")
    print(f"  Bayesian method: {vol_posterior*100:.4f}%")
    print(f"  → Bayesian method is more conservative by incorporating prior information")
    
    # VaR (Value at Risk) estimation
    confidence_level = 0.05  # 95% VaR
    
    # Classical VaR
    var_classical = -np.percentile(returns.values, confidence_level * 100)
    
    # Bayesian VaR (assuming normal distribution)
    var_bayesian = -norm.ppf(confidence_level, 0, vol_posterior)
    
    # CVaR (Conditional Value at Risk) = Expected Shortfall
    # Average of losses smaller than VaR
    returns_sorted = np.sort(returns.values)
    var_threshold_idx = int(confidence_level * len(returns_sorted))
    cvar_classical = -returns_sorted[:var_threshold_idx].mean()
    
    # Bayesian CVaR (assuming normal distribution)
    # CVaR = -E[X | X <= VaR] = -E[X | X <= -VaR]
    # Calculate CVaR for normal distribution
    var_threshold = -var_bayesian / vol_posterior  # Standardize
    cvar_bayesian = -vol_posterior * (norm.pdf(var_threshold) / norm.cdf(var_threshold))
    
    print(f"\n[VaR Estimation (95%)]")
    print(f"  Classical method: {var_classical*100:.4f}%")
    print(f"  Bayesian method: {var_bayesian*100:.4f}%")
    
    print(f"\n[CVaR Estimation (95%) - Expected Shortfall]")
    print(f"  Classical method: {cvar_classical*100:.4f}%")
    print(f"  Bayesian method: {cvar_bayesian*100:.4f}%")
    print(f"  → CVaR is greater than or equal to VaR (average of extreme losses)")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Return distribution
    axes[0].hist(returns.values, bins=50, density=True, alpha=0.7, 
                color='blue', label='Actual Returns')
    
    # Bayesian distribution
    x_range = np.linspace(returns.min(), returns.max(), 100)
    bayesian_pdf = norm.pdf(x_range, 0, vol_posterior)
    axes[0].plot(x_range, bayesian_pdf, 'r-', linewidth=2, 
                label=f'Bayesian Distribution (σ={vol_posterior*100:.2f}%)')
    
    # VaR and CVaR markers
    axes[0].axvline(x=-var_bayesian, color='orange', linestyle='--', 
                   linewidth=2, label=f'Bayesian VaR ({var_bayesian*100:.2f}%)')
    axes[0].axvline(x=-cvar_bayesian, color='red', linestyle='--', 
                   linewidth=2, label=f'Bayesian CVaR ({cvar_bayesian*100:.2f}%)')
    
    axes[0].set_xlabel('Returns', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Return Distribution and Bayesian Estimation', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Volatility estimation over time (rolling window)
    window = 60
    rolling_vol = returns.rolling(window=window).std()
    
    axes[1].plot(returns.index, rolling_vol.values * 100, 'b-', 
                linewidth=1.5, alpha=0.7, label=f'{window}-day Rolling Volatility')
    axes[1].axhline(y=vol_posterior * 100, color='r', linestyle='--', 
                   linewidth=2, label=f'Bayesian Volatility ({vol_posterior*100:.2f}%)')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Volatility (%)', fontsize=12)
    axes[1].set_title('Volatility Estimation Over Time', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'bayesian_risk_{ticker}.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return vol_posterior, var_bayesian, cvar_bayesian


def explain_conjugate_prior():
    """
    4. Conjugate Prior Explanation
    """
    print("\n" + "=" * 60)
    print("4. Conjugate Prior")
    print("=" * 60)
    
    print("\n[Advantages of Conjugate Prior]")
    print("  - Posterior distribution is in the same family as prior distribution")
    print("  - Simple calculation (analytical solution exists)")
    print("  - Easy to update")
    
    print("\n[Main Conjugate Pairs]")
    print("  - Bernoulli/Binomial → Beta prior")
    print("  - Normal distribution (mean) → Normal prior")
    print("  - Normal distribution (variance) → Inverse-gamma prior")
    print("  - Poisson → Gamma prior")
    
    print("\n[Financial Applications]")
    print("  - Volatility estimation: Inverse-gamma prior")
    print("  - Return estimation: Normal prior")
    print("  - Risk event probability: Beta prior")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 4: Bayesian Statistics & Filtering - Bayesian Inference Example")
    print("=" * 60)
    
    # 1. Explain Bayes' theorem
    explain_bayes_theorem()
    
    # 2. Coin flip example
    coin_flip_example()
    
    # 3. Financial risk estimation
    vol, var, cvar = bayesian_risk_estimation('AAPL')
    
    # 4. Explain conjugate prior
    explain_conjugate_prior()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Bayes' theorem: Prior probability → Observation → Posterior probability")
    print("2. Combine prior information with data for better estimation")
    print("3. Conjugate prior simplifies calculations")
    print("4. Useful for financial risk estimation (conservative estimation)")
    print("5. Foundation of Kalman Filter, Prophet models")


if __name__ == "__main__":
    main()

