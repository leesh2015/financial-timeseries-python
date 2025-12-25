"""
Chapter 2: Calculus & Analysis - Ito's Lemma

Core Analogy: "Calculating the rate of change of randomly moving stock prices"
- Brownian Motion: Random movement of stock prices
- Ito's Lemma: Method to differentiate functions of stochastic processes
- Stochastic Differential Equation: Differential equation with stochastic elements

This example demonstrates:
1. Understanding Brownian Motion
2. Mathematical principles of Ito's Lemma
3. Geometric Brownian Motion (GBM) model
4. Foundation of option pricing models
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_ito_lemma():
    """
    1. Mathematical principles of Ito's Lemma
    """
    print("=" * 60)
    print("1. Ito's Lemma")
    print("=" * 60)
    
    print("\n[Brownian Motion]")
    print("  dW_t ~ N(0, dt)")
    print("  → Normal distribution with mean 0, variance dt")
    print("  → Continuous but non-differentiable path")
    
    print("\n[Ordinary Calculus vs Stochastic Calculus]")
    print("  Ordinary Calculus:")
    print("    df = f'(x)dx  (Chain Rule)")
    print("  Stochastic Calculus:")
    print("    df = f'(x)dx + (1/2)f''(x)(dx)²  (Ito's Lemma)")
    print("  → (dx)² = dt (Ito's Rule)")
    
    print("\n[Ito's Lemma]")
    print("  When f(X_t) is a function of stochastic process X_t:")
    print("  df = (∂f/∂t + μ∂f/∂x + (1/2)σ²∂²f/∂x²)dt + σ(∂f/∂x)dW_t")
    print("  Where:")
    print("    dX_t = μdt + σdW_t  (Stochastic Differential Equation)")
    print("    μ: Drift (mean rate of change)")
    print("    σ: Volatility (diffusion coefficient)")
    
    print("\n[Financial Applications]")
    print("  - Stock price model: dS_t = μS_t dt + σS_t dW_t")
    print("  - Option pricing: Foundation of Black-Scholes model")
    print("  - Stochastic volatility models")


def geometric_brownian_motion(S0=100, mu=0.1, sigma=0.2, T=1, n_steps=252):
    """
    2. Geometric Brownian Motion (GBM) Simulation
    """
    print("\n" + "=" * 60)
    print("2. Geometric Brownian Motion (GBM)")
    print("=" * 60)
    
    print(f"\n[GBM Model]")
    print(f"  dS_t = μS_t dt + σS_t dW_t")
    print(f"  Solution: S_t = S₀ exp((μ - σ²/2)t + σW_t)")
    
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    
    # Generate Brownian Motion
    dW = np.random.randn(n_steps) * np.sqrt(dt)
    W = np.cumsum(np.concatenate([[0], dW]))
    
    # GBM path
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    
    print(f"\n[Simulation Parameters]")
    print(f"  Initial price S₀: {S0}")
    print(f"  Drift μ: {mu} ({mu*100}%)")
    print(f"  Volatility σ: {sigma} ({sigma*100}%)")
    print(f"  Period T: {T} years")
    print(f"  Final price: {S[-1]:.2f}")
    print(f"  Expected price: {S0 * np.exp(mu * T):.2f}")
    
    # Simulate multiple paths
    n_paths = 5
    paths = []
    for _ in range(n_paths):
        dW_path = np.random.randn(n_steps) * np.sqrt(dt)
        W_path = np.cumsum(np.concatenate([[0], dW_path]))
        S_path = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W_path)
        paths.append(S_path)
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Single path
    axes[0].plot(t, S, 'b-', linewidth=2, label='GBM Path')
    axes[0].axhline(y=S0, color='r', linestyle='--', alpha=0.7, label=f'Initial Price ({S0})')
    axes[0].set_xlabel('Time (years)', fontsize=12)
    axes[0].set_ylabel('Price', fontsize=12)
    axes[0].set_title('Geometric Brownian Motion (GBM) Path', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Multiple paths
    for i, path in enumerate(paths):
        axes[1].plot(t, path, linewidth=1.5, alpha=0.7, label=f'Path {i+1}')
    axes[1].axhline(y=S0, color='r', linestyle='--', alpha=0.7, label=f'Initial Price ({S0})')
    axes[1].set_xlabel('Time (years)', fontsize=12)
    axes[1].set_ylabel('Price', fontsize=12)
    axes[1].set_title('GBM Multiple Paths Simulation', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'ito_gbm.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return S, t


def black_scholes_option_price(S, K, T, r, sigma, option_type='call'):
    """
    3. Black-Scholes Option Price (Applying Ito's Lemma)
    """
    print("\n" + "=" * 60)
    print("3. Black-Scholes Option Price")
    print("=" * 60)
    
    print("\n[Black-Scholes Formula]")
    print("  Deriving option price using Ito's Lemma")
    print("  C = S₀N(d₁) - Ke^(-rT)N(d₂)")
    print("  Where:")
    print("    d₁ = (ln(S₀/K) + (r + σ²/2)T) / (σ√T)")
    print("    d₂ = d₁ - σ√T")
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    print(f"\n[Option Price Calculation]")
    print(f"  Current stock price S₀: {S:.2f}")
    print(f"  Strike price K: {K:.2f}")
    print(f"  Time to maturity T: {T} years")
    print(f"  Risk-free rate r: {r*100}%")
    print(f"  Volatility σ: {sigma*100}%")
    print(f"  Option type: {option_type}")
    print(f"  Option price: {price:.4f}")
    
    return price


def demonstrate_ito_application():
    """
    4. Ito's Lemma Application Example
    """
    print("\n" + "=" * 60)
    print("4. Ito's Lemma Application Example")
    print("=" * 60)
    
    # Example: f(X_t) = X_t²
    print("\n[Example: f(X_t) = X_t²]")
    print("  When X_t follows dX_t = μdt + σdW_t")
    print("  df = 2X_t dX_t + (1/2)×2×(dX_t)²")
    print("     = 2X_t(μdt + σdW_t) + σ²dt")
    print("     = (2μX_t + σ²)dt + 2σX_t dW_t")
    print("  → Deriving stochastic differential equation for f(X_t) using Ito's Lemma")
    
    # GBM simulation
    S, t = geometric_brownian_motion(S0=100, mu=0.1, sigma=0.2, T=1)
    
    # Option price calculation
    option_price = black_scholes_option_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
    
    print("\n[Financial Applications]")
    print("  1. Stock price model: Simulating stock price paths with GBM")
    print("  2. Option pricing: Deriving Black-Scholes formula")
    print("  3. Risk management: Stochastic volatility models")
    print("  4. Derivative pricing: Pricing functions of complex stochastic processes")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 2: Calculus & Analysis - Ito's Lemma")
    print("=" * 60)
    
    # 1. Explain Ito's Lemma
    explain_ito_lemma()
    
    # 2. GBM simulation
    S, t = geometric_brownian_motion()
    
    # 3. Black-Scholes option price
    option_price = black_scholes_option_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
    
    # 4. Application example
    demonstrate_ito_application()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Ito's Lemma: Method to differentiate functions of stochastic processes")
    print("2. (dW_t)² = dt (Ito's Rule)")
    print("3. GBM: Modeling stochastic movement of stock prices")
    print("4. Black-Scholes: Foundation of option pricing")
    print("5. Core tool for stochastic differential equations")


if __name__ == "__main__":
    main()

