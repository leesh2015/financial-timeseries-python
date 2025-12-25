"""
Chapter 2: Calculus & Analysis - Calculus Principles in GARCH Models

GARCH Model: σ²ₜ = α₀ + α₁ε²ₜ₋₁ + β₁σ²ₜ₋₁
- Model volatility changes over time
- Estimate optimal parameters through calculus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_garch_math():
    """
    Explain mathematical principles of GARCH models
    """
    print("=" * 60)
    print("Mathematical Principles of GARCH Models")
    print("=" * 60)
    
    print("\n[GARCH(1,1) Model]")
    print("  σ²ₜ = α₀ + α₁ε²ₜ₋₁ + β₁σ²ₜ₋₁")
    print("\n  Where:")
    print("    σ²ₜ: Conditional variance at time t (volatility²)")
    print("    εₜ₋₁: Error at time t-1 (return - predicted value)")
    print("    α₀: Constant term")
    print("    α₁: ARCH coefficient (effect of past errors)")
    print("    β₁: GARCH coefficient (effect of past volatility)")
    
    print("\n[Role of Calculus]")
    print("  1. Maximum Likelihood Estimation (MLE): Maximize log-likelihood function")
    print("  2. Gradient descent: Iteratively update parameters")
    print("  3. Differentiation: Calculate gradient of log-likelihood function")
    
    print("\n[Log-Likelihood Function]")
    print("  L(θ) = Σ log f(εₜ | σ²ₜ)")
    print("  → Transform probability density at each time point to log and sum")
    
    print("\n[Optimization]")
    print("  θ* = argmax L(θ)")
    print("  → Find optimal parameters using gradient descent or Newton's method")


def estimate_garch(ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    """
    Estimate GARCH model with real data
    """
    print("\n" + "=" * 60)
    print("GARCH Model Estimation (Real Data)")
    print("=" * 60)
    
    # Download data
    print(f"\nDownloading {ticker} data...")
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna() * 100  # Convert to percentage
    
    print(f"Data period: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"Number of observations: {len(returns)}")
    
    # Estimate GARCH(1,1) model
    print("\nEstimating GARCH(1,1) model...")
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
    fitted_model = model.fit(disp='off')
    
    print("\nEstimation results:")
    print(fitted_model.summary())
    
    # Volatility estimation
    conditional_volatility = fitted_model.conditional_volatility
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Returns
    axes[0].plot(returns.index, returns.values, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_title(f'{ticker} Daily Returns', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Returns (%)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Conditional volatility
    axes[1].plot(conditional_volatility.index, conditional_volatility.values, 
                'r-', linewidth=1.5)
    axes[1].set_title('GARCH Conditional Volatility', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Volatility (%)', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Returns vs volatility comparison
    axes[2].plot(returns.index, np.abs(returns.values), 'b-', linewidth=0.5, 
                alpha=0.5, label='|Returns|')
    axes[2].plot(conditional_volatility.index, conditional_volatility.values, 
                'r-', linewidth=1.5, label='GARCH Volatility')
    axes[2].set_title('Absolute Returns vs GARCH Volatility', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].set_ylabel('Value (%)', fontsize=10)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'garch_volatility_{ticker}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fitted_model, conditional_volatility


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 2: Calculus & Analysis - Calculus Principles in GARCH Models")
    print("=" * 60)
    
    # Explain GARCH mathematical principles
    explain_garch_math()
    
    # Estimate GARCH with real data
    model, volatility = estimate_garch('AAPL')
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. GARCH models capture time-varying volatility")
    print("2. Parameters estimated using Maximum Likelihood Estimation")
    print("3. Log-likelihood function maximized via gradient descent")
    print("4. Calculus (differentiation) is core to the optimization process")
    print("5. Captures volatility clustering in financial time series well")


if __name__ == "__main__":
    main()

