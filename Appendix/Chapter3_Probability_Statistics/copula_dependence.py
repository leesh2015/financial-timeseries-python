"""
Chapter 3: Probability & Time Series Statistics - Dependence Analysis using Copula

Core Analogy: "Panic Room Effect"
- Assets that normally move independently all crash together during crises
- Copula: Model marginal distributions independently, but model dependence structure separately
- Tail Dependence: Correlation in extreme situations

This example demonstrates:
1. Mathematical principles of Copula (Sklar's Theorem)
2. Gaussian vs Clayton Copula
3. Tail Dependence visualization
4. Application to financial risk management
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from scipy.stats import norm
import os
try:
    from copulae import GaussianCopula, ClaytonCopula
    HAS_COPULAE = True
except ImportError:
    HAS_COPULAE = False
    print("Warning: copulae not available. Install with: pip install copulae")

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_copula():
    """
    1. Mathematical Principles of Copula
    """
    print("=" * 60)
    print("1. Mathematical Principles of Copula")
    print("=" * 60)
    
    print("\n[Sklar's Theorem]")
    print("  F(x₁, x₂, ..., xₙ) = C(F₁(x₁), F₂(x₂), ..., Fₙ(xₙ))")
    print("  Where:")
    print("    F: Joint distribution function")
    print("    F_i: Marginal distribution functions (distribution of each variable)")
    print("    C: Copula function (dependency structure)")
    
    print("\n[Advantages of Copula]")
    print("  1. Separate marginal distributions and dependency structure")
    print("  2. Can combine various marginal distributions")
    print("  3. Can model Tail Dependence")
    
    print("\n[Financial Applications]")
    print("  - Normal times: Assets move independently")
    print("  - Crisis: All assets crash simultaneously (Tail Dependence)")
    print("  - Quantify risk in extreme situations using Copula")


def generate_copula_data():
    """
    2. Generate Copula Data
    """
    print("\n" + "=" * 60)
    print("2. Generate Copula Data")
    print("=" * 60)
    
    if not HAS_COPULAE:
        print("\n⚠ copulae package not available, using simulated data.")
        # Simple simulation
        np.random.seed(42)
        n = 1000
        
        # Gaussian Copula (correlation 0.7)
        corr = 0.7
        z1 = np.random.randn(n)
        z2 = corr * z1 + np.sqrt(1 - corr**2) * np.random.randn(n)
        
        # Transform marginal distribution (normal → t distribution)
        x1 = stats.t.ppf(norm.cdf(z1), df=3)
        x2 = stats.t.ppf(norm.cdf(z2), df=3)
        
        return (x1, x2), 'Gaussian'
    
    # Gaussian Copula
    print("\n[Gaussian Copula]")
    gaussian_copula = GaussianCopula(dim=2)
    gaussian_copula.params = 0.7  # Correlation
    
    u_gaussian = gaussian_copula.random(1000)
    x1_gaussian = norm.ppf(u_gaussian[:, 0])
    x2_gaussian = norm.ppf(u_gaussian[:, 1])
    
    # Clayton Copula (lower tail dependence)
    print("[Clayton Copula]")
    clayton_copula = ClaytonCopula(dim=2)
    clayton_copula.params = 2.0  # Dependence parameter
    
    u_clayton = clayton_copula.random(1000)
    x1_clayton = norm.ppf(u_clayton[:, 0])
    x2_clayton = norm.ppf(u_clayton[:, 1])
    
    return {
        'Gaussian': (x1_gaussian, x2_gaussian),
        'Clayton': (x1_clayton, x2_clayton)
    }


def visualize_copula(data_dict):
    """
    3. Copula Visualization
    """
    print("\n" + "=" * 60)
    print("3. Copula Visualization")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    if isinstance(data_dict, dict):
        # Gaussian Copula
        x1_g, x2_g = data_dict['Gaussian']
        axes[0, 0].scatter(x1_g, x2_g, alpha=0.5, s=20, color='blue')
        axes[0, 0].set_xlabel('X₁', fontsize=12)
        axes[0, 0].set_ylabel('X₂', fontsize=12)
        axes[0, 0].set_title('Gaussian Copula (Symmetric Dependence)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Clayton Copula
        x1_c, x2_c = data_dict['Clayton']
        axes[0, 1].scatter(x1_c, x2_c, alpha=0.5, s=20, color='red')
        axes[0, 1].set_xlabel('X₁', fontsize=12)
        axes[0, 1].set_ylabel('X₂', fontsize=12)
        axes[0, 1].set_title('Clayton Copula (Lower Tail Dependence)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Compare Tail Dependence
        # Lower tail (both low values)
        lower_tail_g = np.sum((x1_g < np.percentile(x1_g, 5)) & 
                             (x2_g < np.percentile(x2_g, 5))) / len(x1_g)
        lower_tail_c = np.sum((x1_c < np.percentile(x1_c, 5)) & 
                             (x2_c < np.percentile(x2_c, 5))) / len(x1_c)
        
        axes[1, 0].bar(['Gaussian', 'Clayton'], 
                      [lower_tail_g * 20, lower_tail_c * 20], 
                      color=['blue', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('Lower Tail Dependence (5% × 20)', fontsize=12)
        axes[1, 0].set_title('Tail Dependence Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Upper tail (both high values)
        upper_tail_g = np.sum((x1_g > np.percentile(x1_g, 95)) & 
                             (x2_g > np.percentile(x2_g, 95))) / len(x1_g)
        upper_tail_c = np.sum((x1_c > np.percentile(x1_c, 95)) & 
                             (x2_c > np.percentile(x2_c, 95))) / len(x1_c)
        
        axes[1, 1].bar(['Gaussian', 'Clayton'], 
                      [upper_tail_g * 20, upper_tail_c * 20], 
                      color=['blue', 'red'], alpha=0.7)
        axes[1, 1].set_ylabel('Upper Tail Dependence (5% × 20)', fontsize=12)
        axes[1, 1].set_title('Tail Dependence Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        # Simulation data
        x1, x2 = data_dict
        axes[0, 0].scatter(x1, x2, alpha=0.5, s=20, color='blue')
        axes[0, 0].set_xlabel('X₁', fontsize=12)
        axes[0, 0].set_ylabel('X₂', fontsize=12)
        axes[0, 0].set_title('Simulated Copula Data', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'copula_dependence.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n[Result Interpretation]")
    print("  - Gaussian Copula: Symmetric dependence (similar upper/lower tails)")
    print("  - Clayton Copula: Strong lower tail dependence (simultaneous decline in crisis)")
    print("  - Clayton Copula useful for modeling financial crises")


def financial_copula_analysis(ticker1='AAPL', ticker2='MSFT', 
                               start_date='2020-01-01', end_date='2024-01-01'):
    """
    4. Copula Analysis of Real Financial Data
    """
    print("\n" + "=" * 60)
    print("4. Copula Analysis of Real Financial Data")
    print("=" * 60)
    
    # Download data
    print(f"\nDownloading {ticker1} and {ticker2} data...")
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date)['Close']
    data = data.dropna()
    
    returns1 = data[ticker1].pct_change().dropna()
    returns2 = data[ticker2].pct_change().dropna()
    
    # Align data
    common_dates = returns1.index.intersection(returns2.index)
    returns1 = returns1.loc[common_dates]
    returns2 = returns2.loc[common_dates]
    
    print(f"Data period: {returns1.index[0].date()} ~ {returns1.index[-1].date()}")
    print(f"Number of observations: {len(returns1)}")
    
    # Estimate marginal distributions (assuming normal distribution)
    # In practice, can use more complex distributions (t distribution, GARCH, etc.)
    u1 = norm.cdf((returns1 - returns1.mean()) / returns1.std())
    u2 = norm.cdf((returns2 - returns2.mean()) / returns2.std())
    
    # Calculate correlation
    correlation = np.corrcoef(returns1, returns2)[0, 1]
    print(f"\n[Correlation Coefficient]")
    print(f"  Pearson correlation: {correlation:.4f}")
    
    # Calculate Tail Dependence
    threshold = 0.05
    lower_tail = np.sum((u1 < threshold) & (u2 < threshold)) / len(u1) / threshold
    upper_tail = np.sum((u1 > 1-threshold) & (u2 > 1-threshold)) / len(u1) / threshold
    
    print(f"\n[Tail Dependence]")
    print(f"  Lower tail dependence: {lower_tail:.4f}")
    print(f"  Upper tail dependence: {upper_tail:.4f}")
    print(f"  → Closer to 1 means stronger dependency in extreme situations")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Returns scatter plot
    axes[0, 0].scatter(returns1, returns2, alpha=0.5, s=10, color='blue')
    axes[0, 0].set_xlabel(f'{ticker1} Returns', fontsize=12)
    axes[0, 0].set_ylabel(f'{ticker2} Returns', fontsize=12)
    axes[0, 0].set_title(f'{ticker1} vs {ticker2} Returns', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Copula space (U1, U2)
    axes[0, 1].scatter(u1, u2, alpha=0.5, s=10, color='red')
    axes[0, 1].set_xlabel('U₁ (Uniform)', fontsize=12)
    axes[0, 1].set_ylabel('U₂ (Uniform)', fontsize=12)
    axes[0, 1].set_title('Copula Space', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Lower tail region
    axes[1, 0].scatter(returns1, returns2, alpha=0.3, s=10, color='gray')
    lower_mask = (u1 < threshold) & (u2 < threshold)
    axes[1, 0].scatter(returns1[lower_mask], returns2[lower_mask], 
                      alpha=0.8, s=30, color='red', label='Lower Tail')
    axes[1, 0].set_xlabel(f'{ticker1} Returns', fontsize=12)
    axes[1, 0].set_ylabel(f'{ticker2} Returns', fontsize=12)
    axes[1, 0].set_title('Lower Tail Dependence (During Crisis)', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Upper tail region
    axes[1, 1].scatter(returns1, returns2, alpha=0.3, s=10, color='gray')
    upper_mask = (u1 > 1-threshold) & (u2 > 1-threshold)
    axes[1, 1].scatter(returns1[upper_mask], returns2[upper_mask], 
                      alpha=0.8, s=30, color='green', label='Upper Tail')
    axes[1, 1].set_xlabel(f'{ticker1} Returns', fontsize=12)
    axes[1, 1].set_ylabel(f'{ticker2} Returns', fontsize=12)
    axes[1, 1].set_title('Upper Tail Dependence (During Boom)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'copula_financial_{ticker1}_{ticker2}.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n[Financial Risk Management]")
    print("  - High Tail Dependence: Risk of simultaneous losses in crisis")
    print("  - Limitations of portfolio diversification")
    print("  - Can quantify extreme situation risk using Copula")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 3: Probability & Time Series Statistics - Dependence Analysis using Copula")
    print("=" * 60)
    
    # 1. Explain Copula principles
    explain_copula()
    
    # 2. Generate Copula data
    data_dict = generate_copula_data()
    
    # 3. Visualization
    visualize_copula(data_dict)
    
    # 4. Real financial data analysis
    financial_copula_analysis('AAPL', 'MSFT')
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Copula: Separates marginal distributions and dependence structure")
    print("2. Tail Dependence: Correlation in extreme situations")
    print("3. Clayton Copula: Models lower tail dependence")
    print("4. Quantifies simultaneous loss risk during financial crises")
    print("5. Foundation of Copula models")


if __name__ == "__main__":
    main()

