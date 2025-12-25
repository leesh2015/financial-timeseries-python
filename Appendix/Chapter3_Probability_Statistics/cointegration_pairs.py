"""
Chapter 3: Probability & Time Series Statistics - Cointegration and Pairs Trading

Core Analogy: "Owner and Dog"
- They seem to move independently, but are connected by a leash (long-term equilibrium), so they end up together
- Cointegration: Linear combination of two non-stationary time series is stationary

This example demonstrates:
1. Understanding the concept of cointegration
2. Johansen Test
3. Pairs Trading Strategy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_cointegration():
    """
    Explain the concept of cointegration
    """
    print("=" * 60)
    print("Concept of Cointegration")
    print("=" * 60)
    
    print("\n[Definition]")
    print("  When two non-stationary time series X_t, Y_t exist,")
    print("  If Z_t = Y_t - βX_t is a stationary time series,")
    print("  X_t and Y_t have a cointegration relationship")
    
    print("\n[Analogy: Owner and Dog]")
    print("  - Owner (X_t) and dog (Y_t) each move randomly (non-stationary)")
    print("  - But they are connected by a leash (β), so distance (Z_t) stays within range")
    print("  - If distance increases, they come closer again (mean reversion)")
    
    print("\n[Financial Application]")
    print("  - Price ratio of two related stocks stays within a range")
    print("  - When ratio deviates, expect mean reversion")
    print("  - Basis of pairs trading strategy")


def test_cointegration(ticker1='AAPL', ticker2='MSFT', 
                       start_date='2020-01-01', end_date='2024-01-01'):
    """
    Cointegration test and pairs trading example
    """
    print("\n" + "=" * 60)
    print("Cointegration Test and Pairs Trading")
    print("=" * 60)
    
    # Download data
    print(f"\nDownloading {ticker1} and {ticker2} data...")
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date)['Close']
    data = data.dropna()
    
    price1 = data[ticker1]
    price2 = data[ticker2]
    
    print(f"Data period: {price1.index[0].date()} ~ {price1.index[-1].date()}")
    print(f"Number of observations: {len(price1)}")
    
    # Engle-Granger test (simple method)
    print("\n[Engle-Granger Cointegration Test]")
    # Add constant term (intercept) to the regression
    from statsmodels.tools import add_constant
    price1_with_const = add_constant(price1)
    model = OLS(price2, price1_with_const).fit()
    beta = model.params[1]  # Coefficient for price1 (index 1, since index 0 is constant)
    intercept = model.params[0]  # Intercept term
    spread = price2 - (beta * price1 + intercept)
    
    print(f"  Regression: {ticker2} = {intercept:.4f} + {beta:.4f} × {ticker1} + ε")
    print(f"  Spread mean: {spread.mean():.4f}")
    print(f"  Spread std: {spread.std():.4f}")
    
    # Confirm spread stationarity with ADF test
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(spread.dropna())
    p_value = adf_result[1]
    
    print(f"\n  Spread ADF Test:")
    print(f"    p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"    → Cointegration exists! (p < 0.05)")
        is_cointegrated = True
    else:
        print(f"    → No cointegration (p >= 0.05)")
        is_cointegrated = False
    
    # Johansen test (more accurate method)
    print("\n[Johansen Cointegration Test]")
    data_matrix = np.column_stack([price1.values, price2.values])
    johansen_result = coint_johansen(data_matrix, det_order=0, k_ar_diff=1)
    
    print(f"  Trace statistic: {johansen_result.lr1[0]:.4f}")
    print(f"  Critical value (5%): {johansen_result.cvt[0, 1]:.4f}")
    
    if johansen_result.lr1[0] > johansen_result.cvt[0, 1]:
        print(f"    → Cointegration exists!")
        cointegration_rank = johansen_result.rkt
    else:
        print(f"    → No cointegration")
        cointegration_rank = 0
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Price comparison
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(price1.index, price1.values, 'b-', linewidth=2, label=ticker1, alpha=0.7)
    ax1_twin.plot(price2.index, price2.values, 'r-', linewidth=2, label=ticker2, alpha=0.7)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel(f'{ticker1} Price', fontsize=10, color='b')
    ax1_twin.set_ylabel(f'{ticker2} Price', fontsize=10, color='r')
    ax1.set_title(f'{ticker1} vs {ticker2} Price', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, alpha=0.3)
    
    # Spread
    ax2 = axes[1]
    ax2.plot(spread.index, spread.values, 'g-', linewidth=1.5)
    ax2.axhline(y=spread.mean(), color='r', linestyle='--', label='Mean')
    ax2.axhline(y=spread.mean() + 2*spread.std(), color='orange', linestyle='--', 
               alpha=0.7, label='±2σ')
    ax2.axhline(y=spread.mean() - 2*spread.std(), color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Spread', fontsize=10)
    ax2.set_title(f'Spread: {ticker2} - {beta:.4f} × {ticker1}', 
                 fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Z-score
    z_score = (spread - spread.mean()) / spread.std()
    ax3 = axes[2]
    ax3.plot(z_score.index, z_score.values, 'purple', linewidth=1.5)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.axhline(y=2, color='r', linestyle='--', alpha=0.7, label='±2σ')
    ax3.axhline(y=-2, color='r', linestyle='--', alpha=0.7)
    ax3.fill_between(z_score.index, -2, 2, alpha=0.2, color='green', label='Normal Range')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Z-score', fontsize=10)
    ax3.set_title('Spread Z-score (Pairs Trading Signal)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'cointegration_{ticker1}_{ticker2}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n[Pairs Trading Strategy]")
    print(f"  Z-score > 2: Spread too large → Short {ticker2}, Long {ticker1}")
    print(f"  Z-score < -2: Spread too small → Long {ticker2}, Short {ticker1}")
    print(f"  Z-score → 0: Mean reversion → Close positions")
    
    return spread, z_score, is_cointegrated


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 3: Probability & Time Series Statistics - Cointegration and Pairs Trading")
    print("=" * 60)
    
    # Explain cointegration concept
    explain_cointegration()
    
    # Cointegration test and pairs trading
    spread, z_score, is_cointegrated = test_cointegration('AAPL', 'MSFT')
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Cointegration: Linear combination of two non-stationary time series is stationary")
    print("2. Cointegration relationship means long-term equilibrium relationship")
    print("3. Pairs trading using mean-reverting property of spread")
    print("4. Determine entry/exit points using Z-score")
    print("5. VECM models utilize cointegration")


if __name__ == "__main__":
    main()

