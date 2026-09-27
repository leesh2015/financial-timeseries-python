"""
Chapter 1: Linear Algebra - Multi-Factor Regression Analysis (Fama-French)

Core Analogy: "Explaining Returns with Multiple Factors"
- Single Factor (CAPM): Explained by market beta alone
- Multi-Factor Model: Explained by multiple factors (market, size, value, etc.)
- Regression Analysis: β = (X'X)⁻¹X'y (Least Squares Method)

This example demonstrates:
1. Matrix operations in OLS regression
2. Implementing the Fama-French 3-Factor Model
3. Calculating and interpreting Beta and Alpha
4. Advantages of multi-factor models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_ols_regression():
    """
    1. Matrix Operations in OLS Regression
    """
    print("=" * 60)
    print("1. Matrix Operations in OLS Regression")
    print("=" * 60)
    
    print("\n[Regression Model]")
    print("  y = Xβ + ε")
    print("  Where:")
    print("    y: Dependent variable (n×1) - Stock returns")
    print("    X: Independent variables (n×k) - Factor returns")
    print("    β: Regression coefficients (k×1) - Factor exposures (Beta)")
    print("    ε: Error term (n×1)")
    
    print("\n[Ordinary Least Squares (OLS)]")
    print("  Objective: min ||y - Xβ||²")
    print("  Solution: β = (X'X)⁻¹X'y")
    print("  → Calculate optimal coefficients using matrix operations")
    
    print("\n[Regression Statistics]")
    print("  R² = 1 - (SS_res / SS_tot)")
    print("  → Proportion of variance explained by the model")
    print("  t-statistic = β / SE(β)")
    print("  → Significance test for coefficients")


def load_fama_french_factors(stock_returns=None):
    """
    2. Load Fama-French Factor Data (Simulated)
    If stock_returns is provided, create factors with some correlation to stock returns
    """
    print("\n" + "=" * 60)
    print("2. Fama-French Factor Data")
    print("=" * 60)
    
    # In practice, download from Kenneth French's website
    # Here we use simulated data that has some relationship with stock returns
    np.random.seed(42)
    
    if stock_returns is not None:
        n = len(stock_returns)
        # Ensure stock_returns is 1D array
        stock_returns_1d = np.atleast_1d(stock_returns.values).flatten()
        
        # Create more realistic factors: 
        # Market factor should explain most of stock returns (typical beta ~1.2)
        # Add noise to make it realistic
        market_factor = stock_returns_1d / 1.2 + np.random.randn(n) * (np.std(stock_returns_1d) * 0.3)
        
        # SMB and HML are orthogonal factors (smaller impact)
        # They should be independent of market but still have some relationship
        smb_factor = np.random.randn(n) * np.std(stock_returns_1d) * 0.3
        hml_factor = np.random.randn(n) * np.std(stock_returns_1d) * 0.2
        
        # Add small correlation to stock returns for SMB and HML
        smb_factor = smb_factor + stock_returns_1d * 0.15
        hml_factor = hml_factor + stock_returns_1d * 0.10
        
        # Ensure all factors are 1D
        market_factor = np.atleast_1d(market_factor).flatten()
        smb_factor = np.atleast_1d(smb_factor).flatten()
        hml_factor = np.atleast_1d(hml_factor).flatten()
    else:
        n = 252 * 4  # 4 years of daily data
        # Simulated factors
        market_factor = np.random.randn(n) * 0.01  # Market factor
        smb_factor = np.random.randn(n) * 0.005    # Size factor
        hml_factor = np.random.randn(n) * 0.004    # Value factor
    
    factors = pd.DataFrame({
        'Mkt-RF': market_factor,
        'SMB': smb_factor,
        'HML': hml_factor
    })
    
    print(f"Generated factor data: {n} days")
    print(f"  Market Factor (Mkt-RF): Mean {market_factor.mean():.6f}, Std {market_factor.std():.6f}")
    print(f"  Size Factor (SMB): Mean {smb_factor.mean():.6f}, Std {smb_factor.std():.6f}")
    print(f"  Value Factor (HML): Mean {hml_factor.mean():.6f}, Std {hml_factor.std():.6f}")
    
    return factors


def ols_regression_matrix(y, X):
    """
    3. Perform OLS Regression Using Matrix Operations
    """
    # Ensure y and X are 1D/2D arrays
    y = np.atleast_1d(y).flatten()
    X = np.atleast_2d(X) if X.ndim == 1 else X
    
    # Add constant term
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # β = (X'X)⁻¹X'y
    XTX = np.dot(X_with_const.T, X_with_const)
    XTy = np.dot(X_with_const.T, y)
    beta = np.linalg.solve(XTX, XTy)
    
    # Ensure beta is 1D array
    beta = np.atleast_1d(beta).flatten()
    
    # Predicted values
    y_pred = np.dot(X_with_const, beta)
    
    # Residuals
    residuals = y - y_pred
    
    # R² - ensure we don't divide by zero
    ss_res = np.sum(residuals**2)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    
    if ss_tot == 0:
        r_squared = 0.0
    else:
        r_squared = 1 - (ss_res / ss_tot)
    
    # Standard errors
    mse = ss_res / (len(y) - len(beta))
    var_beta = mse * np.linalg.inv(XTX)
    se_beta = np.sqrt(np.diag(var_beta))
    
    # Ensure se_beta is 1D array
    se_beta = np.atleast_1d(se_beta).flatten()
    
    # t-statistics
    t_stats = beta / se_beta
    
    # Ensure t_stats is 1D array
    t_stats = np.atleast_1d(t_stats).flatten()
    
    return {
        'beta': beta,
        'y_pred': y_pred,
        'residuals': residuals,
        'r_squared': r_squared,
        'se_beta': se_beta,
        't_stats': t_stats
    }


def fama_french_3factor_analysis(ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    """
    4. Fama-French 3-Factor Model Analysis
    """
    print("\n" + "=" * 60)
    print("4. Fama-French 3-Factor Model Analysis")
    print("=" * 60)
    
    # Download stock data
    print(f"\nDownloading {ticker} data...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)['Close']
    stock_returns = stock_data.pct_change().dropna()
    
    # Factor data (simulated with correlation to stock returns)
    factors = load_fama_french_factors(stock_returns=stock_returns)
    
    # Align data (simplified: match lengths)
    min_len = min(len(stock_returns), len(factors))
    stock_returns = stock_returns.iloc[:min_len]
    factors = factors.iloc[:min_len]
    
    print(f"\nData period: {len(stock_returns)} days")
    
    # CAPM (single factor)
    print("\n[CAPM Model]")
    X_capm = factors[['Mkt-RF']].values
    y = stock_returns.values
    result_capm = ols_regression_matrix(y, X_capm)
    
    print(f"  Alpha (α): {result_capm['beta'][0]:.6f}")
    print(f"  Market Beta (β): {result_capm['beta'][1]:.6f}")
    print(f"  R²: {result_capm['r_squared']:.4f}")
    print(f"  t-statistic (Beta): {result_capm['t_stats'][1]:.4f}")
    
    # Fama-French 3-Factor
    print("\n[Fama-French 3-Factor Model]")
    X_ff3 = factors[['Mkt-RF', 'SMB', 'HML']].values
    result_ff3 = ols_regression_matrix(y, X_ff3)
    
    print(f"  Alpha (α): {result_ff3['beta'][0]:.6f}")
    print(f"  Market Beta (β_m): {result_ff3['beta'][1]:.6f}")
    print(f"  Size Beta (β_s): {result_ff3['beta'][2]:.6f}")
    print(f"  Value Beta (β_h): {result_ff3['beta'][3]:.6f}")
    print(f"  R²: {result_ff3['r_squared']:.4f}")
    print(f"  t-statistics:")
    print(f"    Market: {result_ff3['t_stats'][1]:.4f}")
    print(f"    SMB: {result_ff3['t_stats'][2]:.4f}")
    print(f"    HML: {result_ff3['t_stats'][3]:.4f}")
    
    # Model comparison
    print(f"\n[Model Comparison]")
    print(f"  CAPM R²: {result_capm['r_squared']:.4f}")
    print(f"  FF3 R²: {result_ff3['r_squared']:.4f}")
    print(f"  Improvement: {(result_ff3['r_squared'] - result_capm['r_squared'])*100:.2f}%p")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Actual vs Predicted (CAPM)
    axes[0, 0].scatter(y, result_capm['y_pred'], alpha=0.5, s=20, color='blue')
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual Returns', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Returns (CAPM)', fontsize=12)
    axes[0, 0].set_title(f'CAPM: R² = {result_capm["r_squared"]:.4f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Actual vs Predicted (FF3)
    axes[0, 1].scatter(y, result_ff3['y_pred'], alpha=0.5, s=20, color='green')
    axes[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('Actual Returns', fontsize=12)
    axes[0, 1].set_ylabel('Predicted Returns (FF3)', fontsize=12)
    axes[0, 1].set_title(f'Fama-French 3-Factor: R² = {result_ff3["r_squared"]:.4f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Beta comparison
    beta_names = ['Market', 'SMB', 'HML']
    beta_values = result_ff3['beta'][1:]
    beta_errors = result_ff3['se_beta'][1:]
    
    axes[1, 0].bar(beta_names, beta_values, yerr=beta_errors, 
                  alpha=0.7, color=['blue', 'green', 'red'], capsize=5)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 0].set_ylabel('Beta', fontsize=12)
    axes[1, 0].set_title('Fama-French 3-Factor Betas', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # R² comparison
    r2_capm = result_capm['r_squared']
    r2_ff3 = result_ff3['r_squared']
    
    # Clamp R² to reasonable range for display (actual values may be negative)
    r2_display_capm = max(0, min(1, r2_capm))  # Clamp between 0 and 1 for display
    r2_display_ff3 = max(0, min(1, r2_ff3))
    
    # Calculate improvement
    improvement = r2_ff3 - r2_capm
    improvement_pct = (improvement / abs(r2_capm) * 100) if r2_capm != 0 else 0
    
    bars = axes[1, 1].bar(['CAPM', 'FF3'], [r2_display_capm, r2_display_ff3], 
                          alpha=0.7, color=['blue', 'green'], width=0.6)
    
    # Add value labels on bars with actual R² values
    for bar, val, display_val in zip(bars, [r2_capm, r2_ff3], [r2_display_capm, r2_display_ff3]):
        height = bar.get_height()
        # Show actual value (may be negative)
        label_text = f'{val:.4f}'
        if val < 0:
            label_text += '\n(negative)'
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., 
                       max(height, 0.02) if height > 0 else 0.02,
                       label_text,
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add improvement annotation
    if improvement != 0:
        axes[1, 1].annotate(f'Δ = {improvement:.4f}\n({improvement_pct:+.2f}%)',
                           xy=(1, r2_display_ff3), xytext=(1.3, r2_display_ff3),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    axes[1, 1].set_ylabel('R²', fontsize=12)
    axes[1, 1].set_title('Model Explanatory Power Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim(-0.05, 1.05)  # Fixed range to show 0-1 clearly
    axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'factor_regression_{ticker}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.show()
    
    return result_capm, result_ff3


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 1: Linear Algebra - Multi-Factor Regression (Fama-French)")
    print("=" * 60)
    
    # 1. Explain OLS regression
    explain_ols_regression()
    
    # 2. Fama-French 3-Factor analysis
    result_capm, result_ff3 = fama_french_3factor_analysis('AAPL')
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. OLS: β = (X'X)⁻¹X'y (Matrix operations)")
    print("2. CAPM: Explains with market beta alone (Low R²)")
    print("3. Fama-French: Improved explanatory power with multiple factors")
    print("4. Beta: Exposure to each factor")
    print("5. Alpha: Excess returns unexplained by factors")
    print("6. Foundation of Fama-French models")


if __name__ == "__main__":
    main()

