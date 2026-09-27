"""
Chapter 1: Linear Algebra - Matrix Operations in VAR & VECM Models

Core Analogy: "Multiple assets influencing each other"
- VAR: Each asset is influenced by its own and other assets' past values
- VECM: VAR considering long-term equilibrium relationships (cointegration)
- Matrix operations: Efficiently compute multivariate time series using matrices

This example demonstrates:
1. Matrix representation of VAR models
2. Estimating VAR parameters using least squares
3. Cointegration matrix in VECM
4. Calculating IRF (Impulse Response Function)
5. FEVD (Forecast Error Variance Decomposition)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_var_matrix():
    """
    1. Matrix Representation of VAR Models
    """
    print("=" * 60)
    print("1. Matrix Representation of VAR Models")
    print("=" * 60)
    
    print("\n[VAR(1) Model]")
    print("  y_t = A₁y_{t-1} + c + ε_t")
    print("  Where:")
    print("    y_t: n×1 vector (current values of n variables)")
    print("    A₁: n×n matrix (autoregressive coefficients)")
    print("    c: n×1 vector (constant term)")
    print("    ε_t: n×1 vector (error)")
    
    print("\n[Meaning of Matrix Operations]")
    print("  A₁[i,j]: Effect of variable j's past value on variable i's current value")
    print("  Diagonal: Autoregressive coefficients (own past affects own present)")
    print("  Off-diagonal: Cross effects (other variables' past affects own present)")
    
    print("\n[VAR(p) Model]")
    print("  y_t = A₁y_{t-1} + A₂y_{t-2} + ... + Aₚy_{t-p} + c + ε_t")
    print("  → Express influences of multiple lags using matrices")
    
    print("\n[Transform to Matrix Form]")
    print("  Y = XB + E")
    print("  Where:")
    print("    Y: T×n matrix (observations)")
    print("    X: T×(np+1) matrix (explanatory variables)")
    print("    B: (np+1)×n matrix (coefficients)")
    print("    E: T×n matrix (errors)")
    print("  → Least squares: B = (X'X)⁻¹X'Y")


def demonstrate_var_estimation():
    """
    2. VAR Model Estimation (Matrix Operations)
    """
    print("\n" + "=" * 60)
    print("2. VAR Model Estimation (Matrix Operations)")
    print("=" * 60)
    
    # Simple VAR(1) simulation
    np.random.seed(42)
    n_vars = 2
    n_obs = 200
    
    # Coefficient matrix
    A1 = np.array([[0.5, 0.3],
                   [0.2, 0.6]])
    c = np.array([0.1, 0.05])
    
    # Generate data
    y = np.zeros((n_obs, n_vars))
    for t in range(1, n_obs):
        y[t] = A1 @ y[t-1] + c + np.random.randn(n_vars) * 0.1
    
    print(f"\n[Simulation Data]")
    print(f"  Number of variables: {n_vars}")
    print(f"  Number of observations: {n_obs}")
    print(f"  True coefficient matrix A₁:")
    print(A1)
    
    # Fit VAR model
    var_model = VAR(y)
    var_result = var_model.fit(maxlags=1)
    
    print(f"\n[Estimated Coefficient Matrix A₁]")
    print(var_result.coefs[0])
    
    print(f"\n[Coefficient Comparison]")
    print(f"  True vs Estimated Error:")
    print(np.abs(A1 - var_result.coefs[0]))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time series
    axes[0, 0].plot(y[:, 0], 'b-', linewidth=1.5, label='Variable 1', alpha=0.7)
    axes[0, 0].plot(y[:, 1], 'r-', linewidth=1.5, label='Variable 2', alpha=0.7)
    axes[0, 0].set_xlabel('Time', fontsize=12)
    axes[0, 0].set_ylabel('Value', fontsize=12)
    axes[0, 0].set_title('VAR(1) Time Series Data', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coefficient matrix comparison
    im1 = axes[0, 1].imshow(A1, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[0, 1].set_title('True Coefficient Matrix A₁', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_xticklabels(['Var1', 'Var2'])
    axes[0, 1].set_yticklabels(['Var1', 'Var2'])
    for i in range(2):
        for j in range(2):
            text = axes[0, 1].text(j, i, f'{A1[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(var_result.coefs[0], cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[1, 0].set_title('Estimated Coefficient Matrix A₁', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Var1', 'Var2'])
    axes[1, 0].set_yticklabels(['Var1', 'Var2'])
    for i in range(2):
        for j in range(2):
            text = axes[1, 0].text(j, i, f'{var_result.coefs[0][i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=12)
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Error
    error = np.abs(A1 - var_result.coefs[0])
    im3 = axes[1, 1].imshow(error, cmap='Reds', aspect='auto')
    axes[1, 1].set_title('Estimation Error |True - Estimated|', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['Var1', 'Var2'])
    axes[1, 1].set_yticklabels(['Var1', 'Var2'])
    for i in range(2):
        for j in range(2):
            text = axes[1, 1].text(j, i, f'{error[i, j]:.4f}',
                                  ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'var_matrix_estimation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.show()
    
    return var_result, A1


def explain_vecm_matrix():
    """
    3. Matrix Representation of VECM Models
    """
    print("\n" + "=" * 60)
    print("3. Matrix Representation of VECM Models")
    print("=" * 60)
    
    print("\n[VECM Model]")
    print("  Δy_t = αβ'y_{t-1} + Γ₁Δy_{t-1} + ... + ΓₚΔy_{t-p} + ε_t")
    print("  Where:")
    print("    Δy_t: First difference (y_t - y_{t-1})")
    print("    α: Adjustment speed matrix (n×r)")
    print("    β: Cointegration vector matrix (n×r)")
    print("    β'y_{t-1}: Cointegration relationship (deviation from long-run equilibrium)")
    print("    Γ_i: Short-run dynamics coefficient matrix")
    
    print("\n[Meaning of Cointegration]")
    print("  β'y_{t-1} = 0: Long-run equilibrium state")
    print("  β'y_{t-1} ≠ 0: Deviation from equilibrium → Adjustment via α")
    print("  → Matrix αβ' connects long-run relationship to short-run dynamics")
    
    print("\n[Matrix Decomposition]")
    print("  αβ' = Π (cointegration matrix)")
    print("  → Decompose n×n matrix into n×r and r×n (r: cointegration rank)")
    print("  → Efficient computation through dimensionality reduction")


def analyze_financial_var_vecm(tickers=['AAPL', 'MSFT', 'GOOGL'], 
                               start_date='2020-01-01', end_date='2024-01-01'):
    """
    4. Real Financial Data VAR/VECM Analysis
    """
    print("\n" + "=" * 60)
    print("4. Real Financial Data VAR/VECM Analysis")
    print("=" * 60)
    
    # Download data
    print(f"\nDownloading {tickers} data...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data.dropna()
    
    returns = data.pct_change().dropna() * 100  # Percentage
    
    print(f"Data period: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"Number of observations: {len(returns)}")
    print(f"Number of variables: {len(tickers)}")
    
    # VAR model
    print("\n[VAR Model Estimation]")
    var_model = VAR(returns)
    var_result = var_model.fit(maxlags=2, ic='aic')
    
    print(f"Selected lag: {var_result.k_ar}")
    print(f"AIC: {var_result.aic:.4f}")
    
    print(f"\n[VAR Coefficient Matrix (First Lag)]")
    print(var_result.coefs[0])
    print("\nInterpretation:")
    print("  Diagonal: Autoregressive coefficients (own past affects own present)")
    print("  Off-diagonal: Cross effects (other variables' past affects own present)")
    
    # IRF (Impulse Response Function)
    print("\n[IRF Calculation]")
    irf = var_result.irf(10)  # 10 periods
    print("IRF: Effect of shock in one variable on other variables")
    
    # FEVD (Forecast Error Variance Decomposition)
    print("\n[FEVD Calculation]")
    fevd = var_result.fevd(10)
    print("FEVD: Contribution of each variable to forecast error variance")
    
    # VECM model (after cointegration test)
    # Note: VECM requires non-stationary (I(1)) price data, not returns
    print("\n[VECM Model]")
    print("  Note: VECM is applied to price levels (non-stationary), not returns (stationary)")
    print("  VECM captures long-term equilibrium relationships between prices")
    try:
        # Use price levels (log prices) for VECM, not returns
        prices = np.log(data)  # Log prices for VECM
        vecm_model = VECM(prices, k_ar_diff=1, coint_rank=1)
        vecm_result = vecm_model.fit()
        
        print(f"Cointegration rank: {vecm_result.coint_rank}")
        print(f"\n[Cointegration Vector β]")
        print(vecm_result.beta)
        print(f"\n[Adjustment Speed α]")
        print(vecm_result.alpha)
        
        # Cointegration matrix
        coint_matrix = vecm_result.alpha @ vecm_result.beta.T
        print(f"\n[Cointegration Matrix αβ']")
        print(coint_matrix)
        
    except Exception as e:
        print(f"VECM fitting failed: {e}")
        print("  → There may be no cointegration relationship")
        vecm_result = None
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Returns time series
    for i, ticker in enumerate(tickers):
        axes[0, 0].plot(returns.index, returns[ticker].values, 
                       linewidth=1, alpha=0.7, label=ticker)
    axes[0, 0].set_xlabel('Date', fontsize=12)
    axes[0, 0].set_ylabel('Returns (%)', fontsize=12)
    axes[0, 0].set_title('Stock Returns Time Series', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # VAR coefficient matrix
    im1 = axes[0, 1].imshow(var_result.coefs[0], cmap='RdBu', aspect='auto')
    axes[0, 1].set_title('VAR Coefficient Matrix (First Lag)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(range(len(tickers)))
    axes[0, 1].set_yticks(range(len(tickers)))
    axes[0, 1].set_xticklabels(tickers, rotation=45, ha='right')
    axes[0, 1].set_yticklabels(tickers)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            text = axes[0, 1].text(j, i, f'{var_result.coefs[0][i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im1, ax=axes[0, 1])
    
    # IRF
    irf_data = irf.irfs
    for i, ticker in enumerate(tickers):
        axes[1, 0].plot(irf_data[:, i, 0], linewidth=2, label=f'{ticker} → {tickers[0]}', alpha=0.7)
    axes[1, 0].set_xlabel('Lag', fontsize=12)
    axes[1, 0].set_ylabel('IRF', fontsize=12)
    axes[1, 0].set_title(f'IRF: Shock to {tickers[0]}', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # FEVD
    fevd_data = fevd.decomp
    # Use the last available period (FEVD typically has shape [n_periods, n_vars, n_vars])
    # fevd_data[period, shock_var, response_var]
    last_period = fevd_data.shape[0] - 1
    axes[1, 1].bar(range(len(tickers)), fevd_data[last_period, 0, :], alpha=0.7)
    axes[1, 1].set_xlabel('Variable', fontsize=12)
    axes[1, 1].set_ylabel('FEVD (%)', fontsize=12)
    axes[1, 1].set_title(f'FEVD: {tickers[0]} ({last_period + 1} periods ahead)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(range(len(tickers)))
    axes[1, 1].set_xticklabels(tickers, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # VECM cointegration matrix (using prices, not returns)
    if vecm_result is not None:
        im2 = axes[2, 0].imshow(coint_matrix, cmap='RdBu', aspect='auto')
        axes[2, 0].set_title("VECM Cointegration Matrix αβ'", fontsize=12, fontweight='bold')
        axes[2, 0].set_xticks(range(len(tickers)))
        axes[2, 0].set_yticks(range(len(tickers)))
        axes[2, 0].set_xticklabels(tickers, rotation=45, ha='right')
        axes[2, 0].set_yticklabels(tickers)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                text = axes[2, 0].text(j, i, f'{coint_matrix[i, j]:.3f}',
                                      ha="center", va="center", color="black", fontsize=9)
        plt.colorbar(im2, ax=axes[2, 0])
        
        # Cointegration vector
        axes[2, 1].bar(range(len(tickers)), vecm_result.beta[:, 0], alpha=0.7, color='green')
        axes[2, 1].set_xlabel('Variable', fontsize=12)
        axes[2, 1].set_ylabel('Cointegration Vector β', fontsize=12)
        axes[2, 1].set_title('Cointegration Vector (Long-run Equilibrium)', fontsize=12, fontweight='bold')
        axes[2, 1].set_xticks(range(len(tickers)))
        axes[2, 1].set_xticklabels(tickers, rotation=45, ha='right')
        axes[2, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[2, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[2, 0].text(0.5, 0.5, 'VECM fitting failed\n(No cointegration)', 
                       ha='center', va='center', fontsize=14)
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'var_vecm_financial_{"_".join(tickers)}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.show()
    
    return var_result, vecm_result


def explain_matrix_operations():
    """
    5. Summary of Matrix Operations in VAR/VECM
    """
    print("\n" + "=" * 60)
    print("5. Summary of Matrix Operations in VAR/VECM")
    print("=" * 60)
    
    print("\n[Key Matrix Operations]")
    print("  1. Matrix multiplication: A₁y_{t-1} (linear combination of past values)")
    print("  2. Matrix inversion: (X'X)⁻¹ (least squares method)")
    print("  3. Matrix decomposition: αβ' (VECM cointegration matrix)")
    print("  4. Eigenvalue decomposition: Stability testing")
    
    print("\n[Computational Efficiency]")
    print("  - Vectorized operations: Compute all variables simultaneously")
    print("  - Matrix operations: Calculate without loops")
    print("  - NumPy optimization: Fast computation speed")
    
    print("\n[Financial Applications]")
    print("  - Multivariate time series: Analyze mutual influences among assets")
    print("  - Cointegration: Model long-run equilibrium relationships")
    print("  - IRF: Analyze propagation paths of shocks")
    print("  - FEVD: Identify sources of volatility")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 1: Linear Algebra - Matrix Operations in VAR & VECM Models")
    print("=" * 60)
    
    # 1. Explain VAR matrix representation
    explain_var_matrix()
    
    # 2. Demonstrate VAR estimation
    var_result, A1 = demonstrate_var_estimation()
    
    # 3. Explain VECM matrix representation
    explain_vecm_matrix()
    
    # 4. Analyze real financial data
    var_financial, vecm_financial = analyze_financial_var_vecm(['AAPL', 'MSFT', 'GOOGL'])
    
    # 5. Summarize matrix operations
    explain_matrix_operations()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. VAR: y_t = A₁y_{t-1} + ... (matrix multiplication)")
    print("2. VECM: Δy_t = αβ'y_{t-1} + ... (cointegration matrix)")
    print("3. Least squares: B = (X'X)⁻¹X'Y (matrix inversion)")
    print("4. IRF: Propagation path of shocks (matrix powers)")
    print("5. FEVD: Contribution to volatility (matrix decomposition)")
    print("6. Foundation of VAR/VECM models")


if __name__ == "__main__":
    main()

