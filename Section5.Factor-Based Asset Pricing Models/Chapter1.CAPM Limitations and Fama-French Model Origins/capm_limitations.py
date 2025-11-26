"""
Chapter 1: CAPM Limitations Demonstration

This script demonstrates the limitations of CAPM by showing:
1. Low R-squared values in CAPM regressions
2. Non-zero alpha values
3. Size and value effects that CAPM cannot explain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta

# Add Section5 root to path
section4_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, section4_root)

from utils.data_loader import load_ff_factors, load_stock_data
from utils.metrics import calculate_alpha
import warnings

warnings.filterwarnings("ignore")


def run_capm_regression(stock_returns, market_returns, risk_free_rate):
    """
    Run CAPM regression: R_i - R_f = α + β(R_m - R_f) + ε
    
    Parameters:
    -----------
    stock_returns : pd.Series
        Stock returns
    market_returns : pd.Series
        Market returns
    risk_free_rate : pd.Series
        Risk-free rate
    
    Returns:
    --------
    dict
        Regression results
    """
    # Align data
    common_dates = stock_returns.index.intersection(market_returns.index).intersection(risk_free_rate.index)
    stock_ret = stock_returns.loc[common_dates]
    market_ret = market_returns.loc[common_dates]
    rf = risk_free_rate.loc[common_dates]  # Daily data is already daily %
    
    # Calculate excess returns
    stock_excess = stock_ret - rf
    market_excess = market_ret - rf
    
    # Remove NaN
    mask = ~(stock_excess.isna() | market_excess.isna())
    stock_excess_clean = stock_excess[mask]
    market_excess_clean = market_excess[mask]
    
    if len(stock_excess_clean) < 30:
        return None
    
    # OLS regression
    X = market_excess_clean.values.reshape(-1, 1)
    X_with_const = np.column_stack([np.ones(len(X)), X])
    y = stock_excess_clean.values
    
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
    except:
        return None
    
    # Calculate R-squared
    y_pred = X_with_const @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate statistics
    n = len(y)
    k = 2
    mse = ss_res / (n - k) if (n - k) > 0 else 0
    
    if mse > 0:
        var_coeffs = mse * np.linalg.pinv(X_with_const.T @ X_with_const)
        se_coeffs = np.sqrt(np.diag(var_coeffs))
        t_stats = coeffs / se_coeffs
    else:
        se_coeffs = np.full(2, np.nan)
        t_stats = np.full(2, np.nan)
    
    return {
        'alpha': coeffs[0],
        'alpha_se': se_coeffs[0],
        'alpha_tstat': t_stats[0],
        'beta': coeffs[1],
        'beta_se': se_coeffs[1],
        'beta_tstat': t_stats[1],
        'r_squared': r_squared,
        'n_observations': n
    }


def demonstrate_capm_limitations():
    """
    Demonstrate CAPM limitations using multiple stocks
    """
    print("=" * 80)
    print("Chapter 1: CAPM Limitations Demonstration")
    print("=" * 80)
    
    # Load Fama-French 3-factor data (for market and risk-free rate)
    print("\n[Step 1] Loading Fama-French factor data...")
    ff_factors = load_ff_factors('3-factor', 'daily')
    print(f"Loaded {len(ff_factors)} days of factor data")
    print(f"Date range: {ff_factors.index.min()} to {ff_factors.index.max()}")
    
    # Calculate market returns (Mkt-RF + RF)
    market_returns = (ff_factors['Mkt-RF'] / 100) + (ff_factors['RF'] / 100)
    risk_free_rate = ff_factors['RF'] / 100
    
    # Select diverse stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',  # Large cap growth
               'JNJ', 'PG', 'KO', 'WMT', 'XOM', 'CVX',  # Large cap value
               'AMD', 'INTC', 'MU', 'QCOM']  # Mid cap tech
    
    print(f"\n[Step 2] Analyzing {len(tickers)} stocks...")
    
    # Date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5)  # 5 years
    
    results = []
    
    for ticker in tickers:
        try:
            print(f"  Processing {ticker}...", end=' ')
            
            # Load stock data
            stock_data = load_stock_data(ticker, start_date.strftime('%Y-%m-%d'), 
                                        end_date.strftime('%Y-%m-%d'))
            stock_returns = stock_data['Close'].pct_change().dropna()
            
            # Run CAPM regression
            result = run_capm_regression(stock_returns, market_returns, risk_free_rate)
            
            if result:
                result['ticker'] = ticker
                results.append(result)
                print(f"✓ (R²={result['r_squared']:.3f}, α={result['alpha']*250:.2%})")
            else:
                print("✗ (Failed)")
        
        except Exception as e:
            print(f"✗ (Error: {str(e)[:50]})")
            continue
    
    if not results:
        print("\nNo results obtained. Please check data availability.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("CAPM Regression Results Summary")
    print("=" * 80)
    
    print(f"\nAverage R-squared: {results_df['r_squared'].mean():.3f}")
    print(f"Median R-squared: {results_df['r_squared'].median():.3f}")
    print(f"Min R-squared: {results_df['r_squared'].min():.3f}")
    print(f"Max R-squared: {results_df['r_squared'].max():.3f}")
    
    print(f"\nAverage Alpha (annualized): {results_df['alpha'].mean() * 250:.2%}")
    print(f"Stocks with significant alpha (|t-stat| > 2): "
          f"{(np.abs(results_df['alpha_tstat']) > 2).sum()} / {len(results_df)}")
    
    print(f"\nAverage Beta: {results_df['beta'].mean():.2f}")
    print(f"Beta range: {results_df['beta'].min():.2f} to {results_df['beta'].max():.2f}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("Key Findings: CAPM Limitations")
    print("=" * 80)
    
    print("\n1. Low R-squared:")
    print(f"   - Average R² = {results_df['r_squared'].mean():.1%}")
    print(f"   - This means {100 - results_df['r_squared'].mean()*100:.1f}% of return variation is UNEXPLAINED by CAPM")
    print("   - Suggests missing risk factors")
    
    print("\n2. Non-zero Alpha:")
    significant_alpha = results_df[np.abs(results_df['alpha_tstat']) > 2]
    if len(significant_alpha) > 0:
        print(f"   - {len(significant_alpha)} stocks have statistically significant alpha")
        print("   - If CAPM were correct, alpha should be zero")
        print("   - Suggests model misspecification")
    
    print("\n3. Incomplete Risk Explanation:")
    print("   - Market beta alone cannot explain all return variations")
    print("   - Additional factors (size, value) are needed")
    print("   - This motivates multi-factor models like Fama-French")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R-squared distribution
    axes[0, 0].hist(results_df['r_squared'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(results_df['r_squared'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["r_squared"].mean():.3f}')
    axes[0, 0].set_xlabel('R-squared')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('CAPM R-squared Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Alpha distribution
    alpha_annual = results_df['alpha'] * 250
    axes[0, 1].hist(alpha_annual, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', label='Zero Alpha')
    axes[0, 1].axvline(alpha_annual.mean(), color='blue', linestyle='--', 
                       label=f'Mean: {alpha_annual.mean():.2%}')
    axes[0, 1].set_xlabel('Alpha (annualized)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('CAPM Alpha Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # R-squared vs Beta
    axes[1, 0].scatter(results_df['beta'], results_df['r_squared'], alpha=0.6)
    axes[1, 0].set_xlabel('Beta')
    axes[1, 0].set_ylabel('R-squared')
    axes[1, 0].set_title('R-squared vs Beta')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Alpha vs Beta
    axes[1, 1].scatter(results_df['beta'], alpha_annual, alpha=0.6)
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Beta')
    axes[1, 1].set_ylabel('Alpha (annualized)')
    axes[1, 1].set_title('Alpha vs Beta')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_image = os.path.join(script_dir, 'capm_limitations_analysis.png')
    output_csv = os.path.join(script_dir, 'capm_regression_results.csv')
    
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"\n[Visualization] Saved to: {output_image}")
    
    # Save results
    results_df.to_csv(output_csv, index=False)
    print(f"[Results] Saved to: {output_csv}")
    
    print("\n" + "=" * 80)
    print("Conclusion: CAPM has significant limitations")
    print("Multi-factor models (like Fama-French) are needed to better explain stock returns")
    print("=" * 80)


if __name__ == '__main__':
    demonstrate_capm_limitations()

