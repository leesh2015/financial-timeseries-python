"""
Chapter 2: Fama-French 3-Factor Model Implementation

This script demonstrates:
1. Loading Fama-French 3-factor data
2. Calculating factor exposures for individual stocks
3. Interpreting regression results
4. Comparing CAPM vs Fama-French 3-factor model
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

from utils.data_loader import load_ff_factors, load_stock_data, align_data
from utils.factor_utils import calculate_factor_exposures
import warnings

warnings.filterwarnings("ignore")


def compare_capm_vs_ff3factor(ticker, start_date, end_date):
    """
    Compare CAPM and Fama-French 3-factor model for a single stock
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    
    Returns:
    --------
    dict
        Comparison results
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {ticker}: CAPM vs Fama-French 3-Factor Model")
    print(f"{'='*80}")
    
    # Load data
    print("\n[Step 1] Loading data...")
    ff_factors = load_ff_factors('3-factor', 'daily')
    stock_data = load_stock_data(ticker, start_date, end_date)
    
    # Calculate stock returns
    stock_returns = stock_data['Close'].pct_change().dropna()
    
    # Align data
    stock_ret_aligned, factor_data_aligned = align_data(
        stock_data, ff_factors, price_column='Close'
    )
    
    if len(stock_ret_aligned) < 30:
        print(f"Insufficient data: {len(stock_ret_aligned)} observations")
        return None
    
    print(f"Aligned data: {len(stock_ret_aligned)} observations")
    print(f"Date range: {stock_ret_aligned.index.min()} to {stock_ret_aligned.index.max()}")
    
    # CAPM regression
    print("\n[Step 2] Running CAPM regression...")
    market_excess = factor_data_aligned['Mkt-RF'] / 100
    risk_free = factor_data_aligned['RF'] / 100
    stock_excess = stock_ret_aligned - risk_free
    
    # Remove NaN
    mask = ~(stock_excess.isna() | market_excess.isna())
    stock_excess_clean = stock_excess[mask]
    market_excess_clean = market_excess[mask]
    
    # CAPM OLS
    X_capm = market_excess_clean.values.reshape(-1, 1)
    X_capm_const = np.column_stack([np.ones(len(X_capm)), X_capm])
    y_capm = stock_excess_clean.values
    
    coeffs_capm, _, _, _ = np.linalg.lstsq(X_capm_const, y_capm, rcond=None)
    y_pred_capm = X_capm_const @ coeffs_capm
    ss_res_capm = np.sum((y_capm - y_pred_capm) ** 2)
    ss_tot_capm = np.sum((y_capm - np.mean(y_capm)) ** 2)
    r2_capm = 1 - (ss_res_capm / ss_tot_capm) if ss_tot_capm > 0 else 0
    
    capm_results = {
        'alpha': coeffs_capm[0],
        'beta': coeffs_capm[1],
        'r_squared': r2_capm,
        'n_obs': len(y_capm)
    }
    
    print(f"  CAPM Alpha (annualized): {capm_results['alpha'] * 250:.2%}")
    print(f"  CAPM Beta: {capm_results['beta']:.3f}")
    print(f"  CAPM R-squared: {capm_results['r_squared']:.3f}")
    
    # Fama-French 3-factor regression
    print("\n[Step 3] Running Fama-French 3-factor regression...")
    ff3_results = calculate_factor_exposures(
        stock_ret_aligned,
        factor_data_aligned,
        risk_free_col='RF',
        factor_cols=['Mkt-RF', 'SMB', 'HML']
    )
    
    print(f"  FF3 Alpha (annualized): {ff3_results['alpha'] * 250:.2%}")
    print(f"  FF3 Market Beta: {ff3_results['Mkt-RF_beta']:.3f}")
    print(f"  FF3 SMB Beta: {ff3_results['SMB_beta']:.3f}")
    print(f"  FF3 HML Beta: {ff3_results['HML_beta']:.3f}")
    print(f"  FF3 R-squared: {ff3_results['r_squared']:.3f}")
    
    # Comparison
    print("\n[Step 4] Model Comparison:")
    print(f"  R-squared improvement: {ff3_results['r_squared'] - capm_results['r_squared']:.3f}")
    print(f"  R-squared improvement (%): {(ff3_results['r_squared'] / capm_results['r_squared'] - 1) * 100:.1f}%")
    
    # Statistical significance
    print("\n[Step 5] Statistical Significance:")
    print(f"  Alpha t-statistic: {ff3_results['alpha_tstat']:.3f}")
    if abs(ff3_results['alpha_tstat']) > 2:
        print("  → Alpha is statistically significant (market inefficiency or model misspecification)")
    else:
        print("  → Alpha is not statistically significant (market efficiency maintained)")
    
    print(f"  SMB Beta t-statistic: {ff3_results['SMB_tstat']:.3f}")
    if abs(ff3_results['SMB_tstat']) > 2:
        print("  → Size factor is significant")
    
    print(f"  HML Beta t-statistic: {ff3_results['HML_tstat']:.3f}")
    if abs(ff3_results['HML_tstat']) > 2:
        print("  → Value factor is significant")
    
    # Interpretation
    print("\n[Step 6] Factor Exposure Interpretation:")
    if ff3_results['SMB_beta'] > 0:
        print(f"  → Stock has small-cap exposure (SMB beta = {ff3_results['SMB_beta']:.3f})")
        print("  → Expected to earn size premium")
    else:
        print(f"  → Stock has large-cap exposure (SMB beta = {ff3_results['SMB_beta']:.3f})")
        print("  → Expected to earn lower returns than small-caps")
    
    if ff3_results['HML_beta'] > 0:
        print(f"  → Stock has value exposure (HML beta = {ff3_results['HML_beta']:.3f})")
        print("  → Expected to earn value premium")
    else:
        print(f"  → Stock has growth exposure (HML beta = {ff3_results['HML_beta']:.3f})")
        print("  → Expected to earn lower returns than value stocks")
    
    return {
        'ticker': ticker,
        'capm': capm_results,
        'ff3': ff3_results
    }


def main():
    """
    Main function to demonstrate Fama-French 3-factor model
    """
    print("=" * 80)
    print("Chapter 2: Fama-French 3-Factor Model")
    print("=" * 80)
    
    # Example stocks
    tickers = ['AAPL', 'MSFT', 'JNJ', 'WMT']
    
    # Date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5)
    
    results = []
    
    for ticker in tickers:
        try:
            result = compare_capm_vs_ff3factor(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            continue
    
    # Summary
    if results:
        print("\n" + "=" * 80)
        print("Summary: CAPM vs Fama-French 3-Factor Model")
        print("=" * 80)
        
        capm_r2_avg = np.mean([r['capm']['r_squared'] for r in results])
        ff3_r2_avg = np.mean([r['ff3']['r_squared'] for r in results])
        
        print(f"\nAverage CAPM R-squared: {capm_r2_avg:.3f}")
        print(f"Average FF3 R-squared: {ff3_r2_avg:.3f}")
        print(f"Average improvement: {ff3_r2_avg - capm_r2_avg:.3f} ({(ff3_r2_avg/capm_r2_avg - 1)*100:.1f}%)")
        
        print("\nConclusion:")
        print("  - Fama-French 3-factor model explains significantly more return variation")
        print("  - Size and value factors are important risk factors")
        print("  - Multi-factor models are superior to single-factor CAPM")


if __name__ == '__main__':
    main()

