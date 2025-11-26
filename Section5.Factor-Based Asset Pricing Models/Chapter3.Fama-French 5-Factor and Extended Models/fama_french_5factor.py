"""
Chapter 3: Fama-French 5-Factor and 6-Factor Model Implementation

This script demonstrates:
1. Fama-French 5-factor model
2. 6-factor model (adding momentum)
3. Model comparison
4. Factor interpretation
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


def compare_factor_models(ticker, start_date, end_date):
    """
    Compare 3-factor, 5-factor, and 6-factor models
    """
    print(f"\n{'='*80}")
    print(f"Comparing Factor Models for {ticker}")
    print(f"{'='*80}")
    
    # Load stock data using the same method as Chapter 2
    stock_data = load_stock_data(ticker, start_date, end_date)
    ff3 = load_ff_factors('3-factor', 'daily')
    
    # Align data using align_data function (same as Chapter 2)
    stock_ret_aligned, ff3_aligned = align_data(stock_data, ff3, price_column='Close')
    
    if len(stock_ret_aligned) < 30:
        print(f"  Insufficient data: {len(stock_ret_aligned)} observations")
        return None
    
    results = {}
    
    # 3-Factor Model
    print("\n[3-Factor Model]")
    
    try:
        ff3_result = calculate_factor_exposures(
            stock_ret_aligned,
            ff3_aligned,
            factor_cols=['Mkt-RF', 'SMB', 'HML']
        )
        results['3-factor'] = ff3_result
        print(f"  R-squared: {ff3_result['r_squared']:.3f}")
        print(f"  Alpha (annualized): {ff3_result['alpha'] * 250:.2%}")
    except Exception as e:
        print(f"  Error: {str(e)[:50]}")
    
    # 5-Factor Model
    print("\n[5-Factor Model]")
    ff5 = load_ff_factors('5-factor', 'daily')
    # Align with same dates as 3-factor
    ff5_aligned = ff5.loc[stock_ret_aligned.index]
    
    try:
        ff5_result = calculate_factor_exposures(
            stock_ret_aligned,
            ff5_aligned,
            factor_cols=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        )
        results['5-factor'] = ff5_result
        print(f"  R-squared: {ff5_result['r_squared']:.3f}")
        print(f"  Alpha (annualized): {ff5_result['alpha'] * 250:.2%}")
        print(f"  RMW Beta: {ff5_result.get('RMW_beta', np.nan):.3f}")
        print(f"  CMA Beta: {ff5_result.get('CMA_beta', np.nan):.3f}")
    except Exception as e:
        print(f"  Error: {str(e)[:50]}")
    
    # 6-Factor Model
    print("\n[6-Factor Model]")
    ff6 = load_ff_factors('6-factor', 'daily')
    # Align with same dates as 3-factor
    ff6_aligned = ff6.loc[stock_ret_aligned.index]
    
    try:
        ff6_result = calculate_factor_exposures(
            stock_ret_aligned,
            ff6_aligned,
            factor_cols=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        )
        results['6-factor'] = ff6_result
        print(f"  R-squared: {ff6_result['r_squared']:.3f}")
        print(f"  Alpha (annualized): {ff6_result['alpha'] * 250:.2%}")
        print(f"  MOM Beta: {ff6_result.get('MOM_beta', np.nan):.3f}")
    except Exception as e:
        print(f"  Error: {str(e)[:50]}")
    
    # Comparison
    if len(results) > 1:
        print("\n[Model Comparison]")
        print(f"  R-squared improvement (3→5): "
              f"{results.get('5-factor', {}).get('r_squared', 0) - results.get('3-factor', {}).get('r_squared', 0):.3f}")
        print(f"  R-squared improvement (5→6): "
              f"{results.get('6-factor', {}).get('r_squared', 0) - results.get('5-factor', {}).get('r_squared', 0):.3f}")
    
    return results


def main():
    """Main function"""
    print("=" * 80)
    print("Chapter 3: Fama-French 5-Factor and Extended Models")
    print("=" * 80)
    
    tickers = ['AAPL', 'MSFT', 'JNJ', 'WMT']
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5)
    
    all_results = {}
    
    for ticker in tickers:
        try:
            results = compare_factor_models(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            if results:
                all_results[ticker] = results
        except Exception as e:
            print(f"Error with {ticker}: {str(e)}")
            continue
    
    # Summary
    if all_results:
        print("\n" + "=" * 80)
        print("Summary: Model Comparison")
        print("=" * 80)
        
        r2_3f = [r.get('3-factor', {}).get('r_squared', 0) for r in all_results.values()]
        r2_5f = [r.get('5-factor', {}).get('r_squared', 0) for r in all_results.values()]
        r2_6f = [r.get('6-factor', {}).get('r_squared', 0) for r in all_results.values()]
        
        print(f"\nAverage R-squared:")
        print(f"  3-Factor: {np.mean(r2_3f):.3f}")
        print(f"  5-Factor: {np.mean(r2_5f):.3f}")
        print(f"  6-Factor: {np.mean(r2_6f):.3f}")
        
        print("\nConclusion:")
        print("  - Adding RMW and CMA factors improves explanatory power")
        print("  - Adding momentum further improves the model")
        print("  - More factors = better fit, but risk of overfitting")


if __name__ == '__main__':
    main()

