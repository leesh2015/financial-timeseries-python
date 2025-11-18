"""
AIC vs BIC: Model Selection Criteria

This script demonstrates how to use AIC (Akaike Information Criterion) and BIC 
(Bayesian Information Criterion) to select optimal AR and MA orders for ARIMA models.

AIC: Penalizes model complexity less (tends to select more complex models)
BIC: Penalizes model complexity more (tends to select simpler models)
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from utils.data_loader import load_financial_data, preprocess_data
    from utils.config import config
    USE_UTILS = True
except ImportError:
    USE_UTILS = False
    import yfinance as yf

# Configuration
TICKER = 'SPY'
START_DATE = '2020-01-01'
END_DATE = '2023-01-01'
MAX_ORDER = 10  # Maximum AR/MA order to test


def calculate_aic_bic(data, max_order=10):
    """
    Calculate AIC and BIC values for AR and MA models.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data (should be stationary)
    max_order : int
        Maximum order to test
        
    Returns:
    --------
    tuple
        (aic_values_ar, bic_values_ar, aic_values_ma, bic_values_ma)
    """
    aic_values_ar = []
    bic_values_ar = []
    aic_values_ma = []
    bic_values_ma = []
    
    # Test AR models (p, 0, 0)
    print("Testing AR models...")
    for p in range(1, max_order + 1):
        try:
            ar_model = sm.tsa.ARIMA(data, order=(p, 0, 0))
            ar_results = ar_model.fit()
            aic_values_ar.append(ar_results.aic)
            bic_values_ar.append(ar_results.bic)
        except Exception as e:
            print(f"  Warning: AR({p}) model failed: {e}")
            aic_values_ar.append(np.inf)
            bic_values_ar.append(np.inf)
    
    # Test MA models (0, 0, q)
    print("Testing MA models...")
    for q in range(1, max_order + 1):
        try:
            ma_model = sm.tsa.ARIMA(data, order=(0, 0, q))
            ma_results = ma_model.fit()
            aic_values_ma.append(ma_results.aic)
            bic_values_ma.append(ma_results.bic)
        except Exception as e:
            print(f"  Warning: MA({q}) model failed: {e}")
            aic_values_ma.append(np.inf)
            bic_values_ma.append(np.inf)
    
    return aic_values_ar, bic_values_ar, aic_values_ma, bic_values_ma


def main():
    """Main function"""
    # Load data
    if USE_UTILS:
        data = load_financial_data(TICKER, START_DATE, END_DATE)
        data = preprocess_data(data, frequency='D', use_adjusted=False)
        close_prices = data['Close']
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        close_prices = data['Close']
        close_prices = close_prices.asfreq('D')
        close_prices = close_prices.dropna()
    
    # Create log-differenced data (stationary transformation)
    log_diff = np.log(close_prices).diff().dropna()
    
    # Calculate AIC and BIC values
    aic_values_ar, bic_values_ar, aic_values_ma, bic_values_ma = calculate_aic_bic(
        log_diff, max_order=MAX_ORDER
    )
    
    # Find optimal orders
    optimal_p_aic = aic_values_ar.index(min(aic_values_ar)) + 1
    optimal_p_bic = bic_values_ar.index(min(bic_values_ar)) + 1
    optimal_q_aic = aic_values_ma.index(min(aic_values_ma)) + 1
    optimal_q_bic = bic_values_ma.index(min(bic_values_ma)) + 1
    
    # Print results
    print("\n" + "="*60)
    print("Model Selection Results")
    print("="*60)
    print(f'Optimal AR order (AIC): {optimal_p_aic}')
    print(f'Optimal AR order (BIC): {optimal_p_bic}')
    print(f'Optimal MA order (AIC): {optimal_q_aic}')
    print(f'Optimal MA order (BIC): {optimal_q_bic}')
    print("\nNote: BIC typically selects simpler models than AIC")
    
    # Visualize AIC and BIC values
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, MAX_ORDER + 1), aic_values_ar, marker='o', label='AIC (AR)', linewidth=2)
    plt.plot(range(1, MAX_ORDER + 1), bic_values_ar, marker='s', label='BIC (AR)', linewidth=2)
    plt.axvline(optimal_p_aic, color='red', linestyle='--', alpha=0.5, label=f'Optimal AIC: p={optimal_p_aic}')
    plt.axvline(optimal_p_bic, color='blue', linestyle='--', alpha=0.5, label=f'Optimal BIC: p={optimal_p_bic}')
    plt.xlabel('AR Order (p)')
    plt.ylabel('Criterion Value')
    plt.legend()
    plt.title('AIC and BIC Values for AR Order')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, MAX_ORDER + 1), aic_values_ma, marker='o', label='AIC (MA)', linewidth=2)
    plt.plot(range(1, MAX_ORDER + 1), bic_values_ma, marker='s', label='BIC (MA)', linewidth=2)
    plt.axvline(optimal_q_aic, color='red', linestyle='--', alpha=0.5, label=f'Optimal AIC: q={optimal_q_aic}')
    plt.axvline(optimal_q_bic, color='blue', linestyle='--', alpha=0.5, label=f'Optimal BIC: q={optimal_q_bic}')
    plt.xlabel('MA Order (q)')
    plt.ylabel('Criterion Value')
    plt.legend()
    plt.title('AIC and BIC Values for MA Order')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
