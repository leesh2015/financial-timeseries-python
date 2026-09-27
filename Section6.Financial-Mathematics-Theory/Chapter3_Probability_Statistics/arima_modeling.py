"""
Chapter 3: Probability & Time Series Statistics - Probabilistic Foundation of ARIMA Models

Core Analogy: "Predicting the future from past weather"
- AR (AutoRegressive): Past values influence current value
- I (Integrated): Stationarization through differencing
- MA (Moving Average): Past errors influence current value

This example demonstrates:
1. Understanding the probabilistic structure of ARIMA models
2. Model identification through ACF/PACF
3. Optimal parameter selection (AIC/BIC)
4. Application to real financial data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_arima_components():
    """
    1. Explain ARIMA Model Components
    """
    print("=" * 60)
    print("1. ARIMA Model Components")
    print("=" * 60)
    
    print("\n[ARIMA(p, d, q) Model]")
    print("  ARIMA = AR + I + MA")
    
    print("\n[AR(p) - AutoRegressive Model]")
    print("  y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t")
    print("  → Current value is a linear combination of past p values")
    print("  → 'Past determines present'")
    
    print("\n[I(d) - Integration (Differencing)]")
    print("  Difference d times to convert to stationary time series")
    print("  Δy_t = y_t - y_{t-1}  (1st difference)")
    print("  → Remove trend")
    
    print("\n[MA(q) - Moving Average Model]")
    print("  y_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q}")
    print("  → Current value is a linear combination of past q errors")
    print("  → 'Past shocks influence present'")
    
    print("\n[ARIMA(p, d, q) Integrated Model]")
    print("  Differenced series: Δ^d y_t")
    print("  ARIMA: Apply AR(p) + MA(q) to differenced series")
    print("  → Optimal linear prediction model for stationary series")


def generate_arima_example():
    """
    2. ARIMA Model Simulation Example
    """
    print("\n" + "=" * 60)
    print("2. ARIMA Model Simulation")
    print("=" * 60)
    
    np.random.seed(42)
    n = 200
    
    # AR(1) model: y_t = 0.7 * y_{t-1} + ε_t
    print("\n[AR(1) Model Simulation]")
    ar1 = np.zeros(n)
    for i in range(1, n):
        ar1[i] = 0.7 * ar1[i-1] + np.random.randn()
    
    # MA(1) model: y_t = ε_t + 0.5 * ε_{t-1}
    print("[MA(1) Model Simulation]")
    ma1 = np.zeros(n)
    errors_ma = np.random.randn(n)
    for i in range(1, n):
        ma1[i] = errors_ma[i] + 0.5 * errors_ma[i-1]
    
    # ARIMA(1,1,1) model: AR(1) + MA(1) on differenced series, then integrated
    print("[ARIMA(1,1,1) Model Simulation]")
    # First, create ARMA(1,1) on differenced series (stationary)
    arma_diff = np.zeros(n)
    errors_arima = np.random.randn(n)
    for i in range(1, n):
        # AR part
        ar_part = 0.6 * arma_diff[i-1]
        # MA part
        ma_part = errors_arima[i] + 0.4 * errors_arima[i-1]
        # Combined (this is the differenced series)
        arma_diff[i] = ar_part + ma_part
    
    # Then integrate (cumsum) to get non-stationary ARIMA(1,1,1) series
    arima = np.cumsum(arma_diff)
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # AR(1)
    axes[0, 0].plot(ar1, 'b-', linewidth=1.5)
    axes[0, 0].set_title('AR(1): y_t = 0.7y_{t-1} + ε_t', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time', fontsize=10)
    axes[0, 0].set_ylabel('Value', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # AR(1) ACF
    plot_acf(ar1, ax=axes[0, 1], lags=20, title='AR(1) ACF')
    
    # MA(1)
    axes[1, 0].plot(ma1, 'g-', linewidth=1.5)
    axes[1, 0].set_title('MA(1): y_t = ε_t + 0.5ε_{t-1}', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time', fontsize=10)
    axes[1, 0].set_ylabel('Value', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # MA(1) ACF
    plot_acf(ma1, ax=axes[1, 1], lags=20, title='MA(1) ACF')
    
    # ARIMA(1,1,1)
    axes[2, 0].plot(arima, 'r-', linewidth=1.5)
    axes[2, 0].set_title('ARIMA(1,1,1)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Time', fontsize=10)
    axes[2, 0].set_ylabel('Value', fontsize=10)
    axes[2, 0].grid(True, alpha=0.3)
    
    # ARIMA(1,1,1) ACF
    plot_acf(arima, ax=axes[2, 1], lags=20, title='ARIMA(1,1,1) ACF')
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'arima_simulation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return ar1, ma1, arima


def identify_arima_parameters(series, max_p=3, max_d=2, max_q=3):
    """
    3. ARIMA Parameter Identification through ACF/PACF
    """
    print("\n" + "=" * 60)
    print("3. Model Identification through ACF/PACF")
    print("=" * 60)
    
    # Calculate ACF/PACF
    acf_values = acf(series, nlags=20, fft=True)
    pacf_values = pacf(series, nlags=20)
    
    print("\n[ACF (AutoCorrelation Function)]")
    print("  - AR model: Exponentially decreasing (long tail)")
    print("  - MA model: Cuts off after q (close to 0 after q)")
    
    print("\n[PACF (Partial AutoCorrelation Function)]")
    print("  - AR model: Cuts off after p (close to 0 after p)")
    print("  - MA model: Exponentially decreasing")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(series, ax=axes[0], lags=20, title='ACF (Autocorrelation Function)')
    plot_pacf(series, ax=axes[1], lags=20, title='PACF (Partial Autocorrelation Function)')
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'acf_pacf.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return acf_values, pacf_values


def fit_arima_model(ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    """
    4. Apply ARIMA Model to Real Financial Data
    """
    print("\n" + "=" * 60)
    print("4. Apply ARIMA Model to Real Financial Data")
    print("=" * 60)
    
    # Download data
    print(f"\nDownloading {ticker} data...")
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna() * 100  # Percentage
    
    print(f"Data period: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"Number of observations: {len(returns)}")
    
    # ACF/PACF analysis
    print("\n[ACF/PACF Analysis]")
    acf_vals, pacf_vals = identify_arima_parameters(returns)
    
    # Compare multiple ARIMA models
    print("\n[ARIMA Model Comparison]")
    models = [
        (1, 0, 1),  # ARIMA(1,0,1)
        (1, 1, 1),  # ARIMA(1,1,1)
        (2, 0, 2),  # ARIMA(2,0,2)
        (2, 1, 2),  # ARIMA(2,1,2)
    ]
    
    results = []
    for p, d, q in models:
        try:
            model = ARIMA(returns, order=(p, d, q))
            fitted = model.fit()
            results.append({
                'order': (p, d, q),
                'aic': fitted.aic,
                'bic': fitted.bic,
                'model': fitted
            })
            print(f"  ARIMA{p,d,q}: AIC={fitted.aic:.2f}, BIC={fitted.bic:.2f}")
        except Exception as e:
            print(f"  ARIMA{p,d,q}: Fitting failed - {e}")
    
    # Select optimal model (AIC criterion)
    if results:
        best_model = min(results, key=lambda x: x['aic'])
        print(f"\nOptimal model (AIC criterion): ARIMA{best_model['order']}")
        print(f"  AIC: {best_model['aic']:.2f}")
        print(f"  BIC: {best_model['bic']:.2f}")
        
        # Model summary
        print("\n[Model Summary]")
        print(best_model['model'].summary())
        
        # Forecast
        print("\n[Forecast]")
        forecast = best_model['model'].forecast(steps=10)
        print(f"10-day ahead forecast:")
        for i, pred in enumerate(forecast, 1):
            print(f"  Day {i}: {pred:.4f}%")
        
        # Residual analysis
        print("\n[Residual Analysis]")
        residuals = best_model['model'].resid
        ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
        print(f"Ljung-Box test (residual autocorrelation):")
        print(f"  p-value: {ljung_box['lb_pvalue'].iloc[-1]:.6f}")
        if ljung_box['lb_pvalue'].iloc[-1] > 0.05:
            print("  → No autocorrelation in residuals (good model fit)")
        else:
            print("  → Autocorrelation in residuals (model improvement needed)")
        
        # Visualization
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Original data
        axes[0].plot(returns.index, returns.values, 'b-', linewidth=1, alpha=0.7, label='Actual Returns')
        axes[0].plot(returns.index, best_model['model'].fittedvalues, 'r-', 
                    linewidth=1.5, label='ARIMA Fitted Values', alpha=0.8)
        axes[0].set_title(f'{ticker} Returns and ARIMA{best_model["order"]} Fit', 
                         fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Returns (%)', fontsize=10)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        axes[1].plot(residuals.index, residuals.values, 'g-', linewidth=1, alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[1].set_title('Residuals', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Residuals', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[2].hist(residuals.values, bins=50, density=True, alpha=0.7, color='purple')
        axes[2].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Residuals', fontsize=10)
        axes[2].set_ylabel('Density', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, f'arima_fit_{ticker}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_model['model'], forecast
    else:
        print("\nModel fitting failed")
        return None, None


def explain_aic_bic():
    """
    5. AIC/BIC Explanation
    """
    print("\n" + "=" * 60)
    print("5. Model Selection Criteria: AIC vs BIC")
    print("=" * 60)
    
    print("\n[AIC (Akaike Information Criterion)]")
    print("  AIC = -2 × log(Likelihood) + 2 × k")
    print("  k: Number of model parameters")
    print("  → Balance between fit and complexity")
    print("  → Lower is better")
    
    print("\n[BIC (Bayesian Information Criterion)]")
    print("  BIC = -2 × log(Likelihood) + log(n) × k")
    print("  n: Number of observations")
    print("  → BIC imposes larger penalty on complexity than AIC")
    print("  → Prefers simpler models")
    
    print("\n[Selection Criteria]")
    print("  - AIC: Emphasizes prediction accuracy")
    print("  - BIC: Emphasizes model simplicity")
    print("  - Generally select model with both low AIC and BIC")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 3: Probability & Time Series Statistics - Probabilistic Foundation of ARIMA Models")
    print("=" * 60)
    
    # 1. Explain ARIMA components
    explain_arima_components()
    
    # 2. Simulation example
    ar1, ma1, arima = generate_arima_example()
    
    # 3. ACF/PACF analysis
    identify_arima_parameters(ar1)
    
    # 4. Apply to real data
    model, forecast = fit_arima_model('AAPL')
    
    # 5. Explain AIC/BIC
    explain_aic_bic()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. AR: Past values influence current")
    print("2. I: Stationarization through differencing")
    print("3. MA: Past errors influence current")
    print("4. Select appropriate p, q using ACF/PACF")
    print("5. Select optimal model using AIC/BIC")
    print("6. Foundation of time series analysis")


if __name__ == "__main__":
    main()

