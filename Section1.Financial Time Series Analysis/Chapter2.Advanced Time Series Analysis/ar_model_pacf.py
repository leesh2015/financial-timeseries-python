"""
AR Model and PACF (Partial Autocorrelation Function)

This script demonstrates how to use PACF to identify AR model order
and fit an AR(p) model to time series data.

AR(p) Model:
Y_t = φ₁Y_{t-1} + φ₂Y_{t-2} + ... + φₚY_{t-p} + ε_t

where:
- φᵢ: AR coefficients
- ε_t: white noise error term
- p: AR order

PACF (Partial Autocorrelation Function):
- Measures correlation between Y_t and Y_{t-k} after removing
  the effects of intermediate lags (Y_{t-1}, ..., Y_{t-k+1})
- PACF(k) = correlation(Y_t, Y_{t-k} | Y_{t-1}, ..., Y_{t-k+1})
- For AR(p) process: PACF cuts off after lag p
- Significant spikes at lags 1, 2, ..., p indicate AR order

This example generates AR(3) data and uses PACF to identify the order.
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Configuration
RANDOM_SEED = 42
N_SAMPLES = 100
AR_ORDER = 3
PHI_COEFFS = np.array([0.5, -0.3, 0.1])  # AR(3) coefficients
MAX_LAGS = 20


def generate_ar_process(n, phi, seed=None):
    """
    Generate AR(p) process data.
    
    Parameters:
    -----------
    n : int
        Number of observations
    phi : array-like
        AR coefficients [φ₁, φ₂, ..., φₚ]
    seed : int, optional
        Random seed
        
    Returns:
    --------
    np.ndarray
        Generated AR process
    """
    if seed is not None:
        np.random.seed(seed)
    
    p = len(phi)
    errors = np.random.normal(size=n)
    ar_data = np.zeros(n)
    
    for t in range(p, n):
        ar_data[t] = np.sum(phi * ar_data[t-p:t][::-1]) + errors[t]
    
    return ar_data


def main():
    """Main function"""
    # Generate synthetic AR(3) process data
    print(f"{'='*60}")
    print(f"Step 1: Generating AR({AR_ORDER}) Process")
    print("="*60)
    print(f"Model: Y_t = {PHI_COEFFS[0]:.1f}Y_{{t-1}} + {PHI_COEFFS[1]:.1f}Y_{{t-2}} + {PHI_COEFFS[2]:.1f}Y_{{t-3}} + ε_t")
    ar_data = generate_ar_process(N_SAMPLES, PHI_COEFFS, seed=RANDOM_SEED)
    print(f"Generated {len(ar_data)} observations")
    print(f"  Mean: {np.mean(ar_data):.4f}")
    print(f"  Std: {np.std(ar_data):.4f}")
    
    # Plot time series
    plt.figure(figsize=(12, 4))
    plt.plot(ar_data, linewidth=1.5)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'AR({AR_ORDER}) Process', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot PACF
    print(f"\n{'='*60}")
    print(f"Step 2: Partial Autocorrelation Function (PACF)")
    print("="*60)
    print("PACF helps identify AR order:")
    print("  - Significant spikes at lags 1, 2, ..., p indicate AR(p)")
    print("  - PACF cuts off after lag p for AR(p) process")
    
    plt.figure(figsize=(12, 6))
    plot_pacf(ar_data, lags=MAX_LAGS, ax=plt.gca(), method='ywm')
    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Partial Autocorrelation', fontsize=12)
    plt.title(f'Partial Autocorrelation Function (PACF) for AR({AR_ORDER}) Process', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Fit AR(3) model
    print(f"\n{'='*60}")
    print(f"Step 3: Fitting AR({AR_ORDER}) Model")
    print("="*60)
    model = ARIMA(ar_data, order=(AR_ORDER, 0, 0))
    results = model.fit()
    
    print(f"\n{'='*60}")
    print("AR Model Summary")
    print("="*60)
    print(results.summary())
    
    # Compare estimated vs true coefficients
    print(f"\n{'='*60}")
    print("Coefficient Comparison")
    print("="*60)
    estimated_coeffs = results.arparams
    print(f"True coefficients:     {PHI_COEFFS}")
    print(f"Estimated coefficients: {estimated_coeffs}")
    print(f"Difference:            {np.abs(PHI_COEFFS - estimated_coeffs)}")
    print(f"\nModel successfully captures AR({AR_ORDER}) structure")


if __name__ == "__main__":
    main()
