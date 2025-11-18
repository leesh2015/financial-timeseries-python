"""
Random Walk Analysis

This script demonstrates the properties of a random walk process
and how differencing transforms it into a stationary white noise process.

Random Walk:
Y_t = Y_{t-1} + ε_t

where:
- ε_t ~ N(0, σ²): white noise (i.i.d. normal errors)
- Y_0: initial value (often 0)

Properties:
- Non-stationary: variance increases with time
- E[Y_t] = Y_0 (constant mean)
- Var[Y_t] = t × σ² (variance grows linearly with time)
- Unit root process: needs differencing to become stationary

Differencing:
ΔY_t = Y_t - Y_{t-1} = ε_t

After differencing:
- Stationary: E[ΔY_t] = 0, Var[ΔY_t] = σ² (constant)
- White noise: no autocorrelation
- Suitable for time series modeling

This example generates a random walk and demonstrates its properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

# Configuration
MEAN = 0
VARIANCE = 1
SIZE = 1000
RANDOM_SEED = 42


def main():
    """Main function"""
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    print(f"{'='*60}")
    print("Step 1: Generating Random Walk")
    print("="*60)
    print(f"Model: Y_t = Y_{{t-1}} + ε_t, where ε_t ~ N({MEAN}, {VARIANCE})")
    
    # Generate random walk
    errors = np.random.normal(loc=MEAN, scale=np.sqrt(VARIANCE), size=SIZE)
    random_walk = np.cumsum(errors)
    
    print(f"Generated {SIZE} observations")
    print(f"  Initial value: {random_walk[0]:.4f}")
    print(f"  Final value: {random_walk[-1]:.4f}")
    print(f"  Mean: {np.mean(random_walk):.4f}")
    print(f"  Variance: {np.var(random_walk):.4f} (grows with time)")
    
    # Plot random walk
    plt.figure(figsize=(12, 6))
    plt.plot(random_walk, label='Random Walk', linewidth=1.5, color='blue')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Random Walk Process (Non-stationary)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate differences (differencing)
    print(f"\n{'='*60}")
    print("Step 2: Differencing (ΔY_t = Y_t - Y_{t-1} = ε_t)")
    print("="*60)
    diff_series = np.diff(random_walk)
    
    print(f"Differenced series shape: {diff_series.shape}")
    print(f"  Mean: {np.mean(diff_series):.6f} (≈ 0)")
    print(f"  Variance: {np.var(diff_series):.6f} (≈ {VARIANCE}, constant)")
    print(f"  Differenced series is stationary (white noise)")
    
    # Plot differenced series
    plt.figure(figsize=(12, 6))
    plt.plot(diff_series, label='Differenced Series (White Noise)', linewidth=1, color='green', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Difference', fontsize=12)
    plt.title('Differenced Series (Stationary White Noise)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate mean and variance of differenced series
    mean_diff = np.mean(diff_series)
    variance_diff = np.var(diff_series)
    std_diff = np.sqrt(variance_diff)
    
    print(f"\n{'='*60}")
    print("Step 3: Statistical Properties")
    print("="*60)
    print(f"Mean of Differenced Series: {mean_diff:.6f} (≈ 0)")
    print(f"Variance of Differenced Series: {variance_diff:.6f} (≈ {VARIANCE})")
    print(f"Standard Deviation: {std_diff:.6f} (≈ {np.sqrt(VARIANCE)})")
    
    # Plot histogram of differenced series
    print(f"\n{'='*60}")
    print("Step 4: Distribution Analysis")
    print("="*60)
    plt.figure(figsize=(12, 6))
    plt.hist(diff_series, bins=30, density=True, alpha=0.6, color='g', label='Empirical')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_diff, std_diff)
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal(μ={mean_diff:.2f}, σ={std_diff:.2f})')
    plt.xlabel('Difference', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Differenced Series (Should be Normal)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate autocovariance and autocorrelation
    print(f"\n{'='*60}")
    print("Step 5: Autocorrelation Analysis")
    print("="*60)
    autocov = np.correlate(diff_series, diff_series, mode='full')[len(diff_series)-1:]
    autocorr = autocov / autocov[0]
    
    print(f"Autocorrelation at lag 0: {autocorr[0]:.6f} (always 1.0)")
    print(f"Autocorrelation at lag 1: {autocorr[1]:.6f} (≈ 0 for white noise)")
    print(f"Max autocorrelation (lags 1-10): {np.max(np.abs(autocorr[1:11])):.6f}")
    print(f"  No significant autocorrelation (white noise property)")
    
    # Plot autocovariance and autocorrelation
    plt.figure(figsize=(14, 6))
    
    plt.subplot(121)
    plt.bar(range(len(autocov[:20])), autocov[:20], alpha=0.6, align='center', color='blue')
    plt.xlabel('Lag', fontsize=11)
    plt.ylabel('Autocovariance', fontsize=11)
    plt.title('Autocovariance Function', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(122)
    plt.bar(range(len(autocorr[:20])), autocorr[:20], alpha=0.6, align='center', color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Lag', fontsize=11)
    plt.ylabel('Autocorrelation', fontsize=11)
    plt.title('Autocorrelation Function (Should be ~0 for lags > 0)', 
              fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print("Summary")
    print("="*60)
    print("  Random walk is non-stationary (variance grows with time)")
    print("  After differencing: stationary white noise process")
    print("  Differenced series has:")
    print("    - Mean ≈ 0")
    print("    - Constant variance")
    print("    - No autocorrelation (white noise)")
    print("    - Normal distribution")


if __name__ == "__main__":
    main()
