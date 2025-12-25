"""
Chapter 2: Calculus & Analysis - Wavelet Transform

Core Analogy: "Decomposing signals at various frequencies"
- Fourier Transform: Loss of time information
- Wavelet: Preserves both time and frequency information
- Multi-resolution analysis: Finding patterns at various scales

This example demonstrates:
1. Mathematical principles of Wavelet
2. Time series decomposition and noise removal
3. Multi-resolution analysis
4. Application to financial time series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
try:
    import pywt
    HAS_PYWAVELETS = True
except ImportError:
    HAS_PYWAVELETS = False
    print("Warning: pywavelets not available. Install with: pip install pywavelets")

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_wavelet():
    """
    1. Mathematical principles of Wavelet
    """
    print("=" * 60)
    print("1. Mathematical Principles of Wavelet Transform")
    print("=" * 60)
    
    print("\n[Wavelet Transform]")
    print("  W(a,b) = (1/√a) ∫ f(t) × ψ*((t-b)/a) dt")
    print("  Where:")
    print("    ψ(t): Wavelet function (mother wavelet)")
    print("    a: Scale (frequency)")
    print("    b: Position (time)")
    
    print("\n[Advantages of Wavelet]")
    print("  1. Time-frequency localization")
    print("  2. Multi-resolution analysis")
    print("  3. Noise removal")
    print("  4. Feature extraction")
    
    print("\n[Major Wavelets]")
    print("  - Haar: Simplest, step function")
    print("  - Daubechies (db): Smooth function")
    print("  - Coiflet: Symmetry")
    
    print("\n[Financial Applications]")
    print("  - Volatility clustering analysis")
    print("  - Noise removal")
    print("  - Multi-timeframe pattern discovery")


def wavelet_decomposition(series, wavelet='db4', level=4):
    """
    2. Wavelet Decomposition
    """
    if not HAS_PYWAVELETS:
        print("\n⚠ pywavelets package not available, using simulation.")
        # Simple simulation
        n = len(series)
        coeffs = []
        for i in range(level):
            size = n // (2 ** (i + 1))
            coeffs.append((np.random.randn(size), np.random.randn(size)))
        return coeffs, None
    
    # Wavelet decomposition
    coeffs = pywt.wavedec(series, wavelet, level=level)
    
    # Separate coefficients
    cA = coeffs[0]  # Approximation (low frequency)
    cD_list = coeffs[1:]  # Detail (high frequency)
    
    return (cA, cD_list), wavelet


def denoise_with_wavelet(series, wavelet='db4', threshold_mode='soft'):
    """
    3. Noise Removal using Wavelet
    """
    print("\n" + "=" * 60)
    print("3. Wavelet Noise Removal")
    print("=" * 60)
    
    if not HAS_PYWAVELETS:
        print("⚠ pywavelets package is required.")
        return series, series
    
    # Wavelet decomposition
    coeffs = pywt.wavedec(series, wavelet, mode='symmetric')
    
    # Calculate threshold (Universal threshold)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(series)))
    
    # Apply threshold
    coeffs_thresh = list(coeffs)
    coeffs_thresh[1:] = [pywt.threshold(i, threshold, threshold_mode) 
                        for i in coeffs_thresh[1:]]
    
    # Reconstruction
    denoised = pywt.waverec(coeffs_thresh, wavelet, mode='symmetric')
    
    print(f"  Original standard deviation: {series.std():.4f}")
    print(f"  Standard deviation after denoising: {denoised.std():.4f}")
    print(f"  Threshold: {threshold:.4f}")
    
    return denoised, threshold


def analyze_financial_wavelet(ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    """
    4. Wavelet Analysis of Real Financial Data
    """
    print("\n" + "=" * 60)
    print("4. Wavelet Analysis of Real Financial Data")
    print("=" * 60)
    
    # Download data
    print(f"\nDownloading {ticker} data...")
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()
    
    print(f"Data period: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"Number of observations: {len(returns)}")
    
    if not HAS_PYWAVELETS:
        print("\n⚠ pywavelets package is required.")
        return
    
    # Wavelet decomposition
    wavelet = 'db4'
    level = 4
    coeffs = pywt.wavedec(returns.values, wavelet, level=level)
    
    # Reconstruct each level
    reconstructed_levels = []
    for i in range(level + 1):
        coeffs_i = [coeffs[0] if j == 0 else np.zeros_like(coeffs[j]) if j != i+1 else coeffs[j]
                   for j in range(len(coeffs))]
        if i == 0:
            coeffs_i = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        else:
            coeffs_i = [coeffs[0]] + [coeffs[j] if j == i else np.zeros_like(coeffs[j]) 
                       for j in range(1, len(coeffs))]
        recon = pywt.waverec(coeffs_i, wavelet, mode='symmetric')
        reconstructed_levels.append(recon[:len(returns)])
    
    # Noise removal
    denoised, threshold = denoise_with_wavelet(returns.values, wavelet='db4')
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Original vs denoised
    axes[0].plot(returns.index, returns.values, 'b-', linewidth=1, alpha=0.5, label='Original')
    axes[0].plot(returns.index, denoised, 'r-', linewidth=1.5, label='Denoised')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Returns', fontsize=12)
    axes[0].set_title(f'{ticker} Returns: Original vs Denoised', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Multi-resolution decomposition
    axes[1].plot(returns.index, reconstructed_levels[0], 'g-', linewidth=2, 
               label=f'Approximation (Level {level})', alpha=0.8)
    for i in range(1, min(level + 1, len(reconstructed_levels))):
        axes[1].plot(returns.index, reconstructed_levels[i], 
                   linewidth=1.5, label=f'Detail (Level {i})', alpha=0.7)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Returns', fontsize=12)
    axes[1].set_title('Multi-Resolution Decomposition', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Wavelet coefficients (energy)
    energy_by_level = [np.sum(c**2) for c in coeffs]
    axes[2].bar(range(len(energy_by_level)), energy_by_level, alpha=0.7, color='purple')
    axes[2].set_xlabel('Level', fontsize=12)
    axes[2].set_ylabel('Energy (Sum of Squared Coefficients)', fontsize=12)
    axes[2].set_title('Wavelet Coefficient Energy', fontsize=14, fontweight='bold')
    axes[2].set_xticks(range(len(energy_by_level)))
    axes[2].set_xticklabels([f'A{level}'] + [f'D{i}' for i in range(level, 0, -1)])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'wavelet_{ticker}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n[Result Interpretation]")
    print("  - Approximation: Long-term trend (low frequency)")
    print("  - Detail: Short-term fluctuations (high frequency)")
    print("  - Noise removal: High-frequency components removed")
    print("  - Multi-resolution: Pattern discovery at various time scales")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 2: Calculus & Analysis - Wavelet Transform")
    print("=" * 60)
    
    # 1. Explain Wavelet principles
    explain_wavelet()
    
    # 2. Analyze real financial data
    analyze_financial_wavelet('AAPL')
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Wavelet: Time-frequency localization")
    print("2. Multi-resolution analysis: Pattern discovery at various scales")
    print("3. Noise removal: High-frequency components removed")
    print("4. Volatility clustering analysis in financial time series")
    print("5. Wavelet Transform fundamentals")


if __name__ == "__main__":
    main()

