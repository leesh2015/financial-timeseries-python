"""
Chapter 6: Wavelet Transform

This script demonstrates wavelet analysis for:
1. Multi-scale time-frequency analysis
2. Noise removal and signal decomposition
3. Trend and periodicity separation
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_nasdaq_tqqq_data

warnings.filterwarnings("ignore")

try:
    import pywt
    HAS_PYWAVELETS = True
except ImportError:
    HAS_PYWAVELETS = False
    print("Warning: PyWavelets not available. Install with: pip install pywavelets")


def wavelet_decomposition(returns, wavelet='db4', level=5):
    """
    Perform wavelet decomposition
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    wavelet : str
        Wavelet type (e.g., 'db4', 'haar', 'coif2')
    level : int
        Decomposition level
    
    Returns:
    --------
    dict
        Wavelet coefficients and reconstructed components
    """
    if not HAS_PYWAVELETS:
        raise ImportError("PyWavelets is not installed. Install with: pip install pywavelets")
    
    # Perform DWT
    coeffs = pywt.wavedec(returns.values, wavelet, level=level)
    
    # Reconstruct each level using upcoef (more reliable method)
    reconstructed = []
    
    # Reconstruct approximation (low frequency - trend)
    cA = coeffs[0]  # Approximation coefficients
    recon_approx = pywt.upcoef('a', cA, wavelet, level=level)
    recon_approx = recon_approx[:len(returns)]  # Trim to original length
    reconstructed.append(pd.Series(recon_approx, index=returns.index))
    
    # Reconstruct each detail level (high frequency components)
    cD = coeffs[1:]  # Detail coefficients
    for i, cD_i in enumerate(cD):
        # Detail level i+1 (since details start from level 1)
        detail_level = level - i
        recon_detail = pywt.upcoef('d', cD_i, wavelet, level=detail_level)
        recon_detail = recon_detail[:len(returns)]  # Trim to original length
        reconstructed.append(pd.Series(recon_detail, index=returns.index))
    
    return {
        'coeffs': coeffs,
        'approximation': reconstructed[0],  # Low frequency (trend)
        'details': reconstructed[1:],  # High frequency components
        'reconstructed_all': reconstructed,
        'wavelet': wavelet,
        'level': level
    }


def visualize_wavelet_results(data, nasdaq_results, tqqq_results, 
                              nasdaq_returns, tqqq_returns):
    """Visualize wavelet decomposition results"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    
    # NASDAQ original and approximation
    axes[0, 0].plot(nasdaq_returns.index, nasdaq_returns.values, 
                    label='Original Returns', alpha=0.5, linewidth=1, color='gray')
    axes[0, 0].plot(nasdaq_results['approximation'].index, 
                   nasdaq_results['approximation'].values,
                   label='Approximation (Trend)', linewidth=2, color='blue')
    axes[0, 0].set_title('NASDAQ: Original vs Wavelet Approximation', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Returns', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # TQQQ original and approximation
    axes[0, 1].plot(tqqq_returns.index, tqqq_returns.values, 
                    label='Original Returns', alpha=0.5, linewidth=1, color='gray')
    axes[0, 1].plot(tqqq_results['approximation'].index, 
                   tqqq_results['approximation'].values,
                   label='Approximation (Trend)', linewidth=2, color='orange')
    axes[0, 1].set_title('TQQQ: Original vs Wavelet Approximation', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Returns', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # NASDAQ detail components
    for i, detail in enumerate(nasdaq_results['details'][:3]):  # Show first 3 details
        axes[1, 0].plot(detail.index, detail.values, 
                       label=f'Detail Level {i+1}', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_title('NASDAQ: Detail Components (High Frequency)', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Returns', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # TQQQ detail components
    for i, detail in enumerate(tqqq_results['details'][:3]):  # Show first 3 details
        axes[1, 1].plot(detail.index, detail.values, 
                       label=f'Detail Level {i+1}', linewidth=1.5, alpha=0.7)
    axes[1, 1].set_title('TQQQ: Detail Components (High Frequency)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Returns', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # NASDAQ multi-scale analysis
    # Reconstruct signal from approximation only (denoised)
    denoised_nasdaq = nasdaq_results['approximation']
    axes[2, 0].plot(nasdaq_returns.index, nasdaq_returns.values, 
                    label='Original', alpha=0.5, linewidth=1, color='gray')
    axes[2, 0].plot(denoised_nasdaq.index, denoised_nasdaq.values,
                   label='Denoised (Approximation Only)', linewidth=2, color='blue')
    axes[2, 0].set_title('NASDAQ: Denoised Signal', fontsize=14, fontweight='bold')
    axes[2, 0].set_ylabel('Returns', fontsize=12)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # TQQQ multi-scale analysis
    denoised_tqqq = tqqq_results['approximation']
    axes[2, 1].plot(tqqq_returns.index, tqqq_returns.values, 
                   label='Original', alpha=0.5, linewidth=1, color='gray')
    axes[2, 1].plot(denoised_tqqq.index, denoised_tqqq.values,
                   label='Denoised (Approximation Only)', linewidth=2, color='orange')
    axes[2, 1].set_title('TQQQ: Denoised Signal', fontsize=14, fontweight='bold')
    axes[2, 1].set_ylabel('Returns', fontsize=12)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Energy distribution across scales
    nasdaq_energy = [np.sum(c**2) for c in nasdaq_results['coeffs']]
    tqqq_energy = [np.sum(c**2) for c in tqqq_results['coeffs']]
    
    levels = ['Approx'] + [f'Detail {i+1}' for i in range(len(nasdaq_energy)-1)]
    x_pos = np.arange(len(levels))
    
    axes[3, 0].bar(x_pos, nasdaq_energy, color='blue', alpha=0.7)
    axes[3, 0].set_xticks(x_pos)
    axes[3, 0].set_xticklabels(levels, rotation=45)
    axes[3, 0].set_title('NASDAQ: Energy Distribution Across Scales', 
                        fontsize=14, fontweight='bold')
    axes[3, 0].set_ylabel('Energy', fontsize=12)
    axes[3, 0].grid(True, alpha=0.3, axis='y')
    
    axes[3, 1].bar(x_pos, tqqq_energy, color='orange', alpha=0.7)
    axes[3, 1].set_xticks(x_pos)
    axes[3, 1].set_xticklabels(levels, rotation=45)
    axes[3, 1].set_title('TQQQ: Energy Distribution Across Scales', 
                         fontsize=14, fontweight='bold')
    axes[3, 1].set_ylabel('Energy', fontsize=12)
    axes[3, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Wavelet Decomposition Statistics")
    print(f"{'='*60}")
    print(f"\nNASDAQ:")
    print(f"  Original Std: {nasdaq_returns.std():.6f}")
    print(f"  Approximation Std: {nasdaq_results['approximation'].std():.6f}")
    print(f"  Noise Reduction: {(1 - nasdaq_results['approximation'].std() / nasdaq_returns.std()) * 100:.2f}%")
    
    print(f"\nTQQQ:")
    print(f"  Original Std: {tqqq_returns.std():.6f}")
    print(f"  Approximation Std: {tqqq_results['approximation'].std():.6f}")
    print(f"  Noise Reduction: {(1 - tqqq_results['approximation'].std() / tqqq_returns.std()) * 100:.2f}%")


def main():
    """Main function"""
    print("="*60)
    print("Chapter 6: Wavelet Transform")
    print("Multi-Scale Time-Frequency Analysis")
    print("="*60)
    
    if not HAS_PYWAVELETS:
        print("\nError: PyWavelets is not installed.")
        print("Install with: pip install pywavelets")
        return
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    nasdaq = data['nasdaq']
    tqqq = data['tqqq']
    
    nasdaq_returns = nasdaq['Returns'].dropna()
    tqqq_returns = tqqq['Returns'].dropna()
    
    # Apply wavelet decomposition to NASDAQ
    print("\n1. Applying wavelet decomposition to NASDAQ returns...")
    nasdaq_results = wavelet_decomposition(nasdaq_returns, wavelet='db4', level=5)
    
    # Apply wavelet decomposition to TQQQ
    print("2. Applying wavelet decomposition to TQQQ returns...")
    tqqq_results = wavelet_decomposition(tqqq_returns, wavelet='db4', level=5)
    
    # Visualize
    visualize_wavelet_results(data, nasdaq_results, tqqq_results,
                             nasdaq_returns, tqqq_returns)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

