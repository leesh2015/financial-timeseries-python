"""
Time Series Decomposition - Seasonal Decomposition

This script demonstrates seasonal decomposition of time series data
to separate trend, seasonality, and residual components.

Additive Decomposition:
Y_t = Trend_t + Seasonal_t + Residual_t

where:
- Trend_t: Long-term direction (increasing/decreasing)
- Seasonal_t: Recurring patterns (seasonal cycles)
- Residual_t: Random noise (irregular component)

Properties:
- Additive model: Components are added together
- Multiplicative model: Y_t = Trend_t × Seasonal_t × Residual_t
- Decomposition helps identify patterns and make series stationary
- Residuals should be stationary (white noise) for good decomposition

This example uses airline passenger data with clear seasonal patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
DECOMPOSITION_MODEL = 'additive'  # 'additive' or 'multiplicative'


def main():
    """Main function"""
    # Load airline passenger data
    print(f"{'='*60}")
    print("Step 1: Loading Data")
    print("="*60)
    data = pd.read_csv(DATA_URL, index_col='Month', parse_dates=True)
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Decompose time series data
    print(f"\n{'='*60}")
    print(f"Step 2: Seasonal Decomposition ({DECOMPOSITION_MODEL.upper()} Model)")
    print("="*60)
    print(f"Decomposition formula: Y_t = Trend_t + Seasonal_t + Residual_t")
    result = seasonal_decompose(data['Passengers'], model=DECOMPOSITION_MODEL)
    
    # Plot trend, seasonality, and residuals
    print(f"\n{'='*60}")
    print("Step 3: Visualization")
    print("="*60)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Original data
    axes[0].plot(data['Passengers'], label='Original', color='black', linewidth=2)
    axes[0].set_ylabel('Passengers', fontsize=11)
    axes[0].set_title('Original Time Series', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Trend component
    axes[1].plot(result.trend, label='Trend', color='blue', linewidth=2)
    axes[1].set_ylabel('Trend', fontsize=11)
    axes[1].set_title('Trend Component (Long-term Direction)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal component
    axes[2].plot(result.seasonal, label='Seasonality', color='green', linewidth=2)
    axes[2].set_ylabel('Seasonal', fontsize=11)
    axes[2].set_title('Seasonal Component (Recurring Patterns)', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper left', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Residual component
    axes[3].plot(result.resid, label='Residuals', color='red', linewidth=1.5)
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[3].set_xlabel('Date', fontsize=11)
    axes[3].set_ylabel('Residual', fontsize=11)
    axes[3].set_title('Residual Component (Random Noise)', fontsize=12, fontweight='bold')
    axes[3].legend(loc='upper left', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print(f"\n{'='*60}")
    print("Step 4: Component Statistics")
    print("="*60)
    print(f"Original Series:")
    print(f"  Mean: {data['Passengers'].mean():.2f}")
    print(f"  Std: {data['Passengers'].std():.2f}")
    
    print(f"\nTrend Component:")
    print(f"  Mean: {result.trend.mean():.2f}")
    print(f"  Std: {result.trend.std():.2f}")
    
    print(f"\nSeasonal Component:")
    print(f"  Mean: {result.seasonal.mean():.6f} (≈ 0 for additive)")
    print(f"  Std: {result.seasonal.std():.2f}")
    
    print(f"\nResidual Component:")
    print(f"  Mean: {result.resid.mean():.6f} (≈ 0)")
    print(f"  Std: {result.resid.std():.2f}")
    print(f"  Residuals should be stationary (white noise)")


if __name__ == "__main__":
    main()
