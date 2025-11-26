"""
Chapter 3: Prophet Model

This script demonstrates Facebook Prophet for:
1. Long-term trend forecasting
2. Seasonality detection
3. Price prediction
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
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("Warning: Prophet not available. Install with: pip install prophet")


def prophet_forecast(prices, periods=30, yearly_seasonality=True, weekly_seasonality=True):
    """
    Forecast using Prophet model
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    periods : int
        Number of periods to forecast
    yearly_seasonality : bool
        Include yearly seasonality
    weekly_seasonality : bool
        Include weekly seasonality
    
    Returns:
    --------
    dict
        Model and forecast results
    """
    if not HAS_PROPHET:
        raise ImportError("Prophet is not installed. Install with: pip install prophet")
    
    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': prices.index,
        'y': prices.values
    })
    
    # Create and fit model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode='additive'
    )
    
    model.fit(df)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Forecast
    forecast = model.predict(future)
    
    return {
        'model': model,
        'forecast': forecast,
        'future': future
    }


def visualize_prophet_results(data, nasdaq_results, tqqq_results, periods):
    """Visualize Prophet forecast results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # NASDAQ forecast
    nasdaq_forecast = nasdaq_results['forecast']
    nasdaq_model = nasdaq_results['model']
    
    axes[0, 0].plot(nasdaq_forecast['ds'], nasdaq_forecast['yhat'], 
                   label='Forecast', linewidth=2, color='blue')
    axes[0, 0].fill_between(nasdaq_forecast['ds'], 
                           nasdaq_forecast['yhat_lower'], 
                           nasdaq_forecast['yhat_upper'],
                           alpha=0.3, color='blue', label='Confidence Interval')
    axes[0, 0].plot(nasdaq_forecast['ds'][:-periods], 
                   nasdaq_forecast['yhat'][:-periods], 
                   label='Historical Fit', linewidth=1, color='green', alpha=0.7)
    axes[0, 0].set_title('NASDAQ Index: Prophet Forecast', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Price', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # TQQQ forecast
    tqqq_forecast = tqqq_results['forecast']
    tqqq_model = tqqq_results['model']
    
    axes[0, 1].plot(tqqq_forecast['ds'], tqqq_forecast['yhat'], 
                   label='Forecast', linewidth=2, color='orange')
    axes[0, 1].fill_between(tqqq_forecast['ds'], 
                           tqqq_forecast['yhat_lower'], 
                           tqqq_forecast['yhat_upper'],
                           alpha=0.3, color='orange', label='Confidence Interval')
    axes[0, 1].plot(tqqq_forecast['ds'][:-periods], 
                   tqqq_forecast['yhat'][:-periods], 
                   label='Historical Fit', linewidth=1, color='green', alpha=0.7)
    axes[0, 1].set_title('TQQQ ETF: Prophet Forecast', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Price', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # NASDAQ components - plot manually since plot_components doesn't support ax parameter
    nasdaq_components = nasdaq_forecast[['ds', 'trend', 'yearly', 'weekly']].copy()
    if 'yearly' in nasdaq_components.columns:
        axes[1, 0].plot(nasdaq_components['ds'], nasdaq_components['trend'], 
                        label='Trend', linewidth=2, color='blue')
        axes[1, 0].plot(nasdaq_components['ds'], nasdaq_components['yearly'], 
                        label='Yearly', linewidth=1, color='green', alpha=0.7)
        if 'weekly' in nasdaq_components.columns:
            axes[1, 0].plot(nasdaq_components['ds'], nasdaq_components['weekly'], 
                            label='Weekly', linewidth=1, color='orange', alpha=0.7)
    axes[1, 0].set_title('NASDAQ: Prophet Components', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Component Value', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # TQQQ components
    tqqq_components = tqqq_forecast[['ds', 'trend', 'yearly', 'weekly']].copy()
    if 'yearly' in tqqq_components.columns:
        axes[1, 1].plot(tqqq_components['ds'], tqqq_components['trend'], 
                       label='Trend', linewidth=2, color='orange')
        axes[1, 1].plot(tqqq_components['ds'], tqqq_components['yearly'], 
                        label='Yearly', linewidth=1, color='green', alpha=0.7)
        if 'weekly' in tqqq_components.columns:
            axes[1, 1].plot(tqqq_components['ds'], tqqq_components['weekly'], 
                            label='Weekly', linewidth=1, color='red', alpha=0.7)
    axes[1, 1].set_title('TQQQ: Prophet Components', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Component Value', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print forecast summary
    print(f"\n{'='*60}")
    print("Prophet Forecast Summary")
    print(f"{'='*60}")
    print(f"\nNASDAQ Forecast (next {periods} days):")
    print(f"  Current: ${nasdaq_forecast['yhat'].iloc[-periods-1]:.2f}")
    print(f"  Forecast: ${nasdaq_forecast['yhat'].iloc[-1]:.2f}")
    print(f"  Change: {((nasdaq_forecast['yhat'].iloc[-1] / nasdaq_forecast['yhat'].iloc[-periods-1]) - 1) * 100:.2f}%")
    print(f"  Lower bound: ${nasdaq_forecast['yhat_lower'].iloc[-1]:.2f}")
    print(f"  Upper bound: ${nasdaq_forecast['yhat_upper'].iloc[-1]:.2f}")
    
    print(f"\nTQQQ Forecast (next {periods} days):")
    print(f"  Current: ${tqqq_forecast['yhat'].iloc[-periods-1]:.2f}")
    print(f"  Forecast: ${tqqq_forecast['yhat'].iloc[-1]:.2f}")
    print(f"  Change: {((tqqq_forecast['yhat'].iloc[-1] / tqqq_forecast['yhat'].iloc[-periods-1]) - 1) * 100:.2f}%")
    print(f"  Lower bound: ${tqqq_forecast['yhat_lower'].iloc[-1]:.2f}")
    print(f"  Upper bound: ${tqqq_forecast['yhat_upper'].iloc[-1]:.2f}")


def main():
    """Main function"""
    print("="*60)
    print("Chapter 3: Prophet Model")
    print("Index and Leveraged ETF Forecast")
    print("="*60)
    
    if not HAS_PROPHET:
        print("\nError: Prophet is not installed.")
        print("Install with: pip install prophet")
        return
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    nasdaq = data['nasdaq']
    tqqq = data['tqqq']
    
    # Forecast periods
    periods = 30
    
    # Apply Prophet to NASDAQ
    print(f"\n1. Applying Prophet model to NASDAQ (forecasting {periods} days)...")
    nasdaq_results = prophet_forecast(nasdaq['Close'], periods=periods)
    
    # Apply Prophet to TQQQ
    print(f"2. Applying Prophet model to TQQQ (forecasting {periods} days)...")
    tqqq_results = prophet_forecast(tqqq['Close'], periods=periods)
    
    # Visualize
    visualize_prophet_results(data, nasdaq_results, tqqq_results, periods)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

