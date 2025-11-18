"""
Auto-ARIMA Model Selection with Outlier Detection

This script demonstrates how to use Auto-ARIMA to automatically find the optimal
ARIMA model order, and how to handle outliers in the data.

Auto-ARIMA uses information criteria (AIC/BIC) to select the best model automatically.
"""

from pmdarima import auto_arima
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
END_DATE = '2024-12-01'
OUTLIER_THRESHOLD = 3  # Standard deviations for outlier detection


def main():
    """Main function"""
    # Load data
    if USE_UTILS:
        data = load_financial_data(TICKER, START_DATE, END_DATE)
        data = preprocess_data(data, frequency='B', use_adjusted=False)
        close_prices = data['Close']
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        close_prices = data['Close']
        close_prices = close_prices.asfreq('B')
        close_prices = close_prices.dropna()
    
    # Step 1: Use AutoARIMA to find the optimal ARIMA model
    print("="*60)
    print("Step 1: Finding Optimal ARIMA Model with Auto-ARIMA")
    print("="*60)
    model = auto_arima(
        close_prices, 
        seasonal=False,  # Search for a non-seasonal ARIMA model
        trace=True,  # Print the model selection process
        error_action='ignore',  # Ignore errors and proceed
        suppress_warnings=True,  # Suppress warning messages
        stepwise=True  # Perform stepwise search for efficiency
    )
    
    # Print the summary of the model
    print("\nOptimal ARIMA Model Summary:")
    print(model.summary())
    
    # Print the estimated coefficients
    print("\nEstimated coefficients:")
    print(model.params())
    
    # Step 2: Outlier detection and removal
    print("\n" + "="*60)
    print("Step 2: Outlier Detection and Removal")
    print("="*60)
    
    # Extract residuals from the fitted model
    residuals = model.resid()
    
    # Calculate the mean and standard deviation of residuals
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    
    # Identify outliers (using threshold standard deviations)
    outliers = np.abs(residuals - mean_residuals) > OUTLIER_THRESHOLD * std_residuals
    n_outliers = np.sum(outliers)
    
    print(f"Outliers detected: {n_outliers} ({n_outliers/len(residuals)*100:.2f}%)")
    print(f"Threshold: {OUTLIER_THRESHOLD} standard deviations")
    
    # Remove outliers from the original data
    # Note: We need to remove the corresponding data points, not just residuals
    outlier_indices = residuals[outliers].index
    close_prices_no_outliers = close_prices.drop(outlier_indices)
    
    # Step 3: Re-fit model without outliers
    print("\n" + "="*60)
    print("Step 3: Re-fitting Model Without Outliers")
    print("="*60)
    model_no_outliers = auto_arima(
        close_prices_no_outliers, 
        seasonal=False, 
        trace=True, 
        error_action='ignore', 
        suppress_warnings=True, 
        stepwise=True
    )
    
    print("\n" + "="*50)
    print("Model Summary (Without Outliers):")
    print("="*50)
    print(model_no_outliers.summary())
    
    # Print the estimated coefficients without outliers
    print("\nEstimated coefficients without outliers:")
    print(model_no_outliers.params())
    
    # Extract residuals from the model without outliers for plotting
    residuals_no_outliers = model_no_outliers.resid()
    
    # Step 4: Visualize residuals
    print("\n" + "="*60)
    print("Step 4: Residual Analysis")
    print("="*60)
    
    # Plot the histogram of residuals without outliers
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(residuals_no_outliers, kde=True, bins=30, color='skyblue')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals (Without Outliers)')
    plt.grid(True, alpha=0.3)
    
    # Plot the scatter plot of residuals without outliers
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(residuals_no_outliers)), residuals_no_outliers, alpha=0.5, color='skyblue')
    plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero line')
    plt.xlabel('Index')
    plt.ylabel('Residuals')
    plt.title('Residuals Scatter Plot (Without Outliers)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)
    print(f"\nOriginal Model Order: {model.order}")
    print(f"Model Without Outliers Order: {model_no_outliers.order}")


if __name__ == "__main__":
    main()


