"""
Chapter 5: Tree-Based Machine Learning Models

This script demonstrates XGBoost for:
1. Log return prediction
2. Feature importance analysis
3. Early stopping and model optimization
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_nasdaq_tqqq_data, prepare_sequences

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")


def create_features(returns, prices, lookback=20):
    """
    Create technical features for ML models
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    prices : pd.Series
        Price series
    lookback : int
        Lookback period for features
    
    Returns:
    --------
    pd.DataFrame
        Feature matrix
    """
    features = pd.DataFrame(index=returns.index)
    
    # Returns
    features['returns'] = returns
    features['returns_lag1'] = returns.shift(1)
    features['returns_lag2'] = returns.shift(2)
    features['returns_lag3'] = returns.shift(3)
    
    # Moving averages
    features['ma5'] = prices.rolling(5).mean() / prices - 1
    features['ma10'] = prices.rolling(10).mean() / prices - 1
    features['ma20'] = prices.rolling(20).mean() / prices - 1
    
    # Volatility
    features['volatility'] = returns.rolling(lookback).std()
    features['volatility_lag1'] = features['volatility'].shift(1)
    
    # Momentum
    features['momentum'] = prices / prices.shift(lookback) - 1
    
    # RSI (simplified)
    delta = returns
    gain = (delta.where(delta > 0, 0)).rolling(lookback).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Price position
    features['high_low_ratio'] = prices.rolling(lookback).max() / prices.rolling(lookback).min() - 1
    
    return features.dropna()


def xgboost_forecast(features, target, test_size=0.2, lookback=20, n_estimators=500, 
                     early_stopping_rounds=20):
    """
    Forecast using XGBoost with early stopping
    
    Parameters:
    -----------
    features : pd.DataFrame
        Feature matrix
    target : pd.Series
        Target variable (log returns)
    test_size : float
        Test set size ratio
    lookback : int
        Lookback period
    n_estimators : int
        Maximum number of estimators
    early_stopping_rounds : int
        Early stopping rounds
    
    Returns:
    --------
    dict
        Model and predictions
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    # Align data
    common_idx = features.index.intersection(target.index)
    X = features.loc[common_idx].values
    y = target.loc[common_idx].values
    
    # Train-test split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Further split training data for validation (for early stopping)
    val_split_idx = int(len(X_train) * 0.8)
    X_train_fit, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train_fit, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    
    # Create XGBoost model with improved parameters
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,  # Lower learning rate for better generalization
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,  # Regularization
        gamma=0.1,  # Regularization
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        eval_metric='rmse',  # Evaluation metric
        random_state=42,
        n_jobs=-1
    )
    
    # Train with early stopping
    print(f"   Training with early stopping (patience={early_stopping_rounds})...")
    
    # Try different XGBoost API styles for compatibility
    best_iteration = n_estimators
    
    try:
        # Try XGBoost 2.0+ style with callbacks
        from xgboost.callback import EarlyStopping
        early_stopping = EarlyStopping(rounds=early_stopping_rounds, save_best=True)
        model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_train_fit, y_train_fit), (X_val, y_val)],
            callbacks=[early_stopping],
            verbose=False
        )
    except (ImportError, AttributeError, TypeError):
        # Fallback: Try older XGBoost version style with early_stopping_rounds
        try:
            model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_train_fit, y_train_fit), (X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        except TypeError:
            # Last resort: Train without early stopping but with eval_set for monitoring
            print(f"   Warning: Early stopping not supported in this XGBoost version, training without it...")
            model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_train_fit, y_train_fit), (X_val, y_val)],
                verbose=False
            )
    
    # Get best iteration
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else n_estimators
    print(f"   Best iteration: {best_iteration}/{n_estimators}")
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"   Train RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")
    print(f"   Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    return {
        'model': model,
        'y_train': y_train,
        'y_train_pred': y_train_pred,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'best_iteration': best_iteration,
        'feature_importance': feature_importance,
        'test_index': common_idx[split_idx:],
        'feature_names': features.columns.tolist()
    }


def visualize_xgboost_results(nasdaq_results, tqqq_results, nasdaq_features, tqqq_features):
    """Visualize XGBoost results"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # NASDAQ predictions
    axes[0, 0].plot(nasdaq_results['y_test'], label='Actual', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(nasdaq_results['y_test_pred'], label='Predicted', alpha=0.7, linewidth=1.5, linestyle='--')
    axes[0, 0].set_title(f'NASDAQ Log Returns: XGBoost Prediction\nTest RMSE: {nasdaq_results["test_rmse"]:.6f}, R²: {nasdaq_results["test_r2"]:.4f}', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Log Returns', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # TQQQ predictions
    axes[0, 1].plot(tqqq_results['y_test'], label='Actual', alpha=0.7, linewidth=1.5, color='orange')
    axes[0, 1].plot(tqqq_results['y_test_pred'], label='Predicted', alpha=0.7, linewidth=1.5, 
                   linestyle='--', color='red')
    axes[0, 1].set_title(f'TQQQ Log Returns: XGBoost Prediction\nTest RMSE: {tqqq_results["test_rmse"]:.6f}, R²: {tqqq_results["test_r2"]:.4f}', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Log Returns', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # NASDAQ feature importance
    top_features_nasdaq = nasdaq_results['feature_importance'].head(10)
    axes[1, 0].barh(range(len(top_features_nasdaq)), top_features_nasdaq['importance'].values)
    axes[1, 0].set_yticks(range(len(top_features_nasdaq)))
    axes[1, 0].set_yticklabels(top_features_nasdaq['feature'].values)
    axes[1, 0].set_title('NASDAQ: Top 10 Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Importance', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # TQQQ feature importance
    top_features_tqqq = tqqq_results['feature_importance'].head(10)
    axes[1, 1].barh(range(len(top_features_tqqq)), top_features_tqqq['importance'].values, color='orange')
    axes[1, 1].set_yticks(range(len(top_features_tqqq)))
    axes[1, 1].set_yticklabels(top_features_tqqq['feature'].values)
    axes[1, 1].set_title('TQQQ: Top 10 Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Importance', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # Scatter plots
    axes[2, 0].scatter(nasdaq_results['y_test'], nasdaq_results['y_test_pred'], alpha=0.5)
    axes[2, 0].plot([nasdaq_results['y_test'].min(), nasdaq_results['y_test'].max()],
                    [nasdaq_results['y_test'].min(), nasdaq_results['y_test'].max()],
                    'r--', linewidth=2)
    axes[2, 0].set_xlabel('Actual Log Returns', fontsize=12)
    axes[2, 0].set_ylabel('Predicted Log Returns', fontsize=12)
    axes[2, 0].set_title('NASDAQ: Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].scatter(tqqq_results['y_test'], tqqq_results['y_test_pred'], alpha=0.5, color='orange')
    axes[2, 1].plot([tqqq_results['y_test'].min(), tqqq_results['y_test'].max()],
                    [tqqq_results['y_test'].min(), tqqq_results['y_test'].max()],
                    'r--', linewidth=2)
    axes[2, 1].set_xlabel('Actual Log Returns', fontsize=12)
    axes[2, 1].set_ylabel('Predicted Log Returns', fontsize=12)
    axes[2, 1].set_title('TQQQ: Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print(f"\n{'='*60}")
    print("XGBoost Model Performance (Log Returns)")
    print(f"{'='*60}")
    print(f"\nNASDAQ:")
    print(f"  Train RMSE: {nasdaq_results['train_rmse']:.6f}")
    print(f"  Test RMSE: {nasdaq_results['test_rmse']:.6f}")
    print(f"  Train MAE: {nasdaq_results['train_mae']:.6f}")
    print(f"  Test MAE: {nasdaq_results['test_mae']:.6f}")
    print(f"  Train R²: {nasdaq_results['train_r2']:.4f}")
    print(f"  Test R²: {nasdaq_results['test_r2']:.4f}")
    print(f"  Best Iteration: {nasdaq_results['best_iteration']}")
    
    print(f"\nTQQQ:")
    print(f"  Train RMSE: {tqqq_results['train_rmse']:.6f}")
    print(f"  Test RMSE: {tqqq_results['test_rmse']:.6f}")
    print(f"  Train MAE: {tqqq_results['train_mae']:.6f}")
    print(f"  Test MAE: {tqqq_results['test_mae']:.6f}")
    print(f"  Train R²: {tqqq_results['train_r2']:.4f}")
    print(f"  Test R²: {tqqq_results['test_r2']:.4f}")
    print(f"  Best Iteration: {tqqq_results['best_iteration']}")


def main():
    """Main function"""
    print("="*60)
    print("Chapter 5: Tree-Based Machine Learning Models (XGBoost)")
    print("Log Return Prediction with Early Stopping")
    print("="*60)
    
    if not HAS_XGBOOST:
        print("\nError: XGBoost is not installed.")
        print("Install with: pip install xgboost")
        return
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    nasdaq = data['nasdaq']
    tqqq = data['tqqq']
    
    # Create features using Log_Returns
    print("\n1. Creating features from log returns...")
    nasdaq_features = create_features(nasdaq['Log_Returns'].dropna(), nasdaq['Close'])
    tqqq_features = create_features(tqqq['Log_Returns'].dropna(), tqqq['Close'])
    
    # Apply XGBoost to NASDAQ
    print("\n2. Training XGBoost model on NASDAQ log returns...")
    nasdaq_results = xgboost_forecast(
        nasdaq_features, 
        nasdaq['Log_Returns'].dropna(),
        n_estimators=500,
        early_stopping_rounds=20
    )
    
    # Save NASDAQ model
    nasdaq_model_path = os.path.join(models_dir, 'xgboost_nasdaq.pkl')
    import pickle
    with open(nasdaq_model_path, 'wb') as f:
        pickle.dump(nasdaq_results['model'], f)
    print(f"   Saved NASDAQ model to: {nasdaq_model_path}")
    
    # Save NASDAQ metadata
    nasdaq_metadata = {
        'test_rmse': nasdaq_results['test_rmse'],
        'test_mae': nasdaq_results['test_mae'],
        'test_r2': nasdaq_results['test_r2'],
        'train_rmse': nasdaq_results['train_rmse'],
        'train_mae': nasdaq_results['train_mae'],
        'train_r2': nasdaq_results['train_r2'],
        'best_iteration': nasdaq_results['best_iteration'],
        'feature_names': nasdaq_results['feature_names'],
        'lookback': 20
    }
    nasdaq_metadata_path = os.path.join(models_dir, 'xgboost_nasdaq_metadata.json')
    with open(nasdaq_metadata_path, 'w') as f:
        json.dump(nasdaq_metadata, f, indent=2)
    print(f"   Saved NASDAQ metadata to: {nasdaq_metadata_path}")
    
    # Apply XGBoost to TQQQ
    print("\n3. Training XGBoost model on TQQQ log returns...")
    tqqq_results = xgboost_forecast(
        tqqq_features, 
        tqqq['Log_Returns'].dropna(),
        n_estimators=500,
        early_stopping_rounds=20
    )
    
    # Save TQQQ model
    tqqq_model_path = os.path.join(models_dir, 'xgboost_tqqq.pkl')
    with open(tqqq_model_path, 'wb') as f:
        pickle.dump(tqqq_results['model'], f)
    print(f"   Saved TQQQ model to: {tqqq_model_path}")
    
    # Save TQQQ metadata
    tqqq_metadata = {
        'test_rmse': tqqq_results['test_rmse'],
        'test_mae': tqqq_results['test_mae'],
        'test_r2': tqqq_results['test_r2'],
        'train_rmse': tqqq_results['train_rmse'],
        'train_mae': tqqq_results['train_mae'],
        'train_r2': tqqq_results['train_r2'],
        'best_iteration': tqqq_results['best_iteration'],
        'feature_names': tqqq_results['feature_names'],
        'lookback': 20
    }
    tqqq_metadata_path = os.path.join(models_dir, 'xgboost_tqqq_metadata.json')
    with open(tqqq_metadata_path, 'w') as f:
        json.dump(tqqq_metadata, f, indent=2)
    print(f"   Saved TQQQ metadata to: {tqqq_metadata_path}")
    
    # Visualize
    visualize_xgboost_results(nasdaq_results, tqqq_results, nasdaq_features, tqqq_features)
    
    print("\nAnalysis complete!")
    print(f"\nModels saved to: {models_dir}")
    print("You can now use these models in backtest_xgboost.py")


if __name__ == "__main__":
    main()

