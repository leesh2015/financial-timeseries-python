"""
Chapter 5: Tree-Based Machine Learning Models

This script demonstrates XGBoost for:
1. Direction prediction (binary classification: up/down)
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
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, confusion_matrix, classification_report
    )
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


def create_features(returns, prices, lookback=20, max_features=None):
    """
    Create technical features for ML models
    Enhanced with more lag features to match LSTM's lookback window
    
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
    
    # Returns - DO NOT include current returns to avoid leakage
    # features['returns'] = returns  <-- REMOVED LEAKAGE
    
    # Extended lag features to capture more temporal patterns (like LSTM's 60-day lookback)
    # Add more lags to better capture sequential patterns
    for lag in [1, 2, 3, 5, 10, 20, 30, 60]:
        if lag <= len(returns):
            features[f'returns_lag{lag}'] = returns.shift(lag)
    
    # Rolling statistics to capture temporal patterns
    for window in [5, 10, 20, 30, 60]:
        if window <= len(returns):
            # Rolling mean of returns
            features[f'returns_ma{window}'] = returns.rolling(window).mean()
            # Rolling std of returns (volatility)
            features[f'returns_std{window}'] = returns.rolling(window).std()
            # Rolling skewness (asymmetry)
            features[f'returns_skew{window}'] = returns.rolling(window).skew()
    
    # Moving averages (price-based)
    for window in [5, 10, 20, 30, 60]:
        if window <= len(prices):
            features[f'ma{window}'] = prices.rolling(window).mean() / prices - 1
    
    # Volatility features
    features['volatility'] = returns.rolling(lookback).std()
    features['volatility_lag1'] = features['volatility'].shift(1)
    features['volatility_lag5'] = features['volatility'].shift(5)
    
    # Momentum features (multiple timeframes)
    for period in [5, 10, 20, 30, 60]:
        if period <= len(prices):
            features[f'momentum_{period}'] = prices / prices.shift(period) - 1
    
    # RSI (simplified) - multiple timeframes
    delta = returns
    for period in [14, 20, 30]:
        if period <= len(returns):
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # Price position features
    for period in [20, 30, 60]:
        if period <= len(prices):
            features[f'high_low_ratio_{period}'] = prices.rolling(period).max() / prices.rolling(period).min() - 1
            # Price position within range
            rolling_min = prices.rolling(period).min()
            rolling_max = prices.rolling(period).max()
            features[f'price_position_{period}'] = (prices - rolling_min) / (rolling_max - rolling_min + 1e-8)
    
    # Trend features
    for period in [5, 10, 20]:
        if period <= len(returns):
            # Count of up days in period
            features[f'up_days_count_{period}'] = (returns.shift(1) > 0).rolling(period).sum()
            # Average return when positive
            features[f'avg_positive_return_{period}'] = returns.shift(1).rolling(period).apply(
                lambda x: x[x > 0].mean() if (x > 0).any() else 0, raw=False
            )
            # Average return when negative
            features[f'avg_negative_return_{period}'] = returns.shift(1).rolling(period).apply(
                lambda x: x[x < 0].mean() if (x < 0).any() else 0, raw=False
            )
    
    features = features.dropna()
    
    # Feature selection: Remove highly correlated features to reduce noise
    # Always remove highly correlated features regardless of max_features
    if len(features.columns) > 1:
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        # Find highly correlated feature pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        # Find features to drop (highly correlated with others, threshold 0.95)
        to_drop = []
        for column in upper_triangle.columns:
            if any(upper_triangle[column] > 0.95):
                to_drop.append(column)
        
        if len(to_drop) > 0:
            features = features.drop(columns=to_drop)
            print(f"   Removed {len(to_drop)} highly correlated features (correlation > 0.95)")
            print(f"   Remaining features: {len(features.columns)}")
    
    return features


def xgboost_forecast(features, target, test_size=0.2, lookback=20, n_estimators=500, 
                     early_stopping_rounds=100, min_iterations=50):
    """
    Forecast using XGBoost for binary classification (direction prediction)
    
    Parameters:
    -----------
    features : pd.DataFrame
        Feature matrix
    target : pd.Series
        Target variable (log returns) - will be converted to binary (1 if > 0, 0 otherwise)
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
    
    # Shift features to avoid look-ahead bias
    # We want to predict target[t] using features[t-1]
    features_shifted = features.shift(1).dropna()
    
    # Align data
    common_idx = features_shifted.index.intersection(target.index)
    X = features_shifted.loc[common_idx].values
    y_returns = target.loc[common_idx].values
    
    # Convert returns to binary classification: 1 if positive (up), 0 if negative (down)
    y = (y_returns > 0).astype(int)
    
    # Print class distribution
    up_ratio = y.mean()
    print(f"   Class distribution: {up_ratio*100:.1f}% up days, {(1-up_ratio)*100:.1f}% down days")
    
    # Train-test split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Further split training data for validation (for early stopping)
    val_split_idx = int(len(X_train) * 0.8)
    X_train_fit, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train_fit, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    
    # Calculate class weights to handle class imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train_fit)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_fit)
    class_weight_dict = {int(cls): weight for cls, weight in zip(classes, class_weights)}
    scale_pos_weight = class_weight_dict[1] / class_weight_dict[0] if 0 in class_weight_dict and 1 in class_weight_dict else 1.0
    
    print(f"   Class weights: Class 0 (Down): {class_weight_dict.get(0, 1.0):.4f}, Class 1 (Up): {class_weight_dict.get(1, 1.0):.4f}")
    
    # Create XGBoost classifier with improved parameters
    # For XGBoost 2.1.1: early_stopping_rounds must be passed in constructor, not fit()
    # Strategy: Use manual early stopping to ensure minimum iterations
    # First train for minimum iterations, then apply early stopping
    
    # Step 1: Train for minimum iterations without early stopping
    # Using moderate regularization to avoid information loss while preventing overfitting
    print(f"   Training for minimum {min_iterations} iterations first...")
    model_no_early_stop = xgb.XGBClassifier(
        n_estimators=min_iterations,
        max_depth=6,  # Moderate depth to capture patterns without overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,  # Allow more learning, avoid information loss
        gamma=0,  # No minimum loss reduction, avoid information loss
        reg_alpha=0.1,  # Light L1 regularization
        reg_lambda=1.0,  # Moderate L2 regularization
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False
    )
    
    model_no_early_stop.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Step 2: Continue training with early stopping from the minimum iteration point
    # Using moderate regularization to avoid information loss while preventing overfitting
    print(f"   Continuing training with early stopping (patience={early_stopping_rounds})...")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,  # Moderate depth to capture patterns without overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,  # Allow more learning, avoid information loss
        gamma=0,  # No minimum loss reduction, avoid information loss
        reg_alpha=0.1,  # Light L1 regularization
        reg_lambda=1.0,  # Moderate L2 regularization
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False
    )
    
    # Continue training from the previous model
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        xgb_model=model_no_early_stop.get_booster(),  # Continue from previous model
        verbose=False
    )
    
    # Get best iteration
    # Note: Since we trained in two stages, best_iteration from second stage needs to be adjusted
    best_iteration = n_estimators
    try:
        # Method 1: Direct attribute (this will be relative to second stage)
        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
            best_iteration_second_stage = model.best_iteration
            # Add min_iterations because second stage started from min_iterations
            best_iteration = min_iterations + best_iteration_second_stage
            print(f"   Found best_iteration from model attribute (second stage): {best_iteration_second_stage}")
            print(f"   Adjusted best_iteration (including first stage): {best_iteration}")
        # Method 2: From booster
        elif hasattr(model, 'get_booster'):
            try:
                booster = model.get_booster()
                if hasattr(booster, 'best_iteration') and booster.best_iteration is not None:
                    best_iteration_second_stage = booster.best_iteration
                    best_iteration = min_iterations + best_iteration_second_stage
                    print(f"   Found best_iteration from booster (second stage): {best_iteration_second_stage}")
                    print(f"   Adjusted best_iteration (including first stage): {best_iteration}")
            except:
                pass
        # Method 3: From evals_result (combine both stages)
        if best_iteration == n_estimators or best_iteration == 0:
            # Get results from both stages
            evals_result_stage1 = model_no_early_stop.evals_result()
            evals_result_stage2 = model.evals_result()
            
            all_loglosses = []
            if evals_result_stage1 and 'validation_0' in evals_result_stage1:
                all_loglosses.extend(evals_result_stage1['validation_0']['logloss'])
            if evals_result_stage2 and 'validation_0' in evals_result_stage2:
                all_loglosses.extend(evals_result_stage2['validation_0']['logloss'])
            
            if all_loglosses and len(all_loglosses) > 0:
                best_idx = np.argmin(all_loglosses)
                best_iteration = best_idx + 1  # +1 because 0-indexed
                if best_iteration > len(all_loglosses):
                    best_iteration = len(all_loglosses)
                print(f"   Found best_iteration from combined evals_result: {best_iteration} (min logloss at index {best_idx})")
                print(f"   Validation logloss: {all_loglosses[best_idx]:.6f} (iteration {best_iteration})")
    except Exception as e:
        print(f"   Warning: Could not determine best iteration: {e}")
        import traceback
        traceback.print_exc()
        best_iteration = n_estimators
    
    # Ensure minimum iterations (should always be >= min_iterations now)
    if best_iteration < min_iterations:
        print(f"   Warning: Best iteration ({best_iteration}) < min_iterations ({min_iterations})")
        print(f"   Using min_iterations as best_iteration")
        best_iteration = min_iterations
    
    if best_iteration == 0:
        print(f"   Warning: Best iteration is 0, using min_iterations: {min_iterations}")
        best_iteration = min_iterations
    
    print(f"   Best iteration: {best_iteration}/{n_estimators}")
    
    # Predict probabilities
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold using ROC curve (Youden's J statistic)
    try:
        fpr, tpr, thresholds = roc_curve(y_train_fit, model.predict_proba(X_train_fit)[:, 1])
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_threshold = np.clip(optimal_threshold, 0.1, 0.9)
    except:
        optimal_threshold = 0.5
    
    # Convert probabilities to binary predictions
    y_train_pred = (y_train_pred_proba >= optimal_threshold).astype(int)
    y_test_pred = (y_test_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate classification metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    # ROC AUC
    try:
        train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
        test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    except:
        train_roc_auc = 0.0
        test_roc_auc = 0.0
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    print(f"   Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%), Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
    print(f"   Train ROC-AUC: {train_roc_auc:.4f}, Test ROC-AUC: {test_roc_auc:.4f}")
    
    return {
        'model': model,
        'y_train': y_train,
        'y_train_pred': y_train_pred,
        'y_train_pred_proba': y_train_pred_proba,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_pred_proba': y_test_pred_proba,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc,
        'optimal_threshold': optimal_threshold,
        'best_iteration': best_iteration,
        'feature_importance': feature_importance,
        'test_index': common_idx[split_idx:],
        'feature_names': features.columns.tolist()
    }


def visualize_xgboost_results(nasdaq_results, tqqq_results, nasdaq_features, tqqq_features):
    """Visualize XGBoost binary classification results"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # NASDAQ ROC Curve
    fpr_nasdaq, tpr_nasdaq, _ = roc_curve(nasdaq_results['y_test'], nasdaq_results['y_test_pred_proba'])
    axes[0, 0].plot(fpr_nasdaq, tpr_nasdaq, linewidth=2, label=f'ROC (AUC = {nasdaq_results["test_roc_auc"]:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 0].set_title(f'NASDAQ: ROC Curve\nAccuracy: {nasdaq_results["test_accuracy"]:.2%}, F1: {nasdaq_results["test_f1"]:.4f}', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # TQQQ ROC Curve
    fpr_tqqq, tpr_tqqq, _ = roc_curve(tqqq_results['y_test'], tqqq_results['y_test_pred_proba'])
    axes[0, 1].plot(fpr_tqqq, tpr_tqqq, linewidth=2, color='orange', 
                   label=f'ROC (AUC = {tqqq_results["test_roc_auc"]:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title(f'TQQQ: ROC Curve\nAccuracy: {tqqq_results["test_accuracy"]:.2%}, F1: {tqqq_results["test_f1"]:.4f}', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # NASDAQ Feature Importance
    top_features_nasdaq = nasdaq_results['feature_importance'].head(10)
    axes[1, 0].barh(range(len(top_features_nasdaq)), top_features_nasdaq['importance'].values, color='steelblue')
    axes[1, 0].set_yticks(range(len(top_features_nasdaq)))
    axes[1, 0].set_yticklabels(top_features_nasdaq['feature'].values, fontsize=10)
    axes[1, 0].set_xlabel('Importance', fontsize=12)
    axes[1, 0].set_title('NASDAQ: Top 10 Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()
    
    # TQQQ Feature Importance
    top_features_tqqq = tqqq_results['feature_importance'].head(10)
    axes[1, 1].barh(range(len(top_features_tqqq)), top_features_tqqq['importance'].values, color='orange')
    axes[1, 1].set_yticks(range(len(top_features_tqqq)))
    axes[1, 1].set_yticklabels(top_features_tqqq['feature'].values, fontsize=10)
    axes[1, 1].set_xlabel('Importance', fontsize=12)
    axes[1, 1].set_title('TQQQ: Top 10 Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()
    
    # NASDAQ Prediction Probabilities Distribution
    axes[2, 0].hist(nasdaq_results['y_test_pred_proba'][nasdaq_results['y_test'] == 0], 
                    bins=30, alpha=0.5, label='Down Days', color='red', density=True)
    axes[2, 0].hist(nasdaq_results['y_test_pred_proba'][nasdaq_results['y_test'] == 1], 
                    bins=30, alpha=0.5, label='Up Days', color='green', density=True)
    axes[2, 0].axvline(nasdaq_results['optimal_threshold'], color='black', linestyle='--', 
                      linewidth=2, label=f'Threshold: {nasdaq_results["optimal_threshold"]:.3f}')
    axes[2, 0].set_xlabel('Predicted Probability (Up)', fontsize=12)
    axes[2, 0].set_ylabel('Density', fontsize=12)
    axes[2, 0].set_title('NASDAQ: Prediction Probability Distribution', fontsize=14, fontweight='bold')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # TQQQ Prediction Probabilities Distribution
    axes[2, 1].hist(tqqq_results['y_test_pred_proba'][tqqq_results['y_test'] == 0], 
                    bins=30, alpha=0.5, label='Down Days', color='red', density=True)
    axes[2, 1].hist(tqqq_results['y_test_pred_proba'][tqqq_results['y_test'] == 1], 
                    bins=30, alpha=0.5, label='Up Days', color='green', density=True)
    axes[2, 1].axvline(tqqq_results['optimal_threshold'], color='black', linestyle='--', 
                      linewidth=2, label=f'Threshold: {tqqq_results["optimal_threshold"]:.3f}')
    axes[2, 1].set_xlabel('Predicted Probability (Up)', fontsize=12)
    axes[2, 1].set_ylabel('Density', fontsize=12)
    axes[2, 1].set_title('TQQQ: Prediction Probability Distribution', fontsize=14, fontweight='bold')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print(f"\n{'='*60}")
    print("XGBoost Model Performance (Binary Classification)")
    print(f"{'='*60}")
    print(f"\nNASDAQ:")
    print(f"  Classification Threshold: {nasdaq_results['optimal_threshold']:.4f}")
    print(f"  Train Accuracy: {nasdaq_results['train_accuracy']:.4f} ({nasdaq_results['train_accuracy']*100:.2f}%)")
    print(f"  Test Accuracy: {nasdaq_results['test_accuracy']:.4f} ({nasdaq_results['test_accuracy']*100:.2f}%)")
    print(f"  Train Precision: {nasdaq_results['train_precision']:.4f}")
    print(f"  Test Precision: {nasdaq_results['test_precision']:.4f}")
    print(f"  Train Recall: {nasdaq_results['train_recall']:.4f}")
    print(f"  Test Recall: {nasdaq_results['test_recall']:.4f}")
    print(f"  Train F1: {nasdaq_results['train_f1']:.4f}")
    print(f"  Test F1: {nasdaq_results['test_f1']:.4f}")
    print(f"  Train ROC-AUC: {nasdaq_results['train_roc_auc']:.4f}")
    print(f"  Test ROC-AUC: {nasdaq_results['test_roc_auc']:.4f}")
    print(f"  Best Iteration: {nasdaq_results['best_iteration']}")
    
    print(f"\nTQQQ:")
    print(f"  Classification Threshold: {tqqq_results['optimal_threshold']:.4f}")
    print(f"  Train Accuracy: {tqqq_results['train_accuracy']:.4f} ({tqqq_results['train_accuracy']*100:.2f}%)")
    print(f"  Test Accuracy: {tqqq_results['test_accuracy']:.4f} ({tqqq_results['test_accuracy']*100:.2f}%)")
    print(f"  Train Precision: {tqqq_results['train_precision']:.4f}")
    print(f"  Test Precision: {tqqq_results['test_precision']:.4f}")
    print(f"  Train Recall: {tqqq_results['train_recall']:.4f}")
    print(f"  Test Recall: {tqqq_results['test_recall']:.4f}")
    print(f"  Train F1: {tqqq_results['train_f1']:.4f}")
    print(f"  Test F1: {tqqq_results['test_f1']:.4f}")
    print(f"  Train ROC-AUC: {tqqq_results['train_roc_auc']:.4f}")
    print(f"  Test ROC-AUC: {tqqq_results['test_roc_auc']:.4f}")
    print(f"  Best Iteration: {tqqq_results['best_iteration']}")


def main():
    """Main function"""
    print("="*60)
    print("Chapter 5: Tree-Based Machine Learning Models (XGBoost)")
    print("Direction Prediction (Binary Classification) with Early Stopping")
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
        early_stopping_rounds=100,  # Increased patience for better learning
        min_iterations=50  # Minimum iterations before early stopping can trigger
    )
    
    # Save NASDAQ model
    nasdaq_model_path = os.path.join(models_dir, 'xgboost_nasdaq.pkl')
    import pickle
    with open(nasdaq_model_path, 'wb') as f:
        pickle.dump(nasdaq_results['model'], f)
    print(f"   Saved NASDAQ model to: {nasdaq_model_path}")
    
    # Save NASDAQ metadata
    nasdaq_metadata = {
        'test_accuracy': nasdaq_results['test_accuracy'],
        'test_precision': nasdaq_results['test_precision'],
        'test_recall': nasdaq_results['test_recall'],
        'test_f1': nasdaq_results['test_f1'],
        'test_roc_auc': nasdaq_results['test_roc_auc'],
        'train_accuracy': nasdaq_results['train_accuracy'],
        'train_precision': nasdaq_results['train_precision'],
        'train_recall': nasdaq_results['train_recall'],
        'train_f1': nasdaq_results['train_f1'],
        'train_roc_auc': nasdaq_results['train_roc_auc'],
        'optimal_threshold': nasdaq_results['optimal_threshold'],
        'best_iteration': nasdaq_results['best_iteration'],
        'feature_names': nasdaq_results['feature_names'],
        'lookback': 20,
        'model_type': 'binary_classification'
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
        early_stopping_rounds=100,  # Increased patience for better learning
        min_iterations=50  # Minimum iterations before early stopping can trigger
    )
    
    # Save TQQQ model
    tqqq_model_path = os.path.join(models_dir, 'xgboost_tqqq.pkl')
    with open(tqqq_model_path, 'wb') as f:
        pickle.dump(tqqq_results['model'], f)
    print(f"   Saved TQQQ model to: {tqqq_model_path}")
    
    # Save TQQQ metadata
    tqqq_metadata = {
        'test_accuracy': tqqq_results['test_accuracy'],
        'test_precision': tqqq_results['test_precision'],
        'test_recall': tqqq_results['test_recall'],
        'test_f1': tqqq_results['test_f1'],
        'test_roc_auc': tqqq_results['test_roc_auc'],
        'train_accuracy': tqqq_results['train_accuracy'],
        'train_precision': tqqq_results['train_precision'],
        'train_recall': tqqq_results['train_recall'],
        'train_f1': tqqq_results['train_f1'],
        'train_roc_auc': tqqq_results['train_roc_auc'],
        'optimal_threshold': tqqq_results['optimal_threshold'],
        'best_iteration': tqqq_results['best_iteration'],
        'feature_names': tqqq_results['feature_names'],
        'lookback': 20,
        'model_type': 'binary_classification'
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

