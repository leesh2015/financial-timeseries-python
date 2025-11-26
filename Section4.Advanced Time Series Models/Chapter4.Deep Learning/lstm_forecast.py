"""
Chapter 4: Deep Learning Models for Time Series

This script demonstrates LSTM (Long Short-Term Memory) for:
1. Return prediction
2. Long-term dependency learning
3. Complex nonlinear pattern recognition
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

from utils.data_loader import load_nasdaq_tqqq_data, prepare_sequences

warnings.filterwarnings("ignore")

try:
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, confusion_matrix, classification_report
    )
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow/Keras not available. Install with: pip install tensorflow")


def prepare_lstm_data(returns, lookback=60, forecast_horizon=1, test_size=0.2, 
                      prediction_type='direction'):
    """
    Prepare data for LSTM model
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    lookback : int
        Number of time steps to look back
    forecast_horizon : int
        Number of time steps to forecast
    test_size : float
        Test set size ratio
    prediction_type : str
        'direction' for direction prediction (binary classification)
        'value' for return value prediction (regression)
    
    Returns:
    --------
    dict
        Prepared data
    """
    # Scale input features (returns) using TRAINING SPLIT ONLY to avoid leakage
    raw_values = returns.values.reshape(-1, 1)
    split_point = max(lookback, int(len(raw_values) * (1 - test_size)))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(raw_values[:split_point])
    scaled_returns = scaler.transform(raw_values).flatten()
    
    # Create sequences for input (X)
    X, y_raw = prepare_sequences(pd.Series(scaled_returns, index=returns.index),
                                 lookback, forecast_horizon)
    
    # Convert target to direction (binary classification)
    if prediction_type == 'direction':
        # y_raw is the future return value
        # Convert to direction: 1 if positive (up), 0 if negative (down)
        if forecast_horizon == 1:
            y = (y_raw.flatten() > 0).astype(int)
        else:
            # For multi-step, use average direction
            y = (y_raw.mean(axis=1) > 0).astype(int)
    else:
        # Regression: keep original values
        y = y_raw.flatten() if forecast_horizon == 1 else y_raw.mean(axis=1)
    
    # Reshape for LSTM: (samples, time_steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Train-test split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'test_index': returns.index[split_idx + lookback:],
        'prediction_type': prediction_type
    }


def build_lstm_model(lookback=60, forecast_horizon=1, prediction_type='direction'):
    """
    Build LSTM model
    
    Parameters:
    -----------
    lookback : int
        Number of time steps to look back
    forecast_horizon : int
        Number of time steps to forecast
    prediction_type : str
        'direction' for binary classification, 'value' for regression
    
    Returns:
    --------
    model
        Compiled LSTM model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid' if prediction_type == 'direction' else None)
    ])
    
    if prediction_type == 'direction':
        # Binary classification
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        # Regression
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
    
    return model


def lstm_forecast(returns, lookback=60, forecast_horizon=1, epochs=50, batch_size=32,
                  prediction_type='direction'):
    """
    Forecast using LSTM (Direction Prediction - Binary Classification)
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    lookback : int
        Number of time steps to look back
    forecast_horizon : int
        Number of time steps to forecast
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    prediction_type : str
        'direction' for direction prediction (binary classification)
        'value' for return value prediction (regression)
    
    Returns:
    --------
    dict
        Model and predictions
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
    
    # Prepare data
    data = prepare_lstm_data(returns, lookback, forecast_horizon, prediction_type=prediction_type)
    
    # Print data statistics
    print(f"\nData Statistics:")
    print(f"  Total samples: {len(returns)}")
    print(f"  Training samples: {len(data['X_train'])}")
    print(f"  Test samples: {len(data['X_test'])}")
    print(f"  Returns range: [{returns.min():.6f}, {returns.max():.6f}]")
    print(f"  Returns mean: {returns.mean():.6f}, std: {returns.std():.6f}")
    
    if prediction_type == 'direction':
        # Direction statistics
        train_up_ratio = data['y_train'].mean()
        test_up_ratio = data['y_test'].mean()
        print(f"  Training: {train_up_ratio*100:.1f}% up days, {(1-train_up_ratio)*100:.1f}% down days")
        print(f"  Test: {test_up_ratio*100:.1f}% up days, {(1-test_up_ratio)*100:.1f}% down days")
        
        # Calculate class weights to handle class imbalance
        # Use conservative manual weights instead of 'balanced' to avoid over-correction
        classes = np.unique(data['y_train'])
        class_counts = np.bincount(data['y_train'])
        total_samples = len(data['y_train'])
        
        # Calculate balanced weights first
        balanced_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=data['y_train']
        )
        
        # Apply conservative scaling: use sqrt of balanced weights to avoid over-correction
        # This prevents extreme weights that cause the model to flip to predicting only minority class
        conservative_weights = np.sqrt(balanced_weights)
        # Normalize so that the average weight is 1.0
        conservative_weights = conservative_weights / conservative_weights.mean()
        
        class_weight_dict = {int(cls): weight for cls, weight in zip(classes, conservative_weights)}
        
        print(f"\n  Class Weights (conservative to handle imbalance):")
        print(f"    Class 0 (Down): {class_weight_dict[0]:.4f} (balanced would be {balanced_weights[0]:.4f})")
        print(f"    Class 1 (Up): {class_weight_dict[1]:.4f} (balanced would be {balanced_weights[1]:.4f})")
        print(f"    Using conservative sqrt scaling to avoid over-correction")
    else:
        class_weight_dict = None
    
    # Build model
    model = build_lstm_model(lookback, forecast_horizon, prediction_type=prediction_type)
    
    # Print model summary
    print(f"\nModel Architecture:")
    model.summary()
    
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print(f"\nTraining LSTM model (lookback={lookback}, epochs={epochs})...")
    fit_kwargs = {
        'epochs': epochs,
        'batch_size': batch_size,
        'validation_split': 0.2,
        'callbacks': [early_stopping, reduce_lr],
        'verbose': 1
    }
    
    # Add class_weight for binary classification to handle imbalance
    if prediction_type == 'direction' and class_weight_dict is not None:
        fit_kwargs['class_weight'] = class_weight_dict
        print(f"  Using class weights to handle class imbalance")
    
    history = model.fit(
        data['X_train'], data['y_train'],
        **fit_kwargs
    )
    
    # Print training progress and diagnostics
    print(f"\n{'='*60}")
    print("Training Diagnostics")
    print(f"{'='*60}")
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    initial_train_loss = history.history['loss'][0]
    initial_val_loss = history.history['val_loss'][0]
    
    loss_reduction = ((initial_train_loss - final_train_loss) / initial_train_loss * 100) if initial_train_loss > 0 else 0
    val_loss_reduction = ((initial_val_loss - final_val_loss) / initial_val_loss * 100) if initial_val_loss > 0 else 0
    
    print(f"  Initial training loss: {initial_train_loss:.6f}")
    print(f"  Final training loss: {final_train_loss:.6f}")
    print(f"  Training loss reduction: {loss_reduction:.2f}%")
    print(f"  Initial validation loss: {initial_val_loss:.6f}")
    print(f"  Final validation loss: {final_val_loss:.6f}")
    print(f"  Validation loss reduction: {val_loss_reduction:.2f}%")
    
    # Check for overfitting
    overfitting_ratio = (final_train_loss / final_val_loss) if final_val_loss > 0 else 0
    if overfitting_ratio < 0.8:
        print(f"  ⚠️  Warning: Possible overfitting detected!")
        print(f"     Train/Val loss ratio: {overfitting_ratio:.3f} (< 0.8 indicates overfitting)")
    elif overfitting_ratio > 1.2:
        print(f"  ✓ Good: Model is not overfitting")
        print(f"     Train/Val loss ratio: {overfitting_ratio:.3f}")
    else:
        print(f"  ✓ Acceptable: Train/Val loss ratio: {overfitting_ratio:.3f}")
    
    # Check if model is learning
    if loss_reduction < 5:
        print(f"  ⚠️  Warning: Model may not be learning well (loss reduction < 5%)")
    else:
        print(f"  ✓ Model is learning (loss reduction: {loss_reduction:.2f}%)")
    
    # Predict
    y_train_pred_proba = model.predict(data['X_train'], verbose=0).flatten()
    y_test_pred_proba = model.predict(data['X_test'], verbose=0).flatten()
    
    # Find optimal threshold using ROC curve (Youden's J statistic)
    optimal_threshold = 0.5
    if prediction_type == 'direction':
        try:
            # Use validation set to find optimal threshold
            # Split training data to get validation set for threshold optimization
            val_split_idx = int(len(data['X_train']) * 0.8)
            X_val_for_threshold = data['X_train'][val_split_idx:]
            y_val_for_threshold = data['y_train'][val_split_idx:]
            y_val_pred_proba = model.predict(X_val_for_threshold, verbose=0).flatten()
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_val_for_threshold, y_val_pred_proba)
            
            # Find optimal threshold using Youden's J statistic (TPR - FPR)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Clamp threshold to reasonable range [0.1, 0.9]
            optimal_threshold = np.clip(optimal_threshold, 0.1, 0.9)
            
            print(f"\n  Optimal Threshold (from ROC curve): {optimal_threshold:.4f} (default: 0.5)")
        except Exception as e:
            print(f"\n  Warning: Could not find optimal threshold, using default 0.5: {e}")
            optimal_threshold = 0.5
    
    # Convert probabilities to binary predictions using optimal threshold
    y_train_pred = (y_train_pred_proba >= optimal_threshold).astype(int)
    y_test_pred = (y_test_pred_proba >= optimal_threshold).astype(int)
    
    # Get actual labels
    y_train_actual = data['y_train']
    y_test_actual = data['y_test']
    
    # Calculate classification metrics
    train_accuracy = accuracy_score(y_train_actual, y_train_pred)
    test_accuracy = accuracy_score(y_test_actual, y_test_pred)
    train_precision = precision_score(y_train_actual, y_train_pred, zero_division=0)
    test_precision = precision_score(y_test_actual, y_test_pred, zero_division=0)
    train_recall = recall_score(y_train_actual, y_train_pred, zero_division=0)
    test_recall = recall_score(y_test_actual, y_test_pred, zero_division=0)
    train_f1 = f1_score(y_train_actual, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test_actual, y_test_pred, zero_division=0)
    
    # ROC AUC
    try:
        train_roc_auc = roc_auc_score(y_train_actual, y_train_pred_proba)
        test_roc_auc = roc_auc_score(y_test_actual, y_test_pred_proba)
    except:
        train_roc_auc = 0.0
        test_roc_auc = 0.0
    
    # Print prediction statistics
    print(f"\n{'='*60}")
    print("Classification Performance")
    print(f"{'='*60}")
    if prediction_type == 'direction':
        print(f"  Classification Threshold: {optimal_threshold:.4f}")
    print(f"  Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Train Precision: {train_precision:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Train Recall: {train_recall:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")
    print(f"  Train F1-Score: {train_f1:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    print(f"  Train ROC-AUC: {train_roc_auc:.4f}")
    print(f"  Test ROC-AUC: {test_roc_auc:.4f}")
    
    # Baseline accuracy (always predict majority class)
    baseline_accuracy = max(y_test_actual.mean(), 1 - y_test_actual.mean())
    print(f"\n  Baseline Accuracy (majority class): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    
    # Check if model beats baseline
    if test_accuracy > baseline_accuracy:
        improvement = (test_accuracy - baseline_accuracy) * 100
        print(f"  ✓ Model beats baseline by {improvement:.2f}%p")
    else:
        print(f"  ⚠️  Warning: Model does not beat baseline")
    
    # ROC AUC interpretation
    if test_roc_auc < 0.5:
        print(f"  ⚠️  Warning: ROC-AUC < 0.5 - Model performs worse than random")
    elif test_roc_auc < 0.6:
        print(f"  ⚠️  Low ROC-AUC ({test_roc_auc:.4f}) - Model has weak predictive power")
    elif test_roc_auc < 0.7:
        print(f"  ⚠️  Moderate ROC-AUC ({test_roc_auc:.4f}) - Model has some predictive power")
    elif test_roc_auc < 0.8:
        print(f"  ✓ Good ROC-AUC ({test_roc_auc:.4f}) - Model has decent predictive power")
    else:
        print(f"  ✓ Excellent ROC-AUC ({test_roc_auc:.4f}) - Model has strong predictive power")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_actual, y_test_pred)
    print(f"\n  Confusion Matrix (Test):")
    print(f"    True Negatives (Down→Down): {cm[0,0]}")
    print(f"    False Positives (Down→Up): {cm[0,1]}")
    print(f"    False Negatives (Up→Down): {cm[1,0]}")
    print(f"    True Positives (Up→Up): {cm[1,1]}")
    
    return {
        'model': model,
        'history': history,
        'y_train_actual': y_train_actual,
        'y_train_pred': y_train_pred,
        'y_train_pred_proba': y_train_pred_proba,
        'y_test_actual': y_test_actual,
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
        'optimal_threshold': optimal_threshold if prediction_type == 'direction' else None,
        'test_index': data['test_index'],
        'prediction_type': prediction_type
    }


def visualize_lstm_results(nasdaq_results, tqqq_results):
    """Visualize LSTM results (Direction Prediction)"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # NASDAQ predictions - Direction over time
    x_axis = range(len(nasdaq_results['y_test_actual']))
    axes[0, 0].plot(x_axis, nasdaq_results['y_test_actual'], 
                   label='Actual Direction', alpha=0.7, linewidth=1.5, color='blue', marker='o', markersize=3)
    axes[0, 0].plot(x_axis, nasdaq_results['y_test_pred'], 
                   label='Predicted Direction', alpha=0.7, linewidth=1.5, 
                   linestyle='--', color='red', marker='s', markersize=3)
    axes[0, 0].set_title(f'NASDAQ: Direction Prediction\nTest Accuracy: {nasdaq_results["test_accuracy"]:.4f}, ROC-AUC: {nasdaq_results["test_roc_auc"]:.4f}', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Direction (1=Up, 0=Down)', fontsize=12)
    axes[0, 0].set_xlabel('Time Step', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    # TQQQ predictions - Direction over time
    x_axis = range(len(tqqq_results['y_test_actual']))
    axes[0, 1].plot(x_axis, tqqq_results['y_test_actual'], 
                   label='Actual Direction', alpha=0.7, linewidth=1.5, color='orange', marker='o', markersize=3)
    axes[0, 1].plot(x_axis, tqqq_results['y_test_pred'], 
                   label='Predicted Direction', alpha=0.7, linewidth=1.5, 
                   linestyle='--', color='red', marker='s', markersize=3)
    axes[0, 1].set_title(f'TQQQ: Direction Prediction\nTest Accuracy: {tqqq_results["test_accuracy"]:.4f}, ROC-AUC: {tqqq_results["test_roc_auc"]:.4f}', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Direction (1=Up, 0=Down)', fontsize=12)
    axes[0, 1].set_xlabel('Time Step', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-0.1, 1.1)
    
    # Training history - Loss
    axes[1, 0].plot(nasdaq_results['history'].history['loss'], 
                    label='Train Loss', linewidth=2, color='blue')
    axes[1, 0].plot(nasdaq_results['history'].history['val_loss'], 
                   label='Validation Loss', linewidth=2, color='red', linestyle='--')
    axes[1, 0].set_title('NASDAQ: LSTM Training History', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss (Binary Crossentropy)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(tqqq_results['history'].history['loss'], 
                   label='Train Loss', linewidth=2, color='orange')
    axes[1, 1].plot(tqqq_results['history'].history['val_loss'], 
                   label='Validation Loss', linewidth=2, color='red', linestyle='--')
    axes[1, 1].set_title('TQQQ: LSTM Training History', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss (Binary Crossentropy)', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Prediction probabilities
    axes[2, 0].plot(x_axis, nasdaq_results['y_test_pred_proba'], 
                   label='Prediction Probability', alpha=0.7, linewidth=1.5, color='green')
    axes[2, 0].axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Threshold (0.5)')
    axes[2, 0].scatter(x_axis, nasdaq_results['y_test_actual'], 
                      alpha=0.5, color='blue', s=20, label='Actual', zorder=5)
    axes[2, 0].set_xlabel('Time Step', fontsize=12)
    axes[2, 0].set_ylabel('Probability', fontsize=12)
    axes[2, 0].set_title('NASDAQ: Prediction Probabilities', fontsize=14, fontweight='bold')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim(-0.05, 1.05)
    
    axes[2, 1].plot(x_axis, tqqq_results['y_test_pred_proba'], 
                   label='Prediction Probability', alpha=0.7, linewidth=1.5, color='green')
    axes[2, 1].axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Threshold (0.5)')
    axes[2, 1].scatter(x_axis, tqqq_results['y_test_actual'], 
                      alpha=0.5, color='orange', s=20, label='Actual', zorder=5)
    axes[2, 1].set_xlabel('Time Step', fontsize=12)
    axes[2, 1].set_ylabel('Probability', fontsize=12)
    axes[2, 1].set_title('TQQQ: Prediction Probabilities', fontsize=14, fontweight='bold')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print(f"\n{'='*60}")
    print("LSTM Model Performance (Direction Prediction)")
    print(f"{'='*60}")
    print(f"\nNASDAQ:")
    print(f"  Train Accuracy: {nasdaq_results['train_accuracy']:.4f} ({nasdaq_results['train_accuracy']*100:.2f}%)")
    print(f"  Test Accuracy: {nasdaq_results['test_accuracy']:.4f} ({nasdaq_results['test_accuracy']*100:.2f}%)")
    print(f"  Test Precision: {nasdaq_results['test_precision']:.4f}")
    print(f"  Test Recall: {nasdaq_results['test_recall']:.4f}")
    print(f"  Test F1-Score: {nasdaq_results['test_f1']:.4f}")
    print(f"  Test ROC-AUC: {nasdaq_results['test_roc_auc']:.4f}")
    
    print(f"\nTQQQ:")
    print(f"  Train Accuracy: {tqqq_results['train_accuracy']:.4f} ({tqqq_results['train_accuracy']*100:.2f}%)")
    print(f"  Test Accuracy: {tqqq_results['test_accuracy']:.4f} ({tqqq_results['test_accuracy']*100:.2f}%)")
    print(f"  Test Precision: {tqqq_results['test_precision']:.4f}")
    print(f"  Test Recall: {tqqq_results['test_recall']:.4f}")
    print(f"  Test F1-Score: {tqqq_results['test_f1']:.4f}")
    print(f"  Test ROC-AUC: {tqqq_results['test_roc_auc']:.4f}")


def main():
    """Main function"""
    print("="*60)
    print("Chapter 4: Deep Learning Models (LSTM)")
    print("Time Series Forecasting")
    print("="*60)
    
    if not HAS_TENSORFLOW:
        print("\nError: TensorFlow is not installed.")
        print("Install with: pip install tensorflow")
        return
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    nasdaq = data['nasdaq']
    tqqq = data['tqqq']
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Apply LSTM to NASDAQ (using log returns for stationarity)
    print("\n1. Training LSTM model on NASDAQ log returns (Direction Prediction)...")
    nasdaq_results = lstm_forecast(nasdaq['Log_Returns'].dropna(), lookback=60, epochs=50, prediction_type='direction')
    
    # Save NASDAQ model
    nasdaq_model_path = os.path.join(models_dir, 'lstm_nasdaq_direction.h5')
    nasdaq_results['model'].save(nasdaq_model_path)
    print(f"\n   Saved NASDAQ model to: {nasdaq_model_path}")
    
    # Save NASDAQ metadata
    nasdaq_metadata = {
        'optimal_threshold': nasdaq_results['optimal_threshold'],
        'lookback': 60,
        'prediction_type': 'direction',
        'test_accuracy': nasdaq_results['test_accuracy'],
        'test_roc_auc': nasdaq_results['test_roc_auc']
    }
    import json
    nasdaq_metadata_path = os.path.join(models_dir, 'lstm_nasdaq_metadata.json')
    with open(nasdaq_metadata_path, 'w') as f:
        json.dump(nasdaq_metadata, f, indent=2)
    print(f"   Saved NASDAQ metadata to: {nasdaq_metadata_path}")
    
    # Apply LSTM to TQQQ (using log returns for stationarity)
    print("\n2. Training LSTM model on TQQQ log returns (Direction Prediction)...")
    tqqq_results = lstm_forecast(tqqq['Log_Returns'].dropna(), lookback=60, epochs=50, prediction_type='direction')
    
    # Save TQQQ model
    tqqq_model_path = os.path.join(models_dir, 'lstm_tqqq_direction.h5')
    tqqq_results['model'].save(tqqq_model_path)
    print(f"\n   Saved TQQQ model to: {tqqq_model_path}")
    
    # Save TQQQ metadata
    tqqq_metadata = {
        'optimal_threshold': tqqq_results['optimal_threshold'],
        'lookback': 60,
        'prediction_type': 'direction',
        'test_accuracy': tqqq_results['test_accuracy'],
        'test_roc_auc': tqqq_results['test_roc_auc']
    }
    tqqq_metadata_path = os.path.join(models_dir, 'lstm_tqqq_metadata.json')
    with open(tqqq_metadata_path, 'w') as f:
        json.dump(tqqq_metadata, f, indent=2)
    print(f"   Saved TQQQ metadata to: {tqqq_metadata_path}")
    
    # Visualize
    visualize_lstm_results(nasdaq_results, tqqq_results)
    
    print("\nAnalysis complete!")
    print(f"\nModels saved to: {models_dir}")
    print("You can now use these models in backtest_lstm.py")


if __name__ == "__main__":
    main()

