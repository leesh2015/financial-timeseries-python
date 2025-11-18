"""
VECM confidence-based position sizing utilities

This module provides functions for calculating VECM model confidence
and converting it to dynamic position sizing fractions.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def get_vecm_confidence(model_fitted, target_index: str, history: pd.DataFrame, 
                       lower_bound: float, upper_bound: float, predicted_mean: float) -> float:
    """
    Calculate VECM model confidence score.
    
    Confidence is measured by combining multiple indicators:
    1. Prediction interval width: (upper - lower) / predicted_mean (smaller = higher confidence)
    2. Residual standard deviation: smaller = higher confidence
    3. Alpha absolute value: larger = faster convergence = higher confidence
    
    Formula:
    confidence = 0.5 * confidence_interval + 0.3 * confidence_residual + 0.2 * confidence_alpha
    confidence_expanded = 0.2 + (confidence * 0.7)  # Map 0~1 to 0.2~0.9
    
    Parameters:
    -----------
    model_fitted : VECMResults
        Fitted VECM model
    target_index : str
        Target variable name
    history : pd.DataFrame
        Historical data
    lower_bound : float
        Prediction lower bound (same forecast horizon as upper_bound)
    upper_bound : float
        Prediction upper bound (same forecast horizon as lower_bound)
    predicted_mean : float
        Predicted mean value
        
    Returns:
    --------
    float: Confidence score (0~1 range, higher = more confident)
    """
    try:
        target_idx = history.columns.get_loc(target_index)
        
        # 1. Prediction interval width (smaller = higher confidence)
        if predicted_mean > 0:
            interval_width = (upper_bound - lower_bound) / predicted_mean
        else:
            interval_width = 1.0
        
        # 2. Residual standard deviation (smaller = higher confidence)
        residuals = model_fitted.resid
        if residuals.ndim == 2:
            target_residuals = residuals[:, target_idx]
        else:
            target_residuals = residuals
        
        residual_std = np.std(target_residuals)
        if predicted_mean > 0:
            normalized_residual_std = residual_std / predicted_mean
        else:
            normalized_residual_std = 1.0
        
        # 3. Alpha absolute value (larger = faster convergence = higher confidence)
        alpha = model_fitted.alpha
        if alpha.ndim == 2:
            target_alpha = alpha[target_idx, :]
            negative_alphas = target_alpha[target_alpha < 0]
            if len(negative_alphas) > 0:
                alpha_abs_mean = np.mean(np.abs(negative_alphas))
            else:
                alpha_abs_mean = 0.0
        else:
            if target_idx < len(alpha):
                alpha_val = alpha[target_idx]
                alpha_abs_mean = abs(alpha_val) if alpha_val < 0 else 0.0
            else:
                alpha_abs_mean = 0.0
        
        # Normalize each indicator to 0~1 range and combine
        confidence_interval = max(0, 1 - min(interval_width / 1.0, 1.0))
        confidence_residual = max(0, 1 - min(normalized_residual_std / 0.5, 1.0))
        confidence_alpha = min(alpha_abs_mean / 0.05, 1.0)
        
        # Weighted average (prediction interval is most important)
        confidence = (0.5 * confidence_interval + 
                     0.3 * confidence_residual + 
                     0.2 * confidence_alpha)
        
        # Expand confidence range (0~1 to 0.2~0.9 for larger variation)
        confidence_expanded = 0.2 + (confidence * 0.7)
        
        return float(np.clip(confidence_expanded, 0.0, 1.0))
        
    except Exception as e:
        if not hasattr(get_vecm_confidence, '_error_count'):
            get_vecm_confidence._error_count = 0
        if get_vecm_confidence._error_count < 3:
            print(f"Warning: Error calculating VECM confidence: {e}")
            get_vecm_confidence._error_count += 1
        return 0.5  # Default value


def calculate_adaptive_threshold(confidence_history: List[float], 
                                method: str = 'percentile', 
                                percentile: float = 10, 
                                min_threshold: float = 0.2, 
                                max_threshold: float = 0.5) -> float:
    """
    Calculate adaptive threshold based on confidence history.
    
    Methods:
    1. 'percentile': Use lower percentile (default: 10th percentile)
    2. 'min': Use minimum value
    3. 'mean_std': Use mean - standard deviation
    4. 'rolling_min': Use rolling window minimum
    
    Parameters:
    -----------
    confidence_history : List[float]
        Confidence history
    method : str
        Calculation method ('percentile', 'min', 'mean_std', 'rolling_min')
    percentile : float
        Percentile (0~100, default: 10 = lower 10%)
    min_threshold : float
        Minimum threshold (default: 0.2)
    max_threshold : float
        Maximum threshold (default: 0.5)
        
    Returns:
    --------
    float: Calculated adaptive threshold
    """
    if len(confidence_history) < 2:
        return (min_threshold + max_threshold) / 2
    
    conf_array = np.array(confidence_history)
    
    if method == 'percentile':
        threshold = np.percentile(conf_array, percentile)
    elif method == 'min':
        threshold = np.min(conf_array)
    elif method == 'mean_std':
        threshold = np.mean(conf_array) - np.std(conf_array)
    elif method == 'rolling_min':
        window = min(30, len(conf_array))
        threshold = np.min(conf_array[-window:])
    else:
        threshold = np.percentile(conf_array, percentile)
    
    return float(np.clip(threshold, min_threshold, max_threshold))


def normalize_confidence_to_fraction(confidence_current: float, 
                                     confidence_history: List[float],
                                     min_fraction: float = 0.3,
                                     max_fraction: float = 0.7,
                                     window_size: int = 60,
                                     method: str = 'zscore_sigmoid',
                                     absolute_threshold: Optional[float] = None,
                                     threshold_method: str = 'percentile',
                                     threshold_percentile: float = 10) -> float:
    """
    Normalize confidence value to position sizing fraction.
    
    Formula:
    - If confidence_current < absolute_threshold: return min_fraction
    - Otherwise: normalize based on method (zscore_sigmoid, minmax, percentile)
    - Result: min_fraction ~ max_fraction range
    
    Parameters:
    -----------
    confidence_current : float
        Current confidence value
    confidence_history : List[float]
        Confidence history (for rolling window)
    min_fraction : float
        Minimum fraction (default: 0.3)
    max_fraction : float
        Maximum fraction (default: 0.7)
    window_size : int
        Rolling window size (default: 60 days)
    method : str
        Normalization method ('zscore_sigmoid', 'minmax', 'percentile')
    absolute_threshold : Optional[float]
        Absolute confidence threshold
        - None: Dynamic calculation (use threshold_method)
        - float: Fixed value
    threshold_method : str
        Dynamic threshold calculation method (when absolute_threshold=None)
    threshold_percentile : float
        Percentile (0~100, default: 10)
        
    Returns:
    --------
    float: Normalized fraction value (min_fraction ~ max_fraction)
    """
    # Dynamic threshold calculation
    if absolute_threshold is None:
        absolute_threshold = calculate_adaptive_threshold(
            confidence_history, 
            method=threshold_method,
            percentile=threshold_percentile
        )
    
    # Check absolute confidence threshold
    if confidence_current < absolute_threshold:
        return min_fraction
    
    if len(confidence_history) < 2:
        return (min_fraction + max_fraction) / 2
    
    window_conf = confidence_history[-window_size:] if len(confidence_history) >= window_size else confidence_history
    
    if method == 'zscore_sigmoid':
        mean_conf = np.mean(window_conf)
        std_conf = np.std(window_conf) if len(window_conf) > 1 else 0.1
        
        if std_conf < 1e-6:
            # Fallback to minmax if std too small
            min_conf = np.min(window_conf)
            max_conf = np.max(window_conf)
            if max_conf - min_conf < 1e-6:
                return (min_fraction + max_fraction) / 2
            normalized = (confidence_current - min_conf) / (max_conf - min_conf)
            return float(min_fraction + (max_fraction - min_fraction) * normalized)
        else:
            # Z-score with increased sensitivity
            z_score = (confidence_current - mean_conf) / (std_conf * 0.5)
            z_score = np.clip(z_score, -2, 2)
            sigmoid_value = 1 / (1 + np.exp(-z_score))
            normalized = min_fraction + (max_fraction - min_fraction) * sigmoid_value
        
        return float(np.clip(normalized, min_fraction, max_fraction))
    
    elif method == 'minmax':
        min_conf = np.min(window_conf)
        max_conf = np.max(window_conf)
        
        if max_conf - min_conf < 1e-6:
            return (min_fraction + max_fraction) / 2
        
        normalized = (confidence_current - min_conf) / (max_conf - min_conf)
        fraction = min_fraction + (max_fraction - min_fraction) * normalized
        return float(np.clip(fraction, min_fraction, max_fraction))
    
    elif method == 'percentile':
        percentile = np.sum(np.array(window_conf) <= confidence_current) / len(window_conf)
        fraction = min_fraction + (max_fraction - min_fraction) * percentile
        return float(np.clip(fraction, min_fraction, max_fraction))
    
    else:
        return normalize_confidence_to_fraction(confidence_current, confidence_history, 
                                               min_fraction, max_fraction, 
                                               window_size, 'zscore_sigmoid')

