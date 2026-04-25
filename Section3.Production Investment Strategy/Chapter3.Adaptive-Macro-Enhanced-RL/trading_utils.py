import numpy as np
import pandas as pd

def get_alpha_value(model_fitted, target_index, history, method='weighted_mean'):
    """Extracts the Error Correction Speed (alpha) from the VECM."""
    try:
        alpha = model_fitted.alpha
        target_idx = history.columns.get_loc(target_index)
        
        if alpha.ndim == 2:
            if target_idx < alpha.shape[0] and alpha.shape[1] > 0:
                target_alphas = alpha[target_idx, :]
                negative_alphas = target_alphas[target_alphas < 0]
                
                if len(negative_alphas) > 0:
                    if method == 'weighted_mean':
                        abs_negative = np.abs(negative_alphas)
                        weights = abs_negative / np.sum(abs_negative)
                        weighted_alpha = np.sum(negative_alphas * weights)
                        return float(weighted_alpha)
                    elif method == 'sum_negative':
                        return float(np.sum(negative_alphas))
                    elif method == 'min_negative':
                        return float(np.min(negative_alphas))
                    else:
                        abs_negative = np.abs(negative_alphas)
                        weights = abs_negative / np.sum(abs_negative)
                        return float(np.sum(negative_alphas * weights))
                else:
                    return float(np.mean(target_alphas))
            else:
                return 0.0
        elif alpha.ndim == 1:
            if target_idx < len(alpha):
                return float(alpha[target_idx])
            else:
                return 0.0
        else:
            return 0.0
    except Exception as e:
        return 0.0

def get_vecm_confidence(model_fitted, target_index, history, lower_bound, upper_bound, predicted_mean):
    """Calculates the confidence level of the VECM model."""
    try:
        target_idx = history.columns.get_loc(target_index)
        
        # 1. Interval Width
        if predicted_mean > 0:
            interval_width = (upper_bound - lower_bound) / predicted_mean
        else:
            interval_width = 1.0
        
        # 2. Residual Standard Deviation
        residuals = model_fitted.resid
        target_residuals = residuals[:, target_idx] if residuals.ndim == 2 else residuals
        residual_std = np.std(target_residuals)
        normalized_residual_std = residual_std / predicted_mean if predicted_mean > 0 else 1.0
        
        # 3. Absolute Alpha
        alpha = model_fitted.alpha
        if alpha.ndim == 2:
            target_alpha = alpha[target_idx, :]
            negative_alphas = target_alpha[target_alpha < 0]
            alpha_abs_mean = np.mean(np.abs(negative_alphas)) if len(negative_alphas) > 0 else 0.0
        else:
            alpha_val = alpha[target_idx] if target_idx < len(alpha) else 0.0
            alpha_abs_mean = abs(alpha_val) if alpha_val < 0 else 0.0
        
        confidence_interval = max(0, 1 - min(interval_width / 1.0, 1.0))
        confidence_residual = max(0, 1 - min(normalized_residual_std / 0.5, 1.0))
        confidence_alpha = min(alpha_abs_mean / 0.05, 1.0)
        
        confidence = (0.4 * confidence_interval + 0.3 * confidence_residual + 0.3 * confidence_alpha)
        confidence_expanded = 0.2 + (confidence * 0.7)
        
        return float(np.clip(confidence_expanded, 0.0, 1.0))
    except:
        return 0.5

def calculate_adaptive_threshold(confidence_history, method='percentile', percentile=10, min_threshold=0.2, max_threshold=0.5):
    """Calculates adaptive threshold based on historical confidence."""
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

def normalize_confidence_to_fraction(confidence_current, confidence_history, min_fraction=0.3, max_fraction=0.7, 
                                     window_size=60, method='zscore_sigmoid', absolute_threshold=None,
                                     threshold_method='percentile', threshold_percentile=10):
    """Normalizes confidence values into position fractions."""
    if absolute_threshold is None:
        absolute_threshold = calculate_adaptive_threshold(confidence_history, method=threshold_method, percentile=threshold_percentile)
    
    if confidence_current < absolute_threshold:
        return min_fraction
    
    if len(confidence_history) < 2:
        return (min_fraction + max_fraction) / 2
    
    window_conf = confidence_history[-window_size:] if len(confidence_history) >= window_size else confidence_history
    
    if method == 'zscore_sigmoid':
        mean_conf = np.mean(window_conf)
        std_conf = np.std(window_conf) if len(window_conf) > 1 else 0.1
        if std_conf < 1e-6:
            min_c, max_c = np.min(window_conf), np.max(window_conf)
            if max_c - min_c < 1e-6: return (min_fraction + max_fraction) / 2
            return float(min_fraction + (max_fraction - min_fraction) * (confidence_current - min_c) / (max_c - min_c))
        
        z_score = np.clip((confidence_current - mean_conf) / (std_conf * 0.5), -2, 2)
        sigmoid_value = 1 / (1 + np.exp(-z_score))
        return float(np.clip(min_fraction + (max_fraction - min_fraction) * sigmoid_value, min_fraction, max_fraction))
    
    elif method == 'minmax':
        min_c, max_c = np.min(window_conf), np.max(window_conf)
        if max_c - min_c < 1e-6: return (min_fraction + max_fraction) / 2
        return float(np.clip(min_fraction + (max_fraction - min_fraction) * (confidence_current - min_c) / (max_c - min_c), min_fraction, max_fraction))
    
    return min_fraction
