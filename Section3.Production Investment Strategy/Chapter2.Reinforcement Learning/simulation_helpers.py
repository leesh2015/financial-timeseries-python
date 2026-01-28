import numpy as np

def get_alpha_value(model_fitted, target_index, history, method='weighted_mean'):
    """Extract Error Correction Term (alpha) from VECM."""
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
                        return float(np.sum(negative_alphas * weights))
                    elif method == 'sum_negative':
                        return float(np.sum(negative_alphas))
                    elif method == 'min_negative':
                        return float(np.min(negative_alphas))
                return float(np.mean(target_alphas))
        elif alpha.ndim == 1:
            if target_idx < len(alpha):
                return float(alpha[target_idx])
        return 0.0
    except Exception:
        return 0.0

def get_vecm_confidence(model_fitted, target_index, history, lower_bound, upper_bound, predicted_mean):
    """
    Calculate VECM model confidence.
    
    Parameters:
    -----------
    model_fitted : VECMResults
        The fitted VECM model.
    target_index : str
        The name of the target variable.
    history : pd.DataFrame
        The historical data used for fitting.
    lower_bound : float
        The lower bound of the forecast interval. Must be paired with an upper_bound from the same forecast steps.
    upper_bound : float
        The upper bound of the forecast interval. Must be paired with a lower_bound from the same forecast steps.
    predicted_mean : float
        The mean of the forecast.
    
    Returns:
    --------
    float: Confidence score (0~1 range, higher is better)
    """
    try:
        target_idx = history.columns.get_loc(target_index)
        
        interval_width = (upper_bound - lower_bound) / predicted_mean if predicted_mean > 0 else 1.0
        
        residuals = model_fitted.resid[:, target_idx]
        residual_std = np.std(residuals)
        normalized_residual_std = residual_std / predicted_mean if predicted_mean > 0 else 1.0
        
        alpha = model_fitted.alpha
        alpha_abs_mean = 0.0
        if alpha.ndim == 2:
            target_alpha = alpha[target_idx, :]
            negative_alphas = target_alpha[target_alpha < 0]
            if len(negative_alphas) > 0:
                alpha_abs_mean = np.mean(np.abs(negative_alphas))
        
        confidence_interval = max(0, 1 - min(interval_width, 1.0))
        confidence_residual = max(0, 1 - min(normalized_residual_std / 0.5, 1.0))
        confidence_alpha = min(alpha_abs_mean / 0.05, 1.0)
        
        confidence = (0.5 * confidence_interval + 0.3 * confidence_residual + 0.2 * confidence_alpha)
        confidence_expanded = 0.2 + (confidence * 0.7)
        
        return float(np.clip(confidence_expanded, 0.0, 1.0))
        
    except Exception:
        return 0.5

def calculate_adaptive_threshold(confidence_history, method='percentile', percentile=10, min_threshold=0.2, max_threshold=0.5):
    """Calculate adaptive threshold based on past confidence history."""
    if len(confidence_history) < 2:
        return (min_threshold + max_threshold) / 2
    
    conf_array = np.array(confidence_history)
    threshold = np.percentile(conf_array, percentile)
    return float(np.clip(threshold, min_threshold, max_threshold))

def normalize_confidence_to_fraction(confidence_current, confidence_history, **kwargs):
    """Normalize confidence value to a trading fraction."""
    absolute_threshold = kwargs.get('absolute_threshold')
    if absolute_threshold is None:
        absolute_threshold = calculate_adaptive_threshold(
            confidence_history, 
            method=kwargs.get('threshold_method', 'percentile'),
            percentile=kwargs.get('threshold_percentile', 10)
        )
    
    if confidence_current < absolute_threshold:
        return kwargs.get('min_fraction', 0.2)
    
    window_size = kwargs.get('window_size', 60)
    window_conf = confidence_history[-window_size:]
    
    min_conf, max_conf = np.min(window_conf), np.max(window_conf)
    
    if max_conf - min_conf < 1e-6:
        return (kwargs.get('min_fraction', 0.2) + kwargs.get('max_fraction', 0.8)) / 2
        
    normalized = (confidence_current - min_conf) / (max_conf - min_conf)
    fraction = kwargs.get('min_fraction', 0.2) + (kwargs.get('max_fraction', 0.8) - kwargs.get('min_fraction', 0.2)) * normalized
    return float(np.clip(fraction, kwargs.get('min_fraction', 0.2), kwargs.get('max_fraction', 0.8)))