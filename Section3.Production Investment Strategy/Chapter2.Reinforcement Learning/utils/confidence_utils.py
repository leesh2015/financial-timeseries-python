"""
Confidence and position sizing utilities
"""

import numpy as np
from typing import List, Optional

def calculate_adaptive_threshold(confidence_history: List[float], 
                                 method: str = 'percentile', 
                                 percentile: float = 10, 
                                 min_threshold: float = 0.2, 
                                 max_threshold: float = 0.5) -> float:
    """
    Calculate adaptive threshold based on historical confidence.
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
    
    threshold = np.clip(threshold, min_threshold, max_threshold)
    return float(threshold)

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
    Normalize confidence value to a trading fraction.
    """
    if absolute_threshold is None:
        absolute_threshold = calculate_adaptive_threshold(
            confidence_history, 
            method=threshold_method,
            percentile=threshold_percentile
        )
    
    if confidence_current < absolute_threshold:
        return min_fraction
    
    if len(confidence_history) < 2:
        return (min_fraction + max_fraction) / 2
    
    window_conf = confidence_history[-window_size:] if len(confidence_history) >= window_size else confidence_history
    
    if method == 'zscore_sigmoid':
        mean_conf = np.mean(window_conf)
        std_conf = np.std(window_conf) if len(window_conf) > 1 else 0.1
        
        if std_conf < 1e-6:
            min_conf = np.min(window_conf)
            max_conf = np.max(window_conf)
            if max_conf - min_conf < 1e-6:
                return (min_fraction + max_fraction) / 2
            normalized = (confidence_current - min_conf) / (max_conf - min_conf)
            return float(min_fraction + (max_fraction - min_fraction) * normalized)
        else:
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
        return normalize_confidence_to_fraction(confidence_current, confidence_history, min_fraction, max_fraction, 
                                                window_size, 'zscore_sigmoid')
