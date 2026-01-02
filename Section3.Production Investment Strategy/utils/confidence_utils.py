"""
Confidence and threshold calculation utilities
"""

import numpy as np
from typing import List, Union, Optional

def calculate_adaptive_threshold(confidence_history: List[float], 
                               method: str = 'percentile', 
                               percentile: float = 10, 
                               min_threshold: float = 0.2, 
                               max_threshold: float = 0.5) -> float:
    """
    Calculate adaptive threshold based on historical confidence.
    
    Dynamically calculate threshold through various methods:
    1. 'percentile': Use lower percentile (default: 10th percentile)
    2. 'min': Use minimum value
    3. 'mean_std': Use mean - standard deviation
    4. 'rolling_min': Use rolling window minimum
    
    Parameters:
    -----------
    confidence_history : list
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
    float
        Calculated adaptive threshold
    """
    if len(confidence_history) < 2:
        return (min_threshold + max_threshold) / 2
    
    conf_array = np.array(confidence_history)
    
    if method == 'percentile':
        # Use lower percentile (e.g., 10th percentile = lower 10%)
        threshold = np.percentile(conf_array, percentile)
    elif method == 'min':
        # Use minimum value
        threshold = np.min(conf_array)
    elif method == 'mean_std':
        # Use mean - standard deviation (lower 1 sigma)
        threshold = np.mean(conf_array) - np.std(conf_array)
    elif method == 'rolling_min':
        # Use minimum of recent 30 days
        window = min(30, len(conf_array))
        threshold = np.min(conf_array[-window:])
    else:
        # Default: percentile
        threshold = np.percentile(conf_array, percentile)
    
    # Limit to min/max threshold
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
    Normalize confidence value to fraction.
    
    [Modified] Added absolute confidence threshold: Solves problem of using only relative ranking
    - If absolute confidence is too low, use minimum fraction
    - Proper risk management even when overall market is uncertain
    - [Improved] Support dynamic threshold calculation: Adaptive threshold based on historical data
    
    Parameters:
    -----------
    confidence_current : float
        Current confidence value
    confidence_history : list
        Confidence history (for rolling window)
    min_fraction : float
        Minimum fraction (default: 0.3)
    max_fraction : float
        Maximum fraction (default: 0.7)
    window_size : int
        Rolling window size (default: 60 days)
    method : str
        Normalization method
        - 'zscore_sigmoid': Z-score normalization then sigmoid transformation (default, smooth change)
        - 'minmax': Min-Max scaling
        - 'percentile': Percentile-based transformation
    absolute_threshold : float or None
        Absolute confidence threshold
        - None: Dynamic calculation (use threshold_method)
        - float: Use fixed value
    threshold_method : str
        Dynamic threshold calculation method (when absolute_threshold=None)
        - 'percentile': Lower percentile (default: 10th percentile)
        - 'min': Minimum value
        - 'mean_std': Mean - standard deviation
        - 'rolling_min': Minimum of recent 30 days
    threshold_percentile : float
        Percentile (0-100, default: 10)
    
    Returns:
    --------
    float
        Normalized fraction value (min_fraction ~ max_fraction)
    """
    # [Improvement] Dynamic threshold calculation
    if absolute_threshold is None:
        # Calculate dynamically based on historical data
        absolute_threshold = calculate_adaptive_threshold(
            confidence_history, 
            method=threshold_method,
            percentile=threshold_percentile
        )
    
    # [Modified] Absolute confidence threshold check
    if confidence_current < absolute_threshold:
        return min_fraction
    
    if len(confidence_history) < 2:
        return (min_fraction + max_fraction) / 2
    
    window_conf = confidence_history[-window_size:] if len(confidence_history) >= window_size else confidence_history
    
    if method == 'zscore_sigmoid':
        mean_conf = np.mean(window_conf)
        std_conf = np.std(window_conf) if len(window_conf) > 1 else 0.1
        
        if std_conf < 1e-6:
            # If standard deviation is too small, fallback to minmax method
            min_conf = np.min(window_conf)
            max_conf = np.max(window_conf)
            if max_conf - min_conf < 1e-6:
                return (min_fraction + max_fraction) / 2
            normalized = (confidence_current - min_conf) / (max_conf - min_conf)
            return float(min_fraction + (max_fraction - min_fraction) * normalized)
        else:
            # Adjust Z-score more sensitively (increase scale factor)
            z_score = (confidence_current - mean_conf) / (std_conf * 0.5)  # More sensitive
            z_score = np.clip(z_score, -2, 2)  # Reduce range for more extreme response
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
