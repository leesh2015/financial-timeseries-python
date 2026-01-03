"""
Market Regime Detector Module
Categorizes market states based on model prediction patterns and confidence.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import deque

# Weight mapping table per regime
REGIME_WEIGHTS = {
    'bull': {
        'vecm': 0.40,
        'kalman': 0.25,
        'prophet': 0.20,
        'wavelet': 0.10,
        'copula': 0.05
    },
    'bear': {
        'vecm': 0.30,
        'kalman': 0.20,
        'prophet': 0.15,
        'wavelet': 0.15,
        'copula': 0.20
    },
    'sideways': {
        'vecm': 0.35,
        'kalman': 0.30,
        'prophet': 0.15,
        'wavelet': 0.15,
        'copula': 0.05
    },
    'high_vol': {
        'vecm': 0.25,
        'kalman': 0.20,
        'prophet': 0.10,
        'wavelet': 0.20,
        'copula': 0.25
    }
}

class ModelBasedRegimeDetector:
    """
    Regime detection using model predictions and confidence levels.
    """
    
    def __init__(self, window: int = 60, hysteresis_threshold: float = 0.15):
        """
        Parameters:
        -----------
        window : int
            Analysis window size (default: 60 days)
        hysteresis_threshold : float
            Hysteresis filter threshold (default: 0.15)
        """
        self.window = window
        self.hysteresis_threshold = hysteresis_threshold
        
        self.current_regime = None
        self.regime_score_history = deque(maxlen=10)
        self.vecm_pred_history = deque(maxlen=window)
        self.volatility_history = deque(maxlen=252)
    
    def detect_regime(self, 
                      vecm_pred: float,
                      kalman_beta: Optional[float] = None,
                      garch_vol: Optional[float] = None,
                      model_confidences: Optional[Dict[str, float]] = None,
                      copula_risk: Optional[float] = None) -> str:
        """
        Main regime detection function.
        """
        self.vecm_pred_history.append(vecm_pred)
        
        if garch_vol is not None:
            self.volatility_history.append(garch_vol)
        
        if len(self.vecm_pred_history) < 10:
            if self.current_regime is None:
                return 'sideways'
            return self.current_regime
        
        # 1. Analyze trends
        vecm_trend = self._analyze_vecm_trend()
        beta_trend = self._analyze_beta_trend(kalman_beta) if kalman_beta is not None else 'neutral'
        vol_level = self._classify_volatility(garch_vol) if garch_vol is not None else 'medium'
        confidence_consistency = self._analyze_confidence_consistency(
            model_confidences
        ) if model_confidences else 'medium'
        
        # 2. Calculate scores
        regime_scores = self._calculate_regime_scores(
            vecm_trend, beta_trend, vol_level, confidence_consistency, copula_risk
        )
        
        # 3. Classify and filter
        regime = self._classify_regime_by_scores(regime_scores)
        regime = self._apply_hysteresis_filter(regime, regime_scores)
        
        return regime
    
    def _analyze_vecm_trend(self) -> str:
        recent_preds = list(self.vecm_pred_history)
        if len(recent_preds) == 0:
            return 'neutral'
        
        recent_preds = np.array(recent_preds)
        positive_ratio = np.mean(recent_preds > 0)
        avg_pred = np.mean(recent_preds)
        
        if np.abs(avg_pred) > 1e-6:
            cv = np.std(recent_preds) / np.abs(avg_pred)
            if cv > 0.005:
                return 'oscillating'
        
        if positive_ratio > 0.7 and avg_pred > 0.001:
            return 'bullish'
        elif positive_ratio < 0.3 and avg_pred < -0.001:
            return 'bearish'
        else:
            return 'neutral'
    
    def _analyze_beta_trend(self, kalman_beta: float) -> str:
        if kalman_beta > 1.0:
            return 'bullish'
        elif kalman_beta < 0.8:
            return 'bearish'
        else:
            return 'neutral'
    
    def _classify_volatility(self, garch_vol: float) -> str:
        if garch_vol is None or len(self.volatility_history) < 20:
            return 'medium'
        
        vol_array = np.array(list(self.volatility_history))
        vol_percentile = np.percentile(vol_array, [25, 75])
        
        if garch_vol > vol_percentile[1]:
            return 'high'
        elif garch_vol < vol_percentile[0]:
            return 'low'
        else:
            return 'medium'
    
    def _analyze_confidence_consistency(self, model_confidences: Dict[str, float]) -> str:
        confidences = list(model_confidences.values())
        conf_std = np.std(confidences)
        conf_mean = np.mean(confidences)
        
        if conf_std > 0.15:
            return 'inconsistent'
        elif conf_mean > 0.7:
            return 'high'
        elif conf_mean < 0.5:
            return 'low'
        else:
            return 'medium'
    
    def _calculate_regime_scores(self, 
                                vecm_trend: str,
                                beta_trend: str,
                                vol_level: str,
                                confidence_consistency: str,
                                copula_risk: Optional[float] = None) -> Dict[str, float]:
        scores = {'bull': 0.0, 'bear': 0.0, 'sideways': 0.0, 'high_vol': 0.0}
        
        if vecm_trend == 'bullish': scores['bull'] += 0.4
        if beta_trend == 'bullish': scores['bull'] += 0.3
        if vol_level == 'low': scores['bull'] += 0.2
        if confidence_consistency == 'high': scores['bull'] += 0.1
        
        if vecm_trend == 'bearish': scores['bear'] += 0.3
        if vol_level == 'high': scores['bear'] += 0.3
        if copula_risk is not None and copula_risk > 0.7: scores['bear'] += 0.4
        
        if vecm_trend in ['oscillating', 'neutral']: scores['sideways'] += 0.4
        if confidence_consistency == 'inconsistent': scores['sideways'] += 0.3
        if vol_level == 'medium': scores['sideways'] += 0.3
        
        if vol_level == 'high': scores['high_vol'] += 0.5
        if confidence_consistency == 'inconsistent': scores['high_vol'] += 0.3
        if copula_risk is not None and copula_risk > 0.6: scores['high_vol'] += 0.2
        
        return scores
    
    def _classify_regime_by_scores(self, regime_scores: Dict[str, float]) -> str:
        max_score = max(regime_scores.values())
        if max_score < 0.3:
            return 'sideways'
        for regime, score in regime_scores.items():
            if score == max_score:
                return regime
        return 'sideways'
    
    def _apply_hysteresis_filter(self, new_regime: str, regime_scores: Dict[str, float]) -> str:
        if self.current_regime is None:
            self.current_regime = new_regime
            self.regime_score_history.append(regime_scores.get(new_regime, 0.0))
            return new_regime
        
        current_score = regime_scores.get(self.current_regime, 0.0)
        new_score = regime_scores.get(new_regime, 0.0)
        score_diff = new_score - current_score
        
        if score_diff > self.hysteresis_threshold:
            self.current_regime = new_regime
        
        self.regime_score_history.append(regime_scores.get(self.current_regime, 0.0))
        return self.current_regime
    
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        return REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['sideways']).copy()

def create_regime_detector(window: int = 60, hysteresis_threshold: float = 0.15) -> ModelBasedRegimeDetector:
    return ModelBasedRegimeDetector(window=window, hysteresis_threshold=hysteresis_threshold)
