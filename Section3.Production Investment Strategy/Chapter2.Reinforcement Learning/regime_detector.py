"""
Market Regime Detector Module

A system for detecting market regimes based on model prediction patterns.
It classifies the market state using predictions and confidence levels from existing models.
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
        'copula': 0.20  # Strengthen risk management
    },
    'sideways': {
        'vecm': 0.35,
        'kalman': 0.30,  # Strengthen mean reversion
        'prophet': 0.15,
        'wavelet': 0.15,
        'copula': 0.05
    },
    'high_vol': {
        'vecm': 0.25,
        'kalman': 0.20,
        'prophet': 0.10,
        'wavelet': 0.20,  # Strengthen noise filtering
        'copula': 0.25   # Prioritize risk management
    }
}


class ModelBasedRegimeDetector:
    """
    Regime detection using predictions and confidence from existing models.
    Based purely on statistical/mathematical models without technical indicators.
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
        
        # State storage for the hysteresis filter
        self.current_regime = None
        self.regime_score_history = deque(maxlen=10)  # Store scores for the last 10 days
        
        # VECM prediction history (for regime detection)
        self.vecm_pred_history = deque(maxlen=window)
        
        # Volatility history (for percentile calculation)
        self.volatility_history = deque(maxlen=252)  # 1 year of data
    
    def detect_regime(self, 
                     vecm_pred: float,
                     kalman_beta: Optional[float] = None,
                     garch_vol: Optional[float] = None,
                     model_confidences: Optional[Dict[str, float]] = None,
                     copula_risk: Optional[float] = None) -> str:
        """
        Main regime detection function.
        
        Parameters:
        -----------
        vecm_pred : float
            VECM prediction value
        kalman_beta : float, optional
            Kalman Filter tracking beta
        garch_vol : float, optional
            EGARCH volatility
        model_confidences : dict, optional
            Dictionary of each model's confidence
        copula_risk : float, optional
            Copula risk score (0~1)
        
        Returns:
        --------
        str: Regime ('bull', 'bear', 'sideways', 'high_vol')
        """
        # Update VECM prediction history
        self.vecm_pred_history.append(vecm_pred)
        
        # Update volatility history
        if garch_vol is not None:
            self.volatility_history.append(garch_vol)
        
        # Return default value if data is insufficient
        if len(self.vecm_pred_history) < 10:
            if self.current_regime is None:
                return 'sideways'  # Default value
            return self.current_regime
        
        # 1. Analyze VECM prediction patterns
        vecm_trend = self._analyze_vecm_trend()
        
        # 2. Analyze Kalman beta patterns
        beta_trend = self._analyze_beta_trend(kalman_beta) if kalman_beta is not None else 'neutral'
        
        # 3. Classify volatility level
        vol_level = self._classify_volatility(garch_vol) if garch_vol is not None else 'medium'
        
        # 4. Analyze model confidence consistency
        confidence_consistency = self._analyze_confidence_consistency(
            model_confidences
        ) if model_confidences else 'medium'
        
        # 5. Calculate regime scores
        regime_scores = self._calculate_regime_scores(
            vecm_trend, beta_trend, vol_level, confidence_consistency, copula_risk
        )
        
        # 6. Classify regime
        regime = self._classify_regime_by_scores(regime_scores)
        
        # 7. Apply hysteresis filter (for stabilization)
        regime = self._apply_hysteresis_filter(regime, regime_scores)
        
        return regime
    
    def _analyze_vecm_trend(self) -> str:
        """Analyze the directionality of VECM predictions."""
        if len(self.vecm_pred_history) < self.window:
            recent_preds = list(self.vecm_pred_history)
        else:
            recent_preds = list(self.vecm_pred_history)[-self.window:]
        
        if len(recent_preds) == 0:
            return 'neutral'
        
        recent_preds = np.array(recent_preds)
        
        # Calculate the ratio of positive predictions
        positive_ratio = np.mean(recent_preds > 0)
        avg_pred = np.mean(recent_preds)
        
        # Detect oscillation (Coefficient of Variation)
        if np.abs(avg_pred) > 1e-6:
            cv = np.std(recent_preds) / np.abs(avg_pred)
            if cv > 5.0:  # [MODIFIED] Relaxed oscillation criteria (0.005 was too sensitive, causing most to be sideways)
                return 'oscillating'
        
        # Determine directionality
        # [MODIFIED] Readjusted bull market criteria (strengthened after being too sensitive)
        # Acknowledge bull market only if probability > 65% and average > 0.05%
        if positive_ratio > 0.65 and avg_pred > 0.0005:
            return 'bullish'
        elif positive_ratio < 0.35 and avg_pred < -0.0005:
            return 'bearish'
        else:
            return 'neutral'
    
    def _analyze_beta_trend(self, kalman_beta: float) -> str:
        """Analyze Kalman beta tracking patterns."""
        if kalman_beta is None:
            return 'neutral'
        
        if kalman_beta > 1.0:
            return 'bullish'
        elif kalman_beta < 0.8:
            return 'bearish'
        else:
            return 'neutral'
    
    def _classify_volatility(self, garch_vol: float) -> str:
        """Classify EGARCH volatility level."""
        if garch_vol is None or len(self.volatility_history) < 20:
            return 'medium'
        
        vol_array = np.array(list(self.volatility_history))
        vol_percentile = np.percentile(vol_array, [25, 75])
        
        if garch_vol > vol_percentile[1]:  # Upper 75th percentile
            return 'high'
        elif garch_vol < vol_percentile[0]:  # Lower 25th percentile
            return 'low'
        else:
            return 'medium'
    
    def _analyze_confidence_consistency(self, model_confidences: Dict[str, float]) -> str:
        """Analyze model confidence consistency."""
        if not model_confidences or len(model_confidences) < 2:
            return 'medium'
        
        confidences = list(model_confidences.values())
        conf_std = np.std(confidences)
        conf_mean = np.mean(confidences)
        
        # High confidence variance indicates inconsistency (sideways/high-vol)
        if conf_std > 0.15:  # Threshold
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
        """Calculate scores for each regime."""
        scores = {
            'bull': 0.0,
            'bear': 0.0,
            'sideways': 0.0,
            'high_vol': 0.0
        }
        
        # Bull market score
        if vecm_trend == 'bullish':
            scores['bull'] += 0.4  # [MODIFIED] Normalized VECM weight (0.5 -> 0.4)
        if beta_trend == 'bullish':
            scores['bull'] += 0.3
        
        if vol_level == 'low':
            scores['bull'] += 0.2
        if confidence_consistency == 'high':
            scores['bull'] += 0.1
        
        # Bear market score
        if vecm_trend == 'bearish':
            # [MODIFIED] Increased VECM trend score to 0.5 to match bull for balance
            scores['bear'] += 0.4  # [MODIFIED] Normalized VECM weight (0.5 -> 0.4)
        if vol_level == 'high':
            scores['bear'] += 0.3
        if copula_risk is not None and copula_risk > 0.7:
            scores['bear'] += 0.4
        
        # Sideways market score
        if vecm_trend == 'oscillating' or vecm_trend == 'neutral':
            scores['sideways'] += 0.4
        if confidence_consistency == 'inconsistent':
            scores['sideways'] += 0.3
        if vol_level == 'medium':
            scores['sideways'] += 0.3
        
        # High volatility score
        if vol_level == 'high':
            scores['high_vol'] += 0.5
        if confidence_consistency == 'inconsistent':
            scores['high_vol'] += 0.3
        if copula_risk is not None and copula_risk > 0.6:
            scores['high_vol'] += 0.2
        
        return scores
    
    def _classify_regime_by_scores(self, regime_scores: Dict[str, float]) -> str:
        """Classify regime based on scores."""
        max_score = max(regime_scores.values())
        if max_score < 0.3:  # If score is too low, classify as sideways
            return 'sideways'
        
        # Return the regime with the highest score
        for regime, score in regime_scores.items():
            if score == max_score:
                return regime
        return 'sideways'  # Default value
    
    def _apply_hysteresis_filter(self, new_regime: str, regime_scores: Dict[str, float]) -> str:
        """Apply hysteresis filter (to stabilize regime transitions)."""
        if self.current_regime is None:
            self.current_regime = new_regime
            self.regime_score_history.append(regime_scores.get(new_regime, 0.0))
            return new_regime
        
        # Score of the current regime
        current_score = regime_scores.get(self.current_regime, 0.0)
        new_score = regime_scores.get(new_regime, 0.0)
        
        # Switch if the score difference is greater than the threshold
        score_diff = new_score - current_score
        
        if score_diff > self.hysteresis_threshold:
            # Switch to the new regime
            self.current_regime = new_regime
        elif score_diff < -self.hysteresis_threshold:
            # Maintain current regime (score dropped significantly but not enough to switch yet)
            pass
        # else: maintain current regime if score difference is small
        
        self.regime_score_history.append(regime_scores.get(self.current_regime, 0.0))
        return self.current_regime
    
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Return weights for each regime."""
        return REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['sideways']).copy()
    
    def adjust_weights_by_regime(self, 
                                 base_weights: Dict[str, float],
                                 regime: str,
                                 blend_factor: float = 0.5) -> Dict[str, float]:
        """
        Adjust weights by regime.
        
        Parameters:
        -----------
        base_weights : dict
            Base weights (e.g., from Bayesian optimization)
        regime : str
            Current regime
        blend_factor : float
            Blending factor for regime weights and base weights (0~1)
            0.5 = 50% regime weight + 50% base weight
        
        Returns:
        --------
        dict: Adjusted weights
        """
        regime_weights = self.get_regime_weights(regime)
        
        # Blend the two sets of weights
        adjusted_weights = {}
        all_models = set(base_weights.keys()) | set(regime_weights.keys())
        
        for model in all_models:
            base_w = base_weights.get(model, 0.0)
            regime_w = regime_weights.get(model, 0.0)
            adjusted_weights[model] = (1 - blend_factor) * base_w + blend_factor * regime_w
        
        # Normalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        return adjusted_weights


def create_regime_detector(window: int = 60, hysteresis_threshold: float = 0.15) -> ModelBasedRegimeDetector:
    """Helper function to create a regime detector."""
    return ModelBasedRegimeDetector(window=window, hysteresis_threshold=hysteresis_threshold)
