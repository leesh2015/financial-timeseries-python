"""
Phase 7: Reinforcement Learning Agent
Simple Policy-based RL Agent.
"""

import numpy as np
from typing import Optional, Tuple


class SimpleRLPolicy:
    """
    Simple Policy (VECM confidence-based).
    Decisions are based on VECM confidence.
    """
    
    def __init__(self,
                 buy_confidence_threshold=0.7,
                 sell_confidence_threshold=0.5,
                 max_position_ratio=0.5,
                 min_position_ratio_for_sell=0.3,
                 max_position_size=0.8):
        """
        Parameters:
        -----------
        buy_confidence_threshold : float
            Buy confidence threshold (default: 0.7).
        sell_confidence_threshold : float
            Sell confidence threshold (default: 0.5).
        max_position_ratio : float
            Maximum position ratio (default: 0.5).
        min_position_ratio_for_sell : float
            Minimum position ratio for selling (default: 0.3).
        max_position_size : float
            Maximum position size (default: 0.8).
        """
        self.buy_confidence_threshold = buy_confidence_threshold
        self.sell_confidence_threshold = sell_confidence_threshold
        self.max_position_ratio = max_position_ratio
        self.min_position_ratio_for_sell = min_position_ratio_for_sell
        self.max_position_size = max_position_size
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simple policy: VECM confidence-based.
        
        observation: [vecm_pred, vecm_confidence, garch_vol, regime, position_ratio, capital_ratio]
        """
        vecm_confidence = observation[1]  # Confidence
        position_ratio = observation[4]  # Current position ratio
        
        # Optimized buy condition
        if vecm_confidence > self.buy_confidence_threshold and position_ratio < self.max_position_ratio:
            # Buy signal
            position_size = min(self.max_position_size, vecm_confidence)
            signal = 1.0
        # Optimized sell condition
        elif vecm_confidence < self.sell_confidence_threshold and position_ratio > self.min_position_ratio_for_sell:
            # Sell signal
            position_size = position_ratio
            signal = -1.0
        else:
            # Hold
            position_size = position_ratio
            signal = 0.0
        
        action = np.array([position_size, signal], dtype=np.float32)
        return action, None


class VECMRLAgent:
    """
    VECM-based RL Trading Agent (using Simple Policy only).
    """
    
    def __init__(self, 
                 simple_policy_params: dict = None):
        """
        Parameters:
        -----------
        simple_policy_params : dict, optional
            Simple Policy parameters.
        """
        # Simple Policy parameters (optimizable)
        if simple_policy_params is None:
            self.simple_policy_params = {
                'buy_confidence_threshold': 0.7,
                'sell_confidence_threshold': 0.5,
                'max_position_ratio': 0.5,
                'min_position_ratio_for_sell': 0.3,
                'max_position_size': 0.8
            }
        else:
            self.simple_policy_params = simple_policy_params
        
        self.model = SimpleRLPolicy(**self.simple_policy_params)
        self.is_trained = True
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current state observation.
        deterministic : bool
            Whether the prediction is deterministic (Simple Policy is always deterministic).
        
        Returns:
        --------
        action : np.ndarray
            [position_size, signal]
        """
        action, _ = self.model.predict(observation, deterministic)
        return action
