"""
Phase 7: Reinforcement Learning Agent
Simple Policy-based RL Agent for Production Portfolio Management
"""

import numpy as np
from typing import Optional, Tuple


class SimpleRLPolicy:
    """
    Rule-based Policy refined with Bayesian Optimization.
    Integrates VECM Confidence with Macro-economic and Cointegration risk factors.
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
            Threshold for triggering a buy signal (Default: 0.7)
        sell_confidence_threshold : float
            Threshold for triggering a sell signal (Default: 0.5)
        max_position_ratio : float
            Maximum target position ratio (Default: 0.5)
        min_position_ratio_for_sell : float
            Minimum current ratio required to perform a sell action (Default: 0.3)
        max_position_size : float
            Maximum size per individual order (Default: 0.8)
        """
        self.buy_confidence_threshold = buy_confidence_threshold
        self.sell_confidence_threshold = sell_confidence_threshold
        self.max_position_ratio = max_position_ratio
        self.min_position_ratio_for_sell = min_position_ratio_for_sell
        self.max_position_size = max_position_size
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Logic: VECM Confidence + Asset-Specific Soft Penalties (Bayesian Optimized)
        
        Observation Space:
        [0:vecm_pred, 1:vecm_confidence, 2:garch_vol, 3:regime, 4:position_ratio, 
         5:capital_ratio, 6:cl_momentum, 7:zb_momentum, 8:ect_alpha]
        """
        vecm_confidence = observation[1] 
        position_ratio = observation[4]  
        
        # --- Bayesian Optimized Risk Adjustments (Soft Penalties) ---
        macro_penalty = 0.0
        ect_bonus = 0.0
        
        if len(observation) >= 8:
            cl_mom = observation[6] # Oil Momentum
            zb_mom = observation[7] # Treasury Momentum
            
            # Penalties for inflationary (Oil up) and rate-sensitive (ZB down) shocks
            cl_impact = max(0.0, float(cl_mom)) * 5.6399
            zb_impact = max(0.0, -float(zb_mom)) * 1.2647
            macro_penalty = cl_impact + zb_impact
            
            # Detect structural stagflation regimes
            if cl_mom > 0.0733 and zb_mom < -0.0548:
                macro_penalty = max(macro_penalty, 0.5632)
                
            # Error Correction Term (ECT) Alpha Integration
            if len(observation) >= 9:
                ect_alpha = observation[8]
                if ect_alpha > 0:
                    # Divergence Penalty: Price is moving away from equilibrium
                    macro_penalty += float(ect_alpha) * 4.5832
                else:
                    # Convergence Bonus: Price is reverting to equilibrium
                    ect_bonus = -float(ect_alpha) * 1.3929
            
            # Cap the maximum scale of soft penalties to prevent total suppression
            macro_penalty = min(macro_penalty, 0.3957)
            
        # Apply Risk Adjustments to VECM confidence
        # Final Confidence = (Raw Confidence * (1 - Penalty)) + Convergence Bonus
        final_confidence = (vecm_confidence * (1.0 - macro_penalty)) + ect_bonus
        final_confidence = np.clip(final_confidence, 0.0, 1.0)
        
        # --- Execution Logic ---
        # Buy Signal Entry
        if final_confidence > self.buy_confidence_threshold and position_ratio < self.max_position_ratio:
            position_size = min(self.max_position_size, final_confidence)
            signal = 1.0
        # Sell Signal Exit
        elif vecm_confidence < self.sell_confidence_threshold and position_ratio > self.min_position_ratio_for_sell:
            position_size = position_ratio
            signal = -1.0
        else:
            # Hold Position
            position_size = position_ratio
            signal = 0.0
        
        action = np.array([position_size, signal], dtype=np.float32)
        return action, None


class VECMRLAgent:
    """
    VECM-based RL Trading Agent wrapper for integration.
    """
    
    def __init__(self, 
                 simple_policy_params: dict = None):
        """
        Parameters:
        -----------
        simple_policy_params : dict, optional
            Parameters to override the default SimpleRLPolicy settings.
        """
        # Default optimized parameters
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
        Predict Action based on observation.
        """
        action, _ = self.model.predict(observation, deterministic)
        return action
