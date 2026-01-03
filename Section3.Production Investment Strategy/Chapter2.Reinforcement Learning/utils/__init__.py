"""
Utility modules for Chapter 2: Reinforcement Learning
"""

from .vecm_utils import fit_vecm_model, get_alpha_value, get_vecm_confidence
from .garch_utils import find_garch
from .confidence_utils import calculate_adaptive_threshold, normalize_confidence_to_fraction
from .performance_utils import max_drawdown, max_loss
from .regime_utils import ModelBasedRegimeDetector, create_regime_detector
from .rl_utils import VECMRLAgent, SimpleRLPolicy

__all__ = [
    'fit_vecm_model',
    'get_alpha_value',
    'get_vecm_confidence',
    'find_garch',
    'calculate_adaptive_threshold',
    'normalize_confidence_to_fraction',
    'max_drawdown',
    'max_loss',
    'ModelBasedRegimeDetector',
    'create_regime_detector',
    'VECMRLAgent',
    'SimpleRLPolicy'
]
