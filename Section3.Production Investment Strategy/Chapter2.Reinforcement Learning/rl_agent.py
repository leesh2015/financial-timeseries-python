"""
Phase 7: 강화학습 에이전트
Simple Policy 기반 RL 에이전트
"""

import numpy as np
from typing import Optional, Tuple


class SimpleRLPolicy:
    """
    간단한 정책 (VECM 신뢰도 기반)
    VECM 신뢰도 기반 결정
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
            매수 신뢰도 임계값 (기본값: 0.7)
        sell_confidence_threshold : float
            매도 신뢰도 임계값 (기본값: 0.5)
        max_position_ratio : float
            최대 포지션 비율 (기본값: 0.5)
        min_position_ratio_for_sell : float
            매도 최소 포지션 비율 (기본값: 0.3)
        max_position_size : float
            최대 포지션 크기 (기본값: 0.8)
        """
        self.buy_confidence_threshold = buy_confidence_threshold
        self.sell_confidence_threshold = sell_confidence_threshold
        self.max_position_ratio = max_position_ratio
        self.min_position_ratio_for_sell = min_position_ratio_for_sell
        self.max_position_size = max_position_size
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        간단한 정책: VECM 신뢰도 기반 + 매크로 모멘텀 하드 필터(False Positive 방어)
        
        observation: [vecm_pred, vecm_confidence, garch_vol, regime, position_ratio, capital_ratio, cl_mom, zb_mom]
        """
        vecm_confidence = observation[1]  # 신뢰도
        position_ratio = observation[4]  # 현재 포지션 비율
        
        # [추가] 매크로 지표 파싱 (구버전 호환성을 위해 배열 길이 체크)
        if len(observation) >= 8:
            cl_momentum = observation[6]
            zb_momentum = observation[7]
            
            # [거짓 양성 방어 필터] 
            # 원유가 20일 기준 5% 이상 폭등했다면(인플레이션 발작), VECM 신뢰도를 강제로 절반으로 깎아 매수를 억제한다.
            if cl_momentum > 0.05:
                # 국채 하락(금리 인상)까지 겹치면 더 페널티
                if zb_momentum < -0.01:
                    vecm_confidence *= 0.3
                else:
                    vecm_confidence *= 0.5
        
        # 최적화된 매수 조건
        if vecm_confidence > self.buy_confidence_threshold and position_ratio < self.max_position_ratio:
            # 매수 신호
            position_size = min(self.max_position_size, vecm_confidence)
            signal = 1.0
        # 최적화된 매도 조건
        elif vecm_confidence < self.sell_confidence_threshold and position_ratio > self.min_position_ratio_for_sell:
            # 매도 신호
            position_size = position_ratio
            signal = -1.0
        else:
            # 보유
            position_size = position_ratio
            signal = 0.0
        
        action = np.array([position_size, signal], dtype=np.float32)
        return action, None


class VECMRLAgent:
    """
    VECM 기반 RL 거래 에이전트 (Simple Policy만 사용)
    """
    
    def __init__(self, 
                 simple_policy_params: dict = None):
        """
        Parameters:
        -----------
        simple_policy_params : dict, optional
            Simple Policy 파라미터
        """
        # Simple Policy 파라미터 (최적화 가능)
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
        액션 예측
        
        Parameters:
        -----------
        observation : np.ndarray
            현재 상태 관찰값
        deterministic : bool
            결정적 예측 여부 (Simple Policy는 항상 결정적)
        
        Returns:
        --------
        action : np.ndarray
            [position_size, signal]
        """
        action, _ = self.model.predict(observation, deterministic)
        return action
