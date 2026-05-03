import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from coin_models import CoinJumpState, KineticImpact

logger = logging.getLogger(__name__)

# Financial Planck Constant (Symmetric Physics v2.3)
# Based on Binance BTC/USDT Spec: Tick Size (0.1) * Min Volume Step (0.001)
TICK_SIZE = 0.1
MIN_UNIT = 0.001
PLANCK_HF = TICK_SIZE * MIN_UNIT

class HamiltonianCalculator:
    def __init__(self, interaction_coefficient=0.5, vol_multiplier=1.0):
        self.transfer_coef = interaction_coefficient
        self.vol_multiplier = vol_multiplier

    def calculate_kinetic_energy(self, impact_sequence: List[KineticImpact], velocity_only: bool = False, total_vol: float = 0, duration_ms: float = 0) -> float:
        if velocity_only:
            if total_vol <= 0: return 0.0
            dt_ms = max(1.0, duration_ms)
            velocity = 1000.0 / dt_ms
            return total_vol * (velocity ** 2) * self.vol_multiplier

        return sum(imp.get_kinetic_energy() for imp in impact_sequence) * self.vol_multiplier

    def apply_annihilation_interaction(self, buy_vol: float, sell_vol: float, base_t: float) -> float:
        annihilation_vol = min(buy_vol, sell_vol)
        released_energy = annihilation_vol * self.transfer_coef
        return base_t + released_energy

    def quantize(self, energy: float) -> int:
        if PLANCK_HF <= 0: return int(energy)
        return int(np.round(energy / PLANCK_HF))

class VirtualPotentialEstimator:
    def estimate_potential(self, state: CoinJumpState, current_dim: int) -> float:
        # Use top levels based on dimension
        if current_dim == 10:
            asks = sum(p.volume for p in state.initial_potential.asks[:10])
            bids = sum(p.volume for p in state.initial_potential.bids[:10])
            return asks + bids
        else:
            return state.initial_potential.v_sum_5

class TransitionMatrixBuilder:
    def __init__(self, base_dim=5):
        self.base_dim = base_dim
        self.current_dim = base_dim
        self.low_energy_counter = 0 
        self.matrix = np.eye(self.base_dim)

    def build_or_update(self, density: float, buy_vol: float, sell_vol: float):
        # Adaptive Dimension Logic
        if density > 0.8:
            self.low_energy_counter = 0
            if self.current_dim == self.base_dim:
                self.expand_dimension()
        elif density < 0.2:
            self.low_energy_counter += 1
            if self.low_energy_counter >= 3:
                if self.current_dim > self.base_dim:
                    self.contract_dimension()
                self.low_energy_counter = 0
        else:
            self.low_energy_counter = 0
            
        dim = self.current_dim
        self.matrix = np.zeros((dim, dim))
        stay_prob = 1.0 - min(density, 0.9)
        
        vol_sum = buy_vol + sell_vol + 1e-9
        up_weight, down_weight = (buy_vol / vol_sum, sell_vol / vol_sum)
        
        up_prob, down_prob = (1.0 - stay_prob) * up_weight, (1.0 - stay_prob) * down_weight
        
        for i in range(dim):
            actual_stay_prob = stay_prob
            if i == 0: actual_stay_prob += up_prob
            if i == dim - 1: actual_stay_prob += down_prob
            self.matrix[i, i] = actual_stay_prob
            if i > 0: self.matrix[i, i-1] = up_prob 
            if i < dim - 1: self.matrix[i, i+1] = down_prob 
            
        return self.matrix

    def get_n_step_matrix(self, n: float):
        if n <= 1.0: return self.matrix
        full_steps = int(n)
        fraction = n - full_steps
        m_int = np.linalg.matrix_power(self.matrix, full_steps)
        if fraction < 0.001: return m_int
        m_plus = np.dot(m_int, self.matrix)
        return m_int * (1.0 - fraction) + m_plus * fraction

    def expand_dimension(self):
        self.current_dim = 10
        
    def contract_dimension(self):
        self.current_dim = self.base_dim

class CoinEngine:
    def __init__(self, vol_multiplier=1.0):
        self.vol_multiplier = vol_multiplier
        self.hamiltonian = HamiltonianCalculator(vol_multiplier=vol_multiplier)
        self.potential_estimator = VirtualPotentialEstimator()
        self.matrix_builder = TransitionMatrixBuilder(base_dim=5)
        
    def process_state(self, state: CoinJumpState, horizon_n: float = 1.0):
        # 1. Kinetic Energy
        buy_vol = sum(imp.volume for imp in state.impact_sequence if imp.is_buy)
        sell_vol = sum(imp.volume for imp in state.impact_sequence if not imp.is_buy)
        total_vol = buy_vol + sell_vol
        
        # Use historical duration from state (Crucial for backtesting parity)
        duration_ms = state.duration_ms if state.duration_ms > 0 else 1.0

        base_t = self.hamiltonian.calculate_kinetic_energy(
            state.impact_sequence, 
            velocity_only=state.velocity_only,
            total_vol=total_vol,
            duration_ms=duration_ms
        )
        
        t_interacted = self.hamiltonian.apply_annihilation_interaction(buy_vol, sell_vol, base_t)
        q_energy = self.hamiltonian.quantize(t_interacted)
        
        # 2. Potential Field
        v_sum = self.potential_estimator.estimate_potential(state, self.matrix_builder.current_dim)
        q_potential = self.hamiltonian.quantize(v_sum)
        
        density = q_energy / q_potential if q_potential > 0 else 0
        
        # 3. Matrix & Horizon
        self.matrix_builder.build_or_update(density=min(density, 0.9), buy_vol=buy_vol, sell_vol=sell_vol)
        powered_matrix = self.matrix_builder.get_n_step_matrix(horizon_n)
        
        return {
            "quantized_T": q_energy, "quantized_V": q_potential,
            "density": density, "dimension": self.matrix_builder.current_dim,
            "matrix": powered_matrix
        }
