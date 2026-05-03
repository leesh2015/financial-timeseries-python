import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Financial Planck Constant (v2.3 Spec)
TICK_SIZE = 0.01
MIN_TRADE_UNIT = 1
PLANCK_HF = TICK_SIZE * MIN_TRADE_UNIT

@dataclass
class QuantumState:
    jump_id: int 
    bid1: float
    ask1: float
    bid_vol1: int
    ask_vol1: int
    total_t: float
    buy_vol: int
    sell_vol: int
    duration_ms: int
    bid_vols: List[int] = field(default_factory=list)
    ask_vols: List[int] = field(default_factory=list)

class HamiltonianCalculator:
    def __init__(self, interaction_coefficient=0.5, vol_multiplier=1.0):
        self.transfer_coef = interaction_coefficient
        self.vol_multiplier = vol_multiplier

    def calculate_kinetic_energy(self, impact_sequence: List, velocity_only: bool = False, total_vol: float = 0, duration_ms: float = 0) -> float:
        if velocity_only:
            if total_vol <= 0 or duration_ms <= 0: return 0.0
            velocity = 1000.0 / duration_ms
            return total_vol * (velocity ** 2)

        total_t = 0.0
        for imp in impact_sequence:
            # Handle both class-based and dict-based impacts
            if hasattr(imp, 'volume'):
                vol = imp.volume * self.vol_multiplier
                ms = imp.offset_ms
                ns = getattr(imp, 'offset_ns', None)
            else:
                vol = (imp.get("volume") or imp.get("vol") or 0) * self.vol_multiplier
                ms = (imp.get("offset_ms") or imp.get("ms") or 1)
                ns = (imp.get("offset_ns") or imp.get("ns"))
                
            if ns is not None and ns > 0:
                velocity = 1000000.0 / ns
                kinetic = vol * (velocity ** 2)
            else:
                dt_ms = max(1, ms)
                velocity = 1000.0 / dt_ms
                kinetic = vol * (velocity ** 2)
            total_t += kinetic
        return total_t

    def apply_annihilation_interaction(self, buy_vol: int, sell_vol: int, base_t: float) -> float:
        annihilation_vol = min(buy_vol, sell_vol)
        released_energy = annihilation_vol * self.transfer_coef
        return base_t + released_energy

    def quantize(self, energy: float) -> int:
        if PLANCK_HF <= 0: return int(energy)
        return int(np.round(energy / PLANCK_HF))

class VirtualPotentialEstimator:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth 

    def estimate_virtual_potential(self, current_state: QuantumState, prev_t: float) -> np.ndarray:
        potential = np.zeros(self.max_depth)
        a1_vol = current_state.ask_vols[0] if len(current_state.ask_vols) > 0 else current_state.ask_vol1
        b1_vol = current_state.bid_vols[0] if len(current_state.bid_vols) > 0 else current_state.bid_vol1
        potential[1] = a1_vol
        potential[3] = b1_vol
        
        repulsion_force = prev_t * 0.1 
        if len(current_state.ask_vols) >= 2: potential[0] = current_state.ask_vols[1]
        else: potential[0] = a1_vol * (1.0 + repulsion_force)
            
        if len(current_state.bid_vols) >= 2: potential[4] = current_state.bid_vols[1]
        else: potential[4] = b1_vol * (1.0 + repulsion_force)
            
        potential[2] = 0.0 
        return potential

class TransitionMatrixBuilder:
    def __init__(self, base_dim=5):
        self.base_dim = base_dim
        self.current_dim = base_dim
        self.low_energy_counter = 0 
        self.matrix = np.eye(self.base_dim)

    def build_or_update(self, n: float, t_density: float, v_sum: float, buy_vol: int, sell_vol: int):
        density = t_density / v_sum if v_sum > 0 else 0
        if density > 0.8:
            self.low_energy_counter = 0
            if self.current_dim == self.base_dim: self.current_dim = 10
        elif density < 0.2:
            self.low_energy_counter += 1
            if self.low_energy_counter >= 3:
                if self.current_dim > self.base_dim: self.current_dim = self.base_dim
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

class QuantumDemonEngine:
    def __init__(self, vol_multiplier=1.0):
        self.vol_multiplier = vol_multiplier
        self.hamiltonian = HamiltonianCalculator(vol_multiplier=vol_multiplier)
        self.virtual_potential = VirtualPotentialEstimator()
        self.matrix_builder = TransitionMatrixBuilder()
        
    def process_state(self, state: QuantumState, impact_seq: List, horizon_n: float = 1.0, velocity_only: bool = False):
        scaled_buy = int(state.buy_vol * self.vol_multiplier)
        scaled_sell = int(state.sell_vol * self.vol_multiplier)
        
        base_t = self.hamiltonian.calculate_kinetic_energy(
            impact_seq, 
            velocity_only=velocity_only, 
            total_vol=state.buy_vol + state.sell_vol,
            duration_ms=state.duration_ms
        )
        t_interacted = self.hamiltonian.apply_annihilation_interaction(state.buy_vol, state.sell_vol, base_t)
        q_energy = self.hamiltonian.quantize(t_interacted)
        
        v_field = self.virtual_potential.estimate_virtual_potential(state, t_interacted)
        v_sum = np.sum(v_field)
        q_potential = self.hamiltonian.quantize(v_sum)
        
        self.matrix_builder.build_or_update(
            n=1, t_density=q_energy, v_sum=q_potential, 
            buy_vol=scaled_buy, sell_vol=scaled_sell
        )
        powered_matrix = self.matrix_builder.get_n_step_matrix(horizon_n)
        
        return {
            "jump_id": state.jump_id, "quantized_T": q_energy, "quantized_V": q_potential,
            "density": q_energy / q_potential if q_potential > 0 else 0,
            "dimension": self.matrix_builder.current_dim, "matrix": powered_matrix
        }
