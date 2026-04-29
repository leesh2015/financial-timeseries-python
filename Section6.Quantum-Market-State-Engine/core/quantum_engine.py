import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Financial Planck Constant
# Tick Size (0.01) * Min Trade Unit (1)
TICK_SIZE = 0.01
MIN_TRADE_UNIT = 1
PLANCK_HF = TICK_SIZE * MIN_TRADE_UNIT

@dataclass
class QuantumState:
    jump_id: int # n (Number of jump events)
    bid1: float
    ask1: float
    bid_vol1: int
    ask_vol1: int
    total_t: float
    buy_vol: int
    sell_vol: int
    duration_ms: int

class HamiltonianCalculator:
    def __init__(self, interaction_coefficient=0.5):
        self.transfer_coef = interaction_coefficient

    def calculate_kinetic_energy(self, impact_sequence: List) -> float:
        r"""
        Derives kinetic energy (T) from trade data.
        T = \sum V * (1/dt)^2
        """
        total_t = 0.0
        for imp in impact_sequence:
            # Supports both KineticImpact objects or Dicts
            if hasattr(imp, 'volume'):
                vol = imp.volume
                ms = imp.offset_ms
            else:
                vol = imp.get("vol", 0)
                ms = imp.get("ms", 1)
                
            if ms <= 0: ms = 1
            dt_sec = ms / 1000.0
            
            # Acceleration a = 1 / dt
            acceleration = 1.0 / dt_sec
            kinetic = vol * (acceleration ** 2)
            total_t += kinetic
            
        return total_t

    def apply_annihilation_interaction(self, buy_vol: int, sell_vol: int, base_t: float) -> float:
        """
        Annihilation: Buy and sell particles meet, release energy, and disappear.
        The released energy is transferred to the kinetic energy (T) of surrounding particles.
        """
        # Annihilation volume is the minimum of buy and sell (the amount that met and disappeared)
        annihilation_vol = min(buy_vol, sell_vol)
        
        # Energy released by annihilation = Annihilation volume * Transfer coefficient
        released_energy = annihilation_vol * self.transfer_coef
        
        # Add released energy to base kinetic energy (Acceleration effect)
        return base_t + released_energy

    def apply_field_contraction(self, prev_v: float, current_v: float, no_trade_vol: int) -> float:
        """
        Field Contraction: When remaining volume is canceled without trades, the barrier thickness decreases.
        """
        contraction_rate = 1.0
        if current_v < prev_v and no_trade_vol > 0:
            # Adjust critical energy downward by the ratio of canceled volume
            contraction_rate = current_v / prev_v
            
        return contraction_rate

    def quantize(self, energy: float) -> int:
        """
        Normalizes all energy to integer multiples of the Financial Planck Constant (h_f).
        """
        if PLANCK_HF == 0: return int(energy)
        return int(np.round(energy / PLANCK_HF))

class VirtualPotentialEstimator:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth # 5x5 boundary condition

    def estimate_virtual_potential(self, current_state: QuantumState, prev_t: float) -> np.ndarray:
        """
        Renders virtual potential for 2nd~5th levels based on 1st level data and impact repulsion (previous T).
        """
        # Initialize 5x5 potential array (center is current price, index 0~4)
        # 0: Ask2, 1: Ask1, 2: Mid, 3: Bid1, 4: Bid2 (Simplified 5-dimensional model)
        potential = np.zeros(self.max_depth)
        
        # Set 1st level barrier
        potential[1] = current_state.ask_vol1
        potential[3] = current_state.bid_vol1
        
        # Calculate impact repulsion force (Higher T increases the probability of hidden liquidity behind)
        repulsion_force = prev_t * 0.1 
        
        # Reverse calculation of virtual potential (Level 2 and beyond)
        potential[0] = current_state.ask_vol1 * (1.0 + repulsion_force) # Estimated Ask2
        potential[4] = current_state.bid_vol1 * (1.0 + repulsion_force) # Estimated Bid2
        
        # Center (between current prices) has 0 potential (Free space)
        potential[2] = 0.0
        
        return potential

class TransitionMatrixBuilder:
    def __init__(self, base_dim=5):
        self.base_dim = base_dim
        self.current_dim = base_dim
        self.low_energy_counter = 0 # Counter for 3n cycles
        self.matrix = np.eye(self.base_dim)

    def build_or_update(self, n: int, t_density: float, v_sum: float, buy_vol: int, sell_vol: int):
        """
        Transition matrix engine driven solely by the number of event occurrences (n) and energy density (T/V).
        Physical time (sec) variable is completely excluded.
        """
        # Energy Density
        density = t_density / v_sum if v_sum > 0 else 0
        
        # 1. Interrupt and Expansion Logic
        if density > 0.8:
            # Immediate discard of contraction (Interrupt) and expand
            self.low_energy_counter = 0
            if self.current_dim == self.base_dim:
                self.expand_dimension()
                
        # 2. Contraction Logic (3n cycles)
        elif density < 0.2:
            self.low_energy_counter += 1
            if self.low_energy_counter >= 3:
                # Contract dimension upon settling for 3 cycles
                if self.current_dim > self.base_dim:
                    self.contract_dimension()
                self.low_energy_counter = 0
        else:
            self.low_energy_counter = 0
            
        # 3. Transition Matrix Update (Reflect Hamiltonian perturbation and apply asymmetry)
        self._update_matrix_probabilities(density, buy_vol, sell_vol)
        
        # Perform matrix powering corresponding to n times (Accumulation of events)
        return np.linalg.matrix_power(self.matrix, n)
        
    def expand_dimension(self):
        logger.info(f"Energy burst detected! Dimension expansion: {self.current_dim} -> 10")
        self.current_dim = 10
        self.matrix = np.eye(self.current_dim) # Expanded initial phase
        
    def contract_dimension(self):
        logger.info(f"Energy dissipation complete (3n cycle). Dimension contraction: {self.current_dim} -> {self.base_dim}")
        self.current_dim = self.base_dim
        self.matrix = np.eye(self.base_dim)
        
    def _update_matrix_probabilities(self, density: float, buy_vol: int, sell_vol: int):
        """
        Maps the probability of jumping up/down to the transition matrix based on energy density.
        Distributes probabilities asymmetrically based on the ratio of buy/sell volumes.
        """
        dim = self.current_dim
        
        # Calculate directional weights (F = ma, imbalance of force)
        total_vol = buy_vol + sell_vol
        if total_vol > 0:
            up_weight = buy_vol / total_vol
            down_weight = sell_vol / total_vol
        else:
            up_weight = 0.5
            down_weight = 0.5
            
        for i in range(dim):
            # Diagonal (Probability of staying in current state)
            stay_prob = 1.0 - min(density, 0.9)
            
            # Asymmetric diffusion probability (Up/Down jumps)
            up_prob = (1.0 - stay_prob) * up_weight
            down_prob = (1.0 - stay_prob) * down_weight
            
            # Probability of moving out of bounds bounces back to current position (Conservation of probability)
            actual_stay_prob = stay_prob
            if i == 0:
                actual_stay_prob += up_prob
            if i == dim - 1:
                actual_stay_prob += down_prob
                
            self.matrix[i, i] = actual_stay_prob
            
            # Index 0 (Ask2) direction is price increase, Index 4 (Bid2) direction is price decrease
            if i > 0: self.matrix[i, i-1] = up_prob # Upward jump
            if i < dim - 1: self.matrix[i, i+1] = down_prob # Downward jump

class QuantumDemonEngine:
    def __init__(self):
        self.hamiltonian = HamiltonianCalculator()
        self.virtual_potential = VirtualPotentialEstimator()
        self.matrix_builder = TransitionMatrixBuilder()
        
    def process_state(self, state: QuantumState, impact_seq: List[Dict]):
        # 1. Calculate Kinetic Energy
        base_t = self.hamiltonian.calculate_kinetic_energy(impact_seq)
        
        # 2. Apply Annihilation Interaction
        t_interacted = self.hamiltonian.apply_annihilation_interaction(state.buy_vol, state.sell_vol, base_t)
        
        # 3. Energy Quantization (hf)
        q_energy = self.hamiltonian.quantize(t_interacted)
        
        # 4. Potential Energy (Including Virtual Potential)
        v_field = self.virtual_potential.estimate_virtual_potential(state, t_interacted)
        v_sum = np.sum(v_field)
        q_potential = self.hamiltonian.quantize(v_sum)
        
        # 5. Drive Transition Matrix (n, T/V)
        # jump_id represents the number of jumps n. Here we apply n=1 for one step progression.
        # For cumulative powering, state.jump_id can be utilized.
        future_matrix = self.matrix_builder.build_or_update(n=1, t_density=q_energy, v_sum=q_potential, buy_vol=state.buy_vol, sell_vol=state.sell_vol)
        
        return {
            "jump_id": state.jump_id,
            "quantized_T": q_energy,
            "quantized_V": q_potential,
            "density": q_energy / q_potential if q_potential > 0 else 0,
            "dimension": self.matrix_builder.current_dim,
            "matrix": future_matrix
        }
