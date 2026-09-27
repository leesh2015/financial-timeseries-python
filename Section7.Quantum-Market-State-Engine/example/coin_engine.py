import numpy as np
import logging
from scipy.stats import entropy
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
from coin_models import (
    KineticImpact, MarketPotential, CoinJumpState, calculate_physical_horizon
)

logger = logging.getLogger(__name__)

# Financial Planck Constant for Binance BTC/USDT (v3.4.8)
# Tick Size (0.1) * Min Volume Step (0.001) = 0.0001
TICK_SIZE = 0.1
MIN_UNIT = 0.001
PLANCK_HF = TICK_SIZE * MIN_UNIT

class HamiltonianCalculator:
    def __init__(self, planck_hf=0.0001, interaction_coefficient=0.5, vol_multiplier=1.0, max_kinetic=10000000.0):
        self.planck_hf = planck_hf
        self.transfer_coef = interaction_coefficient
        self.vol_multiplier = vol_multiplier
        self.max_kinetic = max_kinetic

    def calculate_kinetic_energy(self, impact_sequence: List, velocity_only: bool = False, total_vol: float = 0, duration_ms: float = 0) -> float:
        if velocity_only:
            if total_vol <= 0: return 0.0
            dt_ms = max(1.0, duration_ms)
            velocity = 1000.0 / dt_ms
            return min(total_vol * (velocity ** 2), self.max_kinetic * 10)

        total_t = 0.0
        for imp in impact_sequence:
            vol = imp.volume * self.vol_multiplier
            ms = max(1, imp.offset_ms)
            ns = getattr(imp, 'offset_ns', None)
                
            if ns is not None and ns > 0:
                velocity = 1000000.0 / max(1, ns)
                kinetic = vol * (velocity ** 2)
            else:
                velocity = 1.0 / ms
                kinetic = vol * (velocity ** 2)
            
            total_t += min(kinetic, self.max_kinetic)
            
        return total_t

    def apply_annihilation_interaction(self, buy_vol: float, sell_vol: float, base_t: float) -> float:
        annihilation_vol = min(buy_vol, sell_vol)
        released_energy = annihilation_vol * self.transfer_coef
        return base_t + released_energy

    def quantize(self, energy: float) -> int:
        if self.planck_hf <= 0: return int(energy)
        return int(np.round(energy / self.planck_hf))

class VirtualPotentialEstimator:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def estimate_virtual_potential(self, state: CoinJumpState, current_dim: int, prev_t: float, spectral_gap: float) -> np.ndarray:
        """
        [v3.4.8] Impedance-Matched Virtual Potential (Zero-Constant Coin Port)
        Maps the 20-level order book into a Physical Potential Field matching current_dim.
        """
        potential = np.zeros(current_dim)
        mid_idx = current_dim // 2
        
        # Group 20 levels into boundaries. Level 1-2 as immediate resistance, Level 3-20 as deep walls.
        ask1_vol = sum(p.volume for p in state.initial_potential.asks[:2])
        bid1_vol = sum(p.volume for p in state.initial_potential.bids[:2])
        ask2_vol = sum(p.volume for p in state.initial_potential.asks[2:20])
        bid2_vol = sum(p.volume for p in state.initial_potential.bids[2:20])
        
        # Immediate Support/Resistance
        potential[mid_idx - 1] = ask1_vol
        potential[mid_idx + 1] = bid1_vol
        
        # Boundary Walls (Ask2/Bid2)
        if current_dim >= 5:
            potential[0] = ask2_vol
            potential[-1] = bid2_vol
            
        return potential

class TransitionMatrixBuilder:
    def __init__(self, base_dim=5):
        self.base_dim = base_dim
        self.current_dim = base_dim
        self.low_energy_counter = 0 
        self.matrix = np.eye(self.base_dim)
        self.spectral_gap = 0.1 # Real-time state order


    def expand_dimension(self):
        self.current_dim = 10
    
    def contract_dimension(self):
        self.current_dim = self.base_dim

    def build_or_update(self, density: float, buy_vol: float, sell_vol: float, spectral_gap: float, v_field: np.ndarray):
        """
        [v3.4] Zero-Constant Phase Transition
        Dimension expansion/contraction driven by Spectral Gap (Order)
        """
        self.spectral_gap = spectral_gap
        rho_high = 1.0 - spectral_gap
        rho_low = 0.5 * rho_high
        
        # Immediate expansion in high order, cautious contraction in chaos
        if density > rho_high:
            self.low_energy_counter = 0
            if self.current_dim == self.base_dim:
                self.expand_dimension()
        elif density < rho_low:
            self.low_energy_counter += 1
            inertia_threshold = max(1, int(1.0 / max(0.001, spectral_gap)))
            if self.low_energy_counter >= inertia_threshold:
                if self.current_dim > self.base_dim:
                    self.contract_dimension()
                self.low_energy_counter = 0
        else:
            self.low_energy_counter = 0
            
        dim = self.current_dim
        self.matrix = np.zeros((dim, dim))
        stay_prob = 1.0 - min(density, 0.9)
        
        # [v3.4.8] Unified Hamiltonian Direction (Execution 50% + Potential 50%)
        total_vol = buy_vol + sell_vol + 1e-9
        exec_bias = (buy_vol - sell_vol) / total_vol
        
        mid_idx = dim // 2
        v_grad = (v_field[mid_idx + 1] - v_field[mid_idx - 1]) / (v_field[mid_idx + 1] + v_field[mid_idx - 1] + 1e-9)
        
        combined_bias = (exec_bias * 0.5) + (v_grad * 0.5)
        
        up_weight = 0.5 + (combined_bias * 0.5)
        down_weight = 1.0 - up_weight
        
        up_prob = (1.0 - stay_prob) * up_weight
        down_prob = (1.0 - stay_prob) * down_weight
        
        for i in range(dim):
            actual_stay_prob = stay_prob
            if i == 0: actual_stay_prob += up_prob
            if i == dim - 1: actual_stay_prob += down_prob
            self.matrix[i, i] = actual_stay_prob
            if i > 0: self.matrix[i, i-1] = up_prob 
            if i < dim - 1: self.matrix[i, i+1] = down_prob 
            
        # [v3.0 Fix] Unitary Safety Check
        row_sums = self.matrix.sum(axis=1)
        for i in range(dim):
            if row_sums[i] > 0:
                self.matrix[i, :] /= row_sums[i]
                
        return self.matrix

    def get_n_step_matrix(self, n: float):
        if n <= 1.0: return self.matrix
        full_steps = int(n)
        fraction = n - full_steps
        m_int = np.linalg.matrix_power(self.matrix, full_steps)
        if fraction < 0.001: 
            res = m_int
        else:
            m_plus = np.dot(m_int, self.matrix)
            res = m_int * (1.0 - fraction) + m_plus * fraction
            
        # [v3.0 Fix] Final Unitary Normalization (Prevents drift after matrix power)
        row_sums = res.sum(axis=1)
        for i in range(len(res)):
            if row_sums[i] > 0:
                res[i, :] /= row_sums[i]
        return res

class CoinEngine:
    """Unified v3.4.8 Wrapper for Coin Logic (Zero-Constant)"""
    def __init__(self, vol_multiplier=1.0):

        self.vol_multiplier = vol_multiplier
        self.hamiltonian = HamiltonianCalculator(planck_hf=PLANCK_HF, vol_multiplier=vol_multiplier)
        self.virtual_potential = VirtualPotentialEstimator()
        self.matrix_builder = TransitionMatrixBuilder(base_dim=5)
        
    def process_state(self, state: CoinJumpState, horizon_n: Optional[float] = None, target_gain: float = 6.0):
        buy_vol = sum(imp.volume for imp in state.impact_sequence if imp.is_buy)
        sell_vol = sum(imp.volume for imp in state.impact_sequence if not imp.is_buy)
        total_vol = buy_vol + sell_vol
        
        base_t = self.hamiltonian.calculate_kinetic_energy(
            state.impact_sequence, 
            velocity_only=state.velocity_only,
            total_vol=total_vol,
            duration_ms=state.duration_ms
        )
        
        t_interacted = self.hamiltonian.apply_annihilation_interaction(buy_vol, sell_vol, base_t)
        q_energy = self.hamiltonian.quantize(t_interacted)
        
        v_field = self.virtual_potential.estimate_virtual_potential(
            state, 
            self.matrix_builder.current_dim,
            t_interacted,
            getattr(state, 'spectral_gap', 0.1)
        )
        v_sum = np.sum(v_field)
        q_potential = self.hamiltonian.quantize(v_sum)
        
        # [v3.4.8] Log-scale scaling with Information Saturation Limit (log2 3)
        raw_density = q_energy / q_potential if q_potential > 0 else 0
        density = min(0.95, raw_density * np.log1p(q_potential / PLANCK_HF) * np.log2(3))
        
        spectral_gap = getattr(state, 'spectral_gap', 0.1)
        self.matrix_builder.build_or_update(
            density=density, 
            buy_vol=buy_vol, 
            sell_vol=sell_vol,
            spectral_gap=spectral_gap,
            v_field=v_field
        )
        
        if horizon_n is None:
            raw_n = calculate_physical_horizon(self.matrix_builder.matrix, target_gain=target_gain)
            # [v3.4.8] Dynamic Coherence Guard (Phase Space Volume Brake)
            # Phase Space Volume (V) = 11 (Macro) * 5 or 10 (Micro) = 55 or 110.
            base_cap = self.last_tc if hasattr(self, 'last_tc') else 55.0
            max_h = base_cap * (2.0 if self.matrix_builder.current_dim > 5 else 1.0)
            horizon_n = min(raw_n, max_h)
            
        powered_matrix = self.matrix_builder.get_n_step_matrix(horizon_n)
        
        try:
            evals = np.linalg.eigvals(self.matrix_builder.matrix)
            sorted_abs_evals = np.sort(np.abs(evals))[::-1]
            gap = 1.0 - sorted_abs_evals[1] if len(sorted_abs_evals) > 1 else 0.5
            self.last_tc = min(55.0, 1.0 / max(0.001, gap))
        except:
            self.last_tc = 55.0
        
        return {
            "quantized_T": q_energy, "quantized_V": q_potential,
            "density": density, "dimension": self.matrix_builder.current_dim,
            "horizon_n": horizon_n,
            "matrix": powered_matrix,
            "base_matrix": self.matrix_builder.matrix
        }

class QuantumAdaptiveCore:
    """[v3.4] Zero-Constant Physical Intelligence Layer (Coin Port)"""
    def __init__(self, history_size=605):
        # [v3.4] Event Horizon confirmed at 605 for 11D stability
        self.history_size = 605
        self.price_history = deque(maxlen=605)
        self.current_tick = TICK_SIZE
        self.is_calibrating = False
        
        self.last_direction = 0
        self.current_run_length = 0
        self.free_paths = deque(maxlen=100)
        self.current_tc = 100.0
        self.current_mfp = 1.0
        self.current_window = 50
        self._collapse_path_count = 0
        self.spectral_gap = 0.1 # Default spectral gap


    def get_phase_changes(self, data: List[float]) -> int:
        """[v3.4] Nyquist-Shannon Criterion: Count significant phase reversals"""
        if len(data) < 2: return 0
        diffs = np.diff(data)
        signs = np.sign(diffs[np.abs(diffs) > 1e-9])
        if len(signs) < 2: return 0
        return np.sum(signs[1:] != signs[:-1])

    def add_event(self, price: float):
        if len(self.price_history) > 0:
            diff = price - self.price_history[-1]
            if abs(diff) > 1e-9:
                self.price_history.append(price)
                direction = np.sign(diff)
                if direction != self.last_direction and self.last_direction != 0:
                    if self.current_run_length > 0:
                        self.free_paths.append(self.current_run_length)
                    self.current_run_length = 1
                else:
                    self.current_run_length += 1
                self.last_direction = direction
                self.current_mfp = np.mean(self.free_paths) if len(self.free_paths) > 0 else 1.0
        else:
            self.price_history.append(price)

    def update_coherence_time(self, matrix: np.ndarray):
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            sorted_evals = np.sort(np.abs(eigenvalues))[::-1]
            if len(sorted_evals) > 1:
                self.spectral_gap = 1.0 - sorted_evals[1]
                if self.spectral_gap > 0.001:
                    self.current_tc = min(1000.0, 1.0 / self.spectral_gap)
                else:
                    self.current_tc = 1000.0
        except Exception:
            pass

    def calculate_entropy(self, prices: Optional[List[float]] = None) -> float:
        """[v3.4] Physics-First Entropy: Requires Nyquist-Shannon Phase Criterion"""
        data = prices if prices is not None else list(self.price_history)
        
        # Nyquist-Shannon Guard: Must have at least 1 full cycle (2 phase changes)
        if self.get_phase_changes(data) < 2: 
            return 2.0 # Maximum uncertainty if not enough information innovation
            
        diffs = np.diff(data)
        signs = np.sign(diffs[np.abs(diffs) > 1e-9])
        counts = {1: 0, -1: 0}
        for s in signs: 
            if s in counts: counts[s] += 1
            
        total = sum(counts.values())
        if total == 0: return 2.0
        
        probs = [count / total for count in counts.values() if count > 0]
        return entropy(probs, base=2)

    def find_optimal_n(self, base_tick=0.1, max_n=5.0) -> float:
        """[v3.4] Adaptive Lens Calibration via Information Innovation Convergence"""
        # Information Innovation Window (w_min) = Tc (Coherence Time)
        w_min = max(20, int(self.current_tc))
        self.current_window = w_min
        
        if len(self.price_history) < w_min: return self.current_tick
        
        best_n = base_tick
        max_q = -1.0
        resolutions = np.arange(base_tick, max_n + 0.1, 0.1)
        prices = np.array(self.price_history)[-w_min:]
        
        for res in resolutions:
            resampled = [prices[0]]
            for p in prices[1:]:
                if abs(p - resampled[-1]) >= res - 1e-9:
                    resampled.append(p)
            
            # Shannon Sampling Guard
            if self.get_phase_changes(resampled) < 2: continue
            
            h = self.calculate_entropy(resampled)
            diffs = np.diff(resampled)
            bias = abs(np.sum(np.sign(diffs))) / len(diffs)
            q = (1.0 - h) * bias * np.log(len(resampled))
            if q > max_q:
                max_q = q
                best_n = res
        return best_n

    def check_coherence_collapse(self) -> bool:
        """
        [v3.4] Zero-Constant Coherence Collapse Detection.
        Replaced all heuristic constants with Information Capacity and Spectral Gap limits.
        """
        current_n = len(self.free_paths)
        if current_n <= self._collapse_path_count: 
            return False
            
        # 1. Information Capacity Limit (Noise Filter)
        # log2(3) is the capacity limit for a 3-state system. 
        # If entropy exceeds this, the signal is physically indistinguishable from thermal noise.
        if self.calculate_entropy() > np.log2(3):
            self._collapse_path_count = current_n
            return True
            
        # 2. Spectral Gap Deceleration (Inertia Collision)
        if len(self.free_paths) >= 2:
            last_path = self.free_paths[-1]
            prev_path = self.free_paths[-2]
            
            # Using the spectral gap as the physical deceleration threshold
            # If the path length collapses faster than the spectral gap allows, it's a collision.
            if prev_path > 0 and (last_path / prev_path) < self.spectral_gap:
                self._collapse_path_count = current_n
                return True
                
        return False
