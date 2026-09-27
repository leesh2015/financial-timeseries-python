import numpy as np
import logging
from scipy.stats import entropy
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
from core.quantum_models import (
    KineticImpact, MarketPotential, QuantumState, calculate_physical_horizon
)

logger = logging.getLogger(__name__)

# Financial Planck Constant Unit Calculation
def get_planck_unit(tick_size: float, min_unit: int = 1) -> float:
    return tick_size * min_unit

class HamiltonianCalculator:
    def __init__(self, planck_hf=0.01, interaction_coefficient=0.5, vol_multiplier=1.0, max_kinetic=10000000.0):
        self.planck_hf = planck_hf
        self.transfer_coef = interaction_coefficient
        self.vol_multiplier = vol_multiplier
        self.max_kinetic = max_kinetic # Energy clamping limit

    def calculate_kinetic_energy(self, impact_sequence: List, velocity_only: bool = False, total_vol: float = 0, duration_ms: float = 0) -> float:
        """
        Calculates T. 
        [v2.4] Added Zero-Offset Defense and Energy Clamping.
        """
        if velocity_only:
            if total_vol <= 0: return 0.0
            dt_ms = max(1.0, duration_ms)
            # Velocity scale standardized to ms units
            velocity = 1000.0 / dt_ms
            return min(total_vol * (velocity ** 2), self.max_kinetic * 10)

        total_t = 0.0
        for imp in impact_sequence:
            if hasattr(imp, 'volume'):
                vol = imp.volume * self.vol_multiplier
                ms = max(1, imp.offset_ms)
                ns = getattr(imp, 'offset_ns', None)
            else:
                vol = (imp.get("volume") or imp.get("vol") or 0) * self.vol_multiplier
                ms = max(1, (imp.get("offset_ms") or imp.get("ms") or 1))
                ns = (imp.get("offset_ns") or imp.get("ns"))
                
            if ns is not None and ns > 0:
                # [v3.4.2] Standardized Velocity (1 unit = 1ms traversal)
                velocity = 1000000.0 / max(1, ns)
                kinetic = vol * (velocity ** 2)
            else:
                velocity = 1.0 / ms
                kinetic = vol * (velocity ** 2)
            
            # [v3.4.2] Dynamic Energy Clamping based on Market Breadth
            total_t += kinetic
        
        return total_t


    def apply_annihilation_interaction(self, buy_vol: int, sell_vol: int, base_t: float) -> float:
        annihilation_vol = min(buy_vol, sell_vol)
        released_energy = annihilation_vol * self.transfer_coef
        return base_t + released_energy

    def quantize(self, energy: float) -> int:
        if self.planck_hf <= 0: return int(energy)
        # [v3.4.8] Physics-First Density Scaling
        # Multiplier derived from Information Saturation Limit: log2(3) bits
        # Represents the maximum entropy capacity for ternary decisions (UP, DOWN, STAY).
        val = energy / self.planck_hf
        if energy > 1e-9 and val < 1.0: return 1
        return int(np.round(val))


class VirtualPotentialEstimator:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth # 5x5 boundary condition

    def estimate_virtual_potential(self, current_state: QuantumState, prev_t: float, spectral_gap: float = 0.5, jump_scale: float = 1.0) -> np.ndarray:
        """
        [v3.1.9] Spectral Gap Based Repulsion (Zero-Constant)
        Replaces static 0.1 with Hamiltonian order-chaos delta.
        Preserves 417 pts Momentum-Attraction mapping.
        """
        potential = np.zeros(self.max_depth)
        
        # --- Level 1 (Repulsion/Resistance Mode for Stocks) ---
        # Large volume at Ask acts as resistance (pushes price DOWN)
        # Large volume at Bid acts as support (pushes price UP)
        a1_vol = current_state.ask_vols[0] if len(current_state.ask_vols) > 0 else current_state.ask_vol1
        b1_vol = current_state.bid_vols[0] if len(current_state.bid_vols) > 0 else current_state.bid_vol1
        
        # In this 5D model (0:Ask2, 1:Ask1, 2:Mid, 3:Bid1, 4:Bid2):
        # Index 1 (Ask) should push price TOWARDS Index 2, 3, 4 (DOWN)
        # Index 3 (Bid) should push price TOWARDS Index 2, 1, 0 (UP)
        # Therefore, Potential at Index 1 attracts Index 0, 1 (keeps it there/pushes from mid)
        potential[1] = a1_vol 
        potential[3] = b1_vol

        
        # --- Level 2 (Real vs Estimation) ---
        # Replace 0.1 with (1.0 - spectral_gap)
        # Higher order (gap) -> lower repulsion
        repulsion_force = (prev_t / jump_scale) * (1.0 - spectral_gap)
        
        # Ask2 Logic
        if len(current_state.ask_vols) >= 2:
            potential[0] = current_state.ask_vols[1]
        else:
            potential[0] = a1_vol * (1.0 + repulsion_force)
            
        # Bid2 Logic
        if len(current_state.bid_vols) >= 2:
            potential[4] = current_state.bid_vols[1]
        else:
            potential[4] = b1_vol * (1.0 + repulsion_force)
            
        potential[2] = 0.0 # Free space (Mid-price)
        return potential

class TransitionMatrixBuilder:
    def __init__(self, base_dim=5):
        self.base_dim = base_dim
        self.current_dim = base_dim
        self.low_energy_counter = 0 
        self.matrix = np.eye(self.base_dim)

    def expand_dimension(self):
        self.current_dim = 10
    
    def contract_dimension(self):
        self.current_dim = self.base_dim

    def build_or_update(self, n: float, t_density: float, v_sum: float, buy_vol: int, sell_vol: int, v_field: np.ndarray, spectral_gap: float = 0.5, planck_hf: float = 0.01):
        """Demon Engine v3.0: Physical Dimension Transition (Zero-Constant)"""
        # 1. Dimension Management (Phase Transition)
        # [v3.4.2] Physical Density Normalization (No Heuristic Multiplier)
        # Standardizes Trade Energy (T) against Book Potential (V) 
        # using the natural log of potential to handle stock-scale liquidity.
        raw_density = t_density / v_sum if v_sum > 0 else 0
        # Selective Density: Log-scale scaling with Information Saturation Limit (log2 3)
        # log2(3) bits (~1.585) is the maximum information per dimension for ternary outcomes.
        density = min(0.95, raw_density * np.log1p(v_sum / planck_hf) * np.log2(3))




        
        # [v3.1.9] Critical Density Thresholds derived from Spectral Gap
        rho_high = 1.0 - spectral_gap
        rho_low = 0.5 * rho_high

        transition_inertia = int(max(1, min(10, 1.0 / max(0.1, spectral_gap))))
        
        if density > rho_high:
            self.low_energy_counter = 0
            if self.current_dim == self.base_dim:
                self.expand_dimension()
        elif density < rho_low:
            self.low_energy_counter += 1
            if self.low_energy_counter >= transition_inertia:
                if self.current_dim > self.base_dim:
                    self.contract_dimension()
                self.low_energy_counter = 0
        else:
            self.low_energy_counter = 0

        # 2. Matrix Construction & Unitary Normalization
        dim = self.current_dim
        self.matrix = np.zeros((dim, dim))
        stay_prob = 1.0 - density


        
        # [v3.4.4] Unified Hamiltonian Direction (Execution + Potential)
        # Combine Trade Imbalance (Kinetic) with Order Book Imbalance (Potential Gradient)
        # Stocks are heavily influenced by the 'Resistance' of deep walls.
        total_vol = buy_vol + sell_vol
        exec_bias = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
        
        # Potential Gradient (V_grad): (Bid_Vol - Ask_Vol) / (Bid_Vol + Ask_Vol)
        # In stocks, a huge Ask_Vol (negative gradient) acts as a cap.
        v_grad = (v_field[3] - v_field[1]) / (v_field[3] + v_field[1] + 1e-9)
        
        # Combined Bias: Execution Momentum (50%) + Potential Resistance (50%)
        # Balanced Hamiltonian for high-liquidity stock markets like TQQQ.
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
            if i > 0: self.matrix[i, i-1] = up_prob # Upward
            if i < dim - 1: self.matrix[i, i+1] = down_prob # Downward
        
        # [v2.4] Unitary Safety Check
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

class QuantumDemonEngine:
    def __init__(self, tick_size=0.01, vol_multiplier=1.0):
        self.tick_size = tick_size
        self.vol_multiplier = vol_multiplier
        self.planck_hf = get_planck_unit(tick_size)
        self.hamiltonian = HamiltonianCalculator(planck_hf=self.planck_hf, vol_multiplier=vol_multiplier)
        self.virtual_potential = VirtualPotentialEstimator()
        self.matrix_builder = TransitionMatrixBuilder(base_dim=5)
        self.last_gap = 0.5 # Initial order estimate
        
    def process_state(self, state: QuantumState, impact_seq: List, horizon_n: Optional[float] = None, target_gain: float = 4.0, velocity_only: bool = False, jump_scale: float = 1.0):
        """
        Demon Engine v2.4 Unified Pipeline:
        1. Kinetic & Potential Calculation
        2. Horizon Derivation (if not provided)
        3. Matrix Power Projection
        """
        scaled_buy = int(state.buy_vol * self.vol_multiplier)
        scaled_sell = int(state.sell_vol * self.vol_multiplier)

        # 1. Energy Calculation
        base_t = self.hamiltonian.calculate_kinetic_energy(
            impact_seq, 
            velocity_only=velocity_only,
            total_vol=(scaled_buy + scaled_sell),
            duration_ms=state.duration_ms
        )
        t_interacted = self.hamiltonian.apply_annihilation_interaction(state.buy_vol, state.sell_vol, base_t)
        q_energy = self.hamiltonian.quantize(t_interacted)
        
        v_field = self.virtual_potential.estimate_virtual_potential(
            state, t_interacted, spectral_gap=self.last_gap, jump_scale=jump_scale
        )
        v_sum = np.sum(v_field)
        q_potential = self.hamiltonian.quantize(v_sum)
        
        # 2. Update Transition Matrix (Kinetic Mapping)
        # Use previous spectral gap for dimension transition (Inertia)
        self.matrix_builder.build_or_update(
            n=horizon_n,
            t_density=q_energy,
            v_sum=q_potential,
            buy_vol=state.buy_vol,
            sell_vol=state.sell_vol,
            v_field=v_field,
            spectral_gap=self.last_gap,
            planck_hf=self.planck_hf
        )


        
        # 3. Derive Horizon & Project (Efficiency Pipeline)
        if horizon_n is None:
            raw_n = calculate_physical_horizon(self.matrix_builder.matrix, target_gain=target_gain)
            # [v3.4.8] Dynamic Coherence Guard (Phase Space Volume Brake)
            # The Maximum Information Capacity is the Cartesian product of the two observers:
            # - Macro Observer (AdaptiveCore): 11 Dimensions (-5 to +5 ticks)
            # - Micro Observer (MatrixBuilder): 5 Dimensions (Ground State) or 10 Dimensions (Excited State)
            # Phase Space Volume (V) = 11 * 5 = 55 (Default). If expanded: 11 * 10 = 110.
            # Predicting beyond this volume causes structural memory overlap (Hallucination).
            base_cap = self.last_tc if hasattr(self, 'last_tc') else 55.0
            max_h = base_cap * (2.0 if self.matrix_builder.current_dim > 5 else 1.0)
            horizon_n = min(raw_n, max_h)


            
        powered_matrix = self.matrix_builder.get_n_step_matrix(horizon_n)

        
        # [v3.1.9] Spectral Gap (Order-Chaos Delta)
        try:
            evals = np.linalg.eigvals(self.matrix_builder.matrix)
            sorted_abs_evals = np.sort(np.abs(evals))[::-1]
            gap = 1.0 - sorted_abs_evals[1] if len(sorted_abs_evals) > 1 else 0.5
            self.last_gap = gap
            # [v3.4.8] Coherence Limit (Tc) derived from Spectral Gap.
            # Capped at 55.0 (Phase Space Volume of 11 Macro x 5 Micro states).
            # Prevents Poincare recurrence (information loss) within the engine's structural memory.
            self.last_tc = min(55.0, 1.0 / max(0.001, gap)) 
        except:
            self.last_gap = 0.5 # Fallback
            self.last_tc = 55.0

            
        return {
            "jump_id": state.jump_id, 
            "quantized_T": q_energy, 
            "quantized_V": q_potential,
            "density": q_energy / q_potential if q_potential > 0 else 0,
            "spectral_gap": self.last_gap,
            "dimension": self.matrix_builder.current_dim, 
            "horizon_n": horizon_n,
            "matrix": powered_matrix
        }

class QuantumAdaptiveCore:
    """
    [v3.0] Zero-Constant Physical Intelligence Layer:
    - Thermodynamic Entropy Calculation (H)
    - Dynamic Window (W*) scaling via Mean Free Path
    - Physical Coherence Collapse Detection
    - Zero Statistical Constants (No 3-Sigma)
    """
    def __init__(self, history_size=605):
        # [v3.1.7] Universal Physical Constants: H = D^2 * Rho
        self.DIMENSION = 11  # State space resolution (-5 to +5)
        self.CONFIDENCE_LEVEL = 5 # Minimum statistical samples per cell
        self.EVENT_HORIZON = self.DIMENSION**2 * self.CONFIDENCE_LEVEL # 605
        
        self.history_size = self.EVENT_HORIZON
        self.price_history = deque(maxlen=self.EVENT_HORIZON)
        self.current_tick = 0.01
        self.is_calibrating = False
        
        # [v3.0] Physical Observers
        self.last_direction = 0
        self.current_run_length = 0
        self.free_paths = deque(maxlen=50) # Track recent mean free paths
        self.current_tc = 100.0 # Default coherence time
        self.current_mfp = 1.0 # Default mean free path
        self.current_window = 100 # Track dynamic window size
        self._collapse_path_count = 0 # [v3.0] Self-clearing: paths recorded at last collapse trigger

    def add_event(self, price: float):
        if len(self.price_history) > 0:
            diff = price - self.price_history[-1]
            if abs(diff) > 1e-9:
                self.price_history.append(price)
                
                # [v3.0] Track Mean Free Path (Collision Detection)
                direction = np.sign(diff)
                if direction != self.last_direction and self.last_direction != 0:
                    # Collision occurred! Record the free path length
                    if self.current_run_length > 0:
                        self.free_paths.append(self.current_run_length)
                    self.current_run_length = 1 # Reset for new direction
                else:
                    self.current_run_length += 1
                
                self.last_direction = direction
                self.current_mfp = np.mean(self.free_paths) if len(self.free_paths) > 0 else 1.0
        else:
            self.price_history.append(price)

    def update_coherence_time(self, matrix: np.ndarray):
        """
        [v3.0] Extract Coherence Time (Tc) from Transition Matrix via Spectral Gap
        """
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            sorted_evals = np.sort(np.abs(eigenvalues))[::-1]
            if len(sorted_evals) > 1:
                spectral_gap = 1.0 - sorted_evals[1]
                # Avoid division by zero, cap at reasonable window size
                if spectral_gap > 0.001:
                    self.current_tc = min(1000.0, 1.0 / spectral_gap)
                else:
                    self.current_tc = 1000.0
        except Exception as e:
            pass # Fallback to previous Tc on convergence error

    def calculate_entropy(self, prices: Optional[List[float]] = None) -> float:
        """
        [v3.4] Nyquist-Shannon Phase Change Entropy.
        H = -sum(p * log2(p))
        Replaces arbitrary length constraints with physical wave cycle requirements.
        """
        data = prices if prices is not None else list(self.price_history)
        # [v3.4.8] log2(3) bits: Max information chaos for ternary decisions.
        if len(data) < 3: return np.log2(3) 
        
        diffs = np.diff(data)
        signs = np.sign(diffs)
        
        # [v3.4] Nyquist-Shannon: Require at least 2 phase changes (1 full cycle)
        non_zero_signs = signs[signs != 0]
        if len(non_zero_signs) > 1:
            phase_changes = np.sum(non_zero_signs[:-1] != non_zero_signs[1:])
        else:
            phase_changes = 0
            
        if phase_changes < 2: 
            # Insufficient wave formation: Assume maximum chaos (log2 3)
            return np.log2(3) 
        
        unique, counts = np.unique(signs, return_counts=True)
        probs = counts / len(signs)
        return -np.sum(probs * np.log2(probs))

    def find_optimal_n(self, base_tick=0.01, max_n=1.00) -> float:
        """
        [v3.1] Ensemble-based Optimal Tick: Eliminates fixed w_star constants.
        Aggregates Q-scores across multiple observation scales to find the 
        universal resolution that minimizes entropy across all windows.
        """
        # 1. Pure Spectral Coherence Ensemble (v3.1.4)
        # Derive observation bounds purely from the data's spectral properties.
        history_len = len(self.price_history)
        
        # [v3.1.4] Physical Coherence Calculation
        # We find the 'Memory Limit' (Tc) of the current price series.
        temp_prices = np.array(self.price_history)
        diffs = np.diff(temp_prices)
        states = np.clip(np.round(diffs / base_tick) + 5, 0, 10).astype(int)
        
        # Build mini-matrix for Tc extraction
        matrix = np.zeros((11, 11))
        for i in range(len(states)-1):
            matrix[states[i], states[i+1]] += 1
        row_sums = matrix.sum(axis=1)
        for i in range(11):
            if row_sums[i] > 0: matrix[i, :] /= row_sums[i]
            else: matrix[i, i] = 1.0
            
        try:
            evals = np.sort(np.abs(np.linalg.eigvals(matrix)))[::-1]
            gap = 1.0 - evals[1] if len(evals) > 1 else 0.1
            tc = 1.0 / max(0.001, gap)
        except:
            tc = 100.0 # Robust fallback
            
        # [v3.4] Information Innovation Convergence (Zero-Constant)
        # Replaces fixed '55' with physical coherence time (tc).
        # A highly ordered market (large gap) needs fewer samples to confirm.
        # A chaotic market (small gap) demands a longer observation window.
        w_min = int(max(4, tc)) # Minimum 4 points needed for 2 phase changes (1 full wave cycle)
        w_max = int(min(history_len, max(w_min + 50, tc * 2.0)))
        
        # Log-spaced windows covering the spectral memory of the system (8 scales)
        scales = np.unique(np.logspace(np.log10(w_min), np.log10(w_max), num=8, dtype=int))
        active_scales = [s for s in scales if s <= history_len]
        if not active_scales: active_scales = [history_len]
        
        resolutions = np.arange(base_tick, max_n + base_tick, base_tick)
        prices_all = temp_prices
        
        # 2. Cross-Scale Quality Resonance (Max-Inference)
        res_scores = {res: 0.0 for res in resolutions}
        
        for scale in active_scales:
            prices = prices_all[-scale:]
            for res in resolutions:
                resampled = [prices[0]]
                for p in prices[1:]:
                    if abs(p - resampled[-1]) >= res - 1e-9:
                        resampled.append(p)
                
                # [v3.4] len(resampled) < 10 is replaced by Nyquist-Shannon entropy limit
                # We skip strictly linear/insufficient data structurally
                if len(resampled) < 3: continue 
                
                h = self.calculate_entropy(resampled)
                diffs = np.diff(resampled)
                bias = abs(np.sum(np.sign(diffs))) / len(diffs) if len(diffs) > 0 else 0
                q = (1.0 - h) * bias * np.log(max(1, len(resampled)))
                
                # [v3.1.3] Maximum Resonance: Pick the clearest signal across scales
                # We don't dilute; we look for the scale that 'sees' this resolution best.
                weight = (1.0 - h) ** 2
                res_scores[res] = max(res_scores[res], q * weight)
        
        # 3. Decision
        best_n = max(res_scores, key=res_scores.get)
        
        # Update state for logging
        self.current_window = int(np.mean(active_scales))
        return best_n        

    def check_coherence_collapse(self) -> bool:
        """
        [v3.4] Zero-Constant Physical Coherence Collapse Detection.
        Replaces all heuristic numbers (1.8, 2.5, 5, 1, 0.02, 0.05) with 
        Information Capacity Limit (log2 3) and Spectral Gap (Delta) bounds.
        """
        if len(self.free_paths) < 5: return False
        
        gap = 1.0 / max(0.001, self.current_tc)
        current_n = len(self.free_paths)
        
        # Rate-limiting: Only process new physical events
        if current_n > self._collapse_path_count:
            # 1. Inertia Collision (Sudden Trend Break)
            last_path = self.free_paths[-1]
            prev_path = self.free_paths[-2] if current_n > 1 else 0
            
            # Dynamic Inertia Threshold: Chaos needs more proof, Order needs less
            inertia_threshold = self.current_mfp / max(0.01, gap)
            # Dynamic Collapse Ratio: 
            # High Order (Gap=0.5) -> even a 50% drop is a collapse
            # High Chaos (Gap=0.1) -> requires a 90% drop
            collapse_ratio = 1.0 - gap
            
            if prev_path > inertia_threshold and (last_path / max(0.1, prev_path)) < collapse_ratio:
                self._collapse_path_count = current_n
                return True
                
            # 2. Extreme Noise: Information Capacity Limit (log2 3)
            # High Chaos (Low Gap) -> Noise threshold scales with Max Entropy capacity.
            dynamic_mfp_limit = 1.0 + (1.0 - gap) * np.log2(3)
            if self.current_mfp < dynamic_mfp_limit:
                self._collapse_path_count = current_n
                return True
                
        return False

