from dataclasses import dataclass, field
from typing import List, Optional
import time

@dataclass
class KineticImpact:
    """Individual trade event that occurred while the price wall was maintained (Kinetic Energy Impact)"""
    offset_ms: int       # Elapsed time from the previous trade or state start (ms)
    volume: int          # Trade volume (Mass m)
    is_buy: bool         # True if buy trade, False if sell trade
    intensity: float     # Trade intensity

    def get_kinetic_energy(self) -> float:
        """T ∝ m * v^2 ∝ volume * (1 / offset_ms)^2 (Non-linear energy calculation)"""
        if self.offset_ms <= 0:
            return 0.0
        # Velocity v = 1 / Δt (Reciprocal of time)
        velocity = 1.0 / self.offset_ms
        return self.volume * (velocity ** 2)

@dataclass
class PotentialEnergyLevel:
    """Potential energy barrier at a specific price level"""
    price: float     
    volume: int      # Collapse critical energy (E_critical)
    is_virtual: bool # Whether it is an estimated virtual barrier
    is_ask: bool     # True for Ask wall, False for Bid wall

@dataclass
class MarketPotential:
    """Order book potential system (5x5 Standardized)"""
    asks: List[PotentialEnergyLevel] = field(default_factory=list)
    bids: List[PotentialEnergyLevel] = field(default_factory=list)
    
    @property
    def total_potential(self) -> int:
        return sum(p.volume for p in self.asks) + sum(p.volume for p in self.bids)

    @property
    def v_sum_5(self) -> int:
        """Sum of standard 5-level potential"""
        return sum(p.volume for p in self.asks[:5]) + sum(p.volume for p in self.bids[:5])

@dataclass
class QuantumJumpState:
    """Master snapshot at the moment of Quantum Jump"""
    jump_id: int                # Market-intrinsic causality time axis (n)
    symbol: str = "QQQ"
    market_type: str = "OVERSEAS_1" # 'DOMESTIC_10' or 'OVERSEAS_1'
    dimension: int = 5          # Matrix dimension (5 or 10)
    
    start_time: float = field(default_factory=time.time) # Unix time of state start
    end_time: Optional[float] = None                     # Unix time of collapse
    
    initial_potential: MarketPotential = field(default_factory=MarketPotential)
    impact_sequence: List[KineticImpact] = field(default_factory=list)
    
    accumulated_buy_vol: int = 0
    accumulated_sell_vol: int = 0

    @property
    def duration_ms(self) -> int:
        if self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return int((time.time() - self.start_time) * 1000)

    @property
    def total_kinetic_energy(self) -> float:
        """Accumulated non-linear kinetic energy T = Σ (vol / Δt)^2"""
        return sum(impact.get_kinetic_energy() for impact in self.impact_sequence)

    def check_collapse(self) -> bool:
        """Check if T > V_sum (Probability wave collapse trigger)"""
        return self.total_kinetic_energy > self.initial_potential.v_sum_5

    @property
    def field_contraction_rate(self) -> float:
        """Space contraction rate due to canceled orders (discrepancy between actual trade vol and vanished volume)"""
        # This logic requires comparison with real-time volume at collapse (handled by external engine)
        return 0.0

def calculate_physical_horizon(matrix, target_gain=4.0) -> int:
    """
    Derives the physical horizon (horizon_n) required to overcome market friction (target_gain)
    from the current transition matrix.
    
    Target_Gain = 4.0 (Ticks): [Spread(1) + Slippage(1~2) + Min Profit & Fees(1)]
    """
    # Assuming matrix is a 5x5 numpy array
    # [2, 0], [2, 1]: Probability of Upward jump / [2, 3], [2, 4]: Probability of Downward jump
    p_up = matrix[2, 0] + matrix[2, 1]
    p_down = matrix[2, 3] + matrix[2, 4]
    bias = abs(p_up - p_down)
    
    if bias > 0.0001:
        n = int(target_gain / bias)
    else:
        n = 10000 
        
    return max(50, n)
