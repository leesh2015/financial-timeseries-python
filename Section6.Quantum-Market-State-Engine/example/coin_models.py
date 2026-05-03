from dataclasses import dataclass, field
from typing import List, Optional
import time
import numpy as np

@dataclass
class KineticImpact:
    """Trade event for BTC. High-precision v2.3 spec."""
    offset_ms: int       
    volume: float        
    is_buy: bool         
    p: float             
    offset_ns: Optional[int] = None # High-precision ns offset

    def get_kinetic_energy(self) -> float:
        """T = m * (1/Δt)^2"""
        # Prioritize Nanosecond precision
        if self.offset_ns is not None and self.offset_ns > 0:
            velocity = 1000000.0 / self.offset_ns
            return self.volume * (velocity ** 2)
            
        # Fallback: Treat 0ms as 1ms (minimum resolution) to preserve high-frequency energy
        dt_ms = max(1, self.offset_ms)
        velocity = 1000.0 / dt_ms
        return self.volume * (velocity ** 2)

@dataclass
class PotentialEnergyLevel:
    """Potential barrier at a specific BTC price level"""
    price: float     
    volume: float      
    is_ask: bool     

@dataclass
class MarketPotential:
    """20-level order book potential system for BTC"""
    asks: List[PotentialEnergyLevel] = field(default_factory=list)
    bids: List[PotentialEnergyLevel] = field(default_factory=list)
    
    @property
    def v_sum_5(self) -> float:
        return sum(p.volume for p in self.asks[:5]) + sum(p.volume for p in self.bids[:5])

@dataclass
class CoinJumpState:
    """Master snapshot of a BTC Quantum Jump (v2.3 Spec)"""
    jump_id: int
    symbol: str = "btcusdt"
    dimension: int = 5
    
    bid1: float = 0.0
    ask1: float = 0.0
    
    start_time: float = field(default_factory=time.perf_counter) 
    server_ts: int = 0
    duration_ms: int = 1 # Historical duration
    arrival_ns: Optional[int] = None
    velocity_only: bool = False
    
    initial_potential: MarketPotential = field(default_factory=MarketPotential)
    impact_sequence: List[KineticImpact] = field(default_factory=list)
    
    @property
    def total_kinetic_energy(self) -> float:
        if self.velocity_only:
            total_vol = sum(imp.volume for imp in self.impact_sequence)
            if total_vol <= 0 or self.duration_ms <= 0: return 0.0
            velocity = 1000.0 / self.duration_ms
            return total_vol * (velocity ** 2)
            
        return sum(impact.get_kinetic_energy() for impact in self.impact_sequence)

def calculate_physical_horizon(matrix, target_gain=6.0) -> float:
    """
    Demon Engine v2.0: High-Precision Physical Horizon Derivation
    Returns float N required to overcome market friction.
    """
    dim = matrix.shape[0]
    mid = dim // 2
    row = matrix[mid]
    
    if dim % 2 == 0:
        # Even (10x10): Perfectly split into 5:5 halves
        p_up = np.sum(row[:mid])
        p_down = np.sum(row[mid:])
    else:
        # Odd (5x5): Symmetric split skipping the center cell
        p_up = np.sum(row[:mid])
        p_down = np.sum(row[mid+1:])
    
    bias = abs(p_up - p_down)
    
    if bias > 0.0001:
        n = target_gain / bias
    else:
        n = 10000 
    
    return float(max(10, n))
