from dataclasses import dataclass, field
from typing import List, Optional
import time
import numpy as np

@dataclass
class KineticImpact:
    """Individual trade event occurred while the wall was maintained (Kinetic Energy Impact)"""
    offset_ms: int       
    volume: int          
    is_buy: bool         
    intensity: float     
    offset_ns: Optional[int] = None # High-precision offset (ns)

    def get_kinetic_energy(self) -> float:
        """T ∝ m * v^2 | Standardized Velocity: 1ms = 1.0"""
        if self.offset_ns is not None and self.offset_ns > 0:
            # 1ms (1,000,000ns) -> v = 1.0
            velocity = 1000000.0 / self.offset_ns
            return self.volume * (velocity ** 2)
            
        dt_ms = max(1, self.offset_ms)
        # 1ms -> v = 1.0
        velocity = 1.0 / dt_ms
        return self.volume * (velocity ** 2)

@dataclass
class PotentialEnergyLevel:
    """Potential energy barrier at a specific price level"""
    price: float     
    volume: int      
    is_virtual: bool 
    is_ask: bool     

@dataclass
class MarketPotential:
    """Order book potential system (5x5 standardization)"""
    asks: List[PotentialEnergyLevel] = field(default_factory=list)
    bids: List[PotentialEnergyLevel] = field(default_factory=list)
    
    @property
    def total_potential(self) -> int:
        return sum(p.volume for p in self.asks) + sum(p.volume for p in self.bids)

    @property
    def v_sum_5(self) -> int:
        return sum(p.volume for p in self.asks[:5]) + sum(p.volume for p in self.bids[:5])

@dataclass
class QuantumState:
    """Master snapshot of a market event (v2.4 Unified Standard)"""
    jump_id: int                
    symbol: str = "TQQQ"
    
    # Price & L1 Volume
    bid1: float = 0.0
    ask1: float = 0.0
    bid_vol1: int = 0
    ask_vol1: int = 0
    
    # Trading Activity
    buy_vol: int = 0
    sell_vol: int = 0
    duration_ms: int = 1
    
    # [Tier 2+] Depth Data
    bid_vols: List[int] = field(default_factory=list)
    ask_vols: List[int] = field(default_factory=list)

    # Physical Context
    server_ts: int = 0
    arrival_ns: Optional[int] = None 
    velocity_only: bool = False      
    dimension: int = 5          
    
    initial_potential: MarketPotential = field(default_factory=MarketPotential)
    impact_sequence: List[KineticImpact] = field(default_factory=list)
    
    @property
    def total_kinetic_energy(self) -> float:
        """T ∝ sum(m * v^2)"""
        if self.velocity_only:
            total_vol = self.buy_vol + self.sell_vol
            if total_vol <= 0 or self.duration_ms <= 0: return 0.0
            # [Standardized] 1ms is base velocity unit
            velocity = 1.0 / self.duration_ms
            return total_vol * (velocity ** 2)
            
        return sum(impact.get_kinetic_energy() for impact in self.impact_sequence)

def calculate_physical_horizon(matrix, target_gain=4.0) -> float:
    """
    Demon Engine v2.0+: High-Precision Physical Horizon Derivation
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
        
    return n
