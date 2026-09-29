"""
Planck Lattice Discretization (Zenodo: 10.5281/zenodo.23042820 Section 1.1)
===========================================================================
Defines the invariant Financial Planck Constant:
    h_f = Delta p_min * Delta v_min
and maps continuous order book price-volume tuples onto a quantized phase-space cell.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class FinancialPlanckCell:
    """
    Fundamental phase-space cell bounded from below by h_f.
    """
    tick_size: float   # Delta p_min (e.g. 0.25 pt for MNQ)
    lot_size: float    # Delta v_min (e.g. 1.0 contract)
    
    @property
    def h_f(self) -> float:
        """The Financial Planck Constant: h_f = Delta p_min * Delta v_min"""
        return self.tick_size * self.lot_size


class PlanckLattice:
    """
    Quantizes price levels and order volume increments onto the discrete financial lattice.
    """
    def __init__(self, tick_size: float = 0.25, lot_size: float = 1.0):
        self.cell = FinancialPlanckCell(tick_size=tick_size, lot_size=lot_size)

    @property
    def h_f(self) -> float:
        return self.cell.h_f

    def quantize_price(self, price: float) -> int:
        """Converts continuous floating price into discrete tick index."""
        return int(np.round(price / self.cell.tick_size))

    def price_to_level(self, price: float, reference_mid: float) -> float:
        """Returns signed integer distance from reference mid-price."""
        return (price - reference_mid) / self.cell.tick_size

    def quantize_order_book(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Quantizes top-K LOB into structured arrays with guaranteed non-zero h_f bounds.
        Returns:
            bids_quantized: (K, 2) array [tick_offset, normalized_volume]
            asks_quantized: (K, 2) array [tick_offset, normalized_volume]
            mid_price: scalar reference
        """
        if not bids or not asks:
            return np.empty((0, 2)), np.empty((0, 2)), 0.0

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2.0

        bids_q = np.array([
            [abs(self.price_to_level(p, mid_price)), max(v / self.cell.lot_size, 1.0)]
            for p, v in bids
        ], dtype=float)

        asks_q = np.array([
            [abs(self.price_to_level(p, mid_price)), max(v / self.cell.lot_size, 1.0)]
            for p, v in asks
        ], dtype=float)

        return bids_q, asks_q, mid_price
