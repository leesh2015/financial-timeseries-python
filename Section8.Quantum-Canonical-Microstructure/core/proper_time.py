"""
Proper-Time & Adaptive Dwell Relaxation Dynamics (Zenodo: 10.5281/zenodo.23042820 Section 3.2, 4, 5.1)
=======================================================================================================
1. Volume-Clocked Proper-Time Evolution:
       dtau = dt * min(max(0.05, Phi_flux / Phi_mean), 20.0)
2. Self-Evolving Dwell Scale:
       tau(k+1) = beta * tau(k) + (1 - beta) * dtau
3. Golden Window Canonical Phase Gating & Entropy Drop Detection.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class ProperTimeState:
    """Tracks intrinsic dwell time and baseline flux."""
    dtau: float          # Volume-clocked elapsed proper time
    tau_relax: float     # Self-evolving characteristic relaxation timescale
    mean_flux: float     # Adaptive baseline information flux (EMA)
    flux_ratio: float    # Instantaneous relative shock intensity


class ProperTimeClock:
    """
    Eliminates temporal sampling artifacts by clocking proper time along information flux geodesics.
    """
    def __init__(self, initial_tau: float = 1.0, eta_flux: float = 1e-3):
        self.tau = max(initial_tau, 0.05)
        self.mean_flux = 1.0
        self.eta_flux = eta_flux

    def update_proper_time(
        self,
        dt_physical: float,
        trade_volume: float,
        delta_depth_volume: float,
        m_eff: float
    ) -> ProperTimeState:
        """
        Calculates dtau and evolves the intrinsic relaxation timescale tau.
        """
        dt_safe = max(dt_physical, 1e-6)
        inst_flux = float(abs(trade_volume) + abs(delta_depth_volume))

        # Update baseline flux (EMA)
        self.mean_flux = (1.0 - self.eta_flux) * self.mean_flux + self.eta_flux * max(inst_flux, 1e-4)
        flux_ratio = inst_flux / max(self.mean_flux, 1e-4)

        # Volume-clocked proper-time increment bounded in [0.05, 20.0]
        speed_factor = float(np.clip(flux_ratio, 0.05, 20.0))
        dtau = float(dt_safe * speed_factor)

        # Self-evolving timescale update along the proper-time geodesic (Section 5.1)
        decay = float(np.exp(-dtau / max(self.tau * m_eff, 1e-6)))
        self.tau = float(decay * self.tau + (1.0 - decay) * dtau)
        self.tau = max(self.tau, 1e-4)

        return ProperTimeState(
            dtau=dtau,
            tau_relax=self.tau,
            mean_flux=self.mean_flux,
            flux_ratio=flux_ratio
        )


class GoldenWindowPhaseGater:
    """
    Canonical Entropy Drop & Structural Window Gater (Section 4 & 6.2).
    Mathematically isolates periods of localized self-organization.
    """
    def __init__(self, entropy_drop_threshold: float = 0.25):
        self.entropy_drop_threshold = entropy_drop_threshold
        self.baseline_entropy = 0.693147  # ln(2) for maximum mixed state

    def evaluate_entropy_drop(self, current_entropy: float) -> Tuple[bool, float]:
        """
        Checks if Delta S = S(tau) - S_baseline satisfies Delta S < -epsilon_S.
        """
        delta_s = current_entropy - self.baseline_entropy
        is_coherent = delta_s < -self.entropy_drop_threshold
        return is_coherent, float(delta_s)

    @staticmethod
    def is_structural_window(utc_hour_float: float) -> Tuple[bool, str]:
        """
        Window-Gated Canonical (Sandwich) regime check (Section 6.2 footnote):
        1. Asia Golden: UTC 01:00 ~ 02:00
        2. US Pre-Open: UTC 13:15 ~ 13:30
        3. US Post-Chaos: UTC 13:45 ~ 14:00 (Avoiding 13:30 ~ 13:45 storm)
        """
        if 1.0 <= utc_hour_float < 2.0:
            return True, "Asia Golden (Liquidity Overlap)"
        if 13.25 <= utc_hour_float < 13.50:
            return True, "US Pre-Open (Anticipation Window)"
        if 13.75 <= utc_hour_float < 14.00:
            return True, "US Post-Chaos (Re-equilibration Window)"
        return False, "Unrestricted / Non-Gated"
