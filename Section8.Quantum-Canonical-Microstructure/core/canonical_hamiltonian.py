"""
Canonical Hamiltonian Dynamics (Zenodo: 10.5281/zenodo.23042820 Section 2 & 5.2)
================================================================================
Constructs the 2-level effective microstructure Hamiltonian:
    H_eff = T + V_eff
and performs exact closed-form SU(2) unitary operator splitting:
    U(dtau) = exp(-i * H_eff * dtau / M_eff)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class MicrostructureHamiltonian:
    """
    Effective Microstructure Hamiltonian:
        H = [[ V_norm^+,          i Delta ],
             [ i Delta,          V_norm^- ]]
    where delta = V_norm^+ - V_norm^- is the potential difference,
    and Delta is the friction-discounted kinetic momentum.
    """
    v_norm_plus: float    # Ask/Up potential energy
    v_norm_minus: float   # Bid/Down potential energy
    delta: float          # Potential difference (v_plus - v_minus)
    kinetic_impulse: float# Friction-gated net momentum (Delta)
    omega_raw: float      # Coupled frequency: sqrt(delta^2 + Delta^2 + epsilon)
    m_eff: float          # Macroscopic logarithmic inertia barrier


class CanonicalHamiltonianBuilder:
    """
    Maps multi-level order book liquidity and aggressive executions into the Hamiltonian operator.
    """
    def __init__(self, mass: float = 2.0, spatial_gamma: float = 0.5, friction_pts: float = 0.5625):
        self.mass = max(mass, 0.1)
        self.spatial_gamma = spatial_gamma
        self.friction_pts = friction_pts

    def compute_virtual_potential(self, levels_quantized: np.ndarray) -> float:
        """
        Projects discrete resting depth into continuous potential well (Section 2.1):
            V_eff = V_L1 + sum_{j=2}^K V_j * exp(-gamma * |x - x_0|)
        """
        if len(levels_quantized) == 0:
            return 1e-4

        v_l1 = levels_quantized[0, 1]
        if len(levels_quantized) == 1:
            return float(v_l1)

        distances = levels_quantized[1:, 0]
        depths = levels_quantized[1:, 1]
        v_tail = np.sum(depths * np.exp(-self.spatial_gamma * distances))
        return float(v_l1 + v_tail)

    def build_hamiltonian(
        self,
        bids_q: np.ndarray,
        asks_q: np.ndarray,
        raw_k_buy: float,
        raw_k_sell: float,
        s_scale: float,
        h_f: float = 0.25
    ) -> MicrostructureHamiltonian:
        """
        Constructs the instantaneous 2-level Hamiltonian.
        """
        v_int_up = self.compute_virtual_potential(asks_q)
        v_int_down = self.compute_virtual_potential(bids_q)

        scale = max(s_scale, 1e-4)
        v_norm_plus = v_int_up / scale
        v_norm_minus = v_int_down / scale
        delta = v_norm_plus - v_norm_minus

        # Friction-gated net kinetic impulse: Delta
        delta_raw = (raw_k_buy - raw_k_sell) / max(1.0, scale)
        friction_barrier = self.friction_pts / scale
        kinetic_impulse = float(np.sign(delta_raw) * max(0.0, abs(delta_raw) - friction_barrier))

        omega_raw = float(np.sqrt(delta**2 + kinetic_impulse**2 + 1e-8))

        # Macroscopic logarithmic inertia scaling (Section 2.1 & 5.1)
        v_sum = max(np.sum(bids_q[:, 1]) + np.sum(asks_q[:, 1]) if len(bids_q) and len(asks_q) else 10.0, 10.0)
        m_eff = self.mass * max(1.0, float(np.log(v_sum / max(h_f, 1e-4))))

        return MicrostructureHamiltonian(
            v_norm_plus=v_norm_plus,
            v_norm_minus=v_norm_minus,
            delta=delta,
            kinetic_impulse=kinetic_impulse,
            omega_raw=omega_raw,
            m_eff=m_eff
        )

    def unitary_step(self, H: MicrostructureHamiltonian, dtau: float) -> np.ndarray:
        """
        Closed analytic SU(2) unitary transformation matrix U(dtau) = exp(-i * H * dtau / M_eff).
        Guaranteed: det(U) = 1 and Tr(U * rho * U_dag) = 1 (Theorem 1).
        """
        omega = H.omega_raw / H.m_eff
        theta = omega * dtau

        c = float(np.cos(theta))
        s = float(np.sin(theta))
        u_delta = (H.delta / H.omega_raw) * s
        u_momentum = (H.kinetic_impulse / H.omega_raw) * s

        # 2x2 complex unitary matrix
        u00 = complex(c, -u_delta)
        u01 = complex(0.0, u_momentum)
        u10 = complex(0.0, u_momentum)
        u11 = complex(c, u_delta)

        return np.array([[u00, u01], [u10, u11]], dtype=complex)
