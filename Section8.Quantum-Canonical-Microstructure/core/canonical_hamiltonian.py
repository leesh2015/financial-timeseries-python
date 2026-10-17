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
    Effective Hermitian Microstructure Hamiltonian (Pauli basis H = delta * sigma_z - Delta * sigma_x):
        H = [[  delta, -Delta ],
             [ -Delta, -delta ]]
    where delta = (V_norm^+ - V_norm^-) / 2 is the longitudinal order book barrier asymmetry,
    and Delta is the friction-discounted transverse kinetic momentum.
    Under U(dtau) = exp(-i * H * dtau / M_eff), the off-diagonal unitary tunneling terms
    evaluate to -i * (-Delta / Omega_raw) * sin(theta) = +i * (Delta / Omega_raw) * sin(theta).
    """
    v_norm_plus: float    # Ask/Up potential energy
    v_norm_minus: float   # Bid/Down potential energy
    delta: float          # Half-difference of normalized potentials
    kinetic_impulse: float# Friction-gated net momentum (Delta)
    omega_raw: float      # Exact Euclidean norm: sqrt(delta^2 + Delta^2)
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
        Constructs the instantaneous 2-level Hermitian Hamiltonian.
        """
        v_int_up = self.compute_virtual_potential(asks_q)
        v_int_down = self.compute_virtual_potential(bids_q)

        scale = max(s_scale, 1e-4)
        v_norm_plus = v_int_up / scale
        v_norm_minus = v_int_down / scale
        delta = 0.5 * (v_norm_plus - v_norm_minus)

        # Friction-gated net kinetic impulse: Delta
        delta_raw = (raw_k_buy - raw_k_sell) / max(1.0, scale)
        friction_barrier = self.friction_pts / scale
        kinetic_impulse = float(np.sign(delta_raw) * max(0.0, abs(delta_raw) - friction_barrier))

        # Exact Euclidean norm without artificial epsilon bias so (delta/omega_raw)^2 + (Delta/omega_raw)^2 == 1
        omega_raw = float(np.hypot(delta, kinetic_impulse))

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
        In exact arithmetic this is an SU(2) matrix; finite floating-point inputs produce
        unitarity and determinant equal to their mathematical values up to rounding error.
        """
        omega_norm = float(np.hypot(H.delta, H.kinetic_impulse))
        if omega_norm == 0.0 or dtau == 0.0:
            return np.eye(2, dtype=complex)

        omega = omega_norm / H.m_eff
        theta = omega * dtau

        c = float(np.cos(theta))
        s = float(np.sin(theta))
        n_z = H.delta / omega_norm
        n_x = H.kinetic_impulse / omega_norm

        u_delta = n_z * s
        u_momentum = n_x * s

        # 2x2 complex SU(2) unitary matrix: c*I - i*s*(n_z * sigma_z - n_x * sigma_x)
        u00 = complex(c, -u_delta)
        u01 = complex(0.0, u_momentum)
        u10 = complex(0.0, u_momentum)
        u11 = complex(c, u_delta)

        return np.array([[u00, u01], [u10, u11]], dtype=complex)
