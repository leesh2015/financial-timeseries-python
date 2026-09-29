"""
Lindblad Dissipation & Open Quantum Microstructure Dynamics (Zenodo: 10.5281/zenodo.23042820 Section 3 & 5.3)
=============================================================================================================
Executes non-unitary operator splitting:
    drho/dtau = -i[H, rho] + L_diss[rho]
Handles asymmetric population pumping and phase decoherence (Dephasing).
Projects state onto 3D Bloch sphere vector and von Neumann information purity.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class QuantumMicrostate:
    """
    Bloch Sphere & Density Matrix Microstate Representation.
    """
    rho: np.ndarray      # 2x2 complex Hermitian density matrix
    r_x: float           # 2 * Re(rho_01)
    r_y: float           # 2 * Im(rho_01)
    r_z: float           # rho_00 - rho_11 (polarization)
    purity: float        # ||r||_2 = sqrt(rx^2 + ry^2 + rz^2) in [0, 1]
    entropy: float       # S(rho) = -Tr(rho ln rho) in [0, ln(2)]
    info_purity: float   # P_info = 1 - S(rho)/ln(2) in [0, 1]


class LindbladDissipator:
    """
    Applies Lindblad jump dissipation, phase decoherence, and computes Bloch thermodynamics.
    """
    def __init__(self, dephase_coupling: float = 0.1):
        self.dephase_coupling = dephase_coupling

    def apply_unitary(self, rho: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Applies exact unitary rotation: rho -> U * rho * U^dagger."""
        u_dag = U.conj().T
        rho_rot = U @ rho @ u_dag
        # Enforce exact numerical Hermiticity & unit trace
        rho_rot = (rho_rot + rho_rot.conj().T) * 0.5
        tr = np.trace(rho_rot).real
        if tr > 0:
            rho_rot /= tr
        return rho_rot

    def apply_dissipation(
        self,
        rho: np.ndarray,
        dtau: float,
        tau_relax: float,
        m_eff: float,
        flow_up: float,
        flow_down: float
    ) -> np.ndarray:
        """
        Non-unitary population pumping and phase dephasing (Section 5.3).
        """
        rho_00 = rho[0, 0].real
        rho_11 = rho[1, 1].real
        rho_01 = rho[0, 1]

        # 1. Asymmetric Transition Rates (Pumping)
        denom = max(tau_relax * m_eff, 1e-6)
        gamma_up = max(flow_up, 0.0) / denom
        gamma_down = max(flow_down, 0.0) / denom
        sum_gamma = gamma_up + gamma_down

        if sum_gamma > 1e-12:
            p_trans = 1.0 - np.exp(-sum_gamma * dtau)
            p_up = (gamma_up / sum_gamma) * p_trans
            p_down = (gamma_down / sum_gamma) * p_trans
        else:
            p_up, p_down = 0.0, 0.0

        # Master equation population exchange
        new_rho_00 = rho_00 * (1.0 - p_down) + rho_11 * p_up
        new_rho_11 = rho_11 * (1.0 - p_up) + rho_00 * p_down

        # Trace normalization
        tot_tr = max(1e-12, new_rho_00 + new_rho_11)
        new_rho_00 = max(0.0, min(1.0, new_rho_00 / tot_tr))
        new_rho_11 = max(0.0, min(1.0, new_rho_11 / tot_tr))

        # 2. Phase Decoherence (Dephasing)
        dephase_rate = (1.0 / max(tau_relax, 1e-6)) + self.dephase_coupling * sum_gamma
        new_rho_01 = rho_01 * np.exp(-dephase_rate * dtau)

        # Cauchy-Schwarz physical boundary condition enforcement: |rho_01| <= sqrt(rho_00 * rho_11)
        c_lim = np.sqrt(new_rho_00 * new_rho_11)
        c_mag = abs(new_rho_01)
        if c_mag > c_lim > 0.0:
            new_rho_01 *= (c_lim / c_mag)

        new_rho = np.array([
            [complex(new_rho_00, 0.0), new_rho_01],
            [new_rho_01.conjugate(), complex(new_rho_11, 0.0)]
        ], dtype=complex)

        return new_rho

    def compute_state_diagnostics(self, rho: np.ndarray) -> QuantumMicrostate:
        """
        Maps rho onto 3D Bloch sphere vector and von Neumann entropy / information purity.
        """
        r_z = float((rho[0, 0] - rho[1, 1]).real)
        r_x = float(2.0 * rho[0, 1].real)
        r_y = float(2.0 * rho[0, 1].imag)

        purity = float(np.sqrt(r_x**2 + r_y**2 + r_z**2))
        purity_clamped = min(0.999999, purity)

        # Von Neumann Entropy: S(rho) = - sum lambda_i ln(lambda_i)
        lam1 = (1.0 + purity_clamped) * 0.5
        lam2 = (1.0 - purity_clamped) * 0.5
        entropy = - (lam1 * np.log(lam1) + (lam2 * np.log(lam2) if lam2 > 1e-12 else 0.0))
        ln2 = 0.6931471805599453
        info_purity = float(max(0.0, 1.0 - (entropy / ln2)))

        return QuantumMicrostate(
            rho=rho,
            r_x=r_x,
            r_y=r_y,
            r_z=r_z,
            purity=purity,
            entropy=float(entropy),
            info_purity=info_purity
        )
