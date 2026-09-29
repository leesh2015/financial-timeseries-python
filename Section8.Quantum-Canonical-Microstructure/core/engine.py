"""
Quantum Canonical Microstructure Engine (Unified Open-System Pipeline)
======================================================================
Zenodo DOI: 10.5281/zenodo.23042820
Integrates:
  1. PlanckLattice (h_f phase-space cell)
  2. CanonicalHamiltonianBuilder (H = T + V_eff & SU(2) U(dtau))
  3. ProperTimeClock (volume-clocked dtau & evolving tau)
  4. LindbladDissipator (asymmetric jump pumping & dephasing)
  5. GoldenWindowPhaseGater (von Neumann entropy collapse & sandwich gating)
"""

from typing import List, Tuple, Dict, Any
import numpy as np

from .planck_lattice import PlanckLattice
from .canonical_hamiltonian import CanonicalHamiltonianBuilder
from .lindblad_dissipation import LindbladDissipator, QuantumMicrostate
from .proper_time import ProperTimeClock, GoldenWindowPhaseGater


class QuantumCanonicalEngine:
    """
    Complete reference implementation of the Open Quantum System Microstructure Model (SSRN: 7533538).
    """
    def __init__(
        self,
        tick_size: float = 0.25,
        lot_size: float = 1.0,
        mass: float = 2.0,
        spatial_gamma: float = 0.5,
        friction_pts: float = 0.5625,
        initial_tau: float = 1.0
    ):
        self.lattice = PlanckLattice(tick_size=tick_size, lot_size=lot_size)
        self.hamiltonian_builder = CanonicalHamiltonianBuilder(
            mass=mass,
            spatial_gamma=spatial_gamma,
            friction_pts=friction_pts
        )
        self.proper_time_clock = ProperTimeClock(initial_tau=initial_tau)
        self.dissipator = LindbladDissipator()
        self.phase_gater = GoldenWindowPhaseGater()

        # Initialize density matrix as maximally mixed thermal ground state (P_up = P_down = 0.5)
        self.rho = np.array([
            [complex(0.5, 0.0), complex(0.0, 0.0)],
            [complex(0.0, 0.0), complex(0.5, 0.0)]
        ], dtype=complex)

        self.s_scale = 100.0  # Dynamic depth thickness barrier scale
        self.last_mid_price = 0.0

    def process_orderbook_snapshot(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        trade_buy_volume: float,
        trade_sell_volume: float,
        dt_physical_sec: float,
        utc_hour_float: float = 13.5
    ) -> Dict[str, Any]:
        """
        Executes a single infinitesimal proper-time transition step.
        """
        # 1. Quantize Order Book on Planck Lattice
        bids_q, asks_q, mid_price = self.lattice.quantize_order_book(bids, asks)
        if mid_price > 0:
            self.last_mid_price = mid_price

        # Delta volume in top levels
        delta_depth = 0.0
        if len(bids_q) and len(asks_q):
            delta_depth = float(abs(bids_q[0, 1] - asks_q[0, 1]))

        total_trade_volume = trade_buy_volume + trade_sell_volume

        # 2. Build Effective Hamiltonian
        H = self.hamiltonian_builder.build_hamiltonian(
            bids_q=bids_q,
            asks_q=asks_q,
            raw_k_buy=trade_buy_volume,
            raw_k_sell=trade_sell_volume,
            s_scale=self.s_scale,
            h_f=self.lattice.h_f
        )

        # 3. Clock Volume-Based Proper Time (dtau) and adapt tau
        pt_state = self.proper_time_clock.update_proper_time(
            dt_physical=dt_physical_sec,
            trade_volume=total_trade_volume,
            delta_depth_volume=delta_depth,
            m_eff=H.m_eff
        )

        # Update dynamic resting thickness scale (s_scale)
        alpha_s = float(np.exp(-pt_state.dtau / max(pt_state.tau_relax * H.m_eff, 1e-6)))
        depth_mean = float((H.v_norm_plus + H.v_norm_minus) * 0.5 * self.s_scale)
        self.s_scale = max(1e-4, alpha_s * self.s_scale + (1.0 - alpha_s) * depth_mean)

        # 4. Step A: Closed Unitary Transformation (Schrödinger Evolution)
        U = self.hamiltonian_builder.unitary_step(H, pt_state.dtau)
        rho_unitary = self.dissipator.apply_unitary(self.rho, U)

        # 5. Step B: Open-System Non-Unitary Dissipation (Lindblad Evolution)
        self.rho = self.dissipator.apply_dissipation(
            rho=rho_unitary,
            dtau=pt_state.dtau,
            tau_relax=pt_state.tau_relax,
            m_eff=H.m_eff,
            flow_up=trade_buy_volume,
            flow_down=trade_sell_volume
        )

        # 6. Extract Quantum Microstate Diagnostics
        microstate = self.dissipator.compute_state_diagnostics(self.rho)

        # 7. Evaluate Phase Gating & Entropy Drop
        is_entropy_drop, delta_s = self.phase_gater.evaluate_entropy_drop(microstate.entropy)
        is_window, window_desc = self.phase_gater.is_structural_window(utc_hour_float)

        return {
            "mid_price": self.last_mid_price,
            "dtau": pt_state.dtau,
            "tau_relax": pt_state.tau_relax,
            "H_omega": H.omega_raw,
            "H_delta": H.delta,
            "r_z": microstate.r_z,
            "r_x": microstate.r_x,
            "r_y": microstate.r_y,
            "purity": microstate.purity,
            "entropy": microstate.entropy,
            "info_purity": microstate.info_purity,
            "is_entropy_drop": is_entropy_drop,
            "delta_entropy": delta_s,
            "in_structural_window": is_window,
            "window_desc": window_desc
        }
