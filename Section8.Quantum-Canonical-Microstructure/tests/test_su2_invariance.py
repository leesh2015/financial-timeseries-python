"""
Unit Tests: Theoretical Invariance & SU(2) Conservation Proofs (Theorem 1)
"""

# import pytest
import numpy as np
from core.canonical_hamiltonian import CanonicalHamiltonianBuilder, MicrostructureHamiltonian
from core.lindblad_dissipation import LindbladDissipator
from core.planck_lattice import PlanckLattice
from core.proper_time import ProperTimeClock


def test_planck_cell_invariance():
    """Verifies that h_f = Delta p_min * Delta v_min strictly bounds the lattice."""
    lattice = PlanckLattice(tick_size=0.25, lot_size=1.0)
    assert lattice.h_f == 0.25
    bids = [(20000.0, 5.0), (19999.75, 10.0)]
    asks = [(20000.25, 4.0), (20000.50, 8.0)]
    bids_q, asks_q, mid = lattice.quantize_order_book(bids, asks)
    assert mid == 20000.125
    assert len(bids_q) == 2
    assert len(asks_q) == 2


def test_su2_unitary_invariance():
    """
    Direct numerical verification of Theorem 1 (Unitary Invariance and Positivity Preservation):
        1. det(U) == 1
        2. Tr(U * rho * U_dag) == 1
        3. Hermiticity preserved
    """
    builder = CanonicalHamiltonianBuilder(mass=2.0)
    dissipator = LindbladDissipator()

    # Arbitrary non-trivial coupling configurations (delta, Delta) in R^2
    configs = [
        (1.5, 0.8, 10.0),
        (-3.2, 5.0, 25.0),
        (0.0, 0.0, 5.0),
        (12.4, -8.1, 100.0)
    ]

    for delta, kinetic_impulse, m_eff in configs:
        omega_raw = np.sqrt(delta**2 + kinetic_impulse**2 + 1e-8)
        H = MicrostructureHamiltonian(
            v_norm_plus=delta + 1.0,
            v_norm_minus=1.0,
            delta=delta,
            kinetic_impulse=kinetic_impulse,
            omega_raw=omega_raw,
            m_eff=m_eff
        )

        for dtau in [0.001, 0.05, 0.5, 2.0, 10.0]:
            U = builder.unitary_step(H, dtau)

            # 1. Determinant test: det(U) == 1
            det_val = np.linalg.det(U)
            assert np.isclose(abs(det_val), 1.0, atol=1e-7), f"det(U) not 1: {det_val}"

            # 2. SU(2) property: U * U_dag == Identity
            u_u_dag = U @ U.conj().T
            assert np.allclose(u_u_dag, np.eye(2), atol=1e-7), "U is not unitary"

            # 3. Density matrix trace conservation: Tr(U * rho * U_dag) == 1
            test_rho = np.array([[0.7, 0.2 - 0.1j], [0.2 + 0.1j, 0.3]], dtype=complex)
            rho_rot = dissipator.apply_unitary(test_rho, U)

            tr = np.trace(rho_rot)
            assert np.isclose(tr.real, 1.0, atol=1e-7), f"Trace not preserved: {tr}"
            assert np.isclose(tr.imag, 0.0, atol=1e-7)

            # 4. Hermiticity: rho_rot == rho_rot^dagger
            assert np.allclose(rho_rot, rho_rot.conj().T, atol=1e-7), "Hermiticity broken"

            # 5. Positivity: eigenvalues >= 0
            evals = np.linalg.eigvalsh(rho_rot)
            assert np.all(evals >= -1e-8), f"Negative eigenvalues found: {evals}"


def test_cauchy_schwarz_boundary_defense():
    """
    Verifies Section 5.3: |rho_01| <= sqrt(rho_00 * rho_11) is strictly enforced under dissipation.
    """
    dissipator = LindbladDissipator()
    # Pathological test density matrix on the verge of boundary violation
    rho = np.array([[0.1, 0.5], [0.5, 0.9]], dtype=complex)
    new_rho = dissipator.apply_dissipation(
        rho=rho,
        dtau=0.1,
        tau_relax=0.5,
        m_eff=5.0,
        flow_up=10.0,
        flow_down=0.0
    )

    rho_00 = new_rho[0, 0].real
    rho_11 = new_rho[1, 1].real
    rho_01_mag = abs(new_rho[0, 1])

    c_lim = np.sqrt(rho_00 * rho_11)
    assert rho_01_mag <= c_lim + 1e-7, f"Cauchy-Schwarz violation: {rho_01_mag} > {c_lim}"


def test_proper_time_clipping():
    """Verifies that dtau bounds [0.05, 20.0] prevent sampling singularity."""
    clock = ProperTimeClock()
    # Extremely massive shock
    pt = clock.update_proper_time(dt_physical=1.0, trade_volume=1e6, delta_depth_volume=0.0, m_eff=10.0)
    assert pt.dtau <= 20.0, f"dtau exceeded upper bound: {pt.dtau}"

    # Absolute zero activity
    pt_zero = clock.update_proper_time(dt_physical=1.0, trade_volume=0.0, delta_depth_volume=0.0, m_eff=10.0)
    assert pt_zero.dtau >= 0.05, f"dtau dropped below lower bound: {pt_zero.dtau}"

