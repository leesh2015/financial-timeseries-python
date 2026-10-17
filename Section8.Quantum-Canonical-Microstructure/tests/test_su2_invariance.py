"""
Numerical Checks: SU(2) Invariance and State-Boundary Guards
"""

# import pytest
import numpy as np
from core.canonical_hamiltonian import CanonicalHamiltonianBuilder, MicrostructureHamiltonian
from core.lindblad_dissipation import LindbladDissipator
from core.planck_lattice import PlanckLattice
from core.proper_time import GoldenWindowPhaseGater, ProperTimeClock


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
    Finite numerical checks of unitary invariance and positivity for selected inputs:
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
        omega_raw = float(np.hypot(delta, kinetic_impulse))
        H = MicrostructureHamiltonian(
            v_norm_plus=delta + 1.0,
            v_norm_minus=1.0,
            delta=delta,
            kinetic_impulse=kinetic_impulse,
            omega_raw=omega_raw,
            m_eff=m_eff
        )

        for dtau in [0.0, 0.001, 0.05, 0.5, 2.0, 10.0]:
            U = builder.unitary_step(H, dtau)
            if dtau == 0.0 or omega_raw == 0.0:
                assert np.array_equal(U, np.eye(2, dtype=complex))

            # 1. Numerical SU(2) determinant check for this finite input.
            det_val = np.linalg.det(U)
            assert np.isclose(det_val.real, 1.0, atol=1e-14), f"Re(det(U)) not 1: {det_val}"
            assert np.isclose(det_val.imag, 0.0, atol=1e-14), f"Im(det(U)) not 0: {det_val}"

            # 2. Numerical unitarity check.
            u_u_dag = U @ U.conj().T
            assert np.allclose(u_u_dag, np.eye(2), atol=1e-14), "U is not strictly unitary"

            # 3. Density matrix trace conservation: Tr(U * rho * U_dag) == 1
            test_rho = np.array([[0.7, 0.2 - 0.1j], [0.2 + 0.1j, 0.3]], dtype=complex)
            rho_rot = U @ test_rho @ U.conj().T
            rho_processed = dissipator.apply_unitary(test_rho, U)
            assert np.allclose(rho_processed, rho_rot, atol=1e-14)

            tr = np.trace(rho_rot)
            assert np.isclose(tr.real, 1.0, atol=1e-14), f"Trace not preserved: {tr}"
            assert np.isclose(tr.imag, 0.0, atol=1e-14)

            # 4. Hermiticity: rho_rot == rho_rot^dagger
            assert np.allclose(rho_rot, rho_rot.conj().T, atol=1e-14), "Hermiticity broken"

            # 5. Positivity: eigenvalues >= 0
            evals = np.linalg.eigvalsh(rho_rot)
            assert np.all(evals >= -1e-14), f"Negative eigenvalues found: {evals}"


def test_small_nonzero_coupling_is_not_treated_as_zero():
    builder = CanonicalHamiltonianBuilder(mass=1.0)
    H = MicrostructureHamiltonian(0.0, 0.0, 1e-16, 0.0, 1e-16, 1.0)
    U = builder.unitary_step(H, 1e16)
    expected = np.array([
        [np.cos(1.0) - 1j * np.sin(1.0), 0.0j],
        [0.0j, np.cos(1.0) + 1j * np.sin(1.0)]
    ])
    assert np.allclose(U, expected, atol=1e-14)


def test_builder_uses_full_potential_difference():
    builder = CanonicalHamiltonianBuilder(mass=2.0, friction_pts=0.0)
    bids = np.array([[0.0, 2.0]])
    asks = np.array([[0.0, 4.0]])
    H = builder.build_hamiltonian(bids, asks, 0.0, 0.0, s_scale=2.0)
    assert H.delta == 1.0


def test_cauchy_schwarz_boundary_defense():
    """Checks projection for a small boundary violation and the zero-bound edge case."""
    dissipator = LindbladDissipator()
    # A slight boundary violation isolates the numerical projection.
    rho = np.array([[0.1, 0.3 + 1e-8], [0.3 + 1e-8, 0.9]], dtype=complex)
    projected = dissipator.apply_dissipation(
        rho=rho, dtau=0.0, tau_relax=0.5, m_eff=5.0,
        flow_up=0.0, flow_down=0.0
    )
    c_lim = np.sqrt(projected[0, 0].real * projected[1, 1].real)
    assert abs(projected[0, 1]) <= c_lim + 1e-14

    # Residual coherence at a zero-population boundary must be projected to zero.
    rho_pure_boundary = np.array([[1.0, 1e-12], [1e-12, 0.0]], dtype=complex)
    clipped = dissipator.apply_dissipation(
        rho=rho_pure_boundary, dtau=0.1, tau_relax=0.5, m_eff=5.0,
        flow_up=0.0, flow_down=0.0
    )
    assert clipped[0, 1] == 0.0j


def test_entropy_gate_separates_step_drop_from_baseline_deficit():
    gater = GoldenWindowPhaseGater(entropy_drop_threshold=0.25)
    baseline = np.log(2.0)

    is_drop, step_delta, deficit = gater.evaluate_entropy_drop(baseline - 0.4)
    assert is_drop
    assert np.isclose(step_delta, -0.4)
    assert np.isclose(deficit, -0.4)

    # Entropy remains below ln(2), but it is recovering rather than dropping.
    is_drop, step_delta, deficit = gater.evaluate_entropy_drop(baseline - 0.3)
    assert not is_drop
    assert np.isclose(step_delta, 0.1)
    assert np.isclose(deficit, -0.3)


def test_proper_time_clipping():
    """Verifies that dtau bounds [0.05, 20.0] prevent sampling singularity."""
    clock = ProperTimeClock()
    # Extremely massive shock
    pt = clock.update_proper_time(dt_physical=1.0, trade_volume=1e6, delta_depth_volume=0.0, m_eff=10.0)
    assert pt.dtau <= 20.0, f"dtau exceeded upper bound: {pt.dtau}"

    # Absolute zero activity
    pt_zero = clock.update_proper_time(dt_physical=1.0, trade_volume=0.0, delta_depth_volume=0.0, m_eff=10.0)
    assert pt_zero.dtau >= 0.05, f"dtau dropped below lower bound: {pt_zero.dtau}"

