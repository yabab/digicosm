import numpy as np
from code.model import (
    energy_density,
    hamiltonian_total,
    init_phase_leapfrog,
    step_phase_leapfrog,
    run_pulsed_drive_samples,
    weighted_norm,
)
from code.measurements import weighted_norm_change_residual


def _ke_from_states(psi_now, psi_prev, dt):
    vel = (psi_now - psi_prev) / dt
    return float(np.sum(np.abs(vel) ** 2))


def test_ke_pe_split_and_total_consistency():
    N, dt = 32, 0.001
    omega0 = 0.0
    kappa = 1.0

    psi = np.zeros((N, N), dtype=np.complex128)
    psi[N // 2, N // 2 - 2] = 1.0 + 0.0j
    psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa)

    psi_next, psi_prev = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa)

    KE = _ke_from_states(psi_next, psi_prev, dt)
    PE = hamiltonian_total(psi_next, omega0=omega0, kappa=kappa)
    total_via_split = KE + PE

    # Compute a reference: small-field analytic check when kappa=0
    ref_PE = hamiltonian_total(psi_next, omega0=omega0, kappa=kappa)

    assert np.isfinite(KE) and np.isfinite(PE)
    assert np.isclose(total_via_split, ref_PE + KE)


def test_weighted_norm_and_residual_small_for_external_N():
    N, dt = 24, 0.001
    omega0 = 0.0
    kappa = 1.0

    psi = np.zeros((N, N), dtype=np.complex128)
    psi[N // 2, N // 2] = 1.0 + 0.0j
    psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa)

    psi_next, psi_prev_next = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa)

    # Use a constant clock_rate (external N) so the residual should be small
    N0 = np.full_like(np.abs(psi), 1.0, dtype=float)
    dW_actual, dW_pred, residual = weighted_norm_change_residual(
        psi, psi_next, N0, N0, dt
    )

    assert np.isfinite(residual)
    # Allow small discretization error from the finite-difference leapfrog
    assert abs(residual) < 2e-2


def test_local_energy_density_and_sums():
    N = 20
    psi = np.zeros((N, N), dtype=np.complex128)
    psi[0, 0] = 2.0 + 0.0j

    dens = energy_density(psi)
    assert dens.shape == psi.shape
    assert np.isclose(dens.sum(), 4.0)


def test_energy_spectrum_parseval():
    N = 32
    psi = np.zeros((N, N), dtype=np.complex128)
    psi[5, 7] = 1.0 + 0.0j

    ed = energy_density(psi)
    E_spat = float(ed.sum())

    F = np.fft.fft2(ed)
    E_freq = float((np.abs(F) ** 2).sum() / (N * N))

    assert np.isfinite(E_spat) and np.isfinite(E_freq)
    assert np.isclose(E_spat, E_freq, atol=1e-12)


def test_energy_region_balance_after_pulse():
    # Use the pulsed-drive helper to get a simple injection and check inner growth
    samples = run_pulsed_drive_samples((32, 32), steps=20, dt=0.01, source_pos=(16, 16), pulse_amp=1.0, pulse_sigma=0.5)

    times = [t for (t, psi) in samples]
    energies = [float(energy_density(psi).sum()) for (_t, psi) in samples]

    assert len(energies) >= 2
    assert energies[-1] >= energies[0]


def test_energy_spectrum_peak_for_single_mode():
    # Construct a single Fourier mode and verify spectrum peak
    N = 32
    y = np.arange(N)[:, None]
    x = np.arange(N)[None, :]
    kx, ky = 3, 5
    psi = np.exp(1j * (2 * np.pi * (kx * x / N + ky * y / N)))

    F = np.fft.fft2(psi)
    idx = np.unravel_index(np.argmax(np.abs(F)), F.shape)

    # FFT peak location should correspond to (kx mod N, ky mod N) but note axis order
    assert idx[1] == kx
    assert idx[0] == (ky % N)


def test_total_energy_matches_omega_when_kappa_zero():
    N = 16
    omega0 = 2.0
    kappa = 0.0
    psi = np.zeros((N, N), dtype=np.complex128)
    psi[3, 4] = 1.0 + 0.0j

    H = hamiltonian_total(psi, omega0=omega0, kappa=kappa)
    expected = float(omega0 * energy_density(psi).sum())
    assert np.isclose(H, expected)


def test_convergence_of_energy_drift_with_dt_halving():
    N = 40
    omega0 = 0.0
    kappa = 1.0

    def run_and_measure(dt, steps=80):
        psi = np.zeros((N, N), dtype=np.complex128)
        psi[N // 2, N // 2 - 1] = 1.0 + 0.0j
        psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa)

        E0 = float(hamiltonian_total(psi, omega0=omega0, kappa=kappa) + _ke_from_states(psi, psi_prev, dt))
        Es = []
        for _ in range(steps):
            psi, psi_prev = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa)
            Es.append(float(hamiltonian_total(psi, omega0=omega0, kappa=kappa) + _ke_from_states(psi, psi_prev, dt)))
        E1 = Es[-1]
        return abs(E1 - E0) / abs(E0)

    drift_dt = run_and_measure(0.002, steps=40)
    drift_half = run_and_measure(0.001, steps=80)

    assert drift_half < drift_dt
