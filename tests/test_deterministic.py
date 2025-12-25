import numpy as np
from code.model import (
    init_phase_leapfrog,
    step_phase_leapfrog,
    run_pulsed_drive_samples,
    laplacian_iso,
    hamiltonian_total,
)


def test_reproducible_random_initial_conditions():
    seed = 12345
    dt = 0.001
    omega0 = 0.0
    kappa = 1.0

    rng = np.random.default_rng(seed)
    psi_a = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    psi_prev_a = init_phase_leapfrog(psi_a, dt, omega0, kappa)
    psi_next_a, psi_prev_a = step_phase_leapfrog(psi_a, psi_prev_a, dt, omega0, kappa)

    rng = np.random.default_rng(seed)
    psi_b = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    psi_prev_b = init_phase_leapfrog(psi_b, dt, omega0, kappa)
    psi_next_b, psi_prev_b = step_phase_leapfrog(psi_b, psi_prev_b, dt, omega0, kappa)

    assert np.allclose(psi_next_a, psi_next_b)
    E_a = float(hamiltonian_total(psi_next_a, omega0=omega0, kappa=kappa) + np.sum(np.abs((psi_next_a - psi_prev_a) / dt) ** 2))
    E_b = float(hamiltonian_total(psi_next_b, omega0=omega0, kappa=kappa) + np.sum(np.abs((psi_next_b - psi_prev_b) / dt) ** 2))
    assert np.isclose(E_a, E_b)


def test_run_pulsed_drive_samples_reproducible_and_shapes():
    samples1 = run_pulsed_drive_samples((20, 20), steps=10, dt=0.01, source_pos=(10, 10), pulse_amp=1.0, pulse_sigma=0.5)
    samples2 = run_pulsed_drive_samples((20, 20), steps=10, dt=0.01, source_pos=(10, 10), pulse_amp=1.0, pulse_sigma=0.5)

    assert len(samples1) == len(samples2)
    for (t1, psi1), (t2, psi2) in zip(samples1, samples2):
        assert t1 == t2
        assert psi1.shape == psi2.shape == (20, 20)
        assert np.allclose(psi1, psi2)


def test_laplacian_zero_on_constant_field():
    arr = np.full((12, 12), 3.14)
    lap = laplacian_iso(arr)
    assert lap.shape == arr.shape
    assert np.allclose(lap, 0.0)
