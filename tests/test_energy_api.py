import numpy as np
from code.model import (
    kinetic_energy,
    kinetic_energy_density,
    potential_energy,
    potential_energy_density,
    total_energy,
    total_energy_density,
    hamiltonian_total,
)


def test_kinetic_density_and_total_match_manual():
    N = 20
    dt = 0.002
    psi = np.zeros((N, N), dtype=np.complex128)
    psi_prev = np.zeros_like(psi)
    psi[5, 5] = 1.0 + 0j

    kd = kinetic_energy_density(psi, psi_prev, dt)
    K = kinetic_energy(psi, psi_prev, dt)

    assert kd.shape == psi.shape
    assert np.isclose(K, kd.sum())


def test_potential_density_and_total_consistent_with_hamiltonian():
    N = 16
    psi = np.zeros((N, N), dtype=np.complex128)
    psi[3, 4] = 1.0 + 0j

    pd = potential_energy_density(psi, omega0=0.5, kappa=1.0)
    P = potential_energy(psi, omega0=0.5, kappa=1.0)

    # hamiltonian_total uses omega0*|psi|^2 + kappa*curvature_proxy summed
    H = hamiltonian_total(psi, omega0=0.5, kappa=1.0)

    assert pd.shape == psi.shape
    assert np.isclose(P, H)


def test_total_energy_consistency():
    N = 18
    dt = 0.001
    psi = np.zeros((N, N), dtype=np.complex128)
    psi_prev = np.zeros_like(psi)
    psi[7, 7] = 1.0 + 0j

    TD = total_energy_density(psi, psi_prev, dt, omega0=0.3, kappa=0.8)
    T = total_energy(psi, psi_prev, dt, omega0=0.3, kappa=0.8)

    assert TD.shape == psi.shape
    assert np.isclose(T, TD.sum())


def test_energy_density_nonnegative():
    N = 12
    dt = 0.001
    psi = np.zeros((N, N), dtype=np.complex128)
    psi_prev = np.zeros_like(psi)
    psi[1, 2] = 1.0 + 0j

    TD = total_energy_density(psi, psi_prev, dt)
    assert (TD >= 0).all()
