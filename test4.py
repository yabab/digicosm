import numpy as np
from model import init_phase_leapfrog, step_phase_leapfrog, hamiltonian_total, energy_density


def test_energy_conservation():
    print("TEST 4: Norm/Hamiltonian stability (phase dynamics)")

    N, steps, dt = 120, 3000, 0.01
    kappa = 1.0
    omega0 = 0.0

    psi = np.zeros((N, N), dtype=np.complex128)
    psi[N // 2 - 5, N // 2 - 30] = 1.0 + 0.0j
    psi[N // 2 + 5, N // 2 - 30] = 1.0 + 0.0j

    v_half = init_phase_leapfrog(psi, dt, omega0, kappa)

    n0 = float(np.sum(energy_density(psi)))
    h0 = hamiltonian_total(psi, omega0=omega0, kappa=kappa)

    norms = []
    hams = []
    for _ in range(steps):
        psi, v_half = step_phase_leapfrog(psi, v_half, dt, omega0, kappa)
        norms.append(float(np.sum(energy_density(psi))))
        hams.append(hamiltonian_total(psi, omega0=omega0, kappa=kappa))

    n1 = norms[-1]
    h1 = hams[-1]

    norm_drift = abs(n1 - n0) / (abs(n0) if n0 != 0 else 1.0)
    ham_drift = abs(h1 - h0) / (abs(h0) if h0 != 0 else 1.0)

    print(f"norm drift = {norm_drift*100:.4f}%")
    print(f"hamiltonian drift = {ham_drift*100:.4f}%")

    if norm_drift < 0.02 and ham_drift < 0.05:
        print("✅ PASS: Approximately conserved")
        return True

    print("❌ FAIL: Drift too large")
    return False


if __name__ == '__main__':
    test_energy_conservation()
