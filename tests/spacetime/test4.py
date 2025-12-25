import numpy as np
from code.model import init_phase_leapfrog, step_phase_leapfrog, hamiltonian_total, energy_density

def test_energy_conservation():
    print("TEST 4: Norm/Hamiltonian stability (phase dynamics)")

    N, steps, dt = 120, 200, 0.0005
    kappa = 1.0
    omega0 = 0.0

    psi = np.zeros((N, N), dtype=np.complex128)
    psi[N // 2 - 5, N // 2 - 30] = 1.0 + 0.0j
    psi[N // 2 + 5, N // 2 - 30] = 1.0 + 0.0j

    psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa)

    def total_energy(psi_now, psi_prev):
        # kinetic ~ |(psi_now - psi_prev)/dt|^2, potential ~ hamiltonian_total
        vel = (psi_now - psi_prev) / dt
        KE = float(np.sum(np.abs(vel) ** 2))
        PE = hamiltonian_total(psi_now, omega0=omega0, kappa=kappa)
        return KE + PE

    E0 = total_energy(psi, psi_prev)

    Es = []
    # debug: show initial magnitudes
    print(f"initial max|psi|={np.max(np.abs(psi)):.6f}, max|psi_prev|={np.max(np.abs(psi_prev)):.6f}")
    for i in range(steps):
        psi_next, psi_prev = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa)
        if i < 5:
            print(f"step {i}: max|psi_next|={np.max(np.abs(psi_next)):.6f}")
        Es.append(total_energy(psi_next, psi_prev))
        psi = psi_next

    E1 = Es[-1]
    E_drift = abs(E1 - E0) / (abs(E0) if E0 != 0 else 1.0)
    print(f"energy drift = {E_drift*100:.4f}%")

    if E_drift < 0.10:
        print("✅ PASS: Approximately conserved")
        return True

    print("❌ FAIL: Drift too large")
    return False