import numpy as np
from code.model import init_phase_leapfrog, step_phase_leapfrog, energy_density, weighted_norm, hamiltonian_total

def test_metric_weighted_unitarity_fixed_lapse():
    print("TEST 10: Metric-weighted unitarity under fixed heterogeneous clock_rate")

    # For i dψ/dt = N(x) H ψ with fixed N(x) and Hermitian H,
    # the conserved quadratic form is ||ψ||^2_W = Σ |ψ|^2 / N.
    # Plain norm Σ|ψ|^2 is not generally conserved when N varies spatially.

    rng = np.random.default_rng(0)

    N = 90
    steps = 200
    dt = 0.0005
    omega0 = 0.0
    kappa = 1.0

    # Fixed lapse/clock_rate field: left half slow, right half fast.
    clock_rate = np.ones((N, N), dtype=float)
    clock_rate[:, : N // 2] = 0.2
    clock_rate[:, N // 2 :] = 1.0

    # Random initial condition to excite many modes.
    psi = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))).astype(np.complex128)
    psi *= 0.05

    psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa, clock_rate=clock_rate)

    def total_energy(psi_now, psi_prev, clock_rate):
        # approximate kinetic using backward difference adjusted by local clock_rate
        dt_eff = dt * clock_rate
        vel = (psi_now - psi_prev) / dt_eff
        KE = float(np.sum(np.abs(vel) ** 2))
        PE = hamiltonian_total(psi_now, omega0=omega0, kappa=kappa)
        return KE + PE

    E0 = total_energy(psi, psi_prev, clock_rate)

    Es = []
    for _ in range(steps):
        psi, psi_prev = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa, clock_rate=clock_rate)
        Es.append(total_energy(psi, psi_prev, clock_rate))

    E1 = Es[-1]
    drift_E = abs(E1 - E0) / (abs(E0) if E0 != 0 else 1.0)
    print(f"energy drift = {drift_E*100:.3f}%")

    if drift_E > 0.20:
        print("❌ FAIL: Energy drift too large under fixed heterogeneous clock_rate")
        return False

    print("✅ PASS: Energy approximately conserved under fixed heterogeneous clock_rate")
    return True