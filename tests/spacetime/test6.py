import numpy as np
from code.model import init_phase_leapfrog, step_phase_leapfrog
from code.measurements import cardinal_diagonal_peak_delta

def test_cardinal_vs_diagonal():
    print("TEST 6: Cardinal vs Diagonal")

    N, steps, dt = 150, 600, 0.02
    kappa = 1.0
    omega0 = 0.0
    psi = np.zeros((N, N), dtype=np.complex128)
    center = N // 2
    psi[center, center] = 1.0 + 0.0j
    v_half = init_phase_leapfrog(psi, dt, omega0, kappa)

    for _ in range(steps):
        psi, v_half = step_phase_leapfrog(psi, v_half, dt, omega0, kappa)

    I = np.abs(psi) ** 2
    result = cardinal_diagonal_peak_delta(I, (center, center))
    print("Δpeak =", result["delta"])
    if result["delta"] < 4:
        print("✅ PASS: Cardinal/diagonal peak positions similar")
        return True
    else:
        print("❌ FAIL: Significant cardinal/diagonal difference")
        return False