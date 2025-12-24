import numpy as np
from model import step_relativistic_2nd_order, energy_relativistic_2nd_order


def test_energy_conservation():
    print("TEST 4: Energy conservation")

    N, steps, dt, c = 100, 2000, 0.02, 1.0
    m = 0.0

    psi_nm1 = np.zeros((N, N), dtype=np.complex128)
    psi_n = np.zeros_like(psi_nm1)

    # Localized initial displacement (zero initial velocity => psi_nm1 == psi_n).
    psi_n[N // 2 - 5, N // 2 - 30] = 1.0 + 0.0j
    psi_n[N // 2 + 5, N // 2 - 30] = 1.0 + 0.0j
    psi_nm1 = psi_n.copy()

    E = []
    for _ in range(steps):
        psi_nm1, psi_n = step_relativistic_2nd_order(psi_nm1, psi_n, dt, c, m)
        E.append(energy_relativistic_2nd_order(psi_nm1, psi_n, dt, c, m))

    drift = abs(E[-1] - E[0]) / abs(E[0]) if E[0] != 0 else float('inf')
    print(f"energy drift = {drift*100:.4f}%")
    if drift < 0.01:
        print("✅ PASS: Energy approximately conserved")
        return True
    else:
        print("❌ FAIL: Energy drift too large")
        return False


if __name__ == '__main__':
    test_energy_conservation()
