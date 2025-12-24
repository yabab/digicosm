import numpy as np
from model import laplacian_iso, step


def test_energy_conservation():
    print("TEST 4: Energy conservation")

    N, steps, dt, c = 100, 2000, 0.02, 1.0
    phi = np.zeros((N, N))
    pi = np.zeros_like(phi)
    phi[N//2-5, N//2-30] = 1.0
    phi[N//2+5, N//2-30] = 1.0

    E = []
    for _ in range(steps):
        phi, pi = step(phi, pi, dt, c, 0.0)
        E.append(0.5*np.sum(pi**2) - 0.5*c**2*np.sum(phi*laplacian_iso(phi)))

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
