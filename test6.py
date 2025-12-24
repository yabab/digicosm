import numpy as np
from model import step_relativistic_2nd_order


def test_cardinal_vs_diagonal():
    print("TEST 6: Cardinal vs Diagonal")

    N, steps, dt, c = 150, 200, 0.05, 1.0
    m = 0.0
    psi_nm1 = np.zeros((N, N), dtype=np.complex128)
    psi_n = np.zeros_like(psi_nm1)
    center = N // 2
    psi_n[center, center] = 1.0 + 0.0j
    psi_nm1 = psi_n.copy()  # zero initial velocity

    for _ in range(steps):
        psi_nm1, psi_n = step_relativistic_2nd_order(psi_nm1, psi_n, dt, c, m)

    I = np.abs(psi_n) ** 2
    # peak along cardinal (to the right)
    row = I[center, center:]
    card_offset = int(np.argmax(row))
    card = card_offset

    # peak along diagonal (down-right)
    diag_line = np.array([I[center + i, center + i] for i in range(N - center)])
    diag_offset = int(np.argmax(diag_line))
    diag = diag_offset

    delta = abs(card - diag)
    print("Δpeak =", delta)
    if delta < 4:
        print("✅ PASS: Cardinal/diagonal peak positions similar")
        return True
    else:
        print("❌ FAIL: Significant cardinal/diagonal difference")
        return False


if __name__ == '__main__':
    test_cardinal_vs_diagonal()
