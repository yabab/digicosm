import numpy as np
from model import step


def test_cardinal_vs_diagonal():
    print("TEST 6: Cardinal vs Diagonal")

    N, steps, dt, c = 150, 200, 0.05, 1.0
    phi = np.zeros((N, N))
    pi = np.zeros_like(phi)
    center = N // 2
    phi[center, center] = 1.0

    for _ in range(steps):
        phi, pi = step(phi, pi, dt, c, 0.0)

    I = phi**2
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
