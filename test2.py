import numpy as np
from model import run_driven_relativistic_wave
from measurements import detect_from_buffer

# ============================================================
# TEST 2 — Two-Source Interference (CORRECT)
# ============================================================

def test_two_source_interference_relativistic():
    print("\nTEST 2: Relativistic Complex Two-Source Interference (Driven)")

    # Increased domain and longer run to produce many fringes
    N = 400
    steps = 1000
    dt = 0.2
    c = 1.0
    m = 0.0
    omega = 4.0
    center = N // 2

    psi_nm1 = np.zeros((N, N), dtype=np.complex128)
    psi_n   = np.zeros_like(psi_nm1)

    # Larger vertical separation and sources placed further left for longer propagation
    d = 80
    x_source = center - 160
    # Use short vertical line sources (extended sources) to increase fringe contrast
    sources = []
    line_half = 10
    amp = 1.0
    for dy in range(-line_half, line_half+1):
        sources.append((center - d + dy, x_source, amp))
        sources.append((center + d + dy, x_source, amp))

    # Single optimized attempt: run once with the extended sources
    # detection routine has been moved to `model.detect_from_buffer`

    # Run the optimized configuration once
    print(f"Running optimized configuration: N={N}, steps={steps}, omega={omega}, separation={d}, x_source={x_source}")
    psi_avg, buf = run_driven_relativistic_wave(psi_nm1, psi_n, steps, dt, c, m, omega, sources, avg_last=160, warmup_frac=0.5)
    best = detect_from_buffer(buf)
    print(f"result -> best peaks = {best['count']}, params={best.get('params')}")
    if best['count'] >= 6:
        print("✅ PASS: Many interference fringes detected")
        return True

    print("❌ FAIL: Insufficient fringes with optimized setup")
    return False

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    test_two_source_interference_relativistic()
