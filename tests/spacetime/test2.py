import numpy as np
from code.model import run_driven_phase_wave
from code.measurements import detect_from_buffer

# ============================================================
# TEST 2 — Two-Source Interference (CORRECT)
# ============================================================

def test_two_source_interference_relativistic():
    print("\nTEST 2: Phase Dynamics Two-Source Interference (Driven)")

    # Increased domain and longer run to produce many fringes
    N = 400
    steps = 1400
    dt = 0.05
    kappa = 1.0
    omega0 = 0.0
    drive_omega = 4.0
    center = N // 2

    psi0 = np.zeros((N, N), dtype=np.complex128)

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
    print(f"Running optimized configuration: N={N}, steps={steps}, drive_omega={drive_omega}, separation={d}, x_source={x_source}")
    psi_avg, buf = run_driven_phase_wave(
        psi0,
        steps,
        dt,
        kappa=kappa,
        omega0=omega0,
        drive_omega=drive_omega,
        sources=sources,
        avg_last=160,
        warmup_frac=0.5,
    )
    best = detect_from_buffer(buf)
    print(f"result -> best peaks = {best['count']}, params={best.get('params')}")
    if best['count'] >= 6:
        print("✅ PASS: Many interference fringes detected")
        return True

    print("❌ FAIL: Insufficient fringes with optimized setup")
    return False