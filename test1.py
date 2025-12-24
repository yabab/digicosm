import numpy as np
from model import run_driven_relativistic_wave, isotropy_ring_error

# ============================================================
# TEST 1 — Single Source Isotropy (Driven, Complex, Relativistic)
# ============================================================

def test_single_source_isotropy_relativistic():
    print("\nTEST 1: Relativistic Complex Single-Source Isotropy (Driven)")

    N = 400
    steps = 1200
    dt = 0.2
    c = 1.0
    m = 0.0
    omega = 4.0
    center = N // 2

    psi_nm1 = np.zeros((N, N), dtype=np.complex128)
    psi_n = np.zeros_like(psi_nm1)

    # Single point source for isotropy test
    amp = 1.0
    sources = [(center, center - 120, amp)]

    psi_avg, _buf = run_driven_relativistic_wave(
        psi_nm1,
        psi_n,
        steps,
        dt,
        c,
        m,
        omega,
        sources,
        avg_last=600,
        warmup_frac=0.5,
    )
    intensity = np.abs(psi_avg) ** 2

    # --- isotropy measurement around the source center ---
    # source center (middle of the vertical line)
    sy = center
    sx = center - 120

    result = isotropy_ring_error(intensity, (sy, sx), dr=3, search_start=5)
    print(f"peak radius = {result['r_peak']}, σ/I = {result['iso_error']:.4f}")

    if result["iso_error"] < 0.05:
        print("✅ PASS: Isotropic steady-state wavefront")
        return True
    else:
        print("❌ FAIL: Anisotropy detected")
        return False

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ok = test_single_source_isotropy_relativistic()
    if not ok:
        print("Test failed.")
    else:
        print("Test passed.")
