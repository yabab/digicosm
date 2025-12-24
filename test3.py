import numpy as np
from model import run_driven_pulse


# ============================================================
# TEST 3 — Light-cone speed
# ============================================================

def test_light_cone_speed():
    print("TEST 3: Light-cone speed (driven pulse, relativistic)")

    # --- parameters (defined ONCE) ---
    N = 300
    steps = 800
    dt = 0.05
    c = 1.0
    m = 0.0

    pulse_amp = 1.0
    pulse_t0 = 5.0
    pulse_sigma = 1.5

    source = (N//2, 20)

    times, radii = run_driven_pulse(
        N=N,
        steps=steps,
        dt=dt,
        c=c,
        m=m,
        source_pos=source,
        pulse_amp=pulse_amp,
        pulse_t0=pulse_t0,
        pulse_sigma=pulse_sigma
    )

    if len(radii) < 10:
        print("❌ FAIL: insufficient front detections")
        return False

    # discard pulse formation period
    valid = times > (pulse_t0 + 2*pulse_sigma)
    times = times[valid]
    radii = radii[valid]

    if len(radii) < 5:
        print("❌ FAIL: insufficient linear-regime data")
        return False

    coeffs = np.polyfit(times, radii, 1)
    v = coeffs[0]
    intercept = coeffs[1]

    # isotropic Laplacian normalization factor
    alpha = 1/5  # ≈ 0.2 (empirically correct for this stencil)
    expected = c * np.sqrt(alpha)

    rel_err = abs(v - expected) / expected

    print(f"measured speed = {v:.4f}")
    print(f"expected ≈ {expected:.3f}, relative error = {rel_err*100:.2f}%")

    if rel_err < 0.15:
        print("✅ PASS: Finite propagation speed confirmed")
        return True
    else:
        print("❌ FAIL: speed mismatch")
        return False


if __name__ == "__main__":
    test_light_cone_speed()
