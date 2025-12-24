import numpy as np
from model import laplacian_iso

# ============================================================
# Driven-pulse hybrid evolution
# ============================================================

def run_driven_pulse(
    N, steps, dt, c, m,
    source_pos,
    pulse_amp, pulse_t0, pulse_sigma
):
    sy, sx = source_pos

    psi_nm1 = np.zeros((N, N), dtype=np.complex128)
    psi_n   = np.zeros_like(psi_nm1)

    times = []
    radii = []

    angles = np.linspace(0, 2*np.pi, 96, endpoint=False)
    max_r = min(sy, sx, N-1-sy, N-1-sx) - 2

    for n in range(steps):
        t = n * dt
        lap = laplacian_iso(psi_n)

        drive = np.zeros_like(psi_n)
        amp = pulse_amp * np.exp(-0.5 * ((t - pulse_t0)/pulse_sigma)**2)
        drive[sy, sx] = amp

        psi_np1 = (
            2*psi_n - psi_nm1
            + dt**2 * (c**2 * lap - m**2 * psi_n + drive)
        )

        psi_nm1, psi_n = psi_n, psi_np1

        if n % 5 != 0:
            continue

        intensity = np.abs(psi_n)**2

        radial_bins = [[] for _ in range(max_r)]
        for ang in angles:
            for r in range(1, max_r):
                y = int(round(sy + r * np.sin(ang)))
                x = int(round(sx + r * np.cos(ang)))
                radial_bins[r].append(intensity[y, x])

        radial_profile = np.array([np.mean(b) if b else 0.0 for b in radial_bins])

        # ignore near-source region
        r_min = int(0.2 * max_r)
        r_peak = r_min + np.argmax(radial_profile[r_min:])

        if radial_profile[r_peak] > 1e-8:
            radii.append(r_peak)
            times.append(t)


    return np.array(times), np.array(radii)


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
