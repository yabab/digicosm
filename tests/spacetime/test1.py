import numpy as np
from code.model import run_pulsed_drive_samples
from code.measurements import front_radii_by_angle

# ============================================================
# TEST 1 — Single Source Isotropy (Driven, Complex, Relativistic)
# ============================================================

def test_single_source_isotropy_relativistic():
    print("\nTEST 1: Phase Dynamics Single-Source Isotropy (Driven)")

    N = 260
    steps = 1400
    dt = 0.02
    kappa = 1.0
    omega0 = 0.0
    center = N // 2

    # Use a short Gaussian pulse to produce a clear propagating front.
    sy = center
    sx = center - 80
    pulse_amp = 1.0
    pulse_t0 = 2.0
    pulse_sigma = 0.6

    samples = run_pulsed_drive_samples(
        (N, N),
        steps,
        dt,
        kappa=kappa,
        omega0=omega0,
        source_pos=(sy, sx),
        pulse_amp=pulse_amp,
        pulse_t0=pulse_t0,
        pulse_sigma=pulse_sigma,
        sample_every=10,
    )

    # Pick a late-enough snapshot where the front is away from the source.
    t_snap = pulse_t0 + 8.0
    snap = min(samples, key=lambda tr: abs(tr[0] - t_snap))
    t_used, psi = snap
    intensity = np.abs(psi) ** 2

    # Threshold as a fraction of max intensity away from the source.
    # (Avoid picking the near-source spike.)
    inner = 8
    yy, xx = np.indices((N, N))
    rr = np.sqrt((yy - sy) ** 2 + (xx - sx) ** 2)
    mask = rr > inner
    I_max = float(np.max(intensity[mask]))
    thresh = max(1e-10, 0.12 * I_max)

    radii = front_radii_by_angle(intensity, (sy, sx), threshold=thresh, angles=72, rmin=10)
    radii = radii[np.isfinite(radii)]
    if radii.size < 40:
        print("❌ FAIL: insufficient angular front detections")
        return False

    r_mean = float(np.mean(radii))
    r_std = float(np.std(radii))
    iso = (r_std / r_mean) if r_mean > 0 else float("inf")
    print(f"t={t_used:.2f}, r_mean={r_mean:.2f}, σ/r={iso:.4f}")

    if iso < 0.06:
        print("✅ PASS: Isotropic propagating front")
        return True

    print("❌ FAIL: Anisotropy detected")
    return False