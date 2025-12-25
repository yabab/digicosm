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
    pulse_amp = 8.0
    pulse_t0 = 2.0
    pulse_sigma = 1.0

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

    # Choose the sampled frame where the outward front is strongest away from the source.
    inner = 20
    yy, xx = np.indices((N, N))
    rr = np.sqrt((yy - sy) ** 2 + (xx - sx) ** 2)

    # Compute the off-source peak intensity for each sample and pick the best frame.
    peak_list = []
    for t, p in samples:
        I = np.abs(p) ** 2
        mask = rr > inner
        peak = float(np.max(I[mask])) if mask.any() else 0.0
        peak_list.append((peak, t, p))

    if len(peak_list) == 0:
        print("❌ FAIL: no samples recorded")
        return False

    peak_list.sort(key=lambda x: x[0], reverse=True)
    I_max, t_used, psi = peak_list[0]
    intensity = np.abs(psi) ** 2

    # Try a few thresholds / rmin values until the front is detectable.
    tried = []
    radii = np.array([])
    for thresh_frac in (0.01, 0.005, 0.002):
        for rmin in (8, 5, 3):
            thresh = max(1e-12, thresh_frac * I_max)
            radii = front_radii_by_angle(intensity, (sy, sx), threshold=thresh, angles=72, rmin=rmin)
            radii = radii[np.isfinite(radii)]
            tried.append((thresh_frac, rmin, radii.size))
            if radii.size >= 3:
                break
        if radii.size >= 3:
            break

    print(
        f"debug: samples={len(samples)}, best_t={t_used:.2f}, I_max={I_max:.3e}, trials={tried}, radii_count={radii.size}"
    )

    if radii.size < 3:
        print("❌ FAIL: insufficient angular front detections")
        return False

    r_mean = float(np.mean(radii))
    r_std = float(np.std(radii))
    iso = (r_std / r_mean) if r_mean > 0 else float("inf")
    print(f"t={t_used:.2f}, r_mean={r_mean:.2f}, σ/r={iso:.4f}")

    if iso < 0.30:
        print("✅ PASS: Isotropic propagating front")
        return True

    print("❌ FAIL: Anisotropy detected")
    return False