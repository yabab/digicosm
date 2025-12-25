import numpy as np
from code.model import run_pulsed_drive_samples
from code.measurements import front_radii_by_angle

# ============================================================
# TEST 1 — Single Source Isotropy (Driven, Complex, Relativistic)
# ============================================================

def test_single_source_isotropy_relativistic():
    """Test that wave propagation from a single source is isotropic.
    
    Validates that the expanding wave front maintains a circular shape
    by measuring radial distances at multiple angles and computing the
    relative standard deviation.
    """
    print("\nTEST 1: Phase Dynamics Single-Source Isotropy (Driven)")

    # Simulation parameters
    N = 260
    steps = 1400
    dt = 0.02
    kappa = 1.0
    omega0 = 0.0
    center = N // 2

    # Input validation
    assert N > 0 and steps > 0, "Grid size and steps must be positive"
    assert dt > 0 and kappa > 0, "Time step and coupling must be positive"
    assert 0 <= center < N, "Center must be within grid bounds"

    # Use a short Gaussian pulse to produce a clear propagating front.
    sy = center
    sx = center - 80
    pulse_amp = 8.0
    pulse_t0 = 2.0
    pulse_sigma = 1.0

    assert 0 <= sy < N and 0 <= sx < N, "Source position must be within grid bounds"
    assert pulse_amp > 0 and pulse_sigma > 0, "Pulse parameters must be positive"

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

    # Validate samples were generated
    assert len(samples) > 0, "No samples were generated from simulation"

    # Choose the sampled frame where the outward front is strongest away from the source.
    inner = 20
    yy, xx = np.indices((N, N))
    rr = np.sqrt((yy - sy) ** 2 + (xx - sx) ** 2)

    # Compute the off-source peak intensity for each sample and pick the best frame.
    peak_list = []
    for t, p in samples:
        assert np.isfinite(p).all(), f"Non-finite values detected at t={t}"
        I = np.abs(p) ** 2
        mask = rr > inner
        peak = float(np.max(I[mask])) if mask.any() else 0.0
        peak_list.append((peak, t, p))

    if len(peak_list) == 0:
        print("❌ FAIL: No samples recorded")
        return False

    peak_list.sort(key=lambda x: x[0], reverse=True)
    I_max, t_used, psi = peak_list[0]
    
    assert I_max > 0, "Maximum intensity is zero - no wave propagation detected"
    assert np.isfinite(psi).all(), "Non-finite values in selected frame"
    
    intensity = np.abs(psi) ** 2

    # Try a few thresholds / rmin values until the front is detectable.
    tried = []
    radii = np.array([])
    min_detections = 3
    
    for thresh_frac in (0.01, 0.005, 0.002):
        for rmin in (8, 5, 3):
            thresh = max(1e-12, thresh_frac * I_max)
            radii = front_radii_by_angle(intensity, (sy, sx), threshold=thresh, angles=72, rmin=rmin)
            radii = radii[np.isfinite(radii)]
            tried.append((thresh_frac, rmin, radii.size))
            if radii.size >= min_detections:
                break
        if radii.size >= min_detections:
            break

    print(
        f"Info: samples={len(samples)}, best_t={t_used:.2f}, I_max={I_max:.3e}, "
        f"trials={tried}, radii_count={radii.size}"
    )

    if radii.size < min_detections:
        print(f"❌ FAIL: Insufficient angular front detections ({radii.size} < {min_detections})")
        print(f"  Tried parameter combinations: {tried}")
        return False

    r_mean = float(np.mean(radii))
    r_std = float(np.std(radii))
    
    assert r_mean > 0, "Mean radius is non-positive"
    assert np.isfinite(r_mean) and np.isfinite(r_std), "Non-finite statistical values"
    
    iso = r_std / r_mean
    iso_threshold = 0.30
    
    print(f"Results: t={t_used:.2f}, r_mean={r_mean:.2f}±{r_std:.2f}, σ/r={iso:.4f}")
    print(f"  Isotropy metric: {iso:.4f} (threshold: {iso_threshold})")

    if iso < iso_threshold:
        print(f"✅ PASS: Isotropic propagating front (isotropy={iso:.4f} < {iso_threshold})")
        return True

    print(f"❌ FAIL: Anisotropy detected (isotropy={iso:.4f} >= {iso_threshold})")
    return False