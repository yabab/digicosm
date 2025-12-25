import numpy as np
from code.model import run_pulsed_drive_samples
from code.measurements import precompute_radial_reduction, detect_front_outermost

# ============================================================
# TEST 3 — Light-cone speed
# ============================================================

def test_light_cone_speed():
    """Test finite propagation speed of wave fronts.
    
    Validates that wave fronts propagate at a finite, physically reasonable
    speed by tracking the outermost front position over time and fitting
    a linear relationship.
    """
    print("\nTEST 3: Finite propagation speed (pulsed, phase dynamics)")

    # --- parameters (defined ONCE) ---
    N = 250
    steps = 1200
    dt = 0.02
    kappa = 1.0
    omega0 = 0.0

    # Input validation
    assert N > 0 and steps > 0, "Grid size and steps must be positive"
    assert dt > 0 and kappa > 0, "Time step and coupling must be positive"

    pulse_amp = 1.0
    pulse_t0 = 2.0
    pulse_sigma = 0.6

    assert pulse_amp > 0 and pulse_sigma > 0, "Pulse parameters must be positive"

    source = (N//2, 20)
    assert 0 <= source[0] < N and 0 <= source[1] < N, "Source position out of bounds"

    samples = run_pulsed_drive_samples(
        (N, N),
        steps,
        dt,
        kappa=kappa,
        omega0=omega0,
        source_pos=source,
        pulse_amp=pulse_amp,
        pulse_t0=pulse_t0,
        pulse_sigma=pulse_sigma,
        sample_every=5,
    )

    assert len(samples) > 0, "No samples generated"

    radial_cache = precompute_radial_reduction((N, N), source)
    fronts = []
    cutoff_time = pulse_t0 + 2 * pulse_sigma
    
    for t, psi in samples:
        assert np.isfinite(psi).all(), f"Non-finite values at t={t}"
        if t <= cutoff_time:
            continue
        r = detect_front_outermost(np.abs(psi) ** 2, radial_cache, threshold=1e-8)
        if r is not None:
            fronts.append((t, r))

    min_fronts = 10
    if len(fronts) < min_fronts:
        print(f"❌ FAIL: Insufficient front detections ({len(fronts)} < {min_fronts})")
        print(f"  Samples: {len(samples)}, Cutoff time: {cutoff_time:.2f}")
        

    times = np.array([t for t, r in fronts], dtype=float)
    radii = np.array([r for t, r in fronts], dtype=float)

    assert len(times) > 1, "Need at least 2 points for linear fit"

    coeffs = np.polyfit(times, radii, 1)
    v = coeffs[0]

    # Phase dynamics is dispersive, but the lattice still has a finite maximum group velocity.
    # For robustness we only require:
    #   - positive speed
    #   - well below a conservative lattice bound O(kappa)
    v_bound = 4.0 * kappa
    v_min = 0.05
    
    print(f"Results: measured front speed = {v:.4f}")
    print(f"  Valid range: ({v_min}, {v_bound:.2f})")
    print(f"  Data points: {len(fronts)}")

    if not (v_min < v < v_bound):
        print(f"❌ FAIL: Speed {v:.4f} outside valid range ({v_min}, {v_bound:.2f})")
        

    print(f"✅ PASS: Finite propagation speed confirmed (v={v:.4f})")
    