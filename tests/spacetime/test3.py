import numpy as np
from code.model import run_pulsed_drive_samples
from code.measurements import precompute_radial_reduction, detect_front_outermost

# ============================================================
# TEST 3 — Light-cone speed
# ============================================================

def test_light_cone_speed():
    print("TEST 3: Finite propagation speed (pulsed, phase dynamics)")

    # --- parameters (defined ONCE) ---
    N = 250
    steps = 1200
    dt = 0.02
    kappa = 1.0
    omega0 = 0.0

    pulse_amp = 1.0
    pulse_t0 = 2.0
    pulse_sigma = 0.6

    source = (N//2, 20)

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

    radial_cache = precompute_radial_reduction((N, N), source)
    fronts = []
    for t, psi in samples:
        if t <= (pulse_t0 + 2 * pulse_sigma):
            continue
        r = detect_front_outermost(np.abs(psi) ** 2, radial_cache, threshold=1e-8)
        if r is not None:
            fronts.append((t, r))

    if len(fronts) < 10:
        print("❌ FAIL: insufficient front detections")
        return False

    times = np.array([t for t, r in fronts], dtype=float)
    radii = np.array([r for t, r in fronts], dtype=float)

    coeffs = np.polyfit(times, radii, 1)
    v = coeffs[0]

    # Phase dynamics is dispersive, but the lattice still has a finite maximum group velocity.
    # For robustness we only require:
    #   - positive speed
    #   - well below a conservative lattice bound O(kappa)
    v_bound = 4.0 * kappa
    print(f"measured front speed = {v:.4f} (bound < {v_bound:.2f})")

    if 0.05 < v < v_bound:
        print("✅ PASS: Finite propagation speed confirmed")
        return True

    print("❌ FAIL: unreasonable speed")
    return False