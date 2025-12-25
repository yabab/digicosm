import numpy as np
from code.model import init_phase_leapfrog, step_phase_leapfrog
from code.measurements import precompute_radial_reduction, detect_front_outermost

def test_kappa_scaling():
    """Test propagation speed scaling with coupling parameter kappa.
    
    Validates that the propagation speed scales approximately as sqrt(kappa),
    consistent with the dispersion relation of the model.
    """
    print("\nTEST 5: Propagation speed scaling with kappa")

    kappa_values = [0.5, 1.0, 2.0]
    speeds = []
    
    for kappa in kappa_values:
        assert kappa > 0, f"Kappa must be positive: {kappa}"
        
        N, steps, dt = 150, 1200, 0.02
        omega0 = 0.0
        center = (N//2, N//2)
        
        # Input validation
        assert N > 0 and steps > 0, "Grid size and steps must be positive"
        assert dt > 0, "Time step must be positive"
        
        psi = np.zeros((N, N), dtype=np.complex128)
        psi[center[0], center[1]] = 1.0 + 0.0j

        psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa)

        radial_cache = precompute_radial_reduction(psi.shape, center)

        fronts = []
        for t in range(steps):
            psi, psi_prev = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa)
            
            if t % 10 == 0:
                assert np.isfinite(psi).all(), f"Non-finite values at step {t} for kappa={kappa}"
                intensity = np.abs(psi) ** 2
                r = detect_front_outermost(intensity, radial_cache)
                if r is not None:
                    fronts.append((t * dt, r))

        min_fronts = 3
        if len(fronts) < min_fronts:
            print(f"❌ FAIL: Insufficient fronts for kappa={kappa} ({len(fronts)} < {min_fronts})")
            return False

        times = np.array([t for t, r in fronts], dtype=float)
        radii = np.array([r for t, r in fronts], dtype=float)

        # Fit only before periodic wrap-around/saturation.
        max_r_fit = (N // 2) - 6
        fit_mask = (radii >= 3) & (radii < max_r_fit)
        times = times[fit_mask]
        radii = radii[fit_mask]

        min_fit_points = 6
        if len(radii) < min_fit_points:
            print(f"❌ FAIL: Insufficient pre-wrap data for kappa={kappa} ({len(radii)} < {min_fit_points})")
            return False

        v = np.polyfit(times, radii, 1)[0]
        assert np.isfinite(v), f"Non-finite velocity for kappa={kappa}"
        speeds.append(v)
        print(f"  kappa={kappa}: speed={v:.4f} (from {len(radii)} points)")

    # Analyze speed ratios
    assert len(speeds) == len(kappa_values), "Mismatch in speeds array"
    ratios = np.array(speeds) / speeds[1]  # Normalize to kappa=1.0
    target = np.array([np.sqrt(0.5), 1.0, np.sqrt(2.0)])
    
    print(f"\nResults:")
    print(f"  Measured speed ratios: {ratios}")
    print(f"  Expected ratios: {target}")
    print(f"  Relative errors: {np.abs(ratios - target) / target * 100}%")
    
    tolerance = 0.30
    ok = np.allclose(ratios, target, rtol=tolerance)
    
    if ok:
        print(f"✅ PASS: Speed scaling matches sqrt(kappa) (tolerance={tolerance})")
    else:
        print(f"❌ FAIL: Speed scaling mismatch (tolerance={tolerance})")
    return ok