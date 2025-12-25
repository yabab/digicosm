import numpy as np
from code.model import init_phase_leapfrog, step_phase_leapfrog
from code.measurements import precompute_radial_reduction, detect_front_outermost

def test_kappa_scaling():
    print("TEST 5: Propagation speed scaling with kappa")

    speeds = []
    for kappa in [0.5, 1.0, 2.0]:
        N, steps, dt = 150, 1200, 0.02
        omega0 = 0.0
        center = (N//2, N//2)
        psi = np.zeros((N, N), dtype=np.complex128)
        psi[center[0], center[1]] = 1.0 + 0.0j

        v_half = init_phase_leapfrog(psi, dt, omega0, kappa)

        radial_cache = precompute_radial_reduction(psi.shape, center)

        fronts = []
        for t in range(steps):
            psi, v_half = step_phase_leapfrog(psi, v_half, dt, omega0, kappa)
            if t % 10 == 0:
                intensity = np.abs(psi) ** 2
                r = detect_front_outermost(intensity, radial_cache)
                if r is not None:
                    fronts.append((t * dt, r))

        if len(fronts) < 3:
            print(f"Insufficient fronts for kappa={kappa}")
            return False

        times = np.array([t for t, r in fronts], dtype=float)
        radii = np.array([r for t, r in fronts], dtype=float)

        # Fit only before periodic wrap-around/saturation.
        max_r_fit = (N // 2) - 6
        fit_mask = (radii >= 3) & (radii < max_r_fit)
        times = times[fit_mask]
        radii = radii[fit_mask]

        if len(radii) < 6:
            print(f"Insufficient pre-wrap data for kappa={kappa}")
            return False

        v = np.polyfit(times, radii, 1)[0]
        speeds.append(v)

    ratios = np.array(speeds) / speeds[1]
    print("speed ratios:", ratios)
    ok = np.allclose(ratios, [0.5, 1.0, 2.0], rtol=0.3)
    if ok:
        print("✅ PASS: c-scaling roughly matches")
    else:
        print("❌ FAIL: c-scaling mismatch")
    return ok