import numpy as np
from model import (
    step_relativistic_2nd_order,
    precompute_radial_reduction,
    detect_front_outermost,
)


def test_c_scaling():
    print("TEST 5: Speed of light scaling")

    speeds = []
    for c in [0.5, 1.0, 2.0]:
        # Low-c runs need longer time to reach measurable radii because the
        # isotropic Laplacian stencil rescales the effective wave speed.
        N, steps, dt = 150, 900, 0.05
        m = 0.0
        center = (N//2, N//2)
        psi_nm1 = np.zeros((N, N), dtype=np.complex128)
        psi_n = np.zeros_like(psi_nm1)
        psi_n[center[0], center[1]] = 1.0 + 0.0j
        psi_nm1 = psi_n.copy()  # zero initial velocity

        radial_cache = precompute_radial_reduction(psi_n.shape, center)

        fronts = []
        for t in range(steps):
            psi_nm1, psi_n = step_relativistic_2nd_order(psi_nm1, psi_n, dt, c, m)
            if t % 10 == 0:
                intensity = np.abs(psi_n) ** 2
                r = detect_front_outermost(intensity, radial_cache)
                if r is not None:
                    fronts.append((t * dt, r))

        if len(fronts) < 3:
            print(f"Insufficient fronts for c={c}")
            return False

        times = np.array([t for t, r in fronts], dtype=float)
        radii = np.array([r for t, r in fronts], dtype=float)

        # Fit only before periodic wrap-around/saturation.
        max_r_fit = (N // 2) - 6
        fit_mask = (radii >= 3) & (radii < max_r_fit)
        times = times[fit_mask]
        radii = radii[fit_mask]

        if len(radii) < 6:
            print(f"Insufficient pre-wrap data for c={c}")
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


if __name__ == '__main__':
    test_c_scaling()
