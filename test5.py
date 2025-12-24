import numpy as np
from model import step


def _precompute_radial_reduction(shape, center):
    """Precompute helpers to compute radial max profiles efficiently."""
    h, w = shape
    cy, cx = center
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = rr.astype(np.int32).ravel()

    order = np.argsort(r_int)
    r_sorted = r_int[order]
    starts = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]])
    r_vals = r_sorted[starts]
    max_r = int(r_vals.max())

    return {
        "order": order,
        "starts": starts,
        "r_vals": r_vals,
        "max_r": max_r,
    }


def detect_front_outermost(intensity, radial_cache, *, rmin=5, threshold=1e-8):
    """Return the *outermost* radius where intensity exceeds threshold.

    Using the outermost radius avoids false negatives/oscillations near the
    source and produces a monotonic front radius until periodic wrap-around.
    """
    flat_sorted = intensity.ravel()[radial_cache["order"]]
    max_per_seg = np.maximum.reduceat(flat_sorted, radial_cache["starts"])

    radial_max = np.zeros(radial_cache["max_r"] + 1, dtype=max_per_seg.dtype)
    radial_max[radial_cache["r_vals"]] = max_per_seg

    for r in range(min(radial_cache["max_r"], intensity.shape[0] // 2 - 1), rmin - 1, -1):
        if radial_max[r] > threshold:
            return r
    return None


def test_c_scaling():
    print("TEST 5: Speed of light scaling")

    speeds = []
    for c in [0.5, 1.0, 2.0]:
        # Low-c runs need longer time to reach measurable radii because the
        # isotropic Laplacian stencil rescales the effective wave speed.
        N, steps, dt = 150, 900, 0.05
        center = (N//2, N//2)
        phi = np.zeros((N, N))
        pi = np.zeros_like(phi)
        phi[center[0], center[1]] = 1.0

        radial_cache = _precompute_radial_reduction(phi.shape, center)

        fronts = []
        for t in range(steps):
            phi, pi = step(phi, pi, dt, c, 0.0)
            if t % 10 == 0:
                r = detect_front_outermost(phi**2, radial_cache)
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
