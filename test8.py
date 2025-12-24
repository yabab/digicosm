import numpy as np
from model import (
    curvature_proxy,
    clock_rate_from_curvature,
    init_phase_leapfrog,
    step_phase_leapfrog,
)
from measurements import estimate_angular_frequency


def test_curvature_induced_time_dilation_local():
    print("TEST 8: Curvature-derived clock_rate produces local time dilation")

    N = 40
    omega0 = 2.5
    kappa = 0.0  # keep sites decoupled so local ω depends only on clock_rate
    dt = 0.01
    steps = 7000

    # Build a field with a localized phase defect to create curvature gradients.
    psi0 = np.ones((N, N), dtype=np.complex128)
    cy, cx = N // 2, N // 2
    psi0[cy - 1 : cy + 2, cx - 1 : cx + 2] *= -1.0  # small pi-phase patch

    curv0 = curvature_proxy(psi0)
    # Map curvature -> clock_rate (higher curvature => slower clock).
    clock_rate = clock_rate_from_curvature(curv0, base=1.0, beta=0.8, mode="rational", clip=(0.05, 2.0))

    max_idx = np.unravel_index(int(np.argmax(curv0)), curv0.shape)
    min_idx = np.unravel_index(int(np.argmin(curv0)), curv0.shape)

    cr_hi = float(clock_rate[max_idx])
    cr_lo = float(clock_rate[min_idx])

    print(f"curv_max at {max_idx}, clock_rate={cr_hi:.4f}")
    print(f"curv_min at {min_idx}, clock_rate={cr_lo:.4f}")

    if not (cr_hi < cr_lo):
        print("❌ FAIL: Expected higher curvature => smaller clock_rate")
        return False

    psi = psi0.copy()
    v_half = init_phase_leapfrog(psi, dt, omega0, kappa, clock_rate=clock_rate)

    times = []
    z_hi = []
    z_lo = []

    for n in range(steps):
        t = n * dt
        psi, v_half = step_phase_leapfrog(psi, v_half, dt, omega0, kappa, clock_rate=clock_rate)
        if n > steps // 3:
            times.append(t)
            z_hi.append(psi[max_idx])
            z_lo.append(psi[min_idx])

    w_hi = estimate_angular_frequency(np.array(z_hi), np.array(times))
    w_lo = estimate_angular_frequency(np.array(z_lo), np.array(times))

    # For kappa=0: ω_local ≈ ω0 * clock_rate(local)
    ratio_hat = w_hi / w_lo if w_lo != 0 else float("inf")
    ratio_exp = cr_hi / cr_lo if cr_lo != 0 else float("inf")
    rel_err = abs(ratio_hat - ratio_exp) / (abs(ratio_exp) if ratio_exp != 0 else 1.0)

    print(f"ω_hi={w_hi:.4f}, ω_lo={w_lo:.4f}")
    print(f"ratio_hat={ratio_hat:.4f}, ratio_exp={ratio_exp:.4f}, rel_err={rel_err*100:.2f}%")

    if rel_err > 0.05:
        print("❌ FAIL: Local dilation ratio mismatch")
        return False

    print("✅ PASS: Curvature-derived clock_rate matches local frequency scaling")
    return True


if __name__ == "__main__":
    test_curvature_induced_time_dilation_local()
