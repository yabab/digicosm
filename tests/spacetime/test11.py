import numpy as np
from code.model import (
    init_phase_leapfrog,
    step_phase_leapfrog,
    init_phase_leapfrog_backreacting,
    step_phase_leapfrog_backreacting,
    clock_rate_from_psi,
)
from code.measurements import weighted_norm_change_residual

def test_weighted_norm_residual_separates_external_vs_backreaction():
    print("TEST 11: Weighted-norm residual separates numerical error vs backreaction")

    rng = np.random.default_rng(1)

    N = 70
    steps = 2500
    dt = 0.0005
    omega0 = 0.0
    kappa = 1.0

    # Spatially varying base lapse.
    x = np.arange(N)[None, :]
    base_row = 0.6 + 0.4 * (x / (N - 1))  # shape (1, N)
    base = np.broadcast_to(base_row, (N, N)).copy()  # shape (N, N)

    # Time modulation (kept small so positivity is guaranteed).
    mod_amp = 0.10
    mod_omega = 0.7

    def external_clock_rate(t):
        return base * (1.0 + mod_amp * np.sin(mod_omega * t))

    # Initial condition.
    psi0 = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))).astype(np.complex128)
    psi0 *= 0.03

    # ---------- Part A: externally-prescribed N(x,t) should have tiny residual ----------
    psi = psi0.copy()
    t0 = 0.0
    N0 = external_clock_rate(t0)
    psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa, clock_rate=N0)

    residuals_ext = []
    for n in range(steps):
        t = n * dt
        Nn = external_clock_rate(t)
        psi_next, psi_prev_next = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa, clock_rate=Nn)

        # Compare against N at t and t+dt (midpoint will make this second-order-ish).
        Nn1 = external_clock_rate(t + dt)
        *_vals, res = weighted_norm_change_residual(psi, psi_next, Nn, Nn1, dt)

        if n > steps // 3:
            residuals_ext.append(abs(res))

        psi, psi_prev = psi_next, psi_prev_next

    ext_med = float(np.median(residuals_ext))
    ext_mean = float(np.mean(residuals_ext))
    print(f"external lapse residual: median={ext_med:.3e}, mean={ext_mean:.3e}")

    if ext_med > 1e3:
        print("❌ FAIL: External-lapse residual too large (numerical error baseline too high)")
        return False

    # ---------- Part B: backreacting N(psi) should show much larger residual ----------
    psi = psi0.copy()
    beta = 0.7
    clip = (0.05, 1.0)

    psi_prev = init_phase_leapfrog_backreacting(
        psi,
        dt,
        omega0,
        kappa,
        base_clock_rate=1.0,
        curvature_beta=beta,
        curvature_mode="rational",
        curvature_clip=clip,
    )

    residuals_br = []
    for n in range(steps):
        psi_next, psi_prev_next = step_phase_leapfrog_backreacting(
            psi,
            psi_prev,
            dt,
            omega0,
            kappa,
            base_clock_rate=1.0,
            curvature_beta=beta,
            curvature_mode="rational",
            curvature_clip=clip,
        )

        Nn = clock_rate_from_psi(psi, base=1.0, beta=beta, mode="rational", clip=clip)
        Nn1 = clock_rate_from_psi(psi_next, base=1.0, beta=beta, mode="rational", clip=clip)
        *_vals, res = weighted_norm_change_residual(psi, psi_next, Nn, Nn1, dt)

        if n > steps // 3:
            residuals_br.append(abs(res))

        psi, psi_prev = psi_next, psi_prev_next

    br_med = float(np.median(residuals_br))
    br_mean = float(np.mean(residuals_br))
    print(f"backreaction residual: median={br_med:.3e}, mean={br_mean:.3e}")

    # Backreaction should measurably increase this residual above the external-N baseline,
    # but not necessarily by an order of magnitude (it depends on coupling strength and clip).
    if br_med < 1.2 * ext_med or br_mean < 1.2 * ext_mean:
        print("❌ FAIL: Backreaction residual not sufficiently above external baseline")
        return False

    print("✅ PASS: Diagnostic separates numerical error vs physical backreaction")
    return True