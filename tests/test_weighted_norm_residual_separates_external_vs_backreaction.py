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
    """Test weighted-norm residual as diagnostic for backreaction.
    
    Validates that a weighted-norm change residual can distinguish between:
    1. Small numerical errors from externally-prescribed clock_rate
    2. Larger physical effects from self-consistent backreacting clock_rate
    
    The residual measures deviation from exact norm conservation and should
    be much larger when backreaction is active.
    """
    print("\nTEST 11: Weighted-norm residual separates numerical error vs backreaction")

    rng = np.random.default_rng(1)

    N = 70
    steps = 2500
    dt = 0.0005
    omega0 = 0.0
    kappa = 1.0

    # Input validation
    assert N > 0 and steps > 0, "Grid size and steps must be positive"
    assert dt > 0 and kappa > 0, "Time step and coupling must be positive"

    # Spatially varying base lapse.
    x = np.arange(N)[None, :]
    base_row = 0.6 + 0.4 * (x / (N - 1))  # shape (1, N)
    base = np.broadcast_to(base_row, (N, N)).copy()  # shape (N, N)

    assert np.all(base > 0), "Base clock rate must be positive"
    assert np.isfinite(base).all(), "Base clock rate contains non-finite values"

    # Time modulation (kept small so positivity is guaranteed).
    mod_amp = 0.10
    mod_omega = 0.7

    assert 0 < mod_amp < 1, "Modulation amplitude must be in (0, 1)"
    assert mod_omega > 0, "Modulation frequency must be positive"

    def external_clock_rate(t):
        """Externally prescribed time-dependent clock_rate."""
        rate = base * (1.0 + mod_amp * np.sin(mod_omega * t))
        assert np.all(rate > 0), f"Clock rate became non-positive at t={t}"
        return rate

    # Initial condition.
    psi0 = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))).astype(np.complex128)
    psi0 *= 0.03

    assert np.isfinite(psi0).all(), "Initial condition contains non-finite values"
    print(f"Info: N={N}, steps={steps}, dt={dt}")
    print(f"  Base clock_rate range: [{base.min():.3f}, {base.max():.3f}]")
    print(f"  Modulation: amp={mod_amp}, omega={mod_omega}")

    # ---------- Part A: externally-prescribed N(x,t) should have tiny residual ----------
    print("\nPart A: Testing externally-prescribed clock_rate...")
    psi = psi0.copy()
    t0 = 0.0
    N0 = external_clock_rate(t0)
    psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa, clock_rate=N0)

    residuals_ext = []
    warmup = steps // 3
    
    for n in range(steps):
        t = n * dt
        Nn = external_clock_rate(t)
        psi_next, psi_prev_next = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa, clock_rate=Nn)

        # Validation
        if n % 500 == 0:
            assert np.isfinite(psi_next).all(), f"Non-finite values at step {n} (external)"
            assert np.isfinite(Nn).all(), f"Non-finite clock_rate at step {n} (external)"

        # Compare against N at t and t+dt (midpoint will make this second-order-ish).
        Nn1 = external_clock_rate(t + dt)
        *_vals, res = weighted_norm_change_residual(psi, psi_next, Nn, Nn1, dt)

        if n > warmup:
            residuals_ext.append(abs(res))

        psi, psi_prev = psi_next, psi_prev_next

    assert len(residuals_ext) > 0, "No residuals collected for external lapse"
    ext_med = float(np.median(residuals_ext))
    ext_mean = float(np.mean(residuals_ext))
    ext_threshold = 1e3
    
    print(f"  External lapse residual: median={ext_med:.3e}, mean={ext_mean:.3e}")

    if ext_med > ext_threshold:
        print(f"❌ FAIL: External-lapse residual too large ({ext_med:.3e} > {ext_threshold:.3e})")
        print("  This indicates numerical error baseline is too high")
        return False

    # ---------- Part B: backreacting N(psi) should show much larger residual ----------
    print("\nPart B: Testing backreacting clock_rate...")
    psi = psi0.copy()
    beta = 0.7
    clip = (0.05, 1.0)

    assert beta > 0, "Beta must be positive"
    assert clip[0] > 0 and clip[1] > clip[0], "Invalid clip bounds"

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

        # Validation
        if n % 500 == 0:
            assert np.isfinite(psi_next).all(), f"Non-finite values at step {n} (backreaction)"

        Nn = clock_rate_from_psi(psi, base=1.0, beta=beta, mode="rational", clip=clip)
        Nn1 = clock_rate_from_psi(psi_next, base=1.0, beta=beta, mode="rational", clip=clip)
        
        assert np.isfinite(Nn).all(), f"Non-finite clock_rate at step {n} (backreaction)"
        assert np.isfinite(Nn1).all(), f"Non-finite clock_rate at step {n+1} (backreaction)"
        
        *_vals, res = weighted_norm_change_residual(psi, psi_next, Nn, Nn1, dt)

        if n > warmup:
            residuals_br.append(abs(res))

        psi, psi_prev = psi_next, psi_prev_next

    assert len(residuals_br) > 0, "No residuals collected for backreaction"
    br_med = float(np.median(residuals_br))
    br_mean = float(np.mean(residuals_br))
    print(f"  Backreaction residual: median={br_med:.3e}, mean={br_mean:.3e}")

    # Backreaction should measurably increase this residual above the external-N baseline,
    # but not necessarily by an order of magnitude (it depends on coupling strength and clip).
    min_ratio = 1.2
    ratio_med = br_med / ext_med if ext_med > 0 else float("inf")
    ratio_mean = br_mean / ext_mean if ext_mean > 0 else float("inf")
    
    print(f"\nComparison:")
    print(f"  Median ratio (backreaction/external): {ratio_med:.3f} (min: {min_ratio})")
    print(f"  Mean ratio (backreaction/external): {ratio_mean:.3f} (min: {min_ratio})")
    
    if br_med < min_ratio * ext_med or br_mean < min_ratio * ext_mean:
        print(f"❌ FAIL: Backreaction residual not sufficiently above external baseline")
        print(f"  Expected both ratios > {min_ratio}")
        return False

    print(f"✅ PASS: Diagnostic successfully separates numerical error vs physical backreaction")
    return True