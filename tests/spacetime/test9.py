import numpy as np
from code.model import (
    init_phase_leapfrog,
    step_phase_leapfrog,
    init_phase_leapfrog_backreacting,
    step_phase_leapfrog_backreacting,
    curvature_proxy,
    clock_rate_from_psi,
    energy_density,
)

def test_backreaction_stability_and_effect():
    """Test curvature backreaction stability and effects.
    
    Validates that dynamic clock_rate computed from field curvature:
    1. Remains numerically stable (no NaN/Inf, bounded amplitude)
    2. Respects clip bounds
    3. Meaningfully changes evolution compared to no backreaction
    4. Correctly maps high curvature to low clock_rate
    5. Preserves energy-like quantities within reasonable bounds
    """
    print("\nTEST 9: Curvature backreaction (dynamic time dilation) stability and effects")

    N = 80
    steps = 2500
    dt = 0.002
    omega0 = 0.0
    kappa = 1.0

    # Input validation
    assert N > 0 and steps > 0, "Grid size and steps must be positive"
    assert dt > 0 and kappa > 0, "Time step and coupling must be positive"

    # Localized disturbance to generate curvature gradients.
    psi0 = np.zeros((N, N), dtype=np.complex128)
    psi0[N // 2 - 3, N // 2 - 18] = 1.0 + 0.0j
    psi0[N // 2 + 3, N // 2 - 18] = 1.0 + 0.0j

    # Reference run (no backreaction).
    print("Info: Running reference simulation (no backreaction)...")
    psi_ref = psi0.copy()
    psi_prev_ref = init_phase_leapfrog(psi_ref, dt, omega0, kappa, clock_rate=1.0)
    
    for i in range(steps):
        psi_ref, psi_prev_ref = step_phase_leapfrog(psi_ref, psi_prev_ref, dt, omega0, kappa, clock_rate=1.0)
        if i % 500 == 0:
            assert np.isfinite(psi_ref).all(), f"Non-finite values in reference at step {i}"

    assert np.isfinite(psi_ref).all(), "Non-finite values in final reference state"
    print("  Reference simulation completed")

    # Backreacting run.
    beta = 0.6
    clip = (0.05, 1.0)

    assert beta > 0, "Beta must be positive"
    assert clip[0] > 0 and clip[1] > clip[0], "Invalid clip bounds"

    print(f"Info: Running backreaction simulation (beta={beta}, clip={clip})...")
    psi = psi0.copy()
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

    min_rates = []
    max_rates = []
    mean_rates = []
    sample_interval = 200

    for n in range(steps):
        psi, psi_prev = step_phase_leapfrog_backreacting(
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

        if (n % sample_interval) == 0:
            # Early stability check
            if not np.isfinite(psi).all():
                print(f"❌ FAIL: NaN/Inf encountered at step {n}")
                return False
            
            rate = clock_rate_from_psi(
                psi,
                base=1.0,
                beta=beta,
                mode="rational",
                clip=clip,
            )
            assert np.isfinite(rate).all(), f"Non-finite clock_rate at step {n}"
            
            min_rates.append(float(np.min(rate)))
            max_rates.append(float(np.max(rate)))
            mean_rates.append(float(np.mean(rate)))

    print("  Backreaction simulation completed")

    # 1) Stability checks.
    print("\nValidation checks:")
    if not np.isfinite(psi).all():
        print("❌ FAIL: NaN/Inf encountered in final state")
        return False

    amp = float(np.max(np.abs(psi)))
    amp_threshold = 10.0
    if amp > amp_threshold:
        print(f"❌ FAIL: Unstable amplitude growth (max|psi|={amp:.3f} > {amp_threshold})")
        return False
    print(f"  1. Amplitude stable: max|psi|={amp:.3f}")

    # 2) Clock-rate is clipped and generally < 1 when curvature exists.
    if min(min_rates) < clip[0] - 1e-12 or max(max_rates) > clip[1] + 1e-12:
        print("❌ FAIL: clock_rate violated clip bounds")
        print(f"  min_rates={min_rates}")
        print(f"  max_rates={max_rates}")
        return False
    print(f"  2. Clock rate within bounds: [{min(min_rates):.4f}, {max(max_rates):.4f}]")

    mean_clock = np.mean(mean_rates)
    if mean_clock >= 1.0:
        print(f"❌ FAIL: Expected mean clock_rate < 1 under positive curvature (got {mean_clock:.4f})")
        return False
    print(f"  3. Mean clock rate reduced: {mean_clock:.4f} < 1.0")

    # 3) Backreaction meaningfully changes evolution.
    diff = float(np.linalg.norm(psi - psi_ref))
    ref_norm = float(np.linalg.norm(psi_ref))
    rel = diff / ref_norm if ref_norm != 0 else float("inf")
    min_effect = 0.2
    
    print(f"  4. Relative difference vs no-backreaction: {rel*100:.3f}% (min: {min_effect*100:.1f}%)")

    if rel < min_effect:
        print(f"❌ FAIL: Backreaction effect too small ({rel*100:.3f}% < {min_effect*100:.1f}%)")
        return False

    # 4) Sanity: higher curvature should imply lower clock rate (by construction).
    curv = curvature_proxy(psi)
    rate = clock_rate_from_psi(psi, base=1.0, beta=beta, mode="rational", clip=clip)
    
    assert np.isfinite(curv).all(), "Non-finite curvature values"
    assert np.isfinite(rate).all(), "Non-finite final clock_rate values"
    
    flat = curv.ravel()
    q_hi = np.quantile(flat, 0.95)
    q_lo = np.quantile(flat, 0.05)
    mean_hi = float(np.mean(rate[curv >= q_hi]))
    mean_lo = float(np.mean(rate[curv <= q_lo]))
    
    print(f"  5. Clock rate vs curvature:")
    print(f"     High curvature (top 5%): clock_rate={mean_hi:.4f}")
    print(f"     Low curvature (bottom 5%): clock_rate={mean_lo:.4f}")

    if not (mean_hi < mean_lo):
        print(f"❌ FAIL: Expected high-curvature => smaller clock_rate")
        print(f"  But got mean_hi={mean_hi:.4f} >= mean_lo={mean_lo:.4f}")
        return False

    # 5) Keep norm-ish bounded (not a strict conservation claim under local lapse).
    norm = float(np.sum(energy_density(psi)))
    norm_threshold = 1e6
    
    if not np.isfinite(norm):
        print("❌ FAIL: Non-finite norm")
        return False
    
    if norm > norm_threshold:
        print(f"❌ FAIL: Norm too large ({norm:.3e} > {norm_threshold:.3e})")
        return False
    
    print(f"  6. Norm bounded: {norm:.3e}")

    print("\n✅ PASS: Backreaction is stable and meaningfully affects dynamics")
    return True