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
    print("TEST 9: Curvature backreaction (dynamic time dilation) is stable and affects evolution")

    N = 80
    steps = 2500
    dt = 0.01
    omega0 = 0.0
    kappa = 1.0

    # Localized disturbance to generate curvature gradients.
    psi0 = np.zeros((N, N), dtype=np.complex128)
    psi0[N // 2 - 3, N // 2 - 18] = 1.0 + 0.0j
    psi0[N // 2 + 3, N // 2 - 18] = 1.0 + 0.0j

    # Reference run (no backreaction).
    psi_ref = psi0.copy()
    v_ref = init_phase_leapfrog(psi_ref, dt, omega0, kappa, clock_rate=1.0)
    for _ in range(steps):
        psi_ref, v_ref = step_phase_leapfrog(psi_ref, v_ref, dt, omega0, kappa, clock_rate=1.0)

    # Backreacting run.
    beta = 0.6
    clip = (0.05, 1.0)

    psi = psi0.copy()
    v_half = init_phase_leapfrog_backreacting(
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

    for n in range(steps):
        psi, v_half = step_phase_leapfrog_backreacting(
            psi,
            v_half,
            dt,
            omega0,
            kappa,
            base_clock_rate=1.0,
            curvature_beta=beta,
            curvature_mode="rational",
            curvature_clip=clip,
        )

        if (n % 200) == 0:
            rate = clock_rate_from_psi(
                psi,
                base=1.0,
                beta=beta,
                mode="rational",
                clip=clip,
            )
            min_rates.append(float(np.min(rate)))
            max_rates.append(float(np.max(rate)))
            mean_rates.append(float(np.mean(rate)))

    # 1) Stability checks.
    if not np.isfinite(psi).all():
        print("❌ FAIL: NaN/Inf encountered")
        return False

    amp = float(np.max(np.abs(psi)))
    if amp > 10.0:
        print(f"❌ FAIL: Unstable amplitude growth (max|psi|={amp:.3f})")
        return False

    # 2) Clock-rate is clipped and generally < 1 when curvature exists.
    if min(min_rates) < clip[0] - 1e-12 or max(max_rates) > clip[1] + 1e-12:
        print("❌ FAIL: clock_rate violated clip bounds")
        print(f"min_rates={min_rates}")
        print(f"max_rates={max_rates}")
        return False

    if np.mean(mean_rates) >= 1.0:
        print("❌ FAIL: Expected mean clock_rate < 1 under positive curvature")
        return False

    # 3) Backreaction meaningfully changes evolution.
    diff = float(np.linalg.norm(psi - psi_ref))
    ref_norm = float(np.linalg.norm(psi_ref))
    rel = diff / (ref_norm if ref_norm != 0 else 1.0)
    print(f"relative difference vs no-backreaction = {rel*100:.3f}%")

    if rel < 0.2:
        print("❌ FAIL: Backreaction effect too small (did it engage?)")
        return False

    # 4) Sanity: higher curvature should imply lower clock rate (by construction).
    curv = curvature_proxy(psi)
    rate = clock_rate_from_psi(psi, base=1.0, beta=beta, mode="rational", clip=clip)
    flat = curv.ravel()
    q_hi = np.quantile(flat, 0.95)
    q_lo = np.quantile(flat, 0.05)
    mean_hi = float(np.mean(rate[curv >= q_hi]))
    mean_lo = float(np.mean(rate[curv <= q_lo]))
    print(f"mean clock_rate (top 5% curvature) = {mean_hi:.4f}")
    print(f"mean clock_rate (bottom 5% curvature) = {mean_lo:.4f}")

    if not (mean_hi < mean_lo):
        print("❌ FAIL: Expected high-curvature region to have smaller clock_rate")
        return False

    # 5) Keep norm-ish bounded (not a strict conservation claim under local lapse).
    norm = float(np.sum(energy_density(psi)))
    if not np.isfinite(norm) or norm > 1e6:
        print("❌ FAIL: Norm blew up")
        return False

    print("✅ PASS: Backreaction is stable and changes dynamics")
    return True