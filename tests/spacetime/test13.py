import numpy as np
from code.model import init_coupled_gravity_matter, step_coupled_gravity_matter, gravity_source_from_psi
from code.measurements import estimate_angular_frequency

def test_coupled_gravity_delayed_frequency_shift():
    print("TEST 13: Coupled gravity causes delayed frequency shift at a probe")

    # Construct a frozen "mass" (density source) plus a probe oscillator.
    # Gravity (clock_rate) propagates at finite speed c_g.
    # The probe's observed coordinate-time frequency should remain unchanged
    # until the gravity wave reaches it.

    N = 121
    dt = 0.002
    steps = 26000  # total t=52.0

    c_g = 1.0
    r_probe = 35

    cy, cx = N // 2, N // 2
    probe = (cy, cx + r_probe)

    # Build a fixed "mass" configuration (gravity source) that excludes the probe.
    # This avoids the probe gravitating itself (which would change clock_rate immediately).
    mass_psi = np.zeros((N, N), dtype=np.complex128)
    mass_psi[cy, cx] = 3.0 + 0.0j
    mass_psi[cy - 1, cx] = 2.0 + 0.0j
    mass_psi[cy + 1, cx] = 2.0 + 0.0j
    mass_psi[cy, cx - 1] = 2.0 + 0.0j
    mass_psi[cy, cx + 1] = 2.0 + 0.0j

    gravity_source = gravity_source_from_psi(
        mass_psi,
        strength=0.6,
        kind="density",
        subtract_mean=False,
    )

    # Matter field includes the probe oscillator amplitude.
    psi0 = mass_psi.copy()
    psi0[probe] = 1.0 + 0.0j

    # Make only the probe rotate (kappa=0 so sites are decoupled).
    Omega = 3.0
    omega = np.zeros((N, N), dtype=float)
    omega[probe] = Omega
    kappa = 0.0

    psi, psi_prev, clock, clock_prev = init_coupled_gravity_matter(
        psi0,
        dt,
        omega,
        kappa,
        clock_rate0=np.ones((N, N), dtype=float),
        gravity_c_g=c_g,
        gravity_gamma=0.0,
        gravity_mu=0.0,
        gravity_base=1.0,
        gravity_source=gravity_source,
        gravity_source_strength=0.0,
        gravity_source_kind="density",
        gravity_subtract_mean=False,
        gravity_clip=(0.05, 2.0),
    )

    # Seed a rotating initial condition at the probe for clear frequency estimation.
    psi_prev[probe] = psi[probe] * np.exp(1j * np.sqrt(Omega) * dt * float(clock[probe]))

    times = []
    z_probe = []
    N_probe = []

    for n in range(steps):
        t = n * dt
        psi, psi_prev, clock, clock_prev = step_coupled_gravity_matter(
            psi,
            psi_prev,
            clock,
            clock_prev,
            dt,
            omega,
            kappa,
            gravity_source=gravity_source,
            gravity_c_g=c_g,
            gravity_gamma=0.0,
            gravity_mu=0.0,
            gravity_base=1.0,
            gravity_source_strength=0.0,
            gravity_source_kind="density",
            gravity_subtract_mean=False,
            gravity_clip=(0.05, 2.0),
        )

        # Sample every step for good phase fit.
        times.append(t)
        z_probe.append(psi[probe])
        N_probe.append(float(clock[probe]))

    times = np.array(times)
    z_probe = np.array(z_probe)
    N_probe = np.array(N_probe)

    # Early window: c_g * t << r_probe
    early = times < (r_probe / c_g) * 0.25
    # Late window: c_g * t > r_probe
    late = times > (r_probe / c_g) * 1.2

    if early.sum() < 50 or late.sum() < 50:
        print("❌ FAIL: Not enough samples in early/late windows")
        return False

    w_early = estimate_angular_frequency(z_probe[early], times[early])
    w_late = estimate_angular_frequency(z_probe[late], times[late])

    N_early = float(np.mean(N_probe[early]))
    N_late = float(np.mean(N_probe[late]))

    # For second-order dynamics ψ_tt = -Ω ψ (kappa=0), the observed angular
    # frequency magnitude scales as sqrt(Ω) * N (clock_rate). Use that mapping.
    w_early_exp = -np.sqrt(Omega) * N_early
    w_late_exp = -np.sqrt(Omega) * N_late

    print(f"N_probe early={N_early:.6f}, late={N_late:.6f}")
    print(f"ω_probe early: hat={w_early:.4f}, exp~={w_early_exp:.4f}")
    print(f"ω_probe late : hat={w_late:.4f}, exp~={w_late_exp:.4f}")

    # 1) No early response
    if abs(N_early - 1.0) > 1e-4:
        print("❌ FAIL: Probe clock_rate changed too early")
        return False

    # 2) Late response exists
    if abs(N_late - 1.0) < 1e-3:
        print("❌ FAIL: Probe clock_rate did not change after wave arrival")
        return False

    # 3) Frequency shift matches change in clock_rate (within tolerance)
    # Allow some wiggle because clock_rate at probe may still be oscillating.
    rel_err_early = abs(w_early - w_early_exp) / (abs(w_early_exp) if w_early_exp != 0 else 1.0)
    rel_err_late = abs(w_late - w_late_exp) / (abs(w_late_exp) if w_late_exp != 0 else 1.0)

    print(f"rel_err early={rel_err_early*100:.2f}%")
    print(f"rel_err late ={rel_err_late*100:.2f}%")

    if rel_err_early > 0.02 or rel_err_late > 0.10:
        print("❌ FAIL: Frequency does not track clock_rate")
        return False

    # 4) Detect the shift
    if abs(w_late - w_early) < 1e-3:
        print("❌ FAIL: No measurable frequency shift")
        return False

    print("✅ PASS: Probe frequency shifts only after gravity arrival")
    return True