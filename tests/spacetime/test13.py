import numpy as np
from code.model import init_coupled_gravity_matter, step_coupled_gravity_matter, gravity_source_from_psi
from code.measurements import estimate_angular_frequency

def test_coupled_gravity_delayed_frequency_shift():
    """Test coupled gravity-matter system with delayed frequency shift.
    
    Validates that:
    1. A gravitational disturbance propagates at finite speed c_g
    2. A distant probe oscillator remains unaffected until the gravity wave arrives
    3. After arrival, the probe's frequency shifts according to local clock_rate change
    4. The frequency shift matches the theoretical prediction from time dilation
    """
    print("\nTEST 13: Coupled gravity causes delayed frequency shift at a probe")

    # Construct a frozen "mass" (density source) plus a probe oscillator.
    # Gravity (clock_rate) propagates at finite speed c_g.
    # The probe's observed coordinate-time frequency should remain unchanged
    # until the gravity wave reaches it.

    N = 121
    dt = 0.002
    steps = 26000  # total t=52.0

    c_g = 1.0
    r_probe = 35

    # Input validation
    assert N > 0 and steps > 0, "Grid size and steps must be positive"
    assert dt > 0 and c_g > 0, "Time step and propagation speed must be positive"
    assert r_probe > 0 and r_probe < N // 2, "Probe distance must be valid"

    cy, cx = N // 2, N // 2
    probe = (cy, cx + r_probe)
    
    assert 0 <= probe[0] < N and 0 <= probe[1] < N, "Probe position out of bounds"
    
    print(f"Info: Grid size={N}, dt={dt}, steps={steps} (t={steps*dt:.1f})")
    print(f"  c_g={c_g}, probe distance={r_probe}, probe position={probe}")

    # Build a fixed "mass" configuration (gravity source) that excludes the probe.
    # This avoids the probe gravitating itself (which would change clock_rate immediately).
    mass_psi = np.zeros((N, N), dtype=np.complex128)
    mass_psi[cy, cx] = 3.0 + 0.0j
    mass_psi[cy - 1, cx] = 2.0 + 0.0j
    mass_psi[cy + 1, cx] = 2.0 + 0.0j
    mass_psi[cy, cx - 1] = 2.0 + 0.0j
    mass_psi[cy, cx + 1] = 2.0 + 0.0j

    assert np.isfinite(mass_psi).all(), "Mass configuration contains non-finite values"
    assert np.abs(mass_psi[probe]) < 1e-10, "Probe should not be part of mass source"

    gravity_source = gravity_source_from_psi(
        mass_psi,
        strength=0.6,
        kind="density",
        subtract_mean=False,
    )
    
    assert np.isfinite(gravity_source).all(), "Gravity source contains non-finite values"
    print(f"  Gravity source: max={np.max(np.abs(gravity_source)):.6f}")

    # Matter field includes the probe oscillator amplitude.
    psi0 = mass_psi.copy()
    psi0[probe] = 1.0 + 0.0j

    # Make only the probe rotate (kappa=0 so sites are decoupled).
    Omega = 3.0
    omega = np.zeros((N, N), dtype=float)
    omega[probe] = Omega
    kappa = 0.0

    assert Omega > 0, "Omega must be positive"
    assert kappa == 0.0, "Test requires decoupled sites (kappa=0)"
    assert omega[probe] == Omega, "Probe frequency not set correctly"
    
    print(f"  Probe oscillator: Ω={Omega}, kappa={kappa}")

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

    # Validate initialization
    assert np.isfinite(psi).all(), "psi contains non-finite values after init"
    assert np.isfinite(psi_prev).all(), "psi_prev contains non-finite values after init"
    assert np.isfinite(clock).all(), "clock contains non-finite values after init"
    assert np.isfinite(clock_prev).all(), "clock_prev contains non-finite values after init"

    # Seed a rotating initial condition at the probe for clear frequency estimation.
    psi_prev[probe] = psi[probe] * np.exp(1j * np.sqrt(Omega) * dt * float(clock[probe]))

    print("  Initialization complete")

    times = []
    z_probe = []
    N_probe = []

    print(f"\nRunning coupled evolution for {steps} steps...")
    check_interval = 5000
    
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

        # Validation at checkpoints
        if n % check_interval == 0:
            assert np.isfinite(psi).all(), f"Non-finite psi at step {n}"
            assert np.isfinite(clock).all(), f"Non-finite clock at step {n}"
            print(f"  Step {n}/{steps} (t={t:.1f}): probe clock_rate={clock[probe]:.6f}")

        # Sample every step for good phase fit.
        times.append(t)
        z_probe.append(psi[probe])
        N_probe.append(float(clock[probe]))

    times = np.array(times)
    z_probe = np.array(z_probe)
    N_probe = np.array(N_probe)

    # Early window: c_g * t << r_probe
    t_early_max = (r_probe / c_g) * 0.25
    # Late window: c_g * t > r_probe
    t_late_min = (r_probe / c_g) * 1.2
    
    early = times < t_early_max
    late = times > t_late_min

    print(f"\nAnalysis windows:")
    print(f"  Early: t < {t_early_max:.2f} ({early.sum()} samples)")
    print(f"  Late: t > {t_late_min:.2f} ({late.sum()} samples)")

    min_samples = 50
    if early.sum() < min_samples or late.sum() < min_samples:
        print(f"❌ FAIL: Insufficient samples in windows (need at least {min_samples} each)")
        return False

    w_early = estimate_angular_frequency(z_probe[early], times[early])
    w_late = estimate_angular_frequency(z_probe[late], times[late])

    N_early = float(np.mean(N_probe[early]))
    N_late = float(np.mean(N_probe[late]))

    # For second-order dynamics ψ_tt = -Ω ψ (kappa=0), the observed angular
    # frequency magnitude scales as sqrt(Ω) * N (clock_rate). Use that mapping.
    w_early_exp = -np.sqrt(Omega) * N_early
    w_late_exp = -np.sqrt(Omega) * N_late

    print(f"\nResults:")
    print(f"  Early phase:")
    print(f"    Mean clock_rate: {N_early:.6f}")
    print(f"    Frequency: ω={w_early:.4f} (expected ≈ {w_early_exp:.4f})")
    print(f"  Late phase:")
    print(f"    Mean clock_rate: {N_late:.6f}")
    print(f"    Frequency: ω={w_late:.4f} (expected ≈ {w_late_exp:.4f})")

    # 1) No early response (causality)
    threshold_early = 1e-4
    if abs(N_early - 1.0) > threshold_early:
        print(f"\n❌ FAIL: Probe clock_rate changed too early")
        print(f"  |N_early - 1.0| = {abs(N_early - 1.0):.6f} > {threshold_early}")
        return False
    print(f"\n✔ Check 1: No early response (|N_early - 1.0| = {abs(N_early - 1.0):.6f})")

    # 2) Late response exists
    threshold_late = 1e-3
    if abs(N_late - 1.0) < threshold_late:
        print(f"❌ FAIL: Probe clock_rate did not change after wave arrival")
        print(f"  |N_late - 1.0| = {abs(N_late - 1.0):.6f} < {threshold_late}")
        return False
    print(f"✔ Check 2: Late response detected (|N_late - 1.0| = {abs(N_late - 1.0):.6f})")

    # 3) Frequency shift matches change in clock_rate (within tolerance)
    # Allow some wiggle because clock_rate at probe may still be oscillating.
    rel_err_early = abs(w_early - w_early_exp) / abs(w_early_exp) if w_early_exp != 0 else float("inf")
    rel_err_late = abs(w_late - w_late_exp) / abs(w_late_exp) if w_late_exp != 0 else float("inf")

    print(f"✔ Check 3: Frequency tracking errors:")
    print(f"  Early: {rel_err_early*100:.2f}% (threshold: 2%)")
    print(f"  Late: {rel_err_late*100:.2f}% (threshold: 10%)")

    if rel_err_early > 0.02 or rel_err_late > 0.10:
        print(f"❌ FAIL: Frequency does not track clock_rate")
        return False

    # 4) Detect the shift
    freq_shift = abs(w_late - w_early)
    min_shift = 1e-3
    
    print(f"✔ Check 4: Frequency shift: |Δω| = {freq_shift:.6f} (min: {min_shift})")
    
    if freq_shift < min_shift:
        print(f"❌ FAIL: No measurable frequency shift ({freq_shift:.6f} < {min_shift})")
        return False

    print("\n✅ PASS: Probe frequency shifts only after gravity arrival")
    return True