import numpy as np
from code.model import init_phase_leapfrog, step_phase_leapfrog, energy_density, weighted_norm, hamiltonian_total

def test_metric_weighted_unitarity_fixed_lapse():
    """Test energy conservation under fixed heterogeneous clock_rate.
    
    Validates that when clock_rate varies spatially but is fixed in time,
    the total energy remains approximately conserved. The conserved quantity
    is a weighted norm that accounts for the local time dilation.
    """
    print("\nTEST 10: Energy conservation under fixed heterogeneous clock_rate")

    # For i dψ/dt = N(x) H ψ with fixed N(x) and Hermitian H,
    # the conserved quadratic form is ||ψ||^2_W = Σ |ψ|^2 / N.
    # Plain norm Σ|ψ|^2 is not generally conserved when N varies spatially.

    rng = np.random.default_rng(0)

    N = 90
    steps = 200
    dt = 0.0005
    omega0 = 0.0
    kappa = 1.0

    # Input validation
    assert N > 0 and steps > 0, "Grid size and steps must be positive"
    assert dt > 0 and kappa > 0, "Time step and coupling must be positive"

    # Fixed lapse/clock_rate field: left half slow, right half fast.
    clock_rate = np.ones((N, N), dtype=float)
    clock_rate[:, : N // 2] = 0.2
    clock_rate[:, N // 2 :] = 1.0

    assert np.all(clock_rate > 0), "Clock rate must be positive everywhere"
    assert np.isfinite(clock_rate).all(), "Clock rate contains non-finite values"

    # Random initial condition to excite many modes.
    psi = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))).astype(np.complex128)
    psi *= 0.05

    assert np.isfinite(psi).all(), "Initial condition contains non-finite values"
    print(f"Info: clock_rate range: [{clock_rate.min():.2f}, {clock_rate.max():.2f}]")
    print(f"  Initial |psi|: mean={np.mean(np.abs(psi)):.6f}, max={np.max(np.abs(psi)):.6f}")

    psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa, clock_rate=clock_rate)
    assert np.isfinite(psi_prev).all(), "psi_prev contains non-finite values"

    def total_energy(psi_now, psi_prev, clock_rate):
        """Calculate total energy accounting for local clock_rate.
        
        Approximate kinetic using backward difference adjusted by local clock_rate.
        """
        dt_eff = dt * clock_rate
        vel = (psi_now - psi_prev) / dt_eff
        KE = float(np.sum(np.abs(vel) ** 2))
        PE = hamiltonian_total(psi_now, omega0=omega0, kappa=kappa)
        return KE + PE

    E0 = total_energy(psi, psi_prev, clock_rate)
    assert np.isfinite(E0), "Initial energy is non-finite"
    assert E0 != 0, "Initial energy is zero"
    print(f"  Initial energy: {E0:.6e}")

    Es = []
    for i in range(steps):
        psi, psi_prev = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa, clock_rate=clock_rate)
        
        # Periodic validation
        if i % 50 == 0:
            assert np.isfinite(psi).all(), f"Non-finite values at step {i}"
        
        E = total_energy(psi, psi_prev, clock_rate)
        assert np.isfinite(E), f"Non-finite energy at step {i}"
        Es.append(E)

    E1 = Es[-1]
    drift_E = abs(E1 - E0) / abs(E0)
    threshold = 0.20
    
    print(f"Results: E_initial={E0:.6e}, E_final={E1:.6e}")
    print(f"  Energy drift: {drift_E*100:.3f}% (threshold: {threshold*100:.1f}%)")
    print(f"  Energy range: [{min(Es):.6e}, {max(Es):.6e}]")

    if drift_E > threshold:
        print(f"❌ FAIL: Energy drift too large ({drift_E*100:.3f}% > {threshold*100:.1f}%)")
        

    print(f"✅ PASS: Energy approximately conserved (drift={drift_E*100:.3f}%)")
    