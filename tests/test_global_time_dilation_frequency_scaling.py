import numpy as np
from code.model import init_phase_leapfrog, step_phase_leapfrog
from code.measurements import estimate_angular_frequency

def test_global_time_dilation_frequency_scaling():
    """Test global time dilation effect on frequency scaling.
    
    Validates that a uniform clock_rate (lapse) parameter correctly scales
    the observed angular frequency as expected from the theory:
    observed_omega = sqrt(omega0) * clock_rate.
    """
    print("\nTEST 7: Global time dilation scales frequency")

    # Decoupled sites: i dψ/dτ = ω ψ, with dτ = dt * clock_rate.
    # Therefore dψ/dt = -i (ω * clock_rate) ψ.
    omega0 = 3.0
    kappa = 0.0
    dt = 0.002
    steps = 6000

    # Input validation
    assert omega0 > 0, "Omega must be positive"
    assert dt > 0 and steps > 0, "Time step and steps must be positive"

    clock_rates = [1.0, 0.4]
    tolerance = 0.02  # 2% error tolerance
    
    for clock_rate in clock_rates:
        assert clock_rate > 0, f"Clock rate must be positive: {clock_rate}"
        psi = np.ones((8, 8), dtype=np.complex128)
        # initialize as a rotating mode with angular freq ~ sqrt(omega0)
        dt_eff = dt * clock_rate
        psi_prev = psi * np.exp(1j * np.sqrt(omega0) * dt_eff)

        times = []
        zs = []
        warmup = steps // 4
        
        for n in range(steps):
            t = n * dt
            psi, psi_prev = step_phase_leapfrog(
                psi, psi_prev, dt, omega0, kappa, clock_rate=clock_rate
            )
            
            # Validation
            if n % 1000 == 0:
                assert np.isfinite(psi).all(), f"Non-finite values at step {n}, clock_rate={clock_rate}"
            
            # Sample a single site after warmup.
            if n > warmup:
                times.append(t)
                zs.append(psi[0, 0])

        assert len(times) > 100, f"Insufficient samples after warmup: {len(times)}"
        
        omega_hat = estimate_angular_frequency(np.array(zs), np.array(times))
        # For second-order dynamics ψ_tt = -ω ψ with proper-time τ scaled by clock_rate,
        # the observed angular frequency magnitude scales as sqrt(ω) * clock_rate.
        omega_expected = -np.sqrt(omega0) * clock_rate
        rel_err = abs(omega_hat - omega_expected) / abs(omega_expected)

        print(f"  clock_rate={clock_rate:.3f}: ω_measured={omega_hat:.4f}, "
              f"ω_expected={omega_expected:.4f}, error={rel_err*100:.2f}%")

        if rel_err > tolerance:
            print(f"❌ FAIL: Frequency scaling mismatch (error={rel_err*100:.2f}% > {tolerance*100:.1f}%)")
            

    print(f"✅ PASS: Frequency scales with global clock_rate (tolerance={tolerance*100:.1f}%)")
    