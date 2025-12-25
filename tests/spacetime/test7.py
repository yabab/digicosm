import numpy as np
from code.model import init_phase_leapfrog, step_phase_leapfrog
from code.measurements import estimate_angular_frequency

def test_global_time_dilation_frequency_scaling():
    print("TEST 7: Global time dilation scales frequency")

    # Decoupled sites: i dψ/dτ = ω ψ, with dτ = dt * clock_rate.
    # Therefore dψ/dt = -i (ω * clock_rate) ψ.
    omega0 = 3.0
    kappa = 0.0
    dt = 0.002
    steps = 6000

    for clock_rate in (1.0, 0.4):
        psi = np.ones((8, 8), dtype=np.complex128)
        # initialize as a rotating mode with angular freq ~ sqrt(omega0)
        dt_eff = dt * clock_rate
        psi_prev = psi * np.exp(1j * np.sqrt(omega0) * dt_eff)

        times = []
        zs = []
        for n in range(steps):
            t = n * dt
            psi, psi_prev = step_phase_leapfrog(
                psi, psi_prev, dt, omega0, kappa, clock_rate=clock_rate
            )
            # Sample a single site.
            if n > steps // 4:
                times.append(t)
                zs.append(psi[0, 0])

        omega_hat = estimate_angular_frequency(np.array(zs), np.array(times))
        # For second-order dynamics ψ_tt = -ω ψ with proper-time τ scaled by clock_rate,
        # the observed angular frequency magnitude scales as sqrt(ω) * clock_rate.
        omega_expected = -np.sqrt(omega0) * clock_rate
        rel_err = abs(omega_hat - omega_expected) / (abs(omega_expected) if omega_expected != 0 else 1.0)

        print(f"clock_rate={clock_rate:.3f} -> ω_hat={omega_hat:.4f}, ω_exp={omega_expected:.4f}, rel_err={rel_err*100:.2f}%")

        if rel_err > 0.02:
            print("❌ FAIL: Frequency scaling mismatch")
            return False

    print("✅ PASS: Frequency scales with global clock_rate")
    return True