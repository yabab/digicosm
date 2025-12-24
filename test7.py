import numpy as np
from model import init_phase_leapfrog, step_phase_leapfrog
from measurements import estimate_angular_frequency


def test_global_time_dilation_frequency_scaling():
    print("TEST 7: Global time dilation scales frequency")

    # Decoupled sites: i dψ/dτ = ω ψ, with dτ = dt * clock_rate.
    # Therefore dψ/dt = -i (ω * clock_rate) ψ.
    omega0 = 3.0
    kappa = 0.0
    dt = 0.01
    steps = 6000

    for clock_rate in (1.0, 0.4):
        psi = np.ones((8, 8), dtype=np.complex128)
        v_half = init_phase_leapfrog(psi, dt, omega0, kappa, clock_rate=clock_rate)

        times = []
        zs = []
        for n in range(steps):
            t = n * dt
            psi, v_half = step_phase_leapfrog(
                psi, v_half, dt, omega0, kappa, clock_rate=clock_rate
            )
            # Sample a single site.
            if n > steps // 4:
                times.append(t)
                zs.append(psi[0, 0])

        omega_hat = estimate_angular_frequency(np.array(zs), np.array(times))
        # Convention: with i dψ/dτ = ω ψ and dτ = dt * clock_rate,
        # we get ψ(t) = exp(-i ω clock_rate t) so phase slope is -ω clock_rate.
        omega_expected = -omega0 * clock_rate
        rel_err = abs(omega_hat - omega_expected) / (abs(omega_expected) if omega_expected != 0 else 1.0)

        print(f"clock_rate={clock_rate:.3f} -> ω_hat={omega_hat:.4f}, ω_exp={omega_expected:.4f}, rel_err={rel_err*100:.2f}%")

        if rel_err > 0.02:
            print("❌ FAIL: Frequency scaling mismatch")
            return False

    print("✅ PASS: Frequency scales with global clock_rate")
    return True


if __name__ == "__main__":
    test_global_time_dilation_frequency_scaling()
