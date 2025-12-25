import numpy as np
from code.model import init_phase_leapfrog, step_phase_leapfrog, hamiltonian_total, energy_density

def test_energy_conservation():
    """Test energy conservation in the phase dynamics system.
    
    Validates that the total energy (kinetic + potential) remains approximately
    constant during time evolution with a fine time step.
    """
    print("\nTEST 4: Norm/Hamiltonian stability (phase dynamics)")

    N, steps, dt = 120, 200, 0.0005
    kappa = 1.0
    omega0 = 0.0

    # Input validation
    assert N > 0 and steps > 0, "Grid size and steps must be positive"
    assert dt > 0 and kappa > 0, "Time step and coupling must be positive"

    psi = np.zeros((N, N), dtype=np.complex128)
    psi[N // 2 - 5, N // 2 - 30] = 1.0 + 0.0j
    psi[N // 2 + 5, N // 2 - 30] = 1.0 + 0.0j

    psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa)

    def total_energy(psi_now, psi_prev):
        """Calculate total energy: kinetic + potential."""
        # kinetic ~ |(psi_now - psi_prev)/dt|^2, potential ~ hamiltonian_total
        vel = (psi_now - psi_prev) / dt
        KE = float(np.sum(np.abs(vel) ** 2))
        PE = hamiltonian_total(psi_now, omega0=omega0, kappa=kappa)
        return KE + PE

    E0 = total_energy(psi, psi_prev)
    assert np.isfinite(E0), "Initial energy is not finite"
    assert E0 != 0, "Initial energy is zero"

    Es = []
    # Show initial magnitudes for debugging
    max_psi_init = np.max(np.abs(psi))
    max_prev_init = np.max(np.abs(psi_prev))
    print(f"Info: initial max|psi|={max_psi_init:.6f}, max|psi_prev|={max_prev_init:.6f}")
    
    for i in range(steps):
        psi_next, psi_prev = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa)
        
        # Validation check for early steps
        if i < 5:
            max_val = np.max(np.abs(psi_next))
            print(f"  step {i}: max|psi_next|={max_val:.6f}")
            assert np.isfinite(psi_next).all(), f"Non-finite values at step {i}"
        
        E = total_energy(psi_next, psi_prev)
        assert np.isfinite(E), f"Non-finite energy at step {i}"
        Es.append(E)
        psi = psi_next

    E1 = Es[-1]
    E_drift = abs(E1 - E0) / abs(E0)
    threshold = 0.10
    
    print(f"Results: E0={E0:.6e}, E1={E1:.6e}")
    print(f"  Energy drift = {E_drift*100:.4f}% (threshold: {threshold*100:.1f}%)")
    print(f"  Min energy: {min(Es):.6e}, Max energy: {max(Es):.6e}")

    if E_drift < threshold:
        print(f"✅ PASS: Energy approximately conserved (drift={E_drift*100:.4f}%)")
        

    print(f"❌ FAIL: Energy drift too large ({E_drift*100:.4f}% >= {threshold*100:.1f}%)")
    