import numpy as np
from code.model import init_phase_leapfrog, step_phase_leapfrog
from code.measurements import cardinal_diagonal_peak_delta

def test_cardinal_vs_diagonal():
    """Test isotropy between cardinal and diagonal directions.
    
    Validates that the peak position is similar along cardinal (horizontal/vertical)
    and diagonal directions, indicating an isotropic Laplacian operator.
    """
    print("\nTEST 6: Cardinal vs Diagonal isotropy")

    N, steps, dt = 150, 600, 0.02
    kappa = 1.0
    omega0 = 0.0
    
    # Input validation
    assert N > 0 and steps > 0, "Grid size and steps must be positive"
    assert dt > 0 and kappa > 0, "Time step and coupling must be positive"
    
    psi = np.zeros((N, N), dtype=np.complex128)
    center = N // 2
    psi[center, center] = 1.0 + 0.0j
    
    psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa)

    for i in range(steps):
        psi, psi_prev = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa)
        if i % 100 == 0:
            assert np.isfinite(psi).all(), f"Non-finite values at step {i}"

    I = np.abs(psi) ** 2
    result = cardinal_diagonal_peak_delta(I, (center, center))
    
    threshold = 4
    delta = result["delta"]
    
    print(f"Results: cardinal peak = {result['card']}, diagonal peak = {result['diag']}")
    print(f"  Peak position delta = {delta} (threshold: {threshold})")
    
    if delta < threshold:
        print(f"✅ PASS: Cardinal/diagonal peak positions similar (Δ={delta} < {threshold})")
        return True
    else:
        print(f"❌ FAIL: Significant cardinal/diagonal difference (Δ={delta} >= {threshold})")
        return False