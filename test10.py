import numpy as np
from model import init_phase_leapfrog, step_phase_leapfrog, energy_density, weighted_norm


def test_metric_weighted_unitarity_fixed_lapse():
    print("TEST 10: Metric-weighted unitarity under fixed heterogeneous clock_rate")

    # For i dψ/dt = N(x) H ψ with fixed N(x) and Hermitian H,
    # the conserved quadratic form is ||ψ||^2_W = Σ |ψ|^2 / N.
    # Plain norm Σ|ψ|^2 is not generally conserved when N varies spatially.

    rng = np.random.default_rng(0)

    N = 90
    steps = 6000
    dt = 0.01
    omega0 = 0.0
    kappa = 1.0

    # Fixed lapse/clock_rate field: left half slow, right half fast.
    clock_rate = np.ones((N, N), dtype=float)
    clock_rate[:, : N // 2] = 0.2
    clock_rate[:, N // 2 :] = 1.0

    # Random initial condition to excite many modes.
    psi = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))).astype(np.complex128)
    psi *= 0.05

    v_half = init_phase_leapfrog(psi, dt, omega0, kappa, clock_rate=clock_rate)

    norm0 = float(np.sum(energy_density(psi)))
    wnorm0 = weighted_norm(psi, clock_rate=clock_rate)

    norms = []
    wnorms = []
    for _ in range(steps):
        psi, v_half = step_phase_leapfrog(psi, v_half, dt, omega0, kappa, clock_rate=clock_rate)
        norms.append(float(np.sum(energy_density(psi))))
        wnorms.append(weighted_norm(psi, clock_rate=clock_rate))

    norm1 = norms[-1]
    wnorm1 = wnorms[-1]

    drift_norm = abs(norm1 - norm0) / (abs(norm0) if norm0 != 0 else 1.0)
    drift_wnorm = abs(wnorm1 - wnorm0) / (abs(wnorm0) if wnorm0 != 0 else 1.0)

    print(f"plain norm drift   = {drift_norm*100:.3f}%")
    print(f"weighted norm drift= {drift_wnorm*100:.3f}%")

    # We expect the weighted drift to be small (numerical error), and
    # substantially smaller than the unweighted drift.
    if drift_wnorm > 0.02:
        print("❌ FAIL: Weighted norm drift too large")
        return False

    if drift_norm < 5 * drift_wnorm:
        print("❌ FAIL: Plain norm did not drift more than weighted norm (unexpected)")
        return False

    print("✅ PASS: Weighted norm is the right conserved quantity under fixed lapse")
    return True


if __name__ == "__main__":
    test_metric_weighted_unitarity_fixed_lapse()
