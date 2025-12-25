import numpy as np
from code.model import (
    laplacian_iso,
    init_phase_leapfrog,
    step_phase_leapfrog,
    clock_rate_from_psi,
)
from code.measurements import precompute_radial_reduction, detect_front_outermost


def test_laplacian_constant_zero():
    """Constant field should have (near) zero isotropic Laplacian (periodic BC)."""
    f = np.ones((16, 16), dtype=float)
    L = laplacian_iso(f)
    assert L.shape == f.shape
    assert np.allclose(L, 0.0, atol=1e-12)


def test_clock_rate_beta_zero():
    """With beta=0 the clock rate should equal the provided base everywhere."""
    psi = np.ones((8, 8), dtype=np.complex128)
    cr = clock_rate_from_psi(psi, base=1.0, beta=0.0, mode="rational", clip=(0.05, 2.0))
    assert cr.shape == psi.shape
    assert np.allclose(cr, 1.0)


def test_leapfrog_step_shapes_and_finite():
    """Init and single-step should preserve shapes and produce finite arrays."""
    psi = np.zeros((12, 12), dtype=np.complex128)
    psi[6, 6] = 1.0 + 0.0j
    dt = 0.01
    omega0 = 0.0
    kappa = 1.0

    psi_prev = init_phase_leapfrog(psi, dt, omega0, kappa)
    psi_next, psi_prev2 = step_phase_leapfrog(psi, psi_prev, dt, omega0, kappa)

    assert psi_next.shape == psi.shape
    assert psi_prev2.shape == psi.shape
    assert np.isfinite(psi_next).all()
    assert np.isfinite(psi_prev2).all()


def test_radial_cache_and_front_detection_empty():
    """Radial helpers should handle zero intensity without crashing and return None."""
    shape = (32, 32)
    center = (16, 16)
    cache = precompute_radial_reduction(shape, center)
    assert "order" in cache and "starts" in cache and "r_vals" in cache

    I = np.zeros(shape, dtype=float)
    r = detect_front_outermost(I, cache, threshold=1e-8)
    assert r is None
