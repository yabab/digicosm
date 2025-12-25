import numpy as np
from code.model import (
    curvature_proxy,
    clock_rate_from_curvature,
    clock_rate_from_psi,
    gravity_source_from_psi,
    weighted_norm,
    init_clock_rate_wave,
    step_clock_rate_wave,
)


def test_curvature_proxy_zero_for_constant_field():
    psi = np.ones((16, 16), dtype=np.complex128)
    curv = curvature_proxy(psi)
    assert curv.shape == psi.shape
    assert np.allclose(curv, 0.0, atol=1e-12)


def test_clock_rate_from_curvature_modes_and_clip():
    # create a simple curvature field
    curv = np.zeros((8, 8), dtype=float)
    curv[0, 0] = 10.0

    # beta=0 should yield base everywhere
    r0 = clock_rate_from_curvature(curv, base=1.0, beta=0.0, mode="exp", clip=(0.05, 2.0))
    assert np.allclose(r0, 1.0)

    # exp mode: high curvature -> significantly reduced rate
    r_exp = clock_rate_from_curvature(curv, base=1.0, beta=0.5, mode="exp", clip=(0.0, 1.0))
    assert r_exp[0, 0] < 1.0
    assert r_exp.min() >= 0.0

    # rational mode behaves similarly but bounded by clip
    r_rat = clock_rate_from_curvature(curv, base=2.0, beta=1.0, mode="rational", clip=(0.1, 5.0))
    assert r_rat[0, 0] <= 2.0
    assert r_rat.min() >= 0.1

    # Unknown mode raises (only when beta != 0)
    try:
        clock_rate_from_curvature(curv, beta=0.5, mode="unknown")
        assert False, "Expected ValueError for unknown mode"
    except ValueError:
        pass


def test_clock_rate_from_psi_matches_composed():
    psi = np.zeros((6, 6), dtype=np.complex128)
    psi[3, 3] = 1.0 + 0.0j

    r1 = clock_rate_from_psi(psi, base=1.0, beta=0.3, mode="rational", clip=(0.01, 10.0))
    curv = curvature_proxy(psi)
    r2 = clock_rate_from_curvature(curv, base=1.0, beta=0.3, mode="rational", clip=(0.01, 10.0))
    assert np.allclose(r1, r2)


def test_gravity_source_and_subtract_mean():
    psi = np.zeros((10, 10), dtype=np.complex128)
    psi[5, 5] = 1.0 + 0.0j

    s1 = gravity_source_from_psi(psi, strength=2.0, kind="density", subtract_mean=False)
    assert s1.shape == psi.shape
    assert float(np.max(s1)) > 0

    s2 = gravity_source_from_psi(psi, strength=2.0, kind="density", subtract_mean=True)
    assert abs(float(np.mean(s2))) < 1e-12

    # curvature kind should also work
    s3 = gravity_source_from_psi(psi, strength=1.5, kind="curvature", subtract_mean=True)
    assert s3.shape == psi.shape

    # unknown kind raises
    try:
        gravity_source_from_psi(psi, kind="invalid")
        assert False, "Expected ValueError for unknown kind"
    except ValueError:
        pass


def test_weighted_norm_and_clip_positive():
    psi = np.ones((4, 4), dtype=np.complex128)
    cr = np.ones_like(psi, dtype=float) * 0.5
    w = weighted_norm(psi, clock_rate=cr)
    assert w > 0

    # Non-positive clock_rate should raise
    try:
        weighted_norm(psi, clock_rate=np.zeros_like(cr))
        assert False, "Expected ValueError for non-positive clock_rate"
    except ValueError:
        pass


def test_init_and_step_clock_rate_wave_clip_behavior():
    # initial uniform clock rate and zero source should remain near base
    base = 1.0
    N = np.ones((8, 8), dtype=float) * base
    dt = 0.01
    prev = init_clock_rate_wave(N, dt, c_g=1.0, gamma=0.0, mu=0.0, base=base, source0=None, clip=(0.2, 2.0))
    assert prev.shape == N.shape
    assert np.all(prev >= 0.2) and np.all(prev <= 2.0)

    # step with zero source should keep values within clip
    nextN = step_clock_rate_wave(N, prev, dt, c_g=1.0, gamma=0.0, mu=0.0, base=base, source=None, clip=(0.2, 2.0))
    assert np.all(nextN >= 0.2) and np.all(nextN <= 2.0)
