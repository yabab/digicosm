import numpy as np
from code.model import compute_clock_rate, clock_rate_from_psi, clock_rate_from_curvature, curvature_proxy


def test_compute_clock_rate_requires_one_input():
    try:
        compute_clock_rate()
        assert False, "Expected ValueError when neither psi nor curvature provided"
    except ValueError:
        pass

    try:
        compute_clock_rate(psi=np.zeros((2,2)), curvature=np.zeros((2,2)))
        assert False, "Expected ValueError when both psi and curvature provided"
    except ValueError:
        pass


def test_compute_clock_rate_matches_existing_apis():
    psi = np.zeros((6,6), dtype=np.complex128)
    psi[2,2] = 1.0 + 0j
    # when passing psi
    out1 = compute_clock_rate(psi=psi, base=1.0, beta=0.2, mode='rational', clip=(0.01,10.0))
    # when passing curvature
    curv = curvature_proxy(psi)
    out2 = compute_clock_rate(curvature=curv, base=1.0, beta=0.2, mode='rational', clip=(0.01,10.0))

    assert out1.shape == psi.shape
    assert out2.shape == psi.shape
    assert np.allclose(out1, out2)


def test_compute_clock_rate_with_per_site_base_and_broadcasting():
    psi = np.zeros((6, 6), dtype=np.complex128)
    psi[2, 2] = 1.0 + 0j
    curv = curvature_proxy(psi)

    # per-site base as a column vector broadcastable to (6,6)
    base_col = np.linspace(0.5, 1.5, 6).reshape(6, 1)
    beta = 0.1

    out = compute_clock_rate(curvature=curv, base=base_col, beta=beta, mode="exp", clip=(0.0, 10.0))
    # expected elementwise: base_col * exp(-beta * curv)
    expected = base_col * np.exp(-beta * curv)
    assert out.shape == curv.shape
    assert np.allclose(out, expected)


def test_compute_clock_rate_incompatible_base_shape_raises():
    psi = np.zeros((5, 5), dtype=np.complex128)
    psi[1, 1] = 1.0 + 0j
    curv = curvature_proxy(psi)

    base_bad = np.ones((3,))
    import pytest

    with pytest.raises(ValueError):
        compute_clock_rate(curvature=curv, base=base_bad, beta=0.1)
