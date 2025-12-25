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
