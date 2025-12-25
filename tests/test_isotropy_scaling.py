import numpy as np
from code.measurements import (
    isotropy_ring_error,
    front_radii_by_angle,
    cardinal_diagonal_peak_delta,
)
from code.model import hamiltonian_total, curvature_proxy


def test_isotropy_ring_error_near_zero_on_synthetic_ring():
    N = 61
    cy = cx = N // 2
    r = 10
    yy, xx = np.indices((N, N))
    rr = np.rint(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(int)
    intensity = np.zeros((N, N), dtype=float)
    intensity[rr == r] = 1.0

    res = isotropy_ring_error(intensity, (cy, cx), dr=1, search_start=3)
    assert res["r_peak"] == r
    # ring is discrete; allow modest angular variation from sampling
    assert res["iso_error"] < 1.5


def test_front_radii_by_angle_returns_consistent_circle():
    N = 80
    cy = cx = N // 2
    r_frac = 0.25
    r = int(min(cy, cx) * r_frac)

    yy, xx = np.indices((N, N))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    intensity = np.zeros((N, N), dtype=float)
    intensity[rr <= r] = 1.0

    radii = front_radii_by_angle(intensity, (cy, cx), threshold=0.5, angles=64, rmin=1)
    radii_valid = radii[~np.isnan(radii)]
    mean_r = float(np.mean(radii_valid))
    std_r = float(np.std(radii_valid))

    assert abs(mean_r - r) < 1.5
    # discrete sampling and pixelization cause small variations; allow ~7% relative
    assert (std_r / max(1.0, mean_r)) < 0.07


def test_cardinal_vs_diagonal_delta_zero_when_symmetric():
    N = 40
    cy = N // 2
    cx = N // 2
    intensity = np.zeros((N, N), dtype=float)
    # create increasing sequence along row and diagonal so maxima align
    for i in range(1, 12):
        intensity[cy, cx + i] = float(i)
        intensity[cy + i, cx + i] = float(i)

    stats = cardinal_diagonal_peak_delta(intensity, (cy, cx))
    assert stats["delta"] == 0


def test_kappa_scaling_affects_hamiltonian_linearly():
    N = 24
    psi = np.zeros((N, N), dtype=np.complex128)
    psi[N // 2, N // 2] = 1.0 + 0j
    psi[N // 2, N // 2 + 1] = -1.0 + 0j

    H0 = hamiltonian_total(psi, omega0=0.0, kappa=0.0)
    H1 = hamiltonian_total(psi, omega0=0.0, kappa=1.0)
    H2 = hamiltonian_total(psi, omega0=0.0, kappa=2.0)

    # H_k - H0 should scale linearly with k
    assert np.isclose(H2 - H0, 2.0 * (H1 - H0), atol=1e-12)


def test_resolution_scaling_mean_radius_consistent_fractional():
    # Build disks at two resolutions with same fractional radius and compare normalized means
    def make_disk(N, frac):
        cy = cx = N // 2
        r = int(min(cy, cx) * frac)
        yy, xx = np.indices((N, N))
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        I = np.zeros((N, N), dtype=float)
        I[rr <= r] = 1.0
        return I, r

    I1, r1 = make_disk(40, 0.2)
    I2, r2 = make_disk(80, 0.2)

    cy1 = cx1 = 40 // 2
    cy2 = cx2 = 80 // 2

    rad1 = front_radii_by_angle(I1, (cy1, cx1), threshold=0.5, angles=64, rmin=1)
    rad2 = front_radii_by_angle(I2, (cy2, cx2), threshold=0.5, angles=64, rmin=1)

    mean1 = float(np.nanmean(rad1))
    mean2 = float(np.nanmean(rad2))

    # normalized radii should be close
    assert abs((mean1 / 40.0) - (mean2 / 80.0)) < 0.02
