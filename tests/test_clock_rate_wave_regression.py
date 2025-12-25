import numpy as np
from code.model import (
    init_clock_rate_wave,
    step_clock_rate_wave,
    laplacian_iso,
)


def test_clock_rate_wave_causality_and_stability():
    # Small grid, localized perturbation; verify wavefront propagates at <= c_g
    N = 101
    c_g = 1.0
    dt = 0.1
    steps = 30

    # initial clock rate baseline
    base = 1.0
    N0 = np.full((N, N), base, dtype=float)
    # local bump at center
    cy = cx = N // 2
    N0[cy, cx] += 1.0

    clip = (1e-6, 10.0)
    N_prev = init_clock_rate_wave(N0, dt, c_g=c_g, base=base, source0=None, clip=clip, laplacian=laplacian_iso)

    N = N0.copy()
    Nprev = N_prev.copy()

    def max_radius_where_perturbed(field, center, threshold=1e-3):
        cy, cx = center
        yy, xx = np.indices(field.shape)
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = np.abs(field - base) > threshold
        if not mask.any():
            return 0.0
        return float(rr[mask].max())

    radii = [max_radius_where_perturbed(N, (cy, cx))]

    max_cells_per_step = 2.0
    for n in range(1, steps + 1):
        N_next = step_clock_rate_wave(N, Nprev, dt, c_g=c_g, base=base, source=None, clip=clip, laplacian=laplacian_iso)
        # checks
        assert np.isfinite(N_next).all()
        assert (N_next >= clip[0] - 1e-12).all()
        assert (N_next <= clip[1] + 1e-12).all()

        r = max_radius_where_perturbed(N_next, (cy, cx))
        radii.append(r)

        # enforce finite per-step propagation in grid-cell units (discrete stencil)
        prev_r = radii[-2]
        delta = r - prev_r
        assert delta <= max_cells_per_step + 1e-12

        Nprev, N = N, N_next

    # ensure front moved outward monotonically (non-decreasing radius)
    assert all(x <= y + 1e-12 for x, y in zip(radii, radii[1:]))
