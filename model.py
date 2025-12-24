"""Compatibility facade.

Project intent is to keep the core dynamics separate from measurement/analysis.
Tests import from `model` for stability, while implementations live in:
  - `physics.py` (dynamics)
  - `measurements.py` (analysis)
"""

from physics import (  # noqa: F401
    laplacian_iso,
    step_relativistic_2nd_order,
    energy_relativistic_2nd_order,
    run_driven_relativistic_wave,
    run_driven_pulse,
)

from measurements import (  # noqa: F401
    precompute_radial_reduction,
    detect_front_outermost,
    isotropy_ring_error,
    cardinal_diagonal_peak_delta,
    detect_from_buffer,
)


__all__ = [
    "laplacian_iso",
    "step_relativistic_2nd_order",
    "energy_relativistic_2nd_order",
    "run_driven_relativistic_wave",
    "run_driven_pulse",
    "precompute_radial_reduction",
    "detect_front_outermost",
    "isotropy_ring_error",
    "cardinal_diagonal_peak_delta",
    "detect_from_buffer",
]
