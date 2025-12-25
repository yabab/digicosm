import numpy as np
from code.species import register_species, get_species, list_species, Species, scalar_leapfrog_step


def test_default_scalar_registered():
    names = list_species()
    assert "scalar" in names


def test_scalar_step_consistent_shapes_and_values():
    N = 24
    phi = np.zeros((N, N), dtype=float)
    phi[N // 2, N // 2] = 1.0
    phi_prev = phi.copy()

    phi_next = scalar_leapfrog_step(phi, phi_prev, 0.01, c=0.5, mu=0.0)
    assert phi_next.shape == phi.shape
    assert np.isfinite(phi_next).all()


def test_register_custom_species_and_step():
    # custom species that scales input each step
    def custom_step(arr, arr_prev, dt, **kwargs):
        return 0.9 * arr

    spec = Species(name="custom", kind="scalar", params={}, step_impl=custom_step)
    register_species(spec)

    s = get_species("custom")
    assert s.name == "custom"
    out = s.step_impl(np.zeros((4, 4)), np.zeros((4, 4)), 0.1)
    assert out.shape == (4, 4)
    assert np.allclose(out, 0.0)
