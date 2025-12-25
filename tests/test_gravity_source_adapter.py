import numpy as np
import pytest
from code.model import (
    gravity_source_from_fields,
    gravity_source_from_psi,
    curvature_proxy,
    energy_density,
)


def test_adapter_prefers_psi_when_present():
    psi = np.zeros((8, 8), dtype=np.complex128)
    psi[3, 3] = 1.0 + 0.0j
    dct = {"psi": psi, "density": energy_density(psi)}

    out_adapter = gravity_source_from_fields(dct, strength=2.0, subtract_mean=True)
    out_direct = gravity_source_from_psi(psi, strength=2.0, kind="curvature", subtract_mean=True)

    assert out_adapter.shape == psi.shape
    assert np.allclose(out_adapter, out_direct)


def test_adapter_accepts_curvature_array():
    psi = np.zeros((6, 6), dtype=np.complex128)
    psi[1, 2] = 1.0 + 0.0j
    curv = curvature_proxy(psi)

    out = gravity_source_from_fields(curv, strength=1.5, kind="curvature", subtract_mean=True)
    # subtract_mean => sum close to 0
    assert abs(float(out.sum())) < 1e-12
    # scaled by strength
    assert np.allclose(out, 1.5 * (curv - curv.mean()))


def test_adapter_accepts_density_array_by_default():
    psi = np.zeros((5, 5), dtype=np.complex128)
    psi[2, 2] = 2.0 + 0.0j
    dens = energy_density(psi)

    out = gravity_source_from_fields(dens, strength=0.5, subtract_mean=False)
    assert np.allclose(out, 0.5 * dens)


def test_adapter_raises_on_uninterpretable_mapping():
    with pytest.raises(ValueError):
        gravity_source_from_fields({"foo": np.ones((3, 3))})
