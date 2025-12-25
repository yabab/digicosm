from dataclasses import dataclass
from typing import Callable, Dict, Any
import numpy as np
from .model import laplacian_iso


@dataclass
class Species:
    name: str
    kind: str
    params: Dict[str, Any]
    step_impl: Callable[..., np.ndarray]


# simple registry for species plugins
_REGISTRY: Dict[str, Species] = {}


def register_species(spec: Species):
    """Register a species plugin by name (overwrites existing)."""
    _REGISTRY[spec.name] = spec


def get_species(name: str) -> Species:
    return _REGISTRY[name]


def list_species() -> list:
    return list(_REGISTRY.keys())


def scalar_leapfrog_step(phi, phi_prev, dt, *, c=1.0, mu=0.0, laplacian=laplacian_iso):
    """POC scalar species leapfrog: phi_tt = c^2 ∇^2 phi - mu^2 phi."""
    phi = np.asarray(phi, dtype=float)
    phi_prev = np.asarray(phi_prev, dtype=float)
    accel = (c ** 2) * laplacian(phi) - (mu ** 2) * phi
    phi_next = 2.0 * phi - phi_prev + (dt ** 2) * accel
    return phi_next


# Convenience: register a default scalar species POC
def _register_default_scalar():
    spec = Species(
        name="scalar",
        kind="scalar",
        params={"c": 1.0, "mu": 0.0},
        step_impl=scalar_leapfrog_step,
    )
    register_species(spec)


# register at import time
_register_default_scalar()
