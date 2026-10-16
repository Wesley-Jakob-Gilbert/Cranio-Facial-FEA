"""Registry of FE solver backends implementing `craniofea.model.Solver`.

Backends are imported lazily so that optional dependencies (JAX, CalculiX,
FEBio) are only needed when that backend is requested.
"""
from __future__ import annotations

import importlib

from craniofea.model import Solver

_REGISTRY: dict[str, tuple[str, str]] = {
    "reference": ("craniofea.fem.reference", "ReferenceSolver"),
    "jax": ("craniofea.fem.jax_solver", "JaxSolver"),
    "calculix": ("craniofea.backends.calculix", "CalculixSolver"),
    "febio": ("craniofea.backends.febio", "FebioSolver"),
}


def available_solvers() -> list[str]:
    return sorted(_REGISTRY)


def get_solver(name: str, **kwargs) -> Solver:
    """Instantiate solver `name` with keyword options."""
    try:
        module, cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown solver {name!r}; choose from {available_solvers()}") from None
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"solver {name!r} is unavailable ({module}: {e})") from e
    return getattr(mod, cls)(**kwargs)
