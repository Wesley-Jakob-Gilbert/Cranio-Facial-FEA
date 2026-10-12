"""External finite-element solvers driven from the craniofea data model.

- :class:`CalculixSolver` / :func:`write_inp`: CalculiX ``ccx`` (C3D4, linear static).
- :class:`FebioSolver` / :func:`write_feb`: FEBio 4 (tet4, spec 4.0 input).

Both implement the :class:`craniofea.model.Solver` protocol and are used to
cross-verify the in-house solvers and to hand models to established codes.
See ``docs/BACKENDS.md`` for the conventions and measured agreement.
"""
from craniofea.backends._common import BackendError
from craniofea.backends.calculix import CalculixSolver, write_inp
from craniofea.backends.febio import FebioSolver, write_feb

__all__ = ["BackendError", "CalculixSolver", "FebioSolver", "write_feb", "write_inp"]
