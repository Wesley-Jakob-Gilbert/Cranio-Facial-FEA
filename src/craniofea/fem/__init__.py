"""Finite-element kernels and solvers for linear elasticity on tet4 meshes.

Modules
-------
constitutive : isotropic / orthotropic Voigt stiffness and Bond rotations
kernels      : vectorised JAX element kernels (gradients, B, K_e, recovery)
jax_solver   : production solver (``JaxSolver``) and the differentiable map
               (``make_differentiable_solver``); importing it enables float64 JAX
reference    : small direct NumPy/SciPy solver used as a test oracle
verification : analytic verification problems and error norms

``jax_solver`` is not imported here so that importing the package does not
change global JAX configuration.
"""
