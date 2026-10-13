"""Implicit-adjoint gradients against central finite differences."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from craniofea.fem.jax_solver import JaxSolver, make_differentiable_solver
from craniofea.fem.verification import heterogeneous_box


@pytest.fixture(scope="module")
def problem():
    return heterogeneous_box((4, 3, 3), seed=7)


def _objective(fem, problem):
    rng = np.random.default_rng(11)
    w = jnp.asarray(rng.normal(size=(len(problem.load_cases), problem.mesh.n_nodes, 3)))
    v = jnp.asarray(rng.uniform(size=(len(problem.load_cases), problem.mesh.n_elems)))

    def L(s, ls):
        out = fem(s, ls)
        u_scale, sed_scale = 1e5, 1e-3      # make both terms O(1) to O(100)
        return u_scale * jnp.sum(w * out["u"]) + sed_scale * jnp.sum(v * out["sed"])

    return L


@pytest.mark.parametrize("method", ["amg", "jax-cg"])
def test_gradient_matches_central_differences(problem, method):
    fem = make_differentiable_solver(problem, method=method, rtol=1e-14)
    L = _objective(fem, problem)
    rng = np.random.default_rng(5)
    s0 = jnp.asarray(rng.uniform(0.5, 1.5, problem.mesh.n_elems))
    l0 = jnp.array([1.0, 0.7, 1.3])
    gs, gl = jax.grad(L, argnums=(0, 1))(s0, l0)

    h = 1e-4
    for seed in range(2):                         # random directions in modulus space
        d = jnp.asarray(np.random.default_rng(seed).normal(size=s0.shape))
        fd = (L(s0 + h * d, l0) - L(s0 - h * d, l0)) / (2 * h)
        assert abs(float(gs @ d) - float(fd)) < 1e-5 * abs(float(fd))
    for e in (0, 17, problem.mesh.n_elems - 1):   # individual elements
        de = jnp.zeros_like(s0).at[e].set(h)
        fd = (L(s0 + de, l0) - L(s0 - de, l0)) / (2 * h)
        assert abs(float(gs[e]) - float(fd)) < 1e-5 * float(jnp.abs(gs).max())
    for c in range(3):                            # load-case scales
        dc = jnp.zeros(3).at[c].set(h)
        fd = (L(s0, l0 + dc) - L(s0, l0 - dc)) / (2 * h)
        assert abs(float(gl[c]) - float(fd)) < 1e-5 * float(jnp.abs(gl).max())


def test_unit_scales_reproduce_solver_and_jit(problem):
    fem = make_differentiable_solver(problem, rtol=1e-13)
    ones = jnp.ones(problem.mesh.n_elems)
    out = jax.jit(fem)(ones)
    ref = JaxSolver(rtol=1e-13).solve(problem)
    for i, name in enumerate(fem.case_names):
        assert np.allclose(out["u"][i], ref[name].u, rtol=0, atol=1e-10 * np.abs(ref[name].u).max())
        assert np.allclose(out["sed"][i], ref[name].sed, rtol=1e-8,
                           atol=1e-10 * ref[name].sed.max())
    sol = fem.solution(out, "gravity")
    assert sol.u.shape == (problem.mesh.n_nodes, 3)
    assert fem.history and all(h["relative_residual"] <= 1e-13 for h in fem.history)


def test_modulus_scaling_is_exact_for_uniform_scale(problem):
    """K(c s) = c K(s): displacements scale by 1/c, energy density by 1/c."""
    fem = make_differentiable_solver(problem, rtol=1e-13)
    s = jnp.ones(problem.mesh.n_elems)
    a, b = fem(s), fem(2.0 * s)
    assert np.allclose(b["u"], 0.5 * a["u"], rtol=0, atol=1e-10 * float(jnp.abs(a["u"]).max()))
    assert np.allclose(b["sed"], 0.5 * a["sed"], rtol=1e-8, atol=1e-12 * float(a["sed"].max()))
