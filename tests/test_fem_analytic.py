"""Verification of the production solver against closed-form solutions."""
import numpy as np
import pytest

from craniofea.fem import verification as V
from craniofea.fem.constitutive import isotropic_C, orthotropic_C, rotate_C
from craniofea.fem.jax_solver import JaxSolver
from craniofea.mesh.primitives import box_mesh
from craniofea.model import LoadCase, Material, Problem

METHODS = ["amg", "jax-cg"]


@pytest.mark.parametrize("method", METHODS)
def test_patch_test_affine_displacement_on_irregular_mesh(method):
    """Boundary data u = G x + c on a jittered mesh: interior is affine, strain constant."""
    m = V.jittered_box(4, amplitude=0.3, seed=1)
    rng = np.random.default_rng(3)
    C = rotate_C(orthotropic_C(12e9, 8e9, 20e9, 3e9, 4.5e9, 5e9, 0.35, 0.2, 0.25),
                 V.random_rotations(1, rng)[0])
    mat = Material(C=np.broadcast_to(C, (m.n_elems, 6, 6)).copy(),
                   mass_density=np.ones(m.n_elems))
    G = rng.normal(size=(3, 3)) * 1e-3
    c = rng.normal(size=3) * 1e-4
    u_aff = m.nodes @ G.T + c
    prob = Problem(m, mat, fixed_nodes=m.node_sets["boundary"], load_cases=[LoadCase("patch")])
    sol = JaxSolver(method=method, rtol=1e-13).solve(prob, prescribed=u_aff)["patch"]
    scale = np.abs(u_aff).max()
    assert np.abs(sol.u - u_aff).max() < 1e-11 * scale
    E = 0.5 * (G + G.T)
    eps = np.array([E[0, 0], E[1, 1], E[2, 2], 2 * E[1, 2], 2 * E[0, 2], 2 * E[0, 1]])
    assert np.abs(sol.strain - eps).max() < 1e-10 * np.abs(eps).max()
    assert np.abs(sol.stress - C @ eps).max() < 1e-10 * np.abs(C @ eps).max()


@pytest.mark.parametrize("method", METHODS)
def test_uniaxial_pressure_bar_is_exact(method):
    m = box_mesh(1.0, 0.2, 0.2, 6, 2, 2)
    E, nu, p = 1.0e9, 0.3, 1.0e5
    mat = Material(C=isotropic_C(np.full(m.n_elems, E), nu), mass_density=np.ones(m.n_elems))
    fixed = np.zeros((m.n_nodes, 3), bool)
    fixed[m.node_sets["xmin"], 0] = True
    fixed[m.node_sets["ymin"], 1] = True
    fixed[m.node_sets["zmin"], 2] = True
    prob = Problem(m, mat, fixed_nodes=np.array([], int), fixed_dofs=fixed,
                   load_cases=[LoadCase("p", pressures={"xmax": p})])
    sol = JaxSolver(method=method, rtol=1e-13).solve(prob)["p"]
    assert np.allclose(sol.stress[:, 0], -p, rtol=1e-9)
    assert np.allclose(sol.stress[:, 1:], 0, atol=1e-8 * p)
    assert np.allclose(sol.u[m.node_sets["xmax"], 0], -p / E, rtol=1e-9)
    assert np.allclose(sol.u[m.node_sets["ymax"], 1], nu * p / E * 0.2, rtol=1e-9)
    assert np.allclose(sol.sed, p**2 / (2 * E), rtol=1e-9)


def _convergence(builder, levels, case):
    solver = JaxSolver(rtol=1e-11)
    h, eu, es = [], [], []
    for k in levels:
        c = builder(k)
        s = solver.solve(c.problem)[case]
        h.append(c.h)
        eu.append(V.l2_error(c.problem.mesh, s.u, c.u_exact)[1])
        es.append(V.stress_l2_error(c.problem.mesh, s.stress, c.stress_exact)[1])
    return np.array(h), np.array(eu), np.array(es)


def test_hanging_column_converges_at_second_order():
    h, eu, es = _convergence(V.hanging_column, (1, 2, 4), "gravity")
    assert eu[-1] < 2e-4
    assert V.observed_rate(h, eu) > 1.85
    assert V.observed_rate(h, es) > 0.95


def test_lame_cylinder_displacement_and_stress_convergence():
    h, eu, es = _convergence(V.lame_cylinder, (2, 4, 8), "pressure")
    assert eu[-1] < 3e-3 and es[-1] < 7e-2
    assert V.observed_rate(h[-2:], eu[-2:]) > 1.9           # O(h^2) in L2 displacement
    assert V.observed_rate(h, es) > 0.85                    # O(h) for constant-stress tets


@pytest.mark.slow
def test_lame_cylinder_fine_meshes():
    h, eu, es = _convergence(V.lame_cylinder, (8, 16, 32), "pressure")
    assert V.observed_rate(h, eu) > 1.95
    assert V.observed_rate(h, es) > 0.95
    assert es[-1] < 2e-2


def _tip(level):
    c = V.cantilever(level)
    s = JaxSolver(rtol=1e-10).solve(c.problem)["tip"]
    return -s.u[c.meta["tip_nodes"], 2].mean(), c.meta["delta_timoshenko"]


def test_cantilever_tip_deflection_converges_monotonically():
    d = [_tip(k) for k in (1, 2, 4)]
    ratio = np.array([a / b for a, b in d])
    assert np.all(np.diff(ratio) > 0) and ratio[-1] < 1.0     # tet4 is stiff: from below
    err = 1 - ratio
    assert err[-1] < 0.25
    assert np.log(err[1] / err[2]) / np.log(2) > 1.0


@pytest.mark.slow
def test_cantilever_approaches_timoshenko():
    d = np.array([_tip(k)[0] for k in (4, 6, 9)])
    delta_t = _tip(1)[1]
    # Richardson extrapolation with the observed order (refinement ratio 1.5)
    q = np.log((d[1] - d[0]) / (d[2] - d[1])) / np.log(1.5)
    d_inf = d[2] + (d[2] - d[1]) / (1.5**q - 1)
    assert abs(d_inf / delta_t - 1) < 0.02
    assert d[2] / delta_t > 0.95
