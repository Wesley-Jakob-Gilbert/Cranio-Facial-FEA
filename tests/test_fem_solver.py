"""Production solver against the direct reference solver, and iterative robustness."""
import numpy as np
import pytest

from craniofea.fem.jax_solver import JaxSolver, stiff_components
from craniofea.fem.reference import ReferenceSolver
from craniofea.fem.verification import heterogeneous_box, layered_box


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.fixture(scope="module")
def hetero():
    prob = heterogeneous_box()
    return prob, ReferenceSolver().solve(prob)


@pytest.mark.parametrize("method", ["amg", "jax-cg"])
def test_matches_reference_on_heterogeneous_orthotropic_model(hetero, method):
    prob, ref = hetero
    sol = JaxSolver(method=method, rtol=1e-13).solve(prob)
    assert set(sol) == set(ref)
    for name in ref:
        s, r = sol[name], ref[name]
        assert s.info["converged"]
        assert _rel(s.u, r.u) < 1e-9
        assert _rel(s.strain, r.strain) < 1e-9
        assert _rel(s.stress, r.stress) < 1e-9
        assert _rel(s.sed, r.sed) < 1e-9
        assert np.all(s.u[prob.constrained_mask()] == 0)


def test_solution_info_and_pattern_reuse(hetero):
    prob, _ = hetero
    solver = JaxSolver()
    first = solver.solve(prob)["pressure"].info
    for key in ("iterations", "relative_residual", "converged", "assembly_s", "setup_s",
                "solve_s", "time_s", "n_dofs", "n_free_dofs"):
        assert key in first
    assert first["relative_residual"] <= 1e-10
    assert len(solver._pattern_cache) == 1
    solver.solve(prob)
    assert len(solver._pattern_cache) == 1


@pytest.fixture(scope="module")
def layered():
    prob = layered_box((14, 10, 8))
    return prob, ReferenceSolver().solve(prob)


def test_stiff_components_split_at_compliant_layer(layered):
    prob, _ = layered
    comps = stiff_components(prob.mesh.tets, prob.material.C, prob.mesh.n_nodes)
    assert len(comps) == 2
    x = prob.mesh.nodes[:, 0]
    mid = 0.5 * x.max()
    assert {bool(np.all(x[c] < mid)) for c in comps} == {True, False}


@pytest.mark.parametrize("deflation", [True, False])
def test_high_contrast_convergence(layered, deflation):
    """340:10:1 stiffness contrast (shell / core / suture) converges to tight tolerance."""
    prob, ref = layered
    sol = JaxSolver(rtol=1e-10, deflation=deflation).solve(prob)
    for name, s in sol.items():
        assert s.info["converged"] and s.info["relative_residual"] <= 1e-10
        assert s.info["iterations"] < 150
        assert _rel(s.u, ref[name].u) < 1e-7
    if deflation:
        assert sol["top"].info["deflation_vectors"] == 12


def test_deflation_reduces_iterations(layered):
    prob, _ = layered
    it = {d: JaxSolver(deflation=d).solve(prob)["top"].info["iterations"] for d in (True, False)}
    assert it[True] < it[False]


def test_high_contrast_jax_cg(layered):
    prob, ref = layered
    sol = JaxSolver(method="jax-cg", rtol=1e-10).solve(prob)
    for name, s in sol.items():
        assert s.info["converged"]
        assert _rel(s.u, ref[name].u) < 1e-6


def test_rejects_inverted_elements(hetero):
    prob, _ = hetero
    tets = prob.mesh.tets.copy()
    tets[0, [1, 2]] = tets[0, [2, 1]]
    bad = type(prob.mesh)(nodes=prob.mesh.nodes, tets=tets)
    with pytest.raises(ValueError, match="non-positive volume"):
        JaxSolver().prepare(type(prob)(bad, prob.material, prob.fixed_nodes, prob.load_cases))
