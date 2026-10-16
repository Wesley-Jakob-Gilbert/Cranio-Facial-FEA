import numpy as np
import pytest

from craniofea.config import MaterialsConfig, StimulusConfig
from craniofea.fem.constitutive import isotropic_C
from craniofea.fem.reference import ReferenceSolver
from craniofea.materials import build_material
from craniofea.mesh.primitives import box_mesh
from craniofea.model import LoadCase, Material, Problem
from craniofea.remodel.demo import labeled_box_mesh
from craniofea.remodel.loading import DailyLoad, build_daily_loading
from craniofea.remodel.stimulus import (
    compute_stimulus,
    cycle_weights,
    daily_stimulus,
    effective_strain,
)


def test_effective_strain_uniaxial_bar_is_axial_strain():
    """Roller-supported bar under pressure p: U = p^2 / 2E, so eps = p / E exactly."""
    m = box_mesh(1.0, 0.2, 0.2, 4, 2, 2)
    E, p = 2.0e9, 1.0e5
    mat = Material(C=isotropic_C(np.full(m.n_elems, E), 0.3), mass_density=np.ones(m.n_elems))
    fixed = np.zeros((m.n_nodes, 3), bool)
    fixed[m.node_sets["xmin"], 0] = fixed[m.node_sets["ymin"], 1] = True
    fixed[m.node_sets["zmin"], 2] = True
    prob = Problem(m, mat, np.array([], int), [LoadCase("p", pressures={"xmax": p})], fixed)
    sol = ReferenceSolver().solve(prob)["p"]
    assert np.allclose(effective_strain(sol.sed, E), p / E, rtol=1e-9)


def test_daily_stimulus_hand_calculation():
    eps = np.array([[1e-3, 0.0], [2e-3, 5e-4]])
    n = np.array([100.0, 1.0])
    psi = daily_stimulus(eps, n, 4.0)
    assert np.isclose(psi[0], (100 * 1e-12 + 16e-12) ** 0.25, rtol=1e-12)
    assert np.isclose(psi[1], 5e-4, rtol=1e-12)
    assert daily_stimulus(np.zeros((2, 3)), n, 4.0).tolist() == [0.0, 0.0, 0.0]
    # very large m: no overflow, Psi -> max strain
    assert np.isclose(daily_stimulus(eps, np.ones(2), 400.0)[0], 2e-3, rtol=1e-2)


def test_static_equivalent_cycles_limits():
    """n_static -> 0: only cyclic events count; n_static -> inf: Psi ~ n^(1/m) eps_static."""
    eps = np.array([[3e-4], [1e-3]])          # [static, cyclic]
    hist = [DailyLoad(LoadCase("s"), static=True),
            DailyLoad(LoadCase("c", cycles_per_day=500.0), static=False)]
    m = 4.0
    psi = [daily_stimulus(eps, cycle_weights(hist, n), m)[0] for n in (0.0, 1.0, 1e3, 1e12)]
    assert np.isclose(psi[0], 500 ** 0.25 * 1e-3)
    assert psi[0] < psi[1] < psi[2] < psi[3]
    assert np.isclose(psi[3] / (1e12 ** 0.25 * 3e-4), 1.0, rtol=1e-6)


def test_compute_stimulus_scales_linearly_with_load():
    """Linear elasticity: scaling every load by k scales every eps_i (and Psi) by k."""
    mesh = labeled_box_mesh(nx=6, ny=4, nz=3)
    from craniofea.config import LoadingConfig, from_dict
    base = LoadingConfig()
    lc2 = from_dict(LoadingConfig, {
        "swallowing": {"pressure": 2 * base.swallowing.pressure},
        "mastication": {"force": 2 * base.mastication.force},
        "gravity": {"enabled": False},
        "scenarios": {"reference": {"resting_pressure": 400.0}}})
    lc1 = from_dict(LoadingConfig, {"gravity": {"enabled": False}})
    mat = build_material(mesh, 1.0, MaterialsConfig())
    solver = ReferenceSolver()
    s1 = compute_stimulus(mesh, mat, build_daily_loading(mesh, lc1, "reference"), solver,
                          StimulusConfig())
    s2 = compute_stimulus(mesh, mat, build_daily_loading(mesh, lc2, "reference"), solver,
                          StimulusConfig())
    assert np.allclose(s2.psi, 2 * s1.psi, rtol=1e-9)
    shares = s1.shares()
    assert np.isclose(sum(shares.values()), 1.0)
    assert set(s1.weights) == {"rest_posture", "swallow", "chew"}


@pytest.mark.parametrize("m", [2.0, 4.0, 8.0])
def test_stimulus_monotone_in_cycles(m):
    eps = np.array([[1e-4, 2e-4]])
    a = daily_stimulus(eps, np.array([10.0]), m)
    b = daily_stimulus(eps, np.array([20.0]), m)
    assert np.allclose(b / a, 2 ** (1 / m))
