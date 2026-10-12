import numpy as np

from craniofea.fem.constitutive import isotropic_C
from craniofea.fem.reference import ReferenceSolver
from craniofea.mesh.primitives import box_mesh
from craniofea.model import LoadCase, Material, Problem


def test_uniaxial_pressure_bar_is_exact():
    """Roller-supported bar compressed by uniform pressure: sigma_xx = -p, eps_xx = -p/E."""
    m = box_mesh(1.0, 0.2, 0.2, 6, 2, 2)
    E, nu, p = 1.0e9, 0.3, 1.0e5
    mat = Material(C=isotropic_C(np.full(m.n_elems, E), nu), mass_density=np.ones(m.n_elems))
    fixed = np.zeros((m.n_nodes, 3), bool)
    fixed[m.node_sets["xmin"], 0] = True
    fixed[m.node_sets["ymin"], 1] = True
    fixed[m.node_sets["zmin"], 2] = True
    prob = Problem(m, mat, fixed_nodes=np.array([], int), fixed_dofs=fixed,
                   load_cases=[LoadCase("p", pressures={"xmax": p})])
    sol = ReferenceSolver().solve(prob)["p"]
    assert np.allclose(sol.stress[:, 0], -p, rtol=1e-9)
    assert np.allclose(sol.stress[:, 1:], 0, atol=1e-6 * p)
    assert np.allclose(sol.u[m.node_sets["xmax"], 0], -p / E * 1.0, rtol=1e-9)
    assert np.allclose(sol.sed, p**2 / (2 * E), rtol=1e-9)
