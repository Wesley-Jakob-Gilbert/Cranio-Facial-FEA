import numpy as np
import pytest

from craniofea.config import LoadingConfig, MaterialsConfig
from craniofea.fem.reference import assemble_rhs, tet4_gradients
from craniofea.materials import build_material
from craniofea.mesh.primitives import face_normals_areas
from craniofea.model import LoadCase, Problem
from craniofea.remodel.demo import labeled_box_mesh
from craniofea.remodel.loading import (
    ap_coordinate,
    bite_nodal_forces,
    build_daily_loading,
    graded_pressure,
)
from craniofea.remodel.simulate import check_mesh


@pytest.fixture(scope="module")
def mesh():
    return labeled_box_mesh(nx=6, ny=4, nz=3)


def test_demo_mesh_has_canonical_labels(mesh):
    check_mesh(mesh)
    M = mesh.n_elems
    layers = np.concatenate([mesh.elem_sets[k] for k in ("cortical", "cancellous", "suture_mid")])
    regions = np.concatenate([mesh.elem_sets[k] for k in mesh.elem_sets if k.startswith("region_")])
    assert np.array_equal(np.sort(layers), np.arange(M))
    assert np.array_equal(np.sort(regions), np.arange(M))
    assert all(len(mesh.elem_sets[k]) > 0 for k in mesh.elem_sets)


@pytest.mark.parametrize("ratio", [1.0, 2.5, 0.5])
def test_graded_pressure_preserves_resultant(mesh, ratio):
    faces = mesh.surface_sets["palate"]
    n, a = face_normals_areas(mesh.nodes, faces)
    p = graded_pressure(mesh, "palate", 2000.0, ratio)
    assert np.isclose(np.sum(p * a), 2000.0 * a.sum())
    t = ap_coordinate(mesh, faces)
    assert np.isclose(p[t == 1].mean() / p[t == 0].mean(), ratio)   # linear AP gradient
    # positive pressure pushes the palate superiorly (into the bone)
    vol = tet4_gradients(mesh.nodes, mesh.tets)[1]
    prob = Problem(mesh, build_material(mesh, 1.0, MaterialsConfig()), mesh.node_sets["fixed"], [])
    f = assemble_rhs(prob, LoadCase("p", pressures={"palate": p}), vol).reshape(-1, 3)
    assert np.allclose(f.sum(axis=0), [0, 0, 2000.0 * a.sum()])


def test_bite_forces_resultant_and_molar_bias(mesh):
    f = bite_nodal_forces(mesh, 120.0, molar_weight=3.0)
    assert np.allclose(f.sum(axis=0), [0, 0, 120.0])
    y = mesh.nodes[:, 1]
    post, ant = f[y < y.mean(), 2].sum(), f[y > y.mean(), 2].sum()
    assert post > ant > 0
    g = bite_nodal_forces(mesh, 100.0, molar_weight=1.0, inclination_deg=30.0)
    assert np.allclose(g.sum(axis=0), [0, 50.0, 100 * np.cos(np.pi / 6)])


def test_scenarios_build_expected_cases(mesh):
    cfg = LoadingConfig()
    names = {s: [d.name for d in build_daily_loading(mesh, cfg, s)] for s in cfg.scenarios}
    assert names["reference"] == ["rest_posture", "swallow", "chew", "gravity"]
    assert names["mewing"] == ["rest_posture", "swallow", "chew", "gravity"]
    assert names["mouth_breathing"] == ["swallow", "chew", "gravity"]
    hist = build_daily_loading(mesh, cfg, "mewing")
    static = {d.name: d.static for d in hist}
    assert static == {"rest_posture": True, "swallow": False, "chew": False, "gravity": True}
    cyc = {d.name: d.case.cycles_per_day for d in hist}
    assert cyc["swallow"] == 600.0 and cyc["chew"] == 1500.0
    rest = hist[0].case.pressures["palate"]
    _, a = face_normals_areas(mesh.nodes, mesh.surface_sets["palate"])
    assert np.isclose(np.sum(rest * a) / a.sum(), 2000.0)


def test_scenario_errors(mesh):
    with pytest.raises(KeyError):
        build_daily_loading(mesh, LoadingConfig(), "nope")
    bad = labeled_box_mesh(nx=6, ny=4, nz=3)
    del bad.surface_sets["occlusal"]
    with pytest.raises(KeyError, match="occlusal"):
        build_daily_loading(bad, LoadingConfig(), "reference")
