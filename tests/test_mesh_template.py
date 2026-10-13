from pathlib import Path

import numpy as np
import pytest

from craniofea.mesh.labeling import LAYERS, REGIONS, validate_labels
from craniofea.mesh.primitives import (
    TET_FACES,
    face_multiplicity,
    face_normals_areas,
    tet_signed_volumes,
)
from craniofea.mesh.template import _extrude, build_template_from_config, build_template_mesh
from craniofea.model import Mesh

MM = 1e-3


@pytest.fixture(scope="module")
def tpl() -> Mesh:
    return build_template_mesh()


def test_template_size_and_orientation(tpl):
    assert tpl.n_elems < 20_000
    vol = tet_signed_volumes(tpl.nodes, tpl.tets)
    assert np.all(vol > 0)
    # Adolescent maxilla without teeth: of the order of 15-25 cm^3.
    assert 12e3 < vol.sum() / MM**3 < 30e3


def test_template_conforming_and_closed(tpl):
    # Every tet face is shared by at most two tets ...
    faces = tpl.tets[:, TET_FACES].reshape(-1, 3)
    assert face_multiplicity(faces).max() <= 2
    # ... and the boundary is a closed 2-manifold (each edge in exactly two triangles).
    b = tpl.surface_sets["all"]
    e = np.sort(b[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert np.all(cnt == 2)


def test_template_surfaces_outward(tpl):
    b = tpl.surface_sets["all"]
    n, a = face_normals_areas(tpl.nodes, b)
    assert np.allclose((n * a[:, None]).sum(0), 0.0, atol=1e-12)
    # Divergence theorem: V = (1/3) sum_f (x_f . n_f) A_f  (positive only if outward).
    x = tpl.nodes[b].mean(axis=1)
    v_surf = np.einsum("ij,ij->i", x, n) @ a / 3.0
    assert np.isclose(v_surf, tet_signed_volumes(tpl.nodes, tpl.tets).sum(), rtol=1e-9)


def test_prism_split_volume_is_exact():
    # Vertical-edged prisms over planar triangles: the 3-tet split is exact.
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    tri = np.array([[0, 1, 2], [1, 3, 2]])
    lo = np.array([0.0, 0.1, 0.2, 0.05])
    hi = np.array([1.0, 1.4, 0.9, 1.2])
    nodes, tets = _extrude(xy, tri, lo, hi, 3)
    expected = sum(0.5 * np.mean((hi - lo)[t]) for t in tri)
    assert np.isclose(np.abs(tet_signed_volumes(nodes, tets)).sum(), expected)


def test_template_labels_valid(tpl):
    assert validate_labels(tpl) == []
    for name in (*LAYERS, *REGIONS):
        assert len(tpl.elem_sets[name]) > 0, name


def test_template_surface_semantics(tpl):
    f = tpl.frame
    si = f.si
    n_pal, a_pal = face_normals_areas(tpl.nodes, tpl.surface_sets["palate"])
    assert np.all(n_pal @ si < 0)                        # palate faces the oral cavity
    n_fix, _ = face_normals_areas(tpl.nodes, tpl.surface_sets["fixed"])
    assert np.all(n_fix @ si > 0.9)                      # cuts face superiorly
    h = lambda s: f.to_local(tpl.nodes[tpl.surface_sets[s]].mean(axis=1))[:, 2]  # noqa: E731
    # Occlusal (crest) is the lowest labelled surface, the cut the highest.
    assert np.median(h("occlusal")) < np.median(h("palate")) < np.median(h("fixed"))
    # Adolescent palatal surface area is of the order of 15-25 cm^2.
    assert 1000 < a_pal.sum() / MM**2 < 2500
    assert np.isin(tpl.node_sets["fixed"], tpl.surface_sets["fixed"]).all()


def test_template_suture_on_midline(tpl):
    x = tpl.frame.to_local(tpl.nodes[tpl.tets[tpl.elem_sets["suture_mid"]]].mean(axis=1))
    assert np.abs(x[:, 0]).max() < 0.5 * MM
    # The slab spans the palate from front to back.
    assert np.ptp(x[:, 1]) > 30 * MM


def test_template_depth_fields(tpl):
    d = tpl.elem_fields["depth"]
    n = tpl.elem_fields["surface_normal"]
    assert d.shape == (tpl.n_elems,) and np.all(d >= 0)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    cort = tpl.elem_sets["cortical"]
    canc = tpl.elem_sets["cancellous"]
    assert d[cort].mean() < d[canc].mean()


def test_template_save_load_roundtrip(tpl, tmp_path):
    p = tmp_path / "tpl.npz"
    tpl.save(p)
    m = Mesh.load(p)
    assert np.array_equal(m.nodes, tpl.nodes) and np.array_equal(m.tets, tpl.tets)
    for group in ("node_sets", "elem_sets", "surface_sets", "elem_fields"):
        a, b = getattr(m, group), getattr(tpl, group)
        assert set(a) == set(b)
        for k in a:
            assert np.array_equal(a[k], b[k]), (group, k)
    for k in ("origin", "lr", "ap", "si"):
        assert np.array_equal(getattr(m.frame, k), getattr(tpl.frame, k))
    assert m.meta["source"] == "parametric_template"


def test_template_from_config_file():
    cfg = Path(__file__).resolve().parents[1] / "configs" / "mesh_template.yaml"
    m = build_template_from_config(cfg)
    assert validate_labels(m) == []


def test_template_parameters_validated():
    with pytest.raises(ValueError):
        build_template_mesh({"not_a_parameter": 1.0})


def test_template_refinement_converges_volume():
    coarse = build_template_mesh({"plan_size": 3.0, "layers": 4})
    fine = build_template_mesh({"plan_size": 1.8, "layers": 8})
    vc = tet_signed_volumes(coarse.nodes, coarse.tets).sum()
    vf = tet_signed_volumes(fine.nodes, fine.tets).sum()
    assert abs(vc - vf) / vf < 0.03
    for m in (coarse, fine):
        assert validate_labels(m) == []
