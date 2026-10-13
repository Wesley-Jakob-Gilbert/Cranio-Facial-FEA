import numpy as np
import pytest

from craniofea.mesh.labeling import (
    DEFAULTS,
    LabelConfig,
    detect_planar_cuts,
    summary_table,
    validate_labels,
)
from craniofea.mesh.primitives import box_mesh, face_normals_areas
from craniofea.mesh.surface import (
    SurfaceQuery,
    closest_point_on_triangles,
    face_components,
)
from craniofea.mesh.template import build_template_mesh
from craniofea.model import AnatomicalFrame

IDENTITY = AnatomicalFrame(origin=np.zeros(3), lr=np.array([1.0, 0, 0]),
                           ap=np.array([0, 1.0, 0]), si=np.array([0, 0, 1.0]))


def test_closest_point_matches_dense_sampling():
    rng = np.random.default_rng(0)
    tri = rng.normal(size=(200, 3, 3))
    p = rng.normal(size=(200, 3)) * 2
    q = closest_point_on_triangles(p, tri[:, 0], tri[:, 1], tri[:, 2])
    d = np.linalg.norm(q - p, axis=1)
    # Dense barycentric samples can never be closer than the exact closest point.
    u, v = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))
    keep = (u + v) <= 1
    u, v = u[keep], v[keep]
    s = (tri[:, None, 0] + u[None, :, None] * (tri[:, None, 1] - tri[:, None, 0])
         + v[None, :, None] * (tri[:, None, 2] - tri[:, None, 0]))
    ds = np.linalg.norm(s - p[:, None], axis=2).min(axis=1)
    assert np.all(d <= ds + 1e-12)
    assert np.all(ds - d < 0.05)


def test_surface_query_box_depth():
    m = box_mesh(1.0, 2.0, 3.0, 4, 4, 4)
    pts = np.array([[0.5, 1.0, 1.5], [0.1, 1.0, 1.5], [0.5, 1.9, 0.2]])
    d, _ = SurfaceQuery(m.nodes, m.surface_sets["all"]).query(pts)
    assert np.allclose(d, [0.5, 0.1, 0.1])


def test_face_components():
    m = box_mesh(1, 1, 1, 2, 2, 2)
    f = np.vstack([m.surface_sets["xmin"], m.surface_sets["xmax"]])
    assert len(np.unique(face_components(f))) == 2
    assert len(np.unique(face_components(m.surface_sets["all"]))) == 1


def test_planar_cut_on_box_is_top_face():
    m = box_mesh(0.02, 0.03, 0.01, 4, 6, 3)
    cuts = detect_planar_cuts(m, IDENTITY, DEFAULTS["fixed"])
    n, a = face_normals_areas(m.nodes, cuts)
    assert np.allclose(n, [0, 0, 1])
    assert np.isclose(a.sum(), 0.02 * 0.03)


def test_planar_cut_tilted_plane():
    m = box_mesh(0.02, 0.02, 0.01, 6, 6, 3)
    m.nodes[:, 2] += 0.08 * m.nodes[:, 0] * (m.nodes[:, 2] > 0.0099)    # ~4.6 deg tilt
    cuts = detect_planar_cuts(m, IDENTITY, DEFAULTS["fixed"])
    assert len(cuts) == len(m.surface_sets["zmax"])


def test_planar_cut_absent_direction():
    m = box_mesh(0.02, 0.02, 0.01, 4, 4, 2)
    cfg = dict(DEFAULTS["fixed"], directions=["superior", "posterior"])
    cuts = detect_planar_cuts(m, IDENTITY, cfg)
    assert len(cuts) == len(m.surface_sets["zmax"]) + len(m.surface_sets["ymin"])


def test_label_config_merge_and_validation():
    cfg = LabelConfig.from_dict({"layers": {"cortical_thickness_mm": 2.0}})
    assert cfg.layers["cortical_thickness_mm"] == 2.0
    assert cfg.layers["suture_half_width_mm"] == DEFAULTS["layers"]["suture_half_width_mm"]
    with pytest.raises(ValueError):
        LabelConfig.from_dict({"cortex": {}})


def test_validate_labels_detects_violations():
    m = build_template_mesh({"plan_size": 3.0, "layers": 4})
    assert validate_labels(m) == []
    m.elem_sets["cortical"] = m.elem_sets["cortical"][1:]
    m.elem_sets["region_basal"] = np.concatenate([m.elem_sets["region_basal"],
                                                  m.elem_sets["region_palatal_vault"][:3]])
    problems = validate_labels(m)
    assert any("cortical" in p and "uncovered" in p for p in problems)
    assert any("region_" in p and "multiply" in p for p in problems)
    assert "palate" in summary_table(m)


def test_cortical_thickness_controls_shell():
    thin = build_template_mesh({"plan_size": 3.0, "layers": 4},
                               {"layers": {"cortical_thickness_mm": 1.0}})
    thick = build_template_mesh({"plan_size": 3.0, "layers": 4},
                                {"layers": {"cortical_thickness_mm": 2.5}})
    assert len(thick.elem_sets["cortical"]) > len(thin.elem_sets["cortical"])
    for m, t in ((thin, 1.0e-3), (thick, 2.5e-3)):
        d = m.elem_fields["depth"]
        assert d[m.elem_sets["cancellous"]].min() >= t
