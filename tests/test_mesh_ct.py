from pathlib import Path

import numpy as np
import pytest

from craniofea.mesh.ct import load_open_full_jaw, normalized_volume
from craniofea.mesh.primitives import box_mesh, face_normals_areas, tet_signed_volumes

ROOT = Path(__file__).resolve().parents[1]
CT_MSH = (ROOT / "data/open_full_jaw/extracted/Patient_3/output/maxilla/volumetric_meshes/"
          "maxilla_volumetric_mesh.msh")
MM = 1e-3


def _write_msh(path, points, tets, tags):
    import meshio

    meshio.write(str(path), meshio.Mesh(points, [("tetra", tets)],
                                        cell_data={"gmsh:geometrical": [tags],
                                                   "gmsh:physical": [tags]}),
                 file_format="gmsh22", binary=False)


@pytest.fixture()
def synthetic_msh(tmp_path):
    """Box in mm: bottom layer 'teeth' (1), a 'PDL' layer (2), the rest 'bone' (3)."""
    b = box_mesh(10.0, 8.0, 6.0, 5, 4, 6)
    zc = b.nodes[b.tets].mean(axis=1)[:, 2]
    tags = np.where(zc < 1.0, 1, np.where(zc < 2.0, 2, 3)).astype(np.int32)
    tets = b.tets.copy()
    tets[::7, [1, 2]] = tets[::7, [2, 1]]                         # some negative tets
    pts = np.vstack([b.nodes, [[50.0, 50.0, 50.0]]])               # one unused node
    flat = np.array([[0, 1, 2, 2]])                                # one degenerate tet
    tets = np.vstack([tets, flat])
    tags = np.append(tags, 3)
    path = tmp_path / "synthetic.msh"
    _write_msh(path, pts, tets, tags)
    return path, b


def test_import_synthetic(synthetic_msh):
    path, box = synthetic_msh
    m = load_open_full_jaw(path)
    assert m.meta["degenerate_removed"] == 1
    assert m.meta["reoriented"] > 0
    vol = tet_signed_volumes(m.nodes, m.tets)
    assert np.all(vol > 0)
    assert np.isclose(vol.sum(), 10 * 8 * 4 * MM**3)              # bone part, in m^3
    assert m.n_nodes == len(np.unique(m.tets))                     # compacted
    # Socket = bone/PDL interface = the bone's bottom face at z = 2 mm.
    s = m.surface_sets["socket"]
    n, a = face_normals_areas(m.nodes, s)
    assert np.isclose(a.sum(), 10 * 8 * MM**2)
    assert np.allclose(n, [0, 0, -1])                              # outward from bone
    assert "tissue_tag" not in m.elem_fields


def test_import_keep_several_tissues(synthetic_msh):
    path, _ = synthetic_msh
    m = load_open_full_jaw(path, keep=("bone", "pdl", "teeth"))
    assert set(np.unique(m.elem_fields["tissue_tag"])) == {1, 2, 3}
    assert "socket" not in m.surface_sets


def test_normalized_volume_scale_invariant():
    reg = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    t = np.array([[0, 1, 2, 3]])
    q = normalized_volume(reg, t)
    assert np.isclose(abs(q[0]), 1.0)
    assert np.isclose(normalized_volume(reg * 1e-3, t)[0], q[0])


# ----------------------------------------------------------------- Open-Full-Jaw data

ct = pytest.mark.skipif(not CT_MSH.exists(), reason="Open-Full-Jaw data not in data/")


@pytest.fixture(scope="module")
def ct_mesh():
    from craniofea.mesh.labeling import build_ct_mesh

    return build_ct_mesh(ROOT / "configs/mesh_ct.yaml", root=ROOT, save=False)


@pytest.mark.ct
@ct
def test_ct_import_counts(ct_mesh):
    assert ct_mesh.n_elems == 420_779
    assert ct_mesh.meta["degenerate_removed"] == 0
    vol = tet_signed_volumes(ct_mesh.nodes, ct_mesh.tets)
    assert np.all(vol > 0)
    assert np.isclose(vol.sum() / MM**3, 15877, rtol=1e-3)
    assert len(ct_mesh.surface_sets["socket"]) == 45_483


@pytest.mark.ct
@ct
def test_ct_frame_matches_dicom_lps(ct_mesh):
    f = ct_mesh.frame
    assert f.lr @ [-1, 0, 0] > 0.99       # LPS: +x is patient left
    assert f.ap @ [0, -1, 0] > 0.99       # +y posterior
    assert f.si @ [0, 0, 1] > 0.99        # +z superior


@pytest.mark.ct
@ct
def test_ct_labels(ct_mesh):
    from craniofea.mesh.labeling import validate_labels

    assert validate_labels(ct_mesh) == []
    si = ct_mesh.frame.si
    n, a = face_normals_areas(ct_mesh.nodes, ct_mesh.surface_sets["palate"])
    assert np.all(n @ si < 0)
    assert 1200 < a.sum() / MM**2 < 2500
    _, a_fix = face_normals_areas(ct_mesh.nodes, ct_mesh.surface_sets["fixed"])
    assert a_fix.sum() / MM**2 > 300
    # Fixed nodes never touch the tooth sockets or the palate.
    for s in ("socket", "palate"):
        assert not np.isin(ct_mesh.node_sets["fixed"], ct_mesh.surface_sets[s]).any(), s
