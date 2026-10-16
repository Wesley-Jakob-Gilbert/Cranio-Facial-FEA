import numpy as np

from craniofea.mesh.primitives import box_mesh, face_normals_areas, tet_signed_volumes


def test_box_mesh_volume_orientation_and_surfaces():
    m = box_mesh(2.0, 1.0, 0.5, 4, 3, 2)
    vol = tet_signed_volumes(m.nodes, m.tets)
    assert np.all(vol > 0)
    assert np.isclose(vol.sum(), 1.0)
    n, a = face_normals_areas(m.nodes, m.surface_sets["all"])
    assert np.isclose(a.sum(), 2 * (2 * 1 + 2 * 0.5 + 1 * 0.5))
    # Divergence theorem: sum of outward n*A over a closed surface is zero.
    assert np.allclose((n * a[:, None]).sum(0), 0, atol=1e-12)
    nz, _ = face_normals_areas(m.nodes, m.surface_sets["zmax"])
    assert np.allclose(nz, [0, 0, 1])
