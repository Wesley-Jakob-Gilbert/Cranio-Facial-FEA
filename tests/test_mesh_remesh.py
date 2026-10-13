import numpy as np
import pytest

from craniofea.mesh.primitives import boundary_faces, box_mesh, orient_tets, tet_signed_volumes
from craniofea.mesh.remesh import remesh_surface, transfer_surface_set
from craniofea.model import Mesh


def test_transfer_surface_set_between_box_meshes():
    a = box_mesh(1.0, 1.0, 1.0, 3, 3, 3)
    b = box_mesh(1.0, 1.0, 1.0, 5, 4, 6)
    got = transfer_surface_set(a, "zmin", b, max_distance=0.05)
    key = lambda f: {tuple(sorted(t)) for t in f.tolist()}  # noqa: E731
    assert key(got) == key(b.surface_sets["zmin"])


def test_remesh_box_surface():
    pytest.importorskip("gmsh")
    src = box_mesh(10.0, 6.0, 4.0, 5, 3, 2)
    nodes, tets = remesh_surface(src.nodes, src.surface_sets["all"], size=1.5, threads=1)
    tets = orient_tets(nodes, tets)
    vol = tet_signed_volumes(nodes, tets)
    assert np.all(vol > 0)
    assert np.isclose(vol.sum(), 240.0, rtol=1e-6)
    faces, _ = boundary_faces(nodes, tets)
    m = Mesh(nodes=nodes, tets=tets, surface_sets={"all": faces})
    assert len(transfer_surface_set(src, "zmax", m, max_distance=0.1)) > 0
