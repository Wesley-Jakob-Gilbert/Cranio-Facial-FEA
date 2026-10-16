import numpy as np
import pytest

from craniofea.mesh.primitives import box_mesh
from craniofea.model import AnatomicalFrame

pv = pytest.importorskip("pyvista")


@pytest.fixture
def mesh():
    m = box_mesh(0.04, 0.05, 0.02, 8, 10, 4)
    m.frame = AnatomicalFrame(m.nodes.mean(0), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    return m


def test_surface_polydata_maps_owner_values(mesh):
    from craniofea.viz.render import surface_polydata

    vals = np.arange(mesh.n_elems, dtype=float)
    poly = surface_polydata(mesh, {"v": vals})
    assert poly.n_cells == len(mesh.surface_sets["all"])
    assert set(np.unique(poly.cell_data["v"])) <= set(vals)


@pytest.mark.slow
def test_render_views(mesh, tmp_path):
    from craniofea.viz.render import render_section, render_surface_field, side_by_side, write_gif

    vals = mesh.nodes[mesh.tets].mean(1)[:, 1]
    a = render_surface_field(mesh, vals, tmp_path / "a.png", view="inferior", label="y")
    b = render_section(mesh, vals, tmp_path / "b.png", plane="lr")
    assert a.ndim == 3 and b.ndim == 3
    write_gif([side_by_side([a, a]), side_by_side([a, a])], tmp_path / "c.gif")
    assert (tmp_path / "c.gif").stat().st_size > 0
