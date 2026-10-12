"""Synthetic labeled box mesh carrying every canonical anatomical label.

Not anatomy: a rectangular slab standing in for a maxilla, used for tests,
CI and quick demonstrations of the remodeling pipeline. Axes: x = lr,
y = ap (posterior -> anterior), z = si (inferior -> superior).

- ``palate``: inferior (z = 0) faces of the central strip |x - xc| < lx/4
- ``occlusal``: inferior faces of the lateral strips |x - xc| > 3 lx/8
- ``fixed``: superior (z = lz) faces, i.e. the junction with the cranial base
- ``suture_mid``: elements with centroid |x - xc| < suture_halfwidth
- ``cortical``: remaining elements with depth < cortical_thickness
- reporting regions split by |x - xc| (vault vs. alveolar), z and y
"""
from __future__ import annotations

import numpy as np

from craniofea.mesh.primitives import boundary_faces, box_mesh, face_normals_areas
from craniofea.model import AnatomicalFrame, Mesh


def labeled_box_mesh(lx: float = 0.040, ly: float = 0.050, lz: float = 0.020,
                     nx: int = 8, ny: int = 6, nz: int = 4,
                     cortical_thickness: float | None = None,
                     suture_halfwidth: float | None = None) -> Mesh:
    """Box mesh with all canonical surface/node/elem sets, elem fields and frame."""
    m = box_mesh(lx, ly, lz, nx, ny, nz)
    hx, hz = lx / nx, lz / nz
    t_cort = 0.99 * min(hx, ly / ny, hz) if cortical_thickness is None else cortical_thickness
    w_sut = 0.5 * hx if suture_halfwidth is None else suture_halfwidth

    nodes, tets = m.nodes, m.tets
    xc = 0.5 * lx
    cen = nodes[tets].mean(axis=1)

    # depth and nearest-face outward normal of each element centroid
    d = np.stack([cen[:, 0], lx - cen[:, 0], cen[:, 1], ly - cen[:, 1],
                  cen[:, 2], lz - cen[:, 2]], axis=1)
    k = np.argmin(d, axis=1)
    normals_tab = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
                            [0, 0, -1], [0, 0, 1]], dtype=float)
    depth = d[np.arange(len(d)), k]
    surface_normal = normals_tab[k]

    # material layers (disjoint, covering): suture first, then cortical shell
    dx = np.abs(cen[:, 0] - xc)
    suture = dx < w_sut
    cortical = ~suture & (depth < t_cort)
    cancellous = ~suture & ~cortical

    # reporting regions (disjoint, covering)
    vault = dx < 0.25 * lx
    alveolar = ~vault & (cen[:, 2] < 0.5 * lz)
    anterior = cen[:, 1] >= 0.5 * ly
    regions = {
        "region_palatal_vault": vault,
        "region_alveolar_anterior": alveolar & anterior,
        "region_alveolar_posterior": alveolar & ~anterior,
        "region_basal": ~vault & ~alveolar,
    }

    faces, _ = boundary_faces(nodes, tets)
    fc = nodes[faces].mean(axis=1)
    fn, _ = face_normals_areas(nodes, faces)
    bottom = fn[:, 2] < -0.99
    top = fn[:, 2] > 0.99
    fdx = np.abs(fc[:, 0] - xc)
    surface_sets = dict(m.surface_sets)
    surface_sets.update({
        "palate": faces[bottom & (fdx < 0.25 * lx)],
        "occlusal": faces[bottom & (fdx > 0.375 * lx)],
        "fixed": faces[top],
        "all": faces,
    })
    node_sets = dict(m.node_sets)
    node_sets["fixed"] = np.unique(surface_sets["fixed"])

    elem_sets = {"cortical": np.flatnonzero(cortical),
                 "cancellous": np.flatnonzero(cancellous),
                 "suture_mid": np.flatnonzero(suture)}
    elem_sets.update({k: np.flatnonzero(v) for k, v in regions.items()})

    frame = AnatomicalFrame(origin=nodes.mean(axis=0), lr=np.array([1.0, 0, 0]),
                            ap=np.array([0, 1.0, 0]), si=np.array([0, 0, 1.0]))
    return Mesh(nodes=nodes, tets=tets, node_sets=node_sets, elem_sets=elem_sets,
                surface_sets=surface_sets,
                elem_fields={"depth": depth, "surface_normal": surface_normal},
                frame=frame,
                meta={"source": "labeled_box_mesh", "dims": [lx, ly, lz],
                      "divisions": [nx, ny, nz], "anatomical": False})
